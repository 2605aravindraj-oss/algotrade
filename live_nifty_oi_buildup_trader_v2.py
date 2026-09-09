"""Variant of live_nifty_oi_buildup_trader.py that adds option-chain
confirmation factors on top of the same futures OI-buildup core signal.

This is a SEPARATE, independent copy -- it does not modify or interact
with live_nifty_oi_buildup_trader.py or its state file. Meant to run in
parallel so the two can be compared.

Same core signal (futures price+OI buildup) and same exit rules as v1.
What's added is an ENTRY FILTER using the option chain (+/-5 strikes
around ATM, same band as live_nifty_insights.py):

  1. OI-wall proximity: skip a fresh entry if the opposing OI wall (the
     resistance call-OI strike for a LONG, the support put-OI strike for
     a SHORT) is within WALL_BUFFER points of spot -- i.e. don't buy
     right into a wall of call writers, don't sell right into a wall of
     put writers.
  2. PCR trend confirmation: skip a fresh LONG entry if PCR is falling
     sharply (calls being written aggressively, working against the
     long); skip a fresh SHORT entry if PCR is rising sharply (puts
     being written aggressively, working against the short).

Exits are UNCHANGED from v1 (still driven purely by the futures buildup
reversing) -- only entries are filtered, so any P&L difference between
the two isolates the effect of the added chain factors.

Requires UPSTOX_ACCESS_TOKEN.

Usage:
    export UPSTOX_ACCESS_TOKEN=...
    python live_nifty_oi_buildup_trader_v2.py
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from io import BytesIO

import requests

from data_sources import upstox_client
from backtest import costs
from backtest.futures_oi_buildup import classify_buildup

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"
STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".live_oi_buildup_state_v2.json")
MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
IST = timezone(timedelta(hours=5, minutes=30))
FORCE_FLAT_TIME = "15:25"
STRIKE_STEP = 50
STRIKE_BAND = 5
WALL_BUFFER = 30  # points; skip entry if the opposing OI wall is closer than this
PCR_MOVE_LIMIT = 0.10  # skip entry if PCR moved more than 10% against the trade direction


def _load_master() -> list[dict]:
    resp = requests.get(MASTER_URL, timeout=30)
    resp.raise_for_status()
    with gzip.open(BytesIO(resp.content)) as f:
        return json.load(f)


def _nearest_future(master: list[dict]) -> dict:
    futs = [d for d in master if d.get("segment") == "NSE_FO" and d.get("underlying_symbol") == "NIFTY"
            and d.get("instrument_type") == "FUT"]
    return min(futs, key=lambda d: d["expiry"])


def _nifty_option_chain(master: list[dict]) -> tuple[str, dict]:
    opts = [d for d in master if d.get("segment") == "NSE_FO" and d.get("underlying_symbol") == "NIFTY"
            and d.get("instrument_type") in ("CE", "PE")]
    nearest_expiry = min(d["expiry"] for d in opts)
    chain = [d for d in opts if d["expiry"] == nearest_expiry]
    expiry_str = datetime.utcfromtimestamp(nearest_expiry / 1000).strftime("%Y-%m-%d")
    lookup = {(d["strike_price"], d["instrument_type"]): d for d in chain}
    return expiry_str, lookup


def _round_to_step(x: float, step: int = STRIKE_STEP) -> int:
    return int(round(x / step) * step)


def _quote(token: str, instrument_key: str) -> dict:
    data = upstox_client.get_quotes([instrument_key], token)["data"]
    return next(iter(data.values()))


def _quotes_batch(token: str, instrument_keys: list[str]) -> dict:
    data = upstox_client.get_quotes(instrument_keys, token)["data"]
    by_key = {}
    for v in data.values():
        by_key[v.get("instrument_token")] = v
    return by_key


def _chain_snapshot(token: str, lookup: dict, atm: int) -> dict:
    """PCR + resistance/support OI walls across +/-STRIKE_BAND strikes."""
    strikes = [atm + i * STRIKE_STEP for i in range(-STRIKE_BAND, STRIKE_BAND + 1)]
    keys_by_strike_type = {}
    all_keys = []
    for s in strikes:
        for t in ("CE", "PE"):
            c = lookup.get((s, t))
            if c:
                keys_by_strike_type[(s, t)] = c["instrument_key"]
                all_keys.append(c["instrument_key"])

    quotes = _quotes_batch(token, all_keys)

    total_call_oi = 0.0
    total_put_oi = 0.0
    max_call_oi = (None, -1)
    max_put_oi = (None, -1)
    for s in strikes:
        ce_key = keys_by_strike_type.get((s, "CE"))
        pe_key = keys_by_strike_type.get((s, "PE"))
        ce = quotes.get(ce_key) if ce_key else None
        pe = quotes.get(pe_key) if pe_key else None
        if ce:
            total_call_oi += ce.get("oi", 0)
            if ce.get("oi", 0) > max_call_oi[1]:
                max_call_oi = (s, ce.get("oi", 0))
        if pe:
            total_put_oi += pe.get("oi", 0)
            if pe.get("oi", 0) > max_put_oi[1]:
                max_put_oi = (s, pe.get("oi", 0))

    pcr = total_put_oi / total_call_oi if total_call_oi else None
    return {"pcr": pcr, "resistance": max_call_oi[0], "support": max_put_oi[0]}


def _load_state() -> dict:
    if os.path.exists(STATE_PATH):
        return json.load(open(STATE_PATH))
    return {"position": None, "prev_future_price": None, "prev_future_oi": None,
            "prev_pcr": None, "trades": [], "cum_pnl": 0.0}


def _save_state(state: dict) -> None:
    json.dump(state, open(STATE_PATH, "w"), indent=2)


def _record_trade(state: dict, direction: str, strike: float, opt_type: str,
                   entry_time: str, entry_price: float, exit_time: str, exit_price: float,
                   lot_size: int, reason: str) -> None:
    fills = [
        costs.Fill(price=entry_price, lot_size=lot_size, side="BUY"),
        costs.Fill(price=exit_price, lot_size=lot_size, side="SELL"),
    ]
    cost = costs.total_cost(fills)
    gross = (exit_price - entry_price) * lot_size
    net = gross - cost
    state["trades"].append({
        "direction": direction, "strike": strike, "opt_type": opt_type,
        "entry_time": entry_time, "entry_price": entry_price,
        "exit_time": exit_time, "exit_price": exit_price,
        "gross_pnl": gross, "cost": cost, "net_pnl": net, "reason": reason,
    })
    state["cum_pnl"] = state.get("cum_pnl", 0.0) + net
    print(f"  CLOSED {direction} {opt_type} {strike}: {entry_price} -> {exit_price}  "
          f"gross={gross:+.2f} cost={cost:.2f} net={net:+.2f}  ({reason})")


def _entry_allowed(direction: str, fut_price: float, snap: dict, prev_pcr: float | None) -> str | None:
    """Returns None if allowed, else a short reason string for why it's blocked.

    A wall only matters if it's actually in front of price in the trade's
    direction: resistance must be ABOVE price to block a long, support must
    be BELOW price to block a short. A wall behind price (already crossed)
    is not an obstacle, regardless of how close the raw point gap looks.
    """
    if direction == "LONG":
        r = snap["resistance"]
        if r is not None and r > fut_price and (r - fut_price) < WALL_BUFFER:
            return f"resistance wall too close ({r}, {r - fut_price:.0f}pt away)"
        if prev_pcr and snap["pcr"] and snap["pcr"] < prev_pcr * (1 - PCR_MOVE_LIMIT):
            return f"PCR falling sharply ({prev_pcr:.2f} -> {snap['pcr']:.2f}, calls being written)"
    else:
        s = snap["support"]
        if s is not None and s < fut_price and (fut_price - s) < WALL_BUFFER:
            return f"support wall too close ({snap['support']}, {fut_price-snap['support']:.0f}pt away)"
        if prev_pcr and snap["pcr"] and snap["pcr"] > prev_pcr * (1 + PCR_MOVE_LIMIT):
            return f"PCR rising sharply ({prev_pcr:.2f} -> {snap['pcr']:.2f}, puts being written)"
    return None


def main() -> None:
    token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not token:
        print("UPSTOX_ACCESS_TOKEN not set.", file=sys.stderr)
        sys.exit(1)

    now_ist = datetime.now(IST)
    time_str = now_ist.strftime("%H:%M")
    print(f"=== OI buildup + chain-factors paper trader (v2) @ {now_ist.strftime('%Y-%m-%d %H:%M:%S')} IST ===")

    if time_str < "09:15" or time_str > "15:30":
        print("Outside market hours (09:15-15:30 IST) -- skipping.")
        return

    state = _load_state()
    master = _load_master()
    fut = _nearest_future(master)
    fut_quote = _quote(token, fut["instrument_key"])
    fut_price = fut_quote["last_price"]
    fut_oi = fut_quote.get("oi", 0)

    expiry, lookup = _nifty_option_chain(master)
    atm = _round_to_step(fut_price)
    snap = _chain_snapshot(token, lookup, atm)

    def _option_contract_and_quote(strike, opt_type):
        c = lookup.get((strike, opt_type))
        if c is None:
            return None, None
        return c, _quote(token, c["instrument_key"])

    position = state.get("position")

    if position is not None and time_str >= FORCE_FLAT_TIME:
        _, q = _option_contract_and_quote(position["strike"], position["opt_type"])
        if q:
            _record_trade(state, position["direction"], position["strike"], position["opt_type"],
                           position["entry_time"], position["entry_price"],
                           now_ist.isoformat(), q["last_price"], position["lot_size"], "eod")
        state["position"] = None
        _save_state(state)
        print(f"Forced flat before close. Cumulative net P&L: Rs {state['cum_pnl']:,.2f}")
        return

    buildup = "Neutral"
    if state.get("prev_future_price") is not None:
        buildup = classify_buildup(fut_price - state["prev_future_price"], fut_oi - state["prev_future_oi"])

    pcr_str = f"{snap['pcr']:.2f}" if snap["pcr"] else "n/a"
    print(f"Futures: {fut_price:.2f} (OI={fut_oi:,.0f})  ATM={atm}  Buildup: {buildup}  "
          f"PCR={pcr_str}  R={snap['resistance']}  S={snap['support']}")

    def _enter(direction: str) -> None:
        block_reason = _entry_allowed(direction, fut_price, snap, state.get("prev_pcr"))
        if block_reason:
            print(f"  {direction} signal present but BLOCKED by chain filter: {block_reason}")
            return
        opt_type = "CE" if direction == "LONG" else "PE"
        contract, q = _option_contract_and_quote(atm, opt_type)
        if contract is None or q is None:
            print("  could not resolve option contract/quote, skipping entry")
            return
        state["position"] = {
            "direction": direction, "strike": atm, "opt_type": opt_type,
            "entry_time": now_ist.isoformat(), "entry_price": q["last_price"],
            "lot_size": contract["lot_size"],
        }
        print(f"  ENTER {direction} {opt_type} {atm} @ {q['last_price']}")

    def _exit(reason: str) -> None:
        nonlocal position
        _, q = _option_contract_and_quote(position["strike"], position["opt_type"])
        exit_price = q["last_price"] if q else position["entry_price"]
        _record_trade(state, position["direction"], position["strike"], position["opt_type"],
                       position["entry_time"], position["entry_price"],
                       now_ist.isoformat(), exit_price, position["lot_size"], reason)
        state["position"] = None
        position = None

    if position is None:
        if buildup == "Long Buildup":
            _enter("LONG")
        elif buildup == "Short Buildup":
            _enter("SHORT")
        else:
            print("  no position, no qualifying signal -- staying flat")
    else:
        if position["direction"] == "LONG" and buildup == "Short Buildup":
            _exit("reverse")
            _enter("SHORT")
        elif position["direction"] == "LONG" and buildup == "Long Unwinding":
            _exit("unwind")
        elif position["direction"] == "SHORT" and buildup == "Long Buildup":
            _exit("reverse")
            _enter("LONG")
        elif position["direction"] == "SHORT" and buildup == "Short Covering":
            _exit("covering")
        else:
            print(f"  holding {position['direction']} {position['opt_type']} {position['strike']} "
                  f"(entry {position['entry_price']})")

    state["prev_future_price"] = fut_price
    state["prev_future_oi"] = fut_oi
    state["prev_pcr"] = snap["pcr"]
    _save_state(state)

    n_trades = len(state["trades"])
    print(f"Trades so far: {n_trades}  Cumulative net P&L: Rs {state['cum_pnl']:,.2f}")


if __name__ == "__main__":
    main()
