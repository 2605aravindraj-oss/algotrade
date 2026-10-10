"""Live forward paper-trading of the NIFTY futures OI-buildup strategy
(backtest/futures_oi_buildup.py), meant to run every 5 minutes during
market hours via a scheduled job.

Each run: fetch the current NIFTY futures LTP+OI, compare to the last
run's reading to classify the OI buildup quadrant, and apply the same
rules validated in the backtest:

    flat  + Long Buildup   -> buy ATM call
    flat  + Short Buildup  -> buy ATM put
    long  + Short Buildup  -> sell the call, immediately buy ATM put (reverse)
    long  + Long Unwinding -> sell the call, go flat
    short + Long Buildup   -> sell the put, immediately buy ATM call (reverse)
    short + Short Covering -> sell the put, go flat

Position and trade log persist in a local state file so P&L accumulates
correctly across runs. Forced flat before 15:25 IST (no overnight).
Requires UPSTOX_ACCESS_TOKEN.

Usage:
    export UPSTOX_ACCESS_TOKEN=...
    python live_nifty_oi_buildup_trader.py
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
STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".live_oi_buildup_state.json")
MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
IST = timezone(timedelta(hours=5, minutes=30))
FORCE_FLAT_TIME = "15:25"
STRIKE_STEP = 50


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


def _load_state() -> dict:
    if os.path.exists(STATE_PATH):
        return json.load(open(STATE_PATH))
    return {"position": None, "prev_future_price": None, "prev_future_oi": None, "trades": [], "cum_pnl": 0.0}


def _save_state(state: dict) -> None:
    json.dump(state, open(STATE_PATH, "w"), indent=2)


def _record_trade(state: dict, direction: str, strike: float, opt_type: str,
                   entry_time: str, entry_price: float, exit_time: str, exit_price: float,
                   lot_size: int, reason: str) -> None:
    entry_side = "BUY"
    exit_side = "SELL"
    fills = [
        costs.Fill(price=entry_price, lot_size=lot_size, side=entry_side),
        costs.Fill(price=exit_price, lot_size=lot_size, side=exit_side),
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


def main() -> None:
    token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not token:
        print("UPSTOX_ACCESS_TOKEN not set.", file=sys.stderr)
        sys.exit(1)

    now_ist = datetime.now(IST)
    time_str = now_ist.strftime("%H:%M")
    print(f"=== OI buildup paper trader @ {now_ist.strftime('%Y-%m-%d %H:%M:%S')} IST ===")

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

    def _option_contract_and_quote(strike, opt_type):
        c = lookup.get((strike, opt_type))
        if c is None:
            return None, None
        return c, _quote(token, c["instrument_key"])

    position = state.get("position")

    # Forced flat before close
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

    print(f"Futures: {fut_price:.2f} (OI={fut_oi:,.0f})  ATM={atm}  Buildup: {buildup}")

    def _enter(direction: str) -> None:
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
    _save_state(state)

    n_trades = len(state["trades"])
    print(f"Trades so far: {n_trades}  Cumulative net P&L: Rs {state['cum_pnl']:,.2f}")


if __name__ == "__main__":
    main()
