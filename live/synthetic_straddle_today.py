"""Check the synthetic straddle breakout strategy
(backtest/synthetic_straddle_breakout_options.py) against TODAY's real
option prices. Read-only -- no order is ever placed, nothing here
touches a real broker account, same "simulated fill, real prices"
spirit as live/ema_sweep_paper_trader.py.

WHY THIS IS A SEPARATE MODULE FROM THE BACKTEST: the backtest only
reads option premiums via Upstox's *expired*-instruments API, which
requires a contract to have already settled -- it has no data at all
for the current week's still-live contract, so it can never be pointed
at "today". This module uses two different, no-auth-required data
sources instead:
    Reference day (yesterday/last completed session): Upstox's regular
        historical-candle endpoint (upstox_client.get_historical_candles)
        -- this works for a still-active (not yet expired) contract too,
        since a day's data publishes the day after, unlike the expired-
        instruments API which needs full settlement.
    Today (Stage 2, live entries): upstox_client.get_intraday_candles --
        "today so far" directly off the exchange feed, no auth needed.
    Current week's contract list: Upstox's public instrument master
        (assets.upstox.com/.../NSE.json.gz), filtered to NIFTY weekly
        options, same approach as live_contract_resolver in
        ema_sweep_paper_trader.py.

Parameters (SL_POINTS, TARGET_POINTS, MIN_DIFF_POINTS, STRIKE_STEP,
CANDLE_MINUTES) mirror the validated defaults in
backtest/synthetic_straddle_breakout_options.py as of this writing
(sl=13/target=26, min_diff_points=10, 1-min candles) -- update both
places together if that tuning changes.

Run directly (`python -m live.synthetic_straddle_today`) for a one-shot
snapshot of where the strategy stands right now: today's reference,
the selected strike, whether min_diff_points would skip the day
entirely, and (if not) each leg's current entry/exit state. This is a
single check, not a poller -- rerun it whenever you want a fresh read;
it does not persist state between runs.
"""
from __future__ import annotations

import gzip
import json
import time
from datetime import date, datetime, timedelta, timezone

import requests

from data_sources import upstox_client
from backtest import options_common as oc
from backtest.sweep_reclaim_breakout import _resample

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"
STRIKE_STEP = 50
STRIKE_SEARCH_RANGE = 2
SL_POINTS = 13.0
TARGET_POINTS = 26.0
MIN_DIFF_POINTS = 10.0
CANDLE_MINUTES = 1
INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
_IST = timezone(timedelta(hours=5, minutes=30))


def _load_nifty_weekly_chain() -> list[dict]:
    resp = requests.get(INSTRUMENTS_URL, timeout=30)
    resp.raise_for_status()
    data = json.loads(gzip.decompress(resp.content))
    return [
        d for d in data
        if d.get("name") == "NIFTY" and d.get("instrument_type") in ("CE", "PE") and d.get("weekly")
    ]


def _nearest_unexpired_expiry(chain: list[dict]) -> tuple[int | None, str | None]:
    now_ms = time.time() * 1000
    future = sorted(set(d["expiry"] for d in chain if d["expiry"] > now_ms))
    if not future:
        return None, None
    nearest = future[0]
    expiry_date = datetime.fromtimestamp(nearest / 1000, tz=timezone.utc).astimezone(_IST).date().isoformat()
    return nearest, expiry_date


def _contract_for_strike(chain: list[dict], expiry_ms: int, strike: float, opt_type: str) -> dict | None:
    for d in chain:
        if d["expiry"] == expiry_ms and d["strike_price"] == strike and d["instrument_type"] == opt_type:
            return d
    return None


def compute_reference(chain: list[dict], expiry_ms: int, ref_date: str, ref_close: float) -> dict | None:
    """Stage 1: same logic as synthetic_straddle_breakout_options.run(),
    against the live weekly chain and the regular historical-candle
    endpoint instead of the expired-instruments API. Returns
    {strike, avg_price, diff, ce_contract, pe_contract} or None if no
    candidate strike had both CE and PE data."""
    atm_guess = oc.round_to_step(ref_close, STRIKE_STEP)
    best = None
    for k in range(-STRIKE_SEARCH_RANGE, STRIKE_SEARCH_RANGE + 1):
        strike = atm_guess + k * STRIKE_STEP
        ce = _contract_for_strike(chain, expiry_ms, strike, "CE")
        pe = _contract_for_strike(chain, expiry_ms, strike, "PE")
        if ce is None or pe is None:
            continue
        ce_hist = upstox_client.get_historical_candles(ce["instrument_key"], "1minute", to_date=ref_date, from_date=ref_date)
        pe_hist = upstox_client.get_historical_candles(pe["instrument_key"], "1minute", to_date=ref_date, from_date=ref_date)
        ce_rows = sorted(ce_hist.get("data", {}).get("candles", []), key=lambda c: c[0])
        pe_rows = sorted(pe_hist.get("data", {}).get("candles", []), key=lambda c: c[0])
        if not ce_rows or not pe_rows:
            continue
        ce_close, pe_close = ce_rows[-1][4], pe_rows[-1][4]
        diff = abs(ce_close - pe_close)
        if best is None or diff < best["diff"]:
            best = {
                "strike": strike, "avg_price": (ce_close + pe_close) / 2, "diff": diff,
                "ce_contract": ce, "pe_contract": pe, "ce_close": ce_close, "pe_close": pe_close,
            }
    return best


def simulate_leg(bars: list[list], avg_price: float) -> dict:
    """Stage 2 for one leg (CE or PE): same entry/exit rules as
    synthetic_straddle_breakout_options.run() (close > avg_price entry,
    SL_POINTS/TARGET_POINTS premium exit). Returns a dict describing
    the leg's current state -- one of 'no_entry', 'closed' (with the
    completed trade's detail), or 'open' (with the still-running
    position's detail)."""
    position = None
    already_traded = False
    closed_trades = []
    for row in bars:
        ts, o, h, l, c, v, oi = row
        if position is not None:
            hit_stop = l <= position["entry_price"] - SL_POINTS
            hit_target = h >= position["entry_price"] + TARGET_POINTS
            if hit_stop or hit_target:
                closed_trades.append({
                    "entry_time": position["entry_time"], "entry_price": position["entry_price"],
                    "exit_time": ts, "exit_price": c,
                    "reason": "stop_loss" if hit_stop else "target",
                })
                position = None
        if position is None and not already_traded and c > avg_price:
            position = {"entry_time": ts, "entry_price": c}
            already_traded = True
    if position is not None:
        last = bars[-1]
        return {"status": "open", "position": position, "current_price": last[4], "closed_trades": closed_trades}
    if closed_trades:
        return {"status": "closed", "closed_trades": closed_trades}
    return {"status": "no_entry", "last_close": bars[-1][4] if bars else None}


def check_today() -> dict:
    """One-shot snapshot: reference, selected strike, min_diff_points
    gate, and each leg's current state. Read-only, makes no auth calls."""
    chain = _load_nifty_weekly_chain()
    expiry_ms, expiry_date = _nearest_unexpired_expiry(chain)
    if expiry_ms is None:
        return {"error": "no unexpired weekly expiry found in the live instrument master"}

    today = date.today()
    trading_days = upstox_client.get_daily_history(
        UNDERLYING_KEY, (today - timedelta(days=10)).isoformat(), (today - timedelta(days=1)).isoformat()
    )
    trading_days.sort(key=lambda d: d["date"])
    if not trading_days:
        return {"error": "no recent completed trading day found"}
    ref_day = trading_days[-1]

    ref = compute_reference(chain, expiry_ms, ref_day["date"], ref_day["close"])
    if ref is None:
        return {"error": "could not compute reference (no strikes with both CE/PE data)"}

    result = {
        "expiry": expiry_date, "ref_date": ref_day["date"], "ref_close": ref_day["close"],
        "strike": ref["strike"], "avg_price": ref["avg_price"], "diff": ref["diff"],
    }
    if ref["diff"] < MIN_DIFF_POINTS:
        result["skipped"] = True
        result["skip_reason"] = f"diff {ref['diff']:.2f} < min_diff_points {MIN_DIFF_POINTS}"
        return result

    result["skipped"] = False
    for label, contract in (("CE", ref["ce_contract"]), ("PE", ref["pe_contract"])):
        rows = sorted(upstox_client.get_intraday_candles(contract["instrument_key"], "1minute"), key=lambda c: c[0])
        bars = _resample(rows, CANDLE_MINUTES)
        result[label] = {"trading_symbol": contract["trading_symbol"], "bar_count": len(bars),
                          **simulate_leg(bars, ref["avg_price"])}
    return result


def main() -> None:
    r = check_today()
    if "error" in r:
        print(f"Error: {r['error']}")
        return
    print(f"Expiry: {r['expiry']}")
    print(f"Reference day: {r['ref_date']}, index close={r['ref_close']}")
    print(f"Best strike: {r['strike']}, diff={r['diff']:.2f}, avg_price={r['avg_price']:.2f}")
    if r["skipped"]:
        print(f"SKIPPED: {r['skip_reason']}")
        return
    for label in ("CE", "PE"):
        leg = r[label]
        print(f"\n[{label}] {leg['trading_symbol']} ({leg['bar_count']} bars today)")
        for t in leg.get("closed_trades", []):
            pnl = t["exit_price"] - t["entry_price"]
            print(f"    {t['reason'].upper()}: entry {t['entry_time']} @ {t['entry_price']:.2f} -> "
                  f"exit {t['exit_time']} @ {t['exit_price']:.2f} (pnl_pts={pnl:+.2f})")
        if leg["status"] == "open":
            pos = leg["position"]
            unrealized = leg["current_price"] - pos["entry_price"]
            print(f"    OPEN: entered {pos['entry_time']} @ {pos['entry_price']:.2f}, "
                  f"current={leg['current_price']:.2f} (unrealized_pts={unrealized:+.2f})")
        elif leg["status"] == "no_entry":
            print(f"    no entry yet (last close={leg['last_close']})")


if __name__ == "__main__":
    main()
