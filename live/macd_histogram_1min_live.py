"""Live, TODAY-ONLY check: MACD(12,26,9) histogram zero-cross on NIFTY 50
spot 1-minute candles, buy ATM CE at the crossing candle's own close.

Not a backtest -- this looks at TODAY's candles only (both for the
index and for the resolved ATM CE contract) and reports whatever
already happened today: no signal yet, an open position, or a position
that's already been stopped out.

SIGNAL: MACD(12,26,9) on the index's own 1-minute closes (continuous
through today's bars so far -- needs the standard 12/26/9 EMA warm-up,
~34 bars/34 minutes before the first real histogram value). The FIRST
bar today whose histogram crosses from <=0 to >0 (a fresh cross, not
merely "is positive") is the entry: buy the ATM CE, filled at that
candle's own close.

STOP-LOSS (as specified -- a dual condition, not price alone and not
histogram alone): the first LATER candle whose close is BELOW the
entry candle's own close AND whose histogram is BELOW the entry
candle's own histogram value. Both must hold on the same candle.
Requiring both avoids a single noisy wick (price dips but momentum
hasn't actually weakened, or vice versa) stopping the trade out.

No target specified -- holds until the stop triggers or the market
closes (forced-flat, read from the last available candle of the day).

STRIKE/EXPIRY: ATM = round-to-50 of the index close at the entry
candle; nearest weekly expiry today, resolved from Upstox's live
instrument master (NSE_FO segment, underlying NIFTY) -- the same
master-loading pattern live_nifty_insights.py uses, since this is a
LIVE (non-expired) contract, not one of this codebase's cached
expired-instruments.

DATA: both the index and the CE contract's candles come from
upstox_client.get_intraday_candles -- public, no auth, current
trading day only.
"""
from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime, timedelta, timezone
from io import BytesIO

import requests

from data_sources import upstox_client
from backtest.technical_rating import _macd

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"
MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
IST = timezone(timedelta(hours=5, minutes=30))
FORCE_FLAT_TIME = "15:25"


def _load_master() -> list[dict]:
    resp = requests.get(MASTER_URL, timeout=30)
    resp.raise_for_status()
    with gzip.open(BytesIO(resp.content)) as f:
        return json.load(f)


def _nearest_nifty_ce(master: list[dict], strike: float) -> dict | None:
    opts = [
        d for d in master
        if d.get("segment") == "NSE_FO" and d.get("underlying_symbol") == "NIFTY"
        and d.get("instrument_type") == "CE"
    ]
    if not opts:
        return None
    nearest_expiry = min(d["expiry"] for d in opts)
    chain = [d for d in opts if d["expiry"] == nearest_expiry]
    return min(chain, key=lambda d: abs(d["strike_price"] - strike))


def _round_to_50(x: float) -> int:
    return int(round(x / 50) * 50)


def run() -> dict:
    rows = sorted(upstox_client.get_intraday_candles(UNDERLYING_KEY, "1minute"), key=lambda r: r[0])
    if len(rows) < 35:
        return {"status": "insufficient_data", "bars_so_far": len(rows)}

    closes = [r[4] for r in rows]
    macd_line, signal_line = _macd(closes, 12, 26, 9)
    histogram = [(m - s) if (m is not None and s is not None) else None for m, s in zip(macd_line, signal_line)]

    entry_idx = None
    prev_hist = None
    for i, hist in enumerate(histogram):
        if hist is None:
            prev_hist = hist
            continue
        if prev_hist is not None and prev_hist <= 0 and hist > 0:
            entry_idx = i
            break
        prev_hist = hist

    if entry_idx is None:
        return {"status": "no_signal_yet", "bars_so_far": len(rows)}

    entry_row = rows[entry_idx]
    entry_close = entry_row[4]
    entry_hist = histogram[entry_idx]
    entry_ts = entry_row[0]

    exit_idx = None
    for j in range(entry_idx + 1, len(rows)):
        c = rows[j][4]
        h = histogram[j]
        if h is None:
            continue
        if c < entry_close and h < entry_hist:
            exit_idx = j
            break

    master = _load_master()
    atm = _round_to_50(entry_close)
    contract = _nearest_nifty_ce(master, atm)
    if contract is None:
        return {"status": "error", "message": "could not resolve a live NIFTY CE contract"}

    opt_rows = sorted(upstox_client.get_intraday_candles(contract["instrument_key"], "1minute"), key=lambda r: r[0])
    if not opt_rows:
        return {"status": "error", "message": "no intraday candles yet for the resolved CE contract"}

    def _premium_at_or_after(ts_target: str) -> tuple[str, float] | None:
        for r in opt_rows:
            if r[0] >= ts_target:
                return r[0], r[4]
        return opt_rows[-1][0], opt_rows[-1][4]

    entry_time, entry_premium = _premium_at_or_after(entry_ts)

    result = {
        "status": "open" if exit_idx is None else "stopped_out",
        "entry_time": entry_ts,
        "entry_index_close": entry_close,
        "entry_histogram": entry_hist,
        "strike": contract["strike_price"],
        "trading_symbol": contract["trading_symbol"],
        "instrument_key": contract["instrument_key"],
        "entry_premium_time": entry_time,
        "entry_premium": entry_premium,
    }

    if exit_idx is not None:
        exit_ts = rows[exit_idx][0]
        exit_index_close = rows[exit_idx][4]
        exit_hist = histogram[exit_idx]
        exit_time, exit_premium = _premium_at_or_after(exit_ts)
        result.update({
            "exit_time": exit_ts,
            "exit_index_close": exit_index_close,
            "exit_histogram": exit_hist,
            "exit_premium_time": exit_time,
            "exit_premium": exit_premium,
            "exit_reason": "stop_loss",
            "pnl_per_unit": round(exit_premium - entry_premium, 2),
        })
    else:
        last_ts, last_premium = opt_rows[-1][0], opt_rows[-1][4]
        result.update({
            "latest_time": last_ts,
            "latest_premium": last_premium,
            "unrealized_pnl_per_unit": round(last_premium - entry_premium, 2),
            "lot_size": contract.get("lot_size"),
        })

    return result


def summary(result: dict) -> str:
    status = result.get("status")
    if status == "insufficient_data":
        return f"Only {result['bars_so_far']} bars so far today -- MACD(12,26,9) needs ~35 to produce a real histogram value. Check back later."
    if status == "no_signal_yet":
        return f"No histogram zero-cross yet today ({result['bars_so_far']} bars so far)."
    if status == "error":
        return f"Error: {result['message']}"

    lines = [
        f"Entry: {result['entry_time']} IST, NIFTY spot close={result['entry_index_close']:.2f}, "
        f"histogram={result['entry_histogram']:.2f}",
        f"Bought: {result['trading_symbol']} (strike {result['strike']:.0f}) @ {result['entry_premium']:.2f} "
        f"(fill time {result['entry_premium_time']})",
    ]
    if status == "stopped_out":
        lines.append(
            f"STOPPED OUT at {result['exit_time']} IST: spot close={result['exit_index_close']:.2f} < entry "
            f"AND histogram={result['exit_histogram']:.2f} < entry histogram."
        )
        lines.append(
            f"Exit premium: {result['exit_premium']:.2f} (fill time {result['exit_premium_time']}) -- "
            f"P&L per unit: {result['pnl_per_unit']:+.2f}"
        )
    else:
        lines.append(
            f"Still OPEN as of {result['latest_time']} IST: premium={result['latest_premium']:.2f} -- "
            f"unrealized P&L per unit: {result['unrealized_pnl_per_unit']:+.2f} (lot size {result['lot_size']})"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    res = run()
    print(summary(res))
