"""Live, TODAY-ONLY check: MACD(12,26,9) histogram zero-cross on NIFTY 50
spot 1-minute candles, buy ATM CE at the crossing candle's own close.

Not a backtest -- this looks at TODAY's candles only (both for the
index and for each resolved ATM CE contract) and reports EVERY signal
that's fired today so far, in order: closed (stopped-out) trades and,
if the most recent one hasn't stopped yet, the currently open trade.
After a stop-loss, scanning resumes from the very next bar looking for
a fresh cross -- a choppy morning can produce several signals in a
row (seen live: 2026-10-07 fired at 10:00 IST, stopped out at 10:01,
then fired again at 10:05 and stayed open -- an earlier version of
this script only reported the first signal of the day and silently
dropped every later one, which is the bug this fixed).

SIGNAL: MACD(12,26,9) on the index's own 1-minute closes (continuous
through today's bars so far -- needs the standard 12/26/9 EMA warm-up,
~34 bars/34 minutes before the first real histogram value). Each bar
whose histogram crosses from <=0 to >0 (a fresh cross, not merely "is
positive") opens a trade: buy the ATM CE, filled at that candle's own
close. Only one position open at a time -- the scan for the NEXT
entry only resumes after the current one's stop triggers.

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
        return {"status": "insufficient_data", "bars_so_far": len(rows), "trades": []}

    closes = [r[4] for r in rows]
    macd_line, signal_line = _macd(closes, 12, 26, 9)
    histogram = [(m - s) if (m is not None and s is not None) else None for m, s in zip(macd_line, signal_line)]

    master = None  # lazy-loaded only once a first signal actually fires
    trades: list[dict] = []
    i = 0
    prev_hist = None
    while i < len(rows):
        hist = histogram[i]
        if hist is None:
            prev_hist = hist
            i += 1
            continue
        if prev_hist is None or not (prev_hist <= 0 and hist > 0):
            prev_hist = hist
            i += 1
            continue

        # fresh cross at bar i -- open a trade
        entry_idx = i
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

        if master is None:
            master = _load_master()
        atm = _round_to_50(entry_close)
        contract = _nearest_nifty_ce(master, atm)
        if contract is None:
            trades.append({"status": "error", "message": "could not resolve a live NIFTY CE contract",
                            "entry_time": entry_ts})
            i = (exit_idx + 1) if exit_idx is not None else len(rows)
            prev_hist = histogram[exit_idx] if exit_idx is not None else hist
            continue

        opt_rows = sorted(upstox_client.get_intraday_candles(contract["instrument_key"], "1minute"), key=lambda r: r[0])
        if not opt_rows:
            trades.append({"status": "error", "message": "no intraday candles yet for the resolved CE contract",
                            "entry_time": entry_ts, "trading_symbol": contract["trading_symbol"]})
            i = (exit_idx + 1) if exit_idx is not None else len(rows)
            prev_hist = histogram[exit_idx] if exit_idx is not None else hist
            continue

        def _premium_at_or_after(ts_target: str, rows_=opt_rows) -> tuple[str, float]:
            for r in rows_:
                if r[0] >= ts_target:
                    return r[0], r[4]
            return rows_[-1][0], rows_[-1][4]

        entry_time, entry_premium = _premium_at_or_after(entry_ts)

        trade = {
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
            trade.update({
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
            trade.update({
                "latest_time": last_ts,
                "latest_premium": last_premium,
                "unrealized_pnl_per_unit": round(last_premium - entry_premium, 2),
                "lot_size": contract.get("lot_size"),
            })

        trades.append(trade)

        if exit_idx is None:
            break  # still open -- nothing more to scan for today
        i = exit_idx + 1
        prev_hist = histogram[exit_idx]

    if not trades:
        return {"status": "no_signal_yet", "bars_so_far": len(rows), "trades": []}
    return {"status": "ok", "trades": trades}


def _trade_summary(trade: dict) -> str:
    if trade.get("status") == "error":
        return f"  [{trade['entry_time']}] Error: {trade['message']}"
    lines = [
        f"  Entry {trade['entry_time']} IST: spot close={trade['entry_index_close']:.2f}, "
        f"histogram={trade['entry_histogram']:.2f}",
        f"    Bought {trade['trading_symbol']} (strike {trade['strike']:.0f}) @ {trade['entry_premium']:.2f} "
        f"(fill {trade['entry_premium_time']})",
    ]
    if trade["status"] == "stopped_out":
        lines.append(
            f"    STOPPED at {trade['exit_time']} IST: spot close={trade['exit_index_close']:.2f} < entry "
            f"AND histogram={trade['exit_histogram']:.2f} < entry histogram"
        )
        lines.append(
            f"    Exit premium {trade['exit_premium']:.2f} (fill {trade['exit_premium_time']}) -- "
            f"P&L/unit: {trade['pnl_per_unit']:+.2f}"
        )
    else:
        lines.append(
            f"    Still OPEN as of {trade['latest_time']} IST: premium={trade['latest_premium']:.2f} -- "
            f"unrealized P&L/unit: {trade['unrealized_pnl_per_unit']:+.2f} (lot size {trade['lot_size']})"
        )
    return "\n".join(lines)


def summary(result: dict) -> str:
    status = result.get("status")
    if status == "insufficient_data":
        return f"Only {result['bars_so_far']} bars so far today -- MACD(12,26,9) needs ~35 to produce a real histogram value. Check back later."
    if status == "no_signal_yet":
        return f"No histogram zero-cross yet today ({result['bars_so_far']} bars so far)."

    trades = result["trades"]
    closed = [t for t in trades if t.get("status") == "stopped_out"]
    total_pnl = sum(t["pnl_per_unit"] for t in closed)
    header = f"{len(trades)} signal(s) today so far:"
    lines = [header, ""]
    for t in trades:
        lines.append(_trade_summary(t))
        lines.append("")
    if closed:
        lines.append(f"Closed trades: {len(closed)}, total P&L/unit so far: {total_pnl:+.2f}")
    return "\n".join(lines).rstrip()


if __name__ == "__main__":
    res = run()
    print(summary(res))
