"""Live, TODAY-ONLY check: MACD(12,26,9) histogram zero-cross on NIFTY 50
spot 1-minute candles -- buy ATM CE on a bullish cross, buy ATM PE on a
bearish cross, each filled at the crossing candle's own close.

Not a backtest -- this looks at TODAY's candles only (for the index and
for each resolved ATM CE/PE contract) and reports EVERY signal that's
fired today so far, in order: closed (stopped-out) trades and, if the
most recent one hasn't stopped yet, the currently open trade. After a
stop-loss, scanning resumes from the very next bar looking for a fresh
cross in EITHER direction -- a choppy morning can produce several
signals in a row, CE and PE alike (seen live: 2026-10-07 fired at 10:00
IST, stopped out at 10:01, then fired again at 10:05 and stayed open --
an earlier version of this script only reported the first signal of
the day and silently dropped every later one, which is the bug this
fixed; a later version only ever looked for bullish crosses/CE, which
is the gap this fixes).

SIGNAL: MACD(12,26,9) on the index's own 1-minute closes. The EMA(12),
EMA(26), and signal EMA(9) are warmed up on HISTORY_DAYS previous
trading days' 1-minute closes (prepended before today's bars, same
public historical-candle data bullish_pattern_screener.py uses for the
index), so they're already converged by the time today's first bar
arrives -- a continuously-running MACD, matching what a charting
platform shows, rather than reseeding from an SMA every morning (an
earlier version of this script did reseed daily, which made its
values diverge from a live chart for the first ~30-40 minutes after
the open). Only crosses that occur ON today's own bars are ever
scanned/traded -- the prior days' bars exist purely for EMA warm-up.
  - A fresh cross from <=0 to >0 (not merely "is positive") opens a
    LONG/CE trade: buy the ATM CE, filled at that candle's own close.
  - A fresh cross from >=0 to <0 opens a SHORT/PE trade: buy the ATM
    PE, filled at that candle's own close.
Only one position open at a time -- the scan for the NEXT entry (CE or
PE, whichever crosses first) only resumes after the current one's stop
triggers.

STOP-LOSS (as specified -- a dual condition, not price alone and not
histogram alone -- mirrored for the two trade directions since a CE
is a bullish bet and a PE a bearish one):
  - CE: the first LATER candle whose close is BELOW the REFERENCE
    candle's own close AND whose histogram is BELOW the reference
    candle's own histogram value.
  - PE: the mirror -- the first LATER candle whose close is ABOVE the
    reference candle's own close AND whose histogram is ABOVE the
    reference candle's own histogram value.
Both conditions must hold on the same candle. Requiring both avoids a
single noisy wick (price moves but momentum hasn't actually weakened,
or vice versa) stopping the trade out. The reference starts as the
entry candle itself.

TARGET + TRAILING (target_rs, default 150.0): once unrealized P&L for
one lot (option premium move x lot_size -- the same formula for CE and
PE, since both are long option positions whose premium rises as the
trade works in its favour) first reaches +Rs150, the stop-loss
REFERENCE switches from the fixed entry candle to the candle that just
set that new peak, and keeps ratcheting forward to whichever later
candle sets a new peak P&L after that -- the dual stop condition above
(mirrored per direction) is then checked against this trailing
reference instead of the entry. This locks in progressively more gain
without capping the upside at a flat Rs150 exit: the trade can keep
running as long as price and histogram don't BOTH fall back below (CE)
or above (PE) the latest peak candle. Before the first time Rs150 is
reached, the stop is anchored to the entry candle exactly as before.

STRIKE/EXPIRY: ATM = round-to-50 of the index close at the entry
candle; nearest weekly expiry today, resolved from Upstox's live
instrument master (NSE_FO segment, underlying NIFTY, instrument_type
CE or PE to match the cross direction) -- the same master-loading
pattern live_nifty_insights.py uses, since this is a LIVE
(non-expired) contract, not one of this codebase's cached
expired-instruments.

DATA: both the index and the CE/PE contract's candles come from
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

from data_sources import cache, upstox_client
from backtest.technical_rating import _macd

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"
MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
IST = timezone(timedelta(hours=5, minutes=30))
FORCE_FLAT_TIME = "15:25"
HISTORY_DAYS = 3  # trading days of 1-minute closes prepended before today, purely to warm up the EMA(12)/EMA(26)/signal-EMA(9) so MACD is continuous (not reseeded each morning)


def _recent_trading_days(n: int) -> list[str]:
    """The last n COMPLETED trading days before today (index calendar,
    public daily-history endpoint -- correctly skips weekends/holidays)."""
    import datetime as _dt
    today = _dt.date.today()
    lookback_from = (today - _dt.timedelta(days=n * 3 + 5)).isoformat()
    days = upstox_client.get_daily_history(UNDERLYING_KEY, lookback_from, (today - _dt.timedelta(days=1)).isoformat())
    return [d["date"] for d in days[-n:]]


def _load_master() -> list[dict]:
    resp = requests.get(MASTER_URL, timeout=30)
    resp.raise_for_status()
    with gzip.open(BytesIO(resp.content)) as f:
        return json.load(f)


def _nearest_nifty_option(master: list[dict], strike: float, instrument_type: str) -> dict | None:
    opts = [
        d for d in master
        if d.get("segment") == "NSE_FO" and d.get("underlying_symbol") == "NIFTY"
        and d.get("instrument_type") == instrument_type
    ]
    if not opts:
        return None
    nearest_expiry = min(d["expiry"] for d in opts)
    chain = [d for d in opts if d["expiry"] == nearest_expiry]
    return min(chain, key=lambda d: abs(d["strike_price"] - strike))


def _round_to_50(x: float) -> int:
    return int(round(x / 50) * 50)


def run(target_rs: float = 150.0) -> dict:
    today_rows = sorted(upstox_client.get_intraday_candles(UNDERLYING_KEY, "1minute"), key=lambda r: r[0])
    if not today_rows:
        return {"status": "insufficient_data", "bars_so_far": 0, "trades": []}

    history_rows: list[list] = []
    for d in _recent_trading_days(HISTORY_DAYS):
        history_rows.extend(cache.get_day_candles_cached(UNDERLYING_KEY, "1minute", d, expired=False))
    history_rows.sort(key=lambda r: r[0])

    rows = history_rows + today_rows  # history is for EMA warm-up only -- crosses are only ever scanned from today_start_idx onward
    today_start_idx = len(history_rows)

    closes = [r[4] for r in rows]
    macd_line, signal_line = _macd(closes, 12, 26, 9)
    histogram = [(m - s) if (m is not None and s is not None) else None for m, s in zip(macd_line, signal_line)]

    if today_start_idx > 0 and histogram[today_start_idx - 1] is None:
        return {"status": "insufficient_data", "bars_so_far": len(today_rows), "trades": []}

    master = None  # lazy-loaded only once a first signal actually fires
    trades: list[dict] = []
    i = today_start_idx
    prev_hist = histogram[today_start_idx - 1] if today_start_idx > 0 else None
    while i < len(rows):
        hist = histogram[i]
        if hist is None:
            prev_hist = hist
            i += 1
            continue
        if prev_hist is None:
            prev_hist = hist
            i += 1
            continue
        if prev_hist <= 0 and hist > 0:
            direction = "CE"
        elif prev_hist >= 0 and hist < 0:
            direction = "PE"
        else:
            prev_hist = hist
            i += 1
            continue

        # fresh cross at bar i -- open a trade
        entry_idx = i
        entry_row = rows[entry_idx]
        entry_close = entry_row[4]
        entry_hist = histogram[entry_idx]
        entry_ts = entry_row[0]

        if master is None:
            master = _load_master()
        atm = _round_to_50(entry_close)
        contract = _nearest_nifty_option(master, atm, direction)
        if contract is None:
            trades.append({"status": "error", "direction": direction,
                            "message": f"could not resolve a live NIFTY {direction} contract",
                            "entry_time": entry_ts})
            i = entry_idx + 1
            prev_hist = hist
            continue

        opt_rows = sorted(upstox_client.get_intraday_candles(contract["instrument_key"], "1minute"), key=lambda r: r[0])
        if not opt_rows:
            trades.append({"status": "error", "direction": direction,
                            "message": f"no intraday candles yet for the resolved {direction} contract",
                            "entry_time": entry_ts, "trading_symbol": contract["trading_symbol"]})
            i = entry_idx + 1
            prev_hist = hist
            continue

        def _premium_at_or_after(ts_target: str, rows_=opt_rows) -> tuple[str, float]:
            for r in rows_:
                if r[0] >= ts_target:
                    return r[0], r[4]
            return rows_[-1][0], rows_[-1][4]

        entry_time, entry_premium = _premium_at_or_after(entry_ts)
        lot_size = contract.get("lot_size") or 1

        # scan forward for the exit: dual-condition stop, anchored to the
        # entry candle until peak P&L/lot first reaches target_rs, then
        # re-anchored ("trailed") to whichever later candle sets each new
        # peak after that.
        ref_close, ref_hist = entry_close, entry_hist
        peak_pnl_per_lot = 0.0
        trailing_active = False
        exit_idx = None
        exit_reason = None
        for j in range(entry_idx + 1, len(rows)):
            c = rows[j][4]
            h = histogram[j]
            if h is None:
                continue
            _, premium_j = _premium_at_or_after(rows[j][0])
            pnl_per_lot = (premium_j - entry_premium) * lot_size
            if pnl_per_lot > peak_pnl_per_lot:
                peak_pnl_per_lot = pnl_per_lot
                if peak_pnl_per_lot >= target_rs:
                    trailing_active = True
                    ref_close, ref_hist = c, h
            stop_hit = (c < ref_close and h < ref_hist) if direction == "CE" else (c > ref_close and h > ref_hist)
            if stop_hit:
                exit_idx = j
                exit_reason = "trailing_stop" if trailing_active else "stop_loss"
                break

        trade = {
            "status": "open" if exit_idx is None else "stopped_out",
            "direction": direction,
            "entry_time": entry_ts,
            "entry_index_close": entry_close,
            "entry_histogram": entry_hist,
            "strike": contract["strike_price"],
            "trading_symbol": contract["trading_symbol"],
            "instrument_key": contract["instrument_key"],
            "entry_premium_time": entry_time,
            "entry_premium": entry_premium,
            "lot_size": lot_size,
            "target_rs": target_rs,
        }

        if exit_idx is not None:
            exit_ts = rows[exit_idx][0]
            exit_index_close = rows[exit_idx][4]
            exit_hist = histogram[exit_idx]
            exit_time, exit_premium = _premium_at_or_after(exit_ts)
            pnl_per_unit = round(exit_premium - entry_premium, 2)
            trade.update({
                "exit_time": exit_ts,
                "exit_index_close": exit_index_close,
                "exit_histogram": exit_hist,
                "exit_premium_time": exit_time,
                "exit_premium": exit_premium,
                "exit_reason": exit_reason,
                "pnl_per_unit": pnl_per_unit,
                "pnl_per_lot": round(pnl_per_unit * lot_size, 2),
                "peak_pnl_per_lot": round(peak_pnl_per_lot, 2),
                "trailing_was_active": trailing_active,
            })
        else:
            last_ts, last_premium = opt_rows[-1][0], opt_rows[-1][4]
            unrealized_pnl_per_lot = round((last_premium - entry_premium) * lot_size, 2)
            trade.update({
                "latest_time": last_ts,
                "latest_premium": last_premium,
                "unrealized_pnl_per_unit": round(last_premium - entry_premium, 2),
                "unrealized_pnl_per_lot": unrealized_pnl_per_lot,
                "peak_pnl_per_lot": round(peak_pnl_per_lot, 2),
                "trailing_active": trailing_active,
                "trailing_ref_close": ref_close if trailing_active else None,
                "trailing_ref_histogram": ref_hist if trailing_active else None,
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
        return f"  [{trade['entry_time']}] Error ({trade.get('direction', '?')}): {trade['message']}"
    direction = trade["direction"]
    lines = [
        f"  Entry {trade['entry_time']} IST ({direction}): spot close={trade['entry_index_close']:.2f}, "
        f"histogram={trade['entry_histogram']:.2f}",
        f"    Bought {trade['trading_symbol']} (strike {trade['strike']:.0f}) @ {trade['entry_premium']:.2f} "
        f"(fill {trade['entry_premium_time']}, lot size {trade['lot_size']})",
    ]
    if trade["status"] == "stopped_out":
        ref_kind = "TRAILED reference (peak after target hit)" if trade["trailing_was_active"] else "entry reference"
        cmp_word = "<" if direction == "CE" else ">"
        lines.append(
            f"    STOPPED ({trade['exit_reason']}) at {trade['exit_time']} IST: spot close="
            f"{trade['exit_index_close']:.2f} {cmp_word} {ref_kind} AND histogram={trade['exit_histogram']:.2f} "
            f"{cmp_word} it, both on the same bar"
        )
        lines.append(
            f"    Exit premium {trade['exit_premium']:.2f} (fill {trade['exit_premium_time']}) -- "
            f"P&L/unit: {trade['pnl_per_unit']:+.2f}  P&L/lot: {trade['pnl_per_lot']:+.2f}  "
            f"(peak P&L/lot reached: {trade['peak_pnl_per_lot']:+.2f}, target was {trade['target_rs']:.0f})"
        )
    else:
        trail_note = (
            f" -- TRAILING ACTIVE (target {trade['target_rs']:.0f} reached, peak {trade['peak_pnl_per_lot']:+.2f}/lot, "
            f"stop now anchored to spot={trade['trailing_ref_close']:.2f}/hist={trade['trailing_ref_histogram']:.2f})"
            if trade["trailing_active"] else f" (target {trade['target_rs']:.0f}/lot not yet reached)"
        )
        lines.append(
            f"    Still OPEN as of {trade['latest_time']} IST: premium={trade['latest_premium']:.2f} -- "
            f"unrealized P&L/unit: {trade['unrealized_pnl_per_unit']:+.2f}, P&L/lot: "
            f"{trade['unrealized_pnl_per_lot']:+.2f}{trail_note}"
        )
    return "\n".join(lines)


def summary(result: dict) -> str:
    status = result.get("status")
    if status == "insufficient_data":
        if result["bars_so_far"] == 0:
            return "No candles for today yet -- market may not have opened. Check back later."
        return (
            f"Only {result['bars_so_far']} bars today, and the {HISTORY_DAYS}-day EMA warm-up "
            "history couldn't produce a real histogram value either -- check back later."
        )
    if status == "no_signal_yet":
        return f"No histogram zero-cross yet today ({result['bars_so_far']} bars so far)."

    trades = result["trades"]
    closed = [t for t in trades if t.get("status") == "stopped_out"]
    total_pnl_unit = sum(t["pnl_per_unit"] for t in closed)
    total_pnl_lot = sum(t["pnl_per_lot"] for t in closed)
    header = f"{len(trades)} signal(s) today so far:"
    lines = [header, ""]
    for t in trades:
        lines.append(_trade_summary(t))
        lines.append("")
    if closed:
        lines.append(
            f"Closed trades: {len(closed)}, total P&L/unit so far: {total_pnl_unit:+.2f}, "
            f"total P&L/lot: {total_pnl_lot:+.2f}"
        )
    return "\n".join(lines).rstrip()


if __name__ == "__main__":
    res = run()
    print(summary(res))
