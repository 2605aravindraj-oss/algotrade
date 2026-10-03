"""Classic Larry Connors-style RSI mean reversion on DAILY bars, for a
single instrument (e.g. a NIFTY 50 constituent stock's NSE_EQ cash
candles).

WHY A SEPARATE MODULE FROM backtest/rsi2_reversion.py: that module is
intraday (1-minute bars, reset every day, forced flat before close,
a same-day VWAP regime filter). None of that carries over to a daily
bar: a daily "VWAP" isn't a meaningful regime filter the way an
intraday session VWAP is, and a daily-bar position is a multi-day
SWING/positional hold by construction, not an intraday scalp -- there
is no same-day force-flat here at all; a position rides until its own
exit condition fires, however many calendar days that takes.

SIGNAL: Wilder RSI(rsi_period) on daily closes (reusing rsi2_
reversion.compute_rsi -- the same calculation, just fed daily closes
instead of 1-minute ones).
    LONG entry:  RSI < oversold
    SHORT entry: RSI > 100 - oversold (mirror), only if allow_short
Both are pure mean-reversion entries with no trend/regime filter
layered on top -- deliberately the simplest version first, to see if
there's a bare edge before adding anything. (The classic Connors
rule also requires price above a long SMA before taking the long
side; that is NOT included here, and would be a natural next
parameter to sweep if the bare version shows promise.)

EXIT (checked in this order): stop_loss_pct (if set) -> RSI crosses
back through exit_threshold (LONG: RSI > exit_threshold; SHORT:
RSI < 100 - exit_threshold) -> max_hold_days (if set, calendar days
since entry, not trading days). No forced-flat -- a position can
carry indefinitely until one of these fires or the data ends.

COSTS: reuses backtest.rsi2_reversion's futures-notional cost
approximation (flat brokerage + ~0.0255% of notional round trip) as
a stand-in for equity costs -- real delivery-style equity costs
differ (STT is higher, 0.1% sell-side, since a multi-day hold
settles as delivery, not an intraday MIS square-off; many discount
brokers charge zero brokerage on delivery). Treat cost figures here
as approximate, same caveat as every other cost model in this
codebase.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from data_sources import upstox_client
from backtest.rsi2_reversion import Trade, compute_rsi

_MAX_CHUNK_DAYS = 365 * 8  # the daily historical-candle endpoint rejects spans beyond ~9-10 years


def _get_daily_history_chunked(instrument_key: str, from_date: str, to_date: str) -> list[dict]:
    """upstox_client.get_daily_history errors (400) on a span much
    beyond ~9-10 years -- a daily RSI strategy needs more history than
    that for a meaningful sample, so fetch in <=8-year chunks and
    concatenate (deduping by date, in case chunk boundaries overlap)."""
    start = datetime.strptime(from_date, "%Y-%m-%d")
    end = datetime.strptime(to_date, "%Y-%m-%d")
    by_date: dict[str, dict] = {}
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=_MAX_CHUNK_DAYS), end)
        chunk = upstox_client.get_daily_history(
            instrument_key, chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        )
        for d in chunk:
            by_date[d["date"]] = d
        chunk_start = chunk_end + timedelta(days=1)
    return list(by_date.values())


def run(
    from_date: str,
    to_date: str,
    instrument_key: str,
    rsi_period: int = 14,
    oversold: float = 30,
    exit_threshold: float = 50,
    stop_loss_pct: float | None = None,
    max_hold_days: int | None = None,
    quantity: int = 100,
    allow_short: bool = False,
) -> list[Trade]:
    days = _get_daily_history_chunked(instrument_key, from_date, to_date)
    days.sort(key=lambda d: d["date"])
    if len(days) < rsi_period + 5:
        return []

    closes = [d["close"] for d in days]
    rsi = compute_rsi(closes, rsi_period)

    trades: list[Trade] = []
    position: Trade | None = None

    for i, day in enumerate(days):
        d = day["date"]
        price = closes[i]
        if rsi[i] is None:
            continue

        if position is not None:
            exit_reason = None
            is_long = position.direction == "LONG"
            if stop_loss_pct is not None:
                hit_sl = (
                    (is_long and price <= position.entry_price * (1 - stop_loss_pct / 100))
                    or (not is_long and price >= position.entry_price * (1 + stop_loss_pct / 100))
                )
                if hit_sl:
                    exit_reason = "stop_loss"
            if exit_reason is None:
                if is_long and rsi[i] > exit_threshold:
                    exit_reason = "rsi_exit"
                elif not is_long and rsi[i] < (100 - exit_threshold):
                    exit_reason = "rsi_exit"
            if exit_reason is None and max_hold_days is not None:
                held_days = (datetime.strptime(d, "%Y-%m-%d") - datetime.strptime(position.date, "%Y-%m-%d")).days
                if held_days >= max_hold_days:
                    exit_reason = "max_hold"
            if exit_reason:
                position.exit_time = d
                position.exit_price = price
                position.exit_reason = exit_reason
                trades.append(position)
                position = None
            continue

        if rsi[i] < oversold:
            position = Trade(date=d, direction="LONG", entry_time=d, entry_price=price, lot_size=quantity)
        elif allow_short and rsi[i] > (100 - oversold):
            position = Trade(date=d, direction="SHORT", entry_time=d, entry_price=price, lot_size=quantity)

    if position is not None:
        position.exit_time = days[-1]["date"]
        position.exit_price = closes[-1]
        position.exit_reason = "data_end"
        trades.append(position)

    return trades


def summary(trades: list[Trade]) -> str:
    from backtest.rsi2_reversion import summary as _summary
    return _summary(trades)
