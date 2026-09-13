"""Intraday MACD-trend + RSI(2)-momentum scalp on 1-minute bars.

Different from backtest/rsi2_reversion.py's mean-reversion logic -- this
trades WITH the trend, using RSI(2) strength as a momentum-continuation
trigger rather than an exhaustion/reversal trigger:

- MACD(12,26,9) on 1-minute closes sets the trend regime:
    uptrend   if MACD line > signal line
    downtrend if MACD line < signal line
  Computed continuously across the whole date range (not reset daily) --
  MACD is a slower trend filter and needs continuity to mean anything.
- Long entry:  uptrend   AND RSI(2) crosses above rsi_entry_high (70)
- Short entry: downtrend AND RSI(2) crosses below rsi_entry_low (30)
- Exit: RSI(2) reverts back through the midline, the MACD trend flips
  against the position, a stop-loss, a max holding time, or forced flat
  before close (no overnight position).

Trades the underlying directly (index/stock), not options.
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest.rsi2_reversion import (
    UNDERLYING_KEY, FORCE_FLAT_TIME, Trade, _minutes_between,
)


def _ema(values: list[float], period: int) -> list[float | None]:
    if len(values) < period:
        return [None] * len(values)
    k = 2 / (period + 1)
    out: list[float | None] = [None] * len(values)
    out[period - 1] = sum(values[:period]) / period
    for i in range(period, len(values)):
        out[i] = values[i] * k + out[i - 1] * (1 - k)
    return out


def compute_macd(
    closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[list[float | None], list[float | None]]:
    """Returns (macd_line, signal_line), index-aligned with closes."""
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = [
        (f - s) if (f is not None and s is not None) else None
        for f, s in zip(ema_fast, ema_slow)
    ]
    macd_values = [v for v in macd_line if v is not None]
    start_idx = next((i for i, v in enumerate(macd_line) if v is not None), len(macd_line))
    sig_ema = _ema(macd_values, signal)
    signal_line: list[float | None] = [None] * len(closes)
    for j, val in enumerate(sig_ema):
        if val is not None:
            signal_line[start_idx + j] = val
    return macd_line, signal_line


def compute_rsi(closes: list[float], period: int = 2) -> list[float | None]:
    n = len(closes)
    rsi: list[float | None] = [None] * n
    if n <= period:
        return rsi
    gains = [0.0] * n
    losses = [0.0] * n
    for i in range(1, n):
        change = closes[i] - closes[i - 1]
        gains[i] = max(change, 0.0)
        losses[i] = max(-change, 0.0)
    avg_gain = sum(gains[1:period + 1]) / period
    avg_loss = sum(losses[1:period + 1]) / period

    def _r(ag: float, al: float) -> float:
        return 100.0 if al == 0 else 100 - (100 / (1 + ag / al))

    rsi[period] = _r(avg_gain, avg_loss)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rsi[i] = _r(avg_gain, avg_loss)
    return rsi


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    rsi_period: int = 2,
    rsi_entry_high: float = 70,
    rsi_entry_low: float = 30,
    rsi_exit_mid: float = 50,
    stop_loss_pct: float = 0.3,
    max_hold_minutes: int | None = 30,
    lot_size: int = 65,
    entry_cutoff: str = "15:00",
) -> list[Trade]:
    trading_days = upstox_client.get_daily_history(underlying_key, from_date, to_date)

    rows: list[list] = []
    for day in trading_days:
        day_candles = cache.get_day_candles_cached(underlying_key, "1minute", day["date"], expired=False)
        rows.extend(sorted(day_candles, key=lambda c: c[0]))
    rows.sort(key=lambda c: c[0])
    if len(rows) < macd_slow + macd_signal + 5:
        return []

    closes = [r[4] for r in rows]
    macd_line, signal_line = compute_macd(closes, macd_fast, macd_slow, macd_signal)
    rsi = compute_rsi(closes, rsi_period)

    trades: list[Trade] = []
    position: Trade | None = None
    prev_rsi: float | None = None

    for i, row in enumerate(rows):
        ts = row[0]
        d = ts[:10]
        time_str = ts[11:16]
        price = closes[i]

        if macd_line[i] is None or signal_line[i] is None or rsi[i] is None:
            prev_rsi = rsi[i]
            continue

        uptrend = macd_line[i] > signal_line[i]

        if position is not None:
            exit_reason = None
            if position.direction == "LONG":
                if rsi[i] < rsi_exit_mid:
                    exit_reason = "rsi_exit"
                elif not uptrend:
                    exit_reason = "trend_flip"
                elif price <= position.entry_price * (1 - stop_loss_pct / 100):
                    exit_reason = "stop_loss"
            else:
                if rsi[i] > rsi_exit_mid:
                    exit_reason = "rsi_exit"
                elif uptrend:
                    exit_reason = "trend_flip"
                elif price >= position.entry_price * (1 + stop_loss_pct / 100):
                    exit_reason = "stop_loss"
            if exit_reason is None and max_hold_minutes and position.entry_time[:10] == d and _minutes_between(position.entry_time[11:16], time_str) >= max_hold_minutes:
                exit_reason = "max_hold"
            if exit_reason is None and time_str >= FORCE_FLAT_TIME:
                exit_reason = "eod"
            if exit_reason:
                position.exit_time = ts
                position.exit_price = price
                position.exit_reason = exit_reason
                trades.append(position)
                position = None
            prev_rsi = rsi[i]
            continue

        if time_str < "09:16" or time_str >= entry_cutoff or prev_rsi is None:
            prev_rsi = rsi[i]
            continue

        if uptrend and prev_rsi <= rsi_entry_high < rsi[i]:
            position = Trade(date=d, direction="LONG", entry_time=ts, entry_price=price, lot_size=lot_size)
        elif not uptrend and prev_rsi >= rsi_entry_low > rsi[i]:
            position = Trade(date=d, direction="SHORT", entry_time=ts, entry_price=price, lot_size=lot_size)

        prev_rsi = rsi[i]

    if position is not None:
        position.exit_time = rows[-1][0]
        position.exit_price = rows[-1][4]
        position.exit_reason = "eod_data_end"
        trades.append(position)

    return trades


def summary(trades: list[Trade]) -> str:
    from backtest.rsi2_reversion import summary as _summary
    return _summary(trades)
