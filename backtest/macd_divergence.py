"""MACD regular divergence on 5-minute NIFTY futures bars, traded on
futures notional (not options).

Swing pivots: a 5-bar fractal (2 bars either side) on price high/low,
confirmed 2 bars after it forms (no lookahead).

Bullish divergence: the two most recent confirmed swing lows show price
making a LOWER low while the MACD(12,26,9) line value at that pivot is
HIGHER than at the prior pivot (momentum weakening despite the deeper
price low) -> LONG at the confirmation bar's close.

Bearish divergence: mirrored on swing highs -> SHORT.

Exit: stop-loss at the triggering pivot's own low/high, target = 2x that
stop distance, else forced flat at FORCE_FLAT_TIME. One position at a
time (a fresh divergence while already in a position is ignored).
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest.futures_oi_buildup import FORCE_FLAT_TIME
from backtest.macd_rsi2_momentum import compute_macd
from backtest.rsi2_5min_sar import _resample_5min
from backtest.rsi2_reversion import Trade, UNDERLYING_KEY


def _find_pivots(bars: list[list], macd_line: list[float | None], fractal: int = 2) -> tuple[list[int], list[int]]:
    """Returns (low_pivot_indices, high_pivot_indices), each index i has a
    confirmed pivot at i-fractal (needs `fractal` bars of lookahead)."""
    lows: list[int] = []
    highs: list[int] = []
    n = len(bars)
    for i in range(fractal, n - fractal):
        if macd_line[i] is None:
            continue
        window_low = [bars[j][3] for j in range(i - fractal, i + fractal + 1)]
        window_high = [bars[j][2] for j in range(i - fractal, i + fractal + 1)]
        if bars[i][3] == min(window_low) and window_low.count(bars[i][3]) == 1:
            lows.append(i)
        if bars[i][2] == max(window_high) and window_high.count(bars[i][2]) == 1:
            highs.append(i)
    return lows, highs


def run(
    from_date: str,
    to_date: str,
    futures_key: str,
    lot_size: int = 65,
    fractal: int = 2,
    target_r_multiple: float = 2.0,
) -> list[Trade]:
    fut_days = upstox_client.get_daily_history(futures_key, from_date, to_date)
    trades: list[Trade] = []

    for day in fut_days:
        d = day["date"]
        rows_1min = sorted(
            cache.get_day_candles_cached(futures_key, "1minute", d, expired=False),
            key=lambda c: c[0],
        )
        if not rows_1min:
            continue
        bars = _resample_5min(rows_1min)
        if len(bars) < 30:
            continue

        closes = [b[4] for b in bars]
        macd_line, signal_line = compute_macd(closes)

        low_pivots, high_pivots = _find_pivots(bars, macd_line, fractal)
        # confirmation happens `fractal` bars after the pivot itself
        low_confirm_at = {p: p + fractal for p in low_pivots}
        high_confirm_at = {p: p + fractal for p in high_pivots}

        position: Trade | None = None
        stop_level = None
        target_level = None

        # Build a schedule: at each confirmation index, what divergence (if any) fires
        confirm_events: dict[int, tuple[str, int, int]] = {}  # confirm_idx -> (direction, pivot1, pivot2)
        for k in range(1, len(low_pivots)):
            p1, p2 = low_pivots[k - 1], low_pivots[k]
            if bars[p2][3] < bars[p1][3] and macd_line[p2] > macd_line[p1]:
                confirm_events[low_confirm_at[p2]] = ("LONG", p1, p2)
        for k in range(1, len(high_pivots)):
            p1, p2 = high_pivots[k - 1], high_pivots[k]
            if bars[p2][2] > bars[p1][2] and macd_line[p2] < macd_line[p1]:
                confirm_events[high_confirm_at[p2]] = ("SHORT", p1, p2)

        for i, bar in enumerate(bars):
            ts, o, h, l, c, v, oi = bar
            time_str = ts[11:16]

            if position is not None:
                if time_str >= FORCE_FLAT_TIME:
                    position.exit_time, position.exit_price, position.exit_reason = ts, c, "eod"
                    trades.append(position)
                    position = None
                elif position.direction == "LONG" and l <= stop_level:
                    position.exit_time, position.exit_price, position.exit_reason = ts, stop_level, "stop_loss"
                    trades.append(position)
                    position = None
                elif position.direction == "LONG" and h >= target_level:
                    position.exit_time, position.exit_price, position.exit_reason = ts, target_level, "target"
                    trades.append(position)
                    position = None
                elif position.direction == "SHORT" and h >= stop_level:
                    position.exit_time, position.exit_price, position.exit_reason = ts, stop_level, "stop_loss"
                    trades.append(position)
                    position = None
                elif position.direction == "SHORT" and l <= target_level:
                    position.exit_time, position.exit_price, position.exit_reason = ts, target_level, "target"
                    trades.append(position)
                    position = None
                continue

            if i in confirm_events and time_str < FORCE_FLAT_TIME:
                direction, p1, p2 = confirm_events[i]
                if direction == "LONG":
                    stop_level = bars[p2][3]
                    risk = c - stop_level
                    if risk <= 0:
                        continue
                    target_level = c + target_r_multiple * risk
                else:
                    stop_level = bars[p2][2]
                    risk = stop_level - c
                    if risk <= 0:
                        continue
                    target_level = c - target_r_multiple * risk
                position = Trade(date=d, direction=direction, entry_time=ts, entry_price=c, lot_size=lot_size)

        if position is not None:
            last = bars[-1]
            position.exit_time, position.exit_price, position.exit_reason = last[0], last[4], "eod_data_end"
            trades.append(position)

    return trades


def summary(trades: list[Trade]) -> str:
    from backtest.rsi2_reversion import summary as _summary
    return _summary(trades)
