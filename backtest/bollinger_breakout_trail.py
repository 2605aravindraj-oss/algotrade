"""Multi-timeframe Bollinger Band breakout, traded on NIFTY futures
notional (not options).

Per day, continuously (can fire more than once, one position at a time):

1. Bollinger Bands(period, std_mult) on 5-minute candles, continuous
   across the whole date range (needs warm-up).
2. Whenever a 5-minute candle CLOSES outside either band (above the
   upper band or below the lower band), mark that candle's own High and
   Low as the pattern levels and note the direction (up/down). This
   replaces any earlier still-pending pattern -- the most recent
   breakout candle is what's being watched.
3. Scan 1-minute candles strictly AFTER that 5-min candle's end time:
   the first one whose High breaks above the pattern High (if the
   5-min breakout was up) -- or whose Low breaks below the pattern Low
   (if down) -- triggers entry, filled at the pattern level itself
   (stop-order style).
4. Once in a trade, ride it with a trailing stop -- no fixed target:
     LONG:  stop = highest 1-min high seen since entry, minus
            trail_points; ratchets up only, never down.
     SHORT: stop = lowest 1-min low seen since entry, plus
            trail_points; ratchets down only, never up.
   Exit the instant a subsequent 1-min bar touches the current stop
   level. While in a trade, new breakout patterns are still tracked
   (so the next setup is ready the moment this trade closes) but no
   new entry is taken until flat. Pass trail_points=None to disable
   this entirely and just hold to end of day instead (see 5).
5. Forced flat at FORCE_FLAT_TIME (no overnight position); any pending
   pattern lapses at day end and a fresh one must form the next day.
   With trail_points=None this is the ONLY exit -- one trade per
   pattern, held all the way to the close.

Reuses rsi2_reversion.Trade and its futures-style cost model.
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest.rsi2_reversion import Trade, FORCE_FLAT_TIME, UNDERLYING_KEY
from backtest.sweep_reclaim_breakout import _resample


def _bollinger_bands(closes: list[float], period: int = 20, std_mult: float = 2.0):
    n = len(closes)
    upper: list[float | None] = [None] * n
    lower: list[float | None] = [None] * n
    for i in range(period - 1, n):
        window = closes[i - period + 1:i + 1]
        m = sum(window) / period
        var = sum((x - m) ** 2 for x in window) / period
        sd = var ** 0.5
        upper[i] = m + std_mult * sd
        lower[i] = m - std_mult * sd
    return upper, lower


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    bb_period: int = 20,
    bb_std: float = 2.0,
    trail_points: float | None = 20.0,
    lot_size: int = 65,
) -> list[Trade]:
    """trail_points=None disables the trailing stop entirely -- once
    entered, the position just holds until forced flat at
    FORCE_FLAT_TIME (or the pattern-search behavior described above for
    when a new entry is taken)."""
    trading_days = upstox_client.get_daily_history(underlying_key, from_date, to_date)
    trades: list[Trade] = []

    for day in trading_days:
        d = day["date"]
        rows_1min = sorted(
            cache.get_day_candles_cached(underlying_key, "1minute", d, expired=False),
            key=lambda c: c[0],
        )
        if not rows_1min:
            continue

        bars_5 = _resample(rows_1min, 5)
        if len(bars_5) < bb_period:
            continue
        closes_5 = [b[4] for b in bars_5]
        upper, lower = _bollinger_bands(closes_5, bb_period, bb_std)

        pattern = None  # dict: direction, high, low, end_time

        def _bar_end_time(bar_ts: str, bar_minutes: int) -> str:
            hh, mm = int(bar_ts[11:13]), int(bar_ts[14:16])
            total = hh * 60 + mm + bar_minutes
            eh, em = divmod(total, 60)
            return f"{eh:02d}:{em:02d}"

        # index 5-min patterns by their end time so we know, scanning
        # 1-min bars in order, exactly when a fresh pattern becomes active
        pattern_events: dict[str, dict] = {}
        for i, bar in enumerate(bars_5):
            if upper[i] is None:
                continue
            ts5, o5, h5, l5, c5, v5, oi5 = bar
            if c5 > upper[i]:
                pattern_events[_bar_end_time(ts5, 5)] = {"direction": "up", "high": h5, "low": l5}
            elif c5 < lower[i]:
                pattern_events[_bar_end_time(ts5, 5)] = {"direction": "down", "high": h5, "low": l5}

        position: Trade | None = None
        stop_level: float | None = None
        extreme: float | None = None  # highest high (long) / lowest low (short) since entry

        for row in rows_1min:
            ts, o, h, l, c, v, oi = row
            time_str = ts[11:16]

            if time_str in pattern_events:
                pattern = pattern_events[time_str]

            if position is not None:
                if time_str >= FORCE_FLAT_TIME:
                    position.exit_time, position.exit_price, position.exit_reason = ts, c, "eod"
                    trades.append(position)
                    position = None
                    pattern = None
                    continue
                if trail_points is not None:
                    if position.direction == "LONG":
                        extreme = max(extreme, h)
                        new_stop = extreme - trail_points
                        stop_level = max(stop_level, new_stop)
                        if l <= stop_level:
                            position.exit_time, position.exit_price, position.exit_reason = ts, stop_level, "trailing_stop"
                            trades.append(position)
                            position = None
                            pattern = None
                    else:
                        extreme = min(extreme, l)
                        new_stop = extreme + trail_points
                        stop_level = min(stop_level, new_stop)
                        if h >= stop_level:
                            position.exit_time, position.exit_price, position.exit_reason = ts, stop_level, "trailing_stop"
                            trades.append(position)
                            position = None
                            pattern = None
                continue

            if pattern is None or time_str >= FORCE_FLAT_TIME:
                continue

            if pattern["direction"] == "up" and h > pattern["high"]:
                entry_price = pattern["high"]
                position = Trade(date=d, direction="LONG", entry_time=ts, entry_price=entry_price, lot_size=lot_size)
                extreme = h
                stop_level = (entry_price - trail_points) if trail_points is not None else None
                pattern = None
            elif pattern["direction"] == "down" and l < pattern["low"]:
                entry_price = pattern["low"]
                position = Trade(date=d, direction="SHORT", entry_time=ts, entry_price=entry_price, lot_size=lot_size)
                extreme = l
                stop_level = (entry_price + trail_points) if trail_points is not None else None
                pattern = None

        if position is not None:
            last = rows_1min[-1]
            position.exit_time, position.exit_price, position.exit_reason = last[0], last[4], "eod_data_end"
            trades.append(position)

    return trades


def summary(trades: list[Trade]) -> str:
    from backtest.sweep_reclaim_breakout import summary as _summary
    return _summary(trades)
