"""Supertrend stop-and-reverse scalp on 5-minute NIFTY futures bars,
traded on futures notional (not options).

Supertrend(period, multiplier) computed with Wilder's ATR, continuous
across the whole date range (needs a warm-up window before it settles).
Standard recursive band formula:

    hl2          = (high + low) / 2
    basic_upper  = hl2 + multiplier * ATR
    basic_lower  = hl2 - multiplier * ATR
    final_upper  = basic_upper if (basic_upper < prev_final_upper or
                                    prev_close > prev_final_upper)
                   else prev_final_upper
    final_lower  = basic_lower if (basic_lower > prev_final_lower or
                                    prev_close < prev_final_lower)
                   else prev_final_lower

Direction flips bullish when close breaks above final_upper, bearish
when close breaks below final_lower (else holds the previous direction,
band trailing as usual). Supertrend line = final_lower while bullish,
final_upper while bearish.

Trading rule (always in the market -- stop and reverse):
    direction flips bullish -> exit any short, go LONG at this bar's close
    direction flips bearish -> exit any long, go SHORT at this bar's close
Forced flat at FORCE_FLAT_TIME (no overnight position); the next flip
after that re-enters as usual. One trade open at a time.
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest.rsi2_5min_sar import _resample_5min
from backtest.rsi2_reversion import Trade, FORCE_FLAT_TIME, UNDERLYING_KEY


def _compute_supertrend(
    bars: list[list], period: int = 10, multiplier: float = 3.0
) -> list[int | None]:
    """Returns a direction series (1 = bullish, -1 = bearish, None during
    ATR warm-up) aligned 1:1 with `bars`."""
    n = len(bars)
    highs = [b[2] for b in bars]
    lows = [b[3] for b in bars]
    closes = [b[4] for b in bars]

    tr = [None] * n
    for i in range(n):
        if i == 0:
            tr[i] = highs[i] - lows[i]
        else:
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )

    atr: list[float | None] = [None] * n
    if n >= period:
        atr[period - 1] = sum(tr[:period]) / period
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    final_upper: list[float | None] = [None] * n
    final_lower: list[float | None] = [None] * n
    direction: list[int | None] = [None] * n

    for i in range(period - 1, n):
        hl2 = (highs[i] + lows[i]) / 2
        basic_upper = hl2 + multiplier * atr[i]
        basic_lower = hl2 - multiplier * atr[i]

        if i == period - 1:
            final_upper[i] = basic_upper
            final_lower[i] = basic_lower
            direction[i] = 1 if closes[i] > basic_upper else -1
            continue

        prev_fu, prev_fl = final_upper[i - 1], final_lower[i - 1]
        final_upper[i] = (
            basic_upper if (basic_upper < prev_fu or closes[i - 1] > prev_fu) else prev_fu
        )
        final_lower[i] = (
            basic_lower if (basic_lower > prev_fl or closes[i - 1] < prev_fl) else prev_fl
        )

        if closes[i] > final_upper[i]:
            direction[i] = 1
        elif closes[i] < final_lower[i]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

    return direction


def run(
    from_date: str,
    to_date: str,
    futures_key: str,
    period: int = 10,
    multiplier: float = 3.0,
    lot_size: int = 65,
    futures_expired: bool = False,
    access_token: str | None = None,
) -> list[Trade]:
    if futures_expired:
        raw_days = upstox_client.get_expired_candles(futures_key, "day", to_date, from_date, access_token)
        fut_days = [{"date": d} for d in sorted({c[0][:10] for c in raw_days})]
    else:
        fut_days = upstox_client.get_daily_history(futures_key, from_date, to_date)

    all_5min: list[list] = []
    for day in fut_days:
        rows_1min = sorted(
            cache.get_day_candles_cached(futures_key, "1minute", day["date"], expired=futures_expired, access_token=access_token),
            key=lambda c: c[0],
        )
        all_5min.extend(_resample_5min(rows_1min))
    all_5min.sort(key=lambda c: c[0])
    if len(all_5min) < period + 2:
        return []

    direction = _compute_supertrend(all_5min, period, multiplier)

    trades: list[Trade] = []
    position: Trade | None = None
    prev_dir: int | None = None
    current_day: str | None = None

    def _close(t: Trade, ts: str, price: float, reason: str) -> None:
        t.exit_time = ts
        t.exit_price = price
        t.exit_reason = reason
        trades.append(t)

    for i, bar in enumerate(all_5min):
        ts, o, h, l, c, v, oi = bar
        d = ts[:10]
        time_str = ts[11:16]
        dirn = direction[i]

        if d != current_day:
            current_day = d
            if position is not None:
                _close(position, ts, o, "eod")
                position = None
            prev_dir = None  # don't carry a stale flip across the overnight gap

        if dirn is None:
            continue

        if time_str >= FORCE_FLAT_TIME:
            if position is not None:
                _close(position, ts, c, "eod")
                position = None
            prev_dir = dirn
            continue

        if prev_dir is not None and dirn != prev_dir:
            if position is not None:
                _close(position, ts, c, "reverse")
            new_dir = "LONG" if dirn == 1 else "SHORT"
            position = Trade(date=d, direction=new_dir, entry_time=ts, entry_price=c, lot_size=lot_size)

        prev_dir = dirn

    if position is not None:
        last = all_5min[-1]
        _close(position, last[0], last[4], "eod_data_end")

    return trades


def summary(trades: list[Trade]) -> str:
    from backtest.rsi2_reversion import summary as _summary
    return _summary(trades)
