"""5-minute RSI(2) breakout stop-and-reverse (SAR) scalp.

- RSI(2) computed on 5-minute closes, continuous across the date range.
- Long entry:  RSI(2) crosses above 70, confirmed by that candle's close
  (this candle's RSI(2) > 70, previous candle's RSI(2) <= 70). Enter long
  AT that candle's close. Stop-loss = that entry candle's low.
- Short entry: mirror -- RSI(2) crosses below 30 confirmed by close,
  enter short at that candle's close, stop-loss = that entry candle's
  high.
- While in a position, each subsequent candle is checked against the
  stop level:
    LONG:  if candle's low <= stop AND candle's close < stop ->
           exit the long AND immediately reverse into a new short, both
           at this candle's close. New short's stop = this candle's high.
    SHORT: mirror (high >= stop AND close > stop -> reverse to long).
  A stop only touched intrabar (low <= stop) but closing back above it
  does NOT trigger -- confirmation requires the close, exactly as
  specified.
- A fresh RSI(2) cross while already in a position changes nothing; only
  the stop-hit-and-close-through rule creates a new entry.
- No separate profit target -- always in the market, long or short,
  until forced flat before the day's close (no overnight position).

Trades the underlying directly (not options) -- same lightweight
futures-style cost model as rsi2_reversion.py.
"""
from __future__ import annotations

from dataclasses import dataclass

from data_sources import cache, upstox_client
from backtest.macd_rsi2_momentum import compute_rsi
from backtest.rsi2_reversion import UNDERLYING_KEY, FORCE_FLAT_TIME, Trade


def _resample_5min(rows_1min: list[list]) -> list[list]:
    """Aggregate 1-minute candles into 5-minute candles aligned to the
    09:15 session open. Works across multiple days in one pass since the
    date prefix in each timestamp keeps buckets from different days apart.
    """
    buckets: dict[str, list] = {}
    order: list[str] = []
    for row in rows_1min:
        ts, o, h, l, c, v, oi = row
        hh, mm = int(ts[11:13]), int(ts[14:16])
        minutes_since_open = (hh * 60 + mm) - (9 * 60 + 15)
        bucket_idx = max(minutes_since_open, 0) // 5
        bucket_start_minutes = 9 * 60 + 15 + bucket_idx * 5
        bh, bm = divmod(bucket_start_minutes, 60)
        bucket_ts = f"{ts[:11]}{bh:02d}:{bm:02d}:00{ts[19:]}"
        if bucket_ts not in buckets:
            buckets[bucket_ts] = [o, h, l, c, v, oi]
            order.append(bucket_ts)
        else:
            b = buckets[bucket_ts]
            b[1] = max(b[1], h)
            b[2] = min(b[2], l)
            b[3] = c
            b[4] += v
            b[5] = oi
    return [[ts] + buckets[ts] for ts in order]


@dataclass
class _OpenPosition:
    direction: str
    entry_time: str
    entry_price: float
    stop_level: float
    date: str


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    rsi_period: int = 2,
    rsi_high: float = 70,
    rsi_low: float = 30,
    lot_size: int = 65,
    entry_cutoff: str = "15:00",
) -> list[Trade]:
    trading_days = upstox_client.get_daily_history(underlying_key, from_date, to_date)

    all_5min: list[list] = []
    for day in trading_days:
        rows_1min = sorted(
            cache.get_day_candles_cached(underlying_key, "1minute", day["date"], expired=False),
            key=lambda c: c[0],
        )
        all_5min.extend(_resample_5min(rows_1min))
    all_5min.sort(key=lambda c: c[0])
    if len(all_5min) < rsi_period + 2:
        return []

    closes = [r[4] for r in all_5min]
    rsi = compute_rsi(closes, rsi_period)

    trades: list[Trade] = []
    position: _OpenPosition | None = None
    prev_rsi: float | None = None

    def _close_trade(pos: _OpenPosition, ts: str, price: float, reason: str) -> None:
        t = Trade(date=pos.date, direction=pos.direction, entry_time=pos.entry_time,
                   entry_price=pos.entry_price, lot_size=lot_size)
        t.exit_time = ts
        t.exit_price = price
        t.exit_reason = reason
        trades.append(t)

    for i, row in enumerate(all_5min):
        ts, o, h, l, c, v, oi = row
        d = ts[:10]
        time_str = ts[11:16]

        if position is not None:
            reversed_this_bar = False
            if position.direction == "LONG" and l <= position.stop_level and c < position.stop_level:
                _close_trade(position, ts, c, "stop_reverse")
                position = _OpenPosition(direction="SHORT", entry_time=ts, entry_price=c, stop_level=h, date=d)
                reversed_this_bar = True
            elif position.direction == "SHORT" and h >= position.stop_level and c > position.stop_level:
                _close_trade(position, ts, c, "stop_reverse")
                position = _OpenPosition(direction="LONG", entry_time=ts, entry_price=c, stop_level=l, date=d)
                reversed_this_bar = True

            if not reversed_this_bar and time_str >= FORCE_FLAT_TIME:
                _close_trade(position, ts, c, "eod")
                position = None

            prev_rsi = rsi[i]
            continue

        if rsi[i] is None or prev_rsi is None:
            prev_rsi = rsi[i]
            continue
        if time_str >= entry_cutoff:
            prev_rsi = rsi[i]
            continue

        if prev_rsi <= rsi_high < rsi[i]:
            position = _OpenPosition(direction="LONG", entry_time=ts, entry_price=c, stop_level=l, date=d)
        elif prev_rsi >= rsi_low > rsi[i]:
            position = _OpenPosition(direction="SHORT", entry_time=ts, entry_price=c, stop_level=h, date=d)

        prev_rsi = rsi[i]

    if position is not None:
        last = all_5min[-1]
        _close_trade(position, last[0], last[4], "eod_data_end")

    return trades


def summary(trades: list[Trade]) -> str:
    from backtest.rsi2_reversion import summary as _summary
    return _summary(trades)
