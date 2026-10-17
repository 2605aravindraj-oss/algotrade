"""The exact same OI-buildup signal and rules as futures_oi_buildup.py,
but realized by trading the futures contract itself (notional), not ATM
options.

This isolates whether the buildup call has real edge on its own, separate
from options-specific noise: strike selection, bid-ask spread on a second
instrument, and theta decay. There's also no cross-instrument fill-timing
risk here (the bug fixed in futures_oi_buildup.py) -- the bar we compute
the signal from IS the instrument being traded, so its own close price at
the bucket's end is exactly the price achievable at the decision instant,
no separate lookup needed.

Same rules as futures_oi_buildup.py:
    flat  + Long Buildup   -> go long
    flat  + Short Buildup  -> go short
    long  + Short Buildup  -> reverse to short
    long  + Long Unwinding -> exit to flat
    short + Long Buildup   -> reverse to long
    short + Short Covering -> exit to flat
Forced flat at FORCE_FLAT_TIME. Uses the futures-style cost model (much
lower than options' since STT only applies sell-side at a lower rate).
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, classify_buildup
from backtest.rsi2_5min_sar import _resample_5min
from backtest.rsi2_reversion import Trade, UNDERLYING_KEY


def run(
    from_date: str,
    to_date: str,
    futures_key: str,
    futures_expired: bool = False,
    access_token: str | None = None,
    lot_size: int = 65,
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
    if len(all_5min) < 2:
        return []

    trades: list[Trade] = []
    position: Trade | None = None
    prev_close = None
    prev_oi = None

    def _close(t: Trade, ts: str, price: float, reason: str) -> None:
        t.exit_time = ts
        t.exit_price = price
        t.exit_reason = reason
        trades.append(t)

    for row in all_5min:
        ts, o, h, l, c, v, oi = row
        d = ts[:10]
        time_str = ts[11:16]

        if prev_close is None:
            prev_close, prev_oi = c, oi
            continue

        buildup = classify_buildup(c - prev_close, oi - prev_oi)

        if position is None:
            if buildup == "Long Buildup":
                position = Trade(date=d, direction="LONG", entry_time=ts, entry_price=c, lot_size=lot_size)
            elif buildup == "Short Buildup":
                position = Trade(date=d, direction="SHORT", entry_time=ts, entry_price=c, lot_size=lot_size)
        else:
            if time_str >= FORCE_FLAT_TIME:
                _close(position, ts, c, "eod")
                position = None
            elif position.direction == "LONG" and buildup == "Short Buildup":
                _close(position, ts, c, "reverse")
                position = Trade(date=d, direction="SHORT", entry_time=ts, entry_price=c, lot_size=lot_size)
            elif position.direction == "LONG" and buildup == "Long Unwinding":
                _close(position, ts, c, "unwind")
                position = None
            elif position.direction == "SHORT" and buildup == "Long Buildup":
                _close(position, ts, c, "reverse")
                position = Trade(date=d, direction="LONG", entry_time=ts, entry_price=c, lot_size=lot_size)
            elif position.direction == "SHORT" and buildup == "Short Covering":
                _close(position, ts, c, "covering")
                position = None

        prev_close, prev_oi = c, oi

    if position is not None:
        last = all_5min[-1]
        _close(position, last[0], last[4], "eod_data_end")

    return trades


def summary(trades: list[Trade]) -> str:
    from backtest.rsi2_reversion import summary as _summary
    return _summary(trades)
