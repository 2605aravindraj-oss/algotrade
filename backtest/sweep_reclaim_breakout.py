"""Multi-timeframe "sweep the 15-min low, reclaim it, then break the
reclaim candle's high" long-only breakout, traded on NIFTY futures
notional (not options).

Per day:
1. The first 15-minute candle (09:15-09:30) sets a reference range
   (High15, Low15).
2. Scan subsequent 3-minute candles for the first one whose low dips
   below Low15 but whose close reclaims back above Low15 (a stop-hunt /
   liquidity-sweep pattern). Mark that candle's own high (High3).
3. Scan subsequent 1-minute candles for the first one whose high breaks
   above High3 -> enter LONG at the High3 breakout level.
4. Exit forced flat at 15:25 (no stop-loss/target -- add one if wanted).
   One trade per day: only the first qualifying setup is taken.

Reuses rsi2_reversion.Trade and its futures-style cost model.
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest.rsi2_reversion import Trade, FORCE_FLAT_TIME, UNDERLYING_KEY


def _resample(rows_1min: list[list], bar_minutes: int) -> list[list]:
    buckets: dict[str, list] = {}
    order: list[str] = []
    for row in rows_1min:
        ts, o, h, l, c, v, oi = row
        hh, mm = int(ts[11:13]), int(ts[14:16])
        minutes_since_open = (hh * 60 + mm) - (9 * 60 + 15)
        bucket_idx = max(minutes_since_open, 0) // bar_minutes
        bucket_start_minutes = 9 * 60 + 15 + bucket_idx * bar_minutes
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


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    lot_size: int = 65,
) -> list[Trade]:
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

        bars_15 = _resample(rows_1min, 15)
        if not bars_15:
            continue
        ref = bars_15[0]  # 09:15-09:30
        ref_end_time = "09:30"
        high15, low15 = ref[2], ref[3]

        bars_3 = [b for b in _resample(rows_1min, 3) if b[0][11:16] >= ref_end_time]

        pattern_bar = None
        for b in bars_3:
            ts, o, h, l, c, v, oi = b
            if l < low15 and c > low15:
                pattern_bar = b
                break
        if pattern_bar is None:
            continue

        high3 = pattern_bar[2]
        pattern_end_time = pattern_bar[0][11:16]

        breakout_bar = None
        for row in rows_1min:
            ts, o, h, l, c, v, oi = row
            if ts[11:16] <= pattern_end_time:
                continue
            if h > high3:
                breakout_bar = row
                break
        if breakout_bar is None:
            continue

        entry_time = breakout_bar[0]
        entry_price = high3  # fill at the breakout level (stop-order style)

        # forced flat at FORCE_FLAT_TIME, else last bar of the day
        exit_row = None
        for row in rows_1min:
            ts = row[0]
            if ts[11:16] >= FORCE_FLAT_TIME:
                exit_row = row
                break
        if exit_row is None:
            exit_row = rows_1min[-1]

        trade = Trade(
            date=d, direction="LONG", entry_time=entry_time, entry_price=entry_price,
            lot_size=lot_size,
        )
        trade.exit_time = exit_row[0]
        trade.exit_price = exit_row[4]
        trade.exit_reason = "eod"
        trades.append(trade)

    return trades


def summary(trades: list[Trade]) -> str:
    ok = [t for t in trades if t.pnl_rupees is not None]
    if not ok:
        return "No trades."
    gross = sum(t.pnl_rupees_gross for t in ok)
    net = sum(t.pnl_rupees for t in ok)
    costs_total = sum(t.cost_rupees for t in ok)
    wins = [t for t in ok if t.pnl_rupees > 0]
    losses = [t for t in ok if t.pnl_rupees <= 0]
    lines = [
        f"Total trades:     {len(ok)}",
        f"Gross win rate:   {100*len(wins)/len(ok):.1f}%",
        f"Gross P&L:        Rs {gross:,.2f}",
        f"Total costs:      Rs {costs_total:,.2f}  (Rs {costs_total/len(ok):.2f}/trade)",
        f"Net P&L:          Rs {net:,.2f}",
        f"Net win rate:     {100*len(wins)/len(ok):.1f}%  ({len(wins)}W / {len(losses)}L)",
        f"Avg win:          Rs {sum(t.pnl_rupees for t in wins)/len(wins):,.2f}" if wins else "Avg win:          n/a",
        f"Avg loss:         Rs {sum(t.pnl_rupees for t in losses)/len(losses):,.2f}" if losses else "Avg loss:         n/a",
        f"Best trade:       Rs {max(t.pnl_rupees for t in ok):,.2f}",
        f"Worst trade:      Rs {min(t.pnl_rupees for t in ok):,.2f}",
    ]
    return "\n".join(lines)
