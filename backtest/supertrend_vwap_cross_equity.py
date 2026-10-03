"""The same SuperTrend+VWAP signal as backtest.supertrend_vwap_cross_
futures, realized on a single NIFTY 50 CONSTITUENT STOCK's own cash
(NSE_EQ) instrument intraday -- no index, no futures, no options.

WHY THIS IS SIMPLER THAN EVEN THE FUTURES VERSION: the index's own
candles carry volume=0 (it's a computed value, not a traded
instrument), which is why the futures/options modules had to use
NIFTY FUTURES as a volume-bearing proxy for VWAP. An individual stock
IS a traded instrument -- its own 1-minute candles carry real volume
directly, so the signal is computed from, and the trade is realized
on, the exact same instrument: no proxy, no per-day contract/expiry
resolution at all. Historical 1-minute data for a stock also comes
from the REGULAR historical-candle endpoint (no auth, no expiry-
based retention limit) rather than the expired-instruments API the
index derivatives modules need -- a stock's own history doesn't
"expire" the way a derivative contract does.

SIGNAL: identical mechanics to backtest.supertrend_vwap_cross_futures
-- SuperTrend(st_period, st_multiplier) direction + a fresh VWAP
cross on candle_minutes bars of the stock's OWN price, filtered by
ema_filter_period, chop_lookback_days/chop_min_efficiency, and
narrow_cpr_max_width_pct, all computed from the STOCK'S OWN daily
history now, not NIFTY's -- every one of those filters needs its own
from-scratch calibration per instrument (the options module found
Bank Nifty needed completely different filter values than NIFTY;
there is no reason to expect any individual stock's values to match
either). The values below are only carried over as an untested
starting point, exactly flagged the same way in the futures module.

EXIT: trend reversal (SuperTrend flips against the position) or
forced-flat at FORCE_FLAT_TIME; sl_points/target_points are in the
stock's own price points (e.g. Rupees for an NSE_EQ stock), off by
default pending their own tuning -- same reasoning as the futures
module's sl_points (a percentage of one instrument's price is not
interchangeable with another's).

POSITION SIZE: `quantity` shares (default 100, a round number giving
roughly the same notional scale as this codebase's other backtests'
1-lot positions for a mid-priced large-cap stock -- NOT a
recommendation, just a consistent sizing convention). LONG and SHORT
both modeled by default (intraday MIS short-selling is standard for
F&O-enabled stocks, which every NIFTY 50 constituent is);
long_only=True (default False) drops every SHORT signal, taking only
LONG entries -- added for a per-side tuning pass where the two sides
clearly don't perform alike. P&L/costs reuse
rsi2_reversion.Trade's futures-notional cost model as an
approximation for intraday equity costs -- real equity intraday STT/
stamp-duty rates differ slightly from F&O's, but are the same order
of magnitude; treat cost figures here as approximate, same caveat
backtest/costs.py already states for its own numbers.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from data_sources import cache, upstox_client
from backtest.futures_oi_buildup import FORCE_FLAT_TIME
from backtest.sweep_reclaim_breakout import _resample
from backtest.supertrend_pivot_options import _compute_supertrend_line
from backtest.ema8_13_trend_sweep_options import _ema
from backtest.rsi2_reversion import Trade


def run(
    from_date: str,
    to_date: str,
    instrument_key: str,
    quantity: int = 100,
    candle_minutes: int = 5,
    st_period: int = 10,
    st_multiplier: float = 3.0,
    sl_points: float | None = None,
    target_points: float | None = None,
    ema_filter_period: int | None = 45,
    chop_lookback_days: int | None = 15,
    chop_min_efficiency: float | None = 0.07,
    narrow_cpr_max_width_pct: float | None = 0.26,
    long_only: bool = False,
    access_token: str | None = None,
) -> list[Trade]:
    trading_days = upstox_client.get_daily_history(instrument_key, from_date, to_date)
    trading_days.sort(key=lambda d: d["date"])
    if len(trading_days) < 1:
        return []

    chop_skip_days: set[str] = set()
    if chop_lookback_days is not None and chop_min_efficiency is not None:
        warmup_from = (
            datetime.strptime(from_date, "%Y-%m-%d") - timedelta(days=chop_lookback_days * 3)
        ).strftime("%Y-%m-%d")
        chop_days = upstox_client.get_daily_history(instrument_key, warmup_from, to_date)
        chop_days.sort(key=lambda d: d["date"])
        chop_closes = {d["date"]: d["close"] for d in chop_days}
        chop_dates = sorted(chop_closes)
        for day in trading_days:
            d = day["date"]
            idx = chop_dates.index(d)
            if idx < chop_lookback_days:
                continue
            window = [chop_closes[dd] for dd in chop_dates[idx - chop_lookback_days:idx]]
            net_move = abs(window[-1] - window[0])
            abs_moves = sum(abs(window[k] - window[k - 1]) for k in range(1, len(window)))
            efficiency = net_move / abs_moves if abs_moves > 0 else 0.0
            if efficiency < chop_min_efficiency:
                chop_skip_days.add(d)

    narrow_cpr_skip_days: set[str] = set()
    if narrow_cpr_max_width_pct is not None:
        cpr_warmup_from = (datetime.strptime(from_date, "%Y-%m-%d") - timedelta(days=10)).strftime("%Y-%m-%d")
        cpr_days = upstox_client.get_daily_history(instrument_key, cpr_warmup_from, to_date)
        cpr_days.sort(key=lambda d: d["date"])
        cpr_by_date = {d["date"]: d for d in cpr_days}
        cpr_dates = sorted(cpr_by_date)
        for day in trading_days:
            d = day["date"]
            idx = cpr_dates.index(d)
            if idx < 1:
                continue
            prev = cpr_by_date[cpr_dates[idx - 1]]
            h, l, c = prev["high"], prev["low"], prev["close"]
            pivot = (h + l + c) / 3
            bc = (h + l) / 2
            tc = 2 * pivot - bc
            width_pct = abs(tc - bc) / c * 100 if c else 0.0
            if width_pct > narrow_cpr_max_width_pct:
                narrow_cpr_skip_days.add(d)

    all_1min: list[list] = []
    for day in trading_days:
        d = day["date"]
        rows = sorted(
            cache.get_day_candles_cached(instrument_key, "1minute", d, expired=False, access_token=access_token),
            key=lambda c: c[0],
        )
        all_1min.extend(rows)
    all_1min.sort(key=lambda c: c[0])
    if len(all_1min) < 2:
        return []

    bars = _resample(all_1min, candle_minutes)
    if len(bars) < st_period + 2:
        return []

    st_dir, _st_line = _compute_supertrend_line(bars, st_period, st_multiplier)
    ema_filter = _ema([b[4] for b in bars], ema_filter_period) if ema_filter_period else None

    trades: list[Trade] = []
    position: Trade | None = None
    current_day: str | None = None
    cum_pv = cum_vol = 0.0
    prev_close = prev_vwap = None

    for i, row in enumerate(bars):
        ts, o, h, l, c, v, oi = row
        d = ts[:10]
        time_str = ts[11:16]

        if d != current_day:
            current_day = d
            position = None
            cum_pv = cum_vol = 0.0
            prev_close = prev_vwap = None

        typical = (h + l + c) / 3
        cum_pv += typical * v
        cum_vol += v
        vwap = (cum_pv / cum_vol) if cum_vol > 0 else None

        def _close(reason: str) -> None:
            nonlocal position
            position.exit_time = ts
            position.exit_price = c
            position.exit_reason = reason
            trades.append(position)
            position = None

        if time_str >= FORCE_FLAT_TIME:
            if position is not None:
                _close("eod")
            prev_close, prev_vwap = c, vwap
            continue

        if position is not None:
            cur_dir = st_dir[i]
            is_long = position.direction == "LONG"
            hit_reversal = cur_dir is not None and ((is_long and cur_dir == -1) or (not is_long and cur_dir == 1))
            hit_sl = sl_points is not None and (
                (is_long and c <= position.entry_price - sl_points)
                or (not is_long and c >= position.entry_price + sl_points)
            )
            hit_target = target_points is not None and (
                (is_long and c >= position.entry_price + target_points)
                or (not is_long and c <= position.entry_price - target_points)
            )
            if hit_sl:
                _close("stop_loss")
            elif hit_target:
                _close("target")
            elif hit_reversal:
                _close("trend_reverse")

        if (position is None and vwap is not None and prev_vwap is not None and prev_close is not None
                and st_dir[i] is not None and d not in chop_skip_days and d not in narrow_cpr_skip_days):
            crossed_above = prev_close <= prev_vwap and c > vwap
            crossed_below = prev_close >= prev_vwap and c < vwap
            direction_label = None
            if st_dir[i] == 1 and crossed_above:
                direction_label = "LONG"
            elif st_dir[i] == -1 and crossed_below and not long_only:
                direction_label = "SHORT"

            if direction_label is not None and ema_filter is not None:
                ema_val = ema_filter[i]
                if ema_val is None:
                    direction_label = None
                elif direction_label == "LONG" and c <= ema_val:
                    direction_label = None
                elif direction_label == "SHORT" and c >= ema_val:
                    direction_label = None

            if direction_label is not None:
                position = Trade(
                    date=d, direction=direction_label, entry_time=ts, entry_price=c,
                    lot_size=quantity,
                )

        prev_close, prev_vwap = c, vwap

    return trades


def summary(trades: list[Trade]) -> str:
    from backtest.rsi2_reversion import summary as _summary
    return _summary(trades)
