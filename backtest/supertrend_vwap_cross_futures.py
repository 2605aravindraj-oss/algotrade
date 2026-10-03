"""The exact same SuperTrend+VWAP signal and regime filters as
backtest.supertrend_vwap_cross_options, but realized by trading NIFTY
FUTURES directly (notional) instead of buying ATM CE/PE options.

WHY THIS IS SIMPLER THAN THE OPTIONS VERSION: no second instrument to
look up. The options module needs a separate option-contract candle
fetch at a computed decision time, because the option's own quotes
don't align exactly with the futures bar that generated the signal.
Here the bar we compute the signal from IS the instrument being
traded, so its own close at the bucket's end is exactly the price
achievable at the decision instant -- same reasoning as
backtest/futures_oi_buildup_notional.py, which this module's P&L
accounting borrows directly (rsi2_reversion.Trade and its
futures-notional cost model: flat Rs 20/order brokerage plus
~0.0255% of notional round-trip -- much lower than options' STT/GST
stack, since there's no premium to buy and no second leg's slippage).

SIGNAL: identical to backtest.supertrend_vwap_cross_options --
SuperTrend(st_period, st_multiplier) direction + a fresh VWAP cross
on candle_minutes futures bars, filtered by ema_filter_period,
chop_lookback_days/chop_min_efficiency, and narrow_cpr_max_width_pct
(see that module's docstring for the full reasoning behind each --
all three are index/futures-price regime reads, portable unchanged
since nothing here depends on options). LONG enters at the signal
bar's own close; SHORT mirrors it.

EXIT: trend reversal (SuperTrend flips against the position) or
forced-flat at FORCE_FLAT_TIME, same as the options version. sl_points
/ target_points are NEW parameters, not ported from the options
module's sl_pct/target_pct -- a percentage of FUTURES PRICE isn't the
same kind of quantity as a percentage of OPTION PREMIUM (futures move
a fraction of a percent intraday; a 15% futures stop would almost
never trigger), so both default to None (unfiltered, ride-until-
reversal only) pending their own from-scratch tuning on this
instrument, exactly like every other parameter in the options module
was tuned from scratch rather than assumed to transfer.

ema_filter_period=45, chop_lookback_days=15/chop_min_efficiency=0.07,
and narrow_cpr_max_width_pct=0.26 are carried over from the options
module's validated NIFTY defaults as a reasonable starting point
(they're pure index/futures regime reads, not options-specific), but
have NOT been independently re-swept against this instrument's own
P&L -- the options module found Bank Nifty needed its own from-scratch
retune for these same filters, so treat these as an untested starting
assumption, not a validated default, until swept here too.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from data_sources import cache, upstox_client
from backtest.futures_oi_buildup import FORCE_FLAT_TIME
from backtest.sweep_reclaim_breakout import _resample
from backtest.supertrend_pivot_options import _compute_supertrend_line
from backtest.fixed_volume_profile_options import _resolve_current_month_futures
from backtest.ema8_13_trend_sweep_options import _ema
from backtest.rsi2_reversion import Trade

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    candle_minutes: int = 5,
    st_period: int = 10,
    st_multiplier: float = 3.0,
    sl_points: float | None = None,
    target_points: float | None = None,
    ema_filter_period: int | None = 45,
    chop_lookback_days: int | None = 15,
    chop_min_efficiency: float | None = 0.07,
    narrow_cpr_max_width_pct: float | None = 0.26,
    access_token: str | None = None,
) -> list[Trade]:
    trading_days = upstox_client.get_daily_history(underlying_key, from_date, to_date)
    trading_days.sort(key=lambda d: d["date"])
    if len(trading_days) < 1:
        return []

    chop_skip_days: set[str] = set()
    if chop_lookback_days is not None and chop_min_efficiency is not None:
        warmup_from = (
            datetime.strptime(from_date, "%Y-%m-%d") - timedelta(days=chop_lookback_days * 3)
        ).strftime("%Y-%m-%d")
        chop_days = upstox_client.get_daily_history(underlying_key, warmup_from, to_date)
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
        cpr_days = upstox_client.get_daily_history(underlying_key, cpr_warmup_from, to_date)
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

    live_futures_cache: dict | None = None

    def _contract_for_day(date_str: str) -> tuple[dict, bool]:
        nonlocal live_futures_cache
        expired_contract = cache.resolve_expired_futures_contract_for_date(underlying_key, date_str, access_token)
        if expired_contract is not None:
            return expired_contract, True
        if live_futures_cache is None:
            live_futures_cache = _resolve_current_month_futures()
        return live_futures_cache, False

    all_1min: list[list] = []
    lot_size_by_day: dict[str, int] = {}
    for day in trading_days:
        d = day["date"]
        contract, expired = _contract_for_day(d)
        lot_size_by_day[d] = contract["lot_size"]
        rows = sorted(
            cache.get_day_candles_cached(contract["instrument_key"], "1minute", d, expired=expired, access_token=access_token),
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
            elif st_dir[i] == -1 and crossed_below:
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
                    lot_size=lot_size_by_day[d],
                )

        prev_close, prev_vwap = c, vwap

    return trades


def summary(trades: list[Trade]) -> str:
    from backtest.rsi2_reversion import summary as _summary
    return _summary(trades)
