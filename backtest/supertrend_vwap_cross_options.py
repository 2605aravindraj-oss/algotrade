"""SuperTrend + VWAP crossover trend-following, realized through real
ATM NIFTY options. 5-minute NIFTY FUTURES candles -- NOT the index,
which carries volume=0 on every candle (verified earlier this
session) and so can't support a genuinely volume-WEIGHTED VWAP, same
reasoning as fixed_volume_profile_options.py. The current/historical
front-month futures contract is resolved per day via
cache.resolve_expired_futures_contract_for_date (the expired-futures
resolver built for that module); SuperTrend, VWAP, and the ATM strike
are all derived from the futures price throughout (futures and index
track closely; a simplification, not exact).

SIGNAL: SuperTrend(10, 3) -- this codebase's own standard default
    period/multiplier (backtest/supertrend.py), not stated in the
    request. VWAP = cumulative(typical_price * volume) / cumulative
    (volume), typical_price=(High+Low+Close)/3, RESET AT EVERY DAY
    BOUNDARY (the standard definition -- VWAP is always a same-day
    running average, never carried across days).
    LONG (buy ATM CE): SuperTrend is bullish (direction=+1) AND price
        CROSSES above VWAP on this candle -- a FRESH cross (previous
        candle's close was at/below VWAP, this candle's close is
        above it), not merely "is above", since the request says
        "crossed".
    SHORT (buy ATM PE): SuperTrend bearish (direction=-1) AND price
        crosses below VWAP -- the mirror.
Entry fills at the signal candle's own close.

EXIT: "stop loss [is] price cross below supertrend" (mirror: above,
for the short side) -- read as the SAME event that flips SuperTrend's
own direction against the position (SuperTrend's direction is BY
DEFINITION whether price is above/below its own line, so "price
crosses below the SuperTrend line" and "SuperTrend direction flips
bearish" are one and the same event, not two separate checks). No
profit target was given, so this is also the ONLY way out short of
the forced-flat close -- ride the trend until SuperTrend reverses,
same "trend_flip" exit convention as supertrend_pivot_options.py
(exit_reason="trend_reverse"). Forced flat at FORCE_FLAT_TIME.

sl_pct (default 0.10): an ADDED protective floor beyond what the
original spec called for, since backtesting it bare (ride-until-
reversal only) showed large drawdowns (-Rs 25,570 to -Rs 36,588
across the 4 tested windows). When set, the position also exits
(exit_reason="stop_loss", checked BEFORE the trend-reversal exit if
both would trigger on the same bar) the moment the option's own
premium falls to entry_price*(1-sl_pct) -- same percentage-of-premium
convention as other modules here (e.g. orb_ema_ride_options.py's 40%
stop). This is a deliberate addition on top of the sourced strategy,
not part of its original rules -- sl_pct=None reproduces the
original unmodified behavior exactly.

sl_pct=0.10 was chosen by sweeping 0.05-0.40 on all 4 established
windows: it sits in a genuine plateau with 0.125 (both far above
every other tested value, not an isolated spike) and gives the best
total net P&L (Rs 42,978 vs Rs 28,070 baseline) AND the best
worst-case single-window drawdown (-Rs 30,888 vs -Rs 36,588
baseline) of any value tested. It improves BOTH net P&L and max
drawdown in 3 of the 4 windows; the 4th (2024-10-01/2025-01-31) is
slightly worse on both (-Rs 1,281 P&L, +Rs 4,213 drawdown) -- a minor
cost for the gain elsewhere. The stop causes more same-day
re-entries after a shaken-out position (trade count roughly doubles
in choppy stretches), which is why very different sl_pct values
produce non-monotonic results across windows -- the mechanism is
path-dependent re-entry behavior, not a simple loss cap.

target_pct (default None -- kept off, see below): a second ADDED
exit, same percentage-of-premium convention as sl_pct -- when set,
the position also exits (exit_reason="target") the moment the
option's own premium rises to entry_price*(1+target_pct), checked
ahead of the trend-reversal exit (SL is still checked first of all
three). Also not part of the original sourced spec -- target_pct=
None preserves the sl_pct-only behavior above exactly.

Swept {0.15, 0.25, 0.35, 0.5, 0.75, 1.0} x sl_pct=0.10 across all 4
established windows: EVERY value makes total net P&L worse than no
target at all (Rs 42,978 with no target vs a best of Rs 21,317 at
target_pct=0.35, and as low as -Rs 20,634 at 0.15); the worst case
(target_pct=0.5/0.75 on the 2026-05-16/2026-09-08 window) blows max
drawdown out to -Rs 45,951 to -Rs 50,235, beyond even the original
bare-strategy number. This strategy's edge comes from occasional
large trend-reversal-exit wins (best trades Rs 17,000-19,500); a
fixed percentage target chops exactly those trades short at a
fraction of their eventual move, trimming the tail that carries the
whole P&L. Conclusion: do not enable target_pct here -- it stays
None by design, not merely by default.

ema_filter_period (default None): a third ADDED entry filter, on top
of the SuperTrend+VWAP cross signal -- when set, an EMA(ema_filter_
period) is computed over the full continuous futures close series
(not reset daily, same continuity as SuperTrend, since a trend filter
needs to see across day boundaries). A LONG signal is only taken if
the futures close is ABOVE the EMA; a SHORT signal only if it's
BELOW. The idea: filter out VWAP crosses that go against the
longer-term trend, which this strategy's whipsaw losses in chop
suggest are disproportionately the losing ones. Not part of the
original sourced spec -- ema_filter_period=None takes every signal,
unfiltered, exactly as before.

Decision-time-correct fills (bucket start + candle_minutes), one
position at a time, everything (VWAP accumulator, pending state)
resets at every day boundary. Requires an Upstox access token
(expired-instruments API, for both the futures leg on older dates and
the option premiums).
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, _bar_at_or_after, _bar_at_or_before
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.sweep_reclaim_breakout import _resample
from backtest.supertrend_pivot_options import _compute_supertrend_line
from backtest.fixed_volume_profile_options import _resolve_current_month_futures
from backtest.ema8_13_trend_sweep_options import _ema

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    candle_minutes: int = 5,
    st_period: int = 10,
    st_multiplier: float = 3.0,
    sl_pct: float | None = 0.10,
    target_pct: float | None = None,
    ema_filter_period: int | None = None,
    access_token: str | None = None,
) -> list[OptionTrade]:
    trading_days = upstox_client.get_daily_history(underlying_key, from_date, to_date)
    trading_days.sort(key=lambda d: d["date"])
    if len(trading_days) < 1:
        return []

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
    for day in trading_days:
        d = day["date"]
        contract, expired = _contract_for_day(d)
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

    expiries = sorted(cache.get_expired_expiries_cached(underlying_key, "options", access_token))
    chain_cache: dict[str, dict] = {}

    def _atm_option_candles(strike, opt_type, date, expiry):
        if expiry not in chain_cache:
            chain_cache[expiry] = oc.build_chain_lookup(
                cache.get_expired_option_chain_cached(underlying_key, expiry, access_token)
            )
        lookup = chain_cache[expiry]
        contract = oc.nearest_contract(lookup, strike, opt_type)
        if contract is None:
            return None, None
        candles = cache.get_day_candles_cached(
            contract["instrument_key"], "1minute", date, expired=True, access_token=access_token
        )
        return contract, sorted(candles, key=lambda c: c[0])

    def _fill(candles, at_time):
        return _bar_at_or_after(candles, at_time) or _bar_at_or_before(candles, at_time)

    trades: list[OptionTrade] = []
    position = None  # dict: direction, entry_time, entry_price, strike, expiry, lot_size, opt_type, date
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

        atm = oc.round_to_step(c, strike_step)
        expiry = next((e for e in expiries if e >= d), None)

        _dh, _dm = divmod(int(time_str[:2]) * 60 + int(time_str[3:5]) + candle_minutes, 60)
        decision_time_str = f"{_dh:02d}:{_dm:02d}"

        def _close(reason: str) -> None:
            nonlocal position
            _, candles = _atm_option_candles(position["strike"], position["opt_type"], position["date"], position["expiry"])
            bar = _fill(candles, decision_time_str) if candles else None
            exit_price = bar[4] if bar else position["entry_price"]
            exit_time = bar[0] if bar else ts
            trades.append(OptionTrade(
                date=position["date"], direction=position["direction"], expiry=position["expiry"],
                strike=position["strike"], entry_time=position["entry_time"], entry_premium=position["entry_price"],
                exit_time=exit_time, exit_premium=exit_price, lot_size=position["lot_size"], exit_reason=reason,
            ))
            position = None

        if time_str >= FORCE_FLAT_TIME:
            if position is not None:
                _close("eod")
            prev_close, prev_vwap = c, vwap
            continue

        if position is not None:
            cur_dir = st_dir[i]
            is_long = position["direction"] == "LONG"
            hit_reversal = cur_dir is not None and ((is_long and cur_dir == -1) or (not is_long and cur_dir == 1))
            hit_sl_pct = hit_target_pct = False
            if sl_pct is not None or target_pct is not None:
                _, candles = _atm_option_candles(position["strike"], position["opt_type"], position["date"], position["expiry"])
                bar = _fill(candles, decision_time_str) if candles else None
                cur_premium = bar[4] if bar else position["entry_price"]
                if sl_pct is not None:
                    hit_sl_pct = cur_premium <= position["entry_price"] * (1 - sl_pct)
                if target_pct is not None:
                    hit_target_pct = cur_premium >= position["entry_price"] * (1 + target_pct)
            if hit_sl_pct:
                _close("stop_loss")
            elif hit_target_pct:
                _close("target")
            elif hit_reversal:
                _close("trend_reverse")

        if position is None and vwap is not None and prev_vwap is not None and prev_close is not None and st_dir[i] is not None and expiry is not None:
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
                opt_type = "CE" if direction_label == "LONG" else "PE"
                contract, candles = _atm_option_candles(atm, opt_type, d, expiry)
                if contract is not None and candles:
                    bar = _fill(candles, decision_time_str)
                    if bar is not None:
                        position = {
                            "direction": direction_label, "entry_time": bar[0], "entry_price": bar[4],
                            "strike": contract["strike_price"], "expiry": expiry,
                            "lot_size": contract["lot_size"], "opt_type": opt_type, "date": d,
                        }

        prev_close, prev_vwap = c, vwap

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
