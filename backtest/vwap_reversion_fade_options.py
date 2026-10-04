"""VWAP mean-reversion FADE, realized through real ATM NIFTY options.
5-minute NIFTY FUTURES candles -- NOT the index (volume=0 on every
index candle, verified in supertrend_vwap_cross_options.py; same
futures-resolution machinery reused here: cache.resolve_expired_
futures_contract_for_date for expired dates, fixed_volume_profile_
options._resolve_current_month_futures as the live fallback).

This is the MIRROR IMAGE of supertrend_vwap_cross_options.py: that
module trades WITH a fresh VWAP cross (trend-following, SuperTrend-
confirmed); this one trades AGAINST a stretched move, betting on a
snap back to the mean (no SuperTrend, no trend confirmation needed --
the whole premise is the opposite of "ride the trend").

VWAP = cumulative(typical_price * volume) / cumulative(volume),
typical_price=(High+Low+Close)/3, RESET AT EVERY DAY BOUNDARY (same
definition as supertrend_vwap_cross_options.py).

SIGNAL: band = vwap * band_pct / 100 (a PERCENTAGE distance from VWAP,
not a fixed point distance -- NIFTY's own level moved from ~24,000 to
~26,000+ across this codebase's data window, so a fixed-point band
would mean a different real stretch at different times; band_pct
keeps the trigger comparable across the whole history).
    LONG  (buy ATM CE): close drops BELOW vwap - band (oversold
        stretch) -- betting on a bounce back UP to the mean.
    SHORT (buy ATM PE): close rises ABOVE vwap + band (overbought
        stretch) -- betting on a pullback back DOWN to the mean.
A FRESH trigger only (the first bar of the day to cross outside the
band); once a position is taken, no new entries are evaluated until it
closes (one_trade_per_day also caps it to one entry total per day,
default True).

EXIT: the fade's own take-profit is baked into the signal itself --
price reverting back to (crossing) VWAP -- plus optional sl_pct/
target_pct (premium %, both off by default) and forced-flat at
FORCE_FLAT_TIME. Unlike the breakout-style modules in this codebase,
a fade's risk is open-ended if the stretch keeps extending instead of
reverting (there is no SuperTrend-flip or opposite-boundary exit here
to cut that short automatically -- that's what sl_pct is for).

STRIKE/EXPIRY: ATM = round_to_step(futures close, strike_step),
nearest expiry on/after the entry day -- same convention as every
other NIFTY options module here.

COSTS: backtest.costs' F&O approximation via
macd_rsi2_momentum_options.OptionTrade, same cost model every other
options module in this codebase uses.
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, _bar_at_or_after, _bar_at_or_before
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.sweep_reclaim_breakout import _resample
from backtest.fixed_volume_profile_options import _resolve_current_month_futures

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    candle_minutes: int = 5,
    band_pct: float = 0.3,
    sl_pct: float | None = None,
    target_pct: float | None = None,
    one_trade_per_day: bool = True,
    long_only: bool = False,
    short_only: bool = False,
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
    traded_today = False

    for row in bars:
        ts, o, h, l, c, v, oi = row
        d = ts[:10]
        time_str = ts[11:16]

        if d != current_day:
            current_day = d
            position = None
            cum_pv = cum_vol = 0.0
            traded_today = False

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
            continue

        if vwap is None:
            continue

        if position is not None:
            is_long = position["direction"] == "LONG"
            hit_reversal = (is_long and c >= vwap) or (not is_long and c <= vwap)
            hit_sl = hit_target = False
            if sl_pct is not None or target_pct is not None:
                _, candles = _atm_option_candles(position["strike"], position["opt_type"], position["date"], position["expiry"])
                bar = _fill(candles, decision_time_str) if candles else None
                cur_premium = bar[4] if bar else position["entry_price"]
                if sl_pct is not None:
                    hit_sl = cur_premium <= position["entry_price"] * (1 - sl_pct)
                if target_pct is not None:
                    hit_target = cur_premium >= position["entry_price"] * (1 + target_pct)
            if hit_sl:
                _close("stop_loss")
            elif hit_target:
                _close("target")
            elif hit_reversal:
                _close("reversion")

        if (position is None and not (one_trade_per_day and traded_today)
                and expiry is not None):
            band = vwap * band_pct / 100
            direction_label = None
            if c < vwap - band and not short_only:
                direction_label = "LONG"
            elif c > vwap + band and not long_only:
                direction_label = "SHORT"

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
                        traded_today = True

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
