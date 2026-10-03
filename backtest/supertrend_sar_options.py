"""Pure SuperTrend stop-and-reverse (SAR), realized through real ATM
NIFTY options, on 5-minute bars.

WHY THIS DIFFERS FROM backtest/supertrend_vwap_cross_options.py: that
module only enters on a SuperTrend+VWAP-cross AGREEMENT (and can sit
flat with no position), and needs NIFTY FUTURES as a volume-bearing
proxy to compute VWAP (the index's own candles carry volume=0). This
strategy has no VWAP filter at all -- SuperTrend is computed from ATR
alone (high/low/close only, no volume needed), so it's computed
directly from the INDEX'S OWN 1-minute candles (verified to carry real
OHLC; no futures proxy needed). Being a pure SAR, it is ALWAYS in the
market once SuperTrend settles: every direction flip immediately
closes the current leg and opens the opposite one, exactly like
backtest/supertrend.py (5-minute NIFTY futures, stop-and-reverse) and
backtest/supertrend_daily_sar.py (daily bars, any NSE_EQ stock) --
this is the same rule, realized through options instead of the
underlying/futures.

SIGNAL: SuperTrend(period, multiplier) on 5-minute index bars
(resampled from 1-minute, continuous across the whole date range).
Direction flips bullish -> exit any PE position, buy ATM CE at this
candle's close. Direction flips bearish -> exit any CE position, buy
ATM PE at this candle's close (long_only=True drops this leg and goes
flat instead, until the next bullish flip).

EXIT: the opposite SuperTrend flip, or forced-flat at FORCE_FLAT_TIME
(no overnight options position -- the weekly-expiry NIFTY chain this
codebase has only covers 2024-10-03 onward, and intraday-only avoids
any expiry-rollover question entirely).

STRIKE/EXPIRY: ATM = round_to_step(index close, strike_step), the
nearest expiry on or after the entry day (same convention as every
other NIFTY options module here).

COSTS: backtest.costs' F&O approximation (flat brokerage + STT on the
sell side + exchange/SEBI/stamp/GST), via macd_rsi2_momentum_options.
OptionTrade -- the same cost model every other options module in this
codebase uses.
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, _bar_at_or_after, _bar_at_or_before
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.sweep_reclaim_breakout import _resample
from backtest.supertrend_pivot_options import _compute_supertrend_line

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    candle_minutes: int = 5,
    st_period: int = 7,
    st_multiplier: float = 3.0,
    long_only: bool = False,
    access_token: str | None = None,
) -> list[OptionTrade]:
    trading_days = upstox_client.get_daily_history(underlying_key, from_date, to_date)
    trading_days.sort(key=lambda d: d["date"])
    if len(trading_days) < 1:
        return []

    all_1min: list[list] = []
    for day in trading_days:
        d = day["date"]
        rows = sorted(
            cache.get_day_candles_cached(underlying_key, "1minute", d, expired=False),
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
    prev_dir: int | None = None

    for i, row in enumerate(bars):
        ts, o, h, l, c, v, oi = row
        d = ts[:10]
        time_str = ts[11:16]

        if d != current_day:
            current_day = d
            position = None
            prev_dir = None

        dirn = st_dir[i]
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
            prev_dir = dirn
            continue

        if dirn is None:
            prev_dir = dirn
            continue

        if prev_dir is not None and dirn != prev_dir and expiry is not None:
            if position is not None:
                _close("flip_flat" if (long_only and dirn == -1) else "reverse")

            if dirn == 1:
                opt_type, direction_label = "CE", "LONG"
            elif not long_only:
                opt_type, direction_label = "PE", "SHORT"
            else:
                opt_type = direction_label = None

            if opt_type is not None:
                contract, candles = _atm_option_candles(atm, opt_type, d, expiry)
                if contract is not None and candles:
                    bar = _fill(candles, decision_time_str)
                    if bar is not None:
                        position = {
                            "direction": direction_label, "entry_time": bar[0], "entry_price": bar[4],
                            "strike": contract["strike_price"], "expiry": expiry,
                            "lot_size": contract["lot_size"], "opt_type": opt_type, "date": d,
                        }

        prev_dir = dirn

    if position is not None:
        last = bars[-1]
        trades.append(OptionTrade(
            date=position["date"], direction=position["direction"], expiry=position["expiry"],
            strike=position["strike"], entry_time=position["entry_time"], entry_premium=position["entry_price"],
            exit_time=last[0], exit_premium=position["entry_price"], lot_size=position["lot_size"],
            exit_reason="data_end",
        ))

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.supertrend_vwap_cross_options import summary as _summary
    return _summary(trades)
