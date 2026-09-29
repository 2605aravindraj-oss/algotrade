"""Max pain / gamma pinning reversion, realized through real ATM NIFTY
options. Daily reference (from the prior session's own open interest),
traded the next day on 5-minute candles.

MAX PAIN: for each candidate strike S (searched across a band of
max_pain_strike_range strikes -- default 15, i.e. +/-750 points at
NIFTY's 50-point step -- around the reference day's own ATM), the
total payout option WRITERS would owe if NIFTY settled exactly at S:
    payout(S) = sum over CE strikes K of OI_K * max(0, S - K)
              + sum over PE strikes K of OI_K * max(0, K - S)
Max pain strike = argmin_S payout(S) -- the level at which option
SELLERS collectively lose the least (equivalently, buyers gain the
least). This is the standard textbook definition -- NOT simply "the
single strike with the most OI" (the two often roughly coincide, but
max pain properly weighs every strike's OI against its distance from
S, not just the one peak). OI is read from the PRIOR trading day's own
option candles -- each contract's LAST available OI value that day --
a fixed, once-per-day snapshot, same "prior session's static
reference" convention as fixed_volume_profile_options.py's volume
profile.

PINNING is a well-documented EXPIRY-DAY effect: as expiry nears, market
makers hedging large short-gamma positions must trade in the direction
that pulls price back toward the strike where their combined exposure
is smallest (loosely, the max pain strike) -- an effect that's weak
early in an expiry's life and strengthens close to it.
near_expiry_only (default True) restricts trading to days within
near_expiry_days (default 2) of the reference day's OWN expiry; set
False to test the reversion idea on every day regardless of days-to-
expiry.

ENTRY, next trading day, same structural shape as
fixed_volume_profile_options.py's poc_magnet mode: whenever a candle's
close is more than pin_distance_points (default 40.0, an assumption)
away from the max pain strike, bet on reversion back toward it: close
> max_pain + distance -> SHORT (buy ATM PE); close < max_pain -
distance -> LONG (buy ATM CE). Entry fills at that candle's own close.
One entry per day (first qualifying candle wins).

EXIT: stop = entry +/- sl_points (default 30.0) further away from max
pain (structural drift-continuation risk, same convention as
poc_magnet); target = the max pain strike itself. Checked against each
later candle's own CLOSE (not high/low -- this is a level-reversion
target/stop, not a wick-touch one). Forced flat at FORCE_FLAT_TIME if
neither hits first. Fill is the option's own premium at that bar's
time -- decision-time-correct (bucket start + candle_minutes), same
convention as every other intraday module here.
"""
from __future__ import annotations

import datetime as _dt

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, _bar_at_or_after, _bar_at_or_before
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.sweep_reclaim_breakout import _resample

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"


def compute_max_pain(
    underlying_key: str,
    expiry: str,
    reference_date: str,
    atm_guess: float,
    strike_step: int,
    strike_range: int,
    access_token: str | None,
) -> float | None:
    """Max pain strike for `expiry`, using OI as of `reference_date`'s
    last available bar for each strike in [-strike_range, strike_range]
    around atm_guess. Returns None if no strikes had usable OI data."""
    lookup = oc.build_chain_lookup(
        cache.get_expired_option_chain_cached(underlying_key, expiry, access_token)
    )
    oi_by_strike_type: dict[tuple[float, str], float] = {}
    for k in range(-strike_range, strike_range + 1):
        strike = atm_guess + k * strike_step
        for opt_type in ("CE", "PE"):
            contract = lookup.get((strike, opt_type))
            if contract is None:
                continue
            candles = cache.get_day_candles_cached(
                contract["instrument_key"], "1minute", reference_date, expired=True, access_token=access_token
            )
            if not candles:
                continue
            last_bar = max(candles, key=lambda c: c[0])
            oi_by_strike_type[(strike, opt_type)] = last_bar[6]

    strikes = sorted({s for s, _ in oi_by_strike_type})
    if not strikes:
        return None

    def payout(settle: float) -> float:
        total = 0.0
        for (strike, opt_type), oi in oi_by_strike_type.items():
            if opt_type == "CE":
                total += oi * max(0.0, settle - strike)
            else:
                total += oi * max(0.0, strike - settle)
        return total

    return min(strikes, key=payout)


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    candle_minutes: int = 5,
    max_pain_strike_range: int = 15,
    near_expiry_only: bool = True,
    near_expiry_days: int = 2,
    pin_distance_points: float = 40.0,
    sl_points: float = 30.0,
    access_token: str | None = None,
) -> list[OptionTrade]:
    trading_days = upstox_client.get_daily_history(underlying_key, from_date, to_date)
    all_1min: list[list] = []
    for day in trading_days:
        rows = sorted(
            cache.get_day_candles_cached(underlying_key, "1minute", day["date"], expired=False),
            key=lambda c: c[0],
        )
        all_1min.extend(rows)
    all_1min.sort(key=lambda c: c[0])
    if len(all_1min) < 2:
        return []

    bars = _resample(all_1min, candle_minutes)
    if not bars:
        return []

    expiries = sorted(cache.get_expired_expiries_cached(underlying_key, "options", access_token))
    chain_cache: dict[str, dict] = {}

    def _lookup(expiry):
        if expiry not in chain_cache:
            chain_cache[expiry] = oc.build_chain_lookup(
                cache.get_expired_option_chain_cached(underlying_key, expiry, access_token)
            )
        return chain_cache[expiry]

    def _index_close(date):
        rows = sorted(
            cache.get_day_candles_cached(underlying_key, "1minute", date, expired=False),
            key=lambda c: c[0],
        )
        return rows[-1][4] if rows else None

    def _atm_option_candles(strike, opt_type, date, expiry):
        lookup = _lookup(expiry)
        contract = oc.nearest_contract(lookup, strike, opt_type)
        if contract is None:
            return None, None
        candles = cache.get_day_candles_cached(
            contract["instrument_key"], "1minute", date, expired=True, access_token=access_token
        )
        return contract, sorted(candles, key=lambda c: c[0])

    # -- reference: max pain per prior trading day --
    max_pain_by_day: dict[str, dict | None] = {}
    for day in trading_days:
        d = day["date"]
        expiry = next((e for e in expiries if e >= d), None)
        idx_close = _index_close(d)
        if expiry is None or idx_close is None:
            max_pain_by_day[d] = None
            continue
        atm_guess = oc.round_to_step(idx_close, strike_step)
        max_pain = compute_max_pain(underlying_key, expiry, d, atm_guess, strike_step, max_pain_strike_range, access_token)
        max_pain_by_day[d] = {"max_pain": max_pain, "expiry": expiry} if max_pain is not None else None

    trades: list[OptionTrade] = []
    trading_dates = [day["date"] for day in trading_days]
    for i in range(1, len(trading_dates)):
        d = trading_dates[i]
        ref = max_pain_by_day[trading_dates[i - 1]]
        if ref is None or ref["expiry"] < d:
            continue
        if near_expiry_only:
            dte = (_dt.date.fromisoformat(ref["expiry"]) - _dt.date.fromisoformat(d)).days
            if dte > near_expiry_days:
                continue

        day_bars = [b for b in bars if b[0][:10] == d]
        if not day_bars:
            continue

        max_pain = ref["max_pain"]
        expiry = ref["expiry"]
        position = None  # dict: direction, entry_time, entry_price, strike, expiry, lot_size, opt_type, date, stop_level, target_level
        already_traded_today = False

        for row in day_bars:
            ts, o, h, l, c, v, oi = row
            time_str = ts[11:16]
            atm = oc.round_to_step(c, strike_step)
            _dh, _dm = divmod(int(time_str[:2]) * 60 + int(time_str[3:5]) + candle_minutes, 60)
            decision_time_str = f"{_dh:02d}:{_dm:02d}"

            def _close(reason: str) -> None:
                nonlocal position
                _, candles = _atm_option_candles(position["strike"], position["opt_type"], position["date"], position["expiry"])
                bar = (_bar_at_or_after(candles, decision_time_str) or _bar_at_or_before(candles, decision_time_str)) if candles else None
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

            if position is not None:
                is_long = position["direction"] == "LONG"
                hit_stop = (c <= position["stop_level"]) if is_long else (c >= position["stop_level"])
                hit_target = (c >= position["target_level"]) if is_long else (c <= position["target_level"])
                if hit_stop:
                    _close("stop_loss")
                elif hit_target:
                    _close("target")

            if position is None and not already_traded_today:
                direction_label = None
                stop_price = None
                if c > max_pain + pin_distance_points:
                    direction_label, stop_price = "SHORT", c + sl_points
                elif c < max_pain - pin_distance_points:
                    direction_label, stop_price = "LONG", c - sl_points

                if direction_label is not None:
                    opt_type = "CE" if direction_label == "LONG" else "PE"
                    contract, candles = _atm_option_candles(atm, opt_type, d, expiry)
                    if contract is not None and candles:
                        bar = _bar_at_or_after(candles, decision_time_str) or _bar_at_or_before(candles, decision_time_str)
                        if bar is not None:
                            position = {
                                "direction": direction_label, "entry_time": bar[0], "entry_price": bar[4],
                                "strike": contract["strike_price"], "expiry": expiry,
                                "lot_size": contract["lot_size"], "opt_type": opt_type, "date": d,
                                "stop_level": stop_price, "target_level": max_pain,
                            }
                            already_traded_today = True

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
