"""Opening Range Breakout (ORB), realized through real ATM NIFTY options,
on 5-minute bars by default.

Classic opening-range strategy: mark the index's own high/low over the
first `opening_range_minutes` minutes of the session (default 15, i.e.
09:15-09:30, using the index's raw 1-minute candles for a precise range
regardless of candle_minutes). Once that window closes, watch the
breakout on `candle_minutes` bars (continuous across the rest of the
day, VWAP-free, CPR-free -- just today's own early range):

    LONG  (buy ATM CE): first close ABOVE the opening-range high
    SHORT (buy ATM PE): first close BELOW the opening-range low

One position at a time; by default at most one entry per day
(one_trade_per_day=True) -- whichever side breaks first. Disable to let
both sides fire independently the same day.

EXIT: the opposite boundary (price reverses back through the OTHER side
of the opening range), sl_pct/target_pct (premium %, both optional, off
by default), or forced-flat at FORCE_FLAT_TIME. All exit filters start
off and are tuned from scratch -- see run_nifty()'s docstring for the
sweep history once tuned.

STRIKE/EXPIRY: ATM = round_to_step(index close, strike_step), nearest
expiry on/after the entry day -- same convention as every other NIFTY
options module here.

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

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"
ENTRY_TIME = "09:15"


def _add_minutes(time_str: str, minutes: int) -> str:
    total = int(time_str[:2]) * 60 + int(time_str[3:5]) + minutes
    return f"{total // 60:02d}:{total % 60:02d}"


def _opening_ranges(underlying_key: str, trading_days: list[dict], opening_range_minutes: int) -> dict[str, dict]:
    """Opening-range high/low for each day, from the index's own raw
    1-minute candles (always precise, independent of candle_minutes)."""
    or_end = _add_minutes(ENTRY_TIME, opening_range_minutes)
    ranges: dict[str, dict] = {}
    for day in trading_days:
        d = day["date"]
        rows = sorted(cache.get_day_candles_cached(underlying_key, "1minute", d, expired=False), key=lambda c: c[0])
        window = [r for r in rows if ENTRY_TIME <= r[0][11:16] < or_end]
        if not window:
            ranges[d] = None
            continue
        ranges[d] = {"high": max(r[2] for r in window), "low": min(r[3] for r in window), "or_end": or_end}
    return ranges


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    opening_range_minutes: int = 15,
    candle_minutes: int = 5,
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

    or_by_date = _opening_ranges(underlying_key, trading_days, opening_range_minutes)

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
    traded_today = False

    for row in bars:
        ts, o, h, l, c, v, oi = row
        d = ts[:10]
        time_str = ts[11:16]

        if d != current_day:
            current_day = d
            position = None
            traded_today = False

        orng = or_by_date.get(d)
        if orng is None:
            continue
        expiry = next((e for e in expiries if e >= d), None)
        if expiry is None:
            continue

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

        # bars still inside the opening-range window itself are never signals
        if time_str < orng["or_end"]:
            continue

        atm = oc.round_to_step(c, strike_step)

        if position is not None:
            is_long = position["direction"] == "LONG"
            hit_reversal = (is_long and c < orng["low"]) or (not is_long and c > orng["high"])
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
                _close("reversal")

        if (position is None and not (one_trade_per_day and traded_today)
                and expiry is not None):
            direction_label = None
            if c > orng["high"] and not short_only:
                direction_label = "LONG"
            elif c < orng["low"] and not long_only:
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
