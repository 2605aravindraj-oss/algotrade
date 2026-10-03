"""Swing pivot (lower-low / higher-high) liquidity-sweep reversal,
realized through real ATM NIFTY options. 5-minute NIFTY 50 INDEX
candles, intraday only.

"Mark the lower lows" and "sweep happened at the lower low" are left
to a trader's eye in the request -- pinned down here to concrete rules:

STAGE 1 -- pivot detection: bar i is a confirmed swing LOW once
    pivot_lookback (default 3) later bars exist and bar i's own Low is
    the minimum Low across the window [i-pivot_lookback, i+pivot_lookback]
    -- a standard fractal pivot. Confirmation is necessarily delayed by
    pivot_lookback bars (we can't know a bar was the local min until
    that many bars have passed), but every bar used for the comparison
    is already in the past relative to when the pivot is confirmed --
    not a look-ahead. Mirror rule for swing HIGHs (max Low -> max High).
    Both reset at every day boundary, like every other 5-min module
    here; no cross-day swing structure.

STAGE 2 -- "lower low" / "higher high" labelling: each newly confirmed
    swing low is compared to the LAST confirmed swing low of the same
    day. If it's LOWER, it's marked a "lower low" (LL): the pattern
    watches for a sweep of this level, with its TARGET set to that
    prior swing low's own value (the next reference level up, since
    price has been making lower lows). If it's not lower, no LL is
    marked, but this swing low still becomes the new "last confirmed
    swing low" reference going forward -- plain rolling swing
    structure, not a strict monotonic-only sequence. A fresh LL
    replaces any earlier still-pending one (same pattern-replacement
    convention as this codebase's other sweep modules). Mirror for
    swing highs -> "higher high" (HH), target = the prior swing high's
    value (a reference level below, since price has been making higher
    highs).

STAGE 3 -- "swi[sic] happened at the lower low" (read as SWEEP,
    consistent with this codebase's other sweep-reclaim modules): once
    an LL is marked and pending, the first later candle (same day)
    whose Low dips below the LL level but whose Close reclaims back
    above it triggers a LONG entry (buy ATM CE) at that candle's own
    close. Mirror: the first later candle whose High pokes above a
    pending HH's level but Close reclaims back below it triggers a
    SHORT entry (buy ATM PE).

STOP-LOSS is a structural INDEX level -- no stop was specified for
this pattern, so (matching this codebase's other structural-stop
sweep modules, e.g. ema8_13_trend_sweep_options.py) the sweep candle's
own opposite extreme is used: LONG stop = sweep candle's own Low,
SHORT stop = sweep candle's own High. TARGET is exactly the prior
swing pivot's own value from stage 2 above (an explicit level from the
request, not a computed multiple) -- checked against each later
candle's index high/low, stop checked first if a single bar would
touch both (conservative). The actual fill is still the option's own
premium at that bar's time -- decision-time-correct (bucket start +
candle_minutes), same convention as every other intraday module here.

Exit: stop-loss, target, or forced flat at FORCE_FLAT_TIME. One
position at a time; a fresh sweep signal while already in a trade is
skipped. Never carries across a day boundary.
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, _bar_at_or_after, _bar_at_or_before
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.sweep_reclaim_breakout import _resample

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    candle_minutes: int = 5,
    pivot_lookback: int = 3,
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
    if len(bars) < 1:
        return []

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
    position = None  # dict: direction, entry_time, entry_price, strike, expiry, lot_size, opt_type, date, stop_level, target_level
    current_day: str | None = None
    day_bars: list[list] = []       # this day's bars only, for pivot windowing
    prev_pivot_low = prev_pivot_high = None
    pending_ll = pending_hh = None  # dict: level, target

    for i, row in enumerate(bars):
        ts, o, h, l, c, v, oi = row
        d = ts[:10]
        time_str = ts[11:16]

        if d != current_day:
            current_day = d
            position = None
            day_bars = []
            prev_pivot_low = prev_pivot_high = None
            pending_ll = pending_hh = None

        day_bars.append(row)
        k = len(day_bars) - 1

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

        if position is not None:
            is_long = position["direction"] == "LONG"
            hit_stop = (l <= position["stop_level"]) if is_long else (h >= position["stop_level"])
            hit_target = (h >= position["target_level"]) if is_long else (l <= position["target_level"])
            if hit_stop:
                _close("stop_loss")
            elif hit_target:
                _close("target")

        # -- pivot confirmation (delayed by pivot_lookback bars) --
        if k >= 2 * pivot_lookback:
            piv_idx = k - pivot_lookback
            window = day_bars[piv_idx - pivot_lookback: piv_idx + pivot_lookback + 1]
            piv_low = day_bars[piv_idx][3]
            piv_high = day_bars[piv_idx][2]
            if piv_low == min(b[3] for b in window):
                if prev_pivot_low is not None and piv_low < prev_pivot_low:
                    pending_ll = {"level": piv_low, "target": prev_pivot_low}
                prev_pivot_low = piv_low
            if piv_high == max(b[2] for b in window):
                if prev_pivot_high is not None and piv_high > prev_pivot_high:
                    pending_hh = {"level": piv_high, "target": prev_pivot_high}
                prev_pivot_high = piv_high

        # -- sweep detection + entry --
        if position is None and pending_ll is not None and expiry is not None:
            if l < pending_ll["level"] and c >= pending_ll["level"]:
                opt_type = "CE"
                contract, candles = _atm_option_candles(atm, opt_type, d, expiry)
                if contract is not None and candles:
                    bar = _fill(candles, decision_time_str)
                    if bar is not None:
                        position = {
                            "direction": "LONG", "entry_time": bar[0], "entry_price": bar[4],
                            "strike": contract["strike_price"], "expiry": expiry,
                            "lot_size": contract["lot_size"], "opt_type": opt_type, "date": d,
                            "stop_level": l, "target_level": pending_ll["target"],
                        }
                pending_ll = None

        if position is None and pending_hh is not None and expiry is not None:
            if h > pending_hh["level"] and c <= pending_hh["level"]:
                opt_type = "PE"
                contract, candles = _atm_option_candles(atm, opt_type, d, expiry)
                if contract is not None and candles:
                    bar = _fill(candles, decision_time_str)
                    if bar is not None:
                        position = {
                            "direction": "SHORT", "entry_time": bar[0], "entry_price": bar[4],
                            "strike": contract["strike_price"], "expiry": expiry,
                            "lot_size": contract["lot_size"], "opt_type": opt_type, "date": d,
                            "stop_level": h, "target_level": pending_hh["target"],
                        }
                pending_hh = None

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
