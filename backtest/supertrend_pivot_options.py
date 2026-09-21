"""Supertrend + Standard Pivot Points (R1/S1) trend-following, realized
through real ATM NIFTY options. Systematized from a discretionary
trading-book strategy ("Ride the Trend with Supertrend"): use R1/S1 as
a breakout filter, Supertrend as both entry confirmation and the exit
signal. Intraday only, forced flat at 15:15 -- the book's own worked
example explicitly closes its intraday trade at 3:15 PM (not this
codebase's usual 15:25 convention), so this module follows the book on
that specific point.

SIGNAL (5-minute bars by default, continuous across the whole date
range -- Supertrend needs warm-up, not reset daily):
    Supertrend(st_period, st_multiplier) -- same recursive Wilder-ATR
    formula as backtest/supertrend.py, but this module also needs the
    trailing LINE value (not just bullish/bearish direction) to place a
    stop against it, so it's recomputed locally rather than reusing
    that module's direction-only helper.
    Standard daily Pivot Points from the PREVIOUS trading day's daily
    H/L/C: P=(H+L+C)/3, R1=2P-L, S1=2P-H. Only R1/S1 are used (per the
    book: "Standard Pivot Points with only S1 and R1 enabled").

ENTRY -- book's own steps, applied every bar (not just the day's first
candle):
    LONG:  a candle CLOSES above R1, Supertrend is bullish (this same
           candle's close is above the Supertrend line), and the candle
           itself is bullish (close > open) -> a buy-stop is placed at
           that candle's own CLOSE (not its High -- the book says
           "entry is triggered above the CLOSING of the bullish
           candle"). The first later candle whose High breaks above
           that close triggers entry (buy ATM CE).
    SHORT: mirror -- close below S1, Supertrend bearish, bearish
           candle -> sell-stop at that candle's own Close; first later
           candle whose Low breaks below it triggers entry (buy ATM PE).
A fresh pattern replaces any earlier still-pending one.

STOP-LOSS AND EXIT are both read off the Supertrend line, matching the
book's two exit descriptions taken literally as two separate checks:
    stop_level = the Supertrend line's value AT ENTRY (book: "stoploss
                 may be placed below/above the Supertrend") -- checked
                 bar-by-bar against the INDEX bar's own high/low, so it
                 can trigger intrabar, before any candle closes.
    trend_flip = the book's stated TARGET mechanism ("target is at the
                 trader's discretion, OR exit when price closes below/
                 above the Supertrend") -- i.e. exit the moment
                 Supertrend's own direction flips against the position
                 (a candle CLOSES on the other side of the line). No
                 fixed profit target exists; every winning trade exits
                 via this flip (or gets forced flat first).
Entries and exits fill at the option's own price as of the deciding
candle's CLOSE (bucket start + candle_minutes), never its start -- same
decision-time-correct fill convention as ema_sweep_breakout_options.py
and ma_fibonacci_options.py.

Exit: stop-loss, trend-flip, or forced flat at 15:15. One position at a
time; never carries across a day boundary. Supertrend/pivot state is
NOT reset daily (needs the continuous multi-day history), only the
pending entry pattern is.
"""
from __future__ import annotations

import datetime as _dt

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import _bar_at_or_after, _bar_at_or_before
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.sweep_reclaim_breakout import _resample

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"
FORCE_FLAT_TIME = "15:15"


def _compute_supertrend_line(
    bars: list[list], period: int = 7, multiplier: float = 3.0
) -> tuple[list[int | None], list[float | None]]:
    """Same recursive Wilder-ATR Supertrend as backtest/supertrend.py's
    _compute_supertrend, but also returns the trailing LINE value
    (final_lower while bullish, final_upper while bearish) -- needed to
    place a stop against it, which the shared direction-only helper
    doesn't expose."""
    n = len(bars)
    highs = [b[2] for b in bars]
    lows = [b[3] for b in bars]
    closes = [b[4] for b in bars]

    tr: list[float] = [0.0] * n
    for i in range(n):
        if i == 0:
            tr[i] = highs[i] - lows[i]
        else:
            tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))

    atr: list[float | None] = [None] * n
    if n >= period:
        atr[period - 1] = sum(tr[:period]) / period
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    final_upper: list[float | None] = [None] * n
    final_lower: list[float | None] = [None] * n
    direction: list[int | None] = [None] * n
    line: list[float | None] = [None] * n

    for i in range(period - 1, n):
        hl2 = (highs[i] + lows[i]) / 2
        basic_upper = hl2 + multiplier * atr[i]
        basic_lower = hl2 - multiplier * atr[i]

        if i == period - 1:
            final_upper[i] = basic_upper
            final_lower[i] = basic_lower
            direction[i] = 1 if closes[i] > basic_upper else -1
        else:
            prev_fu, prev_fl = final_upper[i - 1], final_lower[i - 1]
            final_upper[i] = basic_upper if (basic_upper < prev_fu or closes[i - 1] > prev_fu) else prev_fu
            final_lower[i] = basic_lower if (basic_lower > prev_fl or closes[i - 1] < prev_fl) else prev_fl
            if closes[i] > final_upper[i]:
                direction[i] = 1
            elif closes[i] < final_lower[i]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]

        line[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    return direction, line


def _daily_pivots(underlying_key: str, from_date: str, to_date: str, access_token: str | None = None) -> dict[str, dict]:
    """Standard Pivot Points (P, R1, S1) for each trading day, from the
    PREVIOUS trading day's daily H/L/C. Fetches 15 extra calendar days
    before from_date so the first day in the requested window still has
    a prior day to compute from."""
    buffer_start = (_dt.date.fromisoformat(from_date) - _dt.timedelta(days=15)).isoformat()
    daily = upstox_client.get_daily_history(underlying_key, buffer_start, to_date)
    daily.sort(key=lambda row: row["date"])
    pivots: dict[str, dict] = {}
    for i in range(1, len(daily)):
        prev = daily[i - 1]
        p = (prev["high"] + prev["low"] + prev["close"]) / 3
        pivots[daily[i]["date"]] = {"P": p, "R1": 2 * p - prev["low"], "S1": 2 * p - prev["high"]}
    return pivots


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    candle_minutes: int = 5,
    st_period: int = 7,
    st_multiplier: float = 3.0,
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
    if len(bars) < st_period + 2:
        return []

    direction, line = _compute_supertrend_line(bars, st_period, st_multiplier)
    pivots = _daily_pivots(underlying_key, from_date, to_date, access_token)

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

    trades: list[OptionTrade] = []
    position = None  # dict: direction, entry_time, entry_price(premium), strike, expiry, lot_size, opt_type, date, stop_level(index)
    pattern = None    # dict: direction, trigger_price(index close), stop_level(index)
    current_day: str | None = None

    for i, row in enumerate(bars):
        ts, o, h, l, c, v, oi = row
        d = ts[:10]
        time_str = ts[11:16]

        if d != current_day:
            current_day = d
            pattern = None
            position = None

        atm = oc.round_to_step(c, strike_step)
        expiry = next((e for e in expiries if e >= d), None)

        _dh, _dm = divmod(int(time_str[:2]) * 60 + int(time_str[3:5]) + candle_minutes, 60)
        decision_time_str = f"{_dh:02d}:{_dm:02d}"

        def _enter(direction_label: str, stop_level: float) -> None:
            nonlocal position
            if expiry is None:
                return
            opt_type = "CE" if direction_label == "LONG" else "PE"
            contract, candles = _atm_option_candles(atm, opt_type, d, expiry)
            if contract is None or not candles:
                return
            bar = _bar_at_or_after(candles, decision_time_str) or _bar_at_or_before(candles, decision_time_str)
            if bar is None:
                return
            position = {
                "direction": direction_label, "entry_time": bar[0], "entry_price": bar[4],
                "strike": contract["strike_price"], "expiry": expiry,
                "lot_size": contract["lot_size"], "opt_type": opt_type, "date": d,
                "stop_level": stop_level,
            }

        def _exit(reason: str, fill_time_str: str) -> None:
            nonlocal position
            if position is None:
                return
            _, candles = _atm_option_candles(position["strike"], position["opt_type"], position["date"], position["expiry"])
            bar = None
            if candles:
                bar = _bar_at_or_after(candles, fill_time_str) or _bar_at_or_before(candles, fill_time_str)
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
                _exit("eod", time_str)
            continue

        dirn = direction[i]
        st_line = line[i]

        if position is not None:
            if position["direction"] == "LONG":
                if l <= position["stop_level"]:
                    _exit("stop_loss", decision_time_str)
                elif dirn == -1:
                    _exit("trend_flip", decision_time_str)
            else:
                if h >= position["stop_level"]:
                    _exit("stop_loss", decision_time_str)
                elif dirn == 1:
                    _exit("trend_flip", decision_time_str)

        if position is None and pattern is not None:
            if pattern["direction"] == "LONG" and h > pattern["trigger_price"]:
                _enter("LONG", pattern["stop_level"])
                pattern = None
            elif pattern["direction"] == "SHORT" and l < pattern["trigger_price"]:
                _enter("SHORT", pattern["stop_level"])
                pattern = None

        if position is None and dirn is not None and st_line is not None and d in pivots:
            piv = pivots[d]
            if c > piv["R1"] and dirn == 1 and c > o:
                pattern = {"direction": "LONG", "trigger_price": c, "stop_level": st_line}
            elif c < piv["S1"] and dirn == -1 and c < o:
                pattern = {"direction": "SHORT", "trigger_price": c, "stop_level": st_line}

    if position is not None:
        _, candles = _atm_option_candles(position["strike"], position["opt_type"], position["date"], position["expiry"])
        last_bar = candles[-1] if candles else None
        exit_price = last_bar[4] if last_bar else position["entry_price"]
        exit_time = last_bar[0] if last_bar else all_1min[-1][0]
        trades.append(OptionTrade(
            date=position["date"], direction=position["direction"], expiry=position["expiry"],
            strike=position["strike"], entry_time=position["entry_time"], entry_premium=position["entry_price"],
            exit_time=exit_time, exit_premium=exit_price, lot_size=position["lot_size"], exit_reason="eod_data_end",
        ))

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
