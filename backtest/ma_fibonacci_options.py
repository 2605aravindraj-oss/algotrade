"""200-period Moving Average + Fibonacci retracement pullback-continuation,
realized through real ATM NIFTY options. Systematized from a discretionary
chart-pattern strategy (a trading-book "Moving Average and Fibonacci"
lesson): trade pullbacks that continue in the direction of a 200-bar trend
filter. Intraday only (forced flat at FORCE_FLAT_TIME, no overnight
position) -- the book's own examples hold for days, but this codebase
realizes every strategy through NIFTY WEEKLY options and closes same-day
to avoid overnight theta/gap risk, consistent with every other module here.

The book's rules are chart-drawn and left to a trader's eye (which swing
points to pick, what counts as "at a Fibonacci level," which sub-level
becomes the stop). This module pins each of those down to a concrete,
backtestable rule -- documented below so the gap between "the book" and
"this code" is explicit, not hidden:

TREND FILTER: SMA(200) on candle_minutes bars (default 5-minute,
continuous across the whole date range -- needs substantial warm-up, not
reset daily). Uptrend requires BOTH price above the SMA AND the SMA
itself rising (SMA now > SMA slope_lookback bars ago); downtrend the
mirror. This is the book's own "thing to remember": price above a
FALLING average is flagged as a likely fake breakout, so it's excluded
here rather than just noted as a risk.

SWING POINTS / FIBONACCI LEG: a bar is a (fractal) swing high if its High
is the max over a window of swing_lookback bars on each side of it --
symmetrically for swing lows. Like any lookback-window feature, a swing
point is only knowable once the bars AFTER it have printed, so each one
is revealed exactly swing_lookback bars later than it occurred, never
early. The active Fibonacci leg is the most recent CONFIRMED swing low
followed by a later CONFIRMED swing high (uptrend: retracing that
up-move) or the most recent CONFIRMED swing high followed by a later
CONFIRMED swing low (downtrend: retracing that down-move). The 0%
level is always the leg's trend-direction extreme (the recent high in an
uptrend, the recent low in a downtrend) -- the level price is expected to
travel BACK toward. The 100% level is the pullback's origin.

ENTRY: price must be trading in the "golden zone" between the 38.2% and
61.8% retracement levels (inclusive of the 50% level between them -- the
book calls out all three individually, this treats them as one
continuous zone, the common reading of "the golden zone").
    LONG:  a bullish candle (close > open) whose range touches the zone
           -> buy-stop at that candle's own High; the first later candle
           whose High breaks above it triggers entry (buy ATM CE).
    SHORT: a bearish ENGULFING candle (current bearish, prior bullish,
           current body fully contains the prior candle's body) whose
           range touches the zone -> sell-stop at that candle's own Low;
           the first later candle whose Low breaks below it triggers
           entry (buy ATM PE). The book asks for a plain bullish candle
           on the long side but specifically an engulfing candle on the
           short side -- this module follows that asymmetry as written,
           not a simplification.
A fresh pattern replaces any earlier still-pending one.

STOP-LOSS AND TARGET are on the underlying INDEX (these are chart price
levels in the book, not option-premium points):
    stop   = the nearest Fibonacci sub-level (38.2/50/61.8/100%) beyond
             the entry pattern candle's own stop-side extreme (its Low
             for a LONG, its High for a SHORT) -- "the lower/upper
             Fibonacci level," read as the next one, not the furthest.
    target = the leg's 0% level, UNLESS that would give less than a 1:2
             risk/reward from entry, in which case target = entry +/-
             2*risk instead (the book's explicit "minimum of 1:2").
Checked bar-by-bar against the INDEX bar's own high/low; if a bar's
range would touch both stop and target in the same bar, the stop side
is assumed to trigger first (conservative). Entries and exits fill at
the option's own price as of the deciding candle's CLOSE (bucket start +
candle_minutes), never its start -- same decision-time-correct fill
fix as ema_sweep_breakout_options.py.

Exit: stop-loss, target, or forced flat at FORCE_FLAT_TIME. One position
at a time; a position never carries across a day boundary, and any
pending (not-yet-triggered) pattern lapses at day end too -- the
swing/trend state itself is NOT reset daily (it needs the continuous
multi-day history the book's own charts show).
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, _bar_at_or_after, _bar_at_or_before
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.sweep_reclaim_breakout import _resample

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"
FIB_STOP_LEVELS = (0.382, 0.5, 0.618, 1.0)  # ascending pct, possible stop-loss anchors


def _sma(values: list[float], period: int) -> list[float | None]:
    n = len(values)
    out: list[float | None] = [None] * n
    if n < period:
        return out
    window_sum = sum(values[:period])
    out[period - 1] = window_sum / period
    for i in range(period, n):
        window_sum += values[i] - values[i - period]
        out[i] = window_sum / period
    return out


def _find_confirmed_swings(bars: list[list], lookback: int) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    """Fractal swing highs/lows: bar i is a swing high (low) if its High
    (Low) is the max (min) over [i-lookback, i+lookback]. Returned as
    (confirm_idx, origin_idx, price) tuples -- confirm_idx = origin_idx +
    lookback is the earliest bar index at which this swing is knowable."""
    n = len(bars)
    highs = [b[2] for b in bars]
    lows = [b[3] for b in bars]
    swing_highs = []
    swing_lows = []
    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback:i + lookback + 1]
        if highs[i] == max(window_h):
            swing_highs.append((i + lookback, i, highs[i]))
        window_l = lows[i - lookback:i + lookback + 1]
        if lows[i] == min(window_l):
            swing_lows.append((i + lookback, i, lows[i]))
    return swing_highs, swing_lows


def _fib_level(level0: float, level100: float, pct: float) -> float:
    return level0 + pct * (level100 - level0)


def _next_stop_level(level0: float, level100: float, direction: str, reference_price: float) -> float:
    for pct in FIB_STOP_LEVELS:
        lvl = _fib_level(level0, level100, pct)
        if direction == "LONG" and lvl <= reference_price:
            return lvl
        if direction == "SHORT" and lvl >= reference_price:
            return lvl
    return level100


def _is_bearish_engulfing(prev_row: list, row: list) -> bool:
    po, pc = prev_row[1], prev_row[4]
    o, c = row[1], row[4]
    return pc > po and c < o and o >= pc and c <= po


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    candle_minutes: int = 5,
    sma_period: int = 200,
    slope_lookback: int = 10,
    swing_lookback: int = 5,
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
    if len(bars) < sma_period + slope_lookback + 1:
        return []

    closes = [b[4] for b in bars]
    sma = _sma(closes, sma_period)
    swing_highs, swing_lows = _find_confirmed_swings(bars, swing_lookback)

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
    position = None   # dict: direction, entry_time, entry_price(premium), strike, expiry, lot_size, opt_type, date, stop_level(index), target_level(index)
    pattern = None     # dict: direction, trigger_price(index), stop_level(index), target_level(index)
    current_day: str | None = None
    last_swing_high: tuple[int, float] | None = None  # (origin_idx, price)
    last_swing_low: tuple[int, float] | None = None
    hi_ptr = -1
    lo_ptr = -1

    for i, row in enumerate(bars):
        ts, o, h, l, c, v, oi = row
        d = ts[:10]
        time_str = ts[11:16]

        while hi_ptr + 1 < len(swing_highs) and swing_highs[hi_ptr + 1][0] <= i:
            hi_ptr += 1
            last_swing_high = (swing_highs[hi_ptr][1], swing_highs[hi_ptr][2])
        while lo_ptr + 1 < len(swing_lows) and swing_lows[lo_ptr + 1][0] <= i:
            lo_ptr += 1
            last_swing_low = (swing_lows[lo_ptr][1], swing_lows[lo_ptr][2])

        if d != current_day:
            current_day = d
            pattern = None
            position = None

        atm = oc.round_to_step(c, strike_step)
        expiry = next((e for e in expiries if e >= d), None)

        _dh, _dm = divmod(int(time_str[:2]) * 60 + int(time_str[3:5]) + candle_minutes, 60)
        decision_time_str = f"{_dh:02d}:{_dm:02d}"

        def _enter(direction_label: str, stop_level: float, target_level: float) -> None:
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
                "stop_level": stop_level, "target_level": target_level,
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

        if position is not None:
            if position["direction"] == "LONG":
                if l <= position["stop_level"]:
                    _exit("stop_loss", decision_time_str)
                elif h >= position["target_level"]:
                    _exit("target", decision_time_str)
            else:
                if h >= position["stop_level"]:
                    _exit("stop_loss", decision_time_str)
                elif l <= position["target_level"]:
                    _exit("target", decision_time_str)

        if position is None and pattern is not None:
            if pattern["direction"] == "LONG" and h > pattern["trigger_price"]:
                _enter("LONG", pattern["stop_level"], pattern["target_level"])
                pattern = None
            elif pattern["direction"] == "SHORT" and l < pattern["trigger_price"]:
                _enter("SHORT", pattern["stop_level"], pattern["target_level"])
                pattern = None

        if position is None and sma[i] is not None and sma[i - slope_lookback] is not None:
            uptrend = c > sma[i] and sma[i] > sma[i - slope_lookback]
            downtrend = c < sma[i] and sma[i] < sma[i - slope_lookback]

            up_leg = (
                last_swing_high is not None and last_swing_low is not None
                and last_swing_high[0] > last_swing_low[0] and last_swing_high[1] > last_swing_low[1]
            )
            down_leg = (
                last_swing_high is not None and last_swing_low is not None
                and last_swing_low[0] > last_swing_high[0] and last_swing_low[1] < last_swing_high[1]
            )

            if uptrend and up_leg:
                level0, level100 = last_swing_high[1], last_swing_low[1]
                zone_lo = _fib_level(level0, level100, 0.618)
                zone_hi = _fib_level(level0, level100, 0.382)
                touches_zone = l <= zone_hi and h >= zone_lo
                if touches_zone and c > o:
                    stop_level = _next_stop_level(level0, level100, "LONG", l)
                    risk = h - stop_level
                    target_level = level0 if (level0 - h) >= 2 * risk else h + 2 * risk
                    if risk > 0:
                        pattern = {"direction": "LONG", "trigger_price": h, "stop_level": stop_level, "target_level": target_level}
            elif downtrend and down_leg:
                level0, level100 = last_swing_low[1], last_swing_high[1]
                zone_lo = _fib_level(level0, level100, 0.382)
                zone_hi = _fib_level(level0, level100, 0.618)
                touches_zone = h >= zone_lo and l <= zone_hi
                if touches_zone and i > 0 and _is_bearish_engulfing(bars[i - 1], row):
                    stop_level = _next_stop_level(level0, level100, "SHORT", h)
                    risk = stop_level - l
                    target_level = level0 if (l - level0) >= 2 * risk else l - 2 * risk
                    if risk > 0:
                        pattern = {"direction": "SHORT", "trigger_price": l, "stop_level": stop_level, "target_level": target_level}

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
