"""EMA8/EMA13 crossover -> trend -> consolidation -> EMA sweep
continuation, realized through real ATM NIFTY options. 5-minute NIFTY
50 INDEX candles.

This is a multi-stage pattern with several terms ("trend happened",
"consolidation") left to a trader's eye in the request -- pinned down
here to concrete, testable rules, documented so any misreading is
easy to spot and correct:

STAGE 1 -- crossover: EMA8 crosses EMA13 (continuous across the whole
    date range, needs warm-up) -- bullish (EMA8 from at/below to above
    EMA13) starts an "up" regime, bearish the mirror "down" regime.
    Regime state resets at every day boundary (intraday only, like
    every other 5-min module here) -- the whole crossover-through-entry
    sequence must complete within one session.

STAGE 2 -- "trend happened": since the regime started, track the
    running extreme (highest High for an up regime, lowest Low for a
    down regime). Confirmed once that extreme has moved at least
    trend_min_points (default 15 index points) beyond the crossover
    bar's own Close -- a floor meant to rule out noise, not a claim
    about what a "real" trend requires.

STAGE 3 -- "consolidation": once trend is confirmed, consolidation is
    read as the trend PAUSING -- consolidation_bars (default 3)
    consecutive candles pass with no new extreme being set.

STAGE 4 -- "sweep happened at EMA": once consolidating, watch every
    later candle (still in the same regime) for a sweep of EMA8 (the
    faster/nearer line) IN THE TREND'S OWN DIRECTION -- a pullback
    that pokes through EMA8 but closes back on the trend's side:
        UP regime:   candle Low < EMA8 and Close >= EMA8
        DOWN regime: candle High > EMA8 and Close <= EMA8
    That candle's own High/Low becomes the breakout trigger level (a
    fresh sweep replaces any earlier still-pending one), same
    sweep-then-breakout structure as ema_sweep_breakout_options.py.

ENTRY: the first later candle whose High breaks above the sweep
candle's High (up regime, buy ATM CE) or Low breaks below the sweep
candle's Low (down regime, buy ATM PE) -- i.e. resuming the original
trend after the pullback, not a new reversal.

STOP-LOSS AND TARGET are structural INDEX levels (a genuine chart
stop, not option-premium points), since none were specified for this
pattern and the sweep candle's own range is the natural risk unit,
same convention as this codebase's earlier sweep-breakout modules:
    UP:   stop = sweep candle's Low,  risk = pattern High - Low,
          target = entry + 2*risk (index points)
    DOWN: stop = sweep candle's High, target = entry - 2*risk -- the
          mirror
Checked against each later candle's index high/low; stop checked
first if a single bar would touch both (conservative). The actual
fill is still the option's own premium at that bar's time --
decision-time-correct (bucket start + candle_minutes), same
convention as every other intraday module here.

Exit: stop-loss, target, or forced flat at FORCE_FLAT_TIME. One
position at a time; never carries across a day boundary.
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, _bar_at_or_after, _bar_at_or_before
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.sweep_reclaim_breakout import _resample

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"


def _ema(values: list[float], period: int) -> list[float | None]:
    n = len(values)
    out: list[float | None] = [None] * n
    if n < period:
        return out
    k = 2 / (period + 1)
    out[period - 1] = sum(values[:period]) / period
    for i in range(period, n):
        out[i] = values[i] * k + out[i - 1] * (1 - k)
    return out


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    candle_minutes: int = 5,
    ema_fast: int = 8,
    ema_slow: int = 13,
    trend_min_points: float = 15.0,
    consolidation_bars: int = 3,
    target_multiple: float = 2.0,
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
    if len(bars) < ema_slow + 2:
        return []

    closes = [b[4] for b in bars]
    ema8 = _ema(closes, ema_fast)
    ema13 = _ema(closes, ema_slow)

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
    position = None    # dict: direction, entry_time, entry_price, strike, expiry, lot_size, opt_type, date, stop_level, target_level
    pattern = None      # dict: direction, high, low
    current_day: str | None = None
    regime = None       # "up" / "down" / None
    regime_start_close = None
    extreme_val = None
    extreme_idx = None
    trend_confirmed = False
    consolidating = False
    prev_ema8 = prev_ema13 = None

    for i, row in enumerate(bars):
        ts, o, h, l, c, v, oi = row
        d = ts[:10]
        time_str = ts[11:16]

        if d != current_day:
            current_day = d
            position = None
            pattern = None
            regime = None
            regime_start_close = extreme_val = extreme_idx = None
            trend_confirmed = consolidating = False

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
            prev_ema8, prev_ema13 = ema8[i], ema13[i]
            continue

        if position is not None:
            is_long = position["direction"] == "LONG"
            hit_stop = (l <= position["stop_level"]) if is_long else (h >= position["stop_level"])
            hit_target = (h >= position["target_level"]) if is_long else (l <= position["target_level"])
            if hit_stop:
                _close("stop_loss")
            elif hit_target:
                _close("target")

        # -- crossover detection --
        if prev_ema8 is not None and prev_ema13 is not None and ema8[i] is not None and ema13[i] is not None:
            if prev_ema8 <= prev_ema13 and ema8[i] > ema13[i]:
                regime = "up"
                regime_start_close = c
                extreme_val, extreme_idx = h, i
                trend_confirmed = consolidating = False
                pattern = None
            elif prev_ema8 >= prev_ema13 and ema8[i] < ema13[i]:
                regime = "down"
                regime_start_close = c
                extreme_val, extreme_idx = l, i
                trend_confirmed = consolidating = False
                pattern = None
            elif regime == "up" and ema8[i] < ema13[i]:
                regime = None  # regime broke without a fresh opposite cross this bar
                pattern = None
            elif regime == "down" and ema8[i] > ema13[i]:
                regime = None
                pattern = None

        # -- trend + consolidation tracking --
        if regime == "up":
            if h > extreme_val:
                extreme_val, extreme_idx = h, i
            if not trend_confirmed and extreme_val - regime_start_close >= trend_min_points:
                trend_confirmed = True
            if trend_confirmed and (i - extreme_idx) >= consolidation_bars:
                consolidating = True
        elif regime == "down":
            if l < extreme_val:
                extreme_val, extreme_idx = l, i
            if not trend_confirmed and regime_start_close - extreme_val >= trend_min_points:
                trend_confirmed = True
            if trend_confirmed and (i - extreme_idx) >= consolidation_bars:
                consolidating = True

        # -- sweep detection (only once consolidating) --
        if position is None and consolidating and ema8[i] is not None:
            if regime == "up" and l < ema8[i] and c >= ema8[i]:
                pattern = {"direction": "LONG", "high": h, "low": l}
            elif regime == "down" and h > ema8[i] and c <= ema8[i]:
                pattern = {"direction": "SHORT", "high": h, "low": l}

        # -- breakout / entry --
        if position is None and pattern is not None and expiry is not None:
            triggered = False
            if pattern["direction"] == "LONG" and h > pattern["high"]:
                triggered = True
            elif pattern["direction"] == "SHORT" and l < pattern["low"]:
                triggered = True
            if triggered:
                direction_label = pattern["direction"]
                risk = pattern["high"] - pattern["low"]
                if direction_label == "LONG":
                    stop_level = pattern["low"]
                    target_level = pattern["high"] + target_multiple * risk
                else:
                    stop_level = pattern["high"]
                    target_level = pattern["low"] - target_multiple * risk
                opt_type = "CE" if direction_label == "LONG" else "PE"
                contract, candles = _atm_option_candles(atm, opt_type, d, expiry)
                if contract is not None and candles:
                    bar = _fill(candles, decision_time_str)
                    if bar is not None:
                        position = {
                            "direction": direction_label, "entry_time": bar[0], "entry_price": bar[4],
                            "strike": contract["strike_price"], "expiry": expiry,
                            "lot_size": contract["lot_size"], "opt_type": opt_type, "date": d,
                            "stop_level": stop_level, "target_level": target_level,
                        }
                pattern = None

        prev_ema8, prev_ema13 = ema8[i], ema13[i]

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
