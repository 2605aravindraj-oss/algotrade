"""MACD histogram momentum-exhaustion scalp, realized through real ATM
NIFTY options. 5-minute NIFTY 50 INDEX candles, intraday only.

Several terms in the request are read from a trader's chart-color
intuition ("turns light from dark") rather than given as numbers --
pinned down here to concrete rules, flagged so a misreading is easy to
spot and correct:

STAGE 1 -- MACD(12,26,9) and its histogram (MACD line minus signal
    line) computed on these 5-min closes, continuous across the whole
    date range (needs warm-up, not reset daily -- only the pattern
    state below resets daily).

STAGE 2 -- "bearish macd histogram is strong" / "vice versa": a fresh
    bearish regime starts the moment the histogram crosses from >=0 to
    <0 (mirror: bullish regime on a cross from <=0 to >0). Within that
    regime, track the running most-negative (bearish) / most-positive
    (bullish) histogram value seen so far -- the regime's own "how dark
    has it gotten" extreme. "Strong" is confirmed once that extreme
    reaches at least min_histogram_points (default 6.0) in magnitude --
    chosen from this window's own histogram distribution (2026-05-16 to
    2026-09-08, 5-min NIFTY: median |histogram| ~3.0, 75th percentile
    ~5.7, 90th percentile ~9.3) as a plausibly-real threshold, not
    tuned against P&L -- an assumption to revisit if results look off.

STAGE 3 -- "the moment bearish momentum loses strength turns light
    from dark": once strong is confirmed, the FIRST later candle whose
    histogram is HIGHER than the immediately preceding candle's (less
    negative -- a single deceleration tick) while STILL below zero
    (not yet a full cross back to bullish) is the signal candle --
    read literally as the bar-by-bar color lightening a trader would
    see, not a zero-cross or a multi-bar confirmation. Mirror for the
    bullish/strong case: the first candle whose histogram is LOWER
    than the preceding one (less positive) while still above zero.
    A regime that crosses back through zero before ever getting
    "strong", or before decelerating, is simply abandoned -- no trade
    from it.

ENTRY: "at that candle close, buy/sell options -- scalp some points
    with sl": the signal candle (stage 3) is the entry candle, filled
    at its own close, no separate confirmation bar -- bearish
    exhaustion (histogram decelerating while still negative) buys ATM
    CE (a reversal-up bet), bullish exhaustion buys ATM PE (a
    reversal-down bet).

EXIT -- "scalp some points with sl": no point values were given, so
    this defaults to a small premium-points scalp, same convention as
    this codebase's other premium-SL/target modules (e.g.
    ema_sweep_breakout_options.py): sl_points=8.0, target_points=10.0
    -- an explicit assumption, not derived from the request, chosen
    smaller than this codebase's swing-style SL/targets (10-28) to
    match "scalp." Checked against each later candle's index high/low,
    stop checked first if a single bar would touch both (conservative);
    fill is the option's own premium at that bar's time --
    decision-time-correct (bucket start + candle_minutes), same
    convention as every other intraday module here. Forced flat at
    FORCE_FLAT_TIME if neither hits first.

One position at a time; a fresh signal while already in a trade is
skipped. Everything (regime, extreme tracking, strong/pending state)
resets at every day boundary; a fresh signal candle replaces any
earlier still-pending one within the same regime.
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, _bar_at_or_after, _bar_at_or_before
from backtest.macd_rsi2_momentum import compute_macd
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.sweep_reclaim_breakout import _resample

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    candle_minutes: int = 5,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    min_histogram_points: float = 6.0,
    decel_bars: int = 1,
    sl_points: float = 8.0,
    target_points: float = 10.0,
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
    if len(bars) < macd_slow + macd_signal + 2:
        return []

    closes = [b[4] for b in bars]
    macd_line, signal_line = compute_macd(closes, macd_fast, macd_slow, macd_signal)
    hist = [
        (m - s) if (m is not None and s is not None) else None
        for m, s in zip(macd_line, signal_line)
    ]

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
    position = None       # dict: direction, entry_time, entry_price, strike, expiry, lot_size, opt_type, date, stop_level, target_level
    current_day: str | None = None
    regime = None          # "bearish" / "bullish" / None
    extreme_hist = None
    strong_confirmed = False
    decel_count = 0
    prev_hist = None

    for i, row in enumerate(bars):
        ts, o, h, l, c, v, oi = row
        d = ts[:10]
        time_str = ts[11:16]

        if d != current_day:
            current_day = d
            position = None
            regime = None
            extreme_hist = None
            strong_confirmed = False
            decel_count = 0
            prev_hist = None

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
            prev_hist = hist[i]
            continue

        if position is not None:
            is_long = position["direction"] == "LONG"
            hit_stop = (l <= position["stop_level"]) if is_long else (h >= position["stop_level"])
            hit_target = (h >= position["target_level"]) if is_long else (l <= position["target_level"])
            if hit_stop:
                _close("stop_loss")
            elif hit_target:
                _close("target")

        hi = hist[i]
        if hi is not None:
            # -- regime detection (fresh zero-cross starts/resets tracking) --
            if prev_hist is not None:
                if prev_hist >= 0 and hi < 0:
                    regime, extreme_hist, strong_confirmed, decel_count = "bearish", hi, False, 0
                elif prev_hist <= 0 and hi > 0:
                    regime, extreme_hist, strong_confirmed, decel_count = "bullish", hi, False, 0
                elif regime == "bearish" and hi >= 0:
                    regime = extreme_hist = None
                    strong_confirmed = False
                    decel_count = 0
                elif regime == "bullish" and hi <= 0:
                    regime = extreme_hist = None
                    strong_confirmed = False
                    decel_count = 0

            # -- extreme tracking + strong confirmation + consecutive-deceleration count --
            if regime == "bearish":
                if extreme_hist is None or hi < extreme_hist:
                    extreme_hist = hi
                    decel_count = 0
                elif prev_hist is not None and hi > prev_hist:
                    decel_count += 1
                else:
                    decel_count = 0
                if not strong_confirmed and extreme_hist <= -min_histogram_points:
                    strong_confirmed = True
            elif regime == "bullish":
                if extreme_hist is None or hi > extreme_hist:
                    extreme_hist = hi
                    decel_count = 0
                elif prev_hist is not None and hi < prev_hist:
                    decel_count += 1
                else:
                    decel_count = 0
                if not strong_confirmed and extreme_hist >= min_histogram_points:
                    strong_confirmed = True

            # -- deceleration entry (decel_bars consecutive decelerating bars) --
            if position is None and strong_confirmed and decel_count >= decel_bars and expiry is not None:
                direction_label = None
                if regime == "bearish" and hi < 0:
                    direction_label = "LONG"
                elif regime == "bullish" and hi > 0:
                    direction_label = "SHORT"
                if direction_label is not None:
                    opt_type = "CE" if direction_label == "LONG" else "PE"
                    contract, candles = _atm_option_candles(atm, opt_type, d, expiry)
                    if contract is not None and candles:
                        bar = _fill(candles, decision_time_str)
                        if bar is not None:
                            entry_price = bar[4]
                            stop_level = entry_price - sl_points
                            target_level = entry_price + target_points
                            position = {
                                "direction": direction_label, "entry_time": bar[0], "entry_price": entry_price,
                                "strike": contract["strike_price"], "expiry": expiry,
                                "lot_size": contract["lot_size"], "opt_type": opt_type, "date": d,
                                "stop_level": stop_level, "target_level": target_level,
                            }
                    # consumed either way -- don't re-fire the same deceleration tick
                    strong_confirmed = False

        prev_hist = hi

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
