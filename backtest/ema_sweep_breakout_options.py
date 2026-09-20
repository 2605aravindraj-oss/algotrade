"""EMA9/EMA20 sweep-and-breakout, realized through real ATM NIFTY
options. Intraday only (forced flat at FORCE_FLAT_TIME, no overnight
position).

Multi-timeframe signal: EMA9 and EMA20 are computed on 15-MINUTE bars
(continuous across the whole date range -- needs warm-up; not reset
daily), but the sweep/breakout pattern itself is read off 1-MINUTE
candles against those 15-min EMA lines. This is deliberately slower
and smoother than computing the EMA on 1-minute closes directly (which
produced far too many low-quality signals in an earlier version of
this module) -- a 15-min EMA only moves meaningfully every 15 minutes,
so far fewer 1-min candles will actually cross it.

Decision-time correctness: a 1-minute candle can only react to a
15-min EMA value from a bucket that has ALREADY CLOSED, never the
bucket it's currently inside (that bucket's close, and therefore its
EMA contribution, isn't known yet). Concretely, the 15-min bucket
labeled "09:15" (covering 09:15-09:29) only becomes usable starting at
its close, 09:30 -- so every 1-minute candle from 09:30 up to (but not
including) 09:45 uses THAT bucket's EMA9/EMA20, not the "09:30" bucket
forming under it. This is the same class of fill-pricing/lookahead fix
made in futures_oi_buildup.py, generalized to a signal rather than a
fill price.

A candle "sweeps" an EMA when its range crosses through the (15-min,
decision-time-correct) line but it closes back on the side it started
from:
    sweep ABOVE: candle High > EMA and Close <= EMA
                 (price poked above the average, closed back under it)
    sweep BELOW: candle Low  < EMA and Close >= EMA
                 (price poked below the average, closed back over it)
Either kind of sweep (against EMA9 or EMA20 -- whichever fires) marks
that candle's own High and Low as breakout trigger levels for what
follows. This is deliberately bidirectional: a sweep is read as "price
tested the average and got rejected for now," and it's the FOLLOW-
THROUGH breakout, not the sweep itself, that decides direction --
    next candle's High > pattern High -> BUY  (long, buy ATM CE)
    next candle's Low  < pattern Low  -> SELL (short, buy ATM PE)
A fresh sweep replaces any earlier still-pending one (most recent
evidence wins). If a single candle's range would trigger both the high
and low breakout at once (a wide-range candle), the high breakout is
checked first -- a simplification, not a claim about true intrabar
order, which OHLC bars can't resolve.

Stop-loss and target are on the OPTION PREMIUM itself, not the
underlying index -- a straight points move in whatever was bought
(ATM CE for a LONG signal, ATM PE for a SHORT signal), since either
way the position is a bought option and wants its premium to rise:
    stop   = entry premium - sl_points   (default 5)
    target = entry premium + target_points  (default 5)
Checked bar-by-bar against the OPTION's own 1-minute candle high/low
(not the index's) -- if a bar's range would touch both in the same
bar, the stop side is assumed to trigger first (conservative). This
replaced an earlier version of this module that sized stop/target off
the underlying index (the sweep candle's own range as the risk unit,
with a 1:2/1:3 index-point target and a later trailing-stop variant);
that approach was dropped because an index-level target doesn't
reliably track option premium P&L -- a bought option can lose value
to theta/IV even while the index itself is moving favorably, and gain
even while its target is technically still points away.

Exit: stop-loss, target, or forced flat at FORCE_FLAT_TIME -- there
is no more "exit on the next opposite signal": once in a trade, a
fresh sweep/breakout is tracked (so the next setup is ready) but does
not close the current position early. One position at a time.

The breakout confirmation itself (a 1-min candle's own high/low vs. the
pattern) needs no such correction -- it trades native 1-minute bars for
that part, and a bar's own close fully confirms its own signal there.
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


def _bar_end_ts(bar_ts: str, bar_minutes: int) -> str:
    """Full timestamp (same date/tz as bar_ts) at which this bucket
    closes -- the earliest moment its close/EMA value is usable."""
    hh, mm = int(bar_ts[11:13]), int(bar_ts[14:16])
    total = hh * 60 + mm + bar_minutes
    eh, em = divmod(total, 60)
    return f"{bar_ts[:11]}{eh:02d}:{em:02d}:00{bar_ts[19:]}"


def _align_ema_to_1min(all_1min: list[list], bars15: list[list], ema_15: list[float | None]) -> list[float | None]:
    """For each 1-min candle, the EMA value from the most recently
    COMPLETED 15-min bucket as of that candle's own timestamp (never
    the bucket the candle is currently inside)."""
    effective_from = [_bar_end_ts(b[0], 15) for b in bars15]
    out: list[float | None] = [None] * len(all_1min)
    j = -1
    for idx, row in enumerate(all_1min):
        ts = row[0]
        while j + 1 < len(bars15) and effective_from[j + 1] <= ts:
            j += 1
        if j >= 0:
            out[idx] = ema_15[j]
    return out


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    ema_fast: int = 9,
    ema_slow: int = 20,
    sl_points: float = 5.0,
    target_points: float = 5.0,
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

    bars15 = _resample(all_1min, 15)
    if len(bars15) < ema_slow + 2:
        return []
    closes15 = [b[4] for b in bars15]
    ema9_15 = _ema(closes15, ema_fast)
    ema20_15 = _ema(closes15, ema_slow)
    ema9 = _align_ema_to_1min(all_1min, bars15, ema9_15)
    ema20 = _align_ema_to_1min(all_1min, bars15, ema20_15)

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
    position = None  # dict: direction, entry_time, entry_price(premium), strike, expiry, lot_size, opt_type, date
    pattern = None    # dict: high, low
    current_day: str | None = None

    for i, row in enumerate(all_1min):
        ts, o, h, l, c, v, oi = row
        d = ts[:10]
        time_str = ts[11:16]

        if d != current_day:
            current_day = d
            pattern = None
            # never carry a position across a day boundary
            position = None

        atm = oc.round_to_step(c, strike_step)
        expiry = next((e for e in expiries if e >= d), None)

        def _enter(direction_label: str, pattern_high: float, pattern_low: float) -> None:
            nonlocal position
            if expiry is None:
                return
            opt_type = "CE" if direction_label == "LONG" else "PE"
            contract, candles = _atm_option_candles(atm, opt_type, d, expiry)
            if contract is None or not candles:
                return
            bar = _bar_at_or_after(candles, time_str) or _bar_at_or_before(candles, time_str)
            if bar is None:
                return
            entry_price = bar[4]
            position = {
                "direction": direction_label, "entry_time": bar[0], "entry_price": entry_price,
                "strike": contract["strike_price"], "expiry": expiry,
                "lot_size": contract["lot_size"], "opt_type": opt_type, "date": d,
                "stop_level": entry_price - sl_points, "target_level": entry_price + target_points,
                # bar[0][11:16] -> (high, low), for O(1) same-day premium lookups
                "opt_by_time": {c[0][11:16]: (c[2], c[3]) for c in candles},
            }

        def _exit(reason: str) -> None:
            nonlocal position
            if position is None:
                return
            _, candles = _atm_option_candles(position["strike"], position["opt_type"], position["date"], position["expiry"])
            bar = None
            if candles:
                bar = _bar_at_or_after(candles, time_str) or _bar_at_or_before(candles, time_str)
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
                _exit("eod")
            continue

        if position is not None:
            opt_hl = position["opt_by_time"].get(time_str)
            if opt_hl is not None:
                opt_h, opt_l = opt_hl
                if opt_l <= position["stop_level"]:
                    _exit("stop_loss")
                elif opt_h >= position["target_level"]:
                    _exit("target")

        e9, e20 = ema9[i], ema20[i]
        swept = any(
            ema_val is not None and ((h > ema_val and c <= ema_val) or (l < ema_val and c >= ema_val))
            for ema_val in (e9, e20)
        )
        if swept:
            # the candle that creates the pattern can't also confirm its
            # own breakout -- wait for the next one
            pattern = {"high": h, "low": l}
            continue

        if position is None and pattern is not None:
            if h > pattern["high"]:
                _enter("LONG", pattern["high"], pattern["low"])
                pattern = None
            elif l < pattern["low"]:
                _enter("SHORT", pattern["high"], pattern["low"])
                pattern = None

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
