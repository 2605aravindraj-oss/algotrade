"""Tunable variant of futures_oi_buildup.py for testing fixes to the
whipsaw/noise problems the corrected (post fill-pricing-bug) backtest
exposed. Each parameter defaults to the exact baseline behavior; change
one at a time to isolate its effect.

Parameters:
    bar_minutes:      resample bucket size (baseline: 5)
    confirm_bars:     require this many consecutive identical buildup
                       readings before opening a new leg (baseline: 1,
                       i.e. no confirmation -- act on a single bar)
    auto_reverse:     if False, an opposing signal only exits to flat;
                       a fresh signal is required to re-enter (baseline:
                       True, i.e. exit-and-immediately-flip)
    min_price_move:   ignore buildup classification if the bar's price
                       change is smaller than this (baseline: 0)
    min_oi_move:      same, for OI change (baseline: 0)
    stop_loss_points: hard stop on the option premium, checked against
                       each bar's high/low while in position (baseline:
                       None, i.e. no stop beyond the strategy's own exits)

Reuses the same real options cost model, forced-flat-at-15:25, and
corrected (decision-time, not bucket-start) fill lookup as
futures_oi_buildup.py.
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, UNDERLYING_KEY, BUILDUP_LABELS
from backtest.macd_rsi2_momentum_options import OptionTrade


def classify_buildup(price_change: float, oi_change: float,
                      min_price_move: float = 0.0, min_oi_move: float = 0.0) -> str:
    if abs(price_change) < min_price_move or abs(oi_change) < min_oi_move:
        return "Neutral"
    if price_change == 0 or oi_change == 0:
        return "Neutral"
    return BUILDUP_LABELS[(1 if price_change > 0 else -1, 1 if oi_change > 0 else -1)]


def _resample(rows_1min: list[list], bar_minutes: int) -> list[list]:
    buckets: dict[str, list] = {}
    order: list[str] = []
    for row in rows_1min:
        ts, o, h, l, c, v, oi = row
        hh, mm = int(ts[11:13]), int(ts[14:16])
        minutes_since_open = (hh * 60 + mm) - (9 * 60 + 15)
        bucket_idx = max(minutes_since_open, 0) // bar_minutes
        bucket_start_minutes = 9 * 60 + 15 + bucket_idx * bar_minutes
        bh, bm = divmod(bucket_start_minutes, 60)
        bucket_ts = f"{ts[:11]}{bh:02d}:{bm:02d}:00{ts[19:]}"
        if bucket_ts not in buckets:
            buckets[bucket_ts] = [o, h, l, c, v, oi]
            order.append(bucket_ts)
        else:
            b = buckets[bucket_ts]
            b[1] = max(b[1], h)
            b[2] = min(b[2], l)
            b[3] = c
            b[4] += v
            b[5] = oi
    return [[ts] + buckets[ts] for ts in order]


def _bar_at_or_before(rows, time_str):
    candidates = [r for r in rows if r[0][11:16] <= time_str]
    return candidates[-1] if candidates else None


def _bar_at_or_after(rows, time_str):
    for r in rows:
        if r[0][11:16] >= time_str:
            return r
    return None


def run(
    from_date: str,
    to_date: str,
    futures_key: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    access_token: str | None = None,
    futures_expired: bool = False,
    bar_minutes: int = 5,
    confirm_bars: int = 1,
    auto_reverse: bool = True,
    min_price_move: float = 0.0,
    min_oi_move: float = 0.0,
    stop_loss_points: float | None = None,
) -> list[OptionTrade]:
    if futures_expired:
        raw_days = upstox_client.get_expired_candles(futures_key, "day", to_date, from_date, access_token)
        fut_days = [{"date": d} for d in sorted({c[0][:10] for c in raw_days})]
    else:
        fut_days = upstox_client.get_daily_history(futures_key, from_date, to_date)

    all_bars: list[list] = []
    for day in fut_days:
        rows_1min = sorted(
            cache.get_day_candles_cached(futures_key, "1minute", day["date"], expired=futures_expired, access_token=access_token),
            key=lambda c: c[0],
        )
        all_bars.extend(_resample(rows_1min, bar_minutes))
    all_bars.sort(key=lambda c: c[0])
    if len(all_bars) < 2:
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

    trades: list[OptionTrade] = []
    position = None
    prev_close = None
    prev_oi = None
    reading_streak: list[str] = []  # last `confirm_bars` buildup readings

    for i, row in enumerate(all_bars):
        ts, o, h, l, c, v, oi = row
        d = ts[:10]
        time_str = ts[11:16]
        _dh, _dm = divmod(int(time_str[:2]) * 60 + int(time_str[3:5]) + bar_minutes, 60)
        decision_time_str = f"{_dh:02d}:{_dm:02d}"

        if prev_close is None:
            prev_close, prev_oi = c, oi
            continue

        buildup = classify_buildup(c - prev_close, oi - prev_oi, min_price_move, min_oi_move)
        reading_streak.append(buildup)
        reading_streak[:] = reading_streak[-confirm_bars:]
        confirmed = len(reading_streak) == confirm_bars and len(set(reading_streak)) == 1
        confirmed_buildup = buildup if confirmed else "Neutral"

        atm = oc.round_to_step(c, strike_step)
        expiry = next((e for e in expiries if e >= d), None)

        def _enter(direction: str) -> None:
            nonlocal position
            if expiry is None:
                return
            opt_type = "CE" if direction == "LONG" else "PE"
            contract, candles = _atm_option_candles(atm, opt_type, d, expiry)
            if contract is None or not candles:
                return
            bar = _bar_at_or_after(candles, decision_time_str) or _bar_at_or_before(candles, decision_time_str)
            if bar is None:
                return
            position = {
                "direction": direction, "entry_time": bar[0], "entry_price": bar[4],
                "strike": contract["strike_price"], "expiry": expiry,
                "lot_size": contract["lot_size"], "opt_type": opt_type, "date": d,
            }

        def _exit(reason: str, price_override: float | None = None, time_override: str | None = None) -> None:
            nonlocal position
            if position is None:
                return
            _, candles = _atm_option_candles(position["strike"], position["opt_type"], position["date"], position["expiry"])
            bar = None
            if candles:
                bar = _bar_at_or_after(candles, decision_time_str) or _bar_at_or_before(candles, decision_time_str)
            exit_price = price_override if price_override is not None else (bar[4] if bar else position["entry_price"])
            exit_time = time_override if time_override is not None else (bar[0] if bar else ts)
            trades.append(OptionTrade(
                date=position["date"], direction=position["direction"], expiry=position["expiry"],
                strike=position["strike"], entry_time=position["entry_time"], entry_premium=position["entry_price"],
                exit_time=exit_time, exit_premium=exit_price, lot_size=position["lot_size"], exit_reason=reason,
            ))
            position = None

        # Stop-loss check against this bar's option high/low, before signal logic
        if position is not None and stop_loss_points is not None:
            _, candles = _atm_option_candles(position["strike"], position["opt_type"], position["date"], position["expiry"])
            bar = _bar_at_or_after(candles, decision_time_str) if candles else None
            if bar is not None:
                stop_level = position["entry_price"] - stop_loss_points  # long: premium falling
                if position["direction"] == "LONG" and bar[3] <= stop_level:  # low
                    _exit("stop_loss", price_override=stop_level, time_override=bar[0])
                elif position["direction"] == "SHORT":
                    stop_level_short = position["entry_price"] + stop_loss_points  # short: premium rising hurts
                    if bar[2] >= stop_level_short:  # high
                        _exit("stop_loss", price_override=stop_level_short, time_override=bar[0])

        if position is None:
            if confirmed_buildup == "Long Buildup":
                _enter("LONG")
            elif confirmed_buildup == "Short Buildup":
                _enter("SHORT")
        else:
            if time_str >= FORCE_FLAT_TIME:
                _exit("eod")
            elif position["direction"] == "LONG" and buildup == "Short Buildup":
                _exit("reverse" if auto_reverse else "exit_opposite")
                if auto_reverse:
                    _enter("SHORT")
            elif position["direction"] == "LONG" and buildup == "Long Unwinding":
                _exit("unwind")
            elif position["direction"] == "SHORT" and buildup == "Long Buildup":
                _exit("reverse" if auto_reverse else "exit_opposite")
                if auto_reverse:
                    _enter("LONG")
            elif position["direction"] == "SHORT" and buildup == "Short Covering":
                _exit("covering")

        prev_close, prev_oi = c, oi

    if position is not None:
        _, candles = _atm_option_candles(position["strike"], position["opt_type"], position["date"], position["expiry"])
        last_bar = candles[-1] if candles else None
        exit_price = last_bar[4] if last_bar else position["entry_price"]
        exit_time = last_bar[0] if last_bar else all_bars[-1][0]
        trades.append(OptionTrade(
            date=position["date"], direction=position["direction"], expiry=position["expiry"],
            strike=position["strike"], entry_time=position["entry_time"], entry_premium=position["entry_price"],
            exit_time=exit_time, exit_premium=exit_price, lot_size=position["lot_size"], exit_reason="eod_data_end",
        ))

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
