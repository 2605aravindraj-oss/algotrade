"""Narrow CPR (Central Pivot Range) breakout, realized through real
ATM NIFTY options, on 5-minute bars.

WHY A SEPARATE MODULE FROM backtest/supertrend_vwap_cross_options.py:
that module uses narrow CPR as a SECONDARY GATE on top of a
SuperTrend+VWAP signal (narrow_cpr_max_width_pct=0.26, tuned there --
the single strongest lever found in that strategy's whole tuning
history). Here narrow CPR IS the primary signal: no SuperTrend, no
VWAP -- just "today's CPR is narrow -> trade the breakout of
yesterday's CPR range." Narrow CPR's own threshold is re-tuned from
scratch for this use, not carried over from that module's value --
"predicts a trend day" (a secondary gate) and "triggers a breakout
entry" (a primary signal) are different jobs for the same regime idea,
and this codebase's rule all session has been: nothing transfers
without its own sweep.

CPR, computed from the PRIOR trading day's index H/L/C (no lookahead):
    pivot = (H + L + C) / 3
    BC = (H + L) / 2
    TC = 2*pivot - BC
    width_pct = |TC - BC| / C * 100
Only days where width_pct <= narrow_cpr_max_width_pct are traded at
all -- a day with a wide CPR (yesterday's range was wide, more
indecision) is skipped entirely, no entries taken.

SIGNAL: on a narrow-CPR day, wait for a FRESH breakout of yesterday's
CPR range on 5-minute closes (continuous across the day, VWAP-free):
    LONG  (buy ATM CE): previous close <= TC, this close > TC
    SHORT (buy ATM PE): previous close >= BC, this close < BC
One position at a time; by default at most one entry per day
(one_trade_per_day=True) -- a breakout that fails and re-triggers
later the same day does not re-enter unless disabled.

EXIT: the opposite breakout (TC breakout reverses through BC, or vice
versa), sl_pct/target_pct (premium %, both optional, off by default),
or forced-flat at FORCE_FLAT_TIME. All exit filters start off and are
tuned from scratch -- see run_nifty()'s docstring for the sweep
history once tuned.

STRIKE/EXPIRY: ATM = round_to_step(index close, strike_step), nearest
expiry on/after the entry day -- same convention as every other NIFTY
options module here.

COSTS: backtest.costs' F&O approximation via
macd_rsi2_momentum_options.OptionTrade, same cost model every other
options module in this codebase uses.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, _bar_at_or_after, _bar_at_or_before
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.sweep_reclaim_breakout import _resample

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"


def _daily_cpr(underlying_key: str, trading_days: list[dict], from_date: str, access_token: str | None) -> dict[str, dict]:
    """CPR for each day in trading_days, from the PRIOR trading day's H/L/C.
    Returns {date: {"pivot":, "bc":, "tc":, "width_pct":}}."""
    warmup_from = (datetime.strptime(from_date, "%Y-%m-%d") - timedelta(days=10)).strftime("%Y-%m-%d")
    cpr_days = upstox_client.get_daily_history(underlying_key, warmup_from, trading_days[-1]["date"])
    cpr_days.sort(key=lambda d: d["date"])
    by_date = {d["date"]: d for d in cpr_days}
    dates = sorted(by_date)

    cpr: dict[str, dict] = {}
    for day in trading_days:
        d = day["date"]
        idx = dates.index(d)
        if idx < 1:
            continue
        prev = by_date[dates[idx - 1]]
        h, l, c = prev["high"], prev["low"], prev["close"]
        pivot = (h + l + c) / 3
        bc = (h + l) / 2
        tc = 2 * pivot - bc
        width_pct = abs(tc - bc) / c * 100 if c else 0.0
        cpr[d] = {"pivot": pivot, "bc": min(bc, tc), "tc": max(bc, tc), "width_pct": width_pct}
    return cpr


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    candle_minutes: int = 5,
    narrow_cpr_max_width_pct: float = 0.30,
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

    cpr_by_date = _daily_cpr(underlying_key, trading_days, from_date, access_token)
    narrow_days = {d for d, c in cpr_by_date.items() if c["width_pct"] <= narrow_cpr_max_width_pct}

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
    prev_close: float | None = None
    traded_today = False

    for i, row in enumerate(bars):
        ts, o, h, l, c, v, oi = row
        d = ts[:10]
        time_str = ts[11:16]

        if d != current_day:
            current_day = d
            position = None
            prev_close = None
            traded_today = False

        cpr = cpr_by_date.get(d)
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
            prev_close = c
            continue

        if position is not None:
            is_long = position["direction"] == "LONG"
            hit_reversal = (is_long and c < cpr["bc"]) or (not is_long and c > cpr["tc"])
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

        if (position is None and cpr is not None and d in narrow_days and prev_close is not None
                and expiry is not None and not (one_trade_per_day and traded_today)):
            direction_label = None
            if prev_close <= cpr["tc"] and c > cpr["tc"] and not short_only:
                direction_label = "LONG"
            elif prev_close >= cpr["bc"] and c < cpr["bc"] and not long_only:
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

        prev_close = c

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
