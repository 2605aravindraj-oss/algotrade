"""MACD histogram zero-line crossover, realized through real ATM NIFTY
options, on 5-minute NIFTY 50 SPOT index candles -- not futures. MACD
needs no volume (it's EMA-of-close throughout), so unlike the VWAP-
based modules in this codebase (which need a volume-bearing feed and
fall back to futures), this one can use the index's own candles
directly, exactly as requested.

SIGNAL: MACD(12,26,9) computed continuously across the whole date
range (the EMAs need warm-up, never reset daily -- only the position
state resets at day boundaries). histogram = MACD line - signal line.
    LONG  (buy ATM CE): histogram crosses from <=0 to >0 on this
        candle -- a FRESH cross, not merely "is positive".
    SHORT (buy ATM PE): histogram crosses from >=0 to <0 -- the
        mirror. Entry fills at the signal candle's own close.

EXIT: the opposite cross (histogram crosses back through zero against
the position) -- the natural exit for a continuous regime indicator,
same convention as supertrend_vwap_cross_options.py's "SuperTrend
flips against the position" -- plus optional premium sl_pct/
target_pct (both off by default) and forced-flat at FORCE_FLAT_TIME.

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
from backtest.technical_rating import _macd

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"


def _apply_slippage(price: float, side: str, slippage_pct: float) -> float:
    """Adverse execution: a BUY fills slippage_pct higher, a SELL fills
    slippage_pct lower, than the observed candle price."""
    if slippage_pct <= 0:
        return price
    return price * (1 + slippage_pct) if side == "BUY" else price * (1 - slippage_pct)


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    candle_minutes: int = 5,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sl_pct: float | None = None,
    target_pct: float | None = None,
    slippage_pct: float = 0.0,
    access_token: str | None = None,
) -> list[OptionTrade]:
    trading_days = upstox_client.get_daily_history(underlying_key, from_date, to_date)
    trading_days.sort(key=lambda d: d["date"])
    if len(trading_days) < 1:
        return []

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
    closes = [b[4] for b in bars]
    macd_line, signal_line = _macd(closes, macd_fast, macd_slow, macd_signal)
    histogram = [
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
    position = None  # dict: direction, entry_time, entry_price, strike, expiry, lot_size, opt_type, date
    current_day: str | None = None
    prev_hist: float | None = None

    for i, row in enumerate(bars):
        ts, o, h, l, c, v, oi = row
        d = ts[:10]
        time_str = ts[11:16]

        if d != current_day:
            current_day = d
            position = None
            prev_hist = None

        hist = histogram[i]
        atm = oc.round_to_step(c, strike_step)
        expiry = next((e for e in expiries if e >= d), None)

        _dh, _dm = divmod(int(time_str[:2]) * 60 + int(time_str[3:5]) + candle_minutes, 60)
        decision_time_str = f"{_dh:02d}:{_dm:02d}"

        def _close(reason: str) -> None:
            nonlocal position
            _, candles = _atm_option_candles(position["strike"], position["opt_type"], position["date"], position["expiry"])
            bar = _fill(candles, decision_time_str) if candles else None
            exit_price = _apply_slippage(bar[4], "SELL", slippage_pct) if bar else position["entry_price"]
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
            prev_hist = hist
            continue

        if hist is None:
            prev_hist = hist
            continue

        if position is not None:
            is_long = position["direction"] == "LONG"
            hit_reversal = (is_long and hist < 0) or (not is_long and hist > 0)
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

        if position is None and prev_hist is not None and expiry is not None:
            direction_label = None
            if prev_hist <= 0 and hist > 0:
                direction_label = "LONG"
            elif prev_hist >= 0 and hist < 0:
                direction_label = "SHORT"

            if direction_label is not None:
                opt_type = "CE" if direction_label == "LONG" else "PE"
                contract, candles = _atm_option_candles(atm, opt_type, d, expiry)
                if contract is not None and candles:
                    bar = _fill(candles, decision_time_str)
                    if bar is not None:
                        position = {
                            "direction": direction_label, "entry_time": bar[0],
                            "entry_price": _apply_slippage(bar[4], "BUY", slippage_pct),
                            "strike": contract["strike_price"], "expiry": expiry,
                            "lot_size": contract["lot_size"], "opt_type": opt_type, "date": d,
                        }

        prev_hist = hist

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
