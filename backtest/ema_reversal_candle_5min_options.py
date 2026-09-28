"""EMA9/EMA20 same-candle reversal, realized through real ATM NIFTY
options. 5-minute NIFTY 50 INDEX candles -- the intraday version of
ema_reversal_candle_options.py (which defaulted to daily since no
timeframe was given at the time).

Context + trigger, both on the SAME 5-minute candle:
    LONG (buy ATM CE):  EMA9 < EMA20 (downtrend context) AND this
                         candle's Low is below BOTH EMAs (a wick
                         through both, not necessarily the Open) AND
                         its Close is above BOTH EMAs.
    SHORT (buy ATM PE): EMA9 > EMA20 (uptrend context) AND High above
                         BOTH EMAs AND Close below BOTH EMAs -- the
                         mirror. Loosened from an earlier version that
                         required the Open (not just the Low) below
                         both EMAs -- since a candle's Low is always
                         <= its Open, this admits candles that opened
                         ABOVE the EMAs, wicked down through both
                         intra-bar, and still reclaimed by the close.
EMA9/EMA20 computed on these same 5-minute bars, continuous across the
whole date range (needs warm-up, not reset daily) -- no multi-timeframe
alignment is needed since signal and EMA share one timeframe.

Entry: at the signal candle's own close. Stop-loss is a structural
INDEX level (the signal candle's own Low for a LONG, High for a
SHORT), checked against each later 5-min candle's index high/low.

Exit mode (exit_mode parameter):
    "target" (default): fixed 1:2 risk/reward (risk = |entry - stop|,
        entry = signal candle's own close). Stop checked first if a
        single bar would touch both (conservative).
    "ride": no fixed target -- hold until a later candle's Close
        crosses back through EMA9 against the position.

Intraday only: forced flat at FORCE_FLAT_TIME, never carries a
position across a day boundary -- unlike the daily version, a 5-minute
NIFTY weekly option contract never gets anywhere near its own expiry
within a single session, so the "forced out at expiry, not by the
strategy's own logic" contamination that dominated the daily version's
results doesn't apply here. Entries and exits fill at the option's own
price as of the deciding candle's CLOSE (bucket start + candle_minutes),
same decision-time-correct convention as every other intraday module
in this codebase. One position at a time.
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
    ema_fast: int = 9,
    ema_slow: int = 20,
    exit_mode: str = "target",
    target_multiple: float = 2.0,
    access_token: str | None = None,
) -> list[OptionTrade]:
    if exit_mode not in ("target", "ride"):
        raise ValueError('exit_mode must be "target" or "ride"')

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
    ema9 = _ema(closes, ema_fast)
    ema20 = _ema(closes, ema_slow)

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

    for i, row in enumerate(bars):
        ts, o, h, l, c, v, oi = row
        d = ts[:10]
        time_str = ts[11:16]

        if d != current_day:
            current_day = d
            position = None

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
            if exit_mode == "target":
                hit_stop = (l <= position["stop_level"]) if is_long else (h >= position["stop_level"])
                hit_target = (h >= position["target_level"]) if is_long else (l <= position["target_level"])
                if hit_stop:
                    _close("stop_loss")
                elif hit_target:
                    _close("target")
            else:
                hit_stop = (l <= position["stop_level"]) if is_long else (h >= position["stop_level"])
                if hit_stop:
                    _close("stop_loss")
                elif ema9[i] is not None:
                    if is_long and c < ema9[i]:
                        _close("trend_reverse")
                    elif not is_long and c > ema9[i]:
                        _close("trend_reverse")

        if position is None and ema9[i] is not None and ema20[i] is not None and expiry is not None:
            direction_label = None
            if ema9[i] < ema20[i] and l < ema9[i] and l < ema20[i] and c > ema9[i] and c > ema20[i]:
                direction_label = "LONG"
            elif ema9[i] > ema20[i] and h > ema9[i] and h > ema20[i] and c < ema9[i] and c < ema20[i]:
                direction_label = "SHORT"

            if direction_label is not None:
                stop_level = l if direction_label == "LONG" else h
                risk = (c - stop_level) if direction_label == "LONG" else (stop_level - c)
                if risk > 0:
                    opt_type = "CE" if direction_label == "LONG" else "PE"
                    contract, candles = _atm_option_candles(atm, opt_type, d, expiry)
                    if contract is not None and candles:
                        bar = _fill(candles, decision_time_str)
                        if bar is not None:
                            entry_price = bar[4]
                            target_level = (
                                c + target_multiple * risk if direction_label == "LONG"
                                else c - target_multiple * risk
                            )
                            position = {
                                "direction": direction_label, "entry_time": bar[0], "entry_price": entry_price,
                                "strike": contract["strike_price"], "expiry": expiry,
                                "lot_size": contract["lot_size"], "opt_type": opt_type, "date": d,
                                "stop_level": stop_level, "target_level": target_level,
                            }

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
