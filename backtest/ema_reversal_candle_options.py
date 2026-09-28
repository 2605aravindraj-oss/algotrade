"""EMA9/EMA20 same-candle reversal, realized through real ATM NIFTY
options. Daily NIFTY 50 INDEX candles (this session's most recently
stated timeframe preference -- flag if intraday was actually meant).

Context + trigger, both on the SAME candle (no separate breakout
confirmation candle needed elsewhere in this codebase's other EMA
strategies -- here the signal candle IS the trigger):
    LONG (buy ATM CE):  EMA9 < EMA20 (downtrend context) AND this
                         candle's Low is below BOTH EMAs (a wick
                         through both, not necessarily the Open) AND
                         its Close is above BOTH EMAs -- a full sweep-
                         through-and-reclaim inside one candle.
    SHORT (buy ATM PE): EMA9 > EMA20 (uptrend context) AND High above
                         BOTH EMAs AND Close below BOTH EMAs -- the
                         mirror. Loosened from an earlier version that
                         required the Open (not just the Low) below
                         both EMAs.
Entry: at the signal candle's own close.

Stop-loss is a structural INDEX level (a genuine chart stop, not an
option-premium points value):
    LONG:  stop = signal candle's own Low
    SHORT: stop = signal candle's own High
Checked against each later day's INDEX high/low (not the option's) --
the index level only decides *when* to exit; the fill is still the
option's own last traded price that day.

Exit mode (exit_mode parameter):
    "target" (default): a fixed 1:2 risk/reward. risk = |entry price
        (the signal candle's close) - stop|; target = entry +/-
        2*risk. Stop checked first if a single day's range would
        touch both (conservative).
    "ride": no fixed target -- hold until a later day's Close crosses
        back through EMA9 (the faster/nearer line) against the
        position: below it for a LONG, above it for a SHORT. The
        structural stop still applies throughout either way.

POSITIONAL, not intraday -- a signal can take anywhere from a day to
several weeks to resolve, unlike this codebase's fixed-single-day-hold
daily modules. Since NIFTY options are WEEKLY, a position can outlive
its own contract: if a later day's date passes the held contract's
expiry before the position's own exit condition fires, it's forced
out at the last available price on/before expiry (reason "expiry") --
same convention as supertrend_options.py. One position at a time; a
fresh signal while already in a trade is skipped.
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.macd_rsi2_momentum_options import OptionTrade

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
    ema_fast: int = 9,
    ema_slow: int = 20,
    exit_mode: str = "target",
    target_multiple: float = 2.0,
    access_token: str | None = None,
) -> list[OptionTrade]:
    if exit_mode not in ("target", "ride"):
        raise ValueError('exit_mode must be "target" or "ride"')

    daily = upstox_client.get_daily_history(underlying_key, from_date, to_date)
    daily.sort(key=lambda row: row["date"])
    if len(daily) < ema_slow + 2:
        return []

    closes = [row["close"] for row in daily]
    ema9 = _ema(closes, ema_fast)
    ema20 = _ema(closes, ema_slow)

    expiries = sorted(cache.get_expired_expiries_cached(underlying_key, "options", access_token))
    chain_cache: dict[str, dict] = {}

    def _safe_day_candles(instrument_key: str, date: str) -> list[list]:
        try:
            return cache.get_day_candles_cached(instrument_key, "1minute", date, expired=True, access_token=access_token)
        except RuntimeError:
            return []

    def _atm_option_candles(strike, opt_type, date, expiry):
        if expiry not in chain_cache:
            try:
                chain = cache.get_expired_option_chain_cached(underlying_key, expiry, access_token)
            except RuntimeError:
                chain = []
            chain_cache[expiry] = oc.build_chain_lookup(chain)
        lookup = chain_cache[expiry]
        contract = oc.nearest_contract(lookup, strike, opt_type)
        if contract is None:
            return None, None
        candles = _safe_day_candles(contract["instrument_key"], date)
        return contract, candles

    trades: list[OptionTrade] = []
    position = None  # dict: direction, entry_date, entry_time, entry_price, strike, expiry, lot_size, instrument_key, stop_level, target_level

    def _exit(reason: str, as_of_date: str, use_last_bar: bool = False) -> None:
        nonlocal position
        candles = _safe_day_candles(position["instrument_key"], as_of_date)
        bar = None
        if candles:
            if use_last_bar:
                bar = oc.nearest_bar(candles, pick="last")
            else:
                bar = oc.nearest_bar(candles, pick="last")
        exit_time, exit_price = bar if bar else (f"{as_of_date}T15:30:00+05:30", position["entry_price"])
        trades.append(OptionTrade(
            date=position["entry_date"], direction=position["direction"], expiry=position["expiry"],
            strike=position["strike"], entry_time=position["entry_time"], entry_premium=position["entry_price"],
            exit_time=exit_time, exit_premium=exit_price, lot_size=position["lot_size"], exit_reason=reason,
        ))
        position = None

    for i, day in enumerate(daily):
        d = day["date"]
        o, h, l, c = day["open"], day["high"], day["low"], day["close"]

        if position is not None and d > position["expiry"]:
            _exit("expiry", position["expiry"], use_last_bar=True)

        if position is not None and position["entry_date"] != d:
            is_long = position["direction"] == "LONG"
            if exit_mode == "target":
                hit_stop = (l <= position["stop_level"]) if is_long else (h >= position["stop_level"])
                hit_target = (h >= position["target_level"]) if is_long else (l <= position["target_level"])
                if hit_stop:
                    _exit("stop_loss", d)
                elif hit_target:
                    _exit("target", d)
            else:
                hit_stop = (l <= position["stop_level"]) if is_long else (h >= position["stop_level"])
                if hit_stop:
                    _exit("stop_loss", d)
                elif ema9[i] is not None:
                    if is_long and c < ema9[i]:
                        _exit("trend_reverse", d)
                    elif not is_long and c > ema9[i]:
                        _exit("trend_reverse", d)

        if position is None and ema9[i] is not None and ema20[i] is not None:
            direction_label = None
            if ema9[i] < ema20[i] and l < ema9[i] and l < ema20[i] and c > ema9[i] and c > ema20[i]:
                direction_label = "LONG"
            elif ema9[i] > ema20[i] and h > ema9[i] and h > ema20[i] and c < ema9[i] and c < ema20[i]:
                direction_label = "SHORT"

            if direction_label is not None:
                expiry = next((e for e in expiries if e >= d), None)
                if expiry is not None:
                    opt_type = "CE" if direction_label == "LONG" else "PE"
                    atm = oc.round_to_step(c, strike_step)
                    contract, candles = _atm_option_candles(atm, opt_type, d, expiry)
                    if contract is not None and candles:
                        bar = oc.nearest_bar(candles, pick="last")
                        if bar is not None:
                            entry_time, entry_price = bar
                            stop_level = l if direction_label == "LONG" else h
                            risk = (c - stop_level) if direction_label == "LONG" else (stop_level - c)
                            if risk > 0:
                                target_level = (
                                    c + target_multiple * risk if direction_label == "LONG"
                                    else c - target_multiple * risk
                                )
                                position = {
                                    "direction": direction_label, "entry_date": d, "entry_time": entry_time,
                                    "entry_price": entry_price, "strike": contract["strike_price"], "expiry": expiry,
                                    "lot_size": contract["lot_size"], "instrument_key": contract["instrument_key"],
                                    "stop_level": stop_level, "target_level": target_level,
                                }

    if position is not None:
        _exit("eod_data_end", daily[-1]["date"])

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
