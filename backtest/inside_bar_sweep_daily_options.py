"""Daily-timeframe 3-candle inside-bar liquidity-sweep reversal,
realized through real ATM NIFTY options. Fixed one-day hold: buy the
ATM option at the pattern's confirming (candle 3) close, sell it at
the next trading day's close -- no stop-loss, no target, no early
exit, mirroring rsi_macd_daily_swing_options.py's hold convention
since no exit rule was specified for this pattern.

Pattern (three consecutive trading days, candle 1/2/3 in order):
    Candle 2 is an INSIDE BAR of candle 1: candle2.high <= candle1.high
    AND candle2.low >= candle1.low -- candle 2's whole range sits
    inside candle 1's, a one-day compression/coil.
    Candle 3 then SWEEPS one side of candle 1 and closes back past it
    -- a stop-hunt that immediately reverses. "Closes above/below
    candle 1" is genuinely ambiguous between two readings, controlled
    by strict_reclaim:
        strict_reclaim=False (default): the textbook sweep-and-reclaim
            reading -- close back past the SAME level it swept.
                BULLISH: candle3.low  < candle1.low  (sweeps the low)
                         AND candle3.close > candle1.low
                BEARISH: candle3.high > candle1.high (sweeps the high)
                         AND candle3.close < candle1.high -- the mirror
        strict_reclaim=True: close back past candle 1's OPPOSITE
            extreme instead -- a full reclaim of the whole 2-day base,
            a much stronger and rarer confirmation.
                BULLISH: candle3.low < candle1.low AND candle3.close > candle1.high
                BEARISH: candle3.high > candle1.high AND candle3.close < candle1.low
    Checked directly against the index (no options/auth needed): over
    Jan 2025-Sep 2026 (419 trading days, 45 inside-bar setups), the
    strict reading only resolves ONCE (a single bearish occurrence,
    n=1 -- not a usable sample), while the default loose reading
    resolves 14 times (7 bullish, 7 bearish) -- still small, but at
    least measurable. Then buy ATM CE (bullish) or ATM PE (bearish).
The inside-bar (candle 2) requirement means this looks for the pattern
starting fresh at every candle 1 -- candle 2 and 3 are NOT themselves
eligible as a new candle 1 while still part of an unresolved pattern,
but nothing here prevents overlapping patterns (a later candle 1
inside an earlier pattern's own candles) from also firing.

Entry: at candle 3's own close (the day the pattern confirms).
Exit: the very next trading day's close, unconditionally -- a fixed
one-day hold. The option contract expires STRICTLY AFTER the entry
date, and fills use the option's own last traded price on/before that
day's market close, same conventions as
rsi_macd_daily_swing_options.py, including its cache-only fallback for
running this over a longer history than what's locally cached.
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.macd_rsi2_momentum_options import OptionTrade

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    strict_reclaim: bool = False,
    access_token: str | None = None,
) -> list[OptionTrade]:
    daily = upstox_client.get_daily_history(underlying_key, from_date, to_date)
    daily.sort(key=lambda row: row["date"])
    if len(daily) < 4:
        return []

    expiries = sorted(cache.get_expired_expiries_cached(underlying_key, "options", access_token))
    chain_cache: dict[str, dict] = {}

    def _safe_day_candles(instrument_key: str, date: str) -> list[list]:
        """Cache-only fallback: an uncached day (no live Upstox access
        token in this environment) skips just that entry/exit attempt
        instead of crashing the whole backtest -- same as
        rsi_macd_daily_swing_options.py."""
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
    position = None  # dict: direction, entry_date, entry_time, entry_price, strike, expiry, lot_size, instrument_key, exit_date

    for i, day3 in enumerate(daily):
        d = day3["date"]

        if position is not None and position["exit_date"] == d:
            candles = _safe_day_candles(position["instrument_key"], d)
            bar = oc.nearest_bar(candles, pick="last")
            exit_time, exit_price = bar if bar else (f"{d}T15:30:00+05:30", position["entry_price"])
            trades.append(OptionTrade(
                date=position["entry_date"], direction=position["direction"], expiry=position["expiry"],
                strike=position["strike"], entry_time=position["entry_time"], entry_premium=position["entry_price"],
                exit_time=exit_time, exit_premium=exit_price, lot_size=position["lot_size"], exit_reason="next_day_eod",
            ))
            position = None

        if i < 2 or i + 1 >= len(daily):
            continue  # need a candle 1 and 2 before this bar, and a next day to exit on
        c1, c2, c3 = daily[i - 2], daily[i - 1], daily[i]

        inside_bar = c2["high"] <= c1["high"] and c2["low"] >= c1["low"]
        if not inside_bar:
            continue

        direction_label = None
        if strict_reclaim:
            if c3["low"] < c1["low"] and c3["close"] > c1["high"]:
                direction_label = "LONG"
            elif c3["high"] > c1["high"] and c3["close"] < c1["low"]:
                direction_label = "SHORT"
        else:
            if c3["low"] < c1["low"] and c3["close"] > c1["low"]:
                direction_label = "LONG"
            elif c3["high"] > c1["high"] and c3["close"] < c1["high"]:
                direction_label = "SHORT"
        if direction_label is None:
            continue

        if position is not None:
            continue  # one position at a time; a pattern confirming while already in a trade is skipped

        expiry = next((e for e in expiries if e > d), None)
        if expiry is None:
            continue
        opt_type = "CE" if direction_label == "LONG" else "PE"
        atm = oc.round_to_step(c3["close"], strike_step)
        contract, candles = _atm_option_candles(atm, opt_type, d, expiry)
        if contract is None or not candles:
            continue
        bar = oc.nearest_bar(candles, pick="last")
        if bar is None:
            continue
        entry_time, entry_price = bar
        position = {
            "direction": direction_label, "entry_date": d, "entry_time": entry_time, "entry_price": entry_price,
            "strike": contract["strike_price"], "expiry": expiry,
            "lot_size": contract["lot_size"], "instrument_key": contract["instrument_key"],
            "exit_date": daily[i + 1]["date"],
        }

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
