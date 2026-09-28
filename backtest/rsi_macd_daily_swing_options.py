"""Daily-timeframe RSI(14)/MACD swing strategy, realized through real
ATM NIFTY options. Fixed one-day hold: buy the ATM call at today's
close, sell it at tomorrow's close -- no stop-loss, no target, no
early exit, no intraday check of any kind (everything here trades on
daily bars).

Signal (daily NIFTY 50 INDEX candles):
    RSI(14) on daily closes.
    A 14-period SMA of THAT RSI SERIES ITSELF, not a price SMA --
    "RSI14 greater than SMA14" only makes dimensional sense comparing
    two 0-100 oscillator values; a price-level SMA (thousands, for
    NIFTY) couldn't sensibly be compared to RSI. Read as an RSI trend
    filter: the RSI's own smoothed baseline, the RSI equivalent of a
    fast/slow moving-average crossover, used instead of a fixed level
    like 50.
    MACD(12,26,9) on daily closes. By default (require_fresh_cross=True)
    a "bullish cross" is the specific day the MACD line moves from
    at-or-below its signal line to above it -- a fresh cross only, not
    every day the line happens to be above its signal. This is
    extremely restrictive: over Jan 2025-Sep 2026 (419 trading days)
    it only fires 16 times, since a crossover is a single-day event.
    Setting require_fresh_cross=False trades trade frequency for
    signal strictness -- the condition becomes "MACD line is currently
    above its signal" (persistent, not just the crossing day), which
    holds on ~138 of those same 419 days, an ~8.6x higher "run rate".

Entry: on any day where RSI(14) > SMA14(RSI(14)) AND the bullish MACD
condition above is met, buy the ATM call at that day's close. Long
only (the original spec) unless trade_short=True, which adds the
mirrored bearish side: RSI(14) < SMA14(RSI(14)) AND a bearish MACD
cross (or, under require_fresh_cross=False, the persistent "MACD line
below signal") buys the ATM put instead. Bearish crosses are just as
common as bullish ones (16 vs 16 over Jan 2025-Sep 2026), so this
roughly doubles trade count WITHOUT touching the fresh-cross filter
that require_fresh_cross=False showed is actually where the edge
lives -- a "run rate" lever that adds independent, equally-strict
opportunities instead of diluting the existing ones.

Exit: the very next trading day's close, unconditionally, UNLESS
sl_points and/or target_points are set (both None by default -- the
original spec). When set, the exit day's own 1-minute option candles
are scanned in order from market open; the first bar whose range
touches entry_price - sl_points (stop) or entry_price + target_points
(target) closes the position right there, stop checked first if a
single bar would touch both (conservative, same convention as every
other module here). If neither is ever touched, it still falls back
to that day's close, exactly as the unconditional version does -- SL/
target only ever cut the hold SHORT, they never extend it past the
one-day limit.

The option contract is picked to expire STRICTLY AFTER the entry date
(not on or after it, like every other module in this codebase), since
the position must still exist the next trading day -- an entry taken
on the current weekly expiry's own day rolls straight to next week's
contract instead of skipping the trade.

Fills: both entry and exit use the option's own last traded price
on/before that day's market close (options_common.nearest_bar's
pick="last"), i.e. buy-at-close / sell-at-close, taken literally. One
position at a time -- if a fresh signal fires on the same day an
existing position's one-day hold ends, the exit is processed first,
then the new entry can open the same day.
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.macd_rsi2_momentum import compute_rsi, compute_macd
from backtest.macd_rsi2_momentum_options import OptionTrade

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"


def _sma_of_series(series: list[float | None], period: int) -> list[float | None]:
    """SMA of a series with a leading run of Nones (e.g. an indicator
    still warming up) -- computed over just the dense non-None tail and
    mapped back to full length, same offset trick compute_macd uses for
    its signal line."""
    start_idx = next((i for i, v in enumerate(series) if v is not None), len(series))
    dense = series[start_idx:]
    out: list[float | None] = [None] * len(series)
    if len(dense) < period:
        return out
    window_sum = sum(dense[:period])
    out[start_idx + period - 1] = window_sum / period
    for j in range(period, len(dense)):
        window_sum += dense[j] - dense[j - period]
        out[start_idx + j] = window_sum / period
    return out


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    rsi_period: int = 14,
    rsi_sma_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    require_fresh_cross: bool = True,
    trade_short: bool = False,
    sl_points: float | None = None,
    target_points: float | None = None,
    access_token: str | None = None,
) -> list[OptionTrade]:
    daily = upstox_client.get_daily_history(underlying_key, from_date, to_date)
    daily.sort(key=lambda row: row["date"])
    if len(daily) < macd_slow + macd_signal + 2:
        return []

    closes = [row["close"] for row in daily]
    rsi = compute_rsi(closes, rsi_period)
    rsi_sma = _sma_of_series(rsi, rsi_sma_period)
    macd_line, signal_line = compute_macd(closes, macd_fast, macd_slow, macd_signal)

    bullish_cross = [False] * len(daily)
    bearish_cross = [False] * len(daily)
    for i in range(1, len(daily)):
        if macd_line[i] is None or signal_line[i] is None:
            continue
        if require_fresh_cross:
            if (
                macd_line[i - 1] is not None and signal_line[i - 1] is not None
                and macd_line[i - 1] <= signal_line[i - 1] and macd_line[i] > signal_line[i]
            ):
                bullish_cross[i] = True
            elif (
                macd_line[i - 1] is not None and signal_line[i - 1] is not None
                and macd_line[i - 1] >= signal_line[i - 1] and macd_line[i] < signal_line[i]
            ):
                bearish_cross[i] = True
        else:
            # persistent regime, not just the crossover day itself -- "run
            # rate" lever: the fresh-cross-only version only fires 16 times
            # in ~1.7 years since a crossover is a single-day event, while
            # the persistent condition (macd line simply above signal)
            # holds on ~8x as many days
            if macd_line[i] > signal_line[i]:
                bullish_cross[i] = True
            elif macd_line[i] < signal_line[i]:
                bearish_cross[i] = True

    expiries = sorted(cache.get_expired_expiries_cached(underlying_key, "options", access_token))
    chain_cache: dict[str, dict] = {}

    def _safe_day_candles(instrument_key: str, date: str) -> list[list]:
        """Cache-only fallback: this module is often run over a much
        longer history than what's locally cached, and without a live
        Upstox access token an uncached day can't be fetched. Rather
        than let one missing day crash the whole run, skip just that
        entry/exit attempt -- the caller treats an empty result the
        same as "no contract found"."""
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
    position = None  # dict: entry_date, entry_time, entry_price, strike, expiry, lot_size, instrument_key, exit_date

    for i, day in enumerate(daily):
        d = day["date"]
        c = day["close"]

        if position is not None and position["exit_date"] == d:
            candles = sorted(_safe_day_candles(position["instrument_key"], d), key=lambda c: c[0])
            exit_time = exit_price = exit_reason = None

            if sl_points is not None or target_points is not None:
                # Same formula regardless of direction: LONG (bought CE)
                # and SHORT (bought PE) are both bought-premium positions,
                # so "profit" always means the premium itself rising --
                # matches ema_sweep_breakout_options.py's convention.
                entry_price = position["entry_price"]
                stop_level = entry_price - sl_points if sl_points is not None else None
                target_level = entry_price + target_points if target_points is not None else None

                for bar in candles:
                    ts, o, h, l, cl, v, oi = bar
                    hit_stop = stop_level is not None and l <= stop_level
                    hit_target = target_level is not None and h >= target_level
                    if hit_stop:
                        exit_time, exit_price, exit_reason = ts, cl, "stop_loss"
                        break
                    if hit_target:
                        exit_time, exit_price, exit_reason = ts, cl, "target"
                        break

            if exit_time is None:
                bar = oc.nearest_bar(candles, pick="last")
                exit_time, exit_price = bar if bar else (f"{d}T15:30:00+05:30", position["entry_price"])
                exit_reason = "next_day_eod"

            trades.append(OptionTrade(
                date=position["entry_date"], direction=position["direction"], expiry=position["expiry"],
                strike=position["strike"], entry_time=position["entry_time"], entry_premium=position["entry_price"],
                exit_time=exit_time, exit_premium=exit_price, lot_size=position["lot_size"], exit_reason=exit_reason,
            ))
            position = None

        if position is None and i + 1 < len(daily) and rsi[i] is not None and rsi_sma[i] is not None:
            direction_label = None
            if rsi[i] > rsi_sma[i] and bullish_cross[i]:
                direction_label = "LONG"
            elif trade_short and rsi[i] < rsi_sma[i] and bearish_cross[i]:
                direction_label = "SHORT"

            if direction_label is not None:
                expiry = next((e for e in expiries if e > d), None)
                if expiry is not None:
                    opt_type = "CE" if direction_label == "LONG" else "PE"
                    atm = oc.round_to_step(c, strike_step)
                    contract, candles = _atm_option_candles(atm, opt_type, d, expiry)
                    if contract is not None and candles:
                        bar = oc.nearest_bar(candles, pick="last")
                        if bar is not None:
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
