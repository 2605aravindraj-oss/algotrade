"""Synthetic-ATM straddle breakout, realized through real NIFTY options.
Daily reference computed from one day's option CLOSING prices, traded
the NEXT day on 5-minute candles.

STAGE 1 -- find the "true" ATM strike for day D (a strike-selection
refinement on top of this codebase's usual round-to-nearest-50): among
a small band of strikes around the index close (atm_guess +/-
strike_search_range * strike_step), fetch that day's CE and PE closing
premium (last traded price at/before market close) for each strike,
and pick the strike where |CE_close - PE_close| is SMALLEST -- the
put-call-parity reading of "true" ATM (where a call and a put cost the
same, the strike the forward is trading nearest to), which can differ
from the plain index-rounded strike whenever the market carries a
skew. Call this best_strike; call its avg_price = (CE_close +
PE_close) / 2 the day's reference level.

STAGE 2 -- next trading day, watch that SAME contract (best_strike,
same expiry) on 5-minute candles:
    LONG (buy that CE):  the first 5-min candle whose CE close breaks
                          above the PRIOR day's avg_price.
    SHORT (buy that PE):  the first 5-min candle whose PE close breaks
                          above the PRIOR day's avg_price -- the two
                          legs are watched independently (both can fire
                          the same day; each fires at most once).
Entry fills at that signal candle's own close -- no separate breakout-
confirmation bar, the crossing candle IS the entry, same convention as
ema_reversal_candle_options.py.

EXIT -- "target EOD": no stop-loss, no profit target: once entered,
each leg is held to the forced-flat close (FORCE_FLAT_TIME), the day's
target being simply to hold to end of day, per the request. If the
reference day D's expiry has already passed by the trading day (D was
itself the expiry date), that contract is dead and no trade is taken
(no data to trigger on).

One entry per leg per day; a leg already filled that day is not
re-entered even if its price dips back below avg_price and re-crosses.
Everything resets at the next day boundary (a fresh reference is
computed from THAT day's own close for the day after).
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.sweep_reclaim_breakout import _resample

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    candle_minutes: int = 5,
    strike_search_range: int = 2,
    access_token: str | None = None,
) -> list[OptionTrade]:
    trading_days = upstox_client.get_daily_history(underlying_key, from_date, to_date)
    trading_days.sort(key=lambda row: row["date"])
    if len(trading_days) < 2:
        return []

    expiries = sorted(cache.get_expired_expiries_cached(underlying_key, "options", access_token))
    chain_cache: dict[str, dict] = {}

    def _lookup(expiry):
        if expiry not in chain_cache:
            chain_cache[expiry] = oc.build_chain_lookup(
                cache.get_expired_option_chain_cached(underlying_key, expiry, access_token)
            )
        return chain_cache[expiry]

    def _day_candles(instrument_key, date):
        return cache.get_day_candles_cached(instrument_key, "1minute", date, expired=True, access_token=access_token)

    def _index_close(date):
        rows = sorted(
            cache.get_day_candles_cached(underlying_key, "1minute", date, expired=False),
            key=lambda c: c[0],
        )
        return rows[-1][4] if rows else None

    # -- STAGE 1: compute each day's reference (best_strike, avg_price, expiry) --
    reference: dict[str, dict | None] = {}
    for day in trading_days:
        d = day["date"]
        expiry = next((e for e in expiries if e >= d), None)
        idx_close = _index_close(d)
        if expiry is None or idx_close is None:
            reference[d] = None
            continue
        lookup = _lookup(expiry)
        atm_guess = oc.round_to_step(idx_close, strike_step)
        best = None  # (diff, strike, avg_price)
        for k in range(-strike_search_range, strike_search_range + 1):
            strike = atm_guess + k * strike_step
            ce = lookup.get((strike, "CE"))
            pe = lookup.get((strike, "PE"))
            if ce is None or pe is None:
                continue
            ce_bar = oc.nearest_bar(_day_candles(ce["instrument_key"], d), pick="last")
            pe_bar = oc.nearest_bar(_day_candles(pe["instrument_key"], d), pick="last")
            if ce_bar is None or pe_bar is None:
                continue
            ce_close, pe_close = ce_bar[1], pe_bar[1]
            diff = abs(ce_close - pe_close)
            if best is None or diff < best[0]:
                best = (diff, strike, (ce_close + pe_close) / 2)
        reference[d] = {"strike": best[1], "avg_price": best[2], "expiry": expiry} if best else None

    # -- STAGE 2: trade day D+1 off day D's reference --
    trades: list[OptionTrade] = []
    for i in range(1, len(trading_days)):
        d = trading_days[i]["date"]
        ref = reference[trading_days[i - 1]["date"]]
        if ref is None or ref["expiry"] < d:
            continue
        lookup = _lookup(ref["expiry"])
        ce_contract = lookup.get((ref["strike"], "CE"))
        pe_contract = lookup.get((ref["strike"], "PE"))
        avg_price = ref["avg_price"]

        for contract, opt_type, direction in (
            (ce_contract, "CE", "LONG"),
            (pe_contract, "PE", "SHORT"),
        ):
            if contract is None:
                continue
            bars = _resample(sorted(_day_candles(contract["instrument_key"], d), key=lambda c: c[0]), candle_minutes)
            if not bars:
                continue
            position = None
            for row in bars:
                ts, o, h, l, c, v, oi = row
                time_str = ts[11:16]
                if time_str >= FORCE_FLAT_TIME:
                    if position is not None:
                        trades.append(OptionTrade(
                            date=d, direction=direction, expiry=ref["expiry"], strike=ref["strike"],
                            entry_time=position["entry_time"], entry_premium=position["entry_price"],
                            exit_time=ts, exit_premium=c, lot_size=contract["lot_size"], exit_reason="eod",
                        ))
                        position = None
                    break
                if position is None and c > avg_price:
                    position = {"entry_time": ts, "entry_price": c}
            if position is not None:
                last_ts, last_close = bars[-1][0], bars[-1][4]
                trades.append(OptionTrade(
                    date=d, direction=direction, expiry=ref["expiry"], strike=ref["strike"],
                    entry_time=position["entry_time"], entry_premium=position["entry_price"],
                    exit_time=last_ts, exit_premium=last_close, lot_size=contract["lot_size"], exit_reason="eod_data_end",
                ))

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
