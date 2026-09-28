"""Synthetic-ATM straddle breakout, realized through real NIFTY options.
Daily reference computed from one day's option CLOSING prices, traded
the NEXT day on 1-minute candles (candle_minutes default, retuned down
from an original 5-minute default -- see TIMEFRAME note below).

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
same expiry) on candle_minutes candles (default 1):
    LONG (buy that CE):  the first candle whose CE close breaks above
                          the PRIOR day's avg_price.
    SHORT (buy that PE):  the first candle whose PE close breaks above
                          the PRIOR day's avg_price -- the two legs are
                          watched independently (both can fire the same
                          day; each fires at most once).
Entry fills at that signal candle's own close -- no separate breakout-
confirmation bar, the crossing candle IS the entry, same convention as
ema_reversal_candle_options.py.

TIMEFRAME: candle_minutes defaults to 1 (originally 5). Tested across 4
non-overlapping windows spanning the full available option-data history
(2024-10-01 to 2026-09-08 in 4 chunks) at the sl=14/target=28 setting
below: 1-minute entries beat 5-minute on 3 of 4 windows and roughly
halved the loss on the 4th (net total across all 4 windows: +Rs 21,134
on 1-min vs -Rs 5,214 on 5-min) -- a broad improvement, not concentrated
in one window.

EXIT -- sl_points / target_points, default 13 / 26 (a 1:2 risk/reward,
retuned from the original "target EOD" -- no stop-loss, no profit
target, held to the forced-flat close -- which lost heavily; pass
sl_points=None, target_points=None to restore that original behavior).
Tuning history: on 5-minute candles, sweeping the 1:2 ratio track found
sl_points 11-18 a genuine plateau (not an isolated spike) on the first
window tested, which held up on a second window too -- but a THIRD
window (2024-10-01 to 2025-01-31) came back deeply negative across
that entire plateau, showing the tuning doesn't hold up across all
market regimes. After switching to 1-minute entries (see TIMEFRAME
above), re-swept the 1:2 ratio track and found a new, similarly genuine
plateau at sl_points 9-14 (bounded by noise below and a cliff above) on
the original window, peaking at sl=12. Validated across all 4 windows:
every setting from 9-14 is STILL negative on that same problem window
(2024-10-01 to 2025-01-31, -Rs 2,767 to -Rs 10,140 depending on
setting) -- confirming that specific period is a genuinely hard regime
for this strategy, not fixable by exit tuning -- but the other 3
windows are solidly positive across most of the range. Totals across
all 4 windows: sl=13/target=26 (+Rs 24,729) edges out sl=14/target=28
(+Rs 21,134), a difference small enough to be noise between two
adjacent plateau points; 13/26 was picked as the default per user
preference. Setting either switches that leg to a
premium-points exit, same convention as ema_sweep_breakout_options.py:
    stop_level   = entry premium - sl_points
    target_level = entry premium + target_points
Same formula regardless of direction (CE and PE are both bought-
premium positions, so a "stop" always means the premium ITSELF
falling and a "target" means it rising). Checked against each later
bar's own high/low, stop checked first if a single bar would touch
both (conservative); the actual fill is that deciding bar's own close,
not the literal stop/target level -- same decision-time-correct
convention as every other premium-SL/target module here. Whichever of
sl_points/target_points is left None on a leg that has the other set
just never triggers on that side; if neither is ever hit, the leg
still falls back to a forced-flat EOD exit. If the reference day D's
expiry has already passed by the trading day (D was itself the expiry
date), that contract is dead and no trade is taken (no data to trigger
on).

exit_mode="avg_reverse" is an alternative to the sl_points/target_points
exit above: instead of a premium-points stop/target, close the leg the
first time its OWN close falls back below avg_price -- the mirror of
the entry trigger, on the same reference level, applied identically to
both legs (CE exits below avg_price same as PE does; there's no
separate "vice versa" formula, the rule is symmetric by construction
since both legs only ever enter on a close ABOVE avg_price). No
stop-loss and no profit target in this mode -- sl_points/target_points
are ignored -- so a leg can only exit via this reversal or the
forced-flat EOD fallback if it never reverses.

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
    candle_minutes: int = 1,
    strike_search_range: int = 2,
    sl_points: float | None = 13.0,
    target_points: float | None = 26.0,
    exit_mode: str = "sl_target",
    access_token: str | None = None,
) -> list[OptionTrade]:
    if exit_mode not in ("sl_target", "avg_reverse"):
        raise ValueError('exit_mode must be "sl_target" or "avg_reverse"')
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
            already_traded_today = False
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
                if position is not None:
                    if exit_mode == "avg_reverse":
                        if c < avg_price:
                            trades.append(OptionTrade(
                                date=d, direction=direction, expiry=ref["expiry"], strike=ref["strike"],
                                entry_time=position["entry_time"], entry_premium=position["entry_price"],
                                exit_time=ts, exit_premium=c, lot_size=contract["lot_size"], exit_reason="avg_reverse",
                            ))
                            position = None
                    else:
                        hit_stop = sl_points is not None and l <= position["entry_price"] - sl_points
                        hit_target = target_points is not None and h >= position["entry_price"] + target_points
                        if hit_stop:
                            trades.append(OptionTrade(
                                date=d, direction=direction, expiry=ref["expiry"], strike=ref["strike"],
                                entry_time=position["entry_time"], entry_premium=position["entry_price"],
                                exit_time=ts, exit_premium=c, lot_size=contract["lot_size"], exit_reason="stop_loss",
                            ))
                            position = None
                        elif hit_target:
                            trades.append(OptionTrade(
                                date=d, direction=direction, expiry=ref["expiry"], strike=ref["strike"],
                                entry_time=position["entry_time"], entry_premium=position["entry_price"],
                                exit_time=ts, exit_premium=c, lot_size=contract["lot_size"], exit_reason="target",
                            ))
                            position = None
                if position is None and not already_traded_today and c > avg_price:
                    position = {"entry_time": ts, "entry_price": c}
                    already_traded_today = True
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
