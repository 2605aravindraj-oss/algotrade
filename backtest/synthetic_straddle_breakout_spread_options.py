"""Synthetic-ATM straddle breakout, realized as a BULL CALL SPREAD /
BEAR PUT SPREAD instead of naked option buying. Same Stage 1 reference
computation and Stage 2 entry SIGNAL as
synthetic_straddle_breakout_options.py (duplicated here rather than
imported, since that module's Stage 1 logic lives in closures inside
its own run()) -- only how the signal is REALIZED changes.

LONG signal (CE close > avg_price): buy the ATM CE (long leg, same
strike the naked version would use) AND SELL a further OTM CE
spread_width_strikes strikes above it (short leg) -- a bull call
spread. Mirror: SHORT signal (PE close > avg_price) buys the ATM PE
and sells a further OTM PE spread_width_strikes strikes below it -- a
bear put spread. spread_width_strikes default 2 (100 points at
NIFTY's 50-point strike step) -- an assumption, not derived from the
request; a wider spread costs less net debit but caps profit sooner.

Both legs fill at the SAME signal candle's close (decision-time-
correct, same convention as every other intraday module here). Net
entry cost = long leg's premium - short leg's premium (a debit --
always positive, since the ATM long leg is pricier than the further
OTM short leg in the same direction).

EXIT -- sl_points/target_points (defaults 13/26, carried over
UNCHANGED from the naked-buy version) apply to the SPREAD'S NET VALUE
(long premium - short premium at that moment), not either leg's raw
premium: stop when net value falls to entry_net_debit - sl_points or
below; target when it rises to entry_net_debit + target_points or
above. IMPORTANT CAVEAT: a spread's net value is inherently capped --
it can never exceed the strike width in points (100 at the default
spread_width_strikes=2) -- and moves much less per underlying point
than a naked long option's premium does, so these thresholds, tuned
for a naked option's much larger swings, are very likely mismatched
for a spread's compressed range. Carried over only as a starting
point for this module's own tuning, not a validated choice -- back-
test results here should be read with that in mind before trusting
them. Forced flat at FORCE_FLAT_TIME if neither hits first.

min_diff_points (default 10.0, same meaning/value as the naked
version) still gates whether a day trades at all. one_trade_per_day is
NOT carried over from the naked version in this first cut (its
merged-timeline logic would need to arbitrate between two 2-leg
positions rather than two single legs) -- both CE and PE signals can
still fire independently the same day, each at most once, same as the
naked version's default.
"""
from __future__ import annotations

from dataclasses import dataclass

from data_sources import cache, upstox_client
from backtest import costs, options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME
from backtest.sweep_reclaim_breakout import _resample

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"


@dataclass
class SpreadTrade:
    date: str
    direction: str  # LONG (bull call spread) | SHORT (bear put spread)
    expiry: str
    long_strike: float
    short_strike: float
    entry_time: str
    long_entry_premium: float
    short_entry_premium: float
    exit_time: str
    long_exit_premium: float
    short_exit_premium: float
    lot_size: int
    exit_reason: str

    @property
    def entry_net_debit(self) -> float:
        return self.long_entry_premium - self.short_entry_premium

    @property
    def exit_net_value(self) -> float:
        return self.long_exit_premium - self.short_exit_premium

    @property
    def pnl_points(self) -> float:
        return self.exit_net_value - self.entry_net_debit

    @property
    def pnl_rupees_gross(self) -> float:
        return self.pnl_points * self.lot_size

    @property
    def cost_rupees(self) -> float:
        fills = [
            costs.Fill(price=self.long_entry_premium, lot_size=self.lot_size, side="BUY"),
            costs.Fill(price=self.short_entry_premium, lot_size=self.lot_size, side="SELL"),
            costs.Fill(price=self.long_exit_premium, lot_size=self.lot_size, side="SELL"),
            costs.Fill(price=self.short_exit_premium, lot_size=self.lot_size, side="BUY"),
        ]
        return costs.total_cost(fills)

    @property
    def pnl_rupees(self) -> float:
        return self.pnl_rupees_gross - self.cost_rupees


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    candle_minutes: int = 1,
    strike_search_range: int = 2,
    spread_width_strikes: int = 2,
    sl_points: float | None = 13.0,
    target_points: float | None = 26.0,
    min_diff_points: float | None = 10.0,
    access_token: str | None = None,
) -> list[SpreadTrade]:
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

    # -- STAGE 1: same reference computation as synthetic_straddle_breakout_options.py --
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
        reference[d] = {"strike": best[1], "avg_price": best[2], "diff": best[0], "expiry": expiry} if best else None

    # -- STAGE 2: trade day D+1 off day D's reference, realized as a spread --
    # Long/short legs share the same candle_minutes bucketing, so their bars
    # line up by exact timestamp -- looked up directly, no _bar_at_or_after/
    # before fallback needed (unlike modules that fetch one leg's fill
    # separately from a differently-timed decision point).
    trades: list[SpreadTrade] = []
    for i in range(1, len(trading_days)):
        d = trading_days[i]["date"]
        ref = reference[trading_days[i - 1]["date"]]
        if ref is None or ref["expiry"] < d:
            continue
        if min_diff_points is not None and ref["diff"] < min_diff_points:
            continue
        lookup = _lookup(ref["expiry"])
        avg_price = ref["avg_price"]
        best_strike = ref["strike"]

        for opt_type, direction, short_strike in (
            ("CE", "LONG", best_strike + spread_width_strikes * strike_step),
            ("PE", "SHORT", best_strike - spread_width_strikes * strike_step),
        ):
            long_contract = lookup.get((best_strike, opt_type))
            short_contract = lookup.get((short_strike, opt_type))
            if long_contract is None or short_contract is None:
                continue
            long_bars = _resample(sorted(_day_candles(long_contract["instrument_key"], d), key=lambda c: c[0]), candle_minutes)
            short_bars = _resample(sorted(_day_candles(short_contract["instrument_key"], d), key=lambda c: c[0]), candle_minutes)
            if not long_bars or not short_bars:
                continue
            short_by_ts = {b[0]: b for b in short_bars}

            position = None  # dict: entry_time, long_entry, short_entry
            already_traded_today = False
            for row in long_bars:
                ts, o, h, l, c, v, oi = row
                time_str = ts[11:16]
                short_row = short_by_ts.get(ts)

                if time_str >= FORCE_FLAT_TIME:
                    if position is not None and short_row is not None:
                        trades.append(SpreadTrade(
                            date=d, direction=direction, expiry=ref["expiry"],
                            long_strike=best_strike, short_strike=short_strike,
                            entry_time=position["entry_time"],
                            long_entry_premium=position["long_entry"], short_entry_premium=position["short_entry"],
                            exit_time=ts, long_exit_premium=c, short_exit_premium=short_row[4],
                            lot_size=long_contract["lot_size"], exit_reason="eod",
                        ))
                        position = None
                    continue

                if position is not None and short_row is not None:
                    net_value = c - short_row[4]
                    hit_stop = sl_points is not None and net_value <= position["entry_net_debit"] - sl_points
                    hit_target = target_points is not None and net_value >= position["entry_net_debit"] + target_points
                    reason = "stop_loss" if hit_stop else ("target" if hit_target else None)
                    if reason is not None:
                        trades.append(SpreadTrade(
                            date=d, direction=direction, expiry=ref["expiry"],
                            long_strike=best_strike, short_strike=short_strike,
                            entry_time=position["entry_time"],
                            long_entry_premium=position["long_entry"], short_entry_premium=position["short_entry"],
                            exit_time=ts, long_exit_premium=c, short_exit_premium=short_row[4],
                            lot_size=long_contract["lot_size"], exit_reason=reason,
                        ))
                        position = None

                if position is None and not already_traded_today and short_row is not None and c > avg_price:
                    position = {
                        "entry_time": ts, "long_entry": c, "short_entry": short_row[4],
                        "entry_net_debit": c - short_row[4],
                    }
                    already_traded_today = True

    return trades


def summary(trades: list[SpreadTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
