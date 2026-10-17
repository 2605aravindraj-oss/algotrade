"""Same NIFTY weekly iron condor as backtest.iron_condor.py (short
strangle + protective wings, entered ~09:15 on the nearest weekly
expiry), but exited intraday on a profit target or stop-loss instead
of always holding to the close.

WHY: iron_condor.py's own validated config (short_distance=200,
wing_width=200, hold-to-close) nets Rs 58,096/480 days at zero
slippage but FLIPS NEGATIVE at just 1% per-leg slippage (-Rs 40,026),
because every day pays for all 8 fills (4 legs x entry+exit)
regardless of how the day actually played out -- a day that's already
captured most of its credit by midday still waits for the close,
exposed to afternoon chop that can give part of it back for no
reason. Managing the combined position intraday (exit once most of
the premium is captured, or cut it once losses exceed a multiple of
the credit) is standard professional option-selling discipline, not
a new indicator -- this module tests whether it actually helps here,
not just asserts it.

MECHANISM: identical entry to iron_condor.py (4 legs: sell ATM+/-
short_distance, buy further out by wing_width for defined risk).
From entry_time onward, every 1-minute bar where ALL FOUR legs have a
candle is used to mark the combined position: running_debit = what it
would cost to close all four legs right now (short legs bought back at
their current price, long legs sold at their current price). Compare
against entry_credit (what was collected at 09:15):

    captured_pct = (entry_credit - running_debit) / entry_credit

Exits the whole position (all 4 legs, at that bar's prices) the first
time:
    captured_pct >= profit_target_pct   (take the win, stop watching
                                          it bleed back in afternoon
                                          chop), or
    captured_pct <= -stop_loss_mult     (stop_loss_mult as a multiple
                                          of entry_credit, e.g. 1.5
                                          means "cut it once the loss
                                          is 1.5x what was collected")
Neither hit -> same as before, square off at the day's last candle.

This is a MARK-TO-MARKET exit on the combined 4-leg position's own
premium, not an index-level stop -- same convention as every other
sl_rs-style module in this codebase, just applied to a spread instead
of a single leg.

CAPITAL SIZING FOR A GIVEN BUDGET: iron_condor_lots_for_capital()
estimates how many lots a given capital figure can carry. The TRUE
defined-risk cap per lot is (wing_width - net_credit) * lot_size --
that's the worst case this structure can ever lose, by construction
(NSE collapses the margin requirement for a recognized spread close to
that number). This module doesn't have a live margin API, so it uses
a configurable buffer_multiple (default 1.3) over that theoretical cap
as a conservative stand-in for the real SPAN+exposure margin a broker
will actually block -- confirm the real number on your broker's margin
calculator before sizing a live position; this is an estimate, not a
quote.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from data_sources import upstox_client, cache
from backtest import costs, options_common as oc

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"

_LEG_SIDES = {
    "short_call": ("SELL", "BUY"),
    "short_put": ("SELL", "BUY"),
    "long_call": ("BUY", "SELL"),
    "long_put": ("BUY", "SELL"),
}


def _apply_slippage(price: float, side: str, slippage_pct: float) -> float:
    if slippage_pct <= 0:
        return price
    return price * (1 + slippage_pct) if side == "BUY" else price * (1 - slippage_pct)


@dataclass
class Leg:
    role: str
    strike: float
    trading_symbol: str
    instrument_key: str
    lot_size: int
    entry_price: float | None = None
    exit_price: float | None = None


@dataclass
class DayResult:
    date: str
    expiry: str
    spot_915: float
    atm: float
    legs: list[Leg] = field(default_factory=list)
    entry_credit: float | None = None
    exit_reason: str = ""
    exit_time: str = ""
    pnl_points: float | None = None
    pnl_rupees_gross: float | None = None
    costs_rupees: float | None = None
    pnl_rupees: float | None = None
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.pnl_points is not None


def run(
    from_date: str,
    to_date: str,
    short_distance: int = 200,
    wing_width: int = 200,
    strike_step: int = 50,
    entry_time: str = "09:15",
    profit_target_pct: float = 0.6,
    stop_loss_mult: float = 1.5,
    underlying_key: str = UNDERLYING_KEY,
    slippage_pct: float = 0.0,
    access_token: str | None = None,
) -> list[DayResult]:
    """profit_target_pct (default 0.6): exit once 60% of entry_credit is
    captured. stop_loss_mult (default 1.5): exit once the running loss
    reaches 1.5x entry_credit. Both are checked on every 1-minute bar
    where all 4 legs have a candle; neither hit -> hold to the close,
    same as iron_condor.py. slippage_pct: same adverse per-leg execution
    model as iron_condor.py, applied to entry AND to whichever exit
    (target/stop/close) actually fires."""
    trading_days = upstox_client.get_daily_history(underlying_key, from_date, to_date)
    expiries = sorted(cache.get_expired_expiries_cached(underlying_key, "options", access_token))

    chain_cache: dict[str, dict] = {}
    results: list[DayResult] = []

    for day in trading_days:
        d = day["date"]
        expiry = next((e for e in expiries if e >= d), None)
        if expiry is None:
            results.append(DayResult(date=d, expiry="", spot_915=0, atm=0, note="no expiry found"))
            continue

        spot_rows = cache.get_day_candles_cached(underlying_key, "1minute", d, expired=False)
        spot_entry = oc.nearest_bar(spot_rows, "first", entry_time)
        if spot_entry is None:
            results.append(DayResult(date=d, expiry=expiry, spot_915=0, atm=0, note="no spot candle"))
            continue
        spot_915 = spot_entry[1]

        if expiry not in chain_cache:
            chain_cache[expiry] = oc.build_chain_lookup(
                cache.get_expired_option_chain_cached(underlying_key, expiry, access_token)
            )
        lookup = chain_cache[expiry]
        if not lookup:
            results.append(DayResult(date=d, expiry=expiry, spot_915=spot_915, atm=0, note="empty chain"))
            continue

        atm = oc.round_to_step(spot_915, strike_step)
        wanted = [
            ("short_call", atm + short_distance, "CE"),
            ("short_put", atm - short_distance, "PE"),
            ("long_call", atm + short_distance + wing_width, "CE"),
            ("long_put", atm - short_distance - wing_width, "PE"),
        ]

        legs: list[Leg] = []
        missing = False
        for role, strike, opt_type in wanted:
            contract = oc.nearest_contract(lookup, strike, opt_type)
            if contract is None:
                missing = True
                break
            legs.append(Leg(
                role=role, strike=contract["strike_price"], trading_symbol=contract["trading_symbol"],
                instrument_key=contract["instrument_key"], lot_size=contract["lot_size"],
            ))
        if missing:
            results.append(DayResult(date=d, expiry=expiry, spot_915=spot_915, atm=atm, note="strike not found in chain"))
            continue

        day_result = DayResult(date=d, expiry=expiry, spot_915=spot_915, atm=atm, legs=legs)

        leg_candles: dict[str, list[list]] = {}
        incomplete = False
        for leg in legs:
            candles = cache.get_day_candles_cached(leg.instrument_key, "1minute", d, expired=True, access_token=access_token)
            if not candles:
                incomplete = True
                continue
            leg_candles[leg.role] = sorted(candles, key=lambda c: c[0])
            entry = oc.nearest_bar(candles, "first", entry_time)
            if entry is None:
                incomplete = True
                continue
            entry_side, _ = _LEG_SIDES[leg.role]
            leg.entry_price = _apply_slippage(entry[1], entry_side, slippage_pct)

        if incomplete or any(leg.entry_price is None for leg in legs):
            day_result.note = "missing leg candle data"
            results.append(day_result)
            continue

        by_role = {leg.role: leg for leg in legs}
        entry_credit = (
            by_role["short_call"].entry_price + by_role["short_put"].entry_price
            - by_role["long_call"].entry_price - by_role["long_put"].entry_price
        )
        day_result.entry_credit = entry_credit

        # bar-by-bar mark-to-market on the combined position, using each
        # leg's own candle sequence after entry_time (intersect timestamps
        # so every leg has a price at the bar being evaluated)
        leg_bars_after_entry = {
            role: [row for row in rows if row[0][11:16] >= entry_time]
            for role, rows in leg_candles.items()
        }
        common_ts = sorted(set.intersection(*(set(r[0] for r in rows) for rows in leg_bars_after_entry.values())))

        exit_reason = "close"
        exit_ts = None
        exit_prices: dict[str, float] | None = None
        if entry_credit > 0:
            by_ts = {role: {r[0]: r[4] for r in rows} for role, rows in leg_bars_after_entry.items()}
            for ts in common_ts:
                running_debit = (
                    by_ts["short_call"][ts] + by_ts["short_put"][ts]
                    - by_ts["long_call"][ts] - by_ts["long_put"][ts]
                )
                captured_pct = (entry_credit - running_debit) / entry_credit
                if captured_pct >= profit_target_pct:
                    exit_reason, exit_ts = "target", ts
                    exit_prices = {role: by_ts[role][ts] for role in by_ts}
                    break
                if captured_pct <= -stop_loss_mult:
                    exit_reason, exit_ts = "stop_loss", ts
                    exit_prices = {role: by_ts[role][ts] for role in by_ts}
                    break

        if exit_prices is None:
            exit_candles_ok = True
            exit_prices = {}
            for leg in legs:
                exit_bar = oc.nearest_bar(leg_candles[leg.role], "last")
                if exit_bar is None:
                    exit_candles_ok = False
                    break
                exit_ts = exit_bar[0]
                exit_prices[leg.role] = exit_bar[1]
            if not exit_candles_ok:
                day_result.note = "missing exit candle data"
                results.append(day_result)
                continue

        for leg in legs:
            _, exit_side = _LEG_SIDES[leg.role]
            leg.exit_price = _apply_slippage(exit_prices[leg.role], exit_side, slippage_pct)

        exit_debit = (
            by_role["short_call"].exit_price + by_role["short_put"].exit_price
            - by_role["long_call"].exit_price - by_role["long_put"].exit_price
        )
        pnl_points = entry_credit - exit_debit
        lot_size = legs[0].lot_size
        pnl_gross = pnl_points * lot_size

        fills = []
        for leg in legs:
            entry_side, exit_side = _LEG_SIDES[leg.role]
            fills.append(costs.Fill(price=leg.entry_price, lot_size=leg.lot_size, side=entry_side))
            fills.append(costs.Fill(price=leg.exit_price, lot_size=leg.lot_size, side=exit_side))
        day_costs = costs.total_cost(fills)

        day_result.exit_reason = exit_reason
        day_result.exit_time = exit_ts or ""
        day_result.pnl_points = pnl_points
        day_result.pnl_rupees_gross = pnl_gross
        day_result.costs_rupees = day_costs
        day_result.pnl_rupees = pnl_gross - day_costs
        results.append(day_result)

    return results


def summary(results: list[DayResult]) -> str:
    lines = oc.summary_lines(results)
    ok = [r for r in results if r.ok]
    if ok:
        from collections import Counter
        reasons = Counter(r.exit_reason for r in ok)
        lines.append("")
        lines.append(f"Exit reasons: {dict(reasons)}")
    return "\n".join(lines)


def lots_for_capital(
    capital: float,
    wing_width: int = 200,
    lot_size: int = 75,
    avg_net_credit: float = 0.0,
    buffer_multiple: float = 1.3,
) -> dict:
    """Rough capital sizing for a given iron condor structure. The true
    defined-risk cap per lot is (wing_width - avg_net_credit) * lot_size
    -- the worst case this structure can ever lose. No live margin API
    is wired up here, so buffer_multiple (default 1.3) is applied on top
    as a conservative stand-in for the real SPAN+exposure margin a broker
    will block, which runs somewhat above the bare defined-risk cap.
    CONFIRM THE REAL FIGURE ON YOUR BROKER'S MARGIN CALCULATOR before
    sizing a live position -- this is an estimate, not a quote.
    """
    max_loss_per_lot = max(wing_width - avg_net_credit, 1) * lot_size
    est_margin_per_lot = max_loss_per_lot * buffer_multiple
    lots = int(capital // est_margin_per_lot)
    return {
        "max_loss_per_lot_rupees": max_loss_per_lot,
        "estimated_margin_per_lot_rupees": est_margin_per_lot,
        "lots": lots,
        "capital_deployed_rupees": lots * est_margin_per_lot,
        "capital_at_risk_if_max_loss_rupees": lots * max_loss_per_lot,
    }
