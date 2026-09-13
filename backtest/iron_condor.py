"""Daily intraday iron condor backtest on NIFTY 50 weekly options.

Every trading day: enter a 4-leg iron condor at ~09:15 using whichever
weekly expiry is nearest (>= that day), square off all four legs at the
day's last traded price (~15:29/market close). Strikes are chosen relative
to the day's 09:15 spot price.

    short call = ATM + short_distance          (sell)
    short put  = ATM - short_distance          (sell)
    long call  = short call + wing_width       (buy, protection)
    long put   = short put  - wing_width       (buy, protection)

Requires an Upstox access token (expired-instruments API) since every
expiry involved is, by the time this runs, in the past.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from data_sources import upstox_client, cache
from backtest import costs, options_common as oc

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"

# entry side, exit side for each leg role
_LEG_SIDES = {
    "short_call": ("SELL", "BUY"),
    "short_put": ("SELL", "BUY"),
    "long_call": ("BUY", "SELL"),
    "long_put": ("BUY", "SELL"),
}


@dataclass
class Leg:
    role: str  # short_call | short_put | long_call | long_put
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
    pnl_points: float | None = None
    pnl_rupees_gross: float | None = None
    costs_rupees: float | None = None
    pnl_rupees: float | None = None  # net of costs
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.pnl_points is not None


def run(
    from_date: str,
    to_date: str,
    short_distance: int = 150,
    wing_width: int = 100,
    strike_step: int = 50,
    short_distance_strikes: int | None = None,
    wing_width_strikes: int | None = None,
    entry_time: str = "09:15",
    underlying_key: str = UNDERLYING_KEY,
    max_dte: int | None = None,
    access_token: str | None = None,
) -> list[DayResult]:
    """short_distance/wing_width are in points. If short_distance_strikes /
    wing_width_strikes are given instead, the actual point distance is
    computed per-expiry from that chain's own detected strike spacing
    (needed for equities, whose strike steps vary widely by price).

    max_dte: if set, only enter on days within this many calendar days of
    the nearest expiry (skip the rest) -- useful for monthly-expiry
    underlyings (stocks) where "nearest expiry" is often weeks out and
    there's little theta to harvest most days.
    """
    trading_days = upstox_client.get_daily_history(underlying_key, from_date, to_date)
    expiries = sorted(
        cache.get_expired_expiries_cached(underlying_key, "options", access_token)
    )

    chain_cache: dict[str, dict] = {}
    results: list[DayResult] = []

    for day in trading_days:
        d = day["date"]
        expiry = next((e for e in expiries if e >= d), None)
        if expiry is None:
            results.append(DayResult(date=d, expiry="", spot_915=0, atm=0, note="no expiry found"))
            continue
        if max_dte is not None:
            import datetime as _dt
            dte = (_dt.date.fromisoformat(expiry) - _dt.date.fromisoformat(d)).days
            if dte > max_dte:
                results.append(DayResult(date=d, expiry=expiry, spot_915=0, atm=0, note=f"dte={dte} > max_dte"))
                continue

        spot_candles = cache.get_day_candles_cached(
            underlying_key, "1minute", d, expired=False
        )
        entry_bar = oc.nearest_bar(spot_candles, "first", entry_time)
        if entry_bar is None:
            results.append(DayResult(date=d, expiry=expiry, spot_915=0, atm=0, note="no spot data"))
            continue
        spot_915 = entry_bar[1]

        if expiry not in chain_cache:
            chain_cache[expiry] = oc.build_chain_lookup(
                cache.get_expired_option_chain_cached(underlying_key, expiry, access_token)
            )
        lookup = chain_cache[expiry]
        if not lookup:
            results.append(DayResult(date=d, expiry=expiry, spot_915=spot_915, atm=0, note="empty chain"))
            continue

        if short_distance_strikes is not None:
            step = oc.detect_strike_step(lookup, spot_915)
            atm = oc.round_to_step(spot_915, step)
            eff_short_distance = short_distance_strikes * step
            eff_wing_width = wing_width_strikes * step
        else:
            atm = oc.round_to_step(spot_915, strike_step)
            eff_short_distance = short_distance
            eff_wing_width = wing_width

        wanted = [
            ("short_call", atm + eff_short_distance, "CE"),
            ("short_put", atm - eff_short_distance, "PE"),
            ("long_call", atm + eff_short_distance + eff_wing_width, "CE"),
            ("long_put", atm - eff_short_distance - eff_wing_width, "PE"),
        ]

        legs: list[Leg] = []
        missing = False
        for role, strike, opt_type in wanted:
            contract = oc.nearest_contract(lookup, strike, opt_type)
            if contract is None:
                missing = True
                break
            legs.append(
                Leg(
                    role=role,
                    strike=contract["strike_price"],
                    trading_symbol=contract["trading_symbol"],
                    instrument_key=contract["instrument_key"],
                    lot_size=contract["lot_size"],
                )
            )
        if missing:
            results.append(DayResult(date=d, expiry=expiry, spot_915=spot_915, atm=atm, note="strike not found in chain"))
            continue

        day_result = DayResult(date=d, expiry=expiry, spot_915=spot_915, atm=atm, legs=legs)
        incomplete = False
        for leg in legs:
            candles = cache.get_day_candles_cached(
                leg.instrument_key, "1minute", d, expired=True, access_token=access_token
            )
            entry = oc.nearest_bar(candles, "first", entry_time)
            exit_ = oc.nearest_bar(candles, "last")
            if entry is None or exit_ is None:
                incomplete = True
                continue
            leg.entry_price = entry[1]
            leg.exit_price = exit_[1]

        if incomplete or any(leg.entry_price is None or leg.exit_price is None for leg in legs):
            day_result.note = "missing leg candle data"
            results.append(day_result)
            continue

        by_role = {leg.role: leg for leg in legs}
        entry_credit = (
            by_role["short_call"].entry_price
            + by_role["short_put"].entry_price
            - by_role["long_call"].entry_price
            - by_role["long_put"].entry_price
        )
        exit_debit = (
            by_role["short_call"].exit_price
            + by_role["short_put"].exit_price
            - by_role["long_call"].exit_price
            - by_role["long_put"].exit_price
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

        day_result.pnl_points = pnl_points
        day_result.pnl_rupees_gross = pnl_gross
        day_result.costs_rupees = day_costs
        day_result.pnl_rupees = pnl_gross - day_costs
        results.append(day_result)

    return results


def summary(results: list[DayResult]) -> str:
    return "\n".join(oc.summary_lines(results))
