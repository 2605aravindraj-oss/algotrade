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

import time
from dataclasses import dataclass, field

from data_sources import upstox_client
from backtest import costs

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"
LEG_SLEEP_SECONDS = 0.2

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


def _round_to_step(value: float, step: int) -> int:
    return int(round(value / step) * step)


MARKET_CLOSE = "15:30"


def _nearest_bar(candles: list[list], prefix: str, pick: str) -> tuple[str, float] | None:
    """candles: newest-first rows [timestamp, o, h, l, c, vol, oi].
    pick="first": earliest bar at/after 09:15. pick="last": latest bar at/before
    market close (15:30) -- the raw feed can include post-close settlement
    ticks past 15:30 pinned at the min tick, so we must not just take the
    literal last bar in the response.
    """
    if not candles:
        return None
    rows = sorted(candles, key=lambda c: c[0])
    if pick == "last":
        at_or_before_close = [row for row in rows if row[0][11:16] <= MARKET_CLOSE]
        if not at_or_before_close:
            return None
        row = at_or_before_close[-1]
        return row[0], row[4]
    for row in rows:
        if row[0][11:16] >= "09:15":
            return row[0], row[1]
    return rows[0][0], rows[0][1]


def _build_chain_lookup(chain: list[dict]) -> dict[tuple[float, str], dict]:
    return {(c["strike_price"], c["instrument_type"]): c for c in chain}


def _nearest_contract(lookup: dict, strike: float, opt_type: str) -> dict | None:
    if (strike, opt_type) in lookup:
        return lookup[(strike, opt_type)]
    candidates = [k for k in lookup if k[1] == opt_type]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda k: abs(k[0] - strike))
    return lookup[nearest]


def _detect_strike_step(lookup: dict, atm_guess: float) -> int:
    """Infer the strike spacing actually used near the money for this chain
    (equity option strike steps vary a lot by stock price -- unlike NIFTY's
    flat 50, RELIANCE steps by 20, a Rs 300 stock might step by 5, etc.)."""
    strikes = sorted(set(k[0] for k in lookup))
    if len(strikes) < 2:
        return 50
    nearby = sorted(strikes, key=lambda s: abs(s - atm_guess))[:6]
    diffs = sorted(b - a for a, b in zip(sorted(nearby)[:-1], sorted(nearby)[1:]) if b > a)
    return max(1, int(diffs[0])) if diffs else 50


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
    access_token: str | None = None,
) -> list[DayResult]:
    """short_distance/wing_width are in points. If short_distance_strikes /
    wing_width_strikes are given instead, the actual point distance is
    computed per-expiry from that chain's own detected strike spacing
    (needed for equities, whose strike steps vary widely by price)."""
    trading_days = upstox_client.get_daily_history(underlying_key, from_date, to_date)
    expiries = sorted(
        upstox_client.get_expired_expiries(underlying_key, "options", access_token)
    )

    chain_cache: dict[str, dict] = {}
    results: list[DayResult] = []

    for day in trading_days:
        d = day["date"]
        expiry = next((e for e in expiries if e >= d), None)
        if expiry is None:
            results.append(DayResult(date=d, expiry="", spot_915=0, atm=0, note="no expiry found"))
            continue

        spot_candles = upstox_client.get_historical_candles(
            underlying_key, interval="1minute", to_date=d, from_date=d
        )["data"]["candles"]
        entry_bar = _nearest_bar(spot_candles, entry_time, "first")
        if entry_bar is None:
            results.append(DayResult(date=d, expiry=expiry, spot_915=0, atm=0, note="no spot data"))
            continue
        spot_915 = entry_bar[1]

        if expiry not in chain_cache:
            chain_cache[expiry] = _build_chain_lookup(
                upstox_client.get_expired_option_chain(underlying_key, expiry, access_token)
            )
        lookup = chain_cache[expiry]
        if not lookup:
            results.append(DayResult(date=d, expiry=expiry, spot_915=spot_915, atm=0, note="empty chain"))
            continue

        if short_distance_strikes is not None:
            step = _detect_strike_step(lookup, spot_915)
            atm = _round_to_step(spot_915, step)
            eff_short_distance = short_distance_strikes * step
            eff_wing_width = wing_width_strikes * step
        else:
            atm = _round_to_step(spot_915, strike_step)
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
            contract = _nearest_contract(lookup, strike, opt_type)
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
            candles = upstox_client.get_expired_candles(
                leg.instrument_key, "1minute", d, d, access_token
            )
            time.sleep(LEG_SLEEP_SECONDS)
            entry = _nearest_bar(candles, entry_time, "first")
            exit_ = _nearest_bar(candles, entry_time, "last")
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
    ok = [r for r in results if r.ok]
    if not ok:
        return "No complete trading days."
    pnls = [r.pnl_rupees for r in ok]
    total_costs = sum(r.costs_rupees for r in ok)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    cum = 0.0
    equity = []
    for p in pnls:
        cum += p
        equity.append(cum)
    running_max = equity[0]
    max_dd = 0.0
    for e in equity:
        running_max = max(running_max, e)
        max_dd = min(max_dd, e - running_max)

    lines = [
        f"Trading days:     {len(results)} ({len(ok)} complete, {len(results) - len(ok)} skipped)",
        f"Gross P&L:        Rs {sum(r.pnl_rupees_gross for r in ok):,.2f}",
        f"Costs (STT/GST/etc): Rs {total_costs:,.2f}  (Rs {total_costs / len(ok):,.2f}/day)",
        f"Net P&L:          Rs {sum(pnls):,.2f}",
        f"Win rate:         {len(wins) / len(ok) * 100:.1f}%  ({len(wins)}W / {len(losses)}L)",
        f"Avg win:          Rs {(sum(wins) / len(wins)) if wins else 0:,.2f}",
        f"Avg loss:         Rs {(sum(losses) / len(losses)) if losses else 0:,.2f}",
        f"Best day:         Rs {max(pnls):,.2f}",
        f"Worst day:        Rs {min(pnls):,.2f}",
        f"Max drawdown:     Rs {max_dd:,.2f}",
    ]
    return "\n".join(lines)
