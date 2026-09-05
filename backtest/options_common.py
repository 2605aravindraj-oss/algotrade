"""Shared machinery for the multi-leg options daily backtests (iron condor,
call butterfly, ...): market-close-safe candle lookup, option chain
lookup/resolution, and P&L aggregation across an equity-weighted set of
DayResult-like records.
"""
from __future__ import annotations

MARKET_CLOSE = "15:30"


def round_to_step(value: float, step: int) -> int:
    return int(round(value / step) * step)


def nearest_bar(candles: list[list], pick: str) -> tuple[str, float] | None:
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


def build_chain_lookup(chain: list[dict]) -> dict[tuple[float, str], dict]:
    return {(c["strike_price"], c["instrument_type"]): c for c in chain}


def nearest_contract(lookup: dict, strike: float, opt_type: str) -> dict | None:
    if (strike, opt_type) in lookup:
        return lookup[(strike, opt_type)]
    candidates = [k for k in lookup if k[1] == opt_type]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda k: abs(k[0] - strike))
    return lookup[nearest]


def detect_strike_step(lookup: dict, atm_guess: float) -> int:
    """Infer the strike spacing actually used near the money for this chain
    (equity option strike steps vary a lot by stock price -- unlike NIFTY's
    flat 50, RELIANCE steps by 20, a Rs 300 stock might step by 5, etc.)."""
    strikes = sorted(set(k[0] for k in lookup))
    if len(strikes) < 2:
        return 50
    nearby = sorted(strikes, key=lambda s: abs(s - atm_guess))[:6]
    diffs = sorted(b - a for a, b in zip(sorted(nearby)[:-1], sorted(nearby)[1:]) if b > a)
    return max(1, int(diffs[0])) if diffs else 50


def summary_lines(results: list) -> list[str]:
    """results: DayResult-like objects with .ok, .pnl_rupees, .pnl_rupees_gross,
    .costs_rupees. Shared by every daily options strategy's summary()."""
    ok = [r for r in results if r.ok]
    if not ok:
        return ["No complete trading days."]
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

    return [
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
