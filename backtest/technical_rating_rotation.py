"""Monthly rotation backtest: at each rebalance date, screen the NIFTY 50
for stocks with a Strong Buy Technical Rating (backtest/technical_rating.py)
using only price data available up to that date (no lookahead), equal-weight
the picks, hold to the next rebalance date, then fully liquidate and
rebalance into whatever screens Strong Buy next.

If nothing screens Strong Buy on a given rebalance date, the portfolio sits
in cash for that period (0% return, no trading cost) rather than forcing a
pick -- this tests the rule literally as described, not a "best available"
fallback.

Costs: equity delivery, zero-brokerage discount-broker assumption --
STT 0.1% both sides, stamp duty 0.015% buy side, small exchange/SEBI/GST,
plus a flat ~Rs 20/scrip DP charge on sell (a real, non-negligible retail
cost that's easy to forget). Round-trip works out to roughly 0.25-0.35%
of position value depending on position size.

Simplification: every rebalance fully liquidates and rebuys, even for a
stock held in both consecutive periods -- slightly overstates costs
(a real rebalance would only trade the difference) but keeps the
accounting simple and doesn't affect the *signal's* validity.
"""
from __future__ import annotations

import bisect
import datetime
from dataclasses import dataclass, field

from backtest.technical_rating import rate
from data_sources import instruments, upstox_client

NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE",
    "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NESTLEIND",
    "WIPRO", "ONGC", "NTPC", "POWERGRID", "M&M", "TATASTEEL", "TATAMOTORS",
    "ADANIENT", "ADANIPORTS", "JSWSTEEL", "COALINDIA", "TECHM", "HCLTECH",
    "BAJAJFINSV", "DRREDDY", "GRASIM", "CIPLA", "EICHERMOT", "BRITANNIA",
    "DIVISLAB", "BPCL", "HEROMOTOCO", "HINDALCO", "INDUSINDBK", "SBILIFE",
    "HDFCLIFE", "APOLLOHOSP", "BAJAJ-AUTO", "TATACONSUM", "UPL",
    "SHRIRAMFIN", "LTIM",
]

STT_RATE = 0.001         # both sides, delivery
STAMP_DUTY_BUY_RATE = 0.00015
EXCHANGE_TXN_RATE = 0.0000297
SEBI_FEE_RATE = 0.000001
GST_RATE = 0.18
DP_CHARGE_PER_SCRIP = 20.0  # flat, sell side only


def _equity_buy_cost(value: float) -> float:
    exch = value * EXCHANGE_TXN_RATE
    sebi = value * SEBI_FEE_RATE
    stamp = value * STAMP_DUTY_BUY_RATE
    gst = GST_RATE * (exch + sebi)
    return exch + sebi + stamp + gst


def _equity_sell_cost(value: float, num_scrips: int) -> float:
    exch = value * EXCHANGE_TXN_RATE
    sebi = value * SEBI_FEE_RATE
    stt = value * STT_RATE
    gst = GST_RATE * (exch + sebi)
    return exch + sebi + stt + gst + DP_CHARGE_PER_SCRIP * num_scrips


@dataclass
class Period:
    start_date: str
    end_date: str
    picks: list[str]
    start_value: float
    end_value: float
    costs: float

    @property
    def return_pct(self) -> float:
        return 0.0 if self.start_value == 0 else 100 * (self.end_value - self.start_value) / self.start_value


@dataclass
class RotationResult:
    periods: list[Period] = field(default_factory=list)
    final_value: float = 0.0
    starting_capital: float = 0.0


def _fetch_histories(symbols: list[str], from_date: str, to_date: str) -> dict[str, list[dict]]:
    keys = instruments.resolve_symbols(symbols)
    out: dict[str, list[dict]] = {}
    for sym in symbols:
        key = keys.get(sym)
        if key is None:
            continue
        try:
            daily = upstox_client.get_daily_history(key, from_date, to_date)
        except Exception:
            continue
        if daily:
            out[sym] = sorted(daily, key=lambda c: c["date"])
    return out


def _rebalance_dates(trading_dates: list[str], start_date: str, end_date: str, rebalance_days: int) -> list[str]:
    """Snap a fixed calendar schedule (every `rebalance_days` from
    start_date) onto the nearest trading day at-or-after each target, plus
    a final close-out date (the last trading day at-or-before end_date)."""
    dates: list[str] = []
    cursor = datetime.date.fromisoformat(start_date)
    while True:
        target = cursor.isoformat()
        if target > end_date:
            break
        idx = bisect.bisect_left(trading_dates, target)
        if idx >= len(trading_dates) or trading_dates[idx] > end_date:
            break
        dates.append(trading_dates[idx])
        cursor += datetime.timedelta(days=rebalance_days)

    last_idx = bisect.bisect_right(trading_dates, end_date) - 1
    if last_idx >= 0:
        last_date = trading_dates[last_idx]
        if not dates or dates[-1] != last_date:
            dates.append(last_date)

    seen = set()
    ordered = []
    for d in dates:
        if d not in seen:
            seen.add(d)
            ordered.append(d)
    return ordered


def run(
    start_date: str,
    end_date: str,
    rebalance_days: int = 30,
    starting_capital: float = 1_000_000.0,
    min_label: str = "Strong Buy",
    symbols: list[str] | None = None,
) -> RotationResult:
    symbols = symbols or NIFTY_50
    warmup_from = (datetime.date.fromisoformat(start_date) - datetime.timedelta(days=320)).isoformat()

    histories = _fetch_histories(symbols, warmup_from, end_date)
    calendar_symbol = "RELIANCE" if "RELIANCE" in histories else next(iter(histories))
    all_dates = sorted({c["date"] for c in histories[calendar_symbol]})
    trading_dates = [d for d in all_dates if d >= start_date]
    if not trading_dates:
        return RotationResult(starting_capital=starting_capital, final_value=starting_capital)

    rb_dates = _rebalance_dates(trading_dates, start_date, end_date, rebalance_days)
    if len(rb_dates) < 2:
        return RotationResult(starting_capital=starting_capital, final_value=starting_capital)

    date_index: dict[str, dict[str, int]] = {}  # symbol -> {date: index in its candle list}
    for sym, candles in histories.items():
        date_index[sym] = {c["date"]: i for i, c in enumerate(candles)}

    def _close_on(sym: str, date: str) -> float | None:
        idx = date_index.get(sym, {}).get(date)
        if idx is None:
            return None
        return histories[sym][idx]["close"]

    def _screen(as_of_date: str) -> list[str]:
        picks = []
        for sym, candles in histories.items():
            idx = date_index[sym].get(as_of_date)
            if idx is None:
                continue
            slice_ = candles[:idx + 1]
            r = rate(sym, slice_)
            if r is not None and r.label == min_label:
                picks.append(sym)
        return picks

    result = RotationResult(starting_capital=starting_capital)
    portfolio_value = starting_capital

    for i in range(len(rb_dates) - 1):
        d0, d1 = rb_dates[i], rb_dates[i + 1]
        picks = _screen(d0)
        start_value = portfolio_value

        if not picks:
            result.periods.append(Period(d0, d1, [], start_value, start_value, 0.0))
            continue

        per_stock_capital = start_value / len(picks)
        buy_costs = 0.0
        shares: dict[str, float] = {}
        for sym in picks:
            price = _close_on(sym, d0)
            if price is None or price <= 0:
                continue
            qty = per_stock_capital / price
            shares[sym] = qty
            buy_costs += _equity_buy_cost(qty * price)

        sell_value = 0.0
        sell_costs = 0.0
        for sym, qty in shares.items():
            price = _close_on(sym, d1)
            if price is None:
                # stock delisted/no data mid-period -- treat as flat (no gain/loss) rather than a crash
                price = _close_on(sym, d0)
            sell_value += qty * price
            sell_costs += _equity_sell_cost(qty * price, 1)

        cash_uninvested = start_value - sum(shares[s] * _close_on(s, d0) for s in shares)
        end_value = sell_value + cash_uninvested - buy_costs - sell_costs
        result.periods.append(Period(d0, d1, list(shares.keys()), start_value, end_value, buy_costs + sell_costs))
        portfolio_value = end_value

    result.final_value = portfolio_value
    return result


def summary(result: RotationResult) -> str:
    if not result.periods:
        return "No periods."
    lines = []
    for p in result.periods:
        picks_str = ", ".join(p.picks) if p.picks else "(cash)"
        lines.append(
            f"{p.start_date} -> {p.end_date}: {p.return_pct:+.2f}%  "
            f"[{picks_str}]  costs=Rs {p.costs:,.0f}"
        )
    total_return = 100 * (result.final_value - result.starting_capital) / result.starting_capital
    n_years = len(result.periods) * 30 / 365.25
    cagr = (
        ((result.final_value / result.starting_capital) ** (1 / n_years) - 1) * 100
        if n_years > 0 and result.final_value > 0 else float("nan")
    )
    cash_periods = sum(1 for p in result.periods if not p.picks)
    lines.append("")
    lines.append(f"Periods: {len(result.periods)}  (in cash: {cash_periods})")
    lines.append(f"Starting capital: Rs {result.starting_capital:,.2f}")
    lines.append(f"Final value:      Rs {result.final_value:,.2f}")
    lines.append(f"Total return:     {total_return:+.2f}%")
    lines.append(f"Approx CAGR:      {cagr:+.2f}%")
    return "\n".join(lines)
