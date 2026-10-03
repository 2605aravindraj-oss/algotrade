"""Volatility-targeted NIFTY 50 position-sizing overlay.

NOT a directional strategy -- there is no entry signal, no prediction
of up or down. The position is always long the index; the only
decision is HOW MUCH, scaled by how volatile the index has recently
been. This is a pure risk-sizing overlay, in the spirit of the
"volatility-managed portfolios" literature (e.g. Moreira & Muir):
delever into a volatility spike, lever up when things are calm.

SIZING RULE:
    leverage(t) = clip(target_annual_vol / realized_vol(t-1), 0, max_leverage)
    target_annual_vol = 0.15 (15%)
    realized_vol(t-1) = trailing 20-TRADING-day stdev of NIFTY's own
        daily returns, annualized (* sqrt(252)), computed using only
        returns available THROUGH day t-1 -- no lookahead. The
        leverage applied to day t's return is set using information
        known at the START of day t.
    max_leverage = 2.0, min = 0.0 (leverage is never negative --
        always long, never short, just sized up or down)

Daily strategy return(t) = leverage(t) * NIFTY's own daily return(t).
No transaction costs modeled (a real implementation would need to pay
for daily/weekly leverage adjustments, but the question this module
answers is "does the sizing rule itself carry genuine risk-adjusted
value", which is a pre-cost question).

VALIDATION METHOD: Sharpe ratio (mean daily return / stdev daily
return * sqrt(252), risk-free rate assumed 0) computed over two
independent, non-overlapping windows (split the full history in half
by trading-day count, giving two ~5-6 year windows depending on the
overall span). A vol-targeting overlay can look good on Sharpe for a
boring reason that has nothing to do with real skill: if it happens
to run at a LOWER AVERAGE leverage than 1.0, a lower-vol return stream
can mechanically show a "better" Sharpe even with zero genuine timing
value (Sharpe is scale-invariant in theory, but realized Sharpe on
finite data from a smaller, steadier-vol stream can still look better
by chance). To rule that out, each window ALSO runs a STATIC-LEVERAGE
CONTROL: the SAME constant leverage (that window's own average
REALIZED leverage from the dynamic strategy) applied with NO dynamic
adjustment at all, same window, same underlying index. If the dynamic
strategy's Sharpe beats the static control's Sharpe at an IDENTICAL
average exposure, the vol-timing itself -- not just running smaller on
average -- is adding risk-adjusted value.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from backtest.rsi_daily_reversion import _get_daily_history_chunked

NIFTY_INDEX_KEY = "NSE_INDEX|Nifty 50"
TRADING_DAYS_PER_YEAR = 252


@dataclass
class DailyRow:
    date: str
    nifty_return: float
    realized_vol_annualized: float | None
    leverage: float | None
    strategy_return: float | None


def _daily_returns(closes: list[float]) -> list[float | None]:
    out: list[float | None] = [None] * len(closes)
    for i in range(1, len(closes)):
        out[i] = closes[i] / closes[i - 1] - 1
    return out


def _trailing_realized_vol(returns: list[float | None], i: int, window: int = 20) -> float | None:
    """Annualized stdev of returns[i-window:i] (STRICTLY before index i --
    no lookahead into day i's own return)."""
    if i < window:
        return None
    sample = returns[i - window:i]
    if any(r is None for r in sample):
        return None
    return statistics.pstdev(sample) * math.sqrt(TRADING_DAYS_PER_YEAR)


def run(
    from_date: str,
    to_date: str,
    target_annual_vol: float = 0.15,
    vol_window: int = 20,
    max_leverage: float = 2.0,
    min_leverage: float = 0.0,
) -> list[DailyRow]:
    days = _get_daily_history_chunked(NIFTY_INDEX_KEY, from_date, to_date)
    days.sort(key=lambda d: d["date"])
    closes = [d["close"] for d in days]
    returns = _daily_returns(closes)

    rows: list[DailyRow] = []
    for i, d in enumerate(days):
        ret = returns[i]
        if ret is None:
            rows.append(DailyRow(d["date"], 0.0, None, None, None))
            continue
        vol = _trailing_realized_vol(returns, i, vol_window)
        if vol is None or vol <= 0:
            rows.append(DailyRow(d["date"], ret, vol, None, None))
            continue
        leverage = max(min_leverage, min(max_leverage, target_annual_vol / vol))
        strat_ret = leverage * ret
        rows.append(DailyRow(d["date"], ret, vol, leverage, strat_ret))
    return rows


def sharpe_ratio(returns: list[float]) -> float:
    if len(returns) < 2:
        return float("nan")
    mean = statistics.mean(returns)
    std = statistics.pstdev(returns)
    if std == 0:
        return float("nan")
    return mean / std * math.sqrt(TRADING_DAYS_PER_YEAR)


def static_leverage_returns(rows: list[DailyRow], leverage: float) -> list[float]:
    """Same window's own NIFTY returns, scaled by a single constant
    leverage throughout -- the control for the dynamic strategy."""
    return [leverage * r.nifty_return for r in rows if r.leverage is not None]


def summarize_window(rows: list[DailyRow], label: str) -> dict:
    active = [r for r in rows if r.leverage is not None]
    strat_returns = [r.strategy_return for r in active]
    nifty_returns = [r.nifty_return for r in active]
    avg_leverage = statistics.mean(r.leverage for r in active)
    static_returns = static_leverage_returns(rows, avg_leverage)

    strat_cum = 1.0
    for r in strat_returns:
        strat_cum *= (1 + r)
    nifty_cum = 1.0
    for r in nifty_returns:
        nifty_cum *= (1 + r)
    static_cum = 1.0
    for r in static_returns:
        static_cum *= (1 + r)

    n_years = len(active) / TRADING_DAYS_PER_YEAR
    return {
        "label": label,
        "from": rows[0].date, "to": rows[-1].date,
        "n_days": len(active),
        "avg_leverage": round(avg_leverage, 3),
        "dynamic_sharpe": round(sharpe_ratio(strat_returns), 3),
        "static_control_sharpe": round(sharpe_ratio(static_returns), 3),
        "nifty_sharpe": round(sharpe_ratio(nifty_returns), 3),
        "dynamic_total_return_pct": round((strat_cum - 1) * 100, 2),
        "static_control_total_return_pct": round((static_cum - 1) * 100, 2),
        "nifty_total_return_pct": round((nifty_cum - 1) * 100, 2),
        "dynamic_cagr_pct": round((strat_cum ** (1 / n_years) - 1) * 100, 2) if n_years > 0 else float("nan"),
    }


def run_two_window_validation(from_date: str, to_date: str, **overrides) -> tuple[dict, dict]:
    """Splits the full [from_date, to_date] range into two non-overlapping
    halves by TRADING-DAY count (not calendar midpoint), runs the overlay
    (and its static-leverage control) independently on each half."""
    rows = run(from_date, to_date, **overrides)
    mid = len(rows) // 2
    first_half = rows[:mid]
    second_half = rows[mid:]
    return summarize_window(first_half, "Window 1"), summarize_window(second_half, "Window 2")
