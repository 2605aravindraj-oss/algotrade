"""Vectorized single-instrument backtest engine.

Positions are assumed to be either flat, fully long, or fully short (position
sizing / portfolio allocation is out of scope here). A signal generated from bar
t's close is applied to the return realized over bar t+1, so the backtest never
trades on information it couldn't have had at the time.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from algotrade.backtest.metrics import PerformanceReport, compute_performance
from algotrade.strategies.base import Strategy


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    positions: pd.Series
    returns: pd.Series
    performance: PerformanceReport


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    initial_capital: float = 100_000.0,
    commission_bps: float = 0.0,
    periods_per_year: int = 252,
) -> BacktestResult:
    """Run `strategy` over `df` (must contain a 'close' column) and return equity
    curve, positions, per-bar returns, and summary performance metrics.

    commission_bps: round-trip-agnostic cost charged (in basis points of notional)
    on every bar where the position changes, to approximate brokerage + slippage.
    """
    if "close" not in df.columns:
        raise ValueError("df must contain a 'close' column")
    if df.empty:
        raise ValueError("df is empty")

    positions = strategy.generate_signals(df).reindex(df.index).fillna(0)
    market_returns = df["close"].pct_change().fillna(0)

    # Position at time t decided using data through t is realized as a return over
    # the following bar, so shift by 1 to avoid look-ahead bias.
    active_positions = positions.shift(1).fillna(0)
    strategy_returns = active_positions * market_returns

    if commission_bps > 0:
        position_changes = positions.diff().abs().fillna(positions.abs())
        costs = position_changes * (commission_bps / 10_000)
        strategy_returns = strategy_returns - costs

    equity_curve = initial_capital * (1 + strategy_returns).cumprod()

    performance = compute_performance(
        equity_curve, strategy_returns, positions, periods_per_year=periods_per_year
    )

    return BacktestResult(
        equity_curve=equity_curve,
        positions=positions,
        returns=strategy_returns,
        performance=performance,
    )
