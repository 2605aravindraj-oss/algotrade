from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PerformanceReport:
    total_return_pct: float
    cagr_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    num_trades: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def compute_performance(
    equity_curve: pd.Series,
    strategy_returns: pd.Series,
    positions: pd.Series,
    periods_per_year: int = 252,
) -> PerformanceReport:
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1

    num_periods = len(equity_curve)
    years = max(num_periods / periods_per_year, 1e-9)
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1

    vol = strategy_returns.std(ddof=0) * np.sqrt(periods_per_year)

    mean_return = strategy_returns.mean() * periods_per_year
    sharpe = mean_return / vol if vol > 0 else 0.0

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_drawdown = drawdown.min()

    trade_entries = positions.diff().fillna(positions.iloc[0] if len(positions) else 0) != 0
    num_trades = int(trade_entries.sum())

    winning_periods = (strategy_returns > 0).sum()
    losing_periods = (strategy_returns < 0).sum()
    total_active = winning_periods + losing_periods
    win_rate = winning_periods / total_active if total_active > 0 else 0.0

    return PerformanceReport(
        total_return_pct=round(total_return * 100, 2),
        cagr_pct=round(cagr * 100, 2),
        annualized_volatility_pct=round(vol * 100, 2),
        sharpe_ratio=round(sharpe, 2),
        max_drawdown_pct=round(max_drawdown * 100, 2),
        win_rate_pct=round(win_rate * 100, 2),
        num_trades=num_trades,
    )
