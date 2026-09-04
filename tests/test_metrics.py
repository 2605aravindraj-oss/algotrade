import pandas as pd
import pytest

from algotrade.backtest.metrics import compute_performance


def _series(values, freq="D"):
    index = pd.date_range("2023-01-01", periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=index)


def test_flat_equity_curve_has_zero_return_and_drawdown():
    equity = _series([1000, 1000, 1000, 1000])
    returns = _series([0, 0, 0, 0])
    positions = _series([0, 0, 0, 0])

    report = compute_performance(equity, returns, positions)

    assert report.total_return_pct == 0.0
    assert report.max_drawdown_pct == 0.0
    assert report.num_trades == 0


def test_drawdown_is_negative_after_a_dip():
    equity = _series([1000, 1200, 900, 1100])
    returns = equity.pct_change().fillna(0)
    positions = _series([1, 1, 1, 1])

    report = compute_performance(equity, returns, positions)

    assert report.max_drawdown_pct < 0
    assert report.max_drawdown_pct == pytest.approx(-25.0, abs=0.01)


def test_num_trades_counts_position_changes():
    equity = _series([1000, 1010, 1005, 1020, 1015])
    returns = equity.pct_change().fillna(0)
    positions = _series([0, 1, 1, 0, 1])

    report = compute_performance(equity, returns, positions)

    # 0->1, 1->0, 0->1: three position changes.
    assert report.num_trades == 3
