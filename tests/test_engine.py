import pandas as pd
import pytest

from algotrade.backtest.engine import run_backtest
from algotrade.strategies.base import Strategy
from algotrade.strategies.sma_crossover import SmaCrossover


def _make_df(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=len(closes), freq="D", tz="UTC")
    return pd.DataFrame({"close": closes}, index=index)


class AlwaysLong(Strategy):
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=df.index)


class AlwaysFlat(Strategy):
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=df.index)


def test_always_long_matches_buy_and_hold():
    closes = [100, 105, 110, 108, 115]
    df = _make_df(closes)
    result = run_backtest(df, AlwaysLong(), initial_capital=1000)

    # First bar has no prior signal to act on (shift(1) => 0 exposure on bar 0),
    # so being "always long" earns exactly the buy-and-hold return from bar 0's close.
    expected_final_equity = 1000 * (closes[-1] / closes[0])
    assert result.equity_curve.iloc[-1] == pytest.approx(expected_final_equity)


def test_always_flat_equity_never_moves():
    df = _make_df([100, 105, 95, 120])
    result = run_backtest(df, AlwaysFlat(), initial_capital=1000)
    assert (result.equity_curve == 1000).all()


def test_signal_is_shifted_to_avoid_lookahead():
    # A strategy that goes long only on the final bar shouldn't earn any return,
    # since there's no next bar left to realize it on.
    df = _make_df([100, 100, 100, 200])

    class LongOnLastBar(Strategy):
        def generate_signals(self, df: pd.DataFrame) -> pd.Series:
            signal = pd.Series(0, index=df.index)
            signal.iloc[-1] = 1
            return signal

    result = run_backtest(df, LongOnLastBar(), initial_capital=1000)
    assert result.equity_curve.iloc[-1] == pytest.approx(1000)


def test_commission_reduces_returns():
    df = _make_df([100, 110, 100, 110, 100])
    strategy = SmaCrossover(fast=1, slow=2)

    no_cost = run_backtest(df, strategy, initial_capital=1000, commission_bps=0)
    with_cost = run_backtest(df, strategy, initial_capital=1000, commission_bps=50)

    assert with_cost.equity_curve.iloc[-1] <= no_cost.equity_curve.iloc[-1]


def test_run_backtest_requires_close_column():
    df = pd.DataFrame({"open": [1, 2, 3]})
    with pytest.raises(ValueError):
        run_backtest(df, SmaCrossover())


def test_run_backtest_rejects_empty_df():
    df = pd.DataFrame({"close": []})
    with pytest.raises(ValueError):
        run_backtest(df, SmaCrossover())
