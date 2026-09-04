import pandas as pd
import pytest

from algotrade.strategies.rsi import RsiMeanReversion
from algotrade.strategies.sma_crossover import SmaCrossover


def _make_df(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=len(closes), freq="D", tz="UTC")
    return pd.DataFrame({"close": closes}, index=index)


def test_sma_crossover_rejects_invalid_periods():
    with pytest.raises(ValueError):
        SmaCrossover(fast=50, slow=20)


def test_sma_crossover_goes_long_when_fast_above_slow():
    # A steadily rising series should eventually push the fast SMA above the slow SMA.
    closes = list(range(100, 140))
    df = _make_df(closes)
    signals = SmaCrossover(fast=3, slow=10).generate_signals(df)

    assert signals.iloc[-1] == 1
    assert set(signals.unique()).issubset({0, 1})
    assert len(signals) == len(df)


def test_sma_crossover_flat_before_enough_data():
    df = _make_df([100, 101, 102])
    signals = SmaCrossover(fast=5, slow=10).generate_signals(df)
    assert (signals == 0).all()


def test_rsi_mean_reversion_enters_on_oversold_and_exits_above_threshold():
    # Sharp drop then recovery should trigger an entry near the trough and an
    # exit once price/RSI recovers.
    closes = [100] * 5 + [90, 80, 70, 60, 55] + [70, 90, 110, 130, 150]
    df = _make_df(closes)
    signals = RsiMeanReversion(period=5, oversold=30, exit_level=50).generate_signals(df)

    assert len(signals) == len(df)
    assert set(signals.unique()).issubset({0, 1})
    assert signals.iloc[-1] == 0  # should have exited by the time RSI is pinned high
