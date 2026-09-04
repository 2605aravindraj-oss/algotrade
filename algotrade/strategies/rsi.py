from __future__ import annotations

import pandas as pd

from algotrade.strategies.base import Strategy


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


class RsiMeanReversion(Strategy):
    """Long when RSI dips below `oversold`, flat once it climbs back above `exit_level`."""

    def __init__(self, period: int = 14, oversold: float = 30, exit_level: float = 50):
        self.period = period
        self.oversold = oversold
        self.exit_level = exit_level

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rsi = _rsi(df["close"], self.period)

        position = pd.Series(0, index=df.index)
        in_position = False
        for i, value in enumerate(rsi):
            if not in_position and value < self.oversold:
                in_position = True
            elif in_position and value > self.exit_level:
                in_position = False
            position.iloc[i] = 1 if in_position else 0
        return position
