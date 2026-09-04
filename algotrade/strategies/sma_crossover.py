from __future__ import annotations

import pandas as pd

from algotrade.strategies.base import Strategy


class SmaCrossover(Strategy):
    """Long when the fast SMA is above the slow SMA, flat otherwise."""

    def __init__(self, fast: int = 20, slow: int = 50):
        if fast >= slow:
            raise ValueError("fast period must be smaller than slow period")
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast_sma = df["close"].rolling(self.fast).mean()
        slow_sma = df["close"].rolling(self.slow).mean()
        signal = (fast_sma > slow_sma).astype(int)
        return signal.reindex(df.index).fillna(0).astype(int)
