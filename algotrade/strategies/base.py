"""Base class for backtestable strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """A strategy turns an OHLCV DataFrame into a target-position series.

    generate_signals must return a pd.Series aligned to df.index, valued in
    {-1, 0, 1} meaning short / flat / long. The backtest engine shifts this by
    one bar before applying it to returns, so signals may use the current bar's
    close without introducing look-ahead bias.
    """

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError
