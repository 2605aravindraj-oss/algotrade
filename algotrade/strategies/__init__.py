from algotrade.strategies.base import Strategy
from algotrade.strategies.rsi import RsiMeanReversion
from algotrade.strategies.sma_crossover import SmaCrossover

STRATEGY_REGISTRY = {
    "sma_crossover": SmaCrossover,
    "rsi": RsiMeanReversion,
}

__all__ = ["Strategy", "SmaCrossover", "RsiMeanReversion", "STRATEGY_REGISTRY"]
