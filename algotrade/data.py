"""Fetches historical candle data as a pandas DataFrame, with a local on-disk cache.

Supports two sources:
  - "yahoo" (default): Yahoo Finance via yfinance, no API token required.
  - "upstox": Upstox's historical-candle API, needs UPSTOX_ACCESS_TOKEN.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from algotrade.config import UpstoxConfig, load_config
from algotrade.instruments import find_instrument_key
from algotrade.upstox_client import fetch_historical_candles as fetch_upstox_candles
from algotrade.yahoo_client import fetch_historical_candles as fetch_yahoo_candles

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "oi"]

VALID_SOURCES = {"yahoo", "upstox"}


def _cache_path(source: str, symbol: str, exchange: str, interval: str, start: date, end: date) -> Path:
    name = f"{source}_{exchange}_{symbol}_{interval}_{start.isoformat()}_{end.isoformat()}.csv"
    return CACHE_DIR / name


def candles_to_dataframe(candles: list[list]) -> pd.DataFrame:
    """Converts Upstox's raw [timestamp, open, high, low, close, volume, oi] rows
    into the DataFrame shape shared with the Yahoo Finance client."""
    df = pd.DataFrame(candles, columns=CANDLE_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")
    for col in ["open", "high", "low", "close", "volume", "oi"]:
        df[col] = pd.to_numeric(df[col])
    return df


def get_historical_data(
    symbol: str,
    start: date,
    end: date,
    exchange: str = "NSE_EQ",
    interval: str = "day",
    source: str = "yahoo",
    use_cache: bool = True,
    config: UpstoxConfig | None = None,
) -> pd.DataFrame:
    """Return OHLCV history for `symbol` between `start` and `end` (inclusive), as a
    DataFrame indexed by UTC timestamp. Results are cached to data/cache/ as CSV so
    repeated backtests over the same range don't re-hit the network.
    """
    if source not in VALID_SOURCES:
        raise ValueError(f"source must be one of {sorted(VALID_SOURCES)}, got {source!r}")

    cache_path = _cache_path(source, symbol, exchange, interval, start, end)
    if use_cache and cache_path.exists():
        return pd.read_csv(cache_path, index_col="timestamp", parse_dates=["timestamp"])

    if source == "yahoo":
        df = fetch_yahoo_candles(symbol, exchange, interval, start, end)
    else:
        if config is None:
            config = load_config()
        instrument_key = find_instrument_key(symbol, exchange=exchange)
        candles = fetch_upstox_candles(instrument_key, interval, start, end, config=config)
        df = candles_to_dataframe(candles)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)

    return df
