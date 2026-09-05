"""Fetches historical OHLCV candles from Yahoo Finance via yfinance.

Unlike Upstox, Yahoo Finance needs no API token, which makes it a much easier
default for backtesting. Indian equities are addressed with an exchange suffix
("RELIANCE.NS" on NSE, "RELIANCE.BO" on BSE); other symbols are passed through
as-is so US tickers etc. work unchanged.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf

# Maps our Upstox-style interval names to yfinance's interval strings, so the
# rest of the codebase (CLI, cache keys) doesn't need to know which data source
# is in use. Note Yahoo limits how far back intraday intervals go: "1m" data
# only within the last ~7 days, other sub-daily intervals within ~60 days.
INTERVAL_MAP = {
    "1minute": "1m",
    "30minute": "30m",
    "day": "1d",
    "week": "1wk",
    "month": "1mo",
}

EXCHANGE_SUFFIX = {
    "NSE_EQ": ".NS",
    "BSE_EQ": ".BO",
}


def to_yahoo_ticker(symbol: str, exchange: str) -> str:
    symbol = symbol.upper()
    suffix = EXCHANGE_SUFFIX.get(exchange, "")
    if suffix and not symbol.endswith(suffix):
        return f"{symbol}{suffix}"
    return symbol


def fetch_historical_candles(
    symbol: str,
    exchange: str,
    interval: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Return OHLCV history as a DataFrame indexed by UTC timestamp, with columns
    open/high/low/close/volume/oi (Yahoo has no open-interest data, so oi is 0).
    """
    if interval not in INTERVAL_MAP:
        raise ValueError(f"interval must be one of {sorted(INTERVAL_MAP)}, got {interval!r}")

    ticker = to_yahoo_ticker(symbol, exchange)
    # yfinance's `end` is exclusive; add a day so `end` itself is included.
    raw = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval=INTERVAL_MAP[interval],
        progress=False,
        auto_adjust=False,
    )
    if raw.empty:
        raise ValueError(f"Yahoo Finance returned no data for {ticker!r} between {start} and {end}")

    return normalize_yahoo_dataframe(raw)


def normalize_yahoo_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)

    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"

    df["oi"] = 0
    return df[["open", "high", "low", "close", "volume", "oi"]]
