"""Thin client around Upstox's historical candle REST endpoints."""
from __future__ import annotations

from datetime import date

import requests

from algotrade.config import UpstoxConfig

BASE_URL = "https://api.upstox.com/v2"

# Interval names accepted by the Upstox historical-candle endpoint.
VALID_INTERVALS = {"1minute", "30minute", "day", "week", "month"}


def fetch_historical_candles(
    instrument_key: str,
    interval: str,
    from_date: date,
    to_date: date,
    config: UpstoxConfig | None = None,
) -> list[list]:
    """Fetch raw OHLC(V) candles for one instrument between two dates (inclusive).

    Returns Upstox's raw candle rows: [timestamp, open, high, low, close, volume, oi],
    newest first, as documented at https://upstox.com/developer/api-documentation/.
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(f"interval must be one of {sorted(VALID_INTERVALS)}, got {interval!r}")

    url = (
        f"{BASE_URL}/historical-candle/{instrument_key}/{interval}/"
        f"{to_date.isoformat()}/{from_date.isoformat()}"
    )
    headers = {"Accept": "application/json"}
    if config is not None:
        headers.update(config.auth_header)

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if payload.get("status") != "success":
        raise RuntimeError(f"Upstox API error: {payload}")

    return payload["data"]["candles"]
