"""Minimal client for Yahoo Finance's public chart endpoint.

No API key is required, but requests need a browser-like User-Agent or
Yahoo's edge returns 429s.
"""
from __future__ import annotations

import requests

BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
TIMEOUT = 15
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


def get_chart(symbol: str, interval: str = "1d", range_: str = "5d") -> dict:
    """Fetch OHLCV chart data for a ticker, e.g. "AAPL" or "RELIANCE.NS"."""
    resp = requests.get(
        f"{BASE_URL}/{symbol}",
        params={"interval": interval, "range": range_},
        headers=HEADERS,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("chart", {}).get("error"):
        raise RuntimeError(payload["chart"]["error"])
    return payload


def get_last_price(symbol: str) -> float:
    result = get_chart(symbol, interval="1d", range_="1d")["chart"]["result"][0]
    return result["meta"]["regularMarketPrice"]
