"""Resolves trading symbols (e.g. "RELIANCE" on "NSE_EQ") to Upstox instrument keys.

Upstox identifies instruments by an opaque `instrument_key` (e.g. "NSE_EQ|INE002A01018")
rather than by symbol. The full instrument master is published as a gzipped JSON file;
we download it once and cache it locally, then look symbols up out of the cache.
"""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import requests

INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_PATH = CACHE_DIR / "instruments.json.gz"
CACHE_TTL_SECONDS = 24 * 60 * 60


def _download_instrument_master() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(INSTRUMENTS_URL, timeout=60)
    response.raise_for_status()
    CACHE_PATH.write_bytes(response.content)


def _load_instrument_master(force_refresh: bool = False) -> list[dict]:
    stale = (
        not CACHE_PATH.exists()
        or (time.time() - CACHE_PATH.stat().st_mtime) > CACHE_TTL_SECONDS
    )
    if force_refresh or stale:
        _download_instrument_master()
    with gzip.open(CACHE_PATH, "rt", encoding="utf-8") as f:
        return json.load(f)


def find_instrument_key(
    symbol: str,
    exchange: str = "NSE_EQ",
    force_refresh: bool = False,
) -> str:
    """Look up the Upstox instrument_key for a trading symbol on an exchange segment.

    exchange examples: "NSE_EQ" (equity cash), "NSE_FO" (futures & options), "BSE_EQ".
    """
    symbol = symbol.upper()
    instruments = _load_instrument_master(force_refresh=force_refresh)
    for entry in instruments:
        if (
            entry.get("exchange") == exchange
            and entry.get("trading_symbol", "").upper() == symbol
        ):
            return entry["instrument_key"]
    raise ValueError(f"No instrument found for symbol={symbol!r} exchange={exchange!r}")
