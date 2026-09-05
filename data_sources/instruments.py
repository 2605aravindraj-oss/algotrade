"""Resolve NSE trading symbols (e.g. "TCS") to Upstox instrument keys
(e.g. "NSE_EQ|INE467B01029") using Upstox's public instrument master.
"""
from __future__ import annotations

import gzip
import json
import os
import tempfile
import time
from io import BytesIO

import requests

MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
TIMEOUT = 30
CACHE_PATH = os.path.join(tempfile.gettempdir(), "upstox_nse_instruments.json.gz")
CACHE_TTL_SECONDS = 24 * 60 * 60


def _download_master() -> bytes:
    resp = requests.get(MASTER_URL, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.content


def load_equity_master(force_refresh: bool = False) -> list[dict]:
    """Return all NSE_EQ instrument records, downloading/caching as needed."""
    stale = (
        force_refresh
        or not os.path.exists(CACHE_PATH)
        or time.time() - os.path.getmtime(CACHE_PATH) > CACHE_TTL_SECONDS
    )
    if stale:
        raw = _download_master()
        with open(CACHE_PATH, "wb") as f:
            f.write(raw)
    else:
        with open(CACHE_PATH, "rb") as f:
            raw = f.read()

    with gzip.open(BytesIO(raw)) as f:
        data = json.load(f)
    return [d for d in data if d.get("instrument_type") == "EQ"]


def resolve_symbols(trading_symbols: list[str]) -> dict[str, str | None]:
    """Map trading symbols (e.g. "TCS") to instrument keys. Missing symbols map to None."""
    wanted = set(trading_symbols)
    master = load_equity_master()
    found = {d["trading_symbol"]: d["instrument_key"] for d in master if d["trading_symbol"] in wanted}
    return {s: found.get(s) for s in trading_symbols}
