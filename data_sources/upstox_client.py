"""Minimal client for Upstox's public and authenticated market-data endpoints.

Historical candles are served without authentication. Live quotes and order
placement require an OAuth access token (see https://upstox.com/developer/api-documentation/authentication)
passed via the UPSTOX_ACCESS_TOKEN environment variable or the access_token argument.
"""
from __future__ import annotations

import os
import time
from datetime import date
from urllib.parse import quote

import requests

BASE_URL = "https://api.upstox.com/v2"
TIMEOUT = 15
RETRY_BACKOFF_SECONDS = (2, 4, 8, 16)


def _get_with_retry(url: str, **kwargs) -> requests.Response:
    """GET with retry on transient network errors (timeouts, connection
    resets) using exponential backoff. Does not retry on HTTP error status
    codes -- those are real API responses, not transient failures."""
    last_exc = None
    for attempt, delay in enumerate((0,) + RETRY_BACKOFF_SECONDS):
        if delay:
            time.sleep(delay)
        try:
            return requests.get(url, timeout=TIMEOUT, **kwargs)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_exc = exc
    raise last_exc


def get_historical_candles(
    instrument_key: str,
    interval: str = "day",
    to_date: str | date | None = None,
    from_date: str | date | None = None,
) -> dict:
    """Fetch OHLC candles for an instrument. No auth required.

    instrument_key example: "NSE_EQ|INE002A01018" (Reliance Industries).
    interval: one of "day", "week", "month", "30minute", "1minute", ...
    """
    to_date = to_date or date.today().isoformat()
    path = f"{BASE_URL}/historical-candle/{quote(instrument_key, safe='')}/{interval}/{to_date}"
    if from_date:
        path += f"/{from_date}"
    resp = _get_with_retry(path)
    resp.raise_for_status()
    return resp.json()


def get_daily_history(
    instrument_key: str,
    from_date: str | date,
    to_date: str | date | None = None,
) -> list[dict]:
    """Fetch daily candles as a list of dicts sorted oldest-to-newest.

    Each dict has keys: date, open, high, low, close, volume, oi.
    """
    raw = get_historical_candles(instrument_key, interval="day", to_date=to_date, from_date=from_date)
    candles = raw["data"]["candles"]
    parsed = [
        {
            "date": c[0][:10],
            "open": c[1],
            "high": c[2],
            "low": c[3],
            "close": c[4],
            "volume": c[5],
            "oi": c[6],
        }
        for c in candles
    ]
    parsed.sort(key=lambda c: c["date"])
    return parsed


def get_quotes(instrument_keys: list[str], access_token: str | None = None) -> dict:
    """Fetch live market quotes. Requires a valid OAuth access token."""
    token = access_token or os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not token:
        raise RuntimeError(
            "Upstox live quotes require an access token. Set UPSTOX_ACCESS_TOKEN "
            "or pass access_token explicitly."
        )
    resp = _get_with_retry(
        f"{BASE_URL}/market-quote/quotes",
        params={"symbol": ",".join(instrument_keys)},
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()


def _auth_headers(access_token: str | None) -> dict:
    token = access_token or os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not token:
        raise RuntimeError(
            "This endpoint requires an access token. Set UPSTOX_ACCESS_TOKEN "
            "or pass access_token explicitly."
        )
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def get_expired_expiries(
    underlying_key: str, expiry_type: str = "options", access_token: str | None = None
) -> list[str]:
    """List past expiry dates for an underlying's options/futures. Requires auth."""
    resp = _get_with_retry(
        f"{BASE_URL}/expired-instruments/expiries",
        params={"instrument_key": underlying_key, "expiry_type": expiry_type},
        headers=_auth_headers(access_token),
    )
    resp.raise_for_status()
    return resp.json()["data"]


def get_expired_option_chain(
    underlying_key: str, expiry_date: str | date, access_token: str | None = None
) -> list[dict]:
    """List expired option contracts (all strikes/CE+PE) for one expiry. Requires auth."""
    resp = _get_with_retry(
        f"{BASE_URL}/expired-instruments/option/contract",
        params={"instrument_key": underlying_key, "expiry_date": str(expiry_date)},
        headers=_auth_headers(access_token),
    )
    resp.raise_for_status()
    return resp.json()["data"]


def get_expired_candles(
    expired_instrument_key: str,
    interval: str,
    to_date: str | date,
    from_date: str | date,
    access_token: str | None = None,
) -> list[list]:
    """Fetch candles for an expired instrument key (e.g. "NSE_FO|132352|28-08-2025").
    Requires auth. Returns raw candle rows, newest first.
    """
    path = f"{BASE_URL}/expired-instruments/historical-candle/{quote(expired_instrument_key, safe='')}/{interval}/{to_date}/{from_date}"
    resp = _get_with_retry(path, headers=_auth_headers(access_token))
    resp.raise_for_status()
    return resp.json()["data"]["candles"]
