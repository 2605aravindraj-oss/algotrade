"""Thin wrapper around the Upstox v2 REST API for NIFTY 50 option chain data.

Auth flow (Upstox uses OAuth2 with a daily-expiring access token):
  1. Create an app at https://developer.upstox.com/ to get API_KEY, API_SECRET
     and register a REDIRECT_URI.
  2. Run `python -m nifty_options.cli login` -> open the printed URL, log in,
     and Upstox redirects to REDIRECT_URI with `?code=...`.
  3. Run `python -m nifty_options.cli auth --code <code>` to exchange it for
     an access_token, which is cached in .upstox_token.json (gitignored).
  4. The access token is valid until ~3:30am IST the next day, so step 2-3
     must be repeated once per trading day.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://api.upstox.com/v2"
NIFTY_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
TOKEN_CACHE_PATH = Path(__file__).resolve().parent.parent / ".upstox_token.json"


class UpstoxAuthError(RuntimeError):
    pass


class UpstoxClient:
    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        redirect_uri: str | None = None,
        access_token: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("UPSTOX_API_KEY")
        self.api_secret = api_secret or os.environ.get("UPSTOX_API_SECRET")
        self.redirect_uri = redirect_uri or os.environ.get("UPSTOX_REDIRECT_URI")
        self.access_token = access_token or os.environ.get("UPSTOX_ACCESS_TOKEN") or self._load_cached_token()
        self.session = requests.Session()

    # ---- auth -----------------------------------------------------------

    def _load_cached_token(self) -> str | None:
        if TOKEN_CACHE_PATH.exists():
            try:
                return json.loads(TOKEN_CACHE_PATH.read_text()).get("access_token")
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def _cache_token(self, access_token: str) -> None:
        TOKEN_CACHE_PATH.write_text(json.dumps({"access_token": access_token, "cached_at": time.time()}))

    def login_url(self, state: str = "nifty_options") -> str:
        if not self.api_key or not self.redirect_uri:
            raise UpstoxAuthError("UPSTOX_API_KEY and UPSTOX_REDIRECT_URI must be set to build a login URL")
        return (
            f"{BASE_URL}/login/authorization/dialog"
            f"?response_type=code&client_id={self.api_key}"
            f"&redirect_uri={self.redirect_uri}&state={state}"
        )

    def exchange_code_for_token(self, code: str) -> str:
        if not (self.api_key and self.api_secret and self.redirect_uri):
            raise UpstoxAuthError("UPSTOX_API_KEY, UPSTOX_API_SECRET and UPSTOX_REDIRECT_URI must all be set")
        resp = self.session.post(
            f"{BASE_URL}/login/authorization/token",
            headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
            data={
                "code": code,
                "client_id": self.api_key,
                "client_secret": self.api_secret,
                "redirect_uri": self.redirect_uri,
                "grant_type": "authorization_code",
            },
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        token = payload.get("access_token")
        if not token:
            raise UpstoxAuthError(f"No access_token in response: {payload}")
        self.access_token = token
        self._cache_token(token)
        return token

    # ---- market data ------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.access_token:
            raise UpstoxAuthError(
                "No Upstox access token available. Run `login` + `auth` first, "
                "or set UPSTOX_ACCESS_TOKEN."
            )
        resp = self.session.get(
            f"{BASE_URL}{path}",
            params=params,
            headers={"Accept": "application/json", "Authorization": f"Bearer {self.access_token}"},
            timeout=15,
        )
        if resp.status_code == 401:
            raise UpstoxAuthError(f"Upstox rejected the access token (401): {resp.text}")
        resp.raise_for_status()
        return resp.json()

    def get_option_contracts(self, instrument_key: str = NIFTY_INSTRUMENT_KEY) -> list[dict[str, Any]]:
        """List available option contracts (one row per strike/expiry) - used to discover expiry dates."""
        data = self._get("/option/contract", {"instrument_key": instrument_key})
        return data.get("data", [])

    def get_option_chain(self, expiry_date: str, instrument_key: str = NIFTY_INSTRUMENT_KEY) -> list[dict[str, Any]]:
        """Full option chain (CE+PE per strike) for one expiry, including OI, LTP and greeks."""
        data = self._get("/option/chain", {"instrument_key": instrument_key, "expiry_date": expiry_date})
        return data.get("data", [])

    def nearest_expiry(self, instrument_key: str = NIFTY_INSTRUMENT_KEY) -> str:
        contracts = self.get_option_contracts(instrument_key)
        expiries = sorted({c["expiry"] for c in contracts if c.get("expiry")})
        if not expiries:
            raise RuntimeError("Could not determine any option expiry dates from Upstox")
        return expiries[0]
