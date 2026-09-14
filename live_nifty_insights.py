"""Live NIFTY 50 options snapshot + insights, meant to be run every few
minutes during market hours (09:15-15:30 IST) via a scheduled job.

Fetches the current option chain (a band of strikes around ATM) for the
nearest weekly expiry, plus spot LTP, and reports:
  - spot price and its change since the last run
  - ATM strike and ATM straddle premium, and its change since last run
  - PCR (put OI / call OI) across the fetched strike band
  - the strike with the highest call OI (resistance) and put OI (support)

State is persisted to a small JSON file so each run can show deltas
since the previous one. Requires UPSTOX_ACCESS_TOKEN.

Usage:
    export UPSTOX_ACCESS_TOKEN=...
    python live_nifty_insights.py
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from io import BytesIO

import requests

from data_sources import upstox_client

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"
STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".live_nifty_state.json")
MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
STRIKE_BAND = 5  # strikes each side of ATM
IST = timezone(timedelta(hours=5, minutes=30))


def _load_master() -> list[dict]:
    resp = requests.get(MASTER_URL, timeout=30)
    resp.raise_for_status()
    with gzip.open(BytesIO(resp.content)) as f:
        return json.load(f)


def _nifty_option_chain(master: list[dict]) -> tuple[str, dict]:
    opts = [d for d in master if d.get("segment") == "NSE_FO" and d.get("underlying_symbol") == "NIFTY"
            and d.get("instrument_type") in ("CE", "PE")]
    nearest_expiry = min(d["expiry"] for d in opts)
    chain = [d for d in opts if d["expiry"] == nearest_expiry]
    expiry_str = datetime.utcfromtimestamp(nearest_expiry / 1000).strftime("%Y-%m-%d")
    lookup = {(d["strike_price"], d["instrument_type"]): d for d in chain}
    return expiry_str, lookup


def _round_to_50(x: float) -> int:
    return int(round(x / 50) * 50)


def main() -> None:
    token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not token:
        print("UPSTOX_ACCESS_TOKEN not set.", file=sys.stderr)
        sys.exit(1)

    now_ist = datetime.now(IST)
    print(f"=== NIFTY live snapshot @ {now_ist.strftime('%Y-%m-%d %H:%M:%S')} IST ===")

    if now_ist.strftime("%H:%M") < "09:15" or now_ist.strftime("%H:%M") > "15:30":
        print("Market is outside trading hours (09:15-15:30 IST) -- data below may be stale/last close.")

    master = _load_master()
    expiry, lookup = _nifty_option_chain(master)

    spot_data = upstox_client.get_quotes([UNDERLYING_KEY], token)["data"]
    spot = next(iter(spot_data.values()))["last_price"]
    atm = _round_to_50(spot)

    strikes = [atm + i * 50 for i in range(-STRIKE_BAND, STRIKE_BAND + 1)]
    keys_by_strike_type = {}
    all_keys = []
    for s in strikes:
        for t in ("CE", "PE"):
            c = lookup.get((s, t))
            if c:
                keys_by_strike_type[(s, t)] = c["instrument_key"]
                all_keys.append(c["instrument_key"])

    quotes = upstox_client.get_quotes(all_keys, token)["data"]

    def _q(strike, opt_type):
        key = keys_by_strike_type.get((strike, opt_type))
        if key is None:
            return None
        for v in quotes.values():
            if v.get("instrument_token") == key:
                return v
        return None

    total_call_oi = 0.0
    total_put_oi = 0.0
    max_call_oi = (None, -1)
    max_put_oi = (None, -1)
    for s in strikes:
        ce = _q(s, "CE")
        pe = _q(s, "PE")
        if ce:
            total_call_oi += ce.get("oi", 0)
            if ce.get("oi", 0) > max_call_oi[1]:
                max_call_oi = (s, ce.get("oi", 0))
        if pe:
            total_put_oi += pe.get("oi", 0)
            if pe.get("oi", 0) > max_put_oi[1]:
                max_put_oi = (s, pe.get("oi", 0))

    pcr = total_put_oi / total_call_oi if total_call_oi else None
    atm_ce = _q(atm, "CE")
    atm_pe = _q(atm, "PE")
    straddle = (atm_ce["last_price"] if atm_ce else 0) + (atm_pe["last_price"] if atm_pe else 0)

    prev = {}
    if os.path.exists(STATE_PATH):
        prev = json.load(open(STATE_PATH))

    spot_change = spot - prev["spot"] if "spot" in prev else None
    straddle_change = straddle - prev["straddle"] if "straddle" in prev else None

    print(f"Expiry: {expiry}")
    print(f"Spot: {spot:.2f}" + (f"  ({spot_change:+.2f} since last check)" if spot_change is not None else ""))
    print(f"ATM strike: {atm}")
    print(f"ATM straddle premium: {straddle:.2f}" + (f"  ({straddle_change:+.2f} since last check)" if straddle_change is not None else ""))
    print(f"PCR (OI-based, +/-{STRIKE_BAND} strikes): {pcr:.2f}" if pcr else "PCR: n/a")
    print(f"Highest call OI (resistance): {max_call_oi[0]}  (OI={max_call_oi[1]:,.0f})" if max_call_oi[0] else "")
    print(f"Highest put OI (support):     {max_put_oi[0]}  (OI={max_put_oi[1]:,.0f})" if max_put_oi[0] else "")

    json.dump({"spot": spot, "straddle": straddle, "ts": now_ist.isoformat()}, open(STATE_PATH, "w"))


if __name__ == "__main__":
    main()
