"""SQLite-backed local cache for Upstox market data.

Without this, every backtest re-fetches the same candles/chains from
Upstox's API on every run, even when nothing has changed. This module
wraps the upstox_client calls that dominate that cost (per-contract
1-minute candles, option chains, expiry lists): check the local DB first,
only hit the API for what's missing, and persist the result.

The DB file lives at data/market_cache.db (relative to the repo root) so
it can be committed to git and reused across sessions/machines.
"""
from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager

from data_sources import upstox_client

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "market_cache.db"
)

# Only applied on an actual live API fetch (cache hits are instant).
API_SLEEP_SECONDS = 0.2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    instrument_key TEXT, interval TEXT, date TEXT, ts TEXT,
    open REAL, high REAL, low REAL, close REAL, volume INTEGER, oi INTEGER,
    PRIMARY KEY (instrument_key, interval, ts)
);
CREATE INDEX IF NOT EXISTS idx_candles_lookup ON candles (instrument_key, interval, date);

CREATE TABLE IF NOT EXISTS day_fetched (
    instrument_key TEXT, interval TEXT, date TEXT,
    PRIMARY KEY (instrument_key, interval, date)
);

CREATE TABLE IF NOT EXISTS option_contracts (
    underlying_key TEXT, expiry TEXT, strike REAL, option_type TEXT,
    instrument_key TEXT, lot_size INTEGER, trading_symbol TEXT,
    PRIMARY KEY (underlying_key, expiry, strike, option_type)
);

CREATE TABLE IF NOT EXISTS expiries (
    underlying_key TEXT, expiry_type TEXT, expiry TEXT,
    PRIMARY KEY (underlying_key, expiry_type, expiry)
);
"""


@contextmanager
def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(_SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_expired_expiries_cached(
    underlying_key: str, expiry_type: str = "options", access_token: str | None = None
) -> list[str]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT expiry FROM expiries WHERE underlying_key=? AND expiry_type=? ORDER BY expiry",
            (underlying_key, expiry_type),
        ).fetchall()
        if rows:
            return [r[0] for r in rows]

    expiries = upstox_client.get_expired_expiries(underlying_key, expiry_type, access_token)
    with _conn() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO expiries VALUES (?,?,?)",
            [(underlying_key, expiry_type, e) for e in expiries],
        )
    return expiries


def get_expired_option_chain_cached(
    underlying_key: str, expiry: str, access_token: str | None = None
) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT strike, option_type, instrument_key, lot_size, trading_symbol "
            "FROM option_contracts WHERE underlying_key=? AND expiry=?",
            (underlying_key, str(expiry)),
        ).fetchall()
        if rows:
            return [
                {
                    "strike_price": r[0], "instrument_type": r[1], "instrument_key": r[2],
                    "lot_size": r[3], "trading_symbol": r[4],
                }
                for r in rows
            ]

    chain = upstox_client.get_expired_option_chain(underlying_key, expiry, access_token)
    with _conn() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO option_contracts VALUES (?,?,?,?,?,?,?)",
            [
                (underlying_key, str(expiry), c["strike_price"], c["instrument_type"],
                 c["instrument_key"], c["lot_size"], c["trading_symbol"])
                for c in chain
            ],
        )
    return chain


def get_day_candles_cached(
    instrument_key: str,
    interval: str,
    date: str,
    expired: bool,
    access_token: str | None = None,
) -> list[list]:
    """Candles for a single day. expired=True uses the expired-instruments
    API (needs auth); expired=False uses the public historical-candle
    endpoint (spot/index, or a still-live equity)."""
    with _conn() as conn:
        fetched = conn.execute(
            "SELECT 1 FROM day_fetched WHERE instrument_key=? AND interval=? AND date=?",
            (instrument_key, interval, date),
        ).fetchone()
        if fetched:
            rows = conn.execute(
                "SELECT ts, open, high, low, close, volume, oi FROM candles "
                "WHERE instrument_key=? AND interval=? AND date=? ORDER BY ts",
                (instrument_key, interval, date),
            ).fetchall()
            return [list(r) for r in rows]

    if expired:
        candles = upstox_client.get_expired_candles(instrument_key, interval, date, date, access_token)
    else:
        candles = upstox_client.get_historical_candles(instrument_key, interval, date, date)["data"]["candles"]
    time.sleep(API_SLEEP_SECONDS)

    with _conn() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO candles VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                (instrument_key, interval, c[0][:10], c[0], c[1], c[2], c[3], c[4], c[5], c[6])
                for c in candles
            ],
        )
        conn.execute(
            "INSERT OR IGNORE INTO day_fetched VALUES (?,?,?)",
            (instrument_key, interval, date),
        )
    return candles


def stats() -> dict:
    if not os.path.exists(DB_PATH):
        return {"candle_rows": 0, "days_cached": 0, "contracts": 0, "expiries": 0, "db_size_mb": 0}
    with _conn() as conn:
        return {
            "candle_rows": conn.execute("SELECT COUNT(*) FROM candles").fetchone()[0],
            "days_cached": conn.execute("SELECT COUNT(*) FROM day_fetched").fetchone()[0],
            "contracts": conn.execute("SELECT COUNT(*) FROM option_contracts").fetchone()[0],
            "expiries": conn.execute("SELECT COUNT(*) FROM expiries").fetchone()[0],
            "db_size_mb": os.path.getsize(DB_PATH) / 1_000_000,
        }
