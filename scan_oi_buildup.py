"""Run the NIFTY-futures-OI-buildup backtest (backtest/futures_oi_buildup.py)
across a basket of NSE stocks and compare results.

Individual stock options are monthly-only, so the tradeable window for each
stock ends at the most recent *past* expiry available from the
expired-instruments API (there's no chain data yet for the currently-live
expiry). Strike spacing is auto-detected per stock (options_common's
detect_strike_step) since it varies a lot by price, unlike NIFTY's flat 50.

Results are written incrementally to a JSON file so a partial run isn't
lost if it's interrupted -- each stock requires fresh option-chain backfill
via the Upstox API, which can take a couple of minutes per symbol.

Usage:
    export UPSTOX_ACCESS_TOKEN=...
    python scan_oi_buildup.py --from 2026-07-01 --out results.json
    python scan_oi_buildup.py RELIANCE TCS INFY --from 2026-07-01
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from datetime import datetime
from io import BytesIO

import requests

from backtest import futures_oi_buildup as m
from backtest import options_common as oc
from data_sources import cache, instruments

MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"

NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE",
    "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NESTLEIND",
    "WIPRO", "ONGC", "NTPC", "POWERGRID", "M&M", "TATASTEEL", "TATAMOTORS",
    "ADANIENT", "ADANIPORTS", "JSWSTEEL", "COALINDIA", "TECHM", "HCLTECH",
    "BAJAJFINSV", "DRREDDY", "GRASIM", "CIPLA", "EICHERMOT", "BRITANNIA",
    "DIVISLAB", "BPCL", "HEROMOTOCO", "HINDALCO", "INDUSINDBK", "SBILIFE",
    "HDFCLIFE", "APOLLOHOSP", "BAJAJ-AUTO", "TATACONSUM", "UPL",
    "SHRIRAMFIN", "LTIM",
]


def _load_master() -> list[dict]:
    resp = requests.get(MASTER_URL, timeout=30)
    resp.raise_for_status()
    with gzip.open(BytesIO(resp.content)) as f:
        return json.load(f)


def _nearest_future(master: list[dict], symbol: str) -> dict | None:
    futs = [d for d in master if d.get("segment") == "NSE_FO" and d.get("underlying_symbol") == symbol
            and d.get("instrument_type") == "FUT"]
    if not futs:
        return None
    return min(futs, key=lambda d: d["expiry"])


def run_one(symbol: str, master: list[dict], from_date: str, access_token: str) -> dict:
    fut = _nearest_future(master, symbol)
    if fut is None:
        return {"symbol": symbol, "error": "no futures contract found"}

    eq_keys = instruments.resolve_symbols([symbol])
    underlying_key = eq_keys.get(symbol)
    if underlying_key is None:
        return {"symbol": symbol, "error": "could not resolve equity instrument key"}

    expiries = cache.get_expired_expiries_cached(underlying_key, "options", access_token)
    if not expiries:
        return {"symbol": symbol, "error": "no expired option-chain data"}
    to_date = max(expiries)
    if to_date <= from_date:
        return {"symbol": symbol, "error": f"no tradeable window (last expiry {to_date} <= from {from_date})"}

    fut_days = None
    try:
        from data_sources import upstox_client
        fut_days = upstox_client.get_daily_history(fut["instrument_key"], from_date, to_date)
    except Exception as e:
        return {"symbol": symbol, "error": f"futures history fetch failed: {e}"}
    if not fut_days:
        return {"symbol": symbol, "error": "no futures daily history"}
    approx_price = fut_days[-1]["close"]

    nearest_expiry = min(e for e in expiries if e >= fut_days[0]["date"]) if any(e >= fut_days[0]["date"] for e in expiries) else expiries[0]
    chain = cache.get_expired_option_chain_cached(underlying_key, nearest_expiry, access_token)
    lookup = oc.build_chain_lookup(chain)
    strike_step = oc.detect_strike_step(lookup, approx_price)

    trades = m.run(
        from_date, to_date,
        futures_key=fut["instrument_key"],
        underlying_key=underlying_key,
        strike_step=strike_step,
        access_token=access_token,
    )
    if not trades:
        return {"symbol": symbol, "error": "no trades generated", "to_date": to_date, "strike_step": strike_step}

    gross = sum(t.pnl_rupees_gross for t in trades)
    net = sum(t.pnl_rupees for t in trades)
    costs_total = sum(t.cost_rupees for t in trades)
    wins = [t for t in trades if t.pnl_rupees > 0]
    gross_wins = [t for t in trades if t.pnl_rupees_gross > 0]

    by_day: dict[str, float] = {}
    for t in trades:
        by_day[t.date] = by_day.get(t.date, 0.0) + t.pnl_rupees
    days = sorted(by_day.items())
    win_days = sum(1 for _, pnl in days if pnl > 0)

    running = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        running += t.pnl_rupees
        peak = max(peak, running)
        max_dd = min(max_dd, running - peak)

    return {
        "symbol": symbol,
        "to_date": to_date,
        "strike_step": strike_step,
        "trades": len(trades),
        "trading_days": len(days),
        "winning_days": win_days,
        "winning_days_pct": round(100 * win_days / len(days), 1) if days else None,
        "gross_win_rate_pct": round(100 * len(gross_wins) / len(trades), 1),
        "net_win_rate_pct": round(100 * len(wins) / len(trades), 1),
        "gross_pnl": round(gross, 2),
        "total_costs": round(costs_total, 2),
        "net_pnl": round(net, 2),
        "avg_win": round(sum(t.pnl_rupees for t in wins) / len(wins), 2) if wins else 0,
        "avg_loss": round(sum(t.pnl_rupees for t in trades if t.pnl_rupees <= 0) / max(1, len(trades) - len(wins)), 2) if len(trades) > len(wins) else 0,
        "max_drawdown": round(max_dd, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="*", default=NIFTY_50)
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--out", dest="out_path", default="oi_buildup_scan_results.json")
    args = parser.parse_args()

    token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not token:
        print("UPSTOX_ACCESS_TOKEN not set.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading instrument master...")
    master = _load_master()

    results = []
    if os.path.exists(args.out_path):
        results = json.load(open(args.out_path))
        done_symbols = {r["symbol"] for r in results}
        print(f"Resuming: {len(done_symbols)} symbols already done.")
    else:
        done_symbols = set()

    for symbol in args.symbols:
        if symbol in done_symbols:
            continue
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Running {symbol}...", flush=True)
        try:
            r = run_one(symbol, master, args.from_date, token)
        except Exception as e:
            r = {"symbol": symbol, "error": str(e)}
        results.append(r)
        json.dump(results, open(args.out_path, "w"), indent=2)
        if "error" in r:
            print(f"  {symbol}: ERROR - {r['error']}")
        else:
            print(f"  {symbol}: {r['trades']} trades, net=Rs {r['net_pnl']:,.2f}, "
                  f"net_win_rate={r['net_win_rate_pct']}%, win_days={r['winning_days']}/{r['trading_days']}")

    print(f"\nDone. Results written to {args.out_path}")


if __name__ == "__main__":
    main()
