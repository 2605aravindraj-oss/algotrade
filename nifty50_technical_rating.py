"""Screen the NIFTY 50 constituents for a "Strong Buy" (Strong Bullish)
Technical Rating on the daily timeframe -- see backtest/technical_rating.py
for the exact methodology (a custom approximation of the well-known
moving-average + oscillator consensus rating, not a literal reproduction
of any specific platform's proprietary formula).

Usage:
    python nifty50_technical_rating.py
"""
from __future__ import annotations

import datetime

from backtest.technical_rating import rate
from data_sources import instruments, upstox_client

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


def main() -> None:
    to_date = datetime.date.today().isoformat()
    from_date = (datetime.date.today() - datetime.timedelta(days=420)).isoformat()

    print("Resolving instrument keys...")
    keys = instruments.resolve_symbols(NIFTY_50)

    results = []
    for symbol in NIFTY_50:
        key = keys.get(symbol)
        if key is None:
            print(f"  {symbol}: could not resolve instrument key")
            continue
        try:
            daily = upstox_client.get_daily_history(key, from_date, to_date)
        except Exception as e:
            print(f"  {symbol}: fetch failed - {e}")
            continue
        if not daily:
            print(f"  {symbol}: no daily data")
            continue
        candles = sorted(daily, key=lambda c: c["date"])
        r = rate(symbol, candles)
        if r is None:
            print(f"  {symbol}: not enough history ({len(candles)} bars)")
            continue
        results.append(r)
        print(f"  {symbol}: {r.label}  (rating={r.overall_rating:+.2f}, "
              f"MA {r.ma_buy}B/{r.ma_sell}S, Osc {r.osc_buy}B/{r.osc_sell}S/{r.osc_neutral}N, "
              f"close={r.close:.2f})")

    results.sort(key=lambda r: r.overall_rating, reverse=True)
    strong_buy = [r for r in results if r.label == "Strong Buy"]

    print(f"\n{'='*70}")
    print(f"STRONG BUY on daily timeframe ({len(strong_buy)} of {len(results)}):")
    print(f"{'='*70}")
    if not strong_buy:
        print("  None.")
    for r in strong_buy:
        print(f"  {r.symbol:14s} rating={r.overall_rating:+.2f}  close={r.close:,.2f}  "
              f"MA {r.ma_buy}/{r.ma_buy+r.ma_sell}  Osc {r.osc_buy}B/{r.osc_sell}S/{r.osc_neutral}N")


if __name__ == "__main__":
    main()
