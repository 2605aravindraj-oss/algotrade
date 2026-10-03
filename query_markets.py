"""Demo: fetch recent data from Upstox (historical candles) and Yahoo Finance.

Usage:
    python query_markets.py
"""
from datetime import date, timedelta

from data_sources import upstox_client, yahoo_client

# Reliance Industries: Upstox instrument key vs. Yahoo Finance ticker.
UPSTOX_INSTRUMENT_KEY = "NSE_EQ|INE002A01018"
YAHOO_SYMBOL = "RELIANCE.NS"


def main() -> None:
    print(f"Upstox historical candles for {UPSTOX_INSTRUMENT_KEY}:")
    to_date = date.today()
    from_date = to_date - timedelta(days=7)
    candles = upstox_client.get_historical_candles(
        UPSTOX_INSTRUMENT_KEY, interval="day", to_date=to_date.isoformat(), from_date=from_date.isoformat()
    )["data"]["candles"]
    for candle in candles:
        ts, o, h, l, c, vol, _ = candle
        print(f"  {ts}  O={o} H={h} L={l} C={c} V={vol}")

    print(f"\nYahoo Finance last price for {YAHOO_SYMBOL}:")
    print(f"  {yahoo_client.get_last_price(YAHOO_SYMBOL)}")

    print("\nUpstox live quote (requires UPSTOX_ACCESS_TOKEN):")
    try:
        quotes = upstox_client.get_quotes([UPSTOX_INSTRUMENT_KEY])
        print(f"  {quotes}")
    except RuntimeError as exc:
        print(f"  skipped: {exc}")


if __name__ == "__main__":
    main()
