"""Run the downtrend reversal / break-of-structure + retest backtest across
multiple NSE stocks and compare results.

Usage:
    python scan_structure_reversal.py TCS INFY HDFCBANK RELIANCE
    python scan_structure_reversal.py   # uses a default large-cap basket
"""
import argparse
from datetime import date

from backtest import structure_reversal
from data_sources import instruments, upstox_client

DEFAULT_BASKET = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "ITC", "LT",
    "AXISBANK", "WIPRO", "MARUTI", "SUNPHARMA", "HINDUNILVR", "KOTAKBANK",
    "BAJFINANCE", "ADANIENT", "TATASTEEL", "BHARTIARTL", "ONGC", "NTPC",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="*", default=DEFAULT_BASKET)
    parser.add_argument("--from", dest="from_date", default="2023-01-01")
    parser.add_argument("--to", dest="to_date", default=date.today().isoformat())
    args = parser.parse_args()

    print(f"Resolving {len(args.symbols)} symbols against Upstox's NSE instrument master...")
    keys = instruments.resolve_symbols(args.symbols)
    missing = [s for s, k in keys.items() if k is None]
    if missing:
        print(f"  could not resolve: {', '.join(missing)}")

    rows = []
    for symbol in args.symbols:
        key = keys.get(symbol)
        if key is None:
            continue
        try:
            candles = upstox_client.get_daily_history(key, args.from_date, args.to_date)
            if len(candles) < 60:
                rows.append((symbol, None, "not enough history"))
                continue
            result = structure_reversal.run(candles)
            rows.append((symbol, result, None))
        except Exception as exc:  # network/API hiccup on one symbol shouldn't kill the scan
            rows.append((symbol, None, str(exc)))

    print(f"\n{'Symbol':<12} {'Return':>8} {'BuyHold':>9} {'MaxDD':>8} {'Setups':>7} {'Trades':>7} {'WinRate':>8}")
    for symbol, result, error in rows:
        if error:
            print(f"{symbol:<12} {error}")
            continue
        print(
            f"{symbol:<12} {result.total_return_pct:>+7.2f}% {result.buy_hold_return_pct:>+8.2f}% "
            f"{result.max_drawdown_pct:>7.2f}% {len(result.setups):>7} {len(result.trades):>7} "
            f"{result.win_rate_pct:>7.1f}%"
        )


if __name__ == "__main__":
    main()
