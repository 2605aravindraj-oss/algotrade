"""Backtest an EMA crossover strategy on Upstox historical candles.

Usage:
    python run_ema_backtest.py --instrument "NSE_EQ|INE002A01018" \
        --from 2023-01-01 --fast 12 --slow 26
"""
import argparse
from datetime import date

from backtest import ema_crossover
from data_sources import upstox_client


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", default="NSE_EQ|INE002A01018", help="Upstox instrument key")
    parser.add_argument("--from", dest="from_date", default="2023-01-01")
    parser.add_argument("--to", dest="to_date", default=date.today().isoformat())
    parser.add_argument("--fast", type=int, default=12)
    parser.add_argument("--slow", type=int, default=26)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--trades", type=int, default=10, help="number of recent trades to print")
    args = parser.parse_args()

    print(f"Fetching {args.instrument} daily candles from {args.from_date} to {args.to_date}...")
    candles = upstox_client.get_daily_history(args.instrument, args.from_date, args.to_date)
    print(f"Got {len(candles)} candles.\n")

    result = ema_crossover.run(candles, fast=args.fast, slow=args.slow, initial_capital=args.capital)

    print(f"EMA({args.fast}/{args.slow}) crossover on {args.instrument}")
    print(f"Period: {candles[0]['date']} to {candles[-1]['date']}\n")
    print(result.summary())

    print(f"\nLast {args.trades} trades:")
    for t in result.trades[-args.trades:]:
        exit_str = f"{t.exit_date} @ {t.exit_price}" if t.exit_price else "OPEN"
        ret_str = f"{t.return_pct:+.2f}%" if t.return_pct is not None else "n/a"
        print(f"  {t.entry_date} @ {t.entry_price:.2f}  ->  {exit_str}  ({ret_str})")


if __name__ == "__main__":
    main()
