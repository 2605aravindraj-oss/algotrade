"""Command-line entry point: fetch Upstox historical data and/or run a backtest.

Examples:
    python -m algotrade.cli backtest --symbol RELIANCE --start 2023-01-01 --end 2024-01-01 \\
        --strategy sma_crossover --fast 20 --slow 50

    python -m algotrade.cli fetch-data --symbol INFY --exchange NSE_EQ \\
        --start 2023-01-01 --end 2024-01-01
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from algotrade.backtest.engine import run_backtest
from algotrade.data import get_historical_data
from algotrade.strategies import STRATEGY_REGISTRY


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _build_strategy(args: argparse.Namespace):
    strategy_cls = STRATEGY_REGISTRY[args.strategy]
    if args.strategy == "sma_crossover":
        return strategy_cls(fast=args.fast, slow=args.slow)
    if args.strategy == "rsi":
        return strategy_cls(period=args.rsi_period, oversold=args.oversold, exit_level=args.exit_level)
    return strategy_cls()


def _add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbol", required=True, help="Trading symbol, e.g. RELIANCE")
    parser.add_argument("--exchange", default="NSE_EQ", help="Exchange segment, e.g. NSE_EQ, BSE_EQ, NSE_FO")
    parser.add_argument("--interval", default="day", choices=["1minute", "30minute", "day", "week", "month"])
    parser.add_argument("--start", required=True, type=_parse_date)
    parser.add_argument("--end", required=True, type=_parse_date)
    parser.add_argument("--no-cache", action="store_true", help="Bypass the local data cache")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="algotrade")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch-data", help="Download and cache historical candles")
    _add_common_data_args(fetch_parser)

    backtest_parser = subparsers.add_parser("backtest", help="Run a strategy backtest")
    _add_common_data_args(backtest_parser)
    backtest_parser.add_argument("--strategy", required=True, choices=list(STRATEGY_REGISTRY))
    backtest_parser.add_argument("--capital", type=float, default=100_000.0)
    backtest_parser.add_argument("--commission-bps", type=float, default=0.0)
    backtest_parser.add_argument("--fast", type=int, default=20, help="sma_crossover: fast SMA period")
    backtest_parser.add_argument("--slow", type=int, default=50, help="sma_crossover: slow SMA period")
    backtest_parser.add_argument("--rsi-period", type=int, default=14, help="rsi: lookback period")
    backtest_parser.add_argument("--oversold", type=float, default=30, help="rsi: entry threshold")
    backtest_parser.add_argument("--exit-level", type=float, default=50, help="rsi: exit threshold")
    backtest_parser.add_argument("--plot", metavar="PATH", help="Save an equity curve chart to PATH (e.g. equity.png)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    df = get_historical_data(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        exchange=args.exchange,
        interval=args.interval,
        use_cache=not args.no_cache,
    )

    if args.command == "fetch-data":
        print(f"Fetched {len(df)} candles for {args.symbol} ({args.exchange}, {args.interval}).")
        print(df.tail())
        return 0

    strategy = _build_strategy(args)
    result = run_backtest(
        df,
        strategy,
        initial_capital=args.capital,
        commission_bps=args.commission_bps,
    )

    print(json.dumps(result.performance.as_dict(), indent=2))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        result.equity_curve.plot(ax=ax, title=f"{args.symbol} — {args.strategy} equity curve")
        ax.set_ylabel("Equity")
        fig.tight_layout()
        fig.savefig(args.plot)
        print(f"Saved equity curve chart to {args.plot}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
