"""Backtest a downtrend-reversal / break-of-structure + retest strategy on
Upstox historical candles.

Pattern: downtrend (lower highs, lower lows) -> a higher low prints ->
price breaks above the last lower high -> pulls back to retest that level
-> bounces and continues up -> long entry.

Usage:
    python run_structure_backtest.py --instrument "NSE_EQ|INE002A01018" \
        --from 2023-01-01 --lookback 3
"""
import argparse
from datetime import date

from backtest import structure_reversal
from data_sources import upstox_client


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", default="NSE_EQ|INE002A01018", help="Upstox instrument key")
    parser.add_argument("--from", dest="from_date", default="2023-01-01")
    parser.add_argument("--to", dest="to_date", default=date.today().isoformat())
    parser.add_argument("--lookback", type=int, default=3, help="swing fractal window")
    parser.add_argument("--retest-tolerance", type=float, default=2.0, help="%% above breakout level counted as a retest touch")
    parser.add_argument("--stop-buffer", type=float, default=0.5, help="%% below retest low for the stop-loss")
    parser.add_argument("--reward-risk", type=float, default=2.0)
    parser.add_argument("--capital", type=float, default=100_000.0)
    args = parser.parse_args()

    print(f"Fetching {args.instrument} daily candles from {args.from_date} to {args.to_date}...")
    candles = upstox_client.get_daily_history(args.instrument, args.from_date, args.to_date)
    print(f"Got {len(candles)} candles.\n")

    result = structure_reversal.run(
        candles,
        lookback=args.lookback,
        retest_tolerance_pct=args.retest_tolerance,
        stop_buffer_pct=args.stop_buffer,
        reward_risk=args.reward_risk,
        initial_capital=args.capital,
    )

    print(f"Structure reversal + retest on {args.instrument}")
    print(f"Period: {candles[0]['date']} to {candles[-1]['date']}\n")
    print(result.summary())

    print("\nDetected setups:")
    for s in result.setups:
        print(
            f"  downtrend high {s.downtrend_high_date} @ {s.downtrend_high_price:.2f}"
            f"  ->  higher low {s.higher_low_date} @ {s.higher_low_price:.2f}"
        )
        if s.breakout_date:
            print(f"    breakout {s.breakout_date} @ {s.breakout_price:.2f}")
        if s.retest_date:
            print(f"    retest   {s.retest_date} @ low {s.retest_low:.2f}")
        if s.entry_price:
            exit_str = f"{s.exit_date} @ {s.exit_price:.2f}" if s.exit_price else "OPEN"
            ret_str = f"{s.return_pct:+.2f}%" if s.return_pct is not None else "n/a"
            print(
                f"    entry    {s.entry_date} @ {s.entry_price:.2f}"
                f"  stop {s.stop_loss:.2f}  target {s.target:.2f}"
                f"  ->  exit {exit_str}  ({ret_str})  [{s.outcome}]"
            )
        else:
            print(f"    outcome: {s.outcome}")


if __name__ == "__main__":
    main()
