"""Run the daily intraday iron condor backtest across multiple NSE stocks'
options and compare results.

Individual stock options in India are monthly-only (unlike NIFTY's weekly
expiry), so "nearest expiry" here can be anywhere from 0 to ~30 days out
depending on where in the expiry cycle each day falls -- a materially
different risk profile than the NIFTY weekly version (theta decay per day
is much smaller most of the month; moves are driven more by delta/vega).

Strikes are chosen in units of strikes-from-spot (auto-detected per stock,
since strike spacing varies a lot by price -- unlike NIFTY's flat 50).

Usage:
    export UPSTOX_ACCESS_TOKEN=...
    python scan_iron_condor.py --from 2026-05-01 --to 2026-09-01 \
        --short-distance-strikes 1 --wing-width-strikes 4
    python scan_iron_condor.py TCS INFY RELIANCE --from 2026-05-01 --to 2026-09-01
"""
import argparse

from backtest import iron_condor
from data_sources import instruments

DEFAULT_BASKET = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "ITC", "LT",
    "AXISBANK", "WIPRO", "MARUTI", "SUNPHARMA", "HINDUNILVR", "KOTAKBANK",
    "BAJFINANCE",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="*", default=DEFAULT_BASKET)
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--to", dest="to_date", required=True)
    parser.add_argument("--short-distance-strikes", type=int, default=1)
    parser.add_argument("--wing-width-strikes", type=int, default=4)
    args = parser.parse_args()

    print(f"Resolving {len(args.symbols)} symbols...")
    keys = instruments.resolve_symbols(args.symbols)
    missing = [s for s, k in keys.items() if k is None]
    if missing:
        print(f"  could not resolve: {', '.join(missing)}")

    print(f"{'Symbol':<12} {'Days':>5} {'Gross':>12} {'Costs':>10} {'Net':>12} {'WinRate':>8} {'MaxDD':>10}")
    all_results = {}
    for symbol in args.symbols:
        key = keys.get(symbol)
        if key is None:
            continue
        try:
            results = iron_condor.run(
                args.from_date,
                args.to_date,
                short_distance_strikes=args.short_distance_strikes,
                wing_width_strikes=args.wing_width_strikes,
                underlying_key=key,
            )
            all_results[symbol] = results
            ok = [r for r in results if r.ok]
            if not ok:
                notes = {r.note for r in results if not r.ok}
                print(f"{symbol:<12}  no complete trading days ({', '.join(notes) or 'unknown'})")
                continue
            gross = sum(r.pnl_rupees_gross for r in ok)
            costs_total = sum(r.costs_rupees for r in ok)
            net = sum(r.pnl_rupees for r in ok)
            wins = sum(1 for r in ok if r.pnl_rupees > 0)
            cum = 0.0
            running_max = 0.0
            max_dd = 0.0
            for r in ok:
                cum += r.pnl_rupees
                running_max = max(running_max, cum)
                max_dd = min(max_dd, cum - running_max)
            print(
                f"{symbol:<12} {len(ok):>5} {gross:>12,.2f} {costs_total:>10,.2f} "
                f"{net:>12,.2f} {wins/len(ok)*100:>7.1f}% {max_dd:>10,.2f}"
            )
        except Exception as exc:
            print(f"{symbol:<12}  error: {exc}")

    import pickle
    with open("/tmp/scan_iron_condor_results.pkl", "wb") as f:
        pickle.dump(all_results, f)


if __name__ == "__main__":
    main()
