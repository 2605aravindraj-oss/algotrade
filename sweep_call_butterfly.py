"""Sweep the call butterfly's wing width over a date range and compare
results, to see which width performs best (both raw P&L and risk-adjusted).

Usage:
    export UPSTOX_ACCESS_TOKEN=...
    python sweep_call_butterfly.py --from 2026-05-01 --to 2026-09-01 \
        --widths 50 100 150 200 250 300
"""
import argparse
import statistics

from backtest import call_butterfly
from backtest.iron_condor import UNDERLYING_KEY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--underlying", default=UNDERLYING_KEY)
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--to", dest="to_date", required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[50, 100, 150, 200, 250, 300])
    parser.add_argument("--max-dte", type=int, default=None)
    args = parser.parse_args()

    rows = []
    for width in args.widths:
        print(f"Running wing_width={width}...")
        results = call_butterfly.run(
            args.from_date, args.to_date,
            wing_width=width,
            underlying_key=args.underlying,
            max_dte=args.max_dte,
        )
        ok = [r for r in results if r.ok]
        if not ok:
            print(f"  no complete days")
            continue

        pnls = [r.pnl_rupees for r in ok]
        gross = sum(r.pnl_rupees_gross for r in ok)
        costs_total = sum(r.costs_rupees for r in ok)
        net = sum(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(ok) * 100

        cum = 0.0
        equity = []
        for p in pnls:
            cum += p
            equity.append(cum)
        running_max = equity[0]
        max_dd = 0.0
        for e in equity:
            running_max = max(running_max, e)
            max_dd = min(max_dd, e - running_max)

        mean_pnl = statistics.mean(pnls)
        std_pnl = statistics.stdev(pnls) if len(pnls) > 1 else 0.0
        sharpe_like = mean_pnl / std_pnl if std_pnl else 0.0
        return_over_dd = net / abs(max_dd) if max_dd else float("inf")

        rows.append({
            "width": width, "days": len(ok), "gross": gross, "costs": costs_total,
            "net": net, "win_rate": win_rate,
            "avg_win": sum(wins) / len(wins) if wins else 0,
            "avg_loss": sum(losses) / len(losses) if losses else 0,
            "max_dd": max_dd, "sharpe_like": sharpe_like, "return_over_dd": return_over_dd,
        })

    print()
    print(f"{'Width':>6} {'Days':>5} {'Net P&L':>12} {'WinRate':>8} {'AvgWin':>10} {'AvgLoss':>10} "
          f"{'MaxDD':>10} {'Sharpe':>7} {'Net/|DD|':>9}")
    for r in rows:
        print(
            f"{r['width']:>6} {r['days']:>5} {r['net']:>12,.2f} {r['win_rate']:>7.1f}% "
            f"{r['avg_win']:>10,.2f} {r['avg_loss']:>10,.2f} {r['max_dd']:>10,.2f} "
            f"{r['sharpe_like']:>7.2f} {r['return_over_dd']:>9.2f}"
        )

    if rows:
        best_net = max(rows, key=lambda r: r["net"])
        best_risk = max(rows, key=lambda r: r["return_over_dd"])
        best_sharpe = max(rows, key=lambda r: r["sharpe_like"])
        print()
        print(f"Best by net P&L:       width={best_net['width']}  (Rs {best_net['net']:,.2f})")
        print(f"Best by net/|maxDD|:   width={best_risk['width']}  (ratio {best_risk['return_over_dd']:.2f})")
        print(f"Best by Sharpe-like:   width={best_sharpe['width']}  (ratio {best_sharpe['sharpe_like']:.2f})")


if __name__ == "__main__":
    main()
