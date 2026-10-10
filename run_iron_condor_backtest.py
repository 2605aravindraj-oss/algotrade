"""Backtest a daily intraday NIFTY 50 iron condor: enter ~09:15, square off
all four legs at market close, every trading day.

Requires an Upstox access token (the expired-instruments API), since by the
time this runs every expiry involved is in the past:

    export UPSTOX_ACCESS_TOKEN=...
    python run_iron_condor_backtest.py --from 2025-07-01 --to 2025-08-29

Strikes: short call/put at spot +/- --short-distance points (rounded to the
nearest strike step), long call/put (protection) --wing-width points beyond
those.
"""
import argparse

from backtest import iron_condor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--to", dest="to_date", required=True)
    parser.add_argument("--short-distance", type=int, default=150, help="points OTM for the short strikes")
    parser.add_argument("--wing-width", type=int, default=100, help="points beyond the short strikes for the long wings")
    parser.add_argument("--strike-step", type=int, default=50)
    args = parser.parse_args()

    print(f"Running daily iron condor backtest {args.from_date} to {args.to_date}...")
    print(f"Short distance: {args.short_distance}pt, wing width: {args.wing_width}pt\n")

    results = iron_condor.run(
        args.from_date,
        args.to_date,
        short_distance=args.short_distance,
        wing_width=args.wing_width,
        strike_step=args.strike_step,
    )

    print(iron_condor.summary(results))
    print("\nDaily detail:")
    for r in results:
        if not r.ok:
            print(f"  {r.date}  SKIPPED ({r.note})")
            continue
        strikes = " / ".join(f"{leg.role}={leg.strike:.0f}" for leg in r.legs)
        print(
            f"  {r.date}  expiry={r.expiry}  spot={r.spot_915:.2f}  {strikes}  "
            f"gross=Rs{r.pnl_rupees_gross:>9,.2f}  costs=Rs{r.costs_rupees:>7,.2f}  net=Rs{r.pnl_rupees:>9,.2f}"
        )


if __name__ == "__main__":
    main()
