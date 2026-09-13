"""Backtest a daily intraday call butterfly on NIFTY 50 (or any underlying):
enter ~09:15, square off all legs at market close, every trading day.

    buy  1x lower call   (ATM - wing_width)
    sell 2x middle call  (ATM)
    buy  1x upper call   (ATM + wing_width)

Requires an Upstox access token (the expired-instruments API), since by the
time this runs every expiry involved is in the past:

    export UPSTOX_ACCESS_TOKEN=...
    python run_call_butterfly_backtest.py --from 2026-05-01 --to 2026-09-01 --wing-width 100
"""
import argparse

from backtest import call_butterfly
from backtest.iron_condor import UNDERLYING_KEY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--underlying", default=UNDERLYING_KEY, help="Upstox instrument key, default NIFTY 50 index")
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--to", dest="to_date", required=True)
    parser.add_argument("--wing-width", type=int, default=100, help="points each wing is from ATM")
    parser.add_argument("--strike-step", type=int, default=50)
    parser.add_argument("--max-dte", type=int, default=None, help="only enter within this many days of expiry")
    args = parser.parse_args()

    print(f"Running daily call butterfly backtest {args.from_date} to {args.to_date} on {args.underlying}...")
    print(f"Wing width: {args.wing_width}pt\n")

    results = call_butterfly.run(
        args.from_date,
        args.to_date,
        wing_width=args.wing_width,
        strike_step=args.strike_step,
        underlying_key=args.underlying,
        max_dte=args.max_dte,
    )

    print(call_butterfly.summary(results))
    print("\nDaily detail:")
    for r in results:
        if not r.ok:
            print(f"  {r.date}  SKIPPED ({r.note})")
            continue
        strikes = " / ".join(f"{leg.role}={leg.strike:.0f}" for leg in r.legs)
        print(
            f"  {r.date}  expiry={r.expiry}  spot={r.spot_915:.2f}  {strikes}  "
            f"debit={r.entry_debit:.2f}  gross=Rs{r.pnl_rupees_gross:>9,.2f}  "
            f"costs=Rs{r.costs_rupees:>7,.2f}  net=Rs{r.pnl_rupees:>9,.2f}"
        )


if __name__ == "__main__":
    main()
