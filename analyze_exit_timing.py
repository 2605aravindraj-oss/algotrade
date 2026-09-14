"""Quantify how much an "early close" (sampling exit price a few minutes
before actual market close) distorts the iron condor backtest, and whether
the effect concentrates on 0DTE (same-day expiry) days.

For each trading day, fetches each leg's full 1-minute candle series once
and computes P&L two ways: using the true 15:30 close, and using the last
bar at/before an earlier cutoff (default 15:15) -- mimicking what the
external app's exit price appeared to reflect.

Usage:
    export UPSTOX_ACCESS_TOKEN=...
    python analyze_exit_timing.py --from 2026-05-01 --to 2026-09-01
"""
from __future__ import annotations

import argparse

from backtest import options_common as oc
from backtest.iron_condor import UNDERLYING_KEY
from data_sources import cache, upstox_client


def run(from_date: str, to_date: str, short_distance: int, wing_width: int, early_time: str):
    trading_days = upstox_client.get_daily_history(UNDERLYING_KEY, from_date, to_date)
    expiries = sorted(cache.get_expired_expiries_cached(UNDERLYING_KEY, "options"))
    chain_cache: dict[str, dict] = {}
    rows = []

    for day in trading_days:
        d = day["date"]
        expiry = next((e for e in expiries if e >= d), None)
        if expiry is None:
            continue

        spot_candles = cache.get_day_candles_cached(UNDERLYING_KEY, "1minute", d, expired=False)
        entry_bar = oc.nearest_bar(spot_candles, "first")
        if entry_bar is None:
            continue
        atm = oc.round_to_step(entry_bar[1], 50)

        if expiry not in chain_cache:
            chain_cache[expiry] = oc.build_chain_lookup(
                cache.get_expired_option_chain_cached(UNDERLYING_KEY, expiry)
            )
        lookup = chain_cache[expiry]

        wanted = [
            ("short_call", atm + short_distance, "CE"),
            ("short_put", atm - short_distance, "PE"),
            ("long_call", atm + short_distance + wing_width, "CE"),
            ("long_put", atm - short_distance - wing_width, "PE"),
        ]
        legs = []
        missing = False
        for role, strike, opt_type in wanted:
            contract = oc.nearest_contract(lookup, strike, opt_type)
            if contract is None:
                missing = True
                break
            legs.append((role, contract))
        if missing:
            continue

        leg_data = {}
        incomplete = False
        for role, contract in legs:
            candles = cache.get_day_candles_cached(
                contract["instrument_key"], "1minute", d, expired=True
            )
            entry = oc.nearest_bar(candles, "first")
            exit_true = oc.nearest_bar(candles, "last")
            # temporarily shrink the close cutoff for the "early" read
            orig_close = oc.MARKET_CLOSE
            oc.MARKET_CLOSE = early_time
            exit_early = oc.nearest_bar(candles, "last")
            oc.MARKET_CLOSE = orig_close
            if entry is None or exit_true is None or exit_early is None:
                incomplete = True
                continue
            leg_data[role] = {
                "lot_size": contract["lot_size"],
                "entry": entry[1],
                "exit_true": exit_true[1],
                "exit_early": exit_early[1],
            }
        if incomplete or len(leg_data) != 4:
            continue

        def pnl(exit_key):
            credit = (
                leg_data["short_call"]["entry"] + leg_data["short_put"]["entry"]
                - leg_data["long_call"]["entry"] - leg_data["long_put"]["entry"]
            )
            debit = (
                leg_data["short_call"][exit_key] + leg_data["short_put"][exit_key]
                - leg_data["long_call"][exit_key] - leg_data["long_put"][exit_key]
            )
            return (credit - debit) * leg_data["short_call"]["lot_size"]

        rows.append({
            "date": d,
            "expiry": expiry,
            "dte": (__import__("datetime").date.fromisoformat(expiry) - __import__("datetime").date.fromisoformat(d)).days,
            "pnl_true": pnl("exit_true"),
            "pnl_early": pnl("exit_early"),
        })

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--to", dest="to_date", required=True)
    parser.add_argument("--short-distance", type=int, default=50)
    parser.add_argument("--wing-width", type=int, default=200)
    parser.add_argument("--early-time", default="15:15")
    args = parser.parse_args()

    rows = run(args.from_date, args.to_date, args.short_distance, args.wing_width, args.early_time)

    print(f"{'date':10s} {'exp':10s} {'dte':>3s} {'true':>10s} {'early':>10s} {'diff':>10s} {'flip':>5s}")
    n_flip = 0
    n_flip_0dte = 0
    n_0dte = 0
    diffs = []
    diffs_0dte = []
    for r in rows:
        diff = r["pnl_early"] - r["pnl_true"]
        flip = (r["pnl_true"] > 0) != (r["pnl_early"] > 0)
        diffs.append(diff)
        if r["dte"] == 0:
            n_0dte += 1
            diffs_0dte.append(diff)
        if flip:
            n_flip += 1
            if r["dte"] == 0:
                n_flip_0dte += 1
        print(f"{r['date']:10s} {r['expiry']:10s} {r['dte']:>3d} {r['pnl_true']:>10,.2f} {r['pnl_early']:>10,.2f} {diff:>10,.2f} {'YES' if flip else '':>5s}")

    print()
    print(f"Total days: {len(rows)}  (0DTE: {n_0dte})")
    print(f"Sign flips: {n_flip} ({n_flip/len(rows)*100:.1f}%)  |  0DTE flips: {n_flip_0dte}/{n_0dte}")
    print(f"Avg |diff| all days: Rs {sum(abs(d) for d in diffs)/len(diffs):,.2f}")
    if diffs_0dte:
        print(f"Avg |diff| 0DTE days: Rs {sum(abs(d) for d in diffs_0dte)/len(diffs_0dte):,.2f}")
    non_0dte = [d for r, d in zip(rows, diffs) if r["dte"] != 0]
    if non_0dte:
        print(f"Avg |diff| non-0DTE days: Rs {sum(abs(d) for d in non_0dte)/len(non_0dte):,.2f}")
    print(f"Sum true P&L: Rs {sum(r['pnl_true'] for r in rows):,.2f}")
    print(f"Sum early P&L: Rs {sum(r['pnl_early'] for r in rows):,.2f}")


if __name__ == "__main__":
    main()
