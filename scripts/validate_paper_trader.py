"""Proves live/ema_sweep_paper_trader.py's per-bar logic (_process_bars)
produces IDENTICAL trades to the validated backtest
(backtest/ema_sweep_breakout_options.run(), SL10/target15, trend filter
on) when fed the same historical data through a "replay" data source
instead of live endpoints. This is what justifies calling the live
engine "the same strategy running live" rather than a parallel
reimplementation that could have quietly drifted from what was actually
backtested.

Run: python3 scripts/validate_paper_trader.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

from data_sources import cache, upstox_client
from backtest import ema_sweep_breakout_options as backtest_mod
from backtest import options_common as oc
from live.ema_sweep_paper_trader import Contract, _process_bars, _default_state

FROM_DATE, TO_DATE = "2026-08-01", "2026-08-20"
UNDERLYING_KEY = backtest_mod.UNDERLYING_KEY


def replay_contract_resolver(index_close, opt_type, as_of_date):
    expiries = sorted(cache.get_expired_expiries_cached(UNDERLYING_KEY, "options"))
    expiry = next((e for e in expiries if e >= as_of_date), None)
    if expiry is None:
        return None
    lookup = oc.build_chain_lookup(cache.get_expired_option_chain_cached(UNDERLYING_KEY, expiry))
    atm = oc.round_to_step(index_close, 50)
    contract = oc.nearest_contract(lookup, atm, opt_type)
    if contract is None:
        return None
    return Contract(contract["instrument_key"], contract["strike_price"], contract["lot_size"], expiry)


def replay_option_candles(instrument_key, as_of_date):
    rows = cache.get_day_candles_cached(instrument_key, "1minute", as_of_date, expired=True)
    return sorted(rows, key=lambda c: c[0])


def main():
    trading_days = upstox_client.get_daily_history(UNDERLYING_KEY, FROM_DATE, TO_DATE)
    all_1min = []
    for day in trading_days:
        rows = sorted(cache.get_day_candles_cached(UNDERLYING_KEY, "1minute", day["date"], expired=False), key=lambda c: c[0])
        all_1min.extend(rows)
    all_1min.sort(key=lambda c: c[0])

    state = _default_state()
    state = _process_bars(all_1min, replay_contract_resolver, replay_option_candles, state, log=lambda m: None)

    import json
    replay_trades = []
    with open("data/paper_trades.jsonl") as f:
        for line in f:
            replay_trades.append(json.loads(line))

    backtest_trades = backtest_mod.run(FROM_DATE, TO_DATE, sl_points=10.0, target_points=15.0, trend_filter=True)

    print(f"Backtest module:  {len(backtest_trades)} trades")
    print(f"Live-engine replay: {len(replay_trades)} trades (appended to data/paper_trades.jsonl this run)")

    bt_keys = [(t.date, t.direction, t.entry_time, round(t.entry_premium, 2), t.exit_time, round(t.exit_premium, 2), t.exit_reason) for t in backtest_trades]
    live_keys = [(t["date"], t["direction"], t["entry_time"], round(t["entry_price"], 2), t["exit_time"], round(t["exit_premium"], 2), t["exit_reason"]) for t in replay_trades]

    if bt_keys == live_keys:
        print("MATCH: every trade identical, in order.")
    else:
        print("MISMATCH -- diverged. First few of each:")
        for i in range(min(5, max(len(bt_keys), len(live_keys)))):
            bt = bt_keys[i] if i < len(bt_keys) else None
            lv = live_keys[i] if i < len(live_keys) else None
            flag = "  <-- DIFF" if bt != lv else ""
            print(f"  [{i}] backtest={bt}")
            print(f"       live    ={lv}{flag}")


if __name__ == "__main__":
    main()
