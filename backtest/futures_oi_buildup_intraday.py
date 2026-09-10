"""Backtest of futures_oi_buildup.py's rules against *today's* session so
far, using Upstox's intraday-candle endpoint.

The regular backtest (futures_oi_buildup.run) and the live paper trader
both hit APIs that only cover completed trading days -- the public
historical-candle endpoint publishes a day's data starting the *next* day,
and the expired-instruments API only knows about contracts/expiries that
have already expired. Neither has anything for the day still in progress.

Upstox's intraday-candle endpoint (get_intraday_candles) is the one source
that does cover today, up to the last completed candle -- this module uses
it for both the futures leg and the option legs (via the currently-listed,
not-yet-expired weekly chain), so "today so far" can be backtested the
same way live_nifty_oi_buildup_trader.py trades it forward.
"""
from __future__ import annotations

import gzip
import json
from datetime import date, datetime, timezone, timedelta
from io import BytesIO

import requests

from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, classify_buildup
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.rsi2_5min_sar import _resample_5min
from data_sources.upstox_client import get_intraday_candles

MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
IST = timezone(timedelta(hours=5, minutes=30))


def _load_master() -> list[dict]:
    resp = requests.get(MASTER_URL, timeout=30)
    resp.raise_for_status()
    with gzip.open(BytesIO(resp.content)) as f:
        return json.load(f)


def _nearest_future(master: list[dict], underlying_symbol: str) -> dict:
    futs = [d for d in master if d.get("segment") == "NSE_FO" and d.get("underlying_symbol") == underlying_symbol
            and d.get("instrument_type") == "FUT"]
    return min(futs, key=lambda d: d["expiry"])


def _option_chain(master: list[dict], underlying_symbol: str) -> tuple[str, dict]:
    opts = [d for d in master if d.get("segment") == "NSE_FO" and d.get("underlying_symbol") == underlying_symbol
            and d.get("instrument_type") in ("CE", "PE")]
    nearest_expiry_ms = min(d["expiry"] for d in opts)
    chain = [d for d in opts if d["expiry"] == nearest_expiry_ms]
    expiry_str = datetime.utcfromtimestamp(nearest_expiry_ms / 1000).strftime("%Y-%m-%d")
    return expiry_str, oc.build_chain_lookup(chain)


def run_today(
    underlying_symbol: str = "NIFTY",
    strike_step: int = 50,
) -> list[OptionTrade]:
    today = date.today().isoformat()
    master = _load_master()
    fut = _nearest_future(master, underlying_symbol)
    expiry, chain_lookup = _option_chain(master, underlying_symbol)

    fut_candles = get_intraday_candles(fut["instrument_key"], "1minute")
    rows_1min = sorted(fut_candles, key=lambda c: c[0])
    all_5min = _resample_5min(rows_1min)
    if len(all_5min) < 2:
        return []

    option_candle_cache: dict[str, list[list]] = {}

    def _option_candles(contract: dict) -> list[list]:
        key = contract["instrument_key"]
        if key not in option_candle_cache:
            option_candle_cache[key] = sorted(get_intraday_candles(key, "1minute"), key=lambda c: c[0])
        return option_candle_cache[key]

    trades: list[OptionTrade] = []
    position = None  # dict: direction, entry_time, entry_price, strike, opt_type, lot_size
    prev_close = None
    prev_oi = None

    for row in all_5min:
        ts, o, h, l, c, v, oi = row
        time_str = ts[11:16]

        if prev_close is None:
            prev_close, prev_oi = c, oi
            continue

        buildup = classify_buildup(c - prev_close, oi - prev_oi)
        atm = oc.round_to_step(c, strike_step)

        def _enter(direction: str) -> None:
            nonlocal position
            opt_type = "CE" if direction == "LONG" else "PE"
            contract = oc.nearest_contract(chain_lookup, atm, opt_type)
            if contract is None:
                return
            candles = _option_candles(contract)
            bar = next((r for r in reversed(candles) if r[0][11:16] <= time_str), None) or (candles[0] if candles else None)
            if bar is None:
                return
            position = {
                "direction": direction, "entry_time": bar[0], "entry_price": bar[4],
                "strike": contract["strike_price"], "opt_type": opt_type,
                "lot_size": contract["lot_size"], "contract": contract,
            }

        def _exit(reason: str) -> None:
            nonlocal position
            if position is None:
                return
            candles = _option_candles(position["contract"])
            bar = next((r for r in reversed(candles) if r[0][11:16] <= time_str), None) or (candles[0] if candles else None)
            exit_price = bar[4] if bar else position["entry_price"]
            exit_time = bar[0] if bar else ts
            trades.append(OptionTrade(
                date=today, direction=position["direction"], expiry=expiry,
                strike=position["strike"], entry_time=position["entry_time"], entry_premium=position["entry_price"],
                exit_time=exit_time, exit_premium=exit_price, lot_size=position["lot_size"], exit_reason=reason,
            ))
            position = None

        if position is None:
            if buildup == "Long Buildup":
                _enter("LONG")
            elif buildup == "Short Buildup":
                _enter("SHORT")
        else:
            if time_str >= FORCE_FLAT_TIME:
                _exit("eod")
            elif position["direction"] == "LONG" and buildup == "Short Buildup":
                _exit("reverse")
                _enter("SHORT")
            elif position["direction"] == "LONG" and buildup == "Long Unwinding":
                _exit("unwind")
            elif position["direction"] == "SHORT" and buildup == "Long Buildup":
                _exit("reverse")
                _enter("LONG")
            elif position["direction"] == "SHORT" and buildup == "Short Covering":
                _exit("covering")

        prev_close, prev_oi = c, oi

    if position is not None:
        candles = _option_candles(position["contract"])
        last_bar = candles[-1] if candles else None
        exit_price = last_bar[4] if last_bar else position["entry_price"]
        exit_time = last_bar[0] if last_bar else all_5min[-1][0]
        trades.append(OptionTrade(
            date=today, direction=position["direction"], expiry=expiry,
            strike=position["strike"], entry_time=position["entry_time"], entry_premium=position["entry_price"],
            exit_time=exit_time, exit_premium=exit_price, lot_size=position["lot_size"], exit_reason="in_progress",
        ))

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
