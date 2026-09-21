"""Live paper-trading engine for the EMA9/EMA20 sweep-and-breakout
strategy (backtest/ema_sweep_breakout_options.py: 15-min EMA, 1-min
pattern candle, SL10/target15 on option premium, trend filter on).
Every fill is SIMULATED -- no order is ever placed, nothing here touches
a real broker account or real capital, and no authenticated Upstox call
is made anywhere in this file (none is needed, and no access token is
configured in this environment anyway).

Backtested result for this exact config over 2026-05-16 to 2026-09-08:
405 trades, 39.5% win rate, net P&L -Rs 12,193 at 1 lot (gross P&L was
positive, +Rs 12,104 -- it's transaction costs that make it net
negative at this size). This is being run live to see the same signal
against real market data, not because it has a demonstrated edge.

The per-bar signal/stop/target logic in `_process_bars` is written to
be swappable between LIVE data and REPLAY of historical data (see
`live_data_source` / `replay_data_source` below and
scripts/validate_paper_trader.py), specifically so it can be checked
bar-for-bar against backtest.ema_sweep_breakout_options.run() on a
known historical window before ever being trusted against live prices.
That check is what justifies calling this "the same strategy running
live" rather than a reimplementation that might have quietly drifted
from what was actually backtested.

LIVE data sources (all public, no auth):
    NIFTY 50 index candles: upstox_client.get_intraday_candles
        ("today so far", no auth -- this IS the exchange feed, not a
        derived/delayed source).
    Live NIFTY option instrument keys: Upstox's public instrument
        master (assets.upstox.com/.../NSE.json.gz), filtered to NIFTY
        weekly options; the nearest not-yet-expired weekly is picked,
        ATM by the underlying's current price. The option's own
        premium candles come from the same get_intraday_candles call,
        applied to that contract's instrument_key.

State (open pattern, open paper position, last-processed bar) persists
to data/paper_trader_state.json across restarts -- resuming after a
restart just continues from the last bar it saw, it does not replay
missed bars. Completed paper trades append to data/paper_trades.jsonl,
one JSON object per line.
"""
from __future__ import annotations

import gzip
import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Optional

import requests

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, _bar_at_or_after, _bar_at_or_before
from backtest.ema_sweep_breakout_options import _ema, _align_ema
from backtest.sweep_reclaim_breakout import _resample

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"
STRIKE_STEP = 50
EMA_FAST, EMA_SLOW, EMA_BAR_MINUTES = 9, 20, 15
CANDLE_MINUTES = 1
SL_POINTS, TARGET_POINTS = 10.0, 15.0  # option-premium points, matching the backtested config

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_PATH = os.path.join(_REPO_ROOT, "data", "paper_trader_state.json")
TRADE_LOG_PATH = os.path.join(_REPO_ROOT, "data", "paper_trades.jsonl")
INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
_IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class Contract:
    instrument_key: str
    strike_price: float
    lot_size: int
    expiry: str


DataSource = Callable[[str, str], list[list]]  # (instrument_key, date) -> sorted 1-min candles
ContractResolver = Callable[[float, str, str], Optional[Contract]]  # (index_close, opt_type, date) -> Contract


# ---------------------------------------------------------------- state --

def _default_state() -> dict:
    return {"pattern": None, "position": None, "current_day": None, "last_processed_ts": None, "trade_count": 0}


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return _default_state()


def save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def log_trade(trade: dict, path: str = TRADE_LOG_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(trade) + "\n")


# ------------------------------------------------------ live data glue --

_nifty_weekly_chain_cache: list[dict] | None = None


def _load_nifty_weekly_chain() -> list[dict]:
    global _nifty_weekly_chain_cache
    if _nifty_weekly_chain_cache is not None:
        return _nifty_weekly_chain_cache
    resp = requests.get(INSTRUMENTS_URL, timeout=30)
    resp.raise_for_status()
    data = json.loads(gzip.decompress(resp.content))
    _nifty_weekly_chain_cache = [
        d for d in data
        if d.get("name") == "NIFTY" and d.get("instrument_type") in ("CE", "PE") and d.get("weekly")
    ]
    return _nifty_weekly_chain_cache


def live_contract_resolver(index_close: float, opt_type: str, as_of_date: str) -> Optional[Contract]:
    """Nearest unexpired weekly NIFTY expiry, ATM strike by index_close."""
    chain = _load_nifty_weekly_chain()
    now_ms = time.time() * 1000
    future = sorted(set(d["expiry"] for d in chain if d["expiry"] > now_ms))
    if not future:
        return None
    nearest = future[0]
    expiry_date = datetime.fromtimestamp(nearest / 1000, tz=timezone.utc).astimezone(_IST).date().isoformat()
    atm = oc.round_to_step(index_close, STRIKE_STEP)
    for d in chain:
        if d["expiry"] == nearest and d["strike_price"] == atm and d["instrument_type"] == opt_type:
            return Contract(d["instrument_key"], d["strike_price"], d["lot_size"], expiry_date)
    return None


def live_option_candles(instrument_key: str, as_of_date: str) -> list[list]:
    rows = upstox_client.get_intraday_candles(instrument_key, "1minute")
    return sorted(rows, key=lambda c: c[0])


def live_index_candles(days_back: int = 15) -> list[list]:
    """Prior trading days (cached, immutable) + today's live candles
    (never cached -- today is still in progress)."""
    end = date.today()
    start = end - timedelta(days=days_back)
    trading_days = upstox_client.get_daily_history(UNDERLYING_KEY, start.isoformat(), (end - timedelta(days=1)).isoformat())
    rows: list[list] = []
    for day in trading_days:
        rows.extend(cache.get_day_candles_cached(UNDERLYING_KEY, "1minute", day["date"], expired=False))
    rows.extend(upstox_client.get_intraday_candles(UNDERLYING_KEY, "1minute"))
    rows.sort(key=lambda c: c[0])
    return rows


# ------------------------------------------------- core bar-by-bar step --

def _process_bars(
    all_1min: list[list],
    contract_resolver: ContractResolver,
    option_candles: DataSource,
    state: dict,
    log: Callable[[str], None] = print,
) -> dict:
    """Replays every bar in all_1min strictly after state['last_processed_ts'],
    mutating and returning state. Identical logic to
    ema_sweep_breakout_options.run()'s per-bar loop (day rollover, EOD
    flatten, stop/target check, sweep detection, trend-filtered breakout
    entry, decision-time-correct fills) -- verified to match it bar-for-bar
    in scripts/validate_paper_trader.py. `contract_resolver` and
    `option_candles` are the only points where live vs. historical data
    differs; this function itself doesn't know which it's looking at.
    """
    if len(all_1min) < EMA_SLOW + 2:
        return state

    bars15 = _resample(all_1min, EMA_BAR_MINUTES)
    if len(bars15) < EMA_SLOW + 2:
        return state
    closes15 = [b[4] for b in bars15]
    ema9_15 = _ema(closes15, EMA_FAST)
    ema20_15 = _ema(closes15, EMA_SLOW)
    ema9 = _align_ema(all_1min, bars15, ema9_15, EMA_BAR_MINUTES)
    ema20 = _align_ema(all_1min, bars15, ema20_15, EMA_BAR_MINUTES)

    last_ts = state.get("last_processed_ts")
    pattern = state.get("pattern")
    position = state.get("position")
    current_day = state.get("current_day")
    trade_count = state.get("trade_count", 0)

    opt_cache: dict[tuple[str, str], list[list]] = {}

    def _opt(instrument_key: str, as_of_date: str) -> list[list]:
        key = (instrument_key, as_of_date)
        if key not in opt_cache:
            opt_cache[key] = option_candles(instrument_key, as_of_date)
        return opt_cache[key]

    def _fill(instrument_key: str, as_of_date: str, at_time: str):
        candles = _opt(instrument_key, as_of_date)
        return _bar_at_or_after(candles, at_time) or _bar_at_or_before(candles, at_time)

    def _close_position(reason: str, fill_time_str: str, ts: str) -> None:
        nonlocal position, trade_count
        bar = _fill(position["instrument_key"], position["date"], fill_time_str)
        exit_price = bar[4] if bar else position["entry_price"]
        exit_time = bar[0] if bar else ts
        trade = dict(position, exit_time=exit_time, exit_premium=exit_price, exit_reason=reason)
        log_trade(trade)
        trade_count += 1
        pnl = (exit_price - position["entry_price"]) * position["lot_size"]
        log(f"PAPER EXIT  {position['direction']:5s} {position['opt_type']} {position['strike']:.0f} "
            f"@ {exit_price:.2f}  reason={reason}  pnl=Rs{pnl:+.2f}")
        position = None

    for i, row in enumerate(all_1min):
        ts, o, h, l, c, v, oi = row
        if last_ts is not None and ts <= last_ts:
            continue
        d = ts[:10]
        time_str = ts[11:16]

        if d != current_day:
            current_day = d
            pattern = None
            position = None

        _dh, _dm = divmod(int(time_str[:2]) * 60 + int(time_str[3:5]) + CANDLE_MINUTES, 60)
        decision_time_str = f"{_dh:02d}:{_dm:02d}"

        if time_str >= FORCE_FLAT_TIME:
            if position is not None:
                _close_position("eod", time_str, ts)
            last_ts = ts
            continue

        if position is not None:
            opt_bar = _bar_at_or_before(_opt(position["instrument_key"], position["date"]), time_str)
            if opt_bar is not None:
                oh, ol = opt_bar[2], opt_bar[3]
                if ol <= position["stop_level"]:
                    _close_position("stop_loss", decision_time_str, ts)
                elif oh >= position["target_level"]:
                    _close_position("target", decision_time_str, ts)

        e9, e20 = ema9[i], ema20[i]
        swept = any(
            ema_val is not None and ((h > ema_val and c <= ema_val) or (l < ema_val and c >= ema_val))
            for ema_val in (e9, e20)
        )
        if swept:
            pattern = {"high": h, "low": l}
            last_ts = ts
            continue

        if position is None and pattern is not None:
            direction_label = None
            if h > pattern["high"]:
                if e9 is not None and e20 is not None and e9 > e20:
                    direction_label = "LONG"
                pattern = None
            elif l < pattern["low"]:
                if e9 is not None and e20 is not None and e9 < e20:
                    direction_label = "SHORT"
                pattern = None

            if direction_label is not None:
                opt_type = "CE" if direction_label == "LONG" else "PE"
                contract = contract_resolver(c, opt_type, d)
                if contract is not None:
                    bar = _fill(contract.instrument_key, d, decision_time_str)
                    if bar is not None:
                        entry_price = bar[4]
                        position = {
                            "direction": direction_label, "entry_time": bar[0], "entry_price": entry_price,
                            "strike": contract.strike_price, "expiry": contract.expiry,
                            "lot_size": contract.lot_size, "opt_type": opt_type,
                            "instrument_key": contract.instrument_key, "date": d,
                            "stop_level": entry_price - SL_POINTS, "target_level": entry_price + TARGET_POINTS,
                        }
                        log(f"PAPER ENTRY {direction_label:5s} {opt_type} {contract.strike_price:.0f} "
                            f"@ {entry_price:.2f}  stop={position['stop_level']:.2f}  target={position['target_level']:.2f}")

        last_ts = ts

    state["pattern"] = pattern
    state["position"] = position
    state["current_day"] = current_day
    state["last_processed_ts"] = last_ts
    state["trade_count"] = trade_count
    return state


# ----------------------------------------------------------- live loop --

def poll_once(state: dict, log: Callable[[str], None] = print) -> dict:
    all_1min = live_index_candles()
    if state.get("last_processed_ts") is None:
        # Cold start: the warm-up history is for the EMA calculation only.
        # live_option_candles always returns "today so far" regardless of
        # what date is asked for, so replaying prior days through it would
        # produce nonsense fills -- skip straight to today's first bar.
        today_str = date.today().isoformat()
        prior = [row for row in all_1min if row[0][:10] < today_str]
        if prior:
            state["last_processed_ts"] = prior[-1][0]
            log(f"Cold start: skipping {len(prior)} warm-up bars before {today_str}, EMA still uses them.")
    state = _process_bars(all_1min, live_contract_resolver, live_option_candles, state, log)
    save_state(state)
    return state


def run_live(poll_seconds: int = 30, max_polls: int | None = None, log: Callable[[str], None] = print) -> None:
    """Poll live NIFTY data every poll_seconds and simulate the strategy.
    Runs until the market's forced-flat time has passed for today, or
    max_polls is reached (for a bounded test run)."""
    state = load_state()
    polls = 0
    while True:
        now_ist = datetime.now(_IST)
        time_str = now_ist.strftime("%H:%M")
        try:
            state = poll_once(state, log)
        except Exception as exc:
            log(f"poll error (will retry next cycle): {exc}")
        polls += 1
        pos = state.get("position")
        pat = state.get("pattern")
        log(f"[{now_ist.strftime('%H:%M:%S')}] poll #{polls}  "
            f"position={'FLAT' if pos is None else pos['direction']+' '+pos['opt_type']}  "
            f"pattern={'none' if pat is None else 'pending'}  trades_today={state.get('trade_count', 0)}")
        if time_str >= FORCE_FLAT_TIME and pos is None:
            log("Past forced-flat time and flat -- stopping for today.")
            break
        if max_polls is not None and polls >= max_polls:
            log(f"Reached max_polls={max_polls}, stopping.")
            break
        time.sleep(poll_seconds)


if __name__ == "__main__":
    run_live()
