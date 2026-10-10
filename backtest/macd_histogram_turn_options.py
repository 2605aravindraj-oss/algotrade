"""Backtest: NIFTY 50 spot candles (5-minute by default), buy ATM CE on
a bearish-momentum-fading turn and ATM PE on a bullish-momentum-fading
turn, through real ATM NIFTY options, with a fixed rupee stop-loss/
target.

SIGNAL (the user's own rule, taken literally -- no divergence, no
peaks/troughs, just one bar's histogram and one bar's close against
the bar before it):

  BUY CE: the latest MACD(12,26,9) histogram bar is BELOW zero (this
  is the "only check on a bearish histogram bar" gate) AND it is
  GREATER than the previous histogram bar (momentum fading, less
  negative) AND the latest candle's close is GREATER than the
  previous candle's close (price agrees). Fires at that same bar's
  own close.

  BUY PE: the mirror -- latest histogram bar is ABOVE zero AND LESS
  than the previous bar (momentum fading, less positive) AND the
  latest close is LESS than the previous close. Fires at that bar's
  own close.

This is a one-bar-lag momentum-turn read: it doesn't wait for a
confirmed local peak/trough the way macd_divergence_hump_options.py
does, and it doesn't compare across two separate humps the way a
divergence does -- it's the simplest possible "histogram and price
both just ticked the other way, while still on the losing side of
zero" filter.

MACD is computed CONTINUOUSLY across the whole backtest date range
(not reseeded every morning) -- a daily reset would throw away most
of the 26-period slow EMA's warm-up every single day, and this is
the same choice macd_divergence_hump_screener.py made for its own
(also continuous, daily-bar) histogram. Only the OPTION POSITION
force-flattens at the end of its own entry day; the indicator itself
spans days, same convention as order_block_options.py's structure
tracking.

ENTRY FILL: at the signal bar's own close (via the usual
_decision_time()-based option fill), the same convention every other
sl_rs/target_rs module in this codebase uses -- not the next bar's
open. (The user offered either; this module picks the close-of-
signal-bar convention to stay consistent with the rest of the
codebase. A next-bar-open variant would shift entry_idx by one
position and is a small change if wanted.)

EXIT: fixed rupee P&L stop-loss/target on the option premium (x
lot_size) -- sl_rs (default 300) or target_rs (default 600),
whichever hits first, else force-flat at FORCE_FLAT_TIME (15:25).
Only one position open at a time.

candle_minutes (default 5): resamples the whole continuous 1-minute
index stream first (not per day -- see above). Every option fill and
the FORCE_FLAT_TIME cutoff go through a _decision_time() helper
identical to every other module's.

STRIKE/EXPIRY: ATM = round_to_step(index close, strike_step), nearest
expiry on/after the entry day. Uses the EXPIRED-instruments API (needs
an Upstox access token) for any day whose weekly contract has already
rolled over.
"""
from __future__ import annotations

from dataclasses import dataclass

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, _bar_at_or_after, _bar_at_or_before
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.sweep_reclaim_breakout import _resample
from backtest.technical_rating import _macd

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
STOP_LOSS_RS = 300.0
TARGET_RS = 600.0


@dataclass
class Bar:
    ts: str
    o: float
    h: float
    l: float
    c: float


def _apply_slippage(price: float, side: str, slippage_pct: float) -> float:
    if slippage_pct <= 0:
        return price
    return price * (1 + slippage_pct) if side == "BUY" else price * (1 - slippage_pct)


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    candle_minutes: int = 5,
    macd_fast: int = MACD_FAST,
    macd_slow: int = MACD_SLOW,
    macd_signal: int = MACD_SIGNAL,
    sl_rs: float = STOP_LOSS_RS,
    target_rs: float = TARGET_RS,
    slippage_pct: float = 0.0,
    access_token: str | None = None,
) -> list[OptionTrade]:
    trading_days = upstox_client.get_daily_history(underlying_key, from_date, to_date)
    trading_days.sort(key=lambda d: d["date"])
    if not trading_days:
        return []

    expiries = sorted(cache.get_expired_expiries_cached(underlying_key, "options", access_token))
    chain_cache: dict[str, dict] = {}

    def _atm_option_candles(strike, opt_type, date, expiry):
        if expiry not in chain_cache:
            chain_cache[expiry] = oc.build_chain_lookup(
                cache.get_expired_option_chain_cached(underlying_key, expiry, access_token)
            )
        lookup = chain_cache[expiry]
        contract = oc.nearest_contract(lookup, strike, opt_type)
        if contract is None:
            return None, None
        candles = cache.get_day_candles_cached(
            contract["instrument_key"], "1minute", date, expired=True, access_token=access_token
        )
        return contract, sorted(candles, key=lambda c: c[0])

    def _fill(candles, at_time_str):
        return _bar_at_or_after(candles, at_time_str) or _bar_at_or_before(candles, at_time_str)

    def _decision_time(ts: str) -> str:
        if candle_minutes <= 1:
            return ts[11:16]
        hh, mm = int(ts[11:13]), int(ts[14:16])
        dh, dm = divmod(hh * 60 + mm + candle_minutes, 60)
        return f"{dh:02d}:{dm:02d}"

    all_1min: list[list] = []
    for day in trading_days:
        rows = sorted(cache.get_day_candles_cached(underlying_key, "1minute", day["date"], expired=False), key=lambda r: r[0])
        all_1min.extend(rows)
    all_1min.sort(key=lambda r: r[0])

    min_bars = macd_slow + macd_signal + 3
    if len(all_1min) < min_bars:
        return []

    rows = _resample(all_1min, candle_minutes) if candle_minutes > 1 else all_1min
    if len(rows) < min_bars:
        return []
    bars = [Bar(ts=r[0], o=r[1], h=r[2], l=r[3], c=r[4]) for r in rows]
    closes = [b.c for b in bars]
    macd_line, signal_line = _macd(closes, macd_fast, macd_slow, macd_signal)
    histogram = [(m - s) if (m is not None and s is not None) else None for m, s in zip(macd_line, signal_line)]

    trades: list[OptionTrade] = []

    i = 1
    while i < len(bars):
        bar_date = bars[i].ts[:10]
        decision_time_str = _decision_time(bars[i].ts)
        skip_entries_today = decision_time_str >= FORCE_FLAT_TIME

        h, h_prev = histogram[i], histogram[i - 1]
        close, close_prev = bars[i].c, bars[i - 1].c

        direction_label = opt_type = None
        if not skip_entries_today and h is not None and h_prev is not None:
            if h < 0 and h > h_prev and close > close_prev:
                direction_label, opt_type = "LONG", "CE"
            elif h > 0 and h < h_prev and close < close_prev:
                direction_label, opt_type = "SHORT", "PE"

        if direction_label is None:
            i += 1
            continue

        d = bar_date
        expiry = next((e for e in expiries if e >= d), None)
        if expiry is None:
            i += 1
            continue

        entry_idx = i
        entry_close = bars[entry_idx].c
        entry_ts = bars[entry_idx].ts
        entry_decision_time = decision_time_str
        atm = oc.round_to_step(entry_close, strike_step)

        contract, opt_rows = _atm_option_candles(atm, opt_type, d, expiry)
        if contract is None or not opt_rows:
            i = entry_idx + 1
            continue

        entry_bar = _fill(opt_rows, entry_decision_time)
        if entry_bar is None:
            i = entry_idx + 1
            continue
        entry_price = _apply_slippage(entry_bar[4], "BUY", slippage_pct)
        lot_size = contract["lot_size"]

        day_end_idx = entry_idx
        while day_end_idx + 1 < len(bars) and bars[day_end_idx + 1].ts[:10] == d:
            day_end_idx += 1

        exit_idx = None
        exit_reason = None
        for j in range(entry_idx + 1, day_end_idx + 1):
            j_decision_time = _decision_time(bars[j].ts)
            if j_decision_time >= FORCE_FLAT_TIME:
                exit_idx = j
                exit_reason = "eod"
                break
            bar_j = _fill(opt_rows, j_decision_time)
            premium_j = bar_j[4] if bar_j else entry_price
            pnl_per_lot = (premium_j - entry_price) * lot_size
            if pnl_per_lot <= -sl_rs:
                exit_idx = j
                exit_reason = "stop_loss"
                break
            if pnl_per_lot >= target_rs:
                exit_idx = j
                exit_reason = "target"
                break

        if exit_idx is None:
            exit_idx = day_end_idx
            exit_reason = "eod"

        exit_decision_time = _decision_time(bars[exit_idx].ts)
        exit_bar = _fill(opt_rows, exit_decision_time)
        exit_price = _apply_slippage(exit_bar[4], "SELL", slippage_pct) if exit_bar else entry_price
        exit_time = exit_bar[0] if exit_bar else bars[exit_idx].ts

        trades.append(OptionTrade(
            date=d, direction=direction_label, expiry=expiry, strike=contract["strike_price"],
            entry_time=entry_ts, entry_premium=entry_price, exit_time=exit_time, exit_premium=exit_price,
            lot_size=lot_size, exit_reason=exit_reason,
        ))

        i = exit_idx + 1

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
