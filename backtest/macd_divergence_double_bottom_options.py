"""Backtest: NIFTY 50 spot candles, buy ATM CE on a bullish Double Bottom
confirmed by MACD bullish divergence; buy ATM PE on the bearish mirror
(Double Top + MACD bearish divergence) -- through real ATM NIFTY
options, with a fixed rupee stop-loss/target.

SWING PIVOTS: fractal (left=right=PIVOT_WINDOW bars), same convention
as live/bullish_chart_pattern_screener.py's _swing_points -- a pivot
at index p is only CONFIRMED once PIVOT_WINDOW bars after it exist
(current bar i >= p + PIVOT_WINDOW); using it earlier would be
look-ahead, since a fractal's "rightness" isn't knowable before those
later bars have actually printed.

DOUBLE BOTTOM (bullish): the two most recently CONFIRMED swing lows
L1, L2 (i1 < i2), gap i2-i1 within [MIN_GAP_BARS, MAX_GAP_BARS],
within 0.5x ATR of each other (comparable depth), each the TRUE
extreme of the [i1,i2] span (rules out a deeper untagged dip hiding
between them), with an intervening swing high ("the peak"/neckline)
at least 3.0x ATR above both -- same thresholds
live/bullish_chart_pattern_screener.py's _detect_double_bottom uses,
just evaluated here against a RUNNING pivot list instead of a static
one (and only the latest low-pair is checked each bar, not every
historical pair). STALE if L2 confirmed more than STALE_BARS bars ago
with no breakout since.

MACD BULLISH DIVERGENCE (the new filter on top of the plain double
bottom): the MACD(12,26,9) histogram value AT L2's own bar must be
HIGHER than at L1's own bar (histogram[i2] > histogram[i1]) -- price
making an equal/lower low while momentum makes a HIGHER low is
textbook bullish divergence. MACD is reseeded fresh every trading day
(an SMA seed on that day's own candles), matching every other live/
backtest module's "revert back to daily calculation" convention in
this codebase.

CONFIRMATION: once a valid (L1, L2, peak, divergence) combination
exists, the trade fires the first later bar whose own close breaks
above the peak -- same "breakout" trigger as the live chart-pattern
screener, evaluated using only that bar's own already-known close (no
look-ahead).

DOUBLE TOP (bearish) is the exact mirror: two most recent confirmed
swing highs H1, H2 within 0.5x ATR, an intervening swing low ("the
trough") at least 3.0x ATR below both, MACD histogram[i2] < histogram
[i1] (bearish divergence -- equal/higher high, lower momentum high),
confirmed when a later bar's close breaks below the trough.

ENTRY FILL: the confirming bar's own close (no decision-time offset
needed at candle_minutes=1 -- no resampling happens; see
macd_histogram_1min_dualstop_options.py's _decision_time for the
general case if this is later extended to candle_minutes>1).

EXIT: fixed rupee P&L stop-loss/target on the whole position (premium
move x lot_size), same convention as every sl_rs/target_rs module in
this codebase -- sl_rs (default 300) or target_rs (default 600),
whichever hits first, else force-flat at FORCE_FLAT_TIME (15:25).
Only one position open at a time.

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
from backtest.technical_rating import _macd

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"
PIVOT_WINDOW = 5  # fractal pivot left=right bars (1-minute)
MIN_GAP_BARS = 8
MAX_GAP_BARS = 60
STALE_BARS = 40
DEPTH_ATR_MULT = 0.5    # how close L1/L2 (or H1/H2) must be to each other
PEAK_ATR_MULT = 3.0     # how far the intervening peak/trough must clear both lows/highs
ATR_PERIOD = 14


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


def _atr_at(bars: list[Bar], i: int, period: int = ATR_PERIOD) -> float | None:
    if i < period:
        return None
    trs = []
    for k in range(i - period + 1, i + 1):
        prev_c = bars[k - 1].c
        trs.append(max(bars[k].h - bars[k].l, abs(bars[k].h - prev_c), abs(bars[k].l - prev_c)))
    return sum(trs) / period


def _all_pivots(bars: list[Bar], left: int, right: int) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Every fractal pivot in the day's bars, each tagged by its own
    index p -- the CALLER must only use a pivot once current bar i has
    reached p + right (that's when it's actually confirmed)."""
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    for p in range(left, len(bars) - right):
        window = bars[p - left:p + right + 1]
        if bars[p].h == max(b.h for b in window):
            highs.append((p, bars[p].h))
        if bars[p].l == min(b.l for b in window):
            lows.append((p, bars[p].l))
    return highs, lows


def _double_bottom(bars: list[Bar], histogram: list[float | None], highs, lows, i: int, atr: float) -> tuple[int, float] | None:
    """Checks only the latest confirmed low-pair as of bar i. Returns
    the peak (neckline) level if a valid, divergence-confirmed, not-yet
    -stale double bottom exists, else None."""
    confirmed_lows = [(p, v) for p, v in lows if p + PIVOT_WINDOW <= i]
    if len(confirmed_lows) < 2:
        return None
    (i1, p1), (i2, p2) = confirmed_lows[-2], confirmed_lows[-1]
    if i - i2 > STALE_BARS:
        return None
    gap = i2 - i1
    if gap < MIN_GAP_BARS or gap > MAX_GAP_BARS:
        return None
    if abs(p1 - p2) > atr * DEPTH_ATR_MULT:
        return None
    if min(b.l for b in bars[i1:i2 + 1]) < min(p1, p2) - atr * 0.1:
        return None
    between = [(p, v) for p, v in highs if i1 < p < i2 and p + PIVOT_WINDOW <= i]
    if not between:
        return None
    peak_idx, peak_price = max(between, key=lambda h: h[1])
    if peak_price - max(p1, p2) < atr * PEAK_ATR_MULT:
        return None
    h1, h2 = histogram[i1], histogram[i2]
    if h1 is None or h2 is None or h2 <= h1:
        return None  # no bullish divergence
    return peak_idx, peak_price


def _double_top(bars: list[Bar], histogram: list[float | None], highs, lows, i: int, atr: float) -> tuple[int, float] | None:
    confirmed_highs = [(p, v) for p, v in highs if p + PIVOT_WINDOW <= i]
    if len(confirmed_highs) < 2:
        return None
    (i1, p1), (i2, p2) = confirmed_highs[-2], confirmed_highs[-1]
    if i - i2 > STALE_BARS:
        return None
    gap = i2 - i1
    if gap < MIN_GAP_BARS or gap > MAX_GAP_BARS:
        return None
    if abs(p1 - p2) > atr * DEPTH_ATR_MULT:
        return None
    if max(b.h for b in bars[i1:i2 + 1]) > max(p1, p2) + atr * 0.1:
        return None
    between = [(p, v) for p, v in lows if i1 < p < i2 and p + PIVOT_WINDOW <= i]
    if not between:
        return None
    trough_idx, trough_price = min(between, key=lambda l: l[1])
    if min(p1, p2) - trough_price < atr * PEAK_ATR_MULT:
        return None
    h1, h2 = histogram[i1], histogram[i2]
    if h1 is None or h2 is None or h2 >= h1:
        return None  # no bearish divergence
    return trough_idx, trough_price


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sl_rs: float = 300.0,
    target_rs: float = 600.0,
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

    trades: list[OptionTrade] = []
    min_bars = ATR_PERIOD + 2 * PIVOT_WINDOW + MIN_GAP_BARS + 5

    for day in trading_days:
        d = day["date"]
        rows = sorted(cache.get_day_candles_cached(underlying_key, "1minute", d, expired=False), key=lambda r: r[0])
        if len(rows) < min_bars:
            continue
        bars = [Bar(ts=r[0], o=r[1], h=r[2], l=r[3], c=r[4]) for r in rows]
        closes = [b.c for b in bars]
        macd_line, signal_line = _macd(closes, macd_fast, macd_slow, macd_signal)
        histogram = [(m - s) if (m is not None and s is not None) else None for m, s in zip(macd_line, signal_line)]
        highs, lows = _all_pivots(bars, PIVOT_WINDOW, PIVOT_WINDOW)

        expiry = next((e for e in expiries if e >= d), None)
        if expiry is None:
            continue

        i = ATR_PERIOD + PIVOT_WINDOW + MIN_GAP_BARS
        while i < len(bars):
            time_str = bars[i].ts[11:16]
            if time_str >= FORCE_FLAT_TIME:
                break

            atr = _atr_at(bars, i)
            if atr is None or atr <= 0:
                i += 1
                continue

            direction_label = opt_type = None
            db = _double_bottom(bars, histogram, highs, lows, i, atr)
            if db is not None and bars[i].c > db[1]:
                direction_label, opt_type = "LONG", "CE"
            if direction_label is None:
                dt = _double_top(bars, histogram, highs, lows, i, atr)
                if dt is not None and bars[i].c < dt[1]:
                    direction_label, opt_type = "SHORT", "PE"

            if direction_label is None:
                i += 1
                continue

            entry_idx = i
            entry_close = bars[entry_idx].c
            entry_ts = bars[entry_idx].ts
            atm = oc.round_to_step(entry_close, strike_step)

            contract, opt_rows = _atm_option_candles(atm, opt_type, d, expiry)
            if contract is None or not opt_rows:
                i = entry_idx + 1
                continue

            entry_bar = _fill(opt_rows, time_str)
            if entry_bar is None:
                i = entry_idx + 1
                continue
            entry_price = _apply_slippage(entry_bar[4], "BUY", slippage_pct)
            lot_size = contract["lot_size"]

            exit_idx = None
            exit_reason = None
            for j in range(entry_idx + 1, len(bars)):
                j_time_str = bars[j].ts[11:16]
                if j_time_str >= FORCE_FLAT_TIME:
                    exit_idx = j
                    exit_reason = "eod"
                    break
                bar_j = _fill(opt_rows, j_time_str)
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
                exit_idx = len(bars) - 1
                exit_reason = "eod"

            exit_time_str = bars[exit_idx].ts[11:16]
            exit_bar = _fill(opt_rows, exit_time_str)
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
