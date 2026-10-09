"""Backtest: NIFTY 50 spot 1-minute candles, buy ATM CE when price comes to
a SUPPORT level and a bullish candlestick pattern confirms there; buy ATM
PE when price comes to a RESISTANCE level and a bearish pattern confirms
-- through real ATM NIFTY options, with a fixed rupee stop-loss/target.

SUPPORT/RESISTANCE: fractal swing pivots (left=right=PIVOT_WINDOW bars,
default 5 -- i.e. +/-5 minutes at 1-minute bars), the same convention
live/bullish_chart_pattern_screener.py uses for its swing points, just
tuned for 1-minute noise instead of 5-minute. A pivot is only CONFIRMED
(usable by the strategy) once PIVOT_WINDOW bars after it exist -- using
it earlier would be look-ahead, since a fractal pivot's "rightness" isn't
knowable until those later bars have actually printed. Only pivots formed
within the last PIVOT_LOOKBACK_BARS (default 120, i.e. the last 2 hours)
count as "active" support/resistance -- a swing low from the morning
isn't a relevant level for an afternoon decision. "Price comes to" a
level means the current bar's low (for support) or high (for resistance)
is within PROXIMITY_ATR_MULT x ATR(14) of it -- no "unbroken" check is
applied (the most recent confirmed pivot is used regardless of whether
price already closed through it since), since the proximity + pattern
filters together already do the real signal-quality work.

PATTERNS: classic candlestick reversal patterns, computed strictly on
bars up to and including the current one (no future bars, so no
look-ahead there either) -- adapted from live/bullish_pattern_screener.py
with bearish mirrors added for the PE side:
    BULLISH (checked at support): Bullish Engulfing, Hammer, Bullish
        Harami, Piercing Line, Morning Star, Three White Soldiers,
        Tweezer Bottom.
    BEARISH (checked at resistance): Bearish Engulfing, Shooting Star,
        Bearish Harami, Dark Cloud Cover, Evening Star, Three Black
        Crows, Tweezer Top.
"Long"/"small" body and wick thresholds are relative to the index's own
ATR(14) on 1-minute bars, same reasoning as every ATR-relative filter
elsewhere in this codebase. A trade fires when a bullish pattern
confirms AND price is at support (or the bearish/resistance mirror),
both on the SAME bar.

ENTRY FILL: the bar's own close (no decision-time offset needed, since
no resampling happens at 1-minute -- same convention
macd_histogram_1min_dualstop_options.py uses at candle_minutes=1, and
the live scripts' own "filled at the signal candle's own close").

EXIT: fixed rupee P&L stop-loss/target on the whole position (premium
move x lot_size, same convention as every sl_rs/target_rs module here)
-- sl_rs (default 300) or target_rs (default 600), whichever hits
first, else force-flat at FORCE_FLAT_TIME (15:25). Only one position
open at a time; the entry scan only resumes after the current trade's
exit.

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

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"
ATR_PERIOD = 14
PIVOT_WINDOW = 5  # fractal pivot left=right bars
PIVOT_LOOKBACK_BARS = 120  # only pivots formed within this many bars ago are "active"
PROXIMITY_ATR_MULT = 0.15  # how close price must be to a level to count as "at" it


@dataclass
class Bar:
    ts: str
    o: float
    h: float
    l: float
    c: float

    @property
    def body(self) -> float:
        return abs(self.c - self.o)

    @property
    def upper_wick(self) -> float:
        return self.h - max(self.o, self.c)

    @property
    def lower_wick(self) -> float:
        return min(self.o, self.c) - self.l

    @property
    def bullish(self) -> bool:
        return self.c > self.o

    @property
    def bearish(self) -> bool:
        return self.c < self.o


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


def _confirmed_pivots(bars: list[Bar], i: int, left: int = PIVOT_WINDOW, right: int = PIVOT_WINDOW,
                       lookback: int = PIVOT_LOOKBACK_BARS) -> tuple[tuple[int, float] | None, tuple[int, float] | None]:
    """The most recent CONFIRMED swing low / swing high as of bar i --
    only pivot candidates at index <= i-right are confirmable (a
    fractal's "rightness" isn't knowable before those bars exist), and
    only those formed within `lookback` bars of i count as active."""
    lo = hi = None
    last_confirmable = i - right
    earliest = max(left, i - lookback)
    for p in range(last_confirmable, earliest - 1, -1):
        if p < left or p + right >= len(bars):
            continue
        window = bars[p - left:p + right + 1]
        if lo is None and bars[p].l == min(b.l for b in window):
            lo = (p, bars[p].l)
        if hi is None and bars[p].h == max(b.h for b in window):
            hi = (p, bars[p].h)
        if lo is not None and hi is not None:
            break
    return lo, hi


def _bullish_pattern(bars: list[Bar], i: int, atr: float) -> str | None:
    if atr <= 0:
        return None
    cur = bars[i]
    prev = bars[i - 1] if i >= 1 else None

    if prev is not None and prev.bearish and cur.bullish and cur.o <= prev.c and cur.c >= prev.o and cur.body > prev.body:
        return "Bullish Engulfing"
    if cur.body > 0 and cur.lower_wick >= 2 * cur.body and cur.upper_wick <= 0.3 * cur.body:
        return "Hammer"
    if prev is not None and prev.bearish and cur.bullish and prev.body > atr * 0.5 and cur.o >= min(prev.o, prev.c) and cur.c <= max(prev.o, prev.c):
        return "Bullish Harami"
    if prev is not None and prev.bearish and cur.bullish and prev.body > atr * 0.5:
        midpoint = (prev.o + prev.c) / 2
        if cur.o < prev.l and prev.c < cur.c < prev.o and cur.c > midpoint:
            return "Piercing Line"
    if prev is not None and prev.bearish and cur.bullish and abs(prev.l - cur.l) <= atr * 0.1:
        return "Tweezer Bottom"
    if i >= 2:
        first, star = bars[i - 2], bars[i - 1]
        if (first.bearish and first.body > atr * 0.5 and star.body < atr * 0.3
                and max(star.o, star.c) < first.c and cur.bullish and cur.c > (first.o + first.c) / 2):
            return "Morning Star"
        a, b = bars[i - 2], bars[i - 1]
        if (a.bullish and b.bullish and cur.bullish and b.o > a.o and b.o < a.c
                and cur.o > b.o and cur.o < b.c and b.c > a.c and cur.c > b.c
                and a.body > atr * 0.3 and b.body > atr * 0.3 and cur.body > atr * 0.3):
            return "Three White Soldiers"
    return None


def _bearish_pattern(bars: list[Bar], i: int, atr: float) -> str | None:
    if atr <= 0:
        return None
    cur = bars[i]
    prev = bars[i - 1] if i >= 1 else None

    if prev is not None and prev.bullish and cur.bearish and cur.o >= prev.c and cur.c <= prev.o and cur.body > prev.body:
        return "Bearish Engulfing"
    if cur.body > 0 and cur.upper_wick >= 2 * cur.body and cur.lower_wick <= 0.3 * cur.body:
        return "Shooting Star"
    if prev is not None and prev.bullish and cur.bearish and prev.body > atr * 0.5 and cur.o <= max(prev.o, prev.c) and cur.c >= min(prev.o, prev.c):
        return "Bearish Harami"
    if prev is not None and prev.bullish and cur.bearish and prev.body > atr * 0.5:
        midpoint = (prev.o + prev.c) / 2
        if cur.o > prev.h and prev.o < cur.c < midpoint:
            return "Dark Cloud Cover"
    if prev is not None and prev.bullish and cur.bearish and abs(prev.h - cur.h) <= atr * 0.1:
        return "Tweezer Top"
    if i >= 2:
        first, star = bars[i - 2], bars[i - 1]
        if (first.bullish and first.body > atr * 0.5 and star.body < atr * 0.3
                and min(star.o, star.c) > first.c and cur.bearish and cur.c < (first.o + first.c) / 2):
            return "Evening Star"
        a, b = bars[i - 2], bars[i - 1]
        if (a.bearish and b.bearish and cur.bearish and b.o < a.o and b.o > a.c
                and cur.o < b.o and cur.o > b.c and b.c < a.c and cur.c < b.c
                and a.body > atr * 0.3 and b.body > atr * 0.3 and cur.body > atr * 0.3):
            return "Three Black Crows"
    return None


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
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

    for day in trading_days:
        d = day["date"]
        rows = sorted(cache.get_day_candles_cached(underlying_key, "1minute", d, expired=False), key=lambda r: r[0])
        if len(rows) < ATR_PERIOD + 2 * PIVOT_WINDOW + 5:
            continue
        bars = [Bar(ts=r[0], o=r[1], h=r[2], l=r[3], c=r[4]) for r in rows]

        expiry = next((e for e in expiries if e >= d), None)
        if expiry is None:
            continue

        min_i = ATR_PERIOD + PIVOT_WINDOW + 3
        i = min_i
        while i < len(bars):
            time_str = bars[i].ts[11:16]
            if time_str >= FORCE_FLAT_TIME:
                break

            atr = _atr_at(bars, i)
            if atr is None:
                i += 1
                continue

            support, resistance = _confirmed_pivots(bars, i)
            direction_label = opt_type = pattern = None
            if support is not None and abs(bars[i].l - support[1]) <= atr * PROXIMITY_ATR_MULT:
                pattern = _bullish_pattern(bars, i, atr)
                if pattern is not None:
                    direction_label, opt_type = "LONG", "CE"
            if direction_label is None and resistance is not None and abs(bars[i].h - resistance[1]) <= atr * PROXIMITY_ATR_MULT:
                pattern = _bearish_pattern(bars, i, atr)
                if pattern is not None:
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
