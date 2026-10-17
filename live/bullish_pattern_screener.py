"""Live bullish candlestick-pattern screener, NIFTY 50 stocks, 5-minute bars.

Not a backtest -- this looks at the MOST RECENTLY CLOSED 5-minute candle
(and the 1-2 before it, for multi-candle patterns) for every NIFTY 50
stock, right now, and reports which ones currently show a classic
BULLISH candlestick pattern. Meant to be re-run whenever you want a
fresh read during market hours.

DATA: today's candles come from upstox_client.get_intraday_candles
(public, no auth, current trading day only). The 1-2 trading days
before today come from the public historical-candle endpoint via
data_sources.cache (expired=False -- no auth needed for equities
either, same as the index). Both are 1-minute, merged and resampled to
5-minute bars with backtest.sweep_reclaim_breakout._resample, the same
resampler every other module in this codebase uses.

PATTERNS (candlestick, not geometric chart patterns like triangles/
flags/head-and-shoulders -- those need subjective trendline/pivot
fitting that's noisy at 5-minute resolution; this screener sticks to
well-defined, mechanically-checkable reversal/continuation candles):

    Bullish Engulfing   -- bearish candle followed by a bullish candle
                            whose body fully engulfs the prior body.
    Hammer              -- small body near the top of the bar, a lower
                            wick >= 2x the body, a small/no upper wick,
                            occurring at a local low (downtrend context).
    Inverted Hammer     -- the mirror (long upper wick, small lower
                            wick) at a local low -- weaker on its own,
                            flagged as "needs confirmation".
    Bullish Harami      -- a long bearish candle followed by a small
                            bullish candle whose entire body sits
                            inside the prior candle's body.
    Piercing Line       -- a long bearish candle, then a bullish candle
                            that opens below the prior low and closes
                            above the midpoint of the prior body.
    Morning Star        -- long bearish, small-bodied "star" that gaps
                            down, long bullish closing back above the
                            first candle's midpoint.
    Three White Soldiers -- three consecutive bullish candles, each
                            opening within the prior body and closing
                            at a new high.
    Tweezer Bottom      -- two candles (bearish then bullish) with
                            matching lows.

"Long"/"small" body and wick thresholds are relative to each stock's
own ATR(14) on 5-minute bars, not fixed point values -- a Rs 200 stock
and a Rs 4,000 stock need completely different absolute thresholds,
same reasoning as every ATR-relative filter elsewhere in this codebase.
"Local low" context for Hammer/Inverted Hammer/Morning Star: the
pattern's own low must be within the lowest 3 closes of the preceding
10 bars -- a cheap trend-context filter, not a claim about the stock's
broader trend.

This is a SCREENER, not a signal with a validated backtest -- no
entry/exit, no P&L, no win rate. Treat hits as a shortlist to look at,
not a trade recommendation.
"""
from __future__ import annotations

import statistics
import sys
import time
from dataclasses import dataclass, field

from data_sources import cache, instruments, upstox_client
from backtest.sweep_reclaim_breakout import _resample
from backtest.technical_rating_rotation import NIFTY_50

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"
BAR_MINUTES = 5
HISTORY_DAYS = 3  # trading days of 1-minute history before today, for context/ATR


@dataclass
class PatternHit:
    symbol: str
    pattern: str
    bar_time: str
    close: float
    note: str = ""
    span: int = 2  # how many trailing bars (ending at bar_time) the pattern spans -- for charting


@dataclass
class Bar:
    ts: str
    o: float
    h: float
    l: float
    c: float
    v: float

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


def _recent_trading_days(n: int) -> list[str]:
    """The last n COMPLETED trading days before today (index calendar,
    public daily-history endpoint -- correctly skips weekends/holidays)."""
    import datetime as _dt
    today = _dt.date.today()
    lookback_from = (today - _dt.timedelta(days=n * 3 + 5)).isoformat()
    days = upstox_client.get_daily_history(UNDERLYING_KEY, lookback_from, (today - _dt.timedelta(days=1)).isoformat())
    return [d["date"] for d in days[-n:]]


def _recent_bars(instrument_key: str) -> list[Bar]:
    """This stock's last HISTORY_DAYS trading days (cached, public
    historical-candle endpoint) plus today's candles so far (live,
    uncached), merged, sorted, and resampled to BAR_MINUTES bars."""
    rows_1min: list[list] = []
    for d in _recent_trading_days(HISTORY_DAYS):
        rows_1min.extend(cache.get_day_candles_cached(instrument_key, "1minute", d, expired=False))
    try:
        today_rows = upstox_client.get_intraday_candles(instrument_key, "1minute")
        rows_1min.extend(today_rows)
    except Exception:
        pass
    rows_1min.sort(key=lambda r: r[0])
    bars_5min = _resample(rows_1min, BAR_MINUTES)
    return [Bar(ts=r[0], o=r[1], h=r[2], l=r[3], c=r[4], v=r[5]) for r in bars_5min]


def _atr(bars: list[Bar], period: int = 14) -> float | None:
    if len(bars) < period + 1:
        return None
    trs = []
    for i in range(len(bars) - period, len(bars)):
        prev_c = bars[i - 1].c
        tr = max(bars[i].h - bars[i].l, abs(bars[i].h - prev_c), abs(bars[i].l - prev_c))
        trs.append(tr)
    return statistics.mean(trs)


def _is_local_low(bars: list[Bar], i: int, lookback: int = 10) -> bool:
    """bars[i]'s own low sits within the lowest 3 closes of the preceding
    `lookback` bars -- cheap downtrend-context filter."""
    if i < lookback:
        return False
    window_closes = sorted(b.c for b in bars[i - lookback:i])
    threshold = window_closes[2] if len(window_closes) > 2 else window_closes[-1]
    return bars[i].l <= threshold


def _detect(bars: list[Bar], atr: float) -> list[tuple[str, str, int]]:
    """Returns [(pattern_name, note, span), ...] for patterns firing on
    the LAST bar (bars[-1], the most recently closed candle). span is
    how many trailing bars (ending at bars[-1]) the pattern spans."""
    hits: list[tuple[str, str, int]] = []
    n = len(bars)
    if n < 3 or atr <= 0:
        return hits
    cur, prev = bars[-1], bars[-2]

    # Bullish Engulfing
    if prev.bearish and cur.bullish and cur.o <= prev.c and cur.c >= prev.o and cur.body > prev.body:
        hits.append(("Bullish Engulfing", "", 2))

    # Hammer / Inverted Hammer (local-low context)
    if _is_local_low(bars, n - 1) and cur.body > 0:
        if cur.lower_wick >= 2 * cur.body and cur.upper_wick <= 0.3 * cur.body:
            hits.append(("Hammer", "", 1))
        elif cur.upper_wick >= 2 * cur.body and cur.lower_wick <= 0.3 * cur.body:
            hits.append(("Inverted Hammer", "needs confirmation", 1))

    # Bullish Harami
    if prev.bearish and cur.bullish and prev.body > atr * 0.5 and cur.o >= min(prev.o, prev.c) and cur.c <= max(prev.o, prev.c):
        hits.append(("Bullish Harami", "", 2))

    # Piercing Line
    if prev.bearish and cur.bullish and prev.body > atr * 0.5:
        midpoint = (prev.o + prev.c) / 2
        if cur.o < prev.l and prev.c < cur.c < prev.o and cur.c > midpoint:
            hits.append(("Piercing Line", "", 2))

    # Tweezer Bottom
    if prev.bearish and cur.bullish and abs(prev.l - cur.l) <= atr * 0.1:
        hits.append(("Tweezer Bottom", "", 2))

    # Morning Star (3-bar)
    if n >= 3:
        first, star = bars[-3], bars[-2]
        if (first.bearish and first.body > atr * 0.5 and star.body < atr * 0.3
                and max(star.o, star.c) < first.c and cur.bullish
                and cur.c > (first.o + first.c) / 2):
            hits.append(("Morning Star", "", 3))

    # Three White Soldiers (3-bar)
    if n >= 3:
        a, b, c3 = bars[-3], bars[-2], bars[-1]
        if (a.bullish and b.bullish and c3.bullish
                and b.o > a.o and b.o < a.c and c3.o > b.o and c3.o < b.c
                and b.c > a.c and c3.c > b.c
                and a.body > atr * 0.3 and b.body > atr * 0.3 and c3.body > atr * 0.3):
            hits.append(("Three White Soldiers", "", 3))

    return hits


def _trim_degenerate_tail(bars: list[Bar], min_bars: int = 15) -> list[Bar]:
    """Drop trailing degenerate bars (h == l): this feed's last ~10-15
    minutes of the day often carry only a last-traded-price tick rather
    than a full 1-minute OHLC print, which resamples into a flat,
    zero-range 5-minute bucket that can never match a pattern and would
    otherwise mask a real signal on the bar just before it."""
    while len(bars) > min_bars and bars[-1].h == bars[-1].l:
        bars = bars[:-1]
    return bars


def run(stale_after_bars: int = 2) -> list[PatternHit]:
    """Screens every NIFTY 50 stock for bullish candlestick patterns on
    its most recently closed 5-minute bar. stale_after_bars: also checks
    the bar before the last one, so a pattern doesn't disappear from the
    list the instant a new (patternless) bar closes -- sized to stay
    within a ~10 minute window at the default BAR_MINUTES=5."""
    keys = instruments.resolve_symbols(NIFTY_50)
    hits: list[PatternHit] = []
    for symbol in NIFTY_50:
        key = keys.get(symbol)
        if key is None:
            continue
        try:
            bars = _recent_bars(key)
        except Exception as exc:
            print(f"  {symbol}: skipped ({exc})", file=sys.stderr)
            continue
        bars = _trim_degenerate_tail(bars)
        if len(bars) < 15:
            continue
        atr = _atr(bars)
        if atr is None:
            continue
        for offset in range(stale_after_bars):
            end = len(bars) - offset
            if end < 3:
                break
            for pattern, note, span in _detect(bars[:end], atr):
                hits.append(PatternHit(symbol=symbol, pattern=pattern, bar_time=bars[end - 1].ts, close=bars[end - 1].c, note=note, span=span))
            if offset == 0:
                time.sleep(0)  # placeholder for clarity; no extra sleep needed here
    return hits


def summary(hits: list[PatternHit]) -> str:
    if not hits:
        return "No bullish patterns found on the current 5-minute bar across NIFTY 50."
    lines = [f"{len(hits)} bullish pattern hit(s) across NIFTY 50 (5-minute bars):", ""]
    for h in sorted(hits, key=lambda x: (x.pattern, x.symbol)):
        note = f"  [{h.note}]" if h.note else ""
        lines.append(f"  {h.symbol:<12} {h.pattern:<22} @ {h.bar_time}  close={h.close:.2f}{note}")
    return "\n".join(lines)


if __name__ == "__main__":
    results = run()
    print(summary(results))
