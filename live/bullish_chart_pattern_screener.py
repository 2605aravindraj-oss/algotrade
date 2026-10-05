"""Live bullish GEOMETRIC chart-pattern screener, NIFTY 50 stocks, 5-minute
bars: Double Bottom, Ascending Triangle, Bullish Flag.

Sibling to live/bullish_pattern_screener.py (candlestick patterns --
Engulfing, Hammer, etc., each checkable from 1-3 bars with no geometry
involved). These three are different in kind: each needs swing-pivot
detection and a trendline/level fit across a WINDOW of bars, which is
inherently more subjective than a candlestick rule. Thresholds below
are stated explicitly and are heuristic, not backtested -- this is a
screener to shortlist what to look at, not a validated signal.

SWING PIVOTS (shared basis for all three patterns): a fractal pivot --
bar i is a swing HIGH if its high is the max over [i-left, i+right],
a swing LOW if its low is the min over the same window (left=right=3
bars, i.e. +/-15 minutes at 5-minute bars). A pivot near the tail isn't
confirmed until `right` bars after it exist, same as any real-time
fractal detector -- the very last few bars can never be pivots yet.

DOUBLE BOTTOM: two swing lows L1, L2 (8-60 bars apart, roughly a
40-minute to 5-hour span) within 1.0x ATR of each other (comparable
depth), with an intervening swing high ("the peak"/neckline) at least
2.0x ATR above both -- without a real intervening bounce, two nearby
lows are just noise, not a "W". CONFIRMED when the latest close is
above the peak (the breakout); otherwise reported separately as
FORMING (both lows in place, breakout not yet triggered).

ASCENDING TRIANGLE: the most recent 2-3 swing highs sit within 0.5x
ATR of each other (a flat resistance) while the swing lows in between
them strictly increase (each at least 0.3x ATR above the last --
genuine higher lows, not noise). CONFIRMED when the latest close
breaks above that flat resistance; otherwise FORMING.

BULLISH FLAG: a "pole" -- an impulsive move where cumulative return
over a 6-16 bar window is >= 3.0x ATR with at least 65% of those bars
bullish -- followed immediately by a "flag": 3-10 bars consolidating
in a tight range (high-low spread <= 60% of the pole's own size) that
retraces no more than 50% of the pole. CONFIRMED when the latest close
breaks above the flag's own high; otherwise FORMING.

DATA: same pipeline as bullish_pattern_screener.py -- public
historical-candle endpoint for the last few trading days plus
upstox_client.get_intraday_candles for today, merged and resampled to
5-minute bars, with the same trailing-degenerate-bar trim (this feed's
last ~10-15 minutes of the day often carry only a last-traded-price
tick).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

from data_sources import instruments
from live.bullish_pattern_screener import NIFTY_50, Bar, _atr, _recent_bars, _trim_degenerate_tail

LEFT = RIGHT = 3  # fractal pivot window (bars each side)


@dataclass
class ChartPatternHit:
    symbol: str
    pattern: str
    status: str  # "confirmed" or "forming"
    bar_time: str
    close: float
    level: float  # the neckline/resistance/flag-high being broken (or watched)


def _swing_points(bars: list[Bar]) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    for i in range(LEFT, len(bars) - RIGHT):
        window = bars[i - LEFT:i + RIGHT + 1]
        if bars[i].h == max(b.h for b in window):
            highs.append((i, bars[i].h))
        if bars[i].l == min(b.l for b in window):
            lows.append((i, bars[i].l))
    return highs, lows


def _detect_double_bottom(bars: list[Bar], atr: float, highs, lows) -> ChartPatternHit | None:
    if len(lows) < 2:
        return None
    for j in range(len(lows) - 1, 0, -1):
        i2, p2 = lows[j]
        i1, p1 = lows[j - 1]
        gap = i2 - i1
        if gap < 8 or gap > 60:
            continue
        if abs(p1 - p2) > atr * 1.0:
            continue
        between_peaks = [h for h in highs if i1 < h[0] < i2]
        if not between_peaks:
            continue
        peak_idx, peak_price = max(between_peaks, key=lambda h: h[1])
        if peak_price - max(p1, p2) < atr * 2.0:
            continue
        cur = bars[-1]
        if cur.c > peak_price:
            return ChartPatternHit("", "Double Bottom", "confirmed", cur.ts, cur.c, peak_price)
        return ChartPatternHit("", "Double Bottom", "forming", bars[i2].ts, bars[i2].c, peak_price)
    return None


def _detect_ascending_triangle(bars: list[Bar], atr: float, highs, lows) -> ChartPatternHit | None:
    if len(highs) < 2 or len(lows) < 2:
        return None
    recent_highs = highs[-3:] if len(highs) >= 3 else highs[-2:]
    if max(h[1] for h in recent_highs) - min(h[1] for h in recent_highs) > atr * 0.5:
        return None
    span_start = recent_highs[0][0]
    inner_lows = [l for l in lows if span_start <= l[0] <= len(bars) - 1]
    if len(inner_lows) < 2:
        return None
    rising = all(inner_lows[k][1] > inner_lows[k - 1][1] + atr * 0.3 for k in range(1, len(inner_lows)))
    if not rising:
        return None
    resistance = sum(h[1] for h in recent_highs) / len(recent_highs)
    cur = bars[-1]
    if cur.c > resistance:
        return ChartPatternHit("", "Ascending Triangle", "confirmed", cur.ts, cur.c, resistance)
    return ChartPatternHit("", "Ascending Triangle", "forming", bars[inner_lows[-1][0]].ts, bars[inner_lows[-1][0]].c, resistance)


def _detect_bullish_flag(bars: list[Bar], atr: float) -> ChartPatternHit | None:
    n = len(bars)
    for pole_len in range(16, 5, -1):
        for flag_len in range(3, 11):
            pole_end = n - flag_len
            pole_start = pole_end - pole_len
            if pole_start < 0 or pole_end <= 0 or pole_end >= n:
                continue
            pole = bars[pole_start:pole_end]
            flag = bars[pole_end:n]
            if len(pole) < 2 or len(flag) < 3:
                continue
            pole_move = pole[-1].c - pole[0].o
            if pole_move < atr * 3.0:
                continue
            bullish_frac = sum(1 for b in pole if b.bullish) / len(pole)
            if bullish_frac < 0.65:
                continue
            flag_high = max(b.h for b in flag)
            flag_low = min(b.l for b in flag)
            if flag_high - flag_low > pole_move * 0.6:
                continue
            retrace = pole[-1].c - flag_low
            if retrace > pole_move * 0.5:
                continue
            cur = bars[-1]
            if cur.c > flag_high:
                return ChartPatternHit("", "Bullish Flag", "confirmed", cur.ts, cur.c, flag_high)
            return ChartPatternHit("", "Bullish Flag", "forming", flag[-1].ts, flag[-1].c, flag_high)
    return None


def run() -> list[ChartPatternHit]:
    keys = instruments.resolve_symbols(NIFTY_50)
    hits: list[ChartPatternHit] = []
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
        if len(bars) < 30:
            continue
        atr = _atr(bars)
        if atr is None or atr <= 0:
            continue
        highs, lows = _swing_points(bars)

        for hit in (
            _detect_double_bottom(bars, atr, highs, lows),
            _detect_ascending_triangle(bars, atr, highs, lows),
            _detect_bullish_flag(bars, atr),
        ):
            if hit is not None:
                hit.symbol = symbol
                hits.append(hit)
    return hits


def summary(hits: list[ChartPatternHit]) -> str:
    if not hits:
        return "No bullish chart patterns found across NIFTY 50."
    confirmed = [h for h in hits if h.status == "confirmed"]
    forming = [h for h in hits if h.status == "forming"]
    lines = [f"{len(hits)} bullish chart pattern(s) across NIFTY 50 (5-minute bars):", ""]
    if confirmed:
        lines.append(f"CONFIRMED breakouts ({len(confirmed)}):")
        for h in sorted(confirmed, key=lambda x: (x.pattern, x.symbol)):
            lines.append(f"  {h.symbol:<12} {h.pattern:<20} @ {h.bar_time}  close={h.close:.2f}  broke above {h.level:.2f}")
        lines.append("")
    if forming:
        lines.append(f"FORMING, no breakout yet ({len(forming)}):")
        for h in sorted(forming, key=lambda x: (x.pattern, x.symbol)):
            lines.append(f"  {h.symbol:<12} {h.pattern:<20} @ {h.bar_time}  close={h.close:.2f}  watch breakout above {h.level:.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    results = run()
    print(summary(results))
