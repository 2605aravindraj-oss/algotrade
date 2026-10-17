"""Live screener: NIFTY 50 stocks, DAILY candles, MACD histogram hump
divergence with neckline-breakout confirmation.

Same underlying logic as backtest/macd_divergence_hump_options.py
(the already-built, already-tested intraday NIFTY-index-options
version), ported here to run once per stock per DAILY bar instead of
once per index per intraday bar -- a screener, not a backtest: no
P&L, no entry/exit simulation, just "is a divergence forming or
confirmed right now, for which stocks."

HUMPS: the MACD(12,26,9) histogram (one continuous series over the
whole daily history -- no reseeding; daily bars don't need the
intraday "reset every morning" convention) is segmented into maximal
runs that stay on one side of zero ("humps"). A zero-line crossover
resets hump-tracking state completely -- two peaks/troughs are only
ever compared within the SAME hump, never across a crossover.

LOCAL PEAK/TROUGH: bar k (inside a positive hump) is a local peak if
its histogram value is the strict max over peak_window bars on each
side (2*peak_window+1 bars total, confirmed peak_window bars later);
the negative-hump mirror (strict min) defines a local trough.
peak_window=1 is the literal immediate-neighbor definition.

BEARISH DIVERGENCE: within the same still-open positive hump, a newly
confirmed local peak is LOWER than the previous local peak (momentum
weakening) while the stock's own high at the new peak's day is >= its
high at the previous peak's day (price making an equal/higher high).
BULLISH DIVERGENCE is the exact mirror (negative hump, higher/less-
negative trough, price making an equal/lower low).

NECKLINE: the price extreme BETWEEN the two peaks/troughs being
compared -- the lowest low in between for a bearish divergence (the
pullback low), the highest high in between for a bullish divergence
(the bounce high) -- same definition and confirmation style as
backtest/macd_divergence_hump_options.py's require_neckline_breakout.
A divergence is reported FORMING as soon as it's confirmed, and
CONFIRMED once a later day's close actually breaks the neckline
(below it for bearish, above it for bullish) -- matching how the
geometric chart-pattern screener reports "forming" vs "confirmed".

DATA: upstox_client.get_daily_history per stock (public, no auth --
same endpoint used for the index elsewhere in this codebase), ~200
calendar days of lookback for comfortable MACD + hump warm-up.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date, timedelta

from data_sources import instruments, upstox_client
from backtest.technical_rating import _macd
from backtest.technical_rating_rotation import NIFTY_50

PEAK_WINDOW = 1
LOOKBACK_DAYS = 200  # calendar days of history fetched per stock
STALE_DAYS = 15  # a pending (unconfirmed) divergence older than this is dropped


@dataclass
class Bar:
    ts: str
    o: float
    h: float
    l: float
    c: float


@dataclass
class DivergenceHit:
    symbol: str
    direction: str  # "bullish" (buy-side) or "bearish" (sell-side)
    status: str     # "forming" or "confirmed"
    peak1_date: str
    peak1_price: float
    peak2_date: str
    peak2_price: float
    neckline: float
    bar_date: str   # the day this hit is being reported as of (peak2's day if forming, breakout day if confirmed)
    close: float
    window_start: int = 0
    window_end: int = -1
    markers: list = field(default_factory=list)


def _recent_bars(instrument_key: str) -> list[Bar]:
    from_date = (date.today() - timedelta(days=LOOKBACK_DAYS)).isoformat()
    to_date = (date.today() - timedelta(days=1)).isoformat()
    rows = upstox_client.get_daily_history(instrument_key, from_date, to_date)
    return [Bar(ts=r["date"], o=r["open"], h=r["high"], l=r["low"], c=r["close"]) for r in rows]


def _scan(symbol: str, bars: list[Bar]) -> list[DivergenceHit]:
    closes = [b.c for b in bars]
    macd_line, signal_line = _macd(closes, 12, 26, 9)
    histogram = [(m - s) if (m is not None and s is not None) else None for m, s in zip(macd_line, signal_line)]

    hits: list[DivergenceHit] = []
    hump_sign = 0
    hump_bars: list[tuple[int, float]] = []
    last_peak: tuple[int, float, float] | None = None
    last_trough: tuple[int, float, float] | None = None
    pending: dict | None = None  # {"direction","idx_prev","idx_mid","neckline"}

    for i in range(len(bars)):
        # -- neckline breakout check for a pending divergence --
        if pending is not None:
            if i - pending["idx_mid"] > STALE_DAYS:
                pending = None
            else:
                broke = (
                    (pending["direction"] == "bullish" and bars[i].c > pending["neckline"])
                    or (pending["direction"] == "bearish" and bars[i].c < pending["neckline"])
                )
                if broke:
                    idx_prev, idx_mid = pending["idx_prev"], pending["idx_mid"]
                    hits.append(DivergenceHit(
                        symbol=symbol, direction=pending["direction"], status="confirmed",
                        peak1_date=bars[idx_prev].ts, peak1_price=pending["price_prev"],
                        peak2_date=bars[idx_mid].ts, peak2_price=pending["price_mid"],
                        neckline=pending["neckline"], bar_date=bars[i].ts, close=bars[i].c,
                        window_start=max(0, idx_prev - 3), window_end=-1,
                        markers=[
                            {"index": idx_prev, "label": "1", "price": pending["price_prev"]},
                            {"index": idx_mid, "label": "2", "price": pending["price_mid"]},
                            {"index": i, "label": "Breakout", "price": bars[i].c},
                        ],
                    ))
                    pending = None

        h = histogram[i]
        if h is None or h == 0:
            continue
        sign = 1 if h > 0 else -1
        if sign != hump_sign:
            hump_sign = sign
            hump_bars = []
            last_peak = None
            last_trough = None
        hump_bars.append((i, h))

        window_len = 2 * PEAK_WINDOW + 1
        if len(hump_bars) < window_len:
            continue
        window_slice = hump_bars[-window_len:]
        idx_mid, val_mid = window_slice[PEAK_WINDOW]
        values = [v for _, v in window_slice]

        if hump_sign == 1 and val_mid == max(values) and values.count(val_mid) == 1:
            price_high_mid = bars[idx_mid].h
            if last_peak is not None:
                idx_prev, prev_val, prev_price_high = last_peak
                if val_mid < prev_val and price_high_mid >= prev_price_high:
                    neckline = min(b.l for b in bars[idx_prev:idx_mid + 1])
                    pending = {
                        "direction": "bearish", "idx_prev": idx_prev, "idx_mid": idx_mid,
                        "price_prev": prev_price_high, "price_mid": price_high_mid, "neckline": neckline,
                    }
                    hits.append(DivergenceHit(
                        symbol=symbol, direction="bearish", status="forming",
                        peak1_date=bars[idx_prev].ts, peak1_price=prev_price_high,
                        peak2_date=bars[idx_mid].ts, peak2_price=price_high_mid,
                        neckline=neckline, bar_date=bars[idx_mid].ts, close=bars[idx_mid].c,
                        window_start=max(0, idx_prev - 3), window_end=-1,
                        markers=[
                            {"index": idx_prev, "label": "1", "price": prev_price_high},
                            {"index": idx_mid, "label": "2", "price": price_high_mid},
                        ],
                    ))
            last_peak = (idx_mid, val_mid, price_high_mid)

        elif hump_sign == -1 and val_mid == min(values) and values.count(val_mid) == 1:
            price_low_mid = bars[idx_mid].l
            if last_trough is not None:
                idx_prev, prev_val, prev_price_low = last_trough
                if val_mid > prev_val and price_low_mid <= prev_price_low:
                    neckline = max(b.h for b in bars[idx_prev:idx_mid + 1])
                    pending = {
                        "direction": "bullish", "idx_prev": idx_prev, "idx_mid": idx_mid,
                        "price_prev": prev_price_low, "price_mid": price_low_mid, "neckline": neckline,
                    }
                    hits.append(DivergenceHit(
                        symbol=symbol, direction="bullish", status="forming",
                        peak1_date=bars[idx_prev].ts, peak1_price=prev_price_low,
                        peak2_date=bars[idx_mid].ts, peak2_price=price_low_mid,
                        neckline=neckline, bar_date=bars[idx_mid].ts, close=bars[idx_mid].c,
                        window_start=max(0, idx_prev - 3), window_end=-1,
                        markers=[
                            {"index": idx_prev, "label": "1", "price": prev_price_low},
                            {"index": idx_mid, "label": "2", "price": price_low_mid},
                        ],
                    ))
            last_trough = (idx_mid, val_mid, price_low_mid)

    return hits


def run(recent_days: int = 10) -> list[DivergenceHit]:
    """Only returns hits whose bar_date falls within the last `recent_days`
    calendar days -- a screener should report what's actionable NOW, not
    every divergence in the whole lookback window."""
    keys = instruments.resolve_symbols(NIFTY_50)
    cutoff = (date.today() - timedelta(days=recent_days)).isoformat()
    hits: list[DivergenceHit] = []
    for symbol in NIFTY_50:
        key = keys.get(symbol)
        if key is None:
            continue
        try:
            bars = _recent_bars(key)
        except Exception as exc:
            print(f"  {symbol}: skipped ({exc})", file=sys.stderr)
            continue
        if len(bars) < 60:
            continue
        for hit in _scan(symbol, bars):
            if hit.bar_date >= cutoff:
                hits.append(hit)
    return hits


def summary(hits: list[DivergenceHit]) -> str:
    if not hits:
        return "No MACD divergences found across NIFTY 50 (daily) in the recent window."
    confirmed = [h for h in hits if h.status == "confirmed"]
    forming = [h for h in hits if h.status == "forming"]
    lines = [f"{len(hits)} MACD divergence hit(s) across NIFTY 50 (daily):", ""]
    if confirmed:
        lines.append(f"CONFIRMED (neckline broken) ({len(confirmed)}):")
        for h in sorted(confirmed, key=lambda x: (x.direction, x.symbol)):
            lines.append(
                f"  {h.symbol:<12} {h.direction:<8} peak1 {h.peak1_date}={h.peak1_price:.2f}  "
                f"peak2 {h.peak2_date}={h.peak2_price:.2f}  neckline={h.neckline:.2f}  "
                f"broke {h.bar_date} close={h.close:.2f}"
            )
        lines.append("")
    if forming:
        lines.append(f"FORMING, neckline not yet broken ({len(forming)}):")
        for h in sorted(forming, key=lambda x: (x.direction, x.symbol)):
            lines.append(
                f"  {h.symbol:<12} {h.direction:<8} peak1 {h.peak1_date}={h.peak1_price:.2f}  "
                f"peak2 {h.peak2_date}={h.peak2_price:.2f}  watch neckline={h.neckline:.2f}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    results = run()
    print(summary(results))
