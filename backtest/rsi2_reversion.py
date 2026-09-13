"""Intraday RSI(2) mean-reversion scalp on 1-minute bars.

Classic Larry Connors RSI(2) mean-reversion (from "Short-Term Trading
Strategies That Work"), adapted for intraday scalping and both directions:

- Compute a 2-period RSI on 1-minute closes, reset fresh every trading day
  (no signal carries over from the previous day).
- Use the day's cumulative VWAP as a regime filter: only take longs while
  price is above VWAP, only take shorts while below it.
- Long entry:  RSI(2) < oversold          AND close > VWAP
- Short entry: RSI(2) > 100 - oversold    AND close < VWAP
- Exit: RSI(2) crosses back through the exit threshold, a stop-loss, a
  max holding time, or forced flat before close (no overnight position).

This trades the underlying directly (index or stock), not options -- P&L
is reported in points and in rupees at a configurable lot size.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from data_sources import cache, upstox_client

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"
FORCE_FLAT_TIME = "15:25"

# Rough all-in round-trip cost estimate for NIFTY futures/index trading
# (NOT options -- STT/exchange charges are much lower than options' since
# STT applies only on the sell side and at a lower rate):
#   STT (sell side only): ~0.02%
#   exchange txn charges (both sides): ~0.0035%
#   stamp duty (buy side only): ~0.002%
#   GST/SEBI fee: negligible at this notional
# -> ~0.0255% of notional round trip, i.e. roughly a quarter of what the
# same rate would cost on an options premium-based trade.
BROKERAGE_PER_ORDER = 20.0
OTHER_COST_RATE = 0.000255  # ~0.0255% of notional, round trip


@dataclass
class Trade:
    date: str
    direction: str  # LONG | SHORT
    entry_time: str
    entry_price: float
    exit_time: str | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    lot_size: int = 65

    @property
    def pnl_points(self) -> float | None:
        if self.exit_price is None:
            return None
        return (self.exit_price - self.entry_price) if self.direction == "LONG" else (self.entry_price - self.exit_price)

    @property
    def pnl_rupees_gross(self) -> float | None:
        p = self.pnl_points
        return None if p is None else p * self.lot_size

    @property
    def cost_rupees(self) -> float:
        notional = ((self.entry_price + (self.exit_price or self.entry_price)) / 2) * self.lot_size
        return 2 * BROKERAGE_PER_ORDER + notional * OTHER_COST_RATE

    @property
    def pnl_rupees(self) -> float | None:
        g = self.pnl_rupees_gross
        return None if g is None else g - self.cost_rupees

    @property
    def hold_minutes(self) -> int | None:
        if self.exit_time is None:
            return None
        return _minutes_between(self.entry_time[11:16], self.exit_time[11:16])


def _minutes_between(t1: str, t2: str) -> int:
    h1, m1 = int(t1[:2]), int(t1[3:5])
    h2, m2 = int(t2[:2]), int(t2[3:5])
    return (h2 * 60 + m2) - (h1 * 60 + m1)


def compute_rsi(closes: list[float], period: int = 2) -> list[float | None]:
    """Wilder-smoothed RSI, index-aligned with closes. First `period` entries are None."""
    n = len(closes)
    rsi: list[float | None] = [None] * n
    if n <= period:
        return rsi
    gains = [0.0] * n
    losses = [0.0] * n
    for i in range(1, n):
        change = closes[i] - closes[i - 1]
        gains[i] = max(change, 0.0)
        losses[i] = max(-change, 0.0)

    avg_gain = sum(gains[1:period + 1]) / period
    avg_loss = sum(losses[1:period + 1]) / period

    def _rsi(ag: float, al: float) -> float:
        if al == 0:
            return 100.0
        return 100 - (100 / (1 + ag / al))

    rsi[period] = _rsi(avg_gain, avg_loss)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rsi[i] = _rsi(avg_gain, avg_loss)
    return rsi


def _session_vwap(rows: list[list]) -> list[float]:
    cum_vol = 0.0
    cum_pv = 0.0
    vwap = []
    for row in rows:
        _, o, h, l, c, v, _ = row
        typical = (h + l + c) / 3
        cum_vol += v
        cum_pv += typical * v
        vwap.append(cum_pv / cum_vol if cum_vol > 0 else typical)
    return vwap


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    rsi_period: int = 2,
    oversold: float = 10,
    overbought: float = 70,
    stop_loss_pct: float = 0.3,
    max_hold_minutes: int | None = 30,
    lot_size: int = 65,
    allow_short: bool = True,
    entry_cutoff: str = "15:00",
) -> list[Trade]:
    trading_days = upstox_client.get_daily_history(underlying_key, from_date, to_date)
    trades: list[Trade] = []

    for day in trading_days:
        d = day["date"]
        candles = cache.get_day_candles_cached(underlying_key, "1minute", d, expired=False)
        rows = sorted(candles, key=lambda c: c[0])
        if len(rows) < rsi_period + 5:
            continue

        closes = [r[4] for r in rows]
        vwap = _session_vwap(rows)
        rsi = compute_rsi(closes, rsi_period)

        position: Trade | None = None
        for i, row in enumerate(rows):
            ts = row[0]
            time_str = ts[11:16]
            price = closes[i]
            if rsi[i] is None:
                continue

            if position is not None:
                exit_reason = None
                if position.direction == "LONG":
                    if rsi[i] > overbought:
                        exit_reason = "rsi_exit"
                    elif price <= position.entry_price * (1 - stop_loss_pct / 100):
                        exit_reason = "stop_loss"
                else:
                    if rsi[i] < (100 - overbought):
                        exit_reason = "rsi_exit"
                    elif price >= position.entry_price * (1 + stop_loss_pct / 100):
                        exit_reason = "stop_loss"
                if exit_reason is None and max_hold_minutes and _minutes_between(position.entry_time[11:16], time_str) >= max_hold_minutes:
                    exit_reason = "max_hold"
                if exit_reason is None and time_str >= FORCE_FLAT_TIME:
                    exit_reason = "eod"
                if exit_reason:
                    position.exit_time = ts
                    position.exit_price = price
                    position.exit_reason = exit_reason
                    trades.append(position)
                    position = None
                continue

            if time_str >= entry_cutoff:
                continue
            if rsi[i] < oversold and price > vwap[i]:
                position = Trade(date=d, direction="LONG", entry_time=ts, entry_price=price, lot_size=lot_size)
            elif allow_short and rsi[i] > (100 - oversold) and price < vwap[i]:
                position = Trade(date=d, direction="SHORT", entry_time=ts, entry_price=price, lot_size=lot_size)

        if position is not None:
            position.exit_time = rows[-1][0]
            position.exit_price = rows[-1][4]
            position.exit_reason = "eod_data_end"
            trades.append(position)

    return trades


def summary(trades: list[Trade]) -> str:
    closed = [t for t in trades if t.pnl_rupees is not None]
    if not closed:
        return "No trades."
    pnls = [t.pnl_rupees for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    cum = 0.0
    equity = []
    for p in pnls:
        cum += p
        equity.append(cum)
    running_max = equity[0]
    max_dd = 0.0
    for e in equity:
        running_max = max(running_max, e)
        max_dd = min(max_dd, e - running_max)

    longs = [t for t in closed if t.direction == "LONG"]
    shorts = [t for t in closed if t.direction == "SHORT"]

    def _bucket(ts):
        pnl = [t.pnl_rupees for t in ts]
        w = [p for p in pnl if p > 0]
        return len(ts), (len(w) / len(ts) * 100 if ts else 0), sum(pnl)

    n_long, wr_long, pnl_long = _bucket(longs)
    n_short, wr_short, pnl_short = _bucket(shorts)

    avg_hold = sum(t.hold_minutes for t in closed) / len(closed)

    lines = [
        f"Total trades:     {len(closed)}",
        f"Net P&L:          Rs {sum(pnls):,.2f}",
        f"Win rate:         {len(wins) / len(closed) * 100:.1f}%  ({len(wins)}W / {len(losses)}L)",
        f"Avg win:          Rs {(sum(wins) / len(wins)) if wins else 0:,.2f}",
        f"Avg loss:         Rs {(sum(losses) / len(losses)) if losses else 0:,.2f}",
        f"Avg hold time:    {avg_hold:.1f} min",
        f"Best trade:       Rs {max(pnls):,.2f}",
        f"Worst trade:      Rs {min(pnls):,.2f}",
        f"Max drawdown:     Rs {max_dd:,.2f}",
        f"Long trades:      {n_long}  win rate {wr_long:.1f}%  net Rs {pnl_long:,.2f}",
        f"Short trades:     {n_short}  win rate {wr_short:.1f}%  net Rs {pnl_short:,.2f}",
    ]
    return "\n".join(lines)
