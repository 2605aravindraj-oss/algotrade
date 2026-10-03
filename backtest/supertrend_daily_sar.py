"""Pure SuperTrend stop-and-reverse (SAR) on DAILY bars, for a single
instrument (e.g. a NIFTY 50 constituent stock's NSE_EQ cash candles).

WHY A SEPARATE MODULE FROM backtest/supertrend.py: that module trades
5-minute NIFTY FUTURES bars. This is the same stop-and-reverse rule --
always in the market, flip long/short on every SuperTrend direction
change, one position at a time -- just on a single stock's own daily
candles instead. No VWAP filter, no EMA filter, nothing else gating
entry: unlike backtest/supertrend_vwap_ema_daily.py (which only enters
on a SuperTrend+VWAP-cross agreement and can sit flat), this is always
in the market from the moment SuperTrend settles, exactly as classic
SuperTrend SAR is normally traded.

SIGNAL: SuperTrend(period, multiplier), Wilder ATR, standard recursive
band formula (see backtest/supertrend.py's docstring for the full
derivation -- identical math, just fed daily OHLCV instead of 5-minute
bars).

TRADING RULE: direction flips bullish -> exit any short, go LONG at
that day's close. Direction flips bearish -> exit any long, go SHORT
at that day's close. A position carries indefinitely across trading
days (no forced-flat, like every other daily-bar module in this
codebase) until the next opposite flip or the data ends.

COSTS: reuses backtest.rsi2_reversion's futures-notional cost
approximation, same caveat as every other daily-bar module here.
"""
from __future__ import annotations

from backtest.rsi2_reversion import Trade
from backtest.rsi_daily_reversion import _get_daily_history_chunked
from backtest.supertrend_pivot_options import _compute_supertrend_line


def run(
    from_date: str,
    to_date: str,
    instrument_key: str,
    quantity: int = 100,
    period: int = 10,
    multiplier: float = 3.0,
) -> list[Trade]:
    days = _get_daily_history_chunked(instrument_key, from_date, to_date)
    days.sort(key=lambda d: d["date"])
    if len(days) < period + 2:
        return []

    bars = [[d["date"], d["open"], d["high"], d["low"], d["close"], d["volume"], d["oi"]] for d in days]
    direction, _line = _compute_supertrend_line(bars, period, multiplier)

    trades: list[Trade] = []
    position: Trade | None = None
    prev_dir: int | None = None

    def _close(t: Trade, d: str, price: float, reason: str) -> None:
        t.exit_time = d
        t.exit_price = price
        t.exit_reason = reason
        trades.append(t)

    for i, bar in enumerate(bars):
        d, o, h, l, c, v, oi = bar
        dirn = direction[i]
        if dirn is None:
            continue

        if prev_dir is not None and dirn != prev_dir:
            if position is not None:
                _close(position, d, c, "reverse")
            new_direction = "LONG" if dirn == 1 else "SHORT"
            position = Trade(date=d, direction=new_direction, entry_time=d, entry_price=c, lot_size=quantity)

        prev_dir = dirn

    if position is not None:
        _close(position, bars[-1][0], bars[-1][4], "data_end")

    return trades


RELIANCE_EQUITY_KEY = "NSE_EQ|INE002A01018"


def run_reliance(from_date: str, to_date: str, **overrides) -> list[Trade]:
    """run() against Reliance's own NSE_EQ instrument. No tuned defaults
    yet -- this wrapper exists so a tuned config has a stable place to
    land once a from-scratch sweep is done."""
    return run(from_date, to_date, RELIANCE_EQUITY_KEY, **overrides)


def summary(trades: list[Trade]) -> str:
    from backtest.rsi2_reversion import summary as _summary
    return _summary(trades)
