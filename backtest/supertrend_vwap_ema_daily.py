"""SuperTrend + VWAP cross + EMA trend filter on DAILY bars, for a
single instrument (e.g. a NIFTY 50 constituent stock's NSE_EQ cash
candles).

WHY A SEPARATE MODULE FROM backtest/supertrend_vwap_cross_equity.py:
that module is intraday -- SuperTrend and the VWAP cross are computed
on resampled candle_minutes bars WITHIN each trading day, VWAP resets
to zero every day (a session VWAP), and every position is forced flat
before the close. None of that carries over to a daily bar: a single
day's OHLC candle has no "VWAP" of its own at all (VWAP needs the
intraday distribution of trades within a bar; a daily close is just
one number per day), and a daily-bar position is a genuine multi-day
SWING hold by construction -- there is no same-day force-flat here,
exactly like backtest/rsi_daily_reversion.py.

VWAP SUBSTITUTE: a rolling vwap_lookback_days-wide volume-weighted
average price stands in for the intraday session VWAP's regime-filter
role -- sum(typical price * volume) over the trailing window (ending
at and including the current day, same as the intraday version
including the current partial session), divided by summed volume over
that window, recomputed fresh each day as the window rolls forward.
Where the intraday version resets to a brand new VWAP every trading
day, this rolls a fixed-width window forward one day at a time -- same
role (a volume-weighted reference line whose cross flips bias),
different timescale.

SIGNAL: identical mechanics to the intraday versions -- SuperTrend
(st_period, st_multiplier) direction + a fresh rolling-VWAP cross,
filtered by ema_filter_period (now an EMA of DAILY closes, instead of
an EMA of intraday bar closes), chop_lookback_days/chop_min_efficiency,
and narrow_cpr_max_width_pct (both already computed from the stock's
own daily bars in the intraday modules, so only the signal's own bar
size has changed, not these two filters' math). Every filter still
needs its own from-scratch calibration for THIS bar size and
instrument -- nothing here is assumed to carry over from any intraday
Reliance/NIFTY/Bank Nifty tune, or from the daily RSI reversion tune.
All filters default to None/off; sl_points and target_points too.

EXIT: trend reversal (SuperTrend flips against the position) or
sl_points/target_points (in the stock's own price points) if set. No
forced-flat -- a position can carry indefinitely until one of these
fires or the data ends.

COSTS: reuses backtest.rsi2_reversion's futures-notional cost
approximation, same caveat as every other daily-bar module in this
codebase (real delivery-style equity costs differ somewhat).
"""
from __future__ import annotations

from backtest.ema8_13_trend_sweep_options import _ema
from backtest.rsi2_reversion import Trade
from backtest.rsi_daily_reversion import _get_daily_history_chunked
from backtest.supertrend_pivot_options import _compute_supertrend_line


def _rolling_vwap(bars: list[list], lookback_days: int) -> list[float | None]:
    """Rolling lookback_days-wide VWAP ending at and including bar i:
    sum(typical price * volume) / sum(volume) over that trailing window."""
    n = len(bars)
    pv = [0.0] * n
    vol = [0.0] * n
    for i, b in enumerate(bars):
        _, o, h, l, c, v, _ = b
        typical = (h + l + c) / 3
        pv[i] = typical * v
        vol[i] = v
    out: list[float | None] = [None] * n
    for i in range(n):
        start = i - lookback_days + 1
        if start < 0:
            continue
        window_vol = sum(vol[start:i + 1])
        out[i] = (sum(pv[start:i + 1]) / window_vol) if window_vol > 0 else None
    return out


def run(
    from_date: str,
    to_date: str,
    instrument_key: str,
    quantity: int = 100,
    st_period: int = 10,
    st_multiplier: float = 3.0,
    vwap_lookback_days: int = 20,
    sl_points: float | None = None,
    target_points: float | None = None,
    ema_filter_period: int | None = None,
    chop_lookback_days: int | None = None,
    chop_min_efficiency: float | None = None,
    narrow_cpr_max_width_pct: float | None = None,
    long_only: bool = False,
) -> list[Trade]:
    days = _get_daily_history_chunked(instrument_key, from_date, to_date)
    days.sort(key=lambda d: d["date"])
    if len(days) < st_period + vwap_lookback_days + 2:
        return []

    bars = [[d["date"], d["open"], d["high"], d["low"], d["close"], d["volume"], d["oi"]] for d in days]
    closes = [b[4] for b in bars]

    st_dir, _st_line = _compute_supertrend_line(bars, st_period, st_multiplier)
    vwap = _rolling_vwap(bars, vwap_lookback_days)
    ema_filter = _ema(closes, ema_filter_period) if ema_filter_period else None

    chop_skip_days: set[str] = set()
    if chop_lookback_days is not None and chop_min_efficiency is not None:
        for i in range(chop_lookback_days, len(bars)):
            window = closes[i - chop_lookback_days:i]
            net_move = abs(window[-1] - window[0])
            abs_moves = sum(abs(window[k] - window[k - 1]) for k in range(1, len(window)))
            efficiency = net_move / abs_moves if abs_moves > 0 else 0.0
            if efficiency < chop_min_efficiency:
                chop_skip_days.add(bars[i][0])

    narrow_cpr_skip_days: set[str] = set()
    if narrow_cpr_max_width_pct is not None:
        for i in range(1, len(bars)):
            _, _, h, l, c, _, _ = bars[i - 1]
            pivot = (h + l + c) / 3
            bc = (h + l) / 2
            tc = 2 * pivot - bc
            width_pct = abs(tc - bc) / c * 100 if c else 0.0
            if width_pct > narrow_cpr_max_width_pct:
                narrow_cpr_skip_days.add(bars[i][0])

    trades: list[Trade] = []
    position: Trade | None = None
    prev_close = prev_vwap = None

    for i, b in enumerate(bars):
        d, o, h, l, c, v, oi = b

        if position is not None:
            cur_dir = st_dir[i]
            is_long = position.direction == "LONG"
            hit_reversal = cur_dir is not None and ((is_long and cur_dir == -1) or (not is_long and cur_dir == 1))
            hit_sl = sl_points is not None and (
                (is_long and c <= position.entry_price - sl_points)
                or (not is_long and c >= position.entry_price + sl_points)
            )
            hit_target = target_points is not None and (
                (is_long and c >= position.entry_price + target_points)
                or (not is_long and c <= position.entry_price - target_points)
            )
            exit_reason = "stop_loss" if hit_sl else ("target" if hit_target else ("trend_reverse" if hit_reversal else None))
            if exit_reason:
                position.exit_time = d
                position.exit_price = c
                position.exit_reason = exit_reason
                trades.append(position)
                position = None

        if (position is None and vwap[i] is not None and prev_vwap is not None and prev_close is not None
                and st_dir[i] is not None and d not in chop_skip_days and d not in narrow_cpr_skip_days):
            crossed_above = prev_close <= prev_vwap and c > vwap[i]
            crossed_below = prev_close >= prev_vwap and c < vwap[i]
            direction_label = None
            if st_dir[i] == 1 and crossed_above:
                direction_label = "LONG"
            elif st_dir[i] == -1 and crossed_below and not long_only:
                direction_label = "SHORT"

            if direction_label is not None and ema_filter is not None:
                ema_val = ema_filter[i]
                if ema_val is None:
                    direction_label = None
                elif direction_label == "LONG" and c <= ema_val:
                    direction_label = None
                elif direction_label == "SHORT" and c >= ema_val:
                    direction_label = None

            if direction_label is not None:
                position = Trade(date=d, direction=direction_label, entry_time=d, entry_price=c, lot_size=quantity)

        prev_close, prev_vwap = c, vwap[i]

    if position is not None:
        position.exit_time = bars[-1][0]
        position.exit_price = bars[-1][4]
        position.exit_reason = "data_end"
        trades.append(position)

    return trades


RELIANCE_EQUITY_KEY = "NSE_EQ|INE002A01018"


def run_reliance(from_date: str, to_date: str, **overrides) -> list[Trade]:
    """run() with Reliance's own fine-tuned daily SuperTrend+VWAP config,
    found by sweeping from scratch on the 2015-01-01/2026-09-08 daily
    history.

    BOTH-SIDES signal: every (st_period, st_multiplier) combination
    tested at vwap_lookback_days=20 was net negative or barely positive,
    with the same long/short asymmetry seen in the NIFTY futures module
    (longs positive, shorts deeply negative). Raising vwap_lookback_days
    on the both-sides signal helped up to a point (30-60 days climbed
    from Rs 15,063 to Rs 70,553) then reversed (100+ days went negative
    again) -- not a usable lever on its own, and shorts stayed the drag.

    LONG_ONLY: turns the signal solidly positive almost everywhere --
    24 of 25 (st_period, st_multiplier) combinations tested were net
    positive at vwap_lookback_days=20 alone. Extending vwap_lookback_days
    to 40-70 with long_only on produces a genuine, wide plateau: every
    cell across st_period {7,10} x st_multiplier {2.5,3.0,3.5} x
    vwap_lookback_days {40,50,60,70} nets Rs 11,000-109,000. A finer grid
    around the best point (st_period 8-13, vwap_lookback_days 42-58)
    confirmed it's a real plateau, not a spike -- net P&L moves smoothly
    (Rs 72,000-119,000) and max drawdown is pinned at the same -Rs 15,958
    (the single worst trade) across the whole neighborhood -- no cliff
    edges. st_period=12, st_multiplier=3.0, vwap_lookback_days=50 is the
    best point in that plateau.

    EMA/chop/narrow-CPR filters were all tried on top of this base and
    REJECTED: ema_filter_period only ever matches or weakens the
    unfiltered result (same at 20-45, drops to Rs 22,000-47,000 at
    60+). chop and narrow-CPR each only remove 3-10 of the already-few
    21 trades and move the result non-monotonically as their own
    parameters vary (e.g. chop_lookback_days 10->15->20 at a fixed
    efficiency bounces Rs 129,741 -> 92,906 -> 105,729) -- at this
    sample size that bounce reads as noise, not a plateau, so neither
    filter is trusted; both stay off.

    Final: st_period=12, st_multiplier=3.0, vwap_lookback_days=50,
    long_only=True -- 21 trades, net Rs 118,608.26, 61.9% win rate, max
    drawdown -Rs 15,958.37 (net/drawdown ratio ~7.4, the best of any
    strategy tuned in this codebase so far). Caveat: 21 trades over 11.7
    years (~70-day average hold -- a genuine swing position, not a
    scalp) is a small sample; trust the robustness of the PARAMETER
    PLATEAU more than the exact P&L number, and treat this as a
    promising, not yet fully battle-tested, result.
    """
    overrides.setdefault("st_period", 12)
    overrides.setdefault("st_multiplier", 3.0)
    overrides.setdefault("vwap_lookback_days", 50)
    overrides.setdefault("long_only", True)
    return run(from_date, to_date, RELIANCE_EQUITY_KEY, **overrides)


def summary(trades: list[Trade]) -> str:
    from backtest.rsi2_reversion import summary as _summary
    return _summary(trades)
