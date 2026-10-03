"""Classic Larry Connors-style RSI mean reversion on DAILY bars, for a
single instrument (e.g. a NIFTY 50 constituent stock's NSE_EQ cash
candles).

WHY A SEPARATE MODULE FROM backtest/rsi2_reversion.py: that module is
intraday (1-minute bars, reset every day, forced flat before close,
a same-day VWAP regime filter). None of that carries over to a daily
bar: a daily "VWAP" isn't a meaningful regime filter the way an
intraday session VWAP is, and a daily-bar position is a multi-day
SWING/positional hold by construction, not an intraday scalp -- there
is no same-day force-flat here at all; a position rides until its own
exit condition fires, however many calendar days that takes.

SIGNAL: Wilder RSI(rsi_period) on daily closes (reusing rsi2_
reversion.compute_rsi -- the same calculation, just fed daily closes
instead of 1-minute ones).
    LONG entry:  RSI < oversold
    SHORT entry: RSI > 100 - oversold (mirror), only if allow_short
Both are pure mean-reversion entries with no trend/regime filter
layered on top -- deliberately the simplest version first, to see if
there's a bare edge before adding anything. (The classic Connors
rule also requires price above a long SMA before taking the long
side; that is NOT included here, and would be a natural next
parameter to sweep if the bare version shows promise.)

EXIT (checked in this order): stop_loss_pct (if set) -> target_pct
(if set, a fixed profit target as a percentage of entry price,
checked before the RSI exit so it can lock in a move RSI hasn't
caught up to yet) -> RSI crosses back through exit_threshold (LONG:
RSI > exit_threshold; SHORT: RSI < 100 - exit_threshold) ->
max_hold_days (if set, calendar days since entry, not trading days).
No forced-flat -- a position can carry indefinitely until one of
these fires or the data ends. stop_loss_pct, target_pct and
max_hold_days all default to None (off) -- the tuned Reliance config
relies on the RSI exit alone; see run_reliance_tuned's docstring for
why all three were tried and rejected.

COSTS: reuses backtest.rsi2_reversion's futures-notional cost
approximation (flat brokerage + ~0.0255% of notional round trip) as
a stand-in for equity costs -- real delivery-style equity costs
differ (STT is higher, 0.1% sell-side, since a multi-day hold
settles as delivery, not an intraday MIS square-off; many discount
brokers charge zero brokerage on delivery). Treat cost figures here
as approximate, same caveat as every other cost model in this
codebase.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from data_sources import upstox_client
from backtest.rsi2_reversion import Trade, compute_rsi

_MAX_CHUNK_DAYS = 365 * 8  # the daily historical-candle endpoint rejects spans beyond ~9-10 years


def _get_daily_history_chunked(instrument_key: str, from_date: str, to_date: str) -> list[dict]:
    """upstox_client.get_daily_history errors (400) on a span much
    beyond ~9-10 years -- a daily RSI strategy needs more history than
    that for a meaningful sample, so fetch in <=8-year chunks and
    concatenate (deduping by date, in case chunk boundaries overlap)."""
    start = datetime.strptime(from_date, "%Y-%m-%d")
    end = datetime.strptime(to_date, "%Y-%m-%d")
    by_date: dict[str, dict] = {}
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=_MAX_CHUNK_DAYS), end)
        chunk = upstox_client.get_daily_history(
            instrument_key, chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        )
        for d in chunk:
            by_date[d["date"]] = d
        chunk_start = chunk_end + timedelta(days=1)
    return list(by_date.values())


def run(
    from_date: str,
    to_date: str,
    instrument_key: str,
    rsi_period: int = 14,
    oversold: float = 30,
    exit_threshold: float = 50,
    stop_loss_pct: float | None = None,
    target_pct: float | None = None,
    max_hold_days: int | None = None,
    quantity: int = 100,
    allow_short: bool = False,
) -> list[Trade]:
    days = _get_daily_history_chunked(instrument_key, from_date, to_date)
    days.sort(key=lambda d: d["date"])
    if len(days) < rsi_period + 5:
        return []

    closes = [d["close"] for d in days]
    rsi = compute_rsi(closes, rsi_period)

    trades: list[Trade] = []
    position: Trade | None = None

    for i, day in enumerate(days):
        d = day["date"]
        price = closes[i]
        if rsi[i] is None:
            continue

        if position is not None:
            exit_reason = None
            is_long = position.direction == "LONG"
            if stop_loss_pct is not None:
                hit_sl = (
                    (is_long and price <= position.entry_price * (1 - stop_loss_pct / 100))
                    or (not is_long and price >= position.entry_price * (1 + stop_loss_pct / 100))
                )
                if hit_sl:
                    exit_reason = "stop_loss"
            if exit_reason is None and target_pct is not None:
                hit_target = (
                    (is_long and price >= position.entry_price * (1 + target_pct / 100))
                    or (not is_long and price <= position.entry_price * (1 - target_pct / 100))
                )
                if hit_target:
                    exit_reason = "target"
            if exit_reason is None:
                if is_long and rsi[i] > exit_threshold:
                    exit_reason = "rsi_exit"
                elif not is_long and rsi[i] < (100 - exit_threshold):
                    exit_reason = "rsi_exit"
            if exit_reason is None and max_hold_days is not None:
                held_days = (datetime.strptime(d, "%Y-%m-%d") - datetime.strptime(position.date, "%Y-%m-%d")).days
                if held_days >= max_hold_days:
                    exit_reason = "max_hold"
            if exit_reason:
                position.exit_time = d
                position.exit_price = price
                position.exit_reason = exit_reason
                trades.append(position)
                position = None
            continue

        if rsi[i] < oversold:
            position = Trade(date=d, direction="LONG", entry_time=d, entry_price=price, lot_size=quantity)
        elif allow_short and rsi[i] > (100 - oversold):
            position = Trade(date=d, direction="SHORT", entry_time=d, entry_price=price, lot_size=quantity)

    if position is not None:
        position.exit_time = days[-1]["date"]
        position.exit_price = closes[-1]
        position.exit_reason = "data_end"
        trades.append(position)

    return trades


RELIANCE_EQUITY_KEY = "NSE_EQ|INE002A01018"


def run_reliance_tuned(from_date: str, to_date: str, **overrides) -> list[Trade]:
    """run() with Reliance's own fine-tuned daily RSI config, found
    by sweeping rsi_period x oversold on the 2015-01-01/2026-09-08
    daily history (11.7 years -- long history matters here, since a
    daily mean-reversion dip is infrequent; 2 years gave too few
    trades to judge). {12-20} x {35-45} is a genuinely wide,
    structural plateau (every combination net Rs 37,000-101,000, win
    rates 73-88%), not an isolated spike -- short RSI periods (2-3)
    and low oversold thresholds (10-20, a "deep" oversold read) were
    mostly flat or negative; the edge is in a SLOWER RSI catching a
    MILDER dip. rsi_period=16, oversold=38 is the single best point
    in that plateau: 42 trades, net Rs 101,263, 81.0% win rate, max
    drawdown -Rs 8,916 (equal to the single worst trade -- there was
    never a losing STREAK, just isolated losses). exit_threshold=50
    was independently confirmed best in its own 1D sweep at this
    period/oversold (vs 40-70). stop_loss_pct, target_pct and
    max_hold_days were not swept yet at the time -- left at their
    defaults (off) since the unfiltered result was already strong.
    All three were swept later; see below -- all three stay off.

    stop_loss_pct and target_pct (both off by design -- here's why).
    stop_loss_pct in {2,3,4,5,7,10,15}: every value made max drawdown
    dramatically WORSE (-Rs 21,000 to -Rs 32,000+, vs the unfiltered
    -Rs 8,916) for flat-to-negative net P&L -- a tight stop on a
    mean-reversion dip just converts a trade that would have
    recovered into a realized loss, and cranks trade count up
    (47-79 vs 42) without adding edge. target_pct in
    {3,3.5,4,4.5,5,5.5,6,6.5,7,10,15,20}: net P&L is flat at baseline
    (Rs 101,263) for every value except a single spike at exactly 5
    (Rs 106,590). That spike is NOT a genuine plateau -- target=4.5
    (Rs 99,376) and target=5.5 (Rs 101,038) both sit BELOW it, and an
    entry-matched trade diff (by entry_time, not list index, since
    trade counts differ) showed the gain is ~95% one incidental
    re-entry side effect: a position that exits one day earlier frees
    the single-position slot for an unrelated extra trade the
    baseline never takes, not a repeatable structural edge. Rejected
    for the same reason min_cross_distance_points/require_hold_bar
    were rejected in the NIFTY options module -- isolated spike, not
    a plateau.

    max_hold_days (also off by design). Swept {5,7,10,15,20,25,30,
    40,50,60,90} plus a finer {16..24} grid around an apparent bump
    at 20: EVERY tested value from 5 to 24 makes max drawdown
    substantially worse than the unfiltered -Rs 8,916 (ranging from
    -Rs 13,700 at the mildest cap tried up to -Rs 32,000+ at the
    tightest), while net P&L bounces around noisily above and below
    baseline with no stable improvement (e.g. 103,649 at cap=20,
    109,325 at cap=21, but 93,305 at cap=17 and 90,463 at cap=23 --
    no plateau, just noise). Forcing an early exit on a still-open
    position converts a trade that was heading toward recovery into
    a smaller/negative realized one, the same mechanism that makes
    stop_loss_pct harmful here. Values >=40 converge back toward the
    unfiltered baseline as fewer trades get capped at all (0 capped
    at 90, exactly matching the unfiltered result) -- consistent with
    the cap simply doing damage whenever it actually binds, never
    adding value. All three of stop_loss_pct, target_pct and
    max_hold_days stay off; this strategy works because its mean-
    reversion exits are allowed to run their full course.
    """
    overrides.setdefault("rsi_period", 16)
    overrides.setdefault("oversold", 38)
    overrides.setdefault("exit_threshold", 50)
    return run(from_date, to_date, RELIANCE_EQUITY_KEY, **overrides)


def summary(trades: list[Trade]) -> str:
    from backtest.rsi2_reversion import summary as _summary
    return _summary(trades)
