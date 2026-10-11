"""Backtest: NIFTY 50 spot candles, mark the EMA(9)/EMA(20) crossover
candle's own range, then switch to 1-minute candles and wait for a
sweep of that range's near edge followed by a breakout of its far
edge -- through real ATM NIFTY options, with a stop at the range's own
midpoint (by default) and a fixed 1:2 risk-reward target.

ZONE: computed on 5-minute NIFTY spot candles (not stated in the
request; chosen because the 1-minute switch only makes sense as a
FINER timeframe than whatever the EMA cross itself runs on, and
5-minute matches the EMA cloud pullback strategy's own convention
elsewhere in this codebase). The moment EMA(9) crosses above EMA(20)
(this bar's EMA9 > EMA20, the PRIOR bar's was <=) is a bullish cross;
that crossover candle's own [low, high] becomes the zone. The bearish
mirror (EMA9 crosses below EMA20) marks a zone the same way. EMA runs
CONTINUOUSLY across the whole backtest date range (not reseeded every
morning), same reasoning as order_block_options.py and
ema_cloud_pullback_options.py's own structure tracking -- only the
OPTION POSITION force-flattens daily.

Only ONE zone is tracked at a time, in either direction: a fresh cross
(either direction) replaces whatever zone was pending, discarding it
unresolved -- the most recent cross is what the chart shows as "the"
current zone. The zone becomes visible to the 1-minute series only
once its own 5-minute candle has actually closed (real close instant,
not just its bucket-start label) -- no lookahead.

ENTRY (sweep then breakout, in that order): on 1-minute bars after the
zone exists, a LONG zone is "swept" the first time a bar's LOW trades
at or below the zone's low (price dipped into/through the near edge).
Once swept, the trade fires the first bar (the same bar that just
swept, or a later one) whose CLOSE breaks above the zone's high --
the far edge. A straight breakout with no sweep first does not fire;
the setup just keeps waiting for one. The bearish mirror: swept when
a bar's high trades at or above the zone's high, fires on a close
below the zone's low. Entry fills at the breakout bar's own close
(spot), translated into an ATM option fill the same way as every
other module here.

EXIT: the stop and target are INDEX PRICE levels, not an option
premium percentage -- a chart level, not a Rs figure. stop_at_midpoint
(default True): the stop sits at the zone's own midpoint, (zone_low +
zone_high) / 2, rather than its far-from-entry edge -- tighter than
the original "stop at the swept edge" reading, since the breakout
entry already happens well past the midpoint by construction (the
sweep has to reach one edge, the breakout the other). Set
stop_at_midpoint=False to go back to the original edge-level stop
(zone_low for a LONG, zone_high for a SHORT). For a LONG: risk (index
points) = entry_close - stop_level; target_level = entry_close +
risk_reward * risk, a flat 1:risk_reward reward on the index itself
(default 1:2). The SHORT mirror is symmetric. Later 1-minute index
bars are scanned for the first one whose low/high trades through
stop_level or target_level; whichever triggers, the option's own
premium at that bar's decision time is what gets realized (same fill
convention as every other sl_rs-style module here), not a synthetic
index-based P&L. Forced flat at FORCE_FLAT_TIME (15:25) if neither
level is hit first. Only one position open at a time; zone tracking
continues underneath it, but no new entry fires until flat.

STRIKE/EXPIRY: ATM = round_to_step(index close, strike_step), nearest
expiry on/after the entry day. Uses the EXPIRED-instruments API (needs
an Upstox access token) for any day whose weekly contract has already
rolled over.

FIXED-RUPEE EXIT (sl_rs/target_rs, both None by default): when both are
set, they replace the index-level stop/target entirely -- the option
position exits on its own premium P&L per lot, checked each entry-
timeframe bar at that bar's decision time: pnl_per_lot = (premium -
entry_premium) * lot_size, stop at <= -sl_rs, target at >= target_rs.
Same convention as macd_histogram_turn_options.py and the other
sl_rs-style modules here. risk_reward/stop_at_midpoint are ignored
when this mode is active (stop_level/target_level are still computed
from the zone but unused).

TREND FILTER (trend_filter=False by default): when enabled, gates
entries by a longer EMA (trend_ema_period, default 45) computed on the
same 5-minute zone_bars series the cross itself runs on -- same
convention as supertrend_vwap_cross_options.py's own ema_filter_period.
A LONG only fires while the latest no-lookahead-safe 5-minute close is
above that trend EMA; a SHORT only fires while it's below. "Latest
known" uses the same real-close-instant cursor technique as
zones_available, so the filter never looks ahead. A signal that fails
the filter is NOT discarded -- the zone stays pending (already swept,
if it was) and keeps re-checking the breakout condition on later bars,
so it can still fire once the trend catches up.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, _bar_at_or_after, _bar_at_or_before
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.sweep_reclaim_breakout import _resample
from backtest.technical_rating import _ema

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"
ZONE_TIMEFRAME_MINUTES = 5
ENTRY_TIMEFRAME_MINUTES = 1
EMA_FAST = 9
EMA_SLOW = 20
RISK_REWARD = 2.0
TREND_EMA_PERIOD = 45


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


def _find_zones(zone_bars: list[Bar], ema_fast: int, ema_slow: int) -> list[tuple[int, str, float, float]]:
    """One pass over the zone-timeframe bars: returns
    [(bar_idx, direction, zone_low, zone_high)] for every EMA9/EMA20
    cross, in order."""
    closes = [b.c for b in zone_bars]
    ema9 = _ema(closes, ema_fast)
    ema20 = _ema(closes, ema_slow)
    zones = []
    for i in range(1, len(zone_bars)):
        f0, s0, f1, s1 = ema9[i - 1], ema20[i - 1], ema9[i], ema20[i]
        if f0 is None or s0 is None or f1 is None or s1 is None:
            continue
        if f0 <= s0 and f1 > s1:
            zones.append((i, "LONG", zone_bars[i].l, zone_bars[i].h))
        elif f0 >= s0 and f1 < s1:
            zones.append((i, "SHORT", zone_bars[i].l, zone_bars[i].h))
    return zones


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    zone_minutes: int = ZONE_TIMEFRAME_MINUTES,
    entry_minutes: int = ENTRY_TIMEFRAME_MINUTES,
    ema_fast: int = EMA_FAST,
    ema_slow: int = EMA_SLOW,
    risk_reward: float = RISK_REWARD,
    stop_at_midpoint: bool = True,
    trend_filter: bool = False,
    trend_ema_period: int = TREND_EMA_PERIOD,
    sl_rs: float | None = None,
    target_rs: float | None = None,
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

    def _decision_time(ts: str, minutes: int) -> str:
        if minutes <= 1:
            return ts[11:16]
        hh, mm = int(ts[11:13]), int(ts[14:16])
        dh, dm = divmod(hh * 60 + mm + minutes, 60)
        return f"{dh:02d}:{dm:02d}"

    all_1min: list[list] = []
    for day in trading_days:
        rows = sorted(cache.get_day_candles_cached(underlying_key, "1minute", day["date"], expired=False), key=lambda r: r[0])
        all_1min.extend(rows)
    all_1min.sort(key=lambda r: r[0])

    min_bars = max(ema_slow, trend_ema_period if trend_filter else 0) + 3
    if len(all_1min) < min_bars:
        return []

    zone_rows = _resample(all_1min, zone_minutes) if zone_minutes > 1 else all_1min
    if len(zone_rows) < min_bars:
        return []
    zone_bars = [Bar(ts=r[0], o=r[1], h=r[2], l=r[3], c=r[4]) for r in zone_rows]

    entry_rows = _resample(all_1min, entry_minutes) if entry_minutes > 1 else all_1min
    bars = [Bar(ts=r[0], o=r[1], h=r[2], l=r[3], c=r[4]) for r in entry_rows]

    zones = _find_zones(zone_bars, ema_fast, ema_slow)
    zones_available = sorted(
        ((datetime.fromisoformat(zone_bars[idx].ts) + timedelta(minutes=zone_minutes), direction, zlo, zhi)
         for idx, direction, zlo, zhi in zones),
        key=lambda x: x[0],
    )

    trend_available: list[tuple] = []
    if trend_filter:
        trend_closes = [b.c for b in zone_bars]
        trend_ema = _ema(trend_closes, trend_ema_period)
        trend_available = sorted(
            ((datetime.fromisoformat(zone_bars[idx].ts) + timedelta(minutes=zone_minutes), close, te)
             for idx, (close, te) in enumerate(zip(trend_closes, trend_ema)) if te is not None),
            key=lambda x: x[0],
        )

    trades: list[OptionTrade] = []
    zone_cursor = 0
    trend_cursor = 0
    trend_state: str | None = None  # "UP" / "DOWN", latest known as of this bar
    pending: dict | None = None  # {"direction","zone_low","zone_high","swept"}

    i = 0
    while i < len(bars):
        bar_date = bars[i].ts[:10]
        bar_dt = datetime.fromisoformat(bars[i].ts)
        decision_time_str = _decision_time(bars[i].ts, entry_minutes)
        skip_entries_today = decision_time_str >= FORCE_FLAT_TIME

        # a fresh cross replaces whatever zone was pending, even unresolved
        while zone_cursor < len(zones_available) and zones_available[zone_cursor][0] <= bar_dt:
            _, direction, zlo, zhi = zones_available[zone_cursor]
            pending = {"direction": direction, "zone_low": zlo, "zone_high": zhi, "swept": False}
            zone_cursor += 1

        while trend_cursor < len(trend_available) and trend_available[trend_cursor][0] <= bar_dt:
            _, t_close, t_ema = trend_available[trend_cursor]
            trend_state = "UP" if t_close > t_ema else "DOWN"
            trend_cursor += 1

        direction_label = opt_type = stop_level = target_level = None
        if pending is not None:
            lo, hi, close = bars[i].l, bars[i].h, bars[i].c
            trend_ok_long = (not trend_filter) or trend_state == "UP"
            trend_ok_short = (not trend_filter) or trend_state == "DOWN"
            if pending["direction"] == "LONG":
                if not pending["swept"] and lo <= pending["zone_low"]:
                    pending["swept"] = True
                if pending["swept"] and close > pending["zone_high"] and not skip_entries_today and trend_ok_long:
                    direction_label, opt_type = "LONG", "CE"
                    zone_mid = (pending["zone_low"] + pending["zone_high"]) / 2
                    stop_level = zone_mid if stop_at_midpoint else pending["zone_low"]
                    target_level = close + risk_reward * (close - stop_level)
                    pending = None
            else:
                if not pending["swept"] and hi >= pending["zone_high"]:
                    pending["swept"] = True
                if pending["swept"] and close < pending["zone_low"] and not skip_entries_today and trend_ok_short:
                    direction_label, opt_type = "SHORT", "PE"
                    zone_mid = (pending["zone_low"] + pending["zone_high"]) / 2
                    stop_level = zone_mid if stop_at_midpoint else pending["zone_high"]
                    target_level = close - risk_reward * (stop_level - close)
                    pending = None

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

        use_rupee_exit = sl_rs is not None and target_rs is not None

        exit_idx = None
        exit_reason = None
        for j in range(entry_idx + 1, day_end_idx + 1):
            j_decision_time = _decision_time(bars[j].ts, entry_minutes)
            if j_decision_time >= FORCE_FLAT_TIME:
                exit_idx = j
                exit_reason = "eod"
                break
            if use_rupee_exit:
                bar_j = _fill(opt_rows, j_decision_time)
                premium_j = bar_j[4] if bar_j else entry_price
                pnl_per_lot = (premium_j - entry_price) * lot_size
                if pnl_per_lot <= -sl_rs:
                    exit_idx, exit_reason = j, "stop_loss"
                    break
                if pnl_per_lot >= target_rs:
                    exit_idx, exit_reason = j, "target"
                    break
            elif direction_label == "LONG":
                if bars[j].l <= stop_level:
                    exit_idx, exit_reason = j, "stop_loss"
                    break
                if bars[j].h >= target_level:
                    exit_idx, exit_reason = j, "target"
                    break
            else:
                if bars[j].h >= stop_level:
                    exit_idx, exit_reason = j, "stop_loss"
                    break
                if bars[j].l <= target_level:
                    exit_idx, exit_reason = j, "target"
                    break

        if exit_idx is None:
            exit_idx = day_end_idx
            exit_reason = "eod"

        exit_decision_time = _decision_time(bars[exit_idx].ts, entry_minutes)
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
