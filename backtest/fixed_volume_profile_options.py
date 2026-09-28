"""Fixed (prior-session) volume profile strategies, realized through
real ATM NIFTY options. Three genuinely different rule families built
on the same profile, selected via strategy_mode.

WHY NIFTY FUTURES, NOT THE INDEX: a volume profile needs real traded
volume at each price level. The NIFTY 50 INDEX itself carries volume=0
on every candle (it's a computed index, not a traded instrument) --
verified directly against live data before writing this. NIFTY
FUTURES do carry real volume (confirmed: ~3.7M contracts on a single
recent session), so the profile is built from the current front-month
futures contract's 1-minute candles, via the public (no-auth)
currently-listed endpoint. Futures price is also used for the
strategy's own signal comparisons AND for ATM strike rounding
(index and futures track closely; this is a simplification, not
exact -- a real trader watching a futures-based volume profile
intraday would do the same).

SCOPE LIMIT: unlike this codebase's other strategies, this one is NOT
tested across the 4 historical windows used elsewhere this session.
Backtesting a longer history needs a resolver for EXPIRED (rolled-off)
futures contracts, which doesn't exist in this codebase yet (unlike
options, there's no get_expired_futures_contract helper) -- building
one is future work. For now this only covers the current front-month
contract's own listing window (auto-resolved from Upstox's public
instrument master), roughly 2026-07-01 onward as of this writing.

STAGE 1 -- fixed volume profile, computed once per reference day D
(the PRIOR completed session) from that day's own futures 1-minute
candles, used unchanged for all of day D+1 (a "fixed", not
"developing", profile):
    Each 1-min bar's volume is assigned to ONE price bin -- the bin
    containing that bar's typical price ((High+Low+Close)/3), bin
    width = price_bin_size (default 10.0 points, an assumption; not
    volume split proportionally across the bar's own H-L range, which
    a stricter implementation would do).
    POC (Point of Control) = the bin with the most assigned volume.
    Value Area = POC's bin plus the adjacent bins added one at a time,
    always taking whichever side (above or below the current range)
    has more volume, until the included bins hold >= value_area_pct
    (default 70%, the standard convention) of the day's total volume.
    VAH/VAL = the top/bottom of that included range.

STAGE 2 -- trading day D+1, watched on candle_minutes (default 5)
futures candles, mode-specific:

    strategy_mode="poc_magnet": whenever price is more than
        poc_distance_points (default 60.0 -- roughly a day's typical
        half value-area width, an assumption) away from POC, bet on
        reversion back toward it: price > POC + distance -> SHORT
        (buy ATM PE); price < POC - distance -> LONG (buy ATM CE).
        Stop = entry +/- sl_points further away from POC (structural
        drift-continuation risk); target = POC itself.

    strategy_mode="vah_val_reversion": a candle wicks beyond VAH/VAL
        but CLOSES back inside it (the value area rejects the poke) --
        same sweep-reclaim shape as this codebase's other reclaim
        modules. High > VAH and Close <= VAH -> SHORT (buy PE); Low <
        VAL and Close >= VAL -> LONG (buy CE). Stop = the signal
        candle's own opposite extreme (structural); target = POC.

    strategy_mode="breakout": a candle CLOSES decisively outside the
        value area (not just a wick) -> trade continuation, betting
        the move accelerates into the low-volume area beyond. Close >
        VAH -> LONG (buy CE); Close < VAL -> SHORT (buy PE). Stop =
        the broken level itself (VAH/VAL -- back inside the value area
        means the breakout failed). Target = the breakout level +/-
        target_multiple (default 1.0) * the value area's own width
        (VAH - VAL) -- a standard "measured move" projection.

All modes: entry fills at the signal candle's own close, one position
at a time, a fresh signal while already in a trade is skipped, forced
flat at FORCE_FLAT_TIME, everything resets at the next day boundary
(a fresh profile is computed from THAT day's own close for the day
after). Stop checked before target if a single bar would touch both.
Fill is the option's own premium at that bar's time -- decision-time-
correct (bucket start + candle_minutes), same convention as every
other intraday module here.
"""
from __future__ import annotations

import gzip
import io
import json
from collections import defaultdict

import requests

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, _bar_at_or_after, _bar_at_or_before
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.sweep_reclaim_breakout import _resample

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"
INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"


def _resolve_current_month_futures() -> dict:
    resp = requests.get(INSTRUMENTS_URL, timeout=30)
    resp.raise_for_status()
    with gzip.open(io.BytesIO(resp.content)) as f:
        master = json.load(f)
    futs = [d for d in master if d.get("segment") == "NSE_FO" and d.get("underlying_symbol") == "NIFTY"
            and d.get("instrument_type") == "FUT"]
    if not futs:
        raise RuntimeError("no NIFTY futures contract found in instrument master")
    return min(futs, key=lambda d: d["expiry"])


def compute_profile(rows_1min: list[list], price_bin_size: float, value_area_pct: float) -> dict | None:
    """rows_1min: one day's sorted 1-min [ts,o,h,l,c,v,oi] rows for the
    futures contract. Returns {poc, vah, val, total_volume} or None if
    the day has no volume."""
    vol_by_bin: dict[float, float] = defaultdict(float)
    for row in rows_1min:
        ts, o, h, l, c, v, oi = row
        typical = (h + l + c) / 3
        b = round(typical / price_bin_size) * price_bin_size
        vol_by_bin[b] += v
    total = sum(vol_by_bin.values())
    if total <= 0:
        return None
    poc = max(vol_by_bin, key=vol_by_bin.get)
    included = {poc}
    cum = vol_by_bin[poc]
    while cum < value_area_pct * total:
        lo = min(included) - price_bin_size
        hi = max(included) + price_bin_size
        lo_v = vol_by_bin.get(lo, 0.0)
        hi_v = vol_by_bin.get(hi, 0.0)
        if lo_v == 0.0 and hi_v == 0.0:
            break
        if hi_v >= lo_v:
            included.add(hi)
            cum += hi_v
        else:
            included.add(lo)
            cum += lo_v
    return {"poc": poc, "vah": max(included), "val": min(included), "total_volume": total}


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    candle_minutes: int = 5,
    price_bin_size: float = 10.0,
    value_area_pct: float = 0.70,
    strategy_mode: str = "poc_magnet",
    poc_distance_points: float = 60.0,
    target_multiple: float = 1.0,
    sl_points: float = 40.0,
    access_token: str | None = None,
) -> list[OptionTrade]:
    if strategy_mode not in ("poc_magnet", "vah_val_reversion", "breakout"):
        raise ValueError('strategy_mode must be "poc_magnet", "vah_val_reversion", or "breakout"')

    futures = _resolve_current_month_futures()
    futures_key = futures["instrument_key"]

    fut_days = upstox_client.get_daily_history(futures_key, from_date, to_date)
    fut_days.sort(key=lambda d: d["date"])
    if len(fut_days) < 2:
        return []

    day_1min: dict[str, list[list]] = {}
    for day in fut_days:
        day_1min[day["date"]] = sorted(
            cache.get_day_candles_cached(futures_key, "1minute", day["date"], expired=False),
            key=lambda c: c[0],
        )

    profiles: dict[str, dict | None] = {
        d: compute_profile(rows, price_bin_size, value_area_pct) for d, rows in day_1min.items()
    }

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

    def _fill(candles, at_time):
        return _bar_at_or_after(candles, at_time) or _bar_at_or_before(candles, at_time)

    trades: list[OptionTrade] = []
    for i in range(1, len(fut_days)):
        d = fut_days[i]["date"]
        profile = profiles[fut_days[i - 1]["date"]]
        if profile is None:
            continue
        bars = _resample(day_1min[d], candle_minutes)
        if not bars:
            continue
        expiry = next((e for e in expiries if e >= d), None)
        if expiry is None:
            continue

        poc, vah, val = profile["poc"], profile["vah"], profile["val"]
        position = None  # dict: direction, entry_time, entry_price, strike, expiry, lot_size, opt_type, date, stop_level, target_level
        already_traded_today = False

        for row in bars:
            ts, o, h, l, c, v, oi = row
            time_str = ts[11:16]
            atm = oc.round_to_step(c, strike_step)
            _dh, _dm = divmod(int(time_str[:2]) * 60 + int(time_str[3:5]) + candle_minutes, 60)
            decision_time_str = f"{_dh:02d}:{_dm:02d}"

            def _close(reason: str) -> None:
                nonlocal position
                _, candles = _atm_option_candles(position["strike"], position["opt_type"], position["date"], position["expiry"])
                bar = _fill(candles, decision_time_str) if candles else None
                exit_price = bar[4] if bar else position["entry_price"]
                exit_time = bar[0] if bar else ts
                trades.append(OptionTrade(
                    date=position["date"], direction=position["direction"], expiry=position["expiry"],
                    strike=position["strike"], entry_time=position["entry_time"], entry_premium=position["entry_price"],
                    exit_time=exit_time, exit_premium=exit_price, lot_size=position["lot_size"], exit_reason=reason,
                ))
                position = None

            if time_str >= FORCE_FLAT_TIME:
                if position is not None:
                    _close("eod")
                continue

            if position is not None:
                is_long = position["direction"] == "LONG"
                hit_stop = (c <= position["stop_level"]) if is_long else (c >= position["stop_level"])
                hit_target = (c >= position["target_level"]) if is_long else (c <= position["target_level"])
                if hit_stop:
                    _close("stop_loss")
                elif hit_target:
                    _close("target")

            if position is None and not already_traded_today:
                direction_label = None
                stop_price = target_price = None

                if strategy_mode == "poc_magnet":
                    if c > poc + poc_distance_points:
                        direction_label, stop_price, target_price = "SHORT", c + sl_points, poc
                    elif c < poc - poc_distance_points:
                        direction_label, stop_price, target_price = "LONG", c - sl_points, poc

                elif strategy_mode == "vah_val_reversion":
                    if h > vah and c <= vah:
                        direction_label, stop_price, target_price = "SHORT", h, poc
                    elif l < val and c >= val:
                        direction_label, stop_price, target_price = "LONG", l, poc

                else:  # breakout
                    va_width = vah - val
                    if c > vah:
                        direction_label, stop_price, target_price = "LONG", vah, c + target_multiple * va_width
                    elif c < val:
                        direction_label, stop_price, target_price = "SHORT", val, c - target_multiple * va_width

                if direction_label is not None:
                    opt_type = "CE" if direction_label == "LONG" else "PE"
                    contract, candles = _atm_option_candles(atm, opt_type, d, expiry)
                    if contract is not None and candles:
                        bar = _fill(candles, decision_time_str)
                        if bar is not None:
                            position = {
                                "direction": direction_label, "entry_time": bar[0], "entry_price": bar[4],
                                "strike": contract["strike_price"], "expiry": expiry,
                                "lot_size": contract["lot_size"], "opt_type": opt_type, "date": d,
                                "stop_level": stop_price, "target_level": target_price,
                            }
                            already_traded_today = True

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
