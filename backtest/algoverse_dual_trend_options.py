"""Multi-indicator dual-trend breakout/breakdown strategy, sourced from
an external "Algoverse" strategy card (screenshotted, not written as a
spec), realized through real NIFTY options targeted by DELTA (~0.66)
rather than a fixed strike distance -- since Upstox has no live Greeks
feed, delta is derived per backtest/options_greeks.py (IV backed out
from each candidate strike's own observed premium, then the standard
Black-Scholes delta formula). 5-minute NIFTY 50 INDEX candles
(timeframe not stated on the card; this codebase's usual intraday
default -- an assumption).

SIGNAL -- the card names 4 indicators ("Super Trend, Bollinger Bands,
RSI, and CCI") to "watch both directions" for "either a bullish
breakout or a bearish breakdown", with no periods, thresholds, or
combination rule stated -- pinned down here, each a flagged
assumption:
    SuperTrend(10, 3) -- the common default period/multiplier (not
        this codebase's other SuperTrend card's period=7, since this
        card states no period of its own).
    Bollinger Bands(20, 2 stddev) -- standard default. "Breakout" read
        as a CLOSE beyond the band (not a wick).
    RSI(14) -- standard period. Threshold >60 bullish / <40 bearish --
        a moderate momentum band, not plain >50/<50, since the card
        frames this as a breakout/breakdown FILTER, not a directional
        bias.
    CCI(20) -- standard period. Threshold >+100 bullish / <-100
        bearish (textbook CCI breakout levels).
    ALL FOUR must agree on the SAME candle (a strict conjunction, not
        "most agree" or a scoring system) -- the first such candle
        each day is the "breakout" trigger from the card's step 1.

VALIDATION (card's step 2, "Confirms the breakout past key Pivot Point
levels and recent highs or lows"): checked on that SAME trigger candle
(read as simultaneous, not a separate later bar -- a defensible but
not the only possible reading of "validates the move"):
    Standard daily Pivot Point P=(H+L+C)/3 from the PREVIOUS day's
    daily H/L/C (same convention as supertrend_pivot_options.py) --
    close > P for bullish, close < P for bearish.
    "Recent highs or lows" -- close > the highest High of the prior
    recent_bars (default 20, an assumption -> 100 minutes on 5-min
    candles) candles THIS SAME DAY for bullish; close < the lowest Low
    of the prior recent_bars candles for bearish (a Donchian-channel-
    style check; resets daily, not a multi-day channel).

ENTRY (card's step 3): buy the option, among strikes within
strike_search_range (default 6, i.e. 300 points at NIFTY's 50-point
step) of the index's own ATM, whose own Black-Scholes delta is closest
in magnitude to target_delta (default 0.66, the card's own figure) --
a call for a bullish breakout, a put for a bearish breakdown. Entry
fills at the trigger candle's own close.

EXIT (card's step 4, "Manages the risk"): stop-loss is whichever comes
FIRST (premium check made first if both would trigger on one bar,
conservative) of (a) SuperTrend reversing against the position
(direction flips) or (b) the option's own premium falling
stop_pct (default 0.10, the card's "10% of its value") from entry.
Target is a FIXED RUPEE amount (target_rupees, default 1500, the
card's own stated figure) converted to premium points as
target_rupees / (lot_size * quantity) -- so the points needed HALVES
when quantity doubles on the reversal trade below. Forced flat at
force_flat_time (default "15:14", the card's own stated end time, not
this codebase's usual 15:25/15:30).

REVERSAL TRADE ("If the trade exits in a loss, a defined opposite-side
trade may follow with double the initial quantity if the reversal
trade is confirmed"): if the FIRST trade of the day exits at a net
loss, the OPPOSITE direction's full entry condition (signal +
validation, mirrored) is re-checked starting from that exit candle;
the first later candle (same day) that confirms it triggers the
reversal trade at reversal_quantity_multiple (default 2) times the
base quantity, same stop/target mechanics (target points recomputed
for the larger quantity). max_trades_per_day (default 2, the card's
own stated cap) means at most the first trade plus one reversal -- no
further re-entry regardless of how the reversal trade itself exits.

Entries/exits fill at the deciding candle's own close -- decision-
time-correct (bucket start + candle_minutes), same convention as
every other intraday module here. Requires an Upstox access token
(expired-instruments API).
"""
from __future__ import annotations

import datetime as _dt

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest import options_greeks as greeks
from backtest.futures_oi_buildup import _bar_at_or_after, _bar_at_or_before
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.sweep_reclaim_breakout import _resample
from backtest.supertrend_pivot_options import _compute_supertrend_line, _daily_pivots
from backtest.bollinger_breakout_trail import _bollinger_bands
from backtest.technical_rating import _rsi, _cci

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    candle_minutes: int = 5,
    st_period: int = 10,
    st_multiplier: float = 3.0,
    bb_period: int = 20,
    bb_std: float = 2.0,
    rsi_period: int = 14,
    rsi_bull_threshold: float = 60.0,
    rsi_bear_threshold: float = 40.0,
    cci_period: int = 20,
    cci_threshold: float = 100.0,
    recent_bars: int = 20,
    strike_search_range: int = 6,
    target_delta: float = 0.66,
    stop_pct: float = 0.10,
    target_rupees: float = 1500.0,
    base_quantity: int = 1,
    reversal_quantity_multiple: int = 2,
    max_trades_per_day: int = 2,
    force_flat_time: str = "15:14",
    access_token: str | None = None,
) -> list[OptionTrade]:
    trading_days = upstox_client.get_daily_history(underlying_key, from_date, to_date)
    all_1min: list[list] = []
    for day in trading_days:
        rows = sorted(
            cache.get_day_candles_cached(underlying_key, "1minute", day["date"], expired=False),
            key=lambda c: c[0],
        )
        all_1min.extend(rows)
    all_1min.sort(key=lambda c: c[0])
    if len(all_1min) < 2:
        return []

    bars = _resample(all_1min, candle_minutes)
    if len(bars) < max(bb_period, rsi_period, cci_period, st_period) + 2:
        return []

    highs = [b[2] for b in bars]
    lows = [b[3] for b in bars]
    closes = [b[4] for b in bars]

    st_dir, _st_line = _compute_supertrend_line(bars, st_period, st_multiplier)
    bb_upper, bb_lower = _bollinger_bands(closes, bb_period, bb_std)
    rsi = _rsi(closes, rsi_period)
    cci = _cci(highs, lows, closes, cci_period)
    pivots = _daily_pivots(underlying_key, from_date, to_date, access_token)

    expiries = sorted(cache.get_expired_expiries_cached(underlying_key, "options", access_token))
    chain_cache: dict[str, dict] = {}
    day_candles_cache: dict[tuple[str, str], list[list]] = {}

    def _lookup(expiry):
        if expiry not in chain_cache:
            chain_cache[expiry] = oc.build_chain_lookup(
                cache.get_expired_option_chain_cached(underlying_key, expiry, access_token)
            )
        return chain_cache[expiry]

    def _day_candles(instrument_key, date):
        key = (instrument_key, date)
        if key not in day_candles_cache:
            day_candles_cache[key] = sorted(
                cache.get_day_candles_cached(instrument_key, "1minute", date, expired=True, access_token=access_token),
                key=lambda c: c[0],
            )
        return day_candles_cache[key]

    def _fill(candles, at_time):
        return _bar_at_or_after(candles, at_time) or _bar_at_or_before(candles, at_time)

    def _years_to_expiry(expiry: str, d: str) -> float:
        days = (_dt.date.fromisoformat(expiry) - _dt.date.fromisoformat(d)).days
        return max(days, 0.3) / 365.0

    def _pick_contract(atm: float, opt_type: str, d: str, expiry: str, spot: float, decision_time_str: str):
        """Returns (contract, candles, bar) for the strike whose own
        Black-Scholes delta is closest to target_delta, where `bar` is
        the full [ts,o,h,l,c,v,oi] row already fetched at decision_time
        -- reused directly for entry fill/time, no re-fetch needed."""
        lookup = _lookup(expiry)
        T = _years_to_expiry(expiry, d)
        premiums: dict[float, float] = {}
        contracts: dict[float, dict] = {}
        bars_by_strike: dict[float, list] = {}
        for k in range(-strike_search_range, strike_search_range + 1):
            strike = atm + k * strike_step
            contract = lookup.get((strike, opt_type))
            if contract is None:
                continue
            candles = _day_candles(contract["instrument_key"], d)
            bar = _fill(candles, decision_time_str)
            if bar is None:
                continue
            premiums[strike] = bar[4]
            contracts[strike] = contract
            bars_by_strike[strike] = bar
        best_strike = greeks.strike_by_delta(target_delta, opt_type, spot, T, premiums)
        if best_strike is None:
            return None, None, None
        contract = contracts[best_strike]
        return contract, _day_candles(contract["instrument_key"], d), bars_by_strike[best_strike]

    trades: list[OptionTrade] = []
    position = None  # dict: direction, entry_time, entry_price, strike, expiry, lot_size, opt_type, date, stop_level, target_level, st_entry_dir
    current_day: str | None = None
    day_start_idx = 0
    trades_today = 0
    pending_reversal: str | None = None  # "LONG" / "SHORT" if waiting to confirm a reversal entry, else None

    for i, row in enumerate(bars):
        ts, o, h, l, c, v, oi = row
        d = ts[:10]
        time_str = ts[11:16]

        if d != current_day:
            current_day = d
            day_start_idx = i
            position = None
            trades_today = 0
            pending_reversal = None

        atm = oc.round_to_step(c, strike_step)
        expiry = next((e for e in expiries if e >= d), None)
        piv = pivots.get(d)

        _dh, _dm = divmod(int(time_str[:2]) * 60 + int(time_str[3:5]) + candle_minutes, 60)
        decision_time_str = f"{_dh:02d}:{_dm:02d}"

        def _close(reason: str) -> None:
            nonlocal position, trades_today
            candles = _day_candles(position["contract_key"], position["date"])
            bar = _fill(candles, decision_time_str) if candles else None
            exit_price = bar[4] if bar else position["entry_price"]
            exit_time = bar[0] if bar else ts
            trades.append(OptionTrade(
                date=position["date"], direction=position["direction"], expiry=position["expiry"],
                strike=position["strike"], entry_time=position["entry_time"], entry_premium=position["entry_price"],
                exit_time=exit_time, exit_premium=exit_price, lot_size=position["lot_size"], exit_reason=reason,
            ))
            position = None

        if time_str >= force_flat_time:
            if position is not None:
                _close("eod")
            continue

        # -- exit management --
        if position is not None:
            is_long = position["direction"] == "LONG"
            candles = _day_candles(position["contract_key"], position["date"])
            bar = _fill(candles, decision_time_str)
            cur_premium = bar[4] if bar else position["entry_price"]
            hit_stop_pct = cur_premium <= position["entry_price"] * (1 - stop_pct)
            cur_st_dir = st_dir[i]
            hit_st_reversal = cur_st_dir is not None and (
                (is_long and cur_st_dir == -1) or (not is_long and cur_st_dir == 1)
            )
            hit_target = cur_premium >= position["target_level"] if is_long else cur_premium <= position["target_level"]
            was_loss_candidate = position["direction"]
            if hit_stop_pct:
                _close("stop_loss")
            elif hit_st_reversal:
                _close("trend_reverse")
            elif hit_target:
                _close("target")

            if position is None and trades and trades[-1].date == d:
                last_trade = trades[-1]
                if last_trade.pnl_points < 0 and trades_today < max_trades_per_day:
                    pending_reversal = "SHORT" if was_loss_candidate == "LONG" else "LONG"

        # -- indicator signal (used for both fresh entries and reversal confirmation) --
        bull_signal = bear_signal = False
        if (
            st_dir[i] is not None and bb_upper[i] is not None and bb_lower[i] is not None
            and rsi[i] is not None and cci[i] is not None and piv is not None
            and i - day_start_idx >= recent_bars
        ):
            recent_high = max(highs[i - recent_bars:i])
            recent_low = min(lows[i - recent_bars:i])
            bull_signal = (
                st_dir[i] == 1 and c > bb_upper[i] and rsi[i] > rsi_bull_threshold and cci[i] > cci_threshold
                and c > piv["P"] and c > recent_high
            )
            bear_signal = (
                st_dir[i] == -1 and c < bb_lower[i] and rsi[i] < rsi_bear_threshold and cci[i] < -cci_threshold
                and c < piv["P"] and c < recent_low
            )

        # -- entries --
        if position is None and trades_today < max_trades_per_day and expiry is not None:
            direction_label = None
            quantity = base_quantity
            if pending_reversal is not None:
                if pending_reversal == "LONG" and bull_signal:
                    direction_label, quantity = "LONG", base_quantity * reversal_quantity_multiple
                elif pending_reversal == "SHORT" and bear_signal:
                    direction_label, quantity = "SHORT", base_quantity * reversal_quantity_multiple
            elif trades_today == 0:
                if bull_signal:
                    direction_label = "LONG"
                elif bear_signal:
                    direction_label = "SHORT"

            if direction_label is not None:
                opt_type = "CE" if direction_label == "LONG" else "PE"
                contract, candles, bar = _pick_contract(atm, opt_type, d, expiry, c, decision_time_str)
                if contract is not None and candles and bar is not None:
                    lot_size = contract["lot_size"] * quantity
                    target_pts = target_rupees / lot_size
                    target_level = bar[4] + target_pts if direction_label == "LONG" else bar[4] - target_pts
                    position = {
                        "direction": direction_label, "entry_time": bar[0], "entry_price": bar[4],
                        "strike": contract["strike_price"], "expiry": expiry,
                        "lot_size": lot_size, "opt_type": opt_type, "date": d,
                        "contract_key": contract["instrument_key"],
                        "target_level": target_level,
                    }
                    trades_today += 1
                    pending_reversal = None

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
