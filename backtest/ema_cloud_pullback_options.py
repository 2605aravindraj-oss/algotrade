"""Backtest: NIFTY 50 spot candles (5-minute by default), buy ATM CE when
price pulls back into the EMA(9)/EMA(20) cloud during an uptrend,
consolidates tightly there, then breaks back out above the consolidation
in the trend's direction. Buy ATM PE on the bearish mirror (downtrend,
pullback into the cloud, breakout below) -- through real ATM NIFTY
options, with a fixed rupee stop-loss/target.

TREND: EMA(9) vs EMA(20) on the index's own closes. Uptrend when EMA9 >
EMA20, downtrend when EMA9 < EMA20 -- the "cloud" is the band between
the two EMA values at each bar (its color, in a chart, is which EMA is
on top).

PULLBACK + CONSOLIDATION: a bar "touches" the cloud when its own
high/low range overlaps the EMA9-EMA20 band at that bar (low <=
max(ema9,ema20) and high >= min(ema9,ema20)) -- price has pulled back
into the band, not just approached it. CONSOLIDATION_MIN_BARS
consecutive touching bars, whose combined high-low range is within
CONSOLIDATION_ATR_MULT x ATR(14), count as a genuine basing zone (the
circled area on a chart) rather than a single noisy wick through the
band.

BREAKOUT CONFIRMATION: once a valid consolidation window exists, the
trade fires the first later bar whose own close breaks back out of
that window's own high (uptrend -> buy CE) or low (downtrend -> buy
PE) AND is still on the same side of the trend as when the window
formed (EMA9/EMA20 ordering unchanged) -- a literal trend-continuation
entry, not a reversal. Entry fills at that breakout bar's own close.

candle_minutes (default 5, matching the screenshots this was modeled
on): resamples each day's 1-minute index candles first. Every option
fill and the FORCE_FLAT_TIME cutoff go through a _decision_time()
helper identical to macd_histogram_1min_dualstop_options.py's, since
_resample labels a multi-minute bar by its bucket start, not its
close -- the same look-ahead class already fixed there.

EXIT: fixed rupee P&L stop-loss/target on the whole position (premium
move x lot_size), same convention as every sl_rs/target_rs module in
this codebase -- sl_rs (default 300) or target_rs (default 600),
whichever hits first, else force-flat at FORCE_FLAT_TIME (15:25).
Only one position open at a time.

STRIKE/EXPIRY: ATM = round_to_step(index close, strike_step), nearest
expiry on/after the entry day. Uses the EXPIRED-instruments API (needs
an Upstox access token) for any day whose weekly contract has already
rolled over.
"""
from __future__ import annotations

from dataclasses import dataclass

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, _bar_at_or_after, _bar_at_or_before
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.sweep_reclaim_breakout import _resample
from backtest.technical_rating import _ema

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"
EMA_FAST = 9
EMA_SLOW = 20
ATR_PERIOD = 14
CONSOLIDATION_MIN_BARS = 3
CONSOLIDATION_ATR_MULT = 2.0


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


def _atr_at(bars: list[Bar], i: int, period: int = ATR_PERIOD) -> float | None:
    if i < period:
        return None
    trs = []
    for k in range(i - period + 1, i + 1):
        prev_c = bars[k - 1].c
        trs.append(max(bars[k].h - bars[k].l, abs(bars[k].h - prev_c), abs(bars[k].l - prev_c)))
    return sum(trs) / period


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    candle_minutes: int = 5,
    ema_fast: int = EMA_FAST,
    ema_slow: int = EMA_SLOW,
    consolidation_min_bars: int = CONSOLIDATION_MIN_BARS,
    consolidation_atr_mult: float = CONSOLIDATION_ATR_MULT,
    sl_rs: float = 300.0,
    target_rs: float = 600.0,
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

    def _decision_time(ts: str) -> str:
        """_resample labels a multi-minute bar by its bucket START, not
        its close -- see the identical helper (and its full rationale)
        in macd_histogram_1min_dualstop_options.py. At candle_minutes<=1
        no resampling happens, so this reduces to the bar's own
        timestamp."""
        if candle_minutes <= 1:
            return ts[11:16]
        hh, mm = int(ts[11:13]), int(ts[14:16])
        dh, dm = divmod(hh * 60 + mm + candle_minutes, 60)
        return f"{dh:02d}:{dm:02d}"

    trades: list[OptionTrade] = []
    min_bars = ATR_PERIOD + ema_slow + consolidation_min_bars + 3

    for day in trading_days:
        d = day["date"]
        rows_1min = sorted(cache.get_day_candles_cached(underlying_key, "1minute", d, expired=False), key=lambda r: r[0])
        if len(rows_1min) < min_bars:
            continue
        rows = _resample(rows_1min, candle_minutes) if candle_minutes > 1 else rows_1min
        if len(rows) < min_bars:
            continue
        bars = [Bar(ts=r[0], o=r[1], h=r[2], l=r[3], c=r[4]) for r in rows]
        closes = [b.c for b in bars]
        ema9 = _ema(closes, ema_fast)
        ema20 = _ema(closes, ema_slow)

        expiry = next((e for e in expiries if e >= d), None)
        if expiry is None:
            continue

        i = ATR_PERIOD + ema_slow
        touch_run_start: int | None = None  # start index of the current run of touching bars
        while i < len(bars):
            decision_time_str = _decision_time(bars[i].ts)
            if decision_time_str >= FORCE_FLAT_TIME:
                break

            f, s = ema9[i], ema20[i]
            if f is None or s is None:
                i += 1
                continue
            trend_up = f > s
            trend_down = f < s
            band_hi, band_lo = max(f, s), min(f, s)
            touching = bars[i].l <= band_hi and bars[i].h >= band_lo

            if touching and (trend_up or trend_down):
                if touch_run_start is None:
                    touch_run_start = i
            else:
                touch_run_start = None

            direction_label = opt_type = None
            if touch_run_start is not None and i - touch_run_start + 1 >= consolidation_min_bars:
                window = bars[touch_run_start:i + 1]
                win_hi = max(b.h for b in window)
                win_lo = min(b.l for b in window)
                atr = _atr_at(bars, i)
                if atr is not None and atr > 0 and (win_hi - win_lo) <= atr * consolidation_atr_mult:
                    # scan forward bars (not yet consumed) for the breakout
                    for j in range(i + 1, len(bars)):
                        j_decision_time = _decision_time(bars[j].ts)
                        if j_decision_time >= FORCE_FLAT_TIME:
                            break
                        jf, js = ema9[j], ema20[j]
                        if jf is None or js is None:
                            continue
                        still_up = jf > js
                        still_down = jf < js
                        if trend_up and still_up and bars[j].c > win_hi:
                            direction_label, opt_type, breakout_idx = "LONG", "CE", j
                            break
                        if trend_down and still_down and bars[j].c < win_lo:
                            direction_label, opt_type, breakout_idx = "SHORT", "PE", j
                            break
                        if (trend_up and not still_up) or (trend_down and not still_down):
                            break  # trend itself flipped before any breakout -- window invalidated
                    touch_run_start = None  # this window has been resolved (fired or invalidated) either way

            if direction_label is None:
                i += 1
                continue

            entry_idx = breakout_idx
            entry_close = bars[entry_idx].c
            entry_ts = bars[entry_idx].ts
            entry_decision_time = _decision_time(entry_ts)
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

            exit_idx = None
            exit_reason = None
            for j in range(entry_idx + 1, len(bars)):
                j_decision_time = _decision_time(bars[j].ts)
                if j_decision_time >= FORCE_FLAT_TIME:
                    exit_idx = j
                    exit_reason = "eod"
                    break
                bar_j = _fill(opt_rows, j_decision_time)
                premium_j = bar_j[4] if bar_j else entry_price
                pnl_per_lot = (premium_j - entry_price) * lot_size
                if pnl_per_lot <= -sl_rs:
                    exit_idx = j
                    exit_reason = "stop_loss"
                    break
                if pnl_per_lot >= target_rs:
                    exit_idx = j
                    exit_reason = "target"
                    break

            if exit_idx is None:
                exit_idx = len(bars) - 1
                exit_reason = "eod"

            exit_decision_time = _decision_time(bars[exit_idx].ts)
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
