"""Backtest: NIFTY 50 spot candles (5-minute by default), buy ATM CE when
price pulls back into the EMA(9)/EMA(20) cloud during an uptrend, then
breaks back out above the pullback's own high in the trend's
direction. Buy ATM PE on the bearish mirror (downtrend, pullback into
the cloud, breakout below) -- through real ATM NIFTY options, with a
fixed rupee stop-loss/target.

TREND: EMA(9) vs EMA(20) on the index's own closes. Uptrend when EMA9 >
EMA20, downtrend when EMA9 < EMA20 -- the "cloud" is the band between
the two EMA values at each bar (its color, in a chart, is which EMA is
on top).

PULLBACK + BREAKOUT, by each bar's CLOSE relative to the band (not its
high/low range): during an uptrend, a bar whose close is below
min(ema9,ema20) has genuinely pulled back below the cloud ("dipped").
Once dipped, the trade fires on the first later bar (still in the same
uptrend) whose close is back above max(ema9,ema20) -- "price came
below the bands and broke above the band," the user's own chart
reading, taken literally. The downtrend mirror is identical: a close
above the band sets "dipped" (price popped above the cloud), and the
first later close back below the band, still in the downtrend, fires
a PE entry. Bars whose close sits inside the band, or on the trend
side already (never dipped), don't affect the state.

(An earlier version instead defined "touching" by high/low *range*
overlap with the band and required several such bars within a tight
ATR window before scanning for a breakout. That broke in two ways: a
bar's low can graze the band on a wick even while the close -- and the
rest of the trend -- is pushing to new highs well above it, so the
"pullback" window silently extended across a genuine upswing and
accumulated a high no later close could ever clear; and the
ATR-tightness check only cleared the run state when it PASSED, so one
failing bar left the window open to keep growing instead of resetting.
Together these silently swallowed the 2026-10-06 12:15 breakout the
user flagged by hand, despite the raw EMA9/EMA20/close data matching
their description exactly. Keying off the close, with no range checks
and no minimum-bar/ATR filter, fixes both and matches the chart.)

candle_minutes (default 5, matching the screenshots this was modeled
on): resamples each day's 1-minute index candles first. Every option
fill and the FORCE_FLAT_TIME cutoff go through a _decision_time()
helper identical to macd_histogram_1min_dualstop_options.py's, since
_resample labels a multi-minute bar by its bucket start, not its
close -- the same look-ahead class already fixed there.

EXIT: three stages on the option's own premium P&L, so an early loser
still gets cut quickly but a winner isn't capped:

  1. INITIAL -- below BREAKEVEN_RS (default 300) profit: the only exits
     are the structural dip-stop (the index's own low at the dip that
     triggered the setup, e.g. the 12:10 bar's low for the 2026-10-06
     12:15 LONG) and a trend flip (EMA9/EMA20 cross against the
     position -- the cloud changing color).
  2. BREAKEVEN -- once premium profit reaches BREAKEVEN_RS: the stop
     also moves to the option's own entry price, so from here the
     worst case is flat, not a loss.
  3. TRAILING -- once premium profit reaches TARGET_RS (default 600):
     the premium floor is dropped and the exit switches fully to the
     structural index stop, which is now ratcheted every time a later
     bar repeats the SAME dip-and-reclaim pattern that triggered the
     entry (a fresh, higher pullback low for a LONG, ratcheted up,
     never loosened) -- riding the trend instead of capping the winner
     at a fixed rupee move.

Force-flat at FORCE_FLAT_TIME (15:25) regardless of stage. Only one
position open at a time.

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
BREAKEVEN_RS = 300.0  # premium profit (Rs/lot) at which the stop moves to breakeven
TARGET_RS = 600.0     # premium profit (Rs/lot) at which the stop switches to structural trailing


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


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    candle_minutes: int = 5,
    ema_fast: int = EMA_FAST,
    ema_slow: int = EMA_SLOW,
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
    min_bars = ATR_PERIOD + ema_slow + 3

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
        dipped_up = False    # uptrend: a prior bar closed below the band
        dipped_down = False  # downtrend: a prior bar closed above the band
        dip_low: float | None = None   # lowest low reached while dipped_up
        dip_high: float | None = None  # highest high reached while dipped_down
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
            close = bars[i].c

            if not trend_up:
                dipped_up = False
                dip_low = None
            if not trend_down:
                dipped_down = False
                dip_high = None

            direction_label = opt_type = stop_level = None
            if trend_up:
                if dipped_up and close > band_hi:
                    direction_label, opt_type, stop_level = "LONG", "CE", dip_low
                    dipped_up = False
                    dip_low = None
                elif close < band_lo:
                    dip_low = bars[i].l if dip_low is None else min(dip_low, bars[i].l)
                    dipped_up = True
            elif trend_down:
                if dipped_down and close < band_lo:
                    direction_label, opt_type, stop_level = "SHORT", "PE", dip_high
                    dipped_down = False
                    dip_high = None
                elif close > band_hi:
                    dip_high = bars[i].h if dip_high is None else max(dip_high, bars[i].h)
                    dipped_down = True

            if direction_label is None:
                i += 1
                continue

            entry_idx = i
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
            # three stages on the option's own premium P&L:
            #   "initial"   -- below BREAKEVEN_RS profit: only the structural
            #                  dip stop and a trend flip can exit.
            #   "breakeven" -- >= BREAKEVEN_RS: stop also moves to the
            #                  option's own entry price (give back to flat,
            #                  not a loss).
            #   "trailing"  -- >= TARGET_RS: stop lets go of the premium
            #                  floor and switches fully to the structural
            #                  index trail (ratcheted by each dip-and-reclaim
            #                  cycle), riding the trend instead of capping it.
            stage = "initial"
            trail_stop = stop_level
            trail_dipped = False   # a later bar, mid-trade, has re-dipped past the band
            trail_dip_extreme: float | None = None  # its own low (LONG) / high (SHORT)
            for j in range(entry_idx + 1, len(bars)):
                j_decision_time = _decision_time(bars[j].ts)
                if j_decision_time >= FORCE_FLAT_TIME:
                    exit_idx = j
                    exit_reason = "eod"
                    break

                jf, js = ema9[j], ema20[j]
                if jf is None or js is None:
                    continue
                j_band_hi, j_band_lo = max(jf, js), min(jf, js)
                j_close = bars[j].c

                if direction_label == "LONG" and not (jf > js):
                    exit_idx = j
                    exit_reason = "trend_flip"
                    break
                if direction_label == "SHORT" and not (jf < js):
                    exit_idx = j
                    exit_reason = "trend_flip"
                    break

                bar_j = _fill(opt_rows, j_decision_time)
                premium_j = bar_j[4] if bar_j else entry_price
                pnl_per_lot = (premium_j - entry_price) * lot_size

                if stage == "initial" and pnl_per_lot >= BREAKEVEN_RS:
                    stage = "breakeven"
                if stage == "breakeven" and pnl_per_lot >= TARGET_RS:
                    stage = "trailing"

                if direction_label == "LONG" and trail_stop is not None and bars[j].l <= trail_stop:
                    exit_idx = j
                    exit_reason = "stop_loss"
                    break
                if direction_label == "SHORT" and trail_stop is not None and bars[j].h >= trail_stop:
                    exit_idx = j
                    exit_reason = "stop_loss"
                    break
                if stage == "breakeven" and pnl_per_lot <= 0:
                    exit_idx = j
                    exit_reason = "breakeven"
                    break

                if stage == "trailing":
                    # same dip-and-reclaim pattern as the entry trigger,
                    # continued through the trade: each completed cycle
                    # ratchets the structural stop up (LONG) / down (SHORT)
                    # to its own extreme, never loosening it.
                    if direction_label == "LONG":
                        if trail_dipped and j_close > j_band_hi:
                            if trail_dip_extreme is not None:
                                trail_stop = max(trail_stop, trail_dip_extreme) if trail_stop is not None else trail_dip_extreme
                            trail_dipped = False
                            trail_dip_extreme = None
                        elif j_close < j_band_lo:
                            trail_dip_extreme = bars[j].l if trail_dip_extreme is None else min(trail_dip_extreme, bars[j].l)
                            trail_dipped = True
                    else:
                        if trail_dipped and j_close < j_band_lo:
                            if trail_dip_extreme is not None:
                                trail_stop = min(trail_stop, trail_dip_extreme) if trail_stop is not None else trail_dip_extreme
                            trail_dipped = False
                            trail_dip_extreme = None
                        elif j_close > j_band_hi:
                            trail_dip_extreme = bars[j].h if trail_dip_extreme is None else max(trail_dip_extreme, bars[j].h)
                            trail_dipped = True

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
