"""Backtest of live/macd_histogram_1min_live.py's EXACT entry/exit logic
(histogram zero-cross on NIFTY 50 spot candles, dual-condition
stop-loss, target_rs + trailing) over a date RANGE, through real ATM
NIFTY options -- so the live script's parameters (chiefly target_rs) can
be tuned against more than one day before being adopted as the live
default. candle_minutes (default 1) resamples each day's 1-minute index
candles to that bar size first -- e.g. candle_minutes=5 replays the
exact same signal/stop/target logic a 5-minute version of the live
script would use, for comparison against the 1-minute default.

SIGNAL/STOP/TARGET: identical to live/macd_histogram_1min_live.py --
    - MACD(12,26,9) reseeded FRESH every trading day by default (an
      SMA seed on that day's own candles, matching the live script
      after its "revert back to daily calculation" change). Pass
      continuous=True to instead run ONE EMA across the whole
      from_date/to_date range -- matching
      macd_histogram_zero_cross_options.py's 5-minute module -- so the
      daily SMA-reseed warm-up (~35 bars every morning, a much bigger
      bite out of a 5-minute day than a 1-minute one) doesn't starve
      the signal of any real trading time before continuing it across
      days. Either way, POSITION state (and the entry scan) still
      resets at each day boundary and force-flattens at
      FORCE_FLAT_TIME -- continuous only affects the indicator's own
      warm-up, never carries a position overnight.
    - Fresh cross <=0->>0 buys ATM CE; >=0-><0 buys ATM PE.
    - Dual-condition stop-loss, mirrored by direction (see the live
      script's own docstring for the exact rule).
    - target_rs (default 150.0): once peak P&L/lot first reaches it,
      the stop reference starts trailing to each new peak.
Each bar's own CLOSE time doubles as both the signal decision point and
the fill reference, exactly as the live script treats "filled at the
crossing candle's own close". At candle_minutes=1 that's just the bar's
own timestamp (no resampling happens). At candle_minutes>1, _resample
labels each bar by its bucket START, not its close -- so every fill
(entry, exit, and each bar's P&L/stop check while a position is open)
and the FORCE_FLAT_TIME cutoff go through _decision_time(), which adds
candle_minutes to the bar's label first. Skipping this would be a
genuine look-ahead (pricing a trade as if at a moment before the very
bar that triggered it had finished forming) -- the same bug
macd_histogram_zero_cross_options.py's decision_time_str already
guards against, which an earlier version of this module's
candle_minutes generalization failed to carry over.

Other differences from the live script: this also force-flattens any
still-open position at FORCE_FLAT_TIME (15:25) at day end (the live
script just reports "still open" since it only ever looks at today),
and realizes P&L through OptionTrade's real cost model (brokerage/
STT/slippage), not raw premium difference -- so numbers here are not
directly comparable to the live script's own gross P&L printout.

STRIKE/EXPIRY: ATM = round_to_step(index close, strike_step), nearest
expiry on/after the entry day -- same convention as every other NIFTY
options backtest module here. Uses the EXPIRED-instruments API (needs
an Upstox access token) since most of a past week's weekly contracts
have already rolled over/expired by the time this runs.
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, _bar_at_or_after, _bar_at_or_before
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.sweep_reclaim_breakout import _resample
from backtest.technical_rating import _macd

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"


def _apply_slippage(price: float, side: str, slippage_pct: float) -> float:
    if slippage_pct <= 0:
        return price
    return price * (1 + slippage_pct) if side == "BUY" else price * (1 - slippage_pct)


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    candle_minutes: int = 1,
    continuous: bool = False,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    min_histogram_buffer: float = 0.0,
    target_rs: float = 150.0,
    slippage_pct: float = 0.0,
    access_token: str | None = None,
) -> list[OptionTrade]:
    """min_histogram_buffer: same entry filter as
    macd_histogram_zero_cross_options.py -- requires the histogram to
    clear this many index points PAST zero (not merely cross it) before
    counting as a real signal (prev_hist <= -buffer and hist > buffer for
    a LONG, mirrored for SHORT). 0.0 (default) is the literal zero-cross."""
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
        """_resample labels a multi-minute bar by its bucket START, but
        that bar's own close (and the histogram built from it) isn't
        actually known until candle_minutes later -- any option fill or
        FORCE_FLAT_TIME check must use THIS time, or it's a look-ahead:
        pricing a trade as if at a moment before the very bar that
        triggered it has finished forming. At candle_minutes<=1 bars are
        the raw, already-closed 1-minute candles (no resampling happens),
        so this correctly reduces to the bar's own timestamp, matching
        live/macd_histogram_1min_live.py's "filled at the crossing
        candle's own close" convention exactly -- same pattern
        macd_histogram_zero_cross_options.py already uses."""
        if candle_minutes <= 1:
            return ts[11:16]
        hh, mm = int(ts[11:13]), int(ts[14:16])
        dh, dm = divmod(hh * 60 + mm + candle_minutes, 60)
        return f"{dh:02d}:{dm:02d}"

    def _hist(bar_closes: list[float]) -> list[float | None]:
        macd_line, signal_line = _macd(bar_closes, macd_fast, macd_slow, macd_signal)
        return [(m - s) if (m is not None and s is not None) else None for m, s in zip(macd_line, signal_line)]

    # Build the combined bars/histogram arrays. continuous=True computes ONE
    # EMA warm-up across the whole range (position state still resets daily
    # in the main loop below); continuous=False (default, matches the live
    # script) recomputes a fresh SMA-seeded histogram for each day and
    # concatenates the per-day results.
    bars: list[list] = []
    histogram: list[float | None] = []
    if continuous:
        all_1min: list[list] = []
        for day in trading_days:
            d = day["date"]
            all_1min.extend(cache.get_day_candles_cached(underlying_key, "1minute", d, expired=False))
        all_1min.sort(key=lambda r: r[0])
        if len(all_1min) < 2:
            return []
        bars = _resample(all_1min, candle_minutes) if candle_minutes > 1 else all_1min
        histogram = _hist([b[4] for b in bars])
    else:
        for day in trading_days:
            d = day["date"]
            rows_1min = sorted(cache.get_day_candles_cached(underlying_key, "1minute", d, expired=False), key=lambda r: r[0])
            if len(rows_1min) < 2:
                continue
            day_bars = _resample(rows_1min, candle_minutes) if candle_minutes > 1 else rows_1min
            if len(day_bars) < 2:
                continue
            bars.extend(day_bars)
            histogram.extend(_hist([b[4] for b in day_bars]))

    if len(bars) < 2:
        return []

    trades: list[OptionTrade] = []
    i = 0
    prev_hist = None
    current_day = None
    while i < len(bars):
        ts = bars[i][0]
        d = ts[:10]
        decision_time_str = _decision_time(ts)
        hist = histogram[i]

        if d != current_day:
            current_day = d
            prev_hist = None  # never carry a cross-day artifact into a fresh trading day

        if decision_time_str >= FORCE_FLAT_TIME:
            prev_hist = hist
            i += 1
            continue

        if hist is None:
            prev_hist = hist
            i += 1
            continue
        if prev_hist is None:
            prev_hist = hist
            i += 1
            continue
        if prev_hist <= -min_histogram_buffer and hist > min_histogram_buffer:
            direction_label, opt_type = "LONG", "CE"
        elif prev_hist >= min_histogram_buffer and hist < -min_histogram_buffer:
            direction_label, opt_type = "SHORT", "PE"
        else:
            prev_hist = hist
            i += 1
            continue

        expiry = next((e for e in expiries if e >= d), None)
        if expiry is None:
            prev_hist = hist
            i += 1
            continue

        entry_idx = i
        entry_close = bars[entry_idx][4]
        entry_hist = histogram[entry_idx]
        entry_ts = bars[entry_idx][0]
        atm = oc.round_to_step(entry_close, strike_step)

        contract, opt_rows = _atm_option_candles(atm, opt_type, d, expiry)
        if contract is None or not opt_rows:
            i = entry_idx + 1
            prev_hist = hist
            continue

        entry_bar = _fill(opt_rows, decision_time_str)
        if entry_bar is None:
            i = entry_idx + 1
            prev_hist = hist
            continue
        entry_price = _apply_slippage(entry_bar[4], "BUY", slippage_pct)
        lot_size = contract["lot_size"]

        ref_close, ref_hist = entry_close, entry_hist
        peak_pnl_per_lot = 0.0
        trailing_active = False
        exit_idx = None
        exit_reason = None
        for j in range(entry_idx + 1, len(bars)):
            if bars[j][0][:10] != d:
                exit_idx = j - 1
                exit_reason = "eod"
                break
            j_decision_time_str = _decision_time(bars[j][0])
            if j_decision_time_str >= FORCE_FLAT_TIME:
                exit_idx = j
                exit_reason = "eod"
                break
            c = bars[j][4]
            h = histogram[j]
            if h is None:
                continue
            bar_j = _fill(opt_rows, j_decision_time_str)
            premium_j = bar_j[4] if bar_j else entry_price
            pnl_per_lot = (premium_j - entry_price) * lot_size
            if pnl_per_lot > peak_pnl_per_lot:
                peak_pnl_per_lot = pnl_per_lot
                if peak_pnl_per_lot >= target_rs:
                    trailing_active = True
                    ref_close, ref_hist = c, h
            is_long = direction_label == "LONG"
            stop_hit = (c < ref_close and h < ref_hist) if is_long else (c > ref_close and h > ref_hist)
            if stop_hit:
                exit_idx = j
                exit_reason = "trailing_stop" if trailing_active else "stop_loss"
                break

        if exit_idx is None:
            exit_idx = len(bars) - 1
            exit_reason = "eod"

        exit_decision_time_str = _decision_time(bars[exit_idx][0])
        exit_bar = _fill(opt_rows, exit_decision_time_str)
        exit_price = _apply_slippage(exit_bar[4], "SELL", slippage_pct) if exit_bar else entry_price
        exit_time = exit_bar[0] if exit_bar else bars[exit_idx][0]

        trades.append(OptionTrade(
            date=d, direction=direction_label, expiry=expiry, strike=contract["strike_price"],
            entry_time=entry_ts, entry_premium=entry_price, exit_time=exit_time, exit_premium=exit_price,
            lot_size=lot_size, exit_reason=exit_reason,
        ))

        i = exit_idx + 1
        prev_hist = histogram[exit_idx] if exit_idx < len(histogram) else hist

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
