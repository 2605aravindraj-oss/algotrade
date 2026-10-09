"""Backtest: NIFTY 50 spot 1-minute candles, buy ATM PE on a BEARISH MACD
histogram divergence, buy ATM CE on a BULLISH one -- through real ATM
NIFTY options, with a fixed rupee stop-loss/target. This is the classic
"two peaks/troughs of the histogram, same side of zero, no crossover in
between" divergence read directly off a MACD panel, NOT a price-pivot
double bottom/top (see macd_divergence_double_bottom_options.py for
that, different and separate approach).

HUMPS: the histogram is segmented into maximal runs of bars that stay
on one side of zero ("humps") -- a positive hump ends the instant the
histogram goes <=0, a negative hump ends the instant it goes >=0. Any
comparison below only ever happens between two points INSIDE THE SAME
HUMP -- a zero-line crossover immediately resets the hump's own
tracking state, so a peak from one hump is never compared against a
peak from the next. This is the "no crossover for finding divergence"
requirement.

LOCAL PEAK/TROUGH: bar k (inside a positive hump) is a local peak if
its histogram value is the STRICT max over the peak_window bars on
each side of it (2*peak_window+1 bars total); the negative-hump
mirror (strict min) defines a local trough. peak_window=1 (default)
is the literal single-bar/immediate-neighbor definition -- confirmed
one bar after k. A larger peak_window filters out single-bar noise
(at the cost of confirming peak_window bars later), closer to how a
trendline drawn on a chart connects the visually obvious bar tops/
bottoms rather than every tiny wiggle.

BEARISH DIVERGENCE (buy ATM PE): within the same still-open positive
hump, a newly confirmed local peak is LOWER than the previous local
peak in that same hump (histogram momentum weakening) WHILE the
index's own high at the new peak's bar is >= the index's high at the
previous peak's bar (price making an equal/higher high) -- the
textbook disagreement between price and momentum. Mirrored for:

BULLISH DIVERGENCE (buy ATM CE): within the same negative hump, a
newly confirmed local trough is HIGHER (less negative) than the
previous trough WHILE the index's own low at the new trough's bar is
<= the index's low at the previous trough's bar.

ENTRY: fires immediately once a divergence is confirmed -- one bar
after the weaker peak/trough itself (the earliest bar this can be
known), filled at that bar's own close. Only one position open at a
time; the hump-tracking state keeps updating every bar regardless of
whether a position is open, so a divergence that forms WHILE a trade
is running is still recorded (just not tradable until that position
closes).

MACD(12,26,9) is reseeded fresh every trading day (an SMA seed on
that day's own candles), matching this codebase's established "revert
back to daily calculation" convention.

EXIT: fixed rupee P&L stop-loss/target on the whole position (premium
move x lot_size) -- sl_rs (default 300) or target_rs (default 600),
whichever hits first, else force-flat at FORCE_FLAT_TIME (15:25).

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
from backtest.technical_rating import _macd

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"


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
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    peak_window: int = 1,
    sl_rs: float = 300.0,
    target_rs: float = 600.0,
    slippage_pct: float = 0.0,
    access_token: str | None = None,
) -> list[OptionTrade]:
    """peak_window (default 1): a local peak/trough must be the STRICT
    max/min over `peak_window` bars on each side (2*peak_window+1 bars
    total), not just its two immediate neighbors -- filters out
    single-bar noise at the cost of confirming peak_window bars later.
    peak_window=1 is the literal single-bar definition (the original
    behaviour)."""
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

    trades: list[OptionTrade] = []

    for day in trading_days:
        d = day["date"]
        rows = sorted(cache.get_day_candles_cached(underlying_key, "1minute", d, expired=False), key=lambda r: r[0])
        if len(rows) < macd_slow + macd_signal + 5:
            continue
        bars = [Bar(ts=r[0], o=r[1], h=r[2], l=r[3], c=r[4]) for r in rows]
        closes = [b.c for b in bars]
        macd_line, signal_line = _macd(closes, macd_fast, macd_slow, macd_signal)
        histogram = [(m - s) if (m is not None and s is not None) else None for m, s in zip(macd_line, signal_line)]

        expiry = next((e for e in expiries if e >= d), None)
        if expiry is None:
            continue

        position: dict | None = None
        hump_sign = 0
        hump_bars: list[tuple[int, float]] = []
        last_peak: tuple[int, float, float] | None = None   # (idx, hist_value, price_high)
        last_trough: tuple[int, float, float] | None = None  # (idx, hist_value, price_low)

        for i in range(len(bars)):
            time_str = bars[i].ts[11:16]
            if time_str >= FORCE_FLAT_TIME:
                if position is not None:
                    exit_bar = _fill(position["opt_rows"], time_str)
                    exit_price = _apply_slippage(exit_bar[4], "SELL", slippage_pct) if exit_bar else position["entry_price"]
                    exit_time = exit_bar[0] if exit_bar else bars[i].ts
                    trades.append(OptionTrade(
                        date=d, direction=position["direction"], expiry=expiry, strike=position["strike"],
                        entry_time=position["entry_time"], entry_premium=position["entry_price"],
                        exit_time=exit_time, exit_premium=exit_price, lot_size=position["lot_size"],
                        exit_reason="eod",
                    ))
                    position = None
                break

            # -- exit check for an already-open position, using this bar's option price --
            if position is not None:
                bar_j = _fill(position["opt_rows"], time_str)
                premium_j = bar_j[4] if bar_j else position["entry_price"]
                pnl_per_lot = (premium_j - position["entry_price"]) * position["lot_size"]
                exit_reason = None
                if pnl_per_lot <= -sl_rs:
                    exit_reason = "stop_loss"
                elif pnl_per_lot >= target_rs:
                    exit_reason = "target"
                if exit_reason is not None:
                    exit_price = _apply_slippage(premium_j, "SELL", slippage_pct)
                    trades.append(OptionTrade(
                        date=d, direction=position["direction"], expiry=expiry, strike=position["strike"],
                        entry_time=position["entry_time"], entry_premium=position["entry_price"],
                        exit_time=bar_j[0] if bar_j else bars[i].ts, exit_premium=exit_price,
                        lot_size=position["lot_size"], exit_reason=exit_reason,
                    ))
                    position = None

            # -- hump/divergence tracking, every bar, regardless of position state --
            h = histogram[i]
            if h is not None and h != 0:
                sign = 1 if h > 0 else -1
                if sign != hump_sign:
                    hump_sign = sign
                    hump_bars = []
                    last_peak = None
                    last_trough = None
                hump_bars.append((i, h))

                window_len = 2 * peak_window + 1
                if len(hump_bars) >= window_len:
                    window_slice = hump_bars[-window_len:]
                    idx_mid, val_mid = window_slice[peak_window]
                    values = [v for _, v in window_slice]

                    if hump_sign == 1 and val_mid == max(values) and values.count(val_mid) == 1:
                        price_high_mid = bars[idx_mid].h
                        if last_peak is not None:
                            _, prev_val, prev_price_high = last_peak
                            if val_mid < prev_val and price_high_mid >= prev_price_high and position is None:
                                direction_label, opt_type = "SHORT", "PE"
                                entry_close = bars[i].c
                                atm = oc.round_to_step(entry_close, strike_step)
                                contract, opt_rows = _atm_option_candles(atm, opt_type, d, expiry)
                                if contract is not None and opt_rows:
                                    entry_bar = _fill(opt_rows, time_str)
                                    if entry_bar is not None:
                                        position = {
                                            "direction": direction_label, "strike": contract["strike_price"],
                                            "entry_time": bars[i].ts,
                                            "entry_price": _apply_slippage(entry_bar[4], "BUY", slippage_pct),
                                            "lot_size": contract["lot_size"], "opt_rows": opt_rows,
                                        }
                        last_peak = (idx_mid, val_mid, price_high_mid)

                    elif hump_sign == -1 and val_mid == min(values) and values.count(val_mid) == 1:
                        price_low_mid = bars[idx_mid].l
                        if last_trough is not None:
                            _, prev_val, prev_price_low = last_trough
                            if val_mid > prev_val and price_low_mid <= prev_price_low and position is None:
                                direction_label, opt_type = "LONG", "CE"
                                entry_close = bars[i].c
                                atm = oc.round_to_step(entry_close, strike_step)
                                contract, opt_rows = _atm_option_candles(atm, opt_type, d, expiry)
                                if contract is not None and opt_rows:
                                    entry_bar = _fill(opt_rows, time_str)
                                    if entry_bar is not None:
                                        position = {
                                            "direction": direction_label, "strike": contract["strike_price"],
                                            "entry_time": bars[i].ts,
                                            "entry_price": _apply_slippage(entry_bar[4], "BUY", slippage_pct),
                                            "lot_size": contract["lot_size"], "opt_rows": opt_rows,
                                        }
                        last_trough = (idx_mid, val_mid, price_low_mid)

        # any position still open at the very end of the day's bars (shouldn't
        # normally happen since FORCE_FLAT_TIME closes it first, but guard anyway)
        if position is not None:
            opt_rows = position["opt_rows"]
            last_premium = opt_rows[-1][4]
            exit_price = _apply_slippage(last_premium, "SELL", slippage_pct)
            trades.append(OptionTrade(
                date=d, direction=position["direction"], expiry=expiry, strike=position["strike"],
                entry_time=position["entry_time"], entry_premium=position["entry_price"],
                exit_time=opt_rows[-1][0], exit_premium=exit_price, lot_size=position["lot_size"],
                exit_reason="eod",
            ))

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
