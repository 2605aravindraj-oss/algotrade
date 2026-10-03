"""Opening-range breakout + EMA trend-ride, realized through real ATM
NIFTY options. 10-minute NIFTY 50 INDEX candles.

Range: each day's 09:35-09:45 candle -- the THIRD 10-minute bucket of
the session (NSE opens 09:15, so buckets are 09:15-09:24, 09:25-09:34,
09:35-09:44). That candle's own High and Low become the day's
breakout levels, fixed for the rest of the day. No range = no trade
that day (a day with a data gap right at 09:35 is simply skipped).

EMA(25) on the SAME 10-minute candles, continuous across the whole
date range (needs warm-up, not reset daily). Unlike
ema_sweep_breakout_options.py's multi-timeframe EMA, there's no
decision-time alignment needed here -- the range candle and the EMA
are both on the same 10-minute timeframe, so a bar's own EMA value is
already known at that same bar's own close.

Entry, checked on every candle AFTER the range candle, first
qualifying breakout wins (at most one trade per day):
    LONG (buy ATM CE):  a candle's High breaks above the range High
                         AND that same candle's Close is above EMA25.
    SHORT (buy ATM PE): a candle's Low breaks below the range Low
                         AND that same candle's Close is below EMA25
                         -- the mirror. Whichever direction breaks
                         first wins; the range is spent for the day
                         either way (no second attempt after an exit).

Exit -- "ride the trend", no fixed target:
    trend_reverse: a later candle CLOSES back on the wrong side of
                   EMA25 (below it for a LONG, above it for a SHORT).
    stop_loss:     the OPTION's own premium (not the index) falls 40%
                   from entry -- stop_level = entry_premium * (1 -
                   stop_loss_pct), default stop_loss_pct=0.40. Same
                   formula regardless of direction: LONG (bought CE)
                   and SHORT (bought PE) are both bought-premium
                   positions, so a "stop" always means the premium
                   ITSELF falling, matching every other premium-based
                   stop in this codebase. Checked before the
                   trend-reverse exit if both would trigger on the
                   same candle (conservative).
    eod:           forced flat at FORCE_FLAT_TIME, no overnight
                   position, same as every other intraday module here.

Fills: entry and every exit use the option's own price as of the
deciding candle's CLOSE (bucket start + candle_minutes), never its
start -- same decision-time-correct fill convention as
ema_sweep_breakout_options.py.
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, _bar_at_or_after, _bar_at_or_before
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.sweep_reclaim_breakout import _resample

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"


def _ema(values: list[float], period: int) -> list[float | None]:
    n = len(values)
    out: list[float | None] = [None] * n
    if n < period:
        return out
    k = 2 / (period + 1)
    out[period - 1] = sum(values[:period]) / period
    for i in range(period, n):
        out[i] = values[i] * k + out[i - 1] * (1 - k)
    return out


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    candle_minutes: int = 10,
    ema_period: int = 25,
    stop_loss_pct: float = 0.40,
    range_bucket_start: str = "09:35",
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
    if len(bars) < ema_period + 2:
        return []

    closes = [b[4] for b in bars]
    ema = _ema(closes, ema_period)

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
    position = None  # dict: direction, entry_time, entry_price, strike, expiry, lot_size, opt_type, date, stop_level, opt_by_time
    current_day: str | None = None
    range_high = range_low = None
    range_set_today = False
    entered_today = False

    for i, row in enumerate(bars):
        ts, o, h, l, c, v, oi = row
        d = ts[:10]
        time_str = ts[11:16]

        if d != current_day:
            current_day = d
            position = None
            range_high = range_low = None
            range_set_today = False
            entered_today = False

        atm = oc.round_to_step(c, strike_step)
        expiry = next((e for e in expiries if e >= d), None)

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

        if not range_set_today and time_str == range_bucket_start:
            range_high, range_low = h, l
            range_set_today = True
            continue  # the range candle can't also confirm its own breakout

        if position is not None:
            opt_hl = position["opt_by_time"].get(time_str)
            if opt_hl is not None and opt_hl[1] <= position["stop_level"]:
                _close("stop_loss")
            elif ema[i] is not None:
                if position["direction"] == "LONG" and c < ema[i]:
                    _close("trend_reverse")
                elif position["direction"] == "SHORT" and c > ema[i]:
                    _close("trend_reverse")

        if position is None and not entered_today and range_set_today and ema[i] is not None and expiry is not None:
            direction_label = None
            if h > range_high and c > ema[i]:
                direction_label = "LONG"
            elif l < range_low and c < ema[i]:
                direction_label = "SHORT"

            if direction_label is not None:
                entered_today = True
                opt_type = "CE" if direction_label == "LONG" else "PE"
                contract, candles = _atm_option_candles(atm, opt_type, d, expiry)
                if contract is not None and candles:
                    bar = _fill(candles, decision_time_str)
                    if bar is not None:
                        entry_price = bar[4]
                        opt_bars = _resample(candles, candle_minutes)
                        position = {
                            "direction": direction_label, "entry_time": bar[0], "entry_price": entry_price,
                            "strike": contract["strike_price"], "expiry": expiry,
                            "lot_size": contract["lot_size"], "opt_type": opt_type, "date": d,
                            "stop_level": entry_price * (1 - stop_loss_pct),
                            "opt_by_time": {b[0][11:16]: (b[2], b[3]) for b in opt_bars},
                        }

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
