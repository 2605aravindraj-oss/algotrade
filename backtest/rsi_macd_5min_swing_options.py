"""RSI(14)/MACD swing signal adapted to 5-minute NIFTY 50 INDEX candles,
realized through real ATM NIFTY options -- BOTH directions (buy ATM CE
on a bullish signal, buy ATM PE on a bearish signal), unlike
rsi_macd_daily_swing_options.py's long-only daily version this is
adapted from.

Signal (5-minute bars, continuous across the whole date range -- RSI
and MACD need warm-up, not reset daily):
    RSI(14) and a 14-period SMA of the RSI series ITSELF (see
    rsi_macd_daily_swing_options.py's docstring for why -- "RSI greater
    than SMA" only makes dimensional sense comparing two same-scale
    [0,100] oscillator values, not RSI against a price-level SMA).
    MACD(12,26,9). A FRESH crossover only -- the line moving from
    at/below to above its signal (bullish), or the mirror (bearish) --
    not every bar the line already happens to be on one side.

Entry -- one bar's hold, translating the daily version's "enter at
close, sell next day's close" to this timeframe as literally as
possible: hold for exactly one 5-minute bar, no more:
    LONG (buy ATM CE):  RSI(14) > SMA14(RSI) AND a bullish MACD cross,
                         same bar.
    SHORT (buy ATM PE): RSI(14) < SMA14(RSI) AND a bearish MACD cross,
                         same bar -- the mirror condition. The daily
                         version was long-only by the user's own
                         original spec; this is the natural symmetric
                         extension for trading both CE and PE here.
No stop-loss, no target -- exactly one bar, unconditionally, same as
the daily version. Entry fills at the deciding bar's own close; exit
fills at the NEXT bar's close, both decision-time-correct (bucket
start + candle_minutes), same fill convention as every other intraday
module in this codebase.

Intraday only: forced flat at FORCE_FLAT_TIME (no new entries once
reached; an already-open position still exits normally since its exit
bar was fixed at entry time), never carries a position across a day
boundary, and an entry isn't taken on a day's last bar -- there'd be
no same-day next bar to exit on.
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, _bar_at_or_after, _bar_at_or_before
from backtest.macd_rsi2_momentum import compute_rsi, compute_macd
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.rsi_macd_daily_swing_options import _sma_of_series
from backtest.sweep_reclaim_breakout import _resample

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    candle_minutes: int = 5,
    rsi_period: int = 14,
    rsi_sma_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
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
    if len(bars) < macd_slow + macd_signal + rsi_sma_period + 2:
        return []

    closes = [b[4] for b in bars]
    rsi = compute_rsi(closes, rsi_period)
    rsi_sma = _sma_of_series(rsi, rsi_sma_period)
    macd_line, signal_line = compute_macd(closes, macd_fast, macd_slow, macd_signal)

    bullish_cross = [False] * len(bars)
    bearish_cross = [False] * len(bars)
    for i in range(1, len(bars)):
        if (
            macd_line[i - 1] is not None and signal_line[i - 1] is not None
            and macd_line[i] is not None and signal_line[i] is not None
        ):
            if macd_line[i - 1] <= signal_line[i - 1] and macd_line[i] > signal_line[i]:
                bullish_cross[i] = True
            elif macd_line[i - 1] >= signal_line[i - 1] and macd_line[i] < signal_line[i]:
                bearish_cross[i] = True

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
    position = None  # dict: direction, entry_time, entry_price, strike, expiry, lot_size, opt_type, date, exit_bar_index
    current_day: str | None = None

    for i, row in enumerate(bars):
        ts, o, h, l, c, v, oi = row
        d = ts[:10]
        time_str = ts[11:16]

        if d != current_day:
            current_day = d
            position = None

        atm = oc.round_to_step(c, strike_step)
        expiry = next((e for e in expiries if e >= d), None)

        _dh, _dm = divmod(int(time_str[:2]) * 60 + int(time_str[3:5]) + candle_minutes, 60)
        decision_time_str = f"{_dh:02d}:{_dm:02d}"

        if position is not None and position["exit_bar_index"] == i:
            _, candles = _atm_option_candles(position["strike"], position["opt_type"], position["date"], position["expiry"])
            bar = _fill(candles, decision_time_str) if candles else None
            exit_price = bar[4] if bar else position["entry_price"]
            exit_time = bar[0] if bar else ts
            trades.append(OptionTrade(
                date=position["date"], direction=position["direction"], expiry=position["expiry"],
                strike=position["strike"], entry_time=position["entry_time"], entry_premium=position["entry_price"],
                exit_time=exit_time, exit_premium=exit_price, lot_size=position["lot_size"], exit_reason="next_bar_close",
            ))
            position = None

        if time_str >= FORCE_FLAT_TIME:
            continue

        if position is None and expiry is not None and i + 1 < len(bars) and bars[i + 1][0][:10] == d:
            direction_label = None
            if rsi[i] is not None and rsi_sma[i] is not None:
                if rsi[i] > rsi_sma[i] and bullish_cross[i]:
                    direction_label = "LONG"
                elif rsi[i] < rsi_sma[i] and bearish_cross[i]:
                    direction_label = "SHORT"

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
                            "exit_bar_index": i + 1,
                        }

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
