"""Supertrend stop-and-reverse (backtest/supertrend.py), realized through
ATM options instead of futures notional.

Same Supertrend(period, multiplier) direction series on 5-minute NIFTY
futures bars, continuous across the date range, reset to flat/no-signal
at the start of each day (only trades on an intraday flip, matching
supertrend.py):

    direction flips bullish -> sell any PE, buy ATM CE
    direction flips bearish -> sell any CE, buy ATM PE

Forced flat before FORCE_FLAT_TIME. Each 5-min bucket is labeled by its
start time, but the direction it produces is only knowable once that
bucket's last 1-min candle closes (+bar_minutes) -- exactly the
fill-pricing issue fixed in futures_oi_buildup.py. Fills here are looked
up from that same true decision time, not the bucket's start label.

Optional max_daily_profit / max_daily_loss: once realized net P&L for
the day reaches either threshold, no new entries are taken for the rest
of that day (mirrors futures_oi_buildup.run).
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, _bar_at_or_after, _bar_at_or_before
from backtest.iron_condor import UNDERLYING_KEY
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.rsi2_5min_sar import _resample_5min
from backtest.supertrend import _compute_supertrend


def run(
    from_date: str,
    to_date: str,
    futures_key: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    period: int = 10,
    multiplier: float = 3.0,
    access_token: str | None = None,
    futures_expired: bool = False,
    max_daily_profit: float | None = None,
    max_daily_loss: float | None = None,
) -> list[OptionTrade]:
    if futures_expired:
        raw_days = upstox_client.get_expired_candles(futures_key, "day", to_date, from_date, access_token)
        fut_days = [{"date": d} for d in sorted({c[0][:10] for c in raw_days})]
    else:
        fut_days = upstox_client.get_daily_history(futures_key, from_date, to_date)

    all_5min: list[list] = []
    for day in fut_days:
        rows_1min = sorted(
            cache.get_day_candles_cached(futures_key, "1minute", day["date"], expired=futures_expired, access_token=access_token),
            key=lambda c: c[0],
        )
        all_5min.extend(_resample_5min(rows_1min))
    all_5min.sort(key=lambda c: c[0])
    if len(all_5min) < period + 2:
        return []

    direction = _compute_supertrend(all_5min, period, multiplier)

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

    trades: list[OptionTrade] = []
    position = None  # dict: direction, entry_time, entry_price(premium), strike, expiry, lot_size, opt_type, date
    prev_dir: int | None = None
    current_day: str | None = None
    daily_pnl = 0.0
    day_halted = False

    for i, row in enumerate(all_5min):
        ts, o, h, l, c, v, oi = row
        d = ts[:10]
        time_str = ts[11:16]
        dirn = direction[i]

        if d != current_day:
            current_day = d
            daily_pnl = 0.0
            day_halted = False
            prev_dir = None  # don't carry a stale flip across the overnight gap

        # Bucket labeled by start time; the flip it confirms is only knowable
        # once that bucket's last 1-min candle closes (+5 min). Same fix as
        # futures_oi_buildup.py.
        _dh, _dm = divmod(int(time_str[:2]) * 60 + int(time_str[3:5]) + 5, 60)
        decision_time_str = f"{_dh:02d}:{_dm:02d}"

        if dirn is None:
            continue

        atm = oc.round_to_step(c, strike_step)
        expiry = next((e for e in expiries if e >= d), None)

        def _enter(direction_label: str) -> None:
            nonlocal position
            if expiry is None or day_halted:
                return
            opt_type = "CE" if direction_label == "LONG" else "PE"
            contract, candles = _atm_option_candles(atm, opt_type, d, expiry)
            if contract is None or not candles:
                return
            bar = _bar_at_or_after(candles, decision_time_str) or _bar_at_or_before(candles, decision_time_str)
            if bar is None:
                return
            position = {
                "direction": direction_label, "entry_time": bar[0], "entry_price": bar[4],
                "strike": contract["strike_price"], "expiry": expiry,
                "lot_size": contract["lot_size"], "opt_type": opt_type, "date": d,
            }

        def _exit(reason: str) -> None:
            nonlocal position, daily_pnl, day_halted
            if position is None:
                return
            _, candles = _atm_option_candles(position["strike"], position["opt_type"], position["date"], position["expiry"])
            bar = None
            if candles:
                bar = _bar_at_or_after(candles, decision_time_str) or _bar_at_or_before(candles, decision_time_str)
            exit_price = bar[4] if bar else position["entry_price"]
            exit_time = bar[0] if bar else ts
            trade = OptionTrade(
                date=position["date"], direction=position["direction"], expiry=position["expiry"],
                strike=position["strike"], entry_time=position["entry_time"], entry_premium=position["entry_price"],
                exit_time=exit_time, exit_premium=exit_price, lot_size=position["lot_size"], exit_reason=reason,
            )
            trades.append(trade)
            position = None
            daily_pnl += trade.pnl_rupees
            if (max_daily_profit is not None and daily_pnl >= max_daily_profit) or (
                max_daily_loss is not None and daily_pnl <= -max_daily_loss
            ):
                day_halted = True

        if time_str >= FORCE_FLAT_TIME:
            if position is not None:
                _exit("eod")
            prev_dir = dirn
            continue

        if prev_dir is not None and dirn != prev_dir:
            if position is not None:
                _exit("reverse")
            _enter("LONG" if dirn == 1 else "SHORT")

        prev_dir = dirn

    if position is not None:
        _, candles = _atm_option_candles(position["strike"], position["opt_type"], position["date"], position["expiry"])
        last_bar = candles[-1] if candles else None
        exit_price = last_bar[4] if last_bar else position["entry_price"]
        exit_time = last_bar[0] if last_bar else all_5min[-1][0]
        trades.append(OptionTrade(
            date=position["date"], direction=position["direction"], expiry=position["expiry"],
            strike=position["strike"], entry_time=position["entry_time"], entry_premium=position["entry_price"],
            exit_time=exit_time, exit_premium=exit_price, lot_size=position["lot_size"], exit_reason="eod_data_end",
        ))

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
