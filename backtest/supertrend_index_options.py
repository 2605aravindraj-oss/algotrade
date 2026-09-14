"""Supertrend stop-and-reverse (same signal as supertrend_percent_equity.py
-- computed directly on the NIFTY 50 INDEX, not futures), realized through
real ATM NIFTY options instead of a synthetic percent-of-equity notional.

Signal: Supertrend(period, multiplier) on index bars resampled to
bar_minutes (15 by default, matching the just-validated index backtest),
continuous across the whole date range. Positional, no daily flatten --
    direction flips bullish -> sell any PE, buy ATM CE
    direction flips bearish -> sell any CE, buy ATM PE
Exits: the next opposite flip, the held option's own expiry (forced out,
can't hold past it), or the end of the backtest window.

NIFTY options are WEEKLY, not monthly -- this is a real structural
friction for a positional strategy: even when the Supertrend signal
wants to hold for multiple weeks, the specific option contract bought
on entry expires within days, forcing a rollover (fresh ATM pick,
fresh theta clock) the strategy wouldn't face trading the index or
futures directly. Expect many more "expiry" exits here than "reverse"
exits when the signal's average hold is longer than the option's
remaining life at entry -- that's the real cost this module is built to
surface honestly, not a bug.

Each bar is labeled by its start time; the direction it confirms is
only knowable once that bucket's last 1-min candle closes
(+bar_minutes) -- same fill-pricing fix as futures_oi_buildup.py,
generalized to an arbitrary bar size instead of the fixed +5min.

P&L realized per standard lot (contract["lot_size"], whatever Upstox's
chain reports for that expiry), not percent-of-equity -- consistent
with every other options-realized module in this codebase.
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import _bar_at_or_after, _bar_at_or_before
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.supertrend import _compute_supertrend
from backtest.sweep_reclaim_breakout import _resample

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    bar_minutes: int = 15,
    period: int = 10,
    multiplier: float = 2.5,
    access_token: str | None = None,
    max_daily_profit: float | None = None,
    max_daily_loss: float | None = None,
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
    if not all_1min:
        return []

    bars = _resample(all_1min, bar_minutes)
    if len(bars) < period + 2:
        return []

    direction = _compute_supertrend(bars, period, multiplier)

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

    for i, bar in enumerate(bars):
        ts, o, h, l, c, v, oi = bar
        d = ts[:10]
        time_str = ts[11:16]
        dirn = direction[i]

        if d != current_day:
            current_day = d
            daily_pnl = 0.0
            day_halted = False

        _dh, _dm = divmod(int(time_str[:2]) * 60 + int(time_str[3:5]) + bar_minutes, 60)
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
            bar_ = _bar_at_or_after(candles, decision_time_str) or _bar_at_or_before(candles, decision_time_str)
            if bar_ is None:
                return
            position = {
                "direction": direction_label, "entry_time": bar_[0], "entry_price": bar_[4],
                "strike": contract["strike_price"], "expiry": expiry,
                "lot_size": contract["lot_size"], "opt_type": opt_type, "date": d,
            }

        def _exit(reason: str, as_of_date: str, use_last_bar: bool = False) -> None:
            nonlocal position, daily_pnl, day_halted
            if position is None:
                return
            _, candles = _atm_option_candles(position["strike"], position["opt_type"], as_of_date, position["expiry"])
            bar_ = None
            if candles:
                if use_last_bar:
                    # Forced expiry exit: as_of_date is the contract's LAST
                    # trading day, discovered from a LATER bar (the one that
                    # noticed d > expiry) -- that later bar's decision_time_str
                    # is a different day's clock and must not be matched
                    # against this day's candles (it can land before the
                    # entry time on the same day). Use the day's last traded
                    # price instead, like a real expiry settlement.
                    bar_ = candles[-1]
                else:
                    bar_ = _bar_at_or_after(candles, decision_time_str) or _bar_at_or_before(candles, decision_time_str)
            exit_price = bar_[4] if bar_ else position["entry_price"]
            exit_time = bar_[0] if bar_ else ts
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

        if position is not None and d > position["expiry"]:
            _exit("expiry", position["expiry"], use_last_bar=True)
            prev_dir = dirn
            continue

        if prev_dir is not None and dirn != prev_dir:
            if position is not None:
                _exit("reverse", d)
            _enter("LONG" if dirn == 1 else "SHORT")

        prev_dir = dirn

    if position is not None:
        as_of = min(position["expiry"], bars[-1][0][:10])
        _, candles = _atm_option_candles(position["strike"], position["opt_type"], as_of, position["expiry"])
        last_bar = candles[-1] if candles else None
        exit_price = last_bar[4] if last_bar else position["entry_price"]
        exit_time = last_bar[0] if last_bar else bars[-1][0]
        trades.append(OptionTrade(
            date=position["date"], direction=position["direction"], expiry=position["expiry"],
            strike=position["strike"], entry_time=position["entry_time"], entry_premium=position["entry_price"],
            exit_time=exit_time, exit_premium=exit_price, lot_size=position["lot_size"], exit_reason="eod_data_end",
        ))

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
