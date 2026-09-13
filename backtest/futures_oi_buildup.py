"""NIFTY futures OI-buildup scalp, realized through ATM options.

Classic 4-quadrant open-interest buildup analysis on the current-month
NIFTY futures contract, computed on 5-minute bars:

    price up   + OI up   -> Long Buildup    (fresh longs opening)  -- bullish
    price down + OI up   -> Short Buildup   (fresh shorts opening) -- bearish
    price up   + OI down -> Short Covering  (shorts closing)       -- mildly bullish
    price down + OI down -> Long Unwinding  (longs closing)        -- mildly bearish

Entries only fire on the high-conviction signals (Long/Short Buildup);
the weaker unwind/covering signals are used only to de-risk an existing
position, not to open a fresh one:

    flat   + Long Buildup  -> buy ATM call
    flat   + Short Buildup -> buy ATM put
    long   + Short Buildup -> sell the call, immediately buy ATM put (reverse)
    long   + Long Unwinding -> sell the call, go flat (no reverse)
    short  + Long Buildup  -> sell the put, immediately buy ATM call (reverse)
    short  + Short Covering -> sell the put, go flat (no reverse)

Forced flat before the day's close (no overnight position). Since this
trades a single specific futures contract, it only covers that
contract's active listing window (front-month futures are listed a few
months before expiry) -- it does not roll across contracts.

Futures OI/price history is public (no auth). Realizing the signal
through options requires an Upstox access token (expired-instruments
API) for the option premium history.
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.iron_condor import UNDERLYING_KEY
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.rsi2_5min_sar import _resample_5min

FORCE_FLAT_TIME = "15:25"

BUILDUP_LABELS = {
    (1, 1): "Long Buildup",
    (-1, 1): "Short Buildup",
    (1, -1): "Short Covering",
    (-1, -1): "Long Unwinding",
}


def classify_buildup(price_change: float, oi_change: float) -> str:
    if price_change == 0 or oi_change == 0:
        return "Neutral"
    return BUILDUP_LABELS[(1 if price_change > 0 else -1, 1 if oi_change > 0 else -1)]


def run(
    from_date: str,
    to_date: str,
    futures_key: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    access_token: str | None = None,
    futures_expired: bool = False,
    max_daily_profit: float | None = None,
    max_daily_loss: float | None = None,
) -> list[OptionTrade]:
    """futures_expired=True fetches the futures leg through the
    expired-instruments API instead of the public currently-listed-only
    endpoint -- pass a `futures_key` like "NSE_FO|62329|30-06-2026"
    (from upstox_client.get_expired_expiries(..., "futures") +
    a /expired-instruments/future/contract lookup) to backtest a window
    that predates the currently-listed futures contract's own listing.

    max_daily_profit / max_daily_loss: once realized net P&L for the
    current day reaches +max_daily_profit or -max_daily_loss, any open
    position is force-closed (exit_reason="daily_limit") and no new
    entries are taken for the rest of that day. Both are in rupees and
    both None by default (no cap). Resets at the start of each new day.
    """
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
    if len(all_5min) < 2:
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

    trades: list[OptionTrade] = []
    position = None  # dict: direction, entry_time, entry_price(premium), strike, expiry, lot_size
    prev_close = None
    prev_oi = None
    current_day = None
    daily_pnl = 0.0
    day_halted = False

    for i, row in enumerate(all_5min):
        ts, o, h, l, c, v, oi = row
        d = ts[:10]
        time_str = ts[11:16]

        if d != current_day:
            current_day = d
            daily_pnl = 0.0
            day_halted = False
        # Each 5-min bucket is labeled by its *start* (e.g. "10:30" for the
        # 10:30-10:34 window), but the buildup signal it produces is only
        # knowable once that window's last 1-min candle closes -- +5 min.
        # Option fills must be priced from that point, not the bucket's
        # start label, or every fill uses a price from before the signal
        # that triggered it even existed.
        _dh, _dm = divmod(int(time_str[:2]) * 60 + int(time_str[3:5]) + 5, 60)
        decision_time_str = f"{_dh:02d}:{_dm:02d}"

        if prev_close is None:
            prev_close, prev_oi = c, oi
            continue

        buildup = classify_buildup(c - prev_close, oi - prev_oi)
        atm = oc.round_to_step(c, strike_step)
        expiry = next((e for e in expiries if e >= d), None)

        def _enter(direction: str) -> None:
            nonlocal position
            if expiry is None or day_halted:
                return
            opt_type = "CE" if direction == "LONG" else "PE"
            contract, candles = _atm_option_candles(atm, opt_type, d, expiry)
            if contract is None or not candles:
                return
            bar = _bar_at_or_after(candles, decision_time_str) or _bar_at_or_before(candles, decision_time_str)
            if bar is None:
                return
            position = {
                "direction": direction, "entry_time": bar[0], "entry_price": bar[4],
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

        if position is None:
            if buildup == "Long Buildup":
                _enter("LONG")
            elif buildup == "Short Buildup":
                _enter("SHORT")
        else:
            if time_str >= FORCE_FLAT_TIME:
                _exit("eod")
            elif position["direction"] == "LONG" and buildup == "Short Buildup":
                _exit("reverse")
                _enter("SHORT")
            elif position["direction"] == "LONG" and buildup == "Long Unwinding":
                _exit("unwind")
            elif position["direction"] == "SHORT" and buildup == "Long Buildup":
                _exit("reverse")
                _enter("LONG")
            elif position["direction"] == "SHORT" and buildup == "Short Covering":
                _exit("covering")

        prev_close, prev_oi = c, oi

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


def _bar_at_or_before(rows, time_str):
    candidates = [r for r in rows if r[0][11:16] <= time_str]
    return candidates[-1] if candidates else None


def _bar_at_or_after(rows, time_str):
    for r in rows:
        if r[0][11:16] >= time_str:
            return r
    return None


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
