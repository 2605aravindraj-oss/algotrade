"""Pilot: analyse the whole option chain (not just futures OI) for a
directional signal.

Each 5-minute bar, classify every strike in a band around ATM (both CE
and PE) using that specific option's OWN premium and OI change:

    premium up + OI up   -> buying    (CE: bullish, PE: bearish)
    premium down + OI up -> writing   (CE: bearish, PE: bullish)
    OI down (either)     -> ignored (unwind/covering -- not a fresh
                             directional conviction signal)

Net score = (bullish strike-readings this bar) - (bearish readings).
Entry when |net_score| clears a threshold; exit if it reverses hard or
weakens back through zero; forced flat at FORCE_FLAT_TIME.

Signal comes from the option chain; P&L is realized on the futures
contract's notional move (same "isolate the raw signal" approach used
to test the plain OI-buildup call) so this pilot answers one question
cleanly: does chain-wide analysis produce a better call than the single
futures-OI number did, before we worry about how to realize it via
options.
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME
from backtest.rsi2_5min_sar import _resample_5min
from backtest.rsi2_reversion import Trade, UNDERLYING_KEY


def run(
    from_date: str,
    to_date: str,
    futures_key: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    band_strikes: int = 5,
    net_score_threshold: int = 3,
    access_token: str | None = None,
    lot_size: int = 65,
) -> list[Trade]:
    trading_days = upstox_client.get_daily_history(underlying_key, from_date, to_date)
    expiries = sorted(cache.get_expired_expiries_cached(underlying_key, "options", access_token))
    chain_cache: dict[str, dict] = {}
    trades: list[Trade] = []

    for day in trading_days:
        d = day["date"]

        spot_candles = cache.get_day_candles_cached(underlying_key, "1minute", d, expired=False)
        entry_bar = oc.nearest_bar(spot_candles, "first")
        if entry_bar is None:
            continue
        atm = oc.round_to_step(entry_bar[1], strike_step)

        expiry = next((e for e in expiries if e >= d), None)
        if expiry is None:
            continue
        if expiry not in chain_cache:
            chain_cache[expiry] = oc.build_chain_lookup(
                cache.get_expired_option_chain_cached(underlying_key, expiry, access_token)
            )
        chain = chain_cache[expiry]

        strikes = [atm + i * strike_step for i in range(-band_strikes, band_strikes + 1)]
        per_strike_by_ts: dict[tuple, dict[str, list]] = {}
        for s in strikes:
            for t in ("CE", "PE"):
                contract = oc.nearest_contract(chain, s, t)
                if contract is None:
                    continue
                candles = cache.get_day_candles_cached(
                    contract["instrument_key"], "1minute", d, expired=True, access_token=access_token
                )
                if not candles:
                    continue
                bars5 = _resample_5min(sorted(candles, key=lambda c: c[0]))
                per_strike_by_ts[(s, t)] = {b[0]: b for b in bars5}

        fut_candles = cache.get_day_candles_cached(futures_key, "1minute", d, expired=False)
        if not fut_candles:
            continue
        fut_bars5 = _resample_5min(sorted(fut_candles, key=lambda c: c[0]))

        prev_state: dict[tuple, tuple[float, float]] = {}
        position: Trade | None = None

        def _close(t: Trade, ts: str, price: float, reason: str) -> None:
            t.exit_time = ts
            t.exit_price = price
            t.exit_reason = reason
            trades.append(t)

        for fbar in fut_bars5:
            ts, fo, fh, fl, fc, fv, foi = fbar
            time_str = ts[11:16]

            bullish = 0
            bearish = 0
            for key, series in per_strike_by_ts.items():
                bar = series.get(ts)
                if bar is None:
                    continue
                _, o, h, l, c, v, oi = bar
                if key in prev_state:
                    prev_close, prev_oi = prev_state[key]
                    if oi > prev_oi:
                        opt_type = key[1]
                        buying = c > prev_close
                        if (buying and opt_type == "CE") or (not buying and opt_type == "PE"):
                            bullish += 1
                        else:
                            bearish += 1
                prev_state[key] = (c, oi)
            net_score = bullish - bearish

            if position is None:
                if net_score >= net_score_threshold:
                    position = Trade(date=d, direction="LONG", entry_time=ts, entry_price=fc, lot_size=lot_size)
                elif net_score <= -net_score_threshold:
                    position = Trade(date=d, direction="SHORT", entry_time=ts, entry_price=fc, lot_size=lot_size)
            else:
                if time_str >= FORCE_FLAT_TIME:
                    _close(position, ts, fc, "eod")
                    position = None
                elif position.direction == "LONG" and net_score <= -net_score_threshold:
                    _close(position, ts, fc, "reverse")
                    position = Trade(date=d, direction="SHORT", entry_time=ts, entry_price=fc, lot_size=lot_size)
                elif position.direction == "LONG" and net_score < 0:
                    _close(position, ts, fc, "weaken")
                    position = None
                elif position.direction == "SHORT" and net_score >= net_score_threshold:
                    _close(position, ts, fc, "reverse")
                    position = Trade(date=d, direction="LONG", entry_time=ts, entry_price=fc, lot_size=lot_size)
                elif position.direction == "SHORT" and net_score > 0:
                    _close(position, ts, fc, "weaken")
                    position = None

        if position is not None:
            last = fut_bars5[-1]
            _close(position, last[0], last[4], "eod_data_end")

    return trades


def summary(trades: list[Trade]) -> str:
    from backtest.rsi2_reversion import summary as _summary
    return _summary(trades)
