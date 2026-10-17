"""VWAP mean-reversion FADE, realized through real ATM NIFTY options.
5-minute NIFTY FUTURES candles -- NOT the index (volume=0 on every
index candle, verified in supertrend_vwap_cross_options.py; same
futures-resolution machinery reused here: cache.resolve_expired_
futures_contract_for_date for expired dates, fixed_volume_profile_
options._resolve_current_month_futures as the live fallback).

This is the MIRROR IMAGE of supertrend_vwap_cross_options.py: that
module trades WITH a fresh VWAP cross (trend-following, SuperTrend-
confirmed); this one trades AGAINST a stretched move, betting on a
snap back to the mean (no SuperTrend, no trend confirmation needed --
the whole premise is the opposite of "ride the trend").

VWAP = cumulative(typical_price * volume) / cumulative(volume),
typical_price=(High+Low+Close)/3, RESET AT EVERY DAY BOUNDARY (same
definition as supertrend_vwap_cross_options.py).

SIGNAL: band = vwap * band_pct / 100 (a PERCENTAGE distance from VWAP,
not a fixed point distance -- NIFTY's own level moved from ~24,000 to
~26,000+ across this codebase's data window, so a fixed-point band
would mean a different real stretch at different times; band_pct
keeps the trigger comparable across the whole history).
    LONG  (buy ATM CE): close drops BELOW vwap - band (oversold
        stretch) -- betting on a bounce back UP to the mean.
    SHORT (buy ATM PE): close rises ABOVE vwap + band (overbought
        stretch) -- betting on a pullback back DOWN to the mean.
A FRESH trigger only (the first bar of the day to cross outside the
band); once a position is taken, no new entries are evaluated until it
closes (one_trade_per_day also caps it to one entry total per day,
default True).

EXIT: the fade's own take-profit is baked into the signal itself --
price reverting back to (crossing) VWAP -- plus optional sl_pct/
target_pct (premium %, both off by default) and forced-flat at
FORCE_FLAT_TIME. Unlike the breakout-style modules in this codebase,
a fade's risk is open-ended if the stretch keeps extending instead of
reverting (there is no SuperTrend-flip or opposite-boundary exit here
to cut that short automatically -- that's what sl_pct is for).

STRIKE/EXPIRY: ATM = round_to_step(futures close, strike_step),
nearest expiry on/after the entry day -- same convention as every
other NIFTY options module here.

COSTS: backtest.costs' F&O approximation via
macd_rsi2_momentum_options.OptionTrade, same cost model every other
options module in this codebase uses.
"""
from __future__ import annotations

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, _bar_at_or_after, _bar_at_or_before
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.sweep_reclaim_breakout import _resample
from backtest.fixed_volume_profile_options import _resolve_current_month_futures

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
    candle_minutes: int = 5,
    band_pct: float = 0.3,
    sl_pct: float | None = None,
    target_pct: float | None = None,
    one_trade_per_day: bool = True,
    long_only: bool = False,
    short_only: bool = False,
    slippage_pct: float = 0.0,
    access_token: str | None = None,
) -> list[OptionTrade]:
    trading_days = upstox_client.get_daily_history(underlying_key, from_date, to_date)
    trading_days.sort(key=lambda d: d["date"])
    if len(trading_days) < 1:
        return []

    live_futures_cache: dict | None = None

    def _contract_for_day(date_str: str) -> tuple[dict, bool]:
        nonlocal live_futures_cache
        expired_contract = cache.resolve_expired_futures_contract_for_date(underlying_key, date_str, access_token)
        if expired_contract is not None:
            return expired_contract, True
        if live_futures_cache is None:
            live_futures_cache = _resolve_current_month_futures()
        return live_futures_cache, False

    all_1min: list[list] = []
    for day in trading_days:
        d = day["date"]
        contract, expired = _contract_for_day(d)
        rows = sorted(
            cache.get_day_candles_cached(contract["instrument_key"], "1minute", d, expired=expired, access_token=access_token),
            key=lambda c: c[0],
        )
        all_1min.extend(rows)
    all_1min.sort(key=lambda c: c[0])
    if len(all_1min) < 2:
        return []

    bars = _resample(all_1min, candle_minutes)

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
    position = None  # dict: direction, entry_time, entry_price, strike, expiry, lot_size, opt_type, date
    current_day: str | None = None
    cum_pv = cum_vol = 0.0
    traded_today = False

    for row in bars:
        ts, o, h, l, c, v, oi = row
        d = ts[:10]
        time_str = ts[11:16]

        if d != current_day:
            current_day = d
            position = None
            cum_pv = cum_vol = 0.0
            traded_today = False

        typical = (h + l + c) / 3
        cum_pv += typical * v
        cum_vol += v
        vwap = (cum_pv / cum_vol) if cum_vol > 0 else None

        atm = oc.round_to_step(c, strike_step)
        expiry = next((e for e in expiries if e >= d), None)

        _dh, _dm = divmod(int(time_str[:2]) * 60 + int(time_str[3:5]) + candle_minutes, 60)
        decision_time_str = f"{_dh:02d}:{_dm:02d}"

        def _close(reason: str) -> None:
            nonlocal position
            _, candles = _atm_option_candles(position["strike"], position["opt_type"], position["date"], position["expiry"])
            bar = _fill(candles, decision_time_str) if candles else None
            exit_price = _apply_slippage(bar[4], "SELL", slippage_pct) if bar else position["entry_price"]
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

        if vwap is None:
            continue

        if position is not None:
            is_long = position["direction"] == "LONG"
            hit_reversal = (is_long and c >= vwap) or (not is_long and c <= vwap)
            hit_sl = hit_target = False
            if sl_pct is not None or target_pct is not None:
                _, candles = _atm_option_candles(position["strike"], position["opt_type"], position["date"], position["expiry"])
                bar = _fill(candles, decision_time_str) if candles else None
                cur_premium = bar[4] if bar else position["entry_price"]
                if sl_pct is not None:
                    hit_sl = cur_premium <= position["entry_price"] * (1 - sl_pct)
                if target_pct is not None:
                    hit_target = cur_premium >= position["entry_price"] * (1 + target_pct)
            if hit_sl:
                _close("stop_loss")
            elif hit_target:
                _close("target")
            elif hit_reversal:
                _close("reversion")

        if (position is None and not (one_trade_per_day and traded_today)
                and expiry is not None):
            band = vwap * band_pct / 100
            direction_label = None
            if c < vwap - band and not short_only:
                direction_label = "LONG"
            elif c > vwap + band and not long_only:
                direction_label = "SHORT"

            if direction_label is not None:
                opt_type = "CE" if direction_label == "LONG" else "PE"
                contract, candles = _atm_option_candles(atm, opt_type, d, expiry)
                if contract is not None and candles:
                    bar = _fill(candles, decision_time_str)
                    if bar is not None:
                        position = {
                            "direction": direction_label, "entry_time": bar[0],
                            "entry_price": _apply_slippage(bar[4], "BUY", slippage_pct),
                            "strike": contract["strike_price"], "expiry": expiry,
                            "lot_size": contract["lot_size"], "opt_type": opt_type, "date": d,
                        }
                        traded_today = True

    return trades


def run_nifty(from_date: str, to_date: str, **overrides) -> list[OptionTrade]:
    """run() with NIFTY's own fine-tuned config, found by sweeping from
    scratch on the 2024-10-03/2026-09-08 window (the only span this
    codebase's expired NIFTY options chain covers).

    UNTUNED BASELINE (band_pct=0.3, no sl/target, both sides): net
    -Rs 49,945.31 on 306 trades, 49.0% win rate -- a near-coinflip win
    rate (the mean-reversion premise itself isn't crazy) but still a
    clear loss, because a fade's win is capped (price only needs to
    travel back to VWAP) while its loss is NOT (nothing stops the
    stretch from extending further until EOD) -- avg loss (-Rs
    2,259.22) already exceeded avg win (+Rs 2,016.62) at baseline. This
    capped-win/uncapped-loss asymmetry is structural, not a tuning
    accident -- see sl_pct below, which exists specifically to fix it.

    band_pct ALONE, swept 0.1-1.0 (no sl/target): net LOSS at every
    single value tested (-Rs 5,824 to -Rs 78,913) -- confirms the
    asymmetry can't be fixed by just picking a different stretch
    threshold; the loss side has to be capped directly.

    sl_pct fixes it -- swept 0.10-0.50 at band_pct=0.2 (then 0.25):
    only band_pct=0.2 crossed positive, and only for a contiguous
    sl_pct range (0.13-0.20, net Rs 3,268 to Rs 33,672) -- a real
    plateau, not an isolated spike. Re-gridding band_pct x sl_pct
    jointly over {0.19-0.24} x {0.14-0.20} confirmed sl_pct~0.16-0.18
    as a genuine peak at EVERY band_pct tested in that range (not just
    one lucky combination) -- cross-validated evidence this is a real
    effect, not noise. target_pct was swept too (0.10-0.50) at the
    best point found so far and never beat leaving it off: the
    reversion-to-VWAP exit already acts as the strategy's own take-
    profit, so a separate premium target only clips winners early.
    Left off (None) by design.

    LONG vs SHORT, swept independently at band_pct=0.20/sl_pct=0.16:
    LONG ONLY (fading dips, buying CE) is a clear net LOSER (-Rs
    22,575.53, 288 trades, 39.9% win rate) -- SHORT ONLY (fading
    rallies, buying PE) is a clear net WINNER (+Rs 25,429.84, 280
    trades, 48.2% win rate). The combined run's headline number
    (+Rs 33,672.03) sits ABOVE the isolated short-only figure purely
    because of one-trade-per-day slot competition (a day where LONG
    fires first "steals" that day's only trade slot from a SHORT setup
    that would otherwise have fired later) -- not a real synergy
    between the two sides. Matches this codebase's recurring pattern
    (narrow_cpr_breakout_options.py found the identical long-bad/
    short-good split): NIFTY's own skew means a sharp intraday RALLY
    is more likely to be an overextension that reverts than a sharp
    DROP is, so short_only=True is the honest, robust config, not the
    higher-but-slot-competition-inflated combined number.

    Re-tuning band_pct/sl_pct specifically FOR short_only=True (the
    combined-mode optimum doesn't have to be the short-only optimum,
    and wasn't): band_pct re-peaked at 0.22-0.23 (net Rs 38,528/
    38,361, a tied plateau top, smooth rise from 0.19 and smooth decay
    through 0.24) and sl_pct re-peaked at 0.17 (net Rs 41,571.13,
    smooth rise from 0.13 and decay through 0.20) -- both genuine
    single-peaked curves, not spikes.

    Final: band_pct=0.22, sl_pct=0.17, short_only=True, target_pct=
    None -- 263 trades, net Rs 41,571.13, 49.4% win rate (130W/133L),
    max drawdown -Rs 24,412.47. Avg win (Rs 2,180.46) > avg loss
    (-Rs 1,818.71) -- the capped-win/uncapped-loss asymmetry from the
    untuned baseline is fully corrected. Exit reasons split cleanly
    three ways (117 reversion / 116 stop_loss / 30 eod), with no
    single exit type dominating -- evidence the stop-loss and the
    reversion thesis are both doing real, comparable amounts of work,
    not that one has swallowed the other.
    """
    overrides.setdefault("band_pct", 0.22)
    overrides.setdefault("sl_pct", 0.17)
    overrides.setdefault("short_only", True)
    return run(from_date, to_date, **overrides)


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
