"""SuperTrend + VWAP crossover trend-following, realized through real
ATM NIFTY options. 5-minute NIFTY FUTURES candles -- NOT the index,
which carries volume=0 on every candle (verified earlier this
session) and so can't support a genuinely volume-WEIGHTED VWAP, same
reasoning as fixed_volume_profile_options.py. The current/historical
front-month futures contract is resolved per day via
cache.resolve_expired_futures_contract_for_date (the expired-futures
resolver built for that module); SuperTrend, VWAP, and the ATM strike
are all derived from the futures price throughout (futures and index
track closely; a simplification, not exact).

SIGNAL: SuperTrend(10, 3) -- this codebase's own standard default
    period/multiplier (backtest/supertrend.py), not stated in the
    request. VWAP = cumulative(typical_price * volume) / cumulative
    (volume), typical_price=(High+Low+Close)/3, RESET AT EVERY DAY
    BOUNDARY (the standard definition -- VWAP is always a same-day
    running average, never carried across days).
    LONG (buy ATM CE): SuperTrend is bullish (direction=+1) AND price
        CROSSES above VWAP on this candle -- a FRESH cross (previous
        candle's close was at/below VWAP, this candle's close is
        above it), not merely "is above", since the request says
        "crossed".
    SHORT (buy ATM PE): SuperTrend bearish (direction=-1) AND price
        crosses below VWAP -- the mirror.
Entry fills at the signal candle's own close.

EXIT: "stop loss [is] price cross below supertrend" (mirror: above,
for the short side) -- read as the SAME event that flips SuperTrend's
own direction against the position (SuperTrend's direction is BY
DEFINITION whether price is above/below its own line, so "price
crosses below the SuperTrend line" and "SuperTrend direction flips
bearish" are one and the same event, not two separate checks). No
profit target was given, so this is also the ONLY way out short of
the forced-flat close -- ride the trend until SuperTrend reverses,
same "trend_flip" exit convention as supertrend_pivot_options.py
(exit_reason="trend_reverse"). Forced flat at FORCE_FLAT_TIME.

sl_pct (default 0.15): an ADDED protective floor beyond what the
original spec called for, since backtesting it bare (ride-until-
reversal only) showed large drawdowns (-Rs 25,570 to -Rs 36,588
across the 4 originally-tested windows). When set, the position also
exits (exit_reason="stop_loss", checked BEFORE the trend-reversal
exit if both would trigger on the same bar) the moment the option's
own premium falls to entry_price*(1-sl_pct) -- same
percentage-of-premium convention as other modules here (e.g.
orb_ema_ride_options.py's 40% stop). This is a deliberate addition on
top of the sourced strategy, not part of its original rules --
sl_pct=None reproduces the original unmodified behavior exactly.

sl_pct=0.10 was the first-pass choice (chosen by sweeping 0.05-0.40
on the 4 originally-established windows, BEFORE ema_filter_period and
the chop filter existed -- see their own history below). Once those
two filters were added, 0.10 was never rechecked against the now
higher-quality, less-noisy entry signal -- a known risk of tuning
params sequentially instead of jointly. Re-swept on the full
continuous 2024-10-01/2026-09-08 backtest (738 trades at 0.10) across
{0.15, 0.20, 0.25, 0.30, 0.40, 0.50}: 0.15 is a genuine free upgrade,
not a tradeoff -- win rate up from 23.6% to 27.4%, net P&L essentially
unchanged (Rs 107,815 vs Rs 109,128, -1.2%), AND max drawdown improves
(-Rs 42,343 vs -Rs 46,953, the best of every value tested, 0.10
included). 0.20 pushes win rate further (29.8%) for a similarly small
P&L cost with drawdown roughly flat; beyond 0.25 the trade stops
being worth it (meaningfully lower P&L AND worse drawdown at every
step up to 0.50). The stop causes more same-day re-entries after a
shaken-out position (trade count drops from 738 to 658 at 0.15 as
fewer get shaken out at all), which is why sl_pct's effect on a
window isn't a simple monotonic loss cap -- it's path-dependent
re-entry behavior.

target_pct (default None -- kept off, see below): a second ADDED
exit, same percentage-of-premium convention as sl_pct -- when set,
the position also exits (exit_reason="target") the moment the
option's own premium rises to entry_price*(1+target_pct), checked
ahead of the trend-reversal exit (SL is still checked first of all
three). Also not part of the original sourced spec -- target_pct=
None preserves the sl_pct-only behavior above exactly.

Swept {0.15, 0.25, 0.35, 0.5, 0.75, 1.0} x sl_pct=0.10 across all 4
established windows: EVERY value makes total net P&L worse than no
target at all (Rs 42,978 with no target vs a best of Rs 21,317 at
target_pct=0.35, and as low as -Rs 20,634 at 0.15); the worst case
(target_pct=0.5/0.75 on the 2026-05-16/2026-09-08 window) blows max
drawdown out to -Rs 45,951 to -Rs 50,235, beyond even the original
bare-strategy number. This strategy's edge comes from occasional
large trend-reversal-exit wins (best trades Rs 17,000-19,500); a
fixed percentage target chops exactly those trades short at a
fraction of their eventual move, trimming the tail that carries the
whole P&L. Conclusion: do not enable target_pct here -- it stays
None by design, not merely by default.

ema_filter_period (default 45): a third ADDED entry filter, on top
of the SuperTrend+VWAP cross signal -- when set, an EMA(ema_filter_
period) is computed over the full continuous futures close series
(not reset daily, same continuity as SuperTrend, since a trend filter
needs to see across day boundaries). A LONG signal is only taken if
the futures close is ABOVE the EMA; a SHORT signal only if it's
BELOW. The idea: filter out VWAP crosses that go against the
longer-term trend, which this strategy's whipsaw losses in chop
suggest are disproportionately the losing ones. Not part of the
original sourced spec -- ema_filter_period=None takes every signal,
unfiltered, exactly as before.

ema_filter_period=45 was chosen by sweeping {20, 28, 34, 40, 45, 50,
60, 70, 100, 150, 200} on all 4 established windows. Periods 40-60
form a genuine wide plateau (total net P&L Rs 57,526-64,231), well
above both the shorter end (20: Rs 19,378; 34: Rs 51,032) and the
longer end (100+: falling to Rs 1,617 by 200) -- a real structural
optimum, not an isolated spike. Within that plateau, 45 is the
best-balanced value: the highest total net P&L of any value tested
(Rs 63,830 vs Rs 42,978 with no filter, +48%) AND the best worst-case
single-window drawdown of any value tested (-Rs 27,334 vs -Rs 30,888
baseline). It improves net P&L in 3 of 4 windows and drawdown in 3 of
4 windows (the exception each time, 2025-05-01/2025-09-01, is only
modestly worse on the metric it misses).

chop_lookback_days=15 / chop_min_efficiency=0.07: a fourth ADDED
entry filter, a day-level regime gate on top of all the per-signal
ones above. This is a trend-following strategy (ride until
SuperTrend reverses), and a diagnosis of its one weak window
(2026-05-16/2026-09-08, net near breakeven despite an unchanged
~68% stop-out rate) found that window's NIFTY index was essentially
flat over its full span (net move -0.06%) with the lowest trend
efficiency (net move / sum of daily |moves|) of any tested window --
trend-reversal exits, this strategy's payoff mechanism, earned
~Rs 27/trade there vs Rs 1,194-1,648/trade elsewhere, because
SuperTrend kept flipping back and forth without a sustained move to
ride. When both are set (set either to None for the original
unfiltered behavior), each day's trailing `chop_lookback_days`-
trading-day INDEX closes (ending the prior trading day -- never
today's own still-forming close, so no lookahead) are used to
compute that same efficiency; if it's below chop_min_efficiency, NO
NEW entries are taken that day (an already-open position still
manages its exits normally, and the forced-flat close still
applies). The first `chop_lookback_days` trading days of any run
have no prior window and are never skipped (filter inactive until
enough history exists, rather than blocking a run's own warmup).

lookback=15/threshold=0.07 was chosen by sweeping lookback in
{15, 20, 30} x threshold in {0.03, 0.05, 0.08, 0.12} then refining
around the winner with {12, 15, 18} x {0.06, 0.07, 0.08, 0.09, 0.10}
on all 4 established windows. lookback=15 is a clear local optimum
in its own dimension -- lookback=12 and 18 both underperform it
substantially (W3 turns negative at 12, W4 turns sharply negative at
18) -- and within it, threshold=0.07 gives the highest total net
P&L of every combination tested (Rs 86,969 vs Rs 63,830 with no chop
filter, +36%), with ALL 4 windows positive for the first time (W1
Rs 22,551, W2 Rs 35,298, W3 Rs 19,365, W4 Rs 9,755 -- the previously
weak window). Worst-case single-window drawdown also improves, from
-Rs 27,334 to -Rs 22,161.

min_cross_distance_points (default None -- off): a fifth ADDED entry
filter, aimed directly at the stop-loss rate itself rather than at
resizing it (sl_pct controls how big a loss is paid, this controls
how OFTEN one is paid at all). ~68% of trades are stop-outs across
every sl_pct value tried, which doesn't move with the stop's size --
suggesting a lot of entries are marginal crosses (futures price pokes
a point or two past VWAP, then snaps back) rather than decisive
breaks. When set, a cross is only taken if the futures close is at
least min_cross_distance_points away from VWAP at the signal bar
(checked after the SuperTrend+VWAP-cross condition, before the EMA
filter) -- same bar, same entry timing, just a higher bar for what
counts as a real cross.

require_hold_bar (default False): a sixth ADDED entry filter, testing
persistence instead of magnitude (min_cross_distance_points tests
magnitude and, swept 5-50 points, left the stop-loss SHARE of trades
basically unchanged at 59-66% vs the unfiltered 68%, while collapsing
trade count and usually net P&L too -- a cross's SIZE isn't what
predicts whether it holds). When True, a fresh cross no longer enters
immediately: it's held as a pending signal, and only becomes a real
entry if price is STILL on the correct side of VWAP one bar later
(entering at that next bar's own decision-time-correct fill, one
candle_minutes later than an unheld entry would). If price has
already snapped back by then, the pending signal is discarded with no
trade and no re-arm until a genuinely fresh cross occurs.

Tested True on the full continuous backtest: it DOES cut the
stop-loss share further (58.1% vs ~66% unfiltered) and nudges win
rate up slightly (27.4% -> 27.8%), but net P&L drops 30% (Rs 107,815
-> Rs 75,208) and max drawdown gets WORSE (-Rs 42,343 -> -Rs 47,736).
Waiting one bar for confirmation means entering after the move has
already started, missing the best part of the real trend rides that
pay for everything else. Net loss, not a win -- stays False by
design. Between this and min_cross_distance_points, the ~60-68%
stop-loss rate looks like a structural feature of 5-minute
SuperTrend+VWAP crosses, not something fixable by filtering for a
"better" cross: the edge here comes from losing small and letting a
rare big trend ride pay for the rest (avg loss ~Rs 1,183 vs avg win
~Rs 3,740), not from winning often.

narrow_cpr_max_width_pct (default 0.26): a seventh ADDED entry
filter, a different day-level regime gate from chop_min_efficiency
but aimed at the same problem (not trading on days unlikely to
trend). Central Pivot Range, computed from the PRIOR trading day's
index H/L/C (no lookahead): pivot=(H+L+C)/3, BC=(H+L)/2,
TC=2*pivot-BC, width_pct=|TC-BC|/C*100. A narrow CPR (the pivot
range from yesterday's range was tight) is a common technical-
analysis heuristic for "today is more likely to trend"; a wide one
suggests more of yesterday's indecision carrying over. When set, a
day is skipped for new entries (exits still manage normally) if its
OWN width_pct (from the day before IT) exceeds narrow_cpr_max_width_pct
-- i.e. only genuinely narrow-CPR days get traded. On the 2024-2026
NIFTY daily history, width_pct's own distribution: median 0.148%,
p25 0.068%, p10 0.026%.

narrow_cpr_max_width_pct=0.26 is the strongest single lever found in
this strategy's whole tuning history. Swept {0.05, 0.08, 0.10, 0.15,
0.20, 0.22-0.28, 0.30, 0.32, 0.35, 0.40} on the full continuous
backtest: {0.24, 0.25, 0.26, 0.27, 0.28} form a tight, genuine
plateau (total net P&L Rs 157,435-170,520, ALL FIVE sharing the exact
same max drawdown, -Rs 29,416.25 -- the same single worst stretch
getting filtered out at every value in that band) -- not an isolated
spike. 0.26 is the peak: net P&L Rs 170,520 vs Rs 107,815 with the
filter off (+58%), win rate up (29.4% vs 27.4%), AND max drawdown down
30% (-Rs 29,416 vs -Rs 42,343). Both tails degrade clearly (down to
Rs 41,862 at 0.05; back down to ~Rs 149,000 and worse drawdown by
0.40), confirming this is a real structural optimum.

WALK-FORWARD VALIDATION (vs. the single full-period backtest above):
every number so far was tuned using the WHOLE 2024-10-03/2026-09-08
window at once, which risks fitting that specific history. To check,
sl_pct was re-picked from scratch on an EXPANDING trailing window
(train only on data strictly before each test quarter) and applied
PURELY out-of-sample to the next quarter, chained across 7 quarterly
folds (2025-01-01 through 2026-09-08; 2024-10-03/2024-12-31 spent as
the initial training-only seed, never tested). Every single fold,
independently, re-selected sl_pct=0.15 from {0.10, 0.15, 0.20, 0.25,
None} -- it never flipped to a different value once, which is strong
evidence 0.15 is a real structural optimum rather than a number that
happened to win on this window by luck.

Chained walk-forward result: 445 trades, 29.2% win rate, net
Rs 139,970, max drawdown -Rs 29,416 -- versus the full-period
backtest's 479 trades, 29.2% win rate (IDENTICAL), net Rs 163,227, max
drawdown -Rs 29,416 (also IDENTICAL -- the worst drawdown stretch
falls inside a walk-forward-tested quarter, using the same sl_pct=0.15
either way, so it reproduces exactly). The ~Rs 23,000 P&L gap is
almost entirely just the trades from the initial training-only
quarter (never counted as out-of-sample), not performance lost to
overfitting -- walk-forward recovers ~86% of the full-period net P&L
with an identical win rate and an identical worst case. Fold-by-fold:
6 of 7 quarters net positive (Rs 7,814 to Rs 42,039), one small loss
in the most recent quarter (2026-07-01/2026-09-08: -Rs 2,041) -- a
realistic soft patch, not a blowup. This is the strongest
overfitting-resistance result of any strategy in this codebase so far.

Caveat: only sl_pct was re-optimized per fold -- st_period,
st_multiplier, ema_filter_period, chop_lookback_days/
chop_min_efficiency, and narrow_cpr_max_width_pct were all held fixed
at their already-established values throughout, since a 3-6 month
expanding training window has too few trades to reliably re-sweep six
parameters jointly. This validates "does the already-tuned strategy's
edge survive honest out-of-sample testing", not "would a from-scratch
walk-forward optimizer have found the same parameters" -- a narrower
but still meaningful claim.

Decision-time-correct fills (bucket start + candle_minutes), one
position at a time, everything (VWAP accumulator, pending state)
resets at every day boundary. Requires an Upstox access token
(expired-instruments API, for both the futures leg on older dates and
the option premiums).
"""
from __future__ import annotations

from datetime import datetime, timedelta

from data_sources import cache, upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, _bar_at_or_after, _bar_at_or_before
from backtest.macd_rsi2_momentum_options import OptionTrade
from backtest.sweep_reclaim_breakout import _resample
from backtest.supertrend_pivot_options import _compute_supertrend_line
from backtest.fixed_volume_profile_options import _resolve_current_month_futures
from backtest.ema8_13_trend_sweep_options import _ema

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    candle_minutes: int = 5,
    st_period: int = 10,
    st_multiplier: float = 3.0,
    sl_pct: float | None = 0.15,
    target_pct: float | None = None,
    ema_filter_period: int | None = 45,
    chop_lookback_days: int | None = 15,
    chop_min_efficiency: float | None = 0.07,
    min_cross_distance_points: float | None = None,
    require_hold_bar: bool = False,
    narrow_cpr_max_width_pct: float | None = 0.26,
    access_token: str | None = None,
) -> list[OptionTrade]:
    trading_days = upstox_client.get_daily_history(underlying_key, from_date, to_date)
    trading_days.sort(key=lambda d: d["date"])
    if len(trading_days) < 1:
        return []

    chop_skip_days: set[str] = set()
    if chop_lookback_days is not None and chop_min_efficiency is not None:
        warmup_from = (
            datetime.strptime(from_date, "%Y-%m-%d") - timedelta(days=chop_lookback_days * 3)
        ).strftime("%Y-%m-%d")
        chop_days = upstox_client.get_daily_history(underlying_key, warmup_from, to_date)
        chop_days.sort(key=lambda d: d["date"])
        chop_closes = {d["date"]: d["close"] for d in chop_days}
        chop_dates = sorted(chop_closes)
        for day in trading_days:
            d = day["date"]
            idx = chop_dates.index(d)
            if idx < chop_lookback_days:
                continue
            window = [chop_closes[dd] for dd in chop_dates[idx - chop_lookback_days:idx]]
            net_move = abs(window[-1] - window[0])
            abs_moves = sum(abs(window[k] - window[k - 1]) for k in range(1, len(window)))
            efficiency = net_move / abs_moves if abs_moves > 0 else 0.0
            if efficiency < chop_min_efficiency:
                chop_skip_days.add(d)

    narrow_cpr_skip_days: set[str] = set()
    if narrow_cpr_max_width_pct is not None:
        cpr_warmup_from = (datetime.strptime(from_date, "%Y-%m-%d") - timedelta(days=10)).strftime("%Y-%m-%d")
        cpr_days = upstox_client.get_daily_history(underlying_key, cpr_warmup_from, to_date)
        cpr_days.sort(key=lambda d: d["date"])
        cpr_by_date = {d["date"]: d for d in cpr_days}
        cpr_dates = sorted(cpr_by_date)
        for day in trading_days:
            d = day["date"]
            idx = cpr_dates.index(d)
            if idx < 1:
                continue
            prev = cpr_by_date[cpr_dates[idx - 1]]
            h, l, c = prev["high"], prev["low"], prev["close"]
            pivot = (h + l + c) / 3
            bc = (h + l) / 2
            tc = 2 * pivot - bc
            width_pct = abs(tc - bc) / c * 100 if c else 0.0
            if width_pct > narrow_cpr_max_width_pct:
                narrow_cpr_skip_days.add(d)

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
    if len(bars) < st_period + 2:
        return []

    st_dir, _st_line = _compute_supertrend_line(bars, st_period, st_multiplier)
    ema_filter = _ema([b[4] for b in bars], ema_filter_period) if ema_filter_period else None

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
    pending_signal = None  # dict: direction, atm, opt_type -- a cross awaiting one more bar's confirmation
    current_day: str | None = None
    cum_pv = cum_vol = 0.0
    prev_close = prev_vwap = None

    for i, row in enumerate(bars):
        ts, o, h, l, c, v, oi = row
        d = ts[:10]
        time_str = ts[11:16]

        if d != current_day:
            current_day = d
            position = None
            pending_signal = None
            cum_pv = cum_vol = 0.0
            prev_close = prev_vwap = None

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
            pending_signal = None
            prev_close, prev_vwap = c, vwap
            continue

        if position is not None:
            cur_dir = st_dir[i]
            is_long = position["direction"] == "LONG"
            hit_reversal = cur_dir is not None and ((is_long and cur_dir == -1) or (not is_long and cur_dir == 1))
            hit_sl_pct = hit_target_pct = False
            if sl_pct is not None or target_pct is not None:
                _, candles = _atm_option_candles(position["strike"], position["opt_type"], position["date"], position["expiry"])
                bar = _fill(candles, decision_time_str) if candles else None
                cur_premium = bar[4] if bar else position["entry_price"]
                if sl_pct is not None:
                    hit_sl_pct = cur_premium <= position["entry_price"] * (1 - sl_pct)
                if target_pct is not None:
                    hit_target_pct = cur_premium >= position["entry_price"] * (1 + target_pct)
            if hit_sl_pct:
                _close("stop_loss")
            elif hit_target_pct:
                _close("target")
            elif hit_reversal:
                _close("trend_reverse")

        if position is None and pending_signal is not None and vwap is not None:
            still_valid = (
                (pending_signal["direction"] == "LONG" and c > vwap)
                or (pending_signal["direction"] == "SHORT" and c < vwap)
            )
            if still_valid:
                opt_type = pending_signal["opt_type"]
                contract, candles = _atm_option_candles(pending_signal["atm"], opt_type, d, expiry)
                if contract is not None and candles:
                    bar = _fill(candles, decision_time_str)
                    if bar is not None:
                        position = {
                            "direction": pending_signal["direction"], "entry_time": bar[0], "entry_price": bar[4],
                            "strike": contract["strike_price"], "expiry": expiry,
                            "lot_size": contract["lot_size"], "opt_type": opt_type, "date": d,
                        }
            pending_signal = None

        if (position is None and vwap is not None and prev_vwap is not None and prev_close is not None
                and st_dir[i] is not None and expiry is not None and d not in chop_skip_days
                and d not in narrow_cpr_skip_days):
            crossed_above = prev_close <= prev_vwap and c > vwap
            crossed_below = prev_close >= prev_vwap and c < vwap
            direction_label = None
            if st_dir[i] == 1 and crossed_above:
                direction_label = "LONG"
            elif st_dir[i] == -1 and crossed_below:
                direction_label = "SHORT"

            if direction_label is not None and min_cross_distance_points is not None:
                if abs(c - vwap) < min_cross_distance_points:
                    direction_label = None

            if direction_label is not None and ema_filter is not None:
                ema_val = ema_filter[i]
                if ema_val is None:
                    direction_label = None
                elif direction_label == "LONG" and c <= ema_val:
                    direction_label = None
                elif direction_label == "SHORT" and c >= ema_val:
                    direction_label = None

            if direction_label is not None:
                opt_type = "CE" if direction_label == "LONG" else "PE"
                if require_hold_bar:
                    pending_signal = {"direction": direction_label, "atm": atm, "opt_type": opt_type}
                else:
                    contract, candles = _atm_option_candles(atm, opt_type, d, expiry)
                    if contract is not None and candles:
                        bar = _fill(candles, decision_time_str)
                        if bar is not None:
                            position = {
                                "direction": direction_label, "entry_time": bar[0], "entry_price": bar[4],
                                "strike": contract["strike_price"], "expiry": expiry,
                                "lot_size": contract["lot_size"], "opt_type": opt_type, "date": d,
                            }

        prev_close, prev_vwap = c, vwap

    return trades


BANKNIFTY_UNDERLYING_KEY = "NSE_INDEX|Nifty Bank"


def run_banknifty(from_date: str, to_date: str, access_token: str | None = None, **overrides) -> list[OptionTrade]:
    """run() with Bank Nifty's own validated defaults, NOT the NIFTY
    ones (strike_step=100 -- Bank Nifty's near-the-money strikes step
    by 100, not NIFTY's 50). Every one of NIFTY's filters was
    re-swept from scratch on the full continuous 2024-10-01/2026-09-08
    Bank Nifty backtest rather than assumed to transfer, and most of
    them DON'T transfer:
      sl_pct=None -- every value from 0.10-0.40 underperformed the
        bare "ride until reversal" baseline (Rs 100,394); only 0.50
        (effectively a no-op) roughly tied it. A moderate stop cuts
        Bank Nifty positions off before the real move develops more
        often than it prevents a big loss -- the opposite of NIFTY.
      ema_filter_period=None -- every period tested reduced net P&L
        with only a drawdown trade-off, never dominating the
        unfiltered baseline on both metrics the way ema=45 did for
        NIFTY.
      chop_lookback_days=None, chop_min_efficiency=None -- every
        {15,20} x {0.05-0.20} combination tested badly underperformed
        the unfiltered baseline (several went net negative). The
        trend-efficiency regime read doesn't transfer to Bank Nifty's
        price action.
      narrow_cpr_max_width_pct=0.11 -- the one filter that DOES
        transfer, at a different threshold than NIFTY's 0.26 (Bank
        Nifty's own CPR-width distribution is wider: median 0.174%
        vs NIFTY's 0.148%). Swept 0.05-0.50 then refined around the
        peak: {0.09, 0.11, 0.115, 0.12} form a noisy but real
        plateau (net P&L Rs 109,815-117,863, max drawdown -Rs
        18,241 to -Rs 25,099), clearly above both a tighter filter
        (0.05: Rs 63,832) and a looser one (0.20+: falling back
        toward the unfiltered baseline). 0.11 is the best single
        value by risk-adjusted return: net P&L Rs 117,863 (actually
        ABOVE the Rs 100,394 unfiltered baseline) AND max drawdown
        down 76% (-Rs 18,241 vs -Rs 74,753), net/drawdown ratio 6.46
        vs the next-best region's (0.17-0.18, higher absolute P&L
        around Rs 128,000 but -Rs 44,000 drawdown) ratio of ~2.9.
    """
    overrides.setdefault("underlying_key", BANKNIFTY_UNDERLYING_KEY)
    overrides.setdefault("strike_step", 100)
    overrides.setdefault("sl_pct", None)
    overrides.setdefault("ema_filter_period", None)
    overrides.setdefault("chop_lookback_days", None)
    overrides.setdefault("chop_min_efficiency", None)
    overrides.setdefault("narrow_cpr_max_width_pct", 0.11)
    return run(from_date, to_date, access_token=access_token, **overrides)


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
