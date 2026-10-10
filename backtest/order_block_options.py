"""Backtest: NIFTY 50 spot candles (5-minute by default), trade order
blocks -- a "smart money concepts" pattern -- through real ATM NIFTY
options, with a fixed rupee stop-loss/target.

SWING STRUCTURE: a bar p is a confirmed swing high once PIVOT_WINDOW
bars have passed on both sides and its high is the strict max over
that 2*PIVOT_WINDOW+1-bar window (confirmed PIVOT_WINDOW bars later,
so no lookahead -- same discipline as every other pivot-based module
in this codebase). Swing lows are the mirror (strict min).

ORDER BLOCK: when a later bar's CLOSE breaks above the most recent
still-unbroken confirmed swing high (a bullish break of structure),
scan backward from the bar just before the breakout (up to
OB_LOOKBACK_MAX bars) for the nearest DOWN-close candle -- the last
sell candle before the up-impulse that broke structure. Its own
high-low range becomes the bullish order block's zone: the theory is
that candle is where the last opposing (short) orders got filled
before being overrun, and price tends to return to "mitigate" that
zone before continuing higher. The bearish mirror: a close below the
most recent unbroken swing low scans backward for the nearest UP-close
candle, whose range becomes a bearish order block's zone. Breaking a
swing level consumes it -- a new order block in the same direction
needs a fresh confirmed swing first.

ENTRY (touch + reversal candle, not a bare touch): once an order
block's zone exists, wait for a LATER bar (never the breakout bar
itself -- see below) to trade into it (its own high/low range
overlapping the zone). If that same touching bar's close already
broke all the way through the far side of the zone (fully
invalidating it rather than reacting off it), the order block is
discarded right there. Otherwise the NEXT bar is the confirmation: if
its close reclaims back out the impulse side of the zone (above
zone_high for a bullish OB, below zone_low for a bearish one), the
trade fires at that close; if instead it closes through the far side,
the order block is discarded unconfirmed. Each order block can only
ever produce one trade (or none).

The breakout bar itself is deliberately excluded from touching its own
just-created zone, even though the zone is built from the candle right
before it: a zone that close to current price gets its low/high
grazed by ordinary wick noise on the very next bar almost every time,
which (in an earlier version of this module) registered as an
immediate "touch" at creation and fired the confirmation one bar
later with no real pullback ever having happened -- confirmation-
chasing an already-extended move, not a genuine retest. Manually
tracing three of that version's trades against the raw index bars
confirmed this was happening on all three. Requiring the touch to
land on a bar strictly after creation forces an actual gap/pullback
to occur first.

EXIT: fixed rupee P&L stop-loss/target on the option premium (x
lot_size), same convention as ema_cloud_pullback_options.py and every
other sl_rs/target_rs module in this codebase -- sl_rs (default 300)
or target_rs (default 600), whichever hits first, else force-flat at
FORCE_FLAT_TIME (15:25). Only one position open at a time; order-block
detection/tracking continues underneath it, but no new entry fires
until flat.

candle_minutes (default 5): resamples each day's 1-minute index
candles first. Every option fill and the FORCE_FLAT_TIME cutoff go
through a _decision_time() helper identical to
macd_histogram_1min_dualstop_options.py's.

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
from backtest.sweep_reclaim_breakout import _resample

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"
PIVOT_WINDOW = 3
OB_LOOKBACK_MAX = 10
STOP_LOSS_RS = 300.0
TARGET_RS = 600.0


@dataclass
class Bar:
    ts: str
    o: float
    h: float
    l: float
    c: float


@dataclass
class OrderBlock:
    direction: str  # LONG | SHORT
    zone_low: float
    zone_high: float
    state: str = "pending_touch"  # pending_touch | touched


def _apply_slippage(price: float, side: str, slippage_pct: float) -> float:
    if slippage_pct <= 0:
        return price
    return price * (1 + slippage_pct) if side == "BUY" else price * (1 - slippage_pct)


def _find_order_blocks(bars: list[Bar]) -> list[tuple[int, OrderBlock]]:
    """One forward pass over a day's bars: returns [(created_idx, OrderBlock)]
    in the order each order block's zone became known (created_idx is the
    breakout bar's index -- the zone itself is already fixed by then, no
    lookahead involved in using it from created_idx onward)."""
    n = len(bars)
    highs = [b.h for b in bars]
    lows = [b.l for b in bars]

    swing_high_level: float | None = None
    swing_low_level: float | None = None
    obs: list[tuple[int, OrderBlock]] = []

    for i in range(n):
        p = i - PIVOT_WINDOW
        if p >= PIVOT_WINDOW:
            window_h = highs[p - PIVOT_WINDOW:p + PIVOT_WINDOW + 1]
            if highs[p] == max(window_h) and window_h.count(highs[p]) == 1:
                swing_high_level = highs[p]
            window_l = lows[p - PIVOT_WINDOW:p + PIVOT_WINDOW + 1]
            if lows[p] == min(window_l) and window_l.count(lows[p]) == 1:
                swing_low_level = lows[p]

        if swing_high_level is not None and bars[i].c > swing_high_level:
            for q in range(i - 1, max(-1, i - 1 - OB_LOOKBACK_MAX), -1):
                if bars[q].c < bars[q].o:
                    obs.append((i, OrderBlock("LONG", bars[q].l, bars[q].h)))
                    break
            swing_high_level = None

        if swing_low_level is not None and bars[i].c < swing_low_level:
            for q in range(i - 1, max(-1, i - 1 - OB_LOOKBACK_MAX), -1):
                if bars[q].c > bars[q].o:
                    obs.append((i, OrderBlock("SHORT", bars[q].l, bars[q].h)))
                    break
            swing_low_level = None

    return obs


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    candle_minutes: int = 5,
    sl_rs: float = STOP_LOSS_RS,
    target_rs: float = TARGET_RS,
    slippage_pct: float = 0.0,
    access_token: str | None = None,
) -> list[OptionTrade]:
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
        if candle_minutes <= 1:
            return ts[11:16]
        hh, mm = int(ts[11:13]), int(ts[14:16])
        dh, dm = divmod(hh * 60 + mm + candle_minutes, 60)
        return f"{dh:02d}:{dm:02d}"

    trades: list[OptionTrade] = []
    min_bars = 2 * PIVOT_WINDOW + 5

    for day in trading_days:
        d = day["date"]
        rows_1min = sorted(cache.get_day_candles_cached(underlying_key, "1minute", d, expired=False), key=lambda r: r[0])
        if len(rows_1min) < min_bars:
            continue
        rows = _resample(rows_1min, candle_minutes) if candle_minutes > 1 else rows_1min
        if len(rows) < min_bars:
            continue
        bars = [Bar(ts=r[0], o=r[1], h=r[2], l=r[3], c=r[4]) for r in rows]

        expiry = next((e for e in expiries if e >= d), None)
        if expiry is None:
            continue

        obs_by_created_idx = _find_order_blocks(bars)
        active_obs: list[OrderBlock] = []
        ob_cursor = 0  # next index into obs_by_created_idx to activate

        i = 0
        while i < len(bars):
            decision_time_str = _decision_time(bars[i].ts)
            if decision_time_str >= FORCE_FLAT_TIME:
                break

            direction_label = opt_type = None
            still_active: list[OrderBlock] = []
            for ob in active_obs:
                if direction_label is not None:
                    still_active.append(ob)  # a trade already fired this bar; leave the rest untouched
                    continue

                touching = bars[i].l <= ob.zone_high and bars[i].h >= ob.zone_low
                close = bars[i].c

                if ob.state == "pending_touch":
                    if touching:
                        if ob.direction == "LONG" and close < ob.zone_low:
                            continue  # closed straight through -- invalidated
                        if ob.direction == "SHORT" and close > ob.zone_high:
                            continue  # closed straight through -- invalidated
                        ob.state = "touched"
                        still_active.append(ob)
                    else:
                        still_active.append(ob)
                elif ob.state == "touched":
                    if ob.direction == "LONG" and close > ob.zone_high:
                        direction_label, opt_type = "LONG", "CE"
                    elif ob.direction == "SHORT" and close < ob.zone_low:
                        direction_label, opt_type = "SHORT", "PE"
                    # else: didn't confirm this bar -- discarded either way (not re-added)

            active_obs = still_active

            # newly created order blocks join active_obs only AFTER this
            # bar's touch/confirm processing above -- so the earliest a
            # zone can register a "touch" is the bar AFTER it was created,
            # never the breakout bar itself. That breakout bar's own low
            # routinely wicks back into a zone that close to it just from
            # noise, which was firing a touch+confirm (and so a trade) one
            # bar later with no real pullback ever having happened -- not
            # a retest, just confirmation-chasing an already-extended move.
            while ob_cursor < len(obs_by_created_idx) and obs_by_created_idx[ob_cursor][0] == i:
                active_obs.append(obs_by_created_idx[ob_cursor][1])
                ob_cursor += 1

            if direction_label is None:
                i += 1
                continue

            entry_idx = i
            entry_close = bars[entry_idx].c
            entry_ts = bars[entry_idx].ts
            entry_decision_time = _decision_time(entry_ts)
            atm = oc.round_to_step(entry_close, strike_step)

            contract, opt_rows = _atm_option_candles(atm, opt_type, d, expiry)
            if contract is None or not opt_rows:
                i = entry_idx + 1
                continue

            entry_bar = _fill(opt_rows, entry_decision_time)
            if entry_bar is None:
                i = entry_idx + 1
                continue
            entry_price = _apply_slippage(entry_bar[4], "BUY", slippage_pct)
            lot_size = contract["lot_size"]

            exit_idx = None
            exit_reason = None
            for j in range(entry_idx + 1, len(bars)):
                j_decision_time = _decision_time(bars[j].ts)
                if j_decision_time >= FORCE_FLAT_TIME:
                    exit_idx = j
                    exit_reason = "eod"
                    break
                bar_j = _fill(opt_rows, j_decision_time)
                premium_j = bar_j[4] if bar_j else entry_price
                pnl_per_lot = (premium_j - entry_price) * lot_size
                if pnl_per_lot <= -sl_rs:
                    exit_idx = j
                    exit_reason = "stop_loss"
                    break
                if pnl_per_lot >= target_rs:
                    exit_idx = j
                    exit_reason = "target"
                    break

            if exit_idx is None:
                exit_idx = len(bars) - 1
                exit_reason = "eod"

            exit_decision_time = _decision_time(bars[exit_idx].ts)
            exit_bar = _fill(opt_rows, exit_decision_time)
            exit_price = _apply_slippage(exit_bar[4], "SELL", slippage_pct) if exit_bar else entry_price
            exit_time = exit_bar[0] if exit_bar else bars[exit_idx].ts

            trades.append(OptionTrade(
                date=d, direction=direction_label, expiry=expiry, strike=contract["strike_price"],
                entry_time=entry_ts, entry_premium=entry_price, exit_time=exit_time, exit_premium=exit_price,
                lot_size=lot_size, exit_reason=exit_reason,
            ))

            i = exit_idx + 1

    return trades


def summary(trades: list[OptionTrade]) -> str:
    from backtest.macd_rsi2_momentum_options import summary as _summary
    return _summary(trades)
