"""Check the SuperTrend+VWAP crossover options strategy
(backtest/supertrend_vwap_cross_options.py) against TODAY's real
futures/option prices. Read-only -- no order is ever placed, same
"simulated fill, real prices" spirit as live/synthetic_straddle_today.py
and live/ema_sweep_paper_trader.py.

WHY THIS IS A SEPARATE MODULE FROM THE BACKTEST: the backtest reads
futures/option premiums via Upstox's *expired*-instruments API, which
has no data at all for the still-live current contracts -- it can
never be pointed at "today". This module uses the same two no-auth
data sources as the other live/ tools instead:
    Warmup history (so SuperTrend's ATR has something to compute on
        before today's own bars accumulate): upstox_client.
        get_historical_candles on the CURRENT front-month futures
        contract -- this works pre-expiry too (a day's data publishes
        the day after it closes), unlike the expired-instruments API.
    Today (live): upstox_client.get_intraday_candles -- "today so
        far" directly off the exchange feed, no auth needed.
    Current futures contract: the live instrument master, via
        backtest.fixed_volume_profile_options._resolve_current_month_futures
        (same helper the backtest uses for the still-current month).
    Current option chain: the live instrument master filtered to
        NIFTY CE/PE (same approach as live/synthetic_straddle_today.py's
        _load_nifty_weekly_chain, NOT filtered by the master's own
        `weekly` flag -- that flag is false on a contract's own expiry
        day, a real bug found and fixed in that module).

SIGNAL/EXIT LOGIC: identical to backtest.supertrend_vwap_cross_options
-- SuperTrend(ST_PERIOD, ST_MULTIPLIER) direction + a fresh VWAP cross
on CANDLE_MINUTES futures bars, filtered by EMA_FILTER_PERIOD (only
takes the signal if price is on the trend-confirming side of that
EMA) and by the CHOP_LOOKBACK_DAYS/CHOP_MIN_EFFICIENCY day-level
regime gate (no new entries at all on a day whose trailing INDEX
trend efficiency is too low -- see that module's docstring for the
full reasoning), triggers an ATM CE/PE buy; SL_PCT premium stop
(checked first), fixed profit target (TARGET_PCT, off by design --
see that module's docstring for why), SuperTrend reversal, or
forced-flat at FORCE_FLAT_TIME closes it. VWAP resets at every day
boundary; SuperTrend is computed over the full warmup+today series
(it needs the continuity) but the day's trading -- entries, the open
position, VWAP -- is simulated starting fresh at today's first bar
only, same per-day reset the backtest applies.

The historical warmup window is 7 calendar days of 1-minute futures
candles (comfortably more than ST_PERIOD's 10-bar ATR lookback even
across a weekend/holiday), concatenated with today's intraday
candles before resampling -- so SuperTrend's direction at today's
open is already a settled, warmed-up value, not a cold start.

The latest resampled bar is DROPPED if its own bucket hasn't fully
elapsed yet (wall-clock check against IST) -- otherwise a half-formed
bar could produce a signal or exit based on an incomplete candle.

Run directly (`python -m live.supertrend_vwap_cross_live`) for a
one-shot snapshot of today so far: current SuperTrend direction and
VWAP, each trade that would already have closed today, and the
currently open position (if any). This is a single check, not a
poller -- rerun it whenever you want a fresh read; it persists no
state between runs (same as the other live/ tools), so a position
that opened an hour ago is re-derived from today's bars every time,
not remembered.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from data_sources import upstox_client
from backtest import options_common as oc
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, _bar_at_or_after, _bar_at_or_before
from backtest.sweep_reclaim_breakout import _resample
from backtest.supertrend_pivot_options import _compute_supertrend_line
from backtest.fixed_volume_profile_options import _resolve_current_month_futures
from backtest.ema8_13_trend_sweep_options import _ema
from live.synthetic_straddle_today import _load_nifty_weekly_chain, _nearest_unexpired_expiry, _contract_for_strike

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"
STRIKE_STEP = 50
CANDLE_MINUTES = 5
ST_PERIOD = 10
ST_MULTIPLIER = 3.0
SL_PCT = 0.10
TARGET_PCT = None  # deliberately off -- see backtest module's docstring
EMA_FILTER_PERIOD = 45
CHOP_LOOKBACK_DAYS = 15
CHOP_MIN_EFFICIENCY = 0.07
WARMUP_DAYS = 7
_IST = timezone(timedelta(hours=5, minutes=30))


def _drop_incomplete_last_bar(bars: list[list], candle_minutes: int) -> list[list]:
    if not bars:
        return bars
    now_ist = datetime.now(_IST)
    last_ts = bars[-1][0]
    bh, bm = int(last_ts[11:13]), int(last_ts[14:16])
    bucket_end = now_ist.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
        days=0, hours=bh, minutes=bm + candle_minutes
    )
    if last_ts[:10] == now_ist.date().isoformat() and now_ist < bucket_end:
        return bars[:-1]
    return bars


def _fetch_futures_bars() -> list[list]:
    contract = _resolve_current_month_futures()
    today = date.today()
    hist = upstox_client.get_historical_candles(
        contract["instrument_key"], "1minute",
        to_date=(today - timedelta(days=1)).isoformat(),
        from_date=(today - timedelta(days=WARMUP_DAYS)).isoformat(),
    )
    hist_rows = hist.get("data", {}).get("candles", [])
    today_rows = upstox_client.get_intraday_candles(contract["instrument_key"], "1minute")
    all_rows = sorted(hist_rows + today_rows, key=lambda r: r[0])
    bars = _resample(all_rows, CANDLE_MINUTES)
    return _drop_incomplete_last_bar(bars, CANDLE_MINUTES)


def _fill(candles, at_time):
    return _bar_at_or_after(candles, at_time) or _bar_at_or_before(candles, at_time)


def _is_chop_day() -> bool:
    """Same day-level regime gate as backtest.supertrend_vwap_cross_
    options's chop_lookback_days/chop_min_efficiency: trailing
    CHOP_LOOKBACK_DAYS INDEX closes ending yesterday (no lookahead),
    trend efficiency = net move / sum of |daily moves|."""
    today = date.today()
    days = upstox_client.get_daily_history(
        UNDERLYING_KEY,
        (today - timedelta(days=CHOP_LOOKBACK_DAYS * 3)).isoformat(),
        (today - timedelta(days=1)).isoformat(),
    )
    days.sort(key=lambda d: d["date"])
    if len(days) < CHOP_LOOKBACK_DAYS:
        return False
    window = [d["close"] for d in days[-CHOP_LOOKBACK_DAYS:]]
    net_move = abs(window[-1] - window[0])
    abs_moves = sum(abs(window[k] - window[k - 1]) for k in range(1, len(window)))
    efficiency = net_move / abs_moves if abs_moves > 0 else 0.0
    return efficiency < CHOP_MIN_EFFICIENCY


def check_today() -> dict:
    """One-shot snapshot: today's SuperTrend/VWAP state, trades closed
    today so far, and the current open position (if any). Read-only,
    makes no auth calls."""
    bars = _fetch_futures_bars()
    if len(bars) < ST_PERIOD + 2:
        return {"error": f"not enough futures bars yet ({len(bars)}, need {ST_PERIOD + 2}+) -- "
                          "try again once the market has been open a while, or check back on a trading day"}

    st_dir, _st_line = _compute_supertrend_line(bars, ST_PERIOD, ST_MULTIPLIER)
    ema_filter = _ema([b[4] for b in bars], EMA_FILTER_PERIOD) if EMA_FILTER_PERIOD else None
    chop_today = (
        _is_chop_day() if CHOP_LOOKBACK_DAYS is not None and CHOP_MIN_EFFICIENCY is not None else False
    )

    today_str = date.today().isoformat()
    today_idx = [i for i, b in enumerate(bars) if b[0][:10] == today_str]
    if not today_idx:
        return {"error": f"no futures bars for today ({today_str}) -- market may not be open"}

    chain = _load_nifty_weekly_chain()
    expiry_ms, expiry_date = _nearest_unexpired_expiry(chain)
    if expiry_ms is None:
        return {"error": "no unexpired weekly expiry found in the live instrument master"}

    option_candle_cache: dict[str, list] = {}

    def _option_candles(contract: dict) -> list:
        key = contract["instrument_key"]
        if key not in option_candle_cache:
            option_candle_cache[key] = sorted(
                upstox_client.get_intraday_candles(key, "1minute"), key=lambda c: c[0]
            )
        return option_candle_cache[key]

    def _contract_for(strike: float, opt_type: str) -> dict | None:
        return _contract_for_strike(chain, expiry_ms, strike, opt_type)

    trades: list[dict] = []
    position = None
    cum_pv = cum_vol = 0.0
    prev_close = prev_vwap = None

    for i in today_idx:
        ts, o, h, l, c, v, oi = bars[i]
        time_str = ts[11:16]

        typical = (h + l + c) / 3
        cum_pv += typical * v
        cum_vol += v
        vwap = (cum_pv / cum_vol) if cum_vol > 0 else None

        atm = oc.round_to_step(c, STRIKE_STEP)

        _dh, _dm = divmod(int(time_str[:2]) * 60 + int(time_str[3:5]) + CANDLE_MINUTES, 60)
        decision_time_str = f"{_dh:02d}:{_dm:02d}"

        def _close(reason: str) -> None:
            nonlocal position
            contract = _contract_for(position["strike"], position["opt_type"])
            candles = _option_candles(contract) if contract else []
            bar = _fill(candles, decision_time_str) if candles else None
            exit_price = bar[4] if bar else position["entry_price"]
            exit_time = bar[0] if bar else ts
            trades.append({**position, "exit_time": exit_time, "exit_price": exit_price, "exit_reason": reason})
            position = None

        if time_str >= FORCE_FLAT_TIME:
            if position is not None:
                _close("eod")
            prev_close, prev_vwap = c, vwap
            continue

        if position is not None:
            cur_dir = st_dir[i]
            is_long = position["direction"] == "LONG"
            hit_reversal = cur_dir is not None and ((is_long and cur_dir == -1) or (not is_long and cur_dir == 1))
            hit_sl = hit_target = False
            if SL_PCT is not None or TARGET_PCT is not None:
                contract = _contract_for(position["strike"], position["opt_type"])
                candles = _option_candles(contract) if contract else []
                bar = _fill(candles, decision_time_str) if candles else None
                cur_premium = bar[4] if bar else position["entry_price"]
                if SL_PCT is not None:
                    hit_sl = cur_premium <= position["entry_price"] * (1 - SL_PCT)
                if TARGET_PCT is not None:
                    hit_target = cur_premium >= position["entry_price"] * (1 + TARGET_PCT)
            if hit_sl:
                _close("stop_loss")
            elif hit_target:
                _close("target")
            elif hit_reversal:
                _close("trend_reverse")

        if (position is None and vwap is not None and prev_vwap is not None and prev_close is not None
                and st_dir[i] is not None and not chop_today):
            crossed_above = prev_close <= prev_vwap and c > vwap
            crossed_below = prev_close >= prev_vwap and c < vwap
            direction_label = None
            if st_dir[i] == 1 and crossed_above:
                direction_label = "LONG"
            elif st_dir[i] == -1 and crossed_below:
                direction_label = "SHORT"

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
                contract = _contract_for(atm, opt_type)
                if contract is not None:
                    candles = _option_candles(contract)
                    bar = _fill(candles, decision_time_str) if candles else None
                    if bar is not None:
                        position = {
                            "direction": direction_label, "entry_time": bar[0], "entry_price": bar[4],
                            "strike": contract["strike_price"], "opt_type": opt_type,
                            "trading_symbol": contract["trading_symbol"], "lot_size": contract["lot_size"],
                        }

        prev_close, prev_vwap = c, vwap

    last_i = today_idx[-1]
    return {
        "expiry": expiry_date,
        "last_bar_time": bars[last_i][0],
        "futures_close": bars[last_i][4],
        "supertrend_dir": "bullish" if st_dir[last_i] == 1 else "bearish" if st_dir[last_i] == -1 else "unknown",
        "vwap": vwap,
        "chop_today": chop_today,
        "closed_trades": trades,
        "open_position": position,
    }


def main() -> None:
    r = check_today()
    if "error" in r:
        print(f"Error: {r['error']}")
        return
    print(f"Expiry: {r['expiry']}")
    print(f"Last bar: {r['last_bar_time']}  futures_close={r['futures_close']:.2f}  "
          f"vwap={r['vwap']:.2f}  supertrend={r['supertrend_dir']}")
    if r["chop_today"]:
        print("CHOP FILTER: market regime is choppy -- no new entries will be taken today.")
    if not r["closed_trades"] and not r["open_position"]:
        print("\nNo entries yet today.")
    for t in r["closed_trades"]:
        pnl = (t["exit_price"] - t["entry_price"]) * t["lot_size"]
        print(f"\n[{t['opt_type']}] {t['trading_symbol']} strike={t['strike']}")
        print(f"    {t['exit_reason'].upper()}: entry {t['entry_time']} @ {t['entry_price']:.2f} -> "
              f"exit {t['exit_time']} @ {t['exit_price']:.2f} (pnl=Rs {pnl:+.2f})")
    pos = r["open_position"]
    if pos is not None:
        print(f"\n[{pos['opt_type']}] {pos['trading_symbol']} strike={pos['strike']}")
        print(f"    OPEN: entered {pos['entry_time']} @ {pos['entry_price']:.2f}")


if __name__ == "__main__":
    main()
