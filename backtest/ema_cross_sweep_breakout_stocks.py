"""The same EMA(9)/EMA(20) cross -> sweep -> breakout signal as
backtest.ema_cross_sweep_breakout_options, realized on NIFTY 50
CONSTITUENT STOCKS' own cash (NSE_EQ) instruments intraday -- no
index, no options.

WHY: every exit variant tried on the NIFTY index options version
(fixed 1:risk_reward index-level target, fixed-rupee premium
stop/target, a trend filter, a trailing "ride the trend" stop, a
coarser 1hr/5min timeframe) either lost money net of costs over the
full ~2-year history, or looked positive only because of 1-2 outlier
trades (removing the top 2 trades flipped the 1hr/5min variant's
whole 2-year net from +21.8k to -36). Trading NIFTY 50 stocks' own
cash instruments directly removes the options layer (theta/IV decay
on top of the index's own move) and spreads the signal across 50
independent names instead of betting the whole book on one index's
path -- the same open question ema_cloud_pullback_stocks.py asked of
a different signal.

ZONE/ENTRY: identical mechanics to the options module, per stock --
EMA(9)/EMA(20) cross on the stock's own `zone_minutes` candles (default
5) marks that crossover candle's [low, high] as the zone; once it's
visible (real close instant, no lookahead) the `entry_minutes` series
(default 1) waits for a sweep of the zone's near edge followed by a
breakout close through the far edge. EMA/zone tracking runs
CONTINUOUSLY across the whole backtest date range per stock (not
reseeded every morning), same reasoning as every other module in this
family -- only the POSITION itself force-flattens daily.

EXIT: stop/target are the stock's own PRICE levels (not a % of entry,
not an options premium) -- same "chart level" convention as the index
options version. stop_at_midpoint (default True): stop sits at the
zone's own midpoint; target = entry_close + risk_reward * (entry_close
- stop_level) for a LONG, mirrored for a SHORT (default risk_reward
2.0). No options translation needed -- the stock's own low/high is
checked directly against stop_level/target_level each entry-timeframe
bar. Forced flat at FORCE_FLAT_TIME (15:25) if neither level is hit.
One position per stock at a time; different stocks can be in a trade
simultaneously (unlike the single-index options version).

POSITION SIZE: fixed CAPITAL_PER_TRADE (default Rs 50,000) divided by
entry price, floored to whole shares (minimum 1) -- same convention as
ema_cloud_pullback_stocks.py, so the same code runs across stocks
priced from a few hundred to several thousand rupees.

COSTS: backtest.rsi2_reversion.Trade's cost model (brokerage +
~0.0255% notional round-trip) -- the intraday equity cost model, not
the options module's premium-based one.

DATA: a stock's own 1-minute candles come from the regular historical-
candle endpoint (no auth needed).
"""
from __future__ import annotations

from datetime import datetime, timedelta

from data_sources import cache, instruments, upstox_client
from backtest.ema_cross_sweep_breakout_options import Bar, _find_zones
from backtest.futures_oi_buildup import FORCE_FLAT_TIME
from backtest.rsi2_reversion import Trade
from backtest.sweep_reclaim_breakout import _resample
from backtest.technical_rating_rotation import NIFTY_50

ZONE_TIMEFRAME_MINUTES = 5
ENTRY_TIMEFRAME_MINUTES = 1
EMA_FAST = 9
EMA_SLOW = 20
RISK_REWARD = 2.0
CAPITAL_PER_TRADE = 50_000.0
SWEEP_TOLERANCE_PCT = 0.05


def _drop_zero_volume(rows_1min: list[list]) -> list[list]:
    """NSE equities go quiet for ~14 minutes most days (around 15:15-15:28)
    -- a stretch of zero-volume bars that just repeat the last traded
    price -- then print ONE large-volume bar for the closing auction
    match, often at a different price than the last continuous trade
    (seen on COALINDIA: flat at 410.35 with volume=0 from 15:15-15:28,
    then a single O=H=L=C=411.55 bar at 15:29 with volume=748104). That
    auction print is a real trade, but a one-off settlement match, not
    continuous intraday trading -- left in, it can satisfy a sweep/
    breakout check (or get reported as "last price") on a level that was
    never actually traded through during the session. So this drops not
    just the zero-volume bars but everything in a day from the START of
    its first zero-volume stretch onward, auction print included --
    continuous trading for that day is treated as ending there."""
    by_day: dict[str, list] = {}
    order: list[str] = []
    for r in rows_1min:
        day = r[0][:10]
        if day not in by_day:
            by_day[day] = []
            order.append(day)
        by_day[day].append(r)

    out: list[list] = []
    for day in order:
        day_rows = by_day[day]
        first_zero = next((i for i, r in enumerate(day_rows) if r[5] == 0), None)
        out.extend(day_rows if first_zero is None else day_rows[:first_zero])
    return out


def run_single(
    from_date: str,
    to_date: str,
    instrument_key: str,
    zone_minutes: int = ZONE_TIMEFRAME_MINUTES,
    entry_minutes: int = ENTRY_TIMEFRAME_MINUTES,
    ema_fast: int = EMA_FAST,
    ema_slow: int = EMA_SLOW,
    risk_reward: float = RISK_REWARD,
    stop_at_midpoint: bool = True,
    capital_per_trade: float = CAPITAL_PER_TRADE,
    sweep_tolerance_pct: float = SWEEP_TOLERANCE_PCT,
) -> list[Trade]:
    """Same zone/sweep/breakout logic as ema_cross_sweep_breakout_options.run(),
    on one stock's own cash instrument -- no options translation.

    sweep_tolerance_pct (default 0.05%): the sweep check accepts a high/low
    that gets within this percentage of the zone edge, not just an exact
    touch -- a 1-tick near-miss during real continuous trading (seen on
    ONGC: high reached 221.25 against a 221.26 zone_high) shouldn't silently
    strand the setup. The breakout close check stays exact."""
    trading_days = upstox_client.get_daily_history(instrument_key, from_date, to_date)
    trading_days.sort(key=lambda d: d["date"])
    if not trading_days:
        return []

    def _decision_time(ts: str, minutes: int) -> str:
        if minutes <= 1:
            return ts[11:16]
        hh, mm = int(ts[11:13]), int(ts[14:16])
        dh, dm = divmod(hh * 60 + mm + minutes, 60)
        return f"{dh:02d}:{dm:02d}"

    all_1min: list[list] = []
    for day in trading_days:
        rows = sorted(cache.get_day_candles_cached(instrument_key, "1minute", day["date"], expired=False), key=lambda r: r[0])
        all_1min.extend(rows)
    all_1min.sort(key=lambda r: r[0])
    all_1min = _drop_zero_volume(all_1min)

    min_bars = ema_slow + 3
    if len(all_1min) < min_bars:
        return []

    zone_rows = _resample(all_1min, zone_minutes) if zone_minutes > 1 else all_1min
    if len(zone_rows) < min_bars:
        return []
    zone_bars = [Bar(ts=r[0], o=r[1], h=r[2], l=r[3], c=r[4]) for r in zone_rows]

    entry_rows = _resample(all_1min, entry_minutes) if entry_minutes > 1 else all_1min
    bars = [Bar(ts=r[0], o=r[1], h=r[2], l=r[3], c=r[4]) for r in entry_rows]

    zones = _find_zones(zone_bars, ema_fast, ema_slow)
    zones_available = sorted(
        ((datetime.fromisoformat(zone_bars[idx].ts) + timedelta(minutes=zone_minutes), direction, zlo, zhi)
         for idx, direction, zlo, zhi in zones),
        key=lambda x: x[0],
    )

    trades: list[Trade] = []
    zone_cursor = 0
    pending: dict | None = None

    i = 0
    while i < len(bars):
        bar_date = bars[i].ts[:10]
        bar_dt = datetime.fromisoformat(bars[i].ts)
        decision_time_str = _decision_time(bars[i].ts, entry_minutes)
        skip_entries_today = decision_time_str >= FORCE_FLAT_TIME

        while zone_cursor < len(zones_available) and zones_available[zone_cursor][0] <= bar_dt:
            _, direction, zlo, zhi = zones_available[zone_cursor]
            pending = {"direction": direction, "zone_low": zlo, "zone_high": zhi, "swept": False}
            zone_cursor += 1

        direction_label = stop_level = target_level = None
        if pending is not None:
            lo, hi, close = bars[i].l, bars[i].h, bars[i].c
            if pending["direction"] == "LONG":
                sweep_level = pending["zone_low"] * (1 + sweep_tolerance_pct / 100)
                if not pending["swept"] and lo <= sweep_level:
                    pending["swept"] = True
                if pending["swept"] and close > pending["zone_high"] and not skip_entries_today:
                    direction_label = "LONG"
                    zone_mid = (pending["zone_low"] + pending["zone_high"]) / 2
                    stop_level = zone_mid if stop_at_midpoint else pending["zone_low"]
                    target_level = close + risk_reward * (close - stop_level)
                    pending = None
            else:
                sweep_level = pending["zone_high"] * (1 - sweep_tolerance_pct / 100)
                if not pending["swept"] and hi >= sweep_level:
                    pending["swept"] = True
                if pending["swept"] and close < pending["zone_low"] and not skip_entries_today:
                    direction_label = "SHORT"
                    zone_mid = (pending["zone_low"] + pending["zone_high"]) / 2
                    stop_level = zone_mid if stop_at_midpoint else pending["zone_high"]
                    target_level = close - risk_reward * (stop_level - close)
                    pending = None

        if direction_label is None:
            i += 1
            continue

        d = bar_date
        entry_idx = i
        entry_price = bars[entry_idx].c
        entry_ts = bars[entry_idx].ts
        quantity = max(1, int(capital_per_trade // entry_price))

        day_end_idx = entry_idx
        while day_end_idx + 1 < len(bars) and bars[day_end_idx + 1].ts[:10] == d:
            day_end_idx += 1

        exit_idx = None
        exit_reason = None
        for j in range(entry_idx + 1, day_end_idx + 1):
            j_decision_time = _decision_time(bars[j].ts, entry_minutes)
            if j_decision_time >= FORCE_FLAT_TIME:
                exit_idx = j
                exit_reason = "eod"
                break
            if direction_label == "LONG":
                if bars[j].l <= stop_level:
                    exit_idx, exit_reason = j, "stop_loss"
                    break
                if bars[j].h >= target_level:
                    exit_idx, exit_reason = j, "target"
                    break
            else:
                if bars[j].h >= stop_level:
                    exit_idx, exit_reason = j, "stop_loss"
                    break
                if bars[j].l <= target_level:
                    exit_idx, exit_reason = j, "target"
                    break

        if exit_idx is None:
            exit_idx = day_end_idx
            exit_reason = "eod"

        trades.append(Trade(
            date=d, direction=direction_label, entry_time=entry_ts, entry_price=entry_price,
            exit_time=bars[exit_idx].ts, exit_price=bars[exit_idx].c, exit_reason=exit_reason,
            lot_size=quantity,
        ))

        i = exit_idx + 1

    return trades


def screen_single(
    instrument_key: str,
    zone_minutes: int = ZONE_TIMEFRAME_MINUTES,
    entry_minutes: int = ENTRY_TIMEFRAME_MINUTES,
    ema_fast: int = EMA_FAST,
    ema_slow: int = EMA_SLOW,
    lookback_days: int = 15,
    sweep_tolerance_pct: float = SWEEP_TOLERANCE_PCT,
) -> dict | None:
    """Live setup state for one stock -- NOT a backtest. Replays the same
    zone -> sweep -> breakout state machine as run_single() over the most
    recent `lookback_days` trading days, but reports where things stand
    RIGHT NOW instead of simulating trades. See run_single()'s docstring
    for sweep_tolerance_pct and the zero-volume filter (_drop_zero_volume):

    - "zone_marked": the latest EMA9/EMA20 cross is still unswept -- watch
      for price to sweep the near edge.
    - "swept_pending_breakout": that zone has been swept -- watch for a
      close through the far edge (the breakout trigger).
    - "breakout_fired": the breakout already happened and no new cross has
      superseded it -- shows how long ago.
    - "no_recent_zone": no EMA cross at all in the lookback window.

    Returns None if there isn't enough recent data to compute EMA(ema_slow)
    on the zone timeframe. Every non-None result also carries
    "chart_bars": the last 40 zone-timeframe OHLC bars as [ts,o,h,l,c], for
    drawing a small candlestick chart of the setup without a second fetch.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    far_back = (datetime.now() - timedelta(days=lookback_days * 3 + 10)).strftime("%Y-%m-%d")
    trading_days = upstox_client.get_daily_history(instrument_key, far_back, today)
    trading_days.sort(key=lambda d: d["date"])
    if not trading_days:
        return None
    trading_days = trading_days[-lookback_days:]

    all_1min: list[list] = []
    for day in trading_days:
        rows = sorted(cache.get_day_candles_cached(instrument_key, "1minute", day["date"], expired=False), key=lambda r: r[0])
        all_1min.extend(rows)
    all_1min.sort(key=lambda r: r[0])
    all_1min = _drop_zero_volume(all_1min)

    min_bars = ema_slow + 3
    if len(all_1min) < min_bars:
        return None

    zone_rows = _resample(all_1min, zone_minutes) if zone_minutes > 1 else all_1min
    if len(zone_rows) < min_bars:
        return None
    zone_bars = [Bar(ts=r[0], o=r[1], h=r[2], l=r[3], c=r[4]) for r in zone_rows]

    entry_rows = _resample(all_1min, entry_minutes) if entry_minutes > 1 else all_1min
    bars = [Bar(ts=r[0], o=r[1], h=r[2], l=r[3], c=r[4]) for r in entry_rows]

    zones = _find_zones(zone_bars, ema_fast, ema_slow)
    zones_available = sorted(
        ((datetime.fromisoformat(zone_bars[idx].ts) + timedelta(minutes=zone_minutes), direction, zlo, zhi)
         for idx, direction, zlo, zhi in zones),
        key=lambda x: x[0],
    )
    chart_bars = [[b.ts, b.o, b.h, b.l, b.c] for b in zone_bars[-40:]]
    if not zones_available:
        return {"status": "no_recent_zone", "last_price": bars[-1].c, "last_ts": bars[-1].ts, "chart_bars": chart_bars}

    pending: dict | None = None
    last_fired: dict | None = None
    zone_cursor = 0
    for bar in bars:
        bar_dt = datetime.fromisoformat(bar.ts)
        while zone_cursor < len(zones_available) and zones_available[zone_cursor][0] <= bar_dt:
            zone_ts, direction, zlo, zhi = zones_available[zone_cursor]
            pending = {"direction": direction, "zone_low": zlo, "zone_high": zhi, "swept": False,
                       "zone_ts": zone_ts.isoformat(), "swept_ts": None}
            zone_cursor += 1

        if pending is not None:
            lo, hi, close = bar.l, bar.h, bar.c
            if pending["direction"] == "LONG":
                sweep_level = pending["zone_low"] * (1 + sweep_tolerance_pct / 100)
                if not pending["swept"] and lo <= sweep_level:
                    pending["swept"] = True
                    pending["swept_ts"] = bar.ts
                if pending["swept"] and close > pending["zone_high"]:
                    last_fired = {**pending, "fired_ts": bar.ts, "fired_price": close}
                    pending = None
            else:
                sweep_level = pending["zone_high"] * (1 - sweep_tolerance_pct / 100)
                if not pending["swept"] and hi >= sweep_level:
                    pending["swept"] = True
                    pending["swept_ts"] = bar.ts
                if pending["swept"] and close < pending["zone_low"]:
                    last_fired = {**pending, "fired_ts": bar.ts, "fired_price": close}
                    pending = None

    last_bar = bars[-1]
    last_price, last_ts = last_bar.c, last_bar.ts

    if pending is not None:
        zone_low, zone_high, direction = pending["zone_low"], pending["zone_high"], pending["direction"]
        if not pending["swept"]:
            status = "zone_marked"
            trigger_level = zone_low if direction == "LONG" else zone_high
            distance_pct = 100 * (last_price - trigger_level) / last_price
        else:
            status = "swept_pending_breakout"
            trigger_level = zone_high if direction == "LONG" else zone_low
            distance_pct = 100 * (trigger_level - last_price) / last_price if direction == "LONG" \
                else 100 * (last_price - trigger_level) / last_price
        return {
            "status": status, "direction": direction, "zone_low": zone_low, "zone_high": zone_high,
            "zone_ts": pending["zone_ts"], "swept_ts": pending["swept_ts"], "trigger_level": trigger_level,
            "distance_pct": distance_pct, "last_price": last_price, "last_ts": last_ts, "chart_bars": chart_bars,
        }

    if last_fired is not None:
        return {
            "status": "breakout_fired", "direction": last_fired["direction"],
            "zone_low": last_fired["zone_low"], "zone_high": last_fired["zone_high"],
            "zone_ts": last_fired["zone_ts"], "swept_ts": last_fired["swept_ts"],
            "fired_ts": last_fired["fired_ts"], "fired_price": last_fired["fired_price"],
            "last_price": last_price, "last_ts": last_ts, "chart_bars": chart_bars,
        }

    return {"status": "no_recent_zone", "last_price": last_price, "last_ts": last_ts, "chart_bars": chart_bars}


def screen(
    symbols: list[str] = NIFTY_50,
    **kwargs,
) -> list[dict]:
    """screen_single() across every (resolvable) symbol in `symbols`.
    Returns a list of {symbol, **screen_single() result}, skipping symbols
    that can't be resolved or don't have enough recent data."""
    keys = instruments.resolve_symbols(symbols)
    results = []
    for symbol in symbols:
        key = keys.get(symbol)
        if key is None:
            continue
        try:
            state = screen_single(key, **kwargs)
        except Exception:
            continue
        if state is not None:
            results.append({"symbol": symbol, **state})
    return results


def run_screener(
    from_date: str,
    to_date: str,
    symbols: list[str] = NIFTY_50,
    **kwargs,
) -> dict[str, list[Trade]]:
    """run_single() across every (resolvable) symbol in `symbols`.
    Returns {symbol: trades}; a symbol whose instrument key can't be
    resolved, or whose fetch raises, is simply left out."""
    keys = instruments.resolve_symbols(symbols)
    results: dict[str, list[Trade]] = {}
    for symbol in symbols:
        key = keys.get(symbol)
        if key is None:
            continue
        try:
            trades = run_single(from_date, to_date, key, **kwargs)
        except Exception:
            continue
        if trades:
            results[symbol] = trades
    return results


def summary(results: dict[str, list[Trade]]) -> str:
    from backtest.rsi2_reversion import summary as _summary
    all_trades = [t for trades in results.values() for t in trades]
    if not all_trades:
        return "No trades."
    lines = [_summary(all_trades), "", f"Stocks traded: {len(results)}", ""]
    per_stock = sorted(
        ((sym, sum(t.pnl_rupees for t in trades), len(trades)) for sym, trades in results.items()),
        key=lambda x: -x[1],
    )
    for sym, net, n in per_stock:
        lines.append(f"  {sym:<12} trades={n:<4} net=Rs {net:,.2f}")
    return "\n".join(lines)
