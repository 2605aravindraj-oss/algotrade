"""The same EMA(9)/EMA(20) cloud pullback-continuation signal as
backtest.ema_cloud_pullback_options, realized on NIFTY 50 CONSTITUENT
STOCKS' own cash (NSE_EQ) instruments intraday -- no index, no options.

WHY: the options version never survived a full 2-year, cost-inclusive
backtest (net P&L went from a nominally positive gross figure to
clearly negative once the module's own transaction-cost model was
applied, at every slippage level tested). Options add theta/IV decay
on top of the index's own move, which can erase a correct directional
call. Trading the NIFTY 50 stocks' own cash instruments directly
removes that layer entirely -- the position's P&L tracks the
stock's own price move point-for-point, the same signal, decision
logic and _decision_time look-ahead handling as the options module,
just realized on the underlying instead of a derivative.

SIGNAL: identical to ema_cloud_pullback_options -- per stock, EMA(9)
vs EMA(20) on the stock's own 1-minute closes (resampled to
candle_minutes, default 5) sets the trend; a bar whose CLOSE is below
the band during an uptrend "dips" (and the mirror for a downtrend);
the first later bar, still in the same trend, whose close reclaims
the opposite band edge fires the trade. See that module's docstring
for the full history of why this is keyed off closes, not high/low
range overlap.

VOLATILITY FILTER: enabled by default (vol_filter_mult=1.1), same
mechanics as the options module -- skip a stock's trading day if its
own daily true-range EMA(10) has expanded past vol_filter_mult x the
EMA(30), using only data through the PRIOR day. This was the one
filter that helped (not hurt) the options version's full 2-year
backtest; carried over here as the starting default, but it was
tuned on the INDEX's own volatility regime, not any individual
stock's, so treat 1.1 as an untested starting point for stocks and
re-sweep it before trusting it the way the index version's sweep was
validated.

POSITION SIZE: fixed CAPITAL_PER_TRADE (default Rs 50,000) divided by
the entry price, floored to whole shares (minimum 1) -- lets the same
code run unmodified across NIFTY 50 stocks priced anywhere from a few
hundred to several thousand rupees, unlike a fixed share count.

EXIT: percent-of-entry-price stop-loss/target (stop_pct/target_pct,
default 0.4%/0.8% -- a 1:2 risk-reward echoing the options module's
300/600 ratio, but UNTESTED on stocks; this is a starting point, not
a tuned value) or forced-flat at FORCE_FLAT_TIME (15:25). One position
per stock at a time; different stocks can be in a trade simultaneously
(unlike the single NIFTY-index options version, which could only ever
hold one position across the whole index at once).

COSTS: backtest.rsi2_reversion.Trade's cost model (brokerage +
~0.0255% notional round-trip) -- built for intraday equity/futures,
not options' premium-based costs, so this is the right model for
this module (the options module's biggest problem was its own costs
eating a nominally-positive gross P&L; equity intraday costs are a
different, lower order of magnitude, but still real and included).

DATA: a stock's own 1-minute candles come from the regular historical-
candle endpoint (no auth, no expiry-based retention limit) -- unlike
the options module, no access token is needed at all for this one.
"""
from __future__ import annotations

from data_sources import cache, instruments, upstox_client
from backtest.futures_oi_buildup import FORCE_FLAT_TIME, _bar_at_or_after, _bar_at_or_before
from backtest.rsi2_reversion import Trade
from backtest.sweep_reclaim_breakout import _resample
from backtest.technical_rating import _ema
from backtest.technical_rating_rotation import NIFTY_50

EMA_FAST = 9
EMA_SLOW = 20
CAPITAL_PER_TRADE = 50_000.0
STOP_PCT = 0.4
TARGET_PCT = 0.8
VOL_FAST_PERIOD = 10
VOL_SLOW_PERIOD = 30
VOL_FILTER_MULT = 1.1


def run_single(
    from_date: str,
    to_date: str,
    instrument_key: str,
    candle_minutes: int = 5,
    ema_fast: int = EMA_FAST,
    ema_slow: int = EMA_SLOW,
    capital_per_trade: float = CAPITAL_PER_TRADE,
    stop_pct: float = STOP_PCT,
    target_pct: float = TARGET_PCT,
    enable_vol_filter: bool = True,
    vol_fast_period: int = VOL_FAST_PERIOD,
    vol_slow_period: int = VOL_SLOW_PERIOD,
    vol_filter_mult: float = VOL_FILTER_MULT,
) -> list[Trade]:
    """Same signal/entry logic as ema_cloud_pullback_options.run(), on
    one stock's own cash instrument. See the module docstring for the
    full rationale on sizing, exits and the volatility filter."""
    trading_days = upstox_client.get_daily_history(instrument_key, from_date, to_date)
    trading_days.sort(key=lambda d: d["date"])
    if not trading_days:
        return []

    daily_closes = [dd["close"] for dd in trading_days]
    daily_highs = [dd["high"] for dd in trading_days]
    daily_lows = [dd["low"] for dd in trading_days]
    daily_tr = [daily_highs[0] - daily_lows[0]] + [
        max(daily_highs[k] - daily_lows[k], abs(daily_highs[k] - daily_closes[k - 1]), abs(daily_lows[k] - daily_closes[k - 1]))
        for k in range(1, len(trading_days))
    ]
    vol_fast_series = _ema(daily_tr, vol_fast_period)
    vol_slow_series = _ema(daily_tr, vol_slow_period)

    def _fill(candles, at_time_str):
        return _bar_at_or_after(candles, at_time_str) or _bar_at_or_before(candles, at_time_str)

    def _decision_time(ts: str) -> str:
        """Identical rationale to ema_cloud_pullback_options.py's helper:
        _resample labels a multi-minute bar by its bucket START, not its
        close."""
        if candle_minutes <= 1:
            return ts[11:16]
        hh, mm = int(ts[11:13]), int(ts[14:16])
        dh, dm = divmod(hh * 60 + mm + candle_minutes, 60)
        return f"{dh:02d}:{dm:02d}"

    trades: list[Trade] = []
    min_bars = ema_slow + 3

    for day_idx, day in enumerate(trading_days):
        d = day["date"]

        if enable_vol_filter and day_idx >= 1:
            vol_fast_prev = vol_fast_series[day_idx - 1]
            vol_slow_prev = vol_slow_series[day_idx - 1]
            if vol_fast_prev is not None and vol_slow_prev is not None and vol_fast_prev > vol_filter_mult * vol_slow_prev:
                continue  # realized volatility shock -- sit out the whole day

        rows_1min = sorted(cache.get_day_candles_cached(instrument_key, "1minute", d, expired=False), key=lambda r: r[0])
        if len(rows_1min) < min_bars:
            continue
        rows = _resample(rows_1min, candle_minutes) if candle_minutes > 1 else rows_1min
        if len(rows) < min_bars:
            continue
        closes = [r[4] for r in rows]
        ema9 = _ema(closes, ema_fast)
        ema20 = _ema(closes, ema_slow)

        i = ema_slow
        dipped_up = False
        dipped_down = False
        while i < len(rows):
            decision_time_str = _decision_time(rows[i][0])
            if decision_time_str >= FORCE_FLAT_TIME:
                break

            f, s = ema9[i], ema20[i]
            if f is None or s is None:
                i += 1
                continue
            trend_up = f > s
            trend_down = f < s
            band_hi, band_lo = max(f, s), min(f, s)
            close = rows[i][4]

            if not trend_up:
                dipped_up = False
            if not trend_down:
                dipped_down = False

            direction_label = None
            if trend_up:
                if dipped_up and close > band_hi:
                    direction_label = "LONG"
                    dipped_up = False
                elif close < band_lo:
                    dipped_up = True
            elif trend_down:
                if dipped_down and close < band_lo:
                    direction_label = "SHORT"
                    dipped_down = False
                elif close > band_hi:
                    dipped_down = True

            if direction_label is None:
                i += 1
                continue

            entry_idx = i
            entry_price = rows[entry_idx][4]
            entry_ts = rows[entry_idx][0]
            quantity = max(1, int(capital_per_trade // entry_price))
            stop_price = entry_price * (1 - stop_pct / 100) if direction_label == "LONG" else entry_price * (1 + stop_pct / 100)
            target_price = entry_price * (1 + target_pct / 100) if direction_label == "LONG" else entry_price * (1 - target_pct / 100)

            exit_idx = None
            exit_reason = None
            for j in range(entry_idx + 1, len(rows)):
                j_decision_time = _decision_time(rows[j][0])
                if j_decision_time >= FORCE_FLAT_TIME:
                    exit_idx = j
                    exit_reason = "eod"
                    break
                c_j = rows[j][4]
                if direction_label == "LONG" and c_j <= stop_price:
                    exit_idx = j
                    exit_reason = "stop_loss"
                    break
                if direction_label == "SHORT" and c_j >= stop_price:
                    exit_idx = j
                    exit_reason = "stop_loss"
                    break
                if direction_label == "LONG" and c_j >= target_price:
                    exit_idx = j
                    exit_reason = "target"
                    break
                if direction_label == "SHORT" and c_j <= target_price:
                    exit_idx = j
                    exit_reason = "target"
                    break

            if exit_idx is None:
                exit_idx = len(rows) - 1
                exit_reason = "eod"

            trades.append(Trade(
                date=d, direction=direction_label, entry_time=entry_ts, entry_price=entry_price,
                exit_time=rows[exit_idx][0], exit_price=rows[exit_idx][4], exit_reason=exit_reason,
                lot_size=quantity,
            ))

            i = exit_idx + 1

    return trades


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
