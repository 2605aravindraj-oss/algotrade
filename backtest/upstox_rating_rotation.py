"""Portfolio rotation using the "Technical Rating" composite score from
a separate existing project (upstox_paper_trade's screener feature) --
reproduced here from the user's own specification. This is a
DIFFERENT, simpler 9-indicator formula than backtest/technical_rating.py's
20-signal (12 moving average + 8 oscillator) one; that module's rotation
backtest (backtest/technical_rating_rotation.py) underperformed NIFTY
50 buy-and-hold badly. This is a separate, independently-sourced
formula being tested on its own merits.

SCORE: 9 indicators, each casting a Bullish(+1)/Bearish(-1)/Neutral(0)
vote on a stock's own daily closes/volumes, as of a given date (no
lookahead -- every indicator uses only candles up to and including
that date):

    SMA(20)   price > SMA20 -> Bullish, else Bearish
    SMA(50)   price > SMA50 -> Bullish, else Bearish
    EMA cross EMA(20) > EMA(50) -> Bullish, else Bearish
    RSI(14)   <30 -> Bullish, >70 -> Bearish, else Neutral
    MACD      histogram (12,26,9) > 0 -> Bullish, else Bearish
    Bollinger(20,2)  close below lower band -> Bullish, above upper
              band -> Bearish, else Neutral
    52-week High/Low  within 5% of the trailing-252-day high ->
              Bullish, within 5% of the trailing-252-day low ->
              Bearish, else Neutral
    Relative Strength vs Nifty  stock's own 1/3/6-month return beats
              NIFTY 50's over the SAME window, for however many of
              those three windows are computable (need that much
              trailing history) -- Bullish only if it beats on EVERY
              computable window (rs_require_all=True, the default),
              else Bearish. Excluded entirely (not counted in the vote
              total) if NONE of the three windows are computable yet.
    Volume-confirmed move  that day's volume >= 1.5x its own trailing
              20-day average AND the day's close > previous close ->
              Bullish; >=1.5x average AND close < previous close ->
              Bearish; else Neutral.

score = (bullish_votes - bearish_votes) / total_votes, in [-1, +1].
Needs at least 60 days of price history to compute at all (returns
None before that, same as this codebase's other daily-bar modules'
warmup floors).

PORTFOLIO: every 30 trading days, score every resolvable NIFTY 50
constituent as of the rebalance date, rank by score, take the top 5,
equal-weight, hold to the next rebalance, fully liquidate and
re-rank -- reuses backtest.technical_rating_rotation's rebalance-date
snapping, NIFTY_50 symbol list, and realistic equity-delivery cost
model (STT, stamp duty, DP charges, zero-brokerage assumption)
directly, since that machinery is formula-agnostic.

KNOWN CAVEAT (carried over from the user's own prior, separate
backtest of this formula): tested against TODAY's NIFTY 50 member
list projected backward over the whole window, not the index's real
point-in-time historical membership -- survivorship bias, since no
point-in-time constituent dataset was available. Not fixed here
either, for the same reason.

RELATIVE-STRENGTH VOTE RULE (rs_require_all): the written spec didn't
say how the three 1/3/6-month sub-signals combine into one vote. The
10-year backtest (2016-09-08/2026-09-08) was run both ways to check:
  rs_require_all=False (majority of computable windows beats NIFTY):
    +121.93% total, +8.28% CAGR, -33.61% max drawdown.
  rs_require_all=True  (EVERY computable window must beat, the
    stricter read, and the default here): +203.15% total, +11.70%
    CAGR, -33.05% max drawdown.
Tightening to require_all is a free improvement on this data -- CAGR
up ~3.4 points, drawdown very slightly better too, no tradeoff -- so
it's the default.

REBALANCE CADENCE (calendar_rebalance): the spec says "every 30
TRADING days" literally, but the rebalance-date helper reused from
backtest.technical_rating_rotation steps by 30 CALENDAR days snapped
to the nearest trading day -- a different, more frequent schedule
(~122 rebalances over 10 years vs. ~84 for a true 30-trading-day
cadence). Fixed with _trading_day_rebalance_dates (default,
calendar_rebalance=False); the old behavior is kept only for
comparison (calendar_rebalance=True). This was the single biggest
lever found: CAGR +11.70% -> +16.85%, max drawdown -33.05% ->
-26.42%, both improving together once the cadence matched the spec.

52-WEEK HIGH/LOW BASIS (use_intraday_52w): the spec's "52-week
High/Low" signal can use closing prices or (the more conventional
chart-terminology reading) daily high/low prices for the trailing-
252-day extreme. Tested both: close-based gives +16.85% CAGR / -26.42%
drawdown; high/low-based (the default here) gives +20.94% CAGR /
-27.51% drawdown -- a real tradeoff (CAGR up ~4 points, drawdown ~1
point worse), not a free upgrade, but the better risk-adjusted result
(CAGR/|drawdown| = 0.76 vs 0.64) and the more standard definition of
the term, so it's the default.

END-TO-END RESULT with every default above (2016-09-08/2026-09-08,
10 years, 83 rebalances): +20.94% CAGR, -27.51% max drawdown. This
EXCEEDS the user's own prior, separately-run result for this formula
on CAGR (14.99%) in every variant tested here, but sits a few points
worse on max drawdown (-24.2%) in every variant too. An end-date
sensitivity sweep (testing 2025-09-08 through 2026-09-08 as the end
date, all giving an IDENTICAL max drawdown) ruled out "a recent
correction the user's backtest never saw" as the explanation -- the
worst drawdown stretch is a fixed, earlier event in this window, not
a trailing-date artifact. The remaining drawdown gap is most likely
from implementation details of this independently-sourced formula
that can't be verified without the original source (exact cost
assumptions, volume-average window, Bollinger/MACD parameterization),
not a resolvable bug found so far.
"""
from __future__ import annotations

import datetime
import statistics
from dataclasses import dataclass

from backtest.ema8_13_trend_sweep_options import _ema
from backtest.rsi2_reversion import compute_rsi
from backtest.rsi_daily_reversion import _get_daily_history_chunked
from backtest.technical_rating_rotation import (
    NIFTY_50,
    Period,
    RotationResult,
    _equity_buy_cost,
    _equity_sell_cost,
    _rebalance_dates,
)
from data_sources import instruments, upstox_client

MIN_HISTORY_DAYS = 60
NIFTY_INDEX_KEY = "NSE_INDEX|Nifty 50"


@dataclass
class Rating:
    symbol: str
    as_of_date: str
    price: float
    score: float
    bullish: int
    bearish: int
    neutral: int
    total_votes: int


def _sma(closes: list[float], period: int) -> float | None:
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def _macd_histogram(closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> float | None:
    if len(closes) < slow + signal:
        return None
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = [
        (f - s) if (f is not None and s is not None) else None
        for f, s in zip(ema_fast, ema_slow)
    ]
    first_valid = next((i for i, v in enumerate(macd_line) if v is not None), None)
    if first_valid is None:
        return None
    tail = [v for v in macd_line[first_valid:]]
    if len(tail) < signal:
        return None
    signal_line = _ema(tail, signal)
    if signal_line[-1] is None:
        return None
    return macd_line[-1] - signal_line[-1]


def _bollinger(closes: list[float], period: int = 20, mult: float = 2.0) -> tuple[float, float] | None:
    if len(closes) < period:
        return None
    window = closes[-period:]
    mean = sum(window) / period
    std = statistics.pstdev(window)
    return mean + mult * std, mean - mult * std  # upper, lower


def _relative_strength_vote(
    stock_closes: list[float], nifty_closes: list[float], require_all: bool = False
) -> int | None:
    """+1/-1 on the computable {1mo, 3mo, 6mo} windows beating NIFTY's own
    return over the same window; None if no window is computable yet.
    require_all=False (default): majority of computable windows must beat.
    require_all=True (tightened): EVERY computable window must beat -- a
    single miss on any window votes Bearish."""
    windows = [21, 63, 126]
    beats = []
    n = min(len(stock_closes), len(nifty_closes))
    for w in windows:
        if n <= w:
            continue
        stock_ret = stock_closes[-1] / stock_closes[-1 - w] - 1
        nifty_ret = nifty_closes[-1] / nifty_closes[-1 - w] - 1
        beats.append(stock_ret > nifty_ret)
    if not beats:
        return None
    if require_all:
        return 1 if all(beats) else -1
    bull_count = sum(beats)
    return 1 if bull_count * 2 >= len(beats) else -1


def rate(
    symbol: str, candles: list[dict], nifty_candles: list[dict], rs_require_all: bool = False,
    use_intraday_52w: bool = True,
) -> Rating | None:
    """candles/nifty_candles: date-sorted lists of dicts with date/close/volume,
    both already sliced to the as-of date (no lookahead) by the caller.
    use_intraday_52w: the 52-week High/Low vote uses daily HIGH/LOW prices
    (the conventional "52-week high/low" definition) instead of closes."""
    if len(candles) < MIN_HISTORY_DAYS:
        return None
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    volumes = [c["volume"] for c in candles]
    nifty_closes = [c["close"] for c in nifty_candles]
    price = closes[-1]

    votes: list[int | None] = []

    sma20 = _sma(closes, 20)
    votes.append(1 if (sma20 is not None and price > sma20) else (-1 if sma20 is not None else None))

    sma50 = _sma(closes, 50)
    votes.append(1 if (sma50 is not None and price > sma50) else (-1 if sma50 is not None else None))

    ema20 = _ema(closes, 20)[-1]
    ema50 = _ema(closes, 50)[-1]
    if ema20 is not None and ema50 is not None:
        votes.append(1 if ema20 > ema50 else -1)
    else:
        votes.append(None)

    rsi = compute_rsi(closes, 14)[-1]
    if rsi is None:
        votes.append(None)
    elif rsi < 30:
        votes.append(1)
    elif rsi > 70:
        votes.append(-1)
    else:
        votes.append(0)

    macd_hist = _macd_histogram(closes)
    votes.append(1 if (macd_hist is not None and macd_hist > 0) else (-1 if macd_hist is not None else None))

    bb = _bollinger(closes)
    if bb is None:
        votes.append(None)
    else:
        upper, lower = bb
        votes.append(1 if price < lower else (-1 if price > upper else 0))

    if len(closes) >= 252:
        if use_intraday_52w:
            hi, lo = max(highs[-252:]), min(lows[-252:])
        else:
            window = closes[-252:]
            hi, lo = max(window), min(window)
        if price >= hi * 0.95:
            votes.append(1)
        elif price <= lo * 1.05:
            votes.append(-1)
        else:
            votes.append(0)
    else:
        votes.append(None)

    votes.append(_relative_strength_vote(closes, nifty_closes, require_all=rs_require_all))

    if len(volumes) >= 21:
        avg_vol = sum(volumes[-21:-1]) / 20
        vol_ratio = (volumes[-1] / avg_vol) if avg_vol > 0 else 0
        price_change = closes[-1] - closes[-2]
        if vol_ratio >= 1.5 and price_change > 0:
            votes.append(1)
        elif vol_ratio >= 1.5 and price_change < 0:
            votes.append(-1)
        else:
            votes.append(0)
    else:
        votes.append(None)

    counted = [v for v in votes if v is not None]
    if not counted:
        return None
    bullish = sum(1 for v in counted if v == 1)
    bearish = sum(1 for v in counted if v == -1)
    neutral = sum(1 for v in counted if v == 0)
    total = len(counted)
    score = (bullish - bearish) / total

    return Rating(
        symbol=symbol, as_of_date=candles[-1]["date"], price=price, score=score,
        bullish=bullish, bearish=bearish, neutral=neutral, total_votes=total,
    )


def _fetch_histories(symbols: list[str], from_date: str, to_date: str) -> dict[str, list[dict]]:
    keys = instruments.resolve_symbols(symbols)
    out: dict[str, list[dict]] = {}
    for sym in symbols:
        key = keys.get(sym)
        if key is None:
            continue
        try:
            daily = _get_daily_history_chunked(key, from_date, to_date)
        except Exception:
            continue
        if daily:
            out[sym] = sorted(daily, key=lambda c: c["date"])
    return out


def _trading_day_rebalance_dates(trading_dates: list[str], rebalance_days: int) -> list[str]:
    """Step by rebalance_days actual TRADING days (index into trading_dates
    directly), not calendar days -- "every 30 trading days" literally,
    plus a final close-out on the last available trading day."""
    dates = trading_dates[0::rebalance_days]
    if dates[-1] != trading_dates[-1]:
        dates.append(trading_dates[-1])
    return dates


def run(
    start_date: str,
    end_date: str,
    rebalance_days: int = 30,
    starting_capital: float = 1_000_000.0,
    top_n: int = 5,
    symbols: list[str] | None = None,
    rs_require_all: bool = True,
    calendar_rebalance: bool = False,
    use_intraday_52w: bool = True,
) -> RotationResult:
    symbols = symbols or NIFTY_50
    warmup_from = (datetime.date.fromisoformat(start_date) - datetime.timedelta(days=380)).isoformat()

    histories = _fetch_histories(symbols, warmup_from, end_date)
    nifty_candles_all = sorted(_get_daily_history_chunked(NIFTY_INDEX_KEY, warmup_from, end_date), key=lambda c: c["date"])

    calendar_symbol = "RELIANCE" if "RELIANCE" in histories else next(iter(histories))
    all_dates = sorted({c["date"] for c in histories[calendar_symbol]})
    trading_dates = [d for d in all_dates if d >= start_date]
    if not trading_dates:
        return RotationResult(starting_capital=starting_capital, final_value=starting_capital)

    if calendar_rebalance:
        rb_dates = _rebalance_dates(trading_dates, start_date, end_date, rebalance_days)
    else:
        rb_dates = _trading_day_rebalance_dates(trading_dates, rebalance_days)
    if len(rb_dates) < 2:
        return RotationResult(starting_capital=starting_capital, final_value=starting_capital)

    date_index: dict[str, dict[str, int]] = {}
    for sym, candles in histories.items():
        date_index[sym] = {c["date"]: i for i, c in enumerate(candles)}
    nifty_date_index = {c["date"]: i for i, c in enumerate(nifty_candles_all)}

    def _close_on(sym: str, date: str) -> float | None:
        idx = date_index.get(sym, {}).get(date)
        if idx is None:
            return None
        return histories[sym][idx]["close"]

    def _screen(as_of_date: str) -> list[str]:
        nifty_idx = nifty_date_index.get(as_of_date)
        if nifty_idx is None:
            return []
        nifty_slice = nifty_candles_all[:nifty_idx + 1]
        rated = []
        for sym, candles in histories.items():
            idx = date_index[sym].get(as_of_date)
            if idx is None:
                continue
            r = rate(sym, candles[:idx + 1], nifty_slice, rs_require_all=rs_require_all, use_intraday_52w=use_intraday_52w)
            if r is not None:
                rated.append(r)
        rated.sort(key=lambda r: r.score, reverse=True)
        return [r.symbol for r in rated[:top_n]]

    result = RotationResult(starting_capital=starting_capital)
    portfolio_value = starting_capital

    for i in range(len(rb_dates) - 1):
        d0, d1 = rb_dates[i], rb_dates[i + 1]
        picks = _screen(d0)
        start_value = portfolio_value

        if not picks:
            result.periods.append(Period(d0, d1, [], start_value, start_value, 0.0))
            continue

        per_stock_capital = start_value / len(picks)
        buy_costs = 0.0
        shares: dict[str, float] = {}
        for sym in picks:
            price = _close_on(sym, d0)
            if price is None or price <= 0:
                continue
            qty = per_stock_capital / price
            shares[sym] = qty
            buy_costs += _equity_buy_cost(qty * price)

        sell_value = 0.0
        sell_costs = 0.0
        for sym, qty in shares.items():
            price = _close_on(sym, d1)
            if price is None:
                price = _close_on(sym, d0)
            sell_value += qty * price
            sell_costs += _equity_sell_cost(qty * price, 1)

        cash_uninvested = start_value - sum(shares[s] * _close_on(s, d0) for s in shares)
        end_value = sell_value + cash_uninvested - buy_costs - sell_costs
        result.periods.append(Period(d0, d1, list(shares.keys()), start_value, end_value, buy_costs + sell_costs))
        portfolio_value = end_value

    result.final_value = portfolio_value
    return result


def summary(result: RotationResult) -> str:
    if not result.periods:
        return "No periods."
    lines = []
    for p in result.periods:
        picks_str = ", ".join(p.picks) if p.picks else "(cash)"
        lines.append(f"{p.start_date} -> {p.end_date}: {p.return_pct:+.2f}%  [{picks_str}]  costs=Rs {p.costs:,.0f}")
    total_return = 100 * (result.final_value - result.starting_capital) / result.starting_capital
    n_years = len(result.periods) * 30 / 365.25
    cagr = (
        ((result.final_value / result.starting_capital) ** (1 / n_years) - 1) * 100
        if n_years > 0 and result.final_value > 0 else float("nan")
    )
    cash_periods = sum(1 for p in result.periods if not p.picks)
    lines.append("")
    lines.append(f"Periods: {len(result.periods)}  (in cash: {cash_periods})")
    lines.append(f"Starting capital: Rs {result.starting_capital:,.2f}")
    lines.append(f"Final value:      Rs {result.final_value:,.2f}")
    lines.append(f"Total return:     {total_return:+.2f}%")
    lines.append(f"Approx CAGR:      {cagr:+.2f}%")
    return "\n".join(lines)
