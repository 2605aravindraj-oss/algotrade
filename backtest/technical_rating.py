"""A TradingView-style "Technical Rating" (Moving Averages + Oscillators
consensus), computed from scratch on daily OHLC candles.

This is a custom approximation of the well-known rating scheme popularized
by charting platforms -- not a literal reproduction of anyone's proprietary
formula. It combines two groups of signals, each classified Buy/Sell (moving
averages) or Buy/Sell/Neutral (oscillators):

Moving averages (12): SMA and EMA at periods 10, 20, 30, 50, 100, 200.
    Buy  if close > MA
    Sell if close < MA

Oscillators (8), each with its own standard interpretation:
    RSI(14)            Buy if <30, Sell if >70
    Stochastic(14,3,3) Buy if %K<20 and %K>%D, Sell if %K>80 and %K<%D
    CCI(20)            Buy if <-100 and rising, Sell if >100 and falling
    ADX(14) + DI        Buy if ADX>20 and +DI>-DI and +DI rising,
                        Sell if ADX>20 and -DI>+DI and -DI rising
    Momentum(10)       Buy if close-close[10] > 0, Sell if < 0
    MACD(12,26,9)      Buy if MACD line > signal line, Sell if less
    Williams %R(14)    Buy if <-80, Sell if >-20
    Bull/Bear Power(13) Buy if BullPower>0 and BearPower rising,
                        Sell if BearPower<0 and BullPower falling

Each group's rating = (buy_count - sell_count) / total_in_group, in
[-1, +1]. Overall rating = average of the two group ratings, bucketed:

    >=  0.5            Strong Buy
     0.1 to  0.5        Buy
    -0.1 to  0.1        Neutral
    -0.5 to -0.1        Sell
    <= -0.5            Strong Sell

All computed on the daily timeframe using the latest available candle.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _sma(values: list[float], period: int) -> list[float | None]:
    n = len(values)
    out: list[float | None] = [None] * n
    if n < period:
        return out
    window_sum = sum(values[:period])
    out[period - 1] = window_sum / period
    for i in range(period, n):
        window_sum += values[i] - values[i - period]
        out[i] = window_sum / period
    return out


def _ema(values: list[float], period: int) -> list[float | None]:
    n = len(values)
    out: list[float | None] = [None] * n
    if n < period:
        return out
    k = 2 / (period + 1)
    out[period - 1] = sum(values[:period]) / period
    for i in range(period, n):
        out[i] = values[i] * k + out[i - 1] * (1 - k)
    return out


def _rsi(closes: list[float], period: int = 14) -> list[float | None]:
    n = len(closes)
    rsi: list[float | None] = [None] * n
    if n <= period:
        return rsi
    gains = [0.0] * n
    losses = [0.0] * n
    for i in range(1, n):
        change = closes[i] - closes[i - 1]
        gains[i] = max(change, 0.0)
        losses[i] = max(-change, 0.0)
    avg_gain = sum(gains[1:period + 1]) / period
    avg_loss = sum(losses[1:period + 1]) / period

    def _r(ag: float, al: float) -> float:
        return 100.0 if al == 0 else 100 - (100 / (1 + ag / al))

    rsi[period] = _r(avg_gain, avg_loss)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rsi[i] = _r(avg_gain, avg_loss)
    return rsi


def _macd(closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = [(f - s) if (f is not None and s is not None) else None for f, s in zip(ema_fast, ema_slow)]
    macd_values = [v for v in macd_line if v is not None]
    start_idx = next((i for i, v in enumerate(macd_line) if v is not None), len(macd_line))
    sig_ema = _ema(macd_values, signal)
    signal_line: list[float | None] = [None] * len(closes)
    for j, val in enumerate(sig_ema):
        if val is not None:
            signal_line[start_idx + j] = val
    return macd_line, signal_line


def _stochastic(highs, lows, closes, k_period=14, k_smooth=3, d_smooth=3):
    n = len(closes)
    raw_k: list[float | None] = [None] * n
    for i in range(k_period - 1, n):
        hh = max(highs[i - k_period + 1:i + 1])
        ll = min(lows[i - k_period + 1:i + 1])
        raw_k[i] = 50.0 if hh == ll else 100 * (closes[i] - ll) / (hh - ll)
    raw_vals = [v for v in raw_k if v is not None]
    start = next((i for i, v in enumerate(raw_k) if v is not None), n)
    k_sma = _sma(raw_vals, k_smooth)
    k: list[float | None] = [None] * n
    for j, val in enumerate(k_sma):
        if val is not None:
            k[start + j] = val
    k_vals = [v for v in k if v is not None]
    start_d = next((i for i, v in enumerate(k) if v is not None), n)
    d_sma = _sma(k_vals, d_smooth)
    d: list[float | None] = [None] * n
    for j, val in enumerate(d_sma):
        if val is not None:
            d[start_d + j] = val
    return k, d


def _cci(highs, lows, closes, period=20):
    n = len(closes)
    tp = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(n)]
    tp_sma = _sma(tp, period)
    cci: list[float | None] = [None] * n
    for i in range(period - 1, n):
        window = tp[i - period + 1:i + 1]
        mean = tp_sma[i]
        mad = sum(abs(x - mean) for x in window) / period
        cci[i] = 0.0 if mad == 0 else (tp[i] - mean) / (0.015 * mad)
    return cci


def _adx_di(highs, lows, closes, period=14):
    n = len(closes)
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    tr = [0.0] * n
    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))

    def _wilder_smooth(vals):
        out: list[float | None] = [None] * n
        if n <= period:
            return out
        out[period] = sum(vals[1:period + 1])
        for i in range(period + 1, n):
            out[i] = out[i - 1] - out[i - 1] / period + vals[i]
        return out

    tr_s = _wilder_smooth(tr)
    plus_dm_s = _wilder_smooth(plus_dm)
    minus_dm_s = _wilder_smooth(minus_dm)

    plus_di: list[float | None] = [None] * n
    minus_di: list[float | None] = [None] * n
    for i in range(n):
        if tr_s[i]:
            plus_di[i] = 100 * plus_dm_s[i] / tr_s[i]
            minus_di[i] = 100 * minus_dm_s[i] / tr_s[i]

    dx: list[float | None] = [None] * n
    for i in range(n):
        if plus_di[i] is not None and minus_di[i] is not None and (plus_di[i] + minus_di[i]) > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])

    dx_start = next((i for i, v in enumerate(dx) if v is not None), n)
    dx_vals = [v for v in dx[dx_start:] if v is not None]
    adx: list[float | None] = [None] * n
    if len(dx_vals) >= period:
        adx[dx_start + period - 1] = sum(dx_vals[:period]) / period
        for i in range(dx_start + period, n):
            prev = adx[i - 1]
            adx[i] = (prev * (period - 1) + dx[i]) / period

    return adx, plus_di, minus_di


def _williams_r(highs, lows, closes, period=14):
    n = len(closes)
    out: list[float | None] = [None] * n
    for i in range(period - 1, n):
        hh = max(highs[i - period + 1:i + 1])
        ll = min(lows[i - period + 1:i + 1])
        out[i] = 50.0 if hh == ll else -100 * (hh - closes[i]) / (hh - ll)
    return out


@dataclass
class Rating:
    symbol: str
    close: float
    ma_buy: int = 0
    ma_sell: int = 0
    osc_buy: int = 0
    osc_sell: int = 0
    osc_neutral: int = 0
    ma_details: dict = field(default_factory=dict)
    osc_details: dict = field(default_factory=dict)

    @property
    def ma_rating(self) -> float:
        total = self.ma_buy + self.ma_sell
        return 0.0 if total == 0 else (self.ma_buy - self.ma_sell) / total

    @property
    def osc_rating(self) -> float:
        total = self.osc_buy + self.osc_sell + self.osc_neutral
        return 0.0 if total == 0 else (self.osc_buy - self.osc_sell) / total

    @property
    def overall_rating(self) -> float:
        return (self.ma_rating + self.osc_rating) / 2

    @property
    def label(self) -> str:
        r = self.overall_rating
        if r >= 0.5:
            return "Strong Buy"
        if r >= 0.1:
            return "Buy"
        if r > -0.1:
            return "Neutral"
        if r > -0.5:
            return "Sell"
        return "Strong Sell"


def rate(symbol: str, candles: list[dict]) -> Rating | None:
    """candles: list of {date, open, high, low, close, volume} sorted ascending by date."""
    if len(candles) < 205:
        return None
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]
    i = len(closes) - 1
    close = closes[i]

    r = Rating(symbol=symbol, close=close)

    for period in (10, 20, 30, 50, 100, 200):
        sma = _sma(closes, period)[i]
        ema = _ema(closes, period)[i]
        for name, ma_val in ((f"SMA{period}", sma), (f"EMA{period}", ema)):
            if ma_val is None:
                continue
            if close > ma_val:
                r.ma_buy += 1
                r.ma_details[name] = "Buy"
            else:
                r.ma_sell += 1
                r.ma_details[name] = "Sell"

    def _osc(name, verdict):
        r.osc_details[name] = verdict
        if verdict == "Buy":
            r.osc_buy += 1
        elif verdict == "Sell":
            r.osc_sell += 1
        else:
            r.osc_neutral += 1

    rsi = _rsi(closes, 14)
    if rsi[i] is not None:
        _osc("RSI(14)", "Buy" if rsi[i] < 30 else "Sell" if rsi[i] > 70 else "Neutral")

    k, d = _stochastic(highs, lows, closes)
    if k[i] is not None and d[i] is not None:
        if k[i] < 20 and k[i] > d[i]:
            _osc("Stochastic(14,3,3)", "Buy")
        elif k[i] > 80 and k[i] < d[i]:
            _osc("Stochastic(14,3,3)", "Sell")
        else:
            _osc("Stochastic(14,3,3)", "Neutral")

    cci = _cci(highs, lows, closes, 20)
    if cci[i] is not None and cci[i - 1] is not None:
        if cci[i] < -100 and cci[i] > cci[i - 1]:
            _osc("CCI(20)", "Buy")
        elif cci[i] > 100 and cci[i] < cci[i - 1]:
            _osc("CCI(20)", "Sell")
        else:
            _osc("CCI(20)", "Neutral")

    adx, plus_di, minus_di = _adx_di(highs, lows, closes, 14)
    if adx[i] is not None and plus_di[i] is not None and minus_di[i] is not None and plus_di[i - 1] is not None:
        if adx[i] > 20 and plus_di[i] > minus_di[i] and plus_di[i] > plus_di[i - 1]:
            _osc("ADX(14)/DI", "Buy")
        elif adx[i] > 20 and minus_di[i] > plus_di[i] and minus_di[i] > minus_di[i - 1]:
            _osc("ADX(14)/DI", "Sell")
        else:
            _osc("ADX(14)/DI", "Neutral")

    if i >= 10:
        mom = closes[i] - closes[i - 10]
        _osc("Momentum(10)", "Buy" if mom > 0 else "Sell" if mom < 0 else "Neutral")

    macd_line, signal_line = _macd(closes)
    if macd_line[i] is not None and signal_line[i] is not None:
        _osc("MACD(12,26,9)", "Buy" if macd_line[i] > signal_line[i] else "Sell")

    wr = _williams_r(highs, lows, closes, 14)
    if wr[i] is not None:
        _osc("Williams %R(14)", "Buy" if wr[i] < -80 else "Sell" if wr[i] > -20 else "Neutral")

    ema13 = _ema(closes, 13)
    if ema13[i] is not None and ema13[i - 1] is not None:
        bull = highs[i] - ema13[i]
        bear = lows[i] - ema13[i]
        bull_prev = highs[i - 1] - ema13[i - 1]
        bear_prev = lows[i - 1] - ema13[i - 1]
        if bull > 0 and bear > bear_prev:
            _osc("Bull/Bear Power(13)", "Buy")
        elif bear < 0 and bull < bull_prev:
            _osc("Bull/Bear Power(13)", "Sell")
        else:
            _osc("Bull/Bear Power(13)", "Neutral")

    return r
