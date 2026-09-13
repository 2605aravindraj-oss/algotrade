"""Downtrend reversal, break-of-structure + retest backtest.

Pattern this encodes:
  1. Downtrend: swing highs and swing lows both descending (lower highs,
     lower lows).
  2. Reversal setup: a swing low prints *higher* than the prior swing low
     (a "higher low") while price is still below the last swing high (the
     most recent lower high) -> the downtrend's momentum is weakening.
  3. Break of structure: price closes above that last lower high, i.e. it
     crosses the level that defined the downtrend.
  4. Retest: price pulls back toward the broken level, holds above it
     (doesn't close back below), then closes higher again -> confirms the
     level flipped from resistance to support.
  5. Entry: long at the next bar's open after the retest bounce is
     confirmed. Stop below the retest low; target a fixed reward:risk
     multiple of the resulting risk.

Swing points are detected with a centered fractal (a bar is a swing high/low
if it's the most extreme high/low in a `lookback`-bar window on each side).
A pivot at bar i is only usable in the state machine once bar i+lookback has
been reached, since that's the earliest point it could have been confirmed
in real time -- this keeps the backtest free of lookahead bias.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Setup:
    """One instance of the pattern, from higher-low to entry (or invalidation)."""

    downtrend_high_date: str
    downtrend_high_price: float
    higher_low_date: str
    higher_low_price: float
    breakout_date: str | None = None
    breakout_price: float | None = None
    retest_date: str | None = None
    retest_low: float | None = None
    entry_date: str | None = None
    entry_price: float | None = None
    stop_loss: float | None = None
    target: float | None = None
    exit_date: str | None = None
    exit_price: float | None = None
    outcome: str = "invalidated"  # invalidated | timed_out | stopped_out | target_hit | open

    @property
    def return_pct(self) -> float | None:
        if self.entry_price is None or self.exit_price is None:
            return None
        return (self.exit_price / self.entry_price - 1) * 100


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    setups: list[Setup] = field(default_factory=list)

    @property
    def trades(self) -> list[Setup]:
        return [s for s in self.setups if s.entry_price is not None]

    @property
    def total_return_pct(self) -> float:
        return (self.equity_curve["equity"].iloc[-1] / self.equity_curve["equity"].iloc[0] - 1) * 100

    @property
    def buy_hold_return_pct(self) -> float:
        close = self.equity_curve["close"]
        return (close.iloc[-1] / close.iloc[0] - 1) * 100

    @property
    def max_drawdown_pct(self) -> float:
        equity = self.equity_curve["equity"]
        running_max = equity.cummax()
        return (equity / running_max - 1).min() * 100

    @property
    def win_rate_pct(self) -> float:
        closed = [t for t in self.trades if t.exit_price is not None]
        if not closed:
            return 0.0
        return sum(1 for t in closed if t.return_pct > 0) / len(closed) * 100

    def summary(self) -> str:
        closed = [t for t in self.trades if t.exit_price is not None]
        return "\n".join(
            [
                f"Total return:     {self.total_return_pct:+.2f}%",
                f"Buy & hold:       {self.buy_hold_return_pct:+.2f}%",
                f"Max drawdown:     {self.max_drawdown_pct:.2f}%",
                f"Setups detected:  {len(self.setups)}",
                f"Trades taken:     {len(closed)}"
                + (" (+1 open)" if len(self.trades) > len(closed) else ""),
                f"Win rate:         {self.win_rate_pct:.1f}%",
            ]
        )


def _find_zigzag(df: pd.DataFrame, lookback: int) -> list[dict]:
    """Alternating swing highs/lows, each tagged with the bar index it's
    first usable at (confirmed_at = pivot index + lookback)."""
    highs, lows = df["high"], df["low"]
    n = len(df)
    candidates = []
    for i in range(lookback, n - lookback):
        window_h = highs.iloc[i - lookback : i + lookback + 1]
        if highs.iloc[i] == window_h.max() and (window_h == window_h.max()).sum() == 1:
            candidates.append({"i": i, "type": "high", "price": highs.iloc[i]})
        window_l = lows.iloc[i - lookback : i + lookback + 1]
        if lows.iloc[i] == window_l.min() and (window_l == window_l.min()).sum() == 1:
            candidates.append({"i": i, "type": "low", "price": lows.iloc[i]})
    candidates.sort(key=lambda c: c["i"])

    zigzag: list[dict] = []
    for c in candidates:
        if zigzag and zigzag[-1]["type"] == c["type"]:
            # Same type as the last pivot: keep whichever is more extreme.
            more_extreme = (
                c["price"] > zigzag[-1]["price"]
                if c["type"] == "high"
                else c["price"] < zigzag[-1]["price"]
            )
            if more_extreme:
                zigzag[-1] = c
        else:
            zigzag.append(c)
    for c in zigzag:
        c["confirmed_at"] = c["i"] + lookback
        c["date"] = df["date"].iloc[c["i"]]
    return zigzag


def run(
    candles: list[dict],
    lookback: int = 3,
    retest_tolerance_pct: float = 2.0,
    stop_buffer_pct: float = 0.5,
    reward_risk: float = 2.0,
    breakout_timeout_bars: int = 40,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    df = pd.DataFrame(candles).sort_values("date").reset_index(drop=True)
    zigzag = _find_zigzag(df, lookback)

    setups: list[Setup] = []
    highs: list[dict] = []
    lows: list[dict] = []
    state = "SCANNING"
    current: Setup | None = None
    breakout_bar_i: int | None = None

    cash = initial_capital
    shares = 0.0
    equity = []
    zz_idx = 0

    for i, row in df.iterrows():
        # Absorb any pivots that become confirmed as of this bar.
        while zz_idx < len(zigzag) and zigzag[zz_idx]["confirmed_at"] <= i:
            piv = zigzag[zz_idx]
            (highs if piv["type"] == "high" else lows).append(piv)
            zz_idx += 1

            if state == "SCANNING" and piv["type"] == "low" and len(highs) >= 1 and len(lows) >= 2:
                downtrend = highs[-1]["price"] < (highs[-2]["price"] if len(highs) >= 2 else float("inf"))
                higher_low = lows[-1]["price"] > lows[-2]["price"]
                below_last_high = lows[-1]["price"] < highs[-1]["price"]
                if downtrend and higher_low and below_last_high:
                    current = Setup(
                        downtrend_high_date=highs[-1]["date"],
                        downtrend_high_price=highs[-1]["price"],
                        higher_low_date=piv["date"],
                        higher_low_price=piv["price"],
                    )
                    state = "HL_FORMED"

        if state == "HL_FORMED" and current is not None:
            if row["close"] > current.downtrend_high_price:
                current.breakout_date = row["date"]
                current.breakout_price = row["close"]
                breakout_bar_i = i
                state = "BROKEN"
            elif row["close"] < current.higher_low_price:
                current.outcome = "invalidated"
                setups.append(current)
                current, state = None, "SCANNING"

        elif state == "BROKEN" and current is not None:
            breakout_level = current.downtrend_high_price
            if row["close"] < breakout_level:
                current.outcome = "invalidated"
                setups.append(current)
                current, state = None, "SCANNING"
            else:
                near_level = row["low"] <= breakout_level * (1 + retest_tolerance_pct / 100)
                if near_level and current.retest_date is None:
                    current.retest_date = row["date"]
                    current.retest_low = row["low"]
                elif (
                    current.retest_date is not None
                    and current.entry_price is None
                    and row["close"] > df["close"].iloc[i - 1]
                ):
                    entry_price = df["open"].iloc[i + 1] if i + 1 < len(df) else None
                    if entry_price is not None:
                        risk = entry_price - current.retest_low * (1 - stop_buffer_pct / 100)
                        current.entry_date = df["date"].iloc[i + 1]
                        current.entry_price = entry_price
                        current.stop_loss = current.retest_low * (1 - stop_buffer_pct / 100)
                        current.target = entry_price + reward_risk * risk
                        current.outcome = "open"
                        shares = cash / entry_price
                        cash = 0.0
                        state = "IN_POSITION"
                if state == "BROKEN" and i - breakout_bar_i > breakout_timeout_bars:
                    current.outcome = "timed_out"
                    setups.append(current)
                    current, state = None, "SCANNING"

        elif state == "IN_POSITION" and current is not None:
            if row["low"] <= current.stop_loss:
                current.exit_date, current.exit_price = row["date"], current.stop_loss
                current.outcome = "stopped_out"
                cash, shares = shares * current.exit_price, 0.0
                setups.append(current)
                current, state = None, "SCANNING"
            elif row["high"] >= current.target:
                current.exit_date, current.exit_price = row["date"], current.target
                current.outcome = "target_hit"
                cash, shares = shares * current.exit_price, 0.0
                setups.append(current)
                current, state = None, "SCANNING"

        equity.append(cash + shares * row["close"])

    if current is not None:
        setups.append(current)

    df["equity"] = equity
    return BacktestResult(equity_curve=df[["date", "close", "equity"]], setups=setups)
