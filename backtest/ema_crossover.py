"""Long-only EMA crossover backtest.

Signal: go long when the fast EMA crosses above the slow EMA ("golden
cross"), exit to cash when it crosses back below ("death cross"). To avoid
lookahead bias, a crossover detected on day T's close is filled at day T+1's
open.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Trade:
    entry_date: str
    entry_price: float
    exit_date: str | None = None
    exit_price: float | None = None

    @property
    def return_pct(self) -> float | None:
        if self.exit_price is None:
            return None
        return (self.exit_price / self.entry_price - 1) * 100


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: list[Trade] = field(default_factory=list)

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
        drawdown = equity / running_max - 1
        return drawdown.min() * 100

    @property
    def win_rate_pct(self) -> float:
        closed = [t for t in self.trades if t.exit_price is not None]
        if not closed:
            return 0.0
        wins = sum(1 for t in closed if t.return_pct > 0)
        return wins / len(closed) * 100

    def summary(self) -> str:
        closed = [t for t in self.trades if t.exit_price is not None]
        lines = [
            f"Total return:     {self.total_return_pct:+.2f}%",
            f"Buy & hold:       {self.buy_hold_return_pct:+.2f}%",
            f"Max drawdown:     {self.max_drawdown_pct:.2f}%",
            f"Trades:           {len(closed)} closed"
            + (" (+1 open)" if len(self.trades) > len(closed) else ""),
            f"Win rate:         {self.win_rate_pct:.1f}%",
        ]
        return "\n".join(lines)


def run(
    candles: list[dict],
    fast: int = 12,
    slow: int = 26,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    if fast >= slow:
        raise ValueError("fast period must be shorter than slow period")

    df = pd.DataFrame(candles).sort_values("date").reset_index(drop=True)
    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["signal"] = (df["ema_fast"] > df["ema_slow"]).astype(int)
    # Fill the next day's open on the bar after a crossover is detected.
    df["position"] = df["signal"].shift(1).fillna(0)

    cash = initial_capital
    shares = 0.0
    equity = []
    trades: list[Trade] = []
    open_trade: Trade | None = None

    for i, row in df.iterrows():
        prev_position = df["position"].iloc[i - 1] if i > 0 else 0
        if row["position"] == 1 and prev_position == 0:
            shares = cash / row["open"]
            cash = 0.0
            open_trade = Trade(entry_date=row["date"], entry_price=row["open"])
        elif row["position"] == 0 and prev_position == 1:
            cash = shares * row["open"]
            shares = 0.0
            if open_trade is not None:
                open_trade.exit_date = row["date"]
                open_trade.exit_price = row["open"]
                trades.append(open_trade)
                open_trade = None
        equity.append(cash + shares * row["close"])

    if open_trade is not None:
        trades.append(open_trade)

    df["equity"] = equity
    return BacktestResult(equity_curve=df[["date", "close", "equity"]], trades=trades)
