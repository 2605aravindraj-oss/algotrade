"""Supertrend stop-and-reverse, unfiltered, percent-of-equity backtest --
built to reproduce a TradingView strategy-tester-style run for direct
comparison: 100% of equity per trade, commission % per side, slippage in
ticks, on the NIFTY 50 INDEX (not futures -- the index has full 1-minute
history back through 2025 with no expiry/contract-roll complications,
unlike futures which would need monthly-contract stitching over a
~1.7-year window).

Reuses backtest.supertrend._compute_supertrend (same Wilder-ATR recursive
band formula) on bars resampled to an arbitrary timeframe (15-minute by
default here, vs. the 5-minute bars used elsewhere in this session).

Trading rule -- always in the market, no filter, no stop-loss, no target:
    direction flips bullish -> exit any short, go LONG at this bar's close
    direction flips bearish -> exit any long, go SHORT at this bar's close
No entry on the very first direction value (nothing to flip from yet);
every subsequent flip trades. A position can carry overnight/across many
days -- this is the "positional" behavior, matching a plain
ta.supertrend() stop-and-reverse strategy with no daily reset.

Position sizing / costs, applied to match the TradingView reference run:
    100% of current equity re-invested each trade (compounding)
    commission_rate applied to notional on both entry and exit
    slippage_ticks * tick_size applied against the trader on both fills
Equity is marked to market at the close of every bar (not just at trade
boundaries) so a daily equity curve exists even through a position held
for many days -- needed for a real max-drawdown and Sharpe calculation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from data_sources import cache, upstox_client
from backtest.supertrend import _compute_supertrend
from backtest.sweep_reclaim_breakout import _resample

UNDERLYING_KEY = "NSE_INDEX|Nifty 50"
TICK_SIZE = 0.05  # NIFTY 50 index minimum tick


@dataclass
class Trade:
    direction: str
    entry_time: str
    entry_price: float
    exit_time: str | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    equity_before: float = 0.0
    equity_after: float | None = None

    @property
    def return_pct(self) -> float | None:
        if self.equity_after is None:
            return None
        return 100 * (self.equity_after - self.equity_before) / self.equity_before


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    daily_equity: list[tuple[str, float]] = field(default_factory=list)
    starting_capital: float = 0.0
    final_equity: float = 0.0


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    bar_minutes: int = 15,
    period: int = 10,
    multiplier: float = 2.5,
    starting_capital: float = 100_000.0,
    commission_rate: float = 0.0003,   # 0.03% per side
    slippage_ticks: float = 1.0,
    tick_size: float = TICK_SIZE,
) -> BacktestResult:
    trading_days = upstox_client.get_daily_history(underlying_key, from_date, to_date)
    all_1min: list[list] = []
    for day in trading_days:
        rows = sorted(
            cache.get_day_candles_cached(underlying_key, "1minute", day["date"], expired=False),
            key=lambda c: c[0],
        )
        all_1min.extend(rows)
    all_1min.sort(key=lambda c: c[0])
    if not all_1min:
        return BacktestResult(starting_capital=starting_capital, final_equity=starting_capital)

    bars = _resample(all_1min, bar_minutes)
    if len(bars) < period + 2:
        return BacktestResult(starting_capital=starting_capital, final_equity=starting_capital)

    direction = _compute_supertrend(bars, period, multiplier)
    slippage_pts = slippage_ticks * tick_size

    trades: list[Trade] = []
    position: dict | None = None
    prev_dir: int | None = None
    equity = starting_capital
    daily_marks: dict[str, float] = {}

    def _mtm(price: float) -> float:
        if position is None:
            return equity
        if position["direction"] == "LONG":
            return position["qty"] * price
        return position["equity_after_entry_fee"] + position["qty"] * (position["entry_price_eff"] - price)

    def _open(direction_label: str, price: float, ts: str) -> None:
        nonlocal position
        entry_price_eff = price + slippage_pts if direction_label == "LONG" else price - slippage_pts
        entry_commission = equity * commission_rate
        equity_after_entry_fee = equity - entry_commission
        qty = equity_after_entry_fee / entry_price_eff
        position = {
            "direction": direction_label, "entry_price_eff": entry_price_eff, "qty": qty,
            "equity_after_entry_fee": equity_after_entry_fee,
            "entry_time": ts, "entry_price": price, "equity_before": equity,
        }

    def _close(price: float, ts: str, reason: str) -> None:
        nonlocal position, equity
        exit_price_eff = price - slippage_pts if position["direction"] == "LONG" else price + slippage_pts
        mtm_before_fee = _mtm(exit_price_eff)
        exit_commission = mtm_before_fee * commission_rate
        new_equity = mtm_before_fee - exit_commission
        trades.append(Trade(
            direction=position["direction"], entry_time=position["entry_time"], entry_price=position["entry_price"],
            exit_time=ts, exit_price=price, exit_reason=reason,
            equity_before=position["equity_before"], equity_after=new_equity,
        ))
        equity = new_equity
        position = None

    for i, bar in enumerate(bars):
        ts, o, h, l, c, v, oi = bar
        d = ts[:10]
        dirn = direction[i]
        if dirn is None:
            continue

        if prev_dir is not None and dirn != prev_dir:
            if position is not None:
                _close(c, ts, "reverse")
            _open("LONG" if dirn == 1 else "SHORT", c, ts)

        prev_dir = dirn
        daily_marks[d] = _mtm(c)  # last mark of the day wins (dict overwrite, dates in order)

    if position is not None:
        last = bars[-1]
        _close(last[4], last[0], "eod_data_end")
        daily_marks[last[0][:10]] = equity

    daily_equity = sorted(daily_marks.items())
    return BacktestResult(trades=trades, daily_equity=daily_equity, starting_capital=starting_capital, final_equity=equity)


def summary(result: BacktestResult) -> str:
    ok = [t for t in result.trades if t.equity_after is not None]
    if not ok:
        return "No trades."

    net_profit_pct = 100 * (result.final_equity - result.starting_capital) / result.starting_capital
    wins = [t for t in ok if t.equity_after > t.equity_before]
    losses = [t for t in ok if t.equity_after <= t.equity_before]
    gross_profit = sum(t.equity_after - t.equity_before for t in wins)
    gross_loss = sum(t.equity_before - t.equity_after for t in losses)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    win_rate = 100 * len(wins) / len(ok)

    peak = result.starting_capital
    max_dd = 0.0
    for _, val in result.daily_equity:
        peak = max(peak, val)
        max_dd = min(max_dd, 100 * (val - peak) / peak)

    vals = [v for _, v in result.daily_equity]
    daily_returns = [(vals[i] - vals[i - 1]) / vals[i - 1] for i in range(1, len(vals)) if vals[i - 1] != 0]
    if daily_returns:
        mean_r = sum(daily_returns) / len(daily_returns)
        var = sum((r - mean_r) ** 2 for r in daily_returns) / len(daily_returns)
        std_r = var ** 0.5
        sharpe = (mean_r / std_r) * (252 ** 0.5) if std_r > 0 else float("nan")
    else:
        sharpe = float("nan")

    return "\n".join([
        f"Total trades:     {len(ok)}",
        f"Net profit:       {net_profit_pct:+.2f}%  (Rs {result.starting_capital:,.2f} -> Rs {result.final_equity:,.2f})",
        f"Profit factor:    {profit_factor:.2f}",
        f"Win rate:         {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)",
        f"Max drawdown:     {max_dd:.2f}%",
        f"Sharpe ratio:     {sharpe:.2f}  (annualized from daily equity marks, rf=0)",
    ])
