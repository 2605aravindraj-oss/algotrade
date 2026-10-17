"""Same MACD-trend + RSI(2)-momentum signals as macd_rsi2_momentum.py, but
realized through NIFTY weekly ATM options instead of full underlying
notional -- to test whether the much smaller option premium (and
correspondingly smaller absolute transaction costs) changes the
economics of a signal that trades too often for full futures notional.

Entry/exit TIMING is identical to the underlying-based signals (the same
RSI/MACD/stop-loss-on-underlying/max-hold/EOD triggers, evaluated on spot
price); only the instrument used to realize P&L differs:
    LONG signal  -> buy the ATM call at entry, sell it at exit
    SHORT signal -> buy the ATM put at entry, sell it at exit

Requires an Upstox access token (expired-instruments API) for the option
premium history.
"""
from __future__ import annotations

from dataclasses import dataclass

from data_sources import cache
from backtest import costs, options_common as oc
from backtest.macd_rsi2_momentum import run as run_underlying_signals
from backtest.iron_condor import UNDERLYING_KEY


@dataclass
class OptionTrade:
    date: str
    direction: str  # LONG (bought call) | SHORT (bought put)
    expiry: str
    strike: float
    entry_time: str
    entry_premium: float
    exit_time: str
    exit_premium: float
    lot_size: int
    exit_reason: str

    @property
    def pnl_points(self) -> float:
        return self.exit_premium - self.entry_premium

    @property
    def pnl_rupees_gross(self) -> float:
        return self.pnl_points * self.lot_size

    @property
    def cost_rupees(self) -> float:
        fills = [
            costs.Fill(price=self.entry_premium, lot_size=self.lot_size, side="BUY"),
            costs.Fill(price=self.exit_premium, lot_size=self.lot_size, side="SELL"),
        ]
        return costs.total_cost(fills)

    @property
    def pnl_rupees(self) -> float:
        return self.pnl_rupees_gross - self.cost_rupees


def _to_minutes(t: str) -> int:
    return int(t[:2]) * 60 + int(t[3:5])


def _nearest_bar_to_time(rows: list[list], time_str: str) -> list | None:
    if not rows:
        return None
    target = _to_minutes(time_str)
    return min(rows, key=lambda r: abs(_to_minutes(r[0][11:16]) - target))


def run(
    from_date: str,
    to_date: str,
    underlying_key: str = UNDERLYING_KEY,
    strike_step: int = 50,
    access_token: str | None = None,
    **signal_kwargs,
) -> list[OptionTrade]:
    underlying_trades = run_underlying_signals(from_date, to_date, underlying_key=underlying_key, **signal_kwargs)

    expiries = sorted(cache.get_expired_expiries_cached(underlying_key, "options", access_token))
    chain_cache: dict[str, dict] = {}
    option_trades: list[OptionTrade] = []

    for t in underlying_trades:
        if t.exit_price is None:
            continue
        d = t.date
        expiry = next((e for e in expiries if e >= d), None)
        if expiry is None:
            continue

        if expiry not in chain_cache:
            chain_cache[expiry] = oc.build_chain_lookup(
                cache.get_expired_option_chain_cached(underlying_key, expiry, access_token)
            )
        lookup = chain_cache[expiry]

        atm = oc.round_to_step(t.entry_price, strike_step)
        opt_type = "CE" if t.direction == "LONG" else "PE"
        contract = oc.nearest_contract(lookup, atm, opt_type)
        if contract is None:
            continue

        candles = cache.get_day_candles_cached(
            contract["instrument_key"], "1minute", d, expired=True, access_token=access_token
        )
        rows = sorted(candles, key=lambda c: c[0])
        entry_bar = _nearest_bar_to_time(rows, t.entry_time[11:16])
        exit_bar = _nearest_bar_to_time(rows, t.exit_time[11:16])
        if entry_bar is None or exit_bar is None:
            continue

        option_trades.append(OptionTrade(
            date=d, direction=t.direction, expiry=expiry, strike=contract["strike_price"],
            entry_time=entry_bar[0], entry_premium=entry_bar[4],
            exit_time=exit_bar[0], exit_premium=exit_bar[4],
            lot_size=contract["lot_size"], exit_reason=t.exit_reason,
        ))

    return option_trades


def summary(trades: list[OptionTrade]) -> str:
    if not trades:
        return "No trades."
    pnls = [t.pnl_rupees for t in trades]
    gross = sum(t.pnl_rupees_gross for t in trades)
    total_costs = sum(t.cost_rupees for t in trades)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_wins = [t for t in trades if t.pnl_points > 0]

    cum = 0.0
    equity = []
    for p in pnls:
        cum += p
        equity.append(cum)
    running_max = equity[0]
    max_dd = 0.0
    for e in equity:
        running_max = max(running_max, e)
        max_dd = min(max_dd, e - running_max)

    lines = [
        f"Total trades:     {len(trades)}",
        f"Gross win rate:   {len(gross_wins) / len(trades) * 100:.1f}%",
        f"Gross P&L:        Rs {gross:,.2f}",
        f"Total costs:      Rs {total_costs:,.2f}  (Rs {total_costs / len(trades):,.2f}/trade)",
        f"Net P&L:          Rs {sum(pnls):,.2f}",
        f"Net win rate:     {len(wins) / len(trades) * 100:.1f}%  ({len(wins)}W / {len(losses)}L)",
        f"Avg win:          Rs {(sum(wins) / len(wins)) if wins else 0:,.2f}",
        f"Avg loss:         Rs {(sum(losses) / len(losses)) if losses else 0:,.2f}",
        f"Best trade:       Rs {max(pnls):,.2f}",
        f"Worst trade:      Rs {min(pnls):,.2f}",
        f"Max drawdown:     Rs {max_dd:,.2f}",
    ]
    return "\n".join(lines)
