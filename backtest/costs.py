"""Approximate Indian F&O options transaction costs (2025/26 rates).

Applied per executed fill (buy or sell), on the fill's premium value
(price * lot_size):

- Brokerage: flat Rs 20 per executed order (typical discount broker rate).
- STT (Securities Transaction Tax): 0.1% of premium, sell side only
  (charged whenever an option is sold, whether opening a short or closing
  a long).
- Exchange transaction charges: ~0.03503% of premium, both sides.
- SEBI turnover fee: Rs 10/crore (0.0001%) of premium, both sides.
- Stamp duty: 0.003% of premium, buy side only.
- GST: 18% on (brokerage + exchange charges + SEBI fee).

These are approximations for cost-awareness, not a substitute for a real
contract note -- rates change periodically and brokers vary.
"""
from __future__ import annotations

from dataclasses import dataclass

BROKERAGE_PER_ORDER = 20.0
STT_SELL_RATE = 0.001
EXCHANGE_TXN_RATE = 0.0003503
SEBI_FEE_RATE = 0.000001
STAMP_DUTY_BUY_RATE = 0.00003
GST_RATE = 0.18


@dataclass
class Fill:
    price: float
    lot_size: int
    side: str  # "BUY" or "SELL"


def fill_cost(fill: Fill) -> float:
    premium_value = fill.price * fill.lot_size
    brokerage = BROKERAGE_PER_ORDER
    exchange_txn = premium_value * EXCHANGE_TXN_RATE
    sebi_fee = premium_value * SEBI_FEE_RATE
    stt = premium_value * STT_SELL_RATE if fill.side == "SELL" else 0.0
    stamp_duty = premium_value * STAMP_DUTY_BUY_RATE if fill.side == "BUY" else 0.0
    gst = GST_RATE * (brokerage + exchange_txn + sebi_fee)
    return brokerage + exchange_txn + sebi_fee + stt + stamp_duty + gst


def total_cost(fills: list[Fill]) -> float:
    return sum(fill_cost(f) for f in fills)
