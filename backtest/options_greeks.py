"""Black-Scholes option pricing, implied volatility, and delta.

Built for strategies that target an option by DELTA rather than by a
fixed strike-distance-in-points (e.g. "buy the option whose delta is
near 0.66"): Upstox has no live Greeks feed, so delta has to be
derived from the option's own observed premium -- back out its
implied volatility via Black-Scholes inversion, then plug that into
the standard delta formula.

RISK-FREE RATE: a fixed 7% per annum (RISK_FREE_RATE) -- a stable
approximation for short-term Indian rates, not pulled from any live
source. Delta is not very sensitive to r over the few-day holding
periods this codebase's strategies use, so this is a reasonable
simplification, but it's an assumption, not a fetched value.
"""
from __future__ import annotations

import math

RISK_FREE_RATE = 0.07


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def bs_price(S: float, K: float, T: float, sigma: float, opt_type: str, r: float = RISK_FREE_RATE) -> float:
    """European option price. T in years, sigma annualized. Falls back
    to intrinsic value if T or sigma is non-positive (expiry moment)."""
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if opt_type == "CE" else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt_type == "CE":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_delta(S: float, K: float, T: float, sigma: float, opt_type: str, r: float = RISK_FREE_RATE) -> float:
    """Call delta in [0,1], put delta in [-1,0]."""
    if T <= 0 or sigma <= 0:
        if opt_type == "CE":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1) if opt_type == "CE" else _norm_cdf(d1) - 1.0


def implied_vol(
    price: float, S: float, K: float, T: float, opt_type: str, r: float = RISK_FREE_RATE,
    tol: float = 1e-4, max_iter: int = 50,
) -> float | None:
    """Newton-Raphson with a bisection fallback. Returns None if the
    quoted price is below intrinsic value or T<=0 (no valid IV)."""
    intrinsic = max(0.0, S - K) if opt_type == "CE" else max(0.0, K - S)
    if price < intrinsic - 1e-6 or T <= 0:
        return None

    sigma = 0.3
    for _ in range(max_iter):
        price_est = bs_price(S, K, T, sigma, opt_type, r)
        diff = price_est - price
        if abs(diff) < tol:
            return sigma
        d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
        vega = S * _norm_pdf(d1) * math.sqrt(T)
        if vega < 1e-8:
            break
        sigma -= diff / vega
        sigma = min(max(sigma, 0.001), 5.0)

    lo, hi = 0.001, 5.0
    for _ in range(100):
        mid = (lo + hi) / 2
        p = bs_price(S, K, T, mid, opt_type, r)
        if abs(p - price) < tol:
            return mid
        if p < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def strike_by_delta(
    target_delta: float,
    opt_type: str,
    spot: float,
    T: float,
    strike_premiums: dict[float, float],
    r: float = RISK_FREE_RATE,
) -> float | None:
    """Pick the strike (among strike_premiums, {strike: observed
    premium at this moment}) whose own Black-Scholes delta (IV backed
    out from ITS OWN premium) is closest in magnitude to target_delta
    (always pass target_delta as a positive number, e.g. 0.66, for
    both calls and puts -- matched against |delta|). Returns None if
    no candidate strike had a valid IV."""
    best_strike = None
    best_diff = None
    for strike, premium in strike_premiums.items():
        iv = implied_vol(premium, spot, strike, T, opt_type, r)
        if iv is None:
            continue
        delta = bs_delta(spot, strike, T, iv, opt_type, r)
        diff = abs(abs(delta) - target_delta)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_strike = strike
    return best_strike
