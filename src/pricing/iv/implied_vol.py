"""
Robust implied volatility solver using damped Newton-Raphson with safeguards.
Falls back to bisection if Newton fails. Works for both calls and puts.
"""
from __future__ import annotations
import math
from typing import Literal, Optional
from ..black_scholes import bs_price, bs_vega

OptionType = Literal["call", "put"]


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: OptionType = "call",
    initial_sigma: float = 0.2,
    tol: float = 1e-7,
    max_iter: int = 100,
    sigma_bounds: tuple[float, float] = (1e-6, 5.0),
) -> Optional[float]:
    """
    Compute Black-Scholes implied volatility for a European option.

    Returns None if no solution in bounds.
    """
    low, high = sigma_bounds
    sigma = max(min(initial_sigma, high), low)

    # No-arbitrage price bounds (rough):
    # For calls: max(0, S*e^{-qT} - K*e^{-rT}) <= C <= S*e^{-qT}
    # For puts:  max(0, K*e^{-rT} - S*e^{-qT}) <= P <= K*e^{-rT}
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)
    if option_type == "call":
        intrinsic_lower = max(0.0, S * df_q - K * df_r)
        upper = S * df_q
    else:
        intrinsic_lower = max(0.0, K * df_r - S * df_q)
        upper = K * df_r

    if price < intrinsic_lower - 1e-12 or price > upper + 1e-12 or T <= 0:
        return None

    # Newton iterations with damping
    for _ in range(max_iter):
        model = bs_price(S, K, T, r, q, sigma, option_type)
        vega = bs_vega(S, K, T, r, q, sigma)
        diff = model - price
        if abs(diff) < tol:
            return sigma
        if vega <= 1e-12:  # flat vega -> break
            break
        step = diff / vega
        # damped update
        for damp in (1.0, 0.5, 0.25, 0.1):
            trial = sigma - damp * step
            if low <= trial <= high:
                trial_price = bs_price(S, K, T, r, q, trial, option_type)
                if abs(trial_price - price) <= abs(diff):
                    sigma = trial
                    break
        else:
            break

    # Bisection fallback
    left, right = low, high
    f_left = bs_price(S, K, T, r, q, left, option_type) - price
    f_right = bs_price(S, K, T, r, q, right, option_type) - price
    if f_left * f_right > 0:
        return None
    for _ in range(200):
        mid = 0.5 * (left + right)
        f_mid = bs_price(S, K, T, r, q, mid, option_type) - price
        if abs(f_mid) < tol:
            return mid
        if f_left * f_mid <= 0:
            right, f_right = mid, f_mid
        else:
            left, f_left = mid, f_mid
    return 0.5 * (left + right)


__all__ = ["implied_vol"]

