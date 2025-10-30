"""
Black-Scholes Greeks (delta, theta) with dividend yield q.
"""
from __future__ import annotations
import math
from .black_scholes import _d1, _d2, _norm_cdf, _norm_pdf
from typing import Literal

OptionType = Literal["call", "put"]


def bs_delta(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: OptionType = "call") -> float:
    d1 = _d1(S, K, T, r, q, sigma)
    df_q = math.exp(-q * T)
    if option_type == "call":
        return df_q * _norm_cdf(d1)
    else:
        return -df_q * _norm_cdf(-d1)


def bs_theta(
    S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: OptionType = "call"
) -> float:
    """
    Returns theta per year (change in price per year). To get per-day, divide by 365.
    """
    d1 = _d1(S, K, T, r, q, sigma)
    d2 = _d2(d1, sigma, T)
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)
    term1 = - (S * df_q * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    if option_type == "call":
        return term1 - r * K * df_r * _norm_cdf(d2) + q * S * df_q * _norm_cdf(d1)
    else:
        return term1 + r * K * df_r * _norm_cdf(-d2) - q * S * df_q * _norm_cdf(-d1)


__all__ = ["bs_delta", "bs_theta"]

