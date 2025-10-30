"""
Black-Scholes pricing utilities (no external deps).

Supports dividend yield q. Prices European options and exposes Vega for IV solvers.
"""
from __future__ import annotations
import math
from typing import Literal

OptionType = Literal["call", "put"]


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    # Using erf for numerical stability
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError("Invalid inputs for d1: ensure T>0, sigma>0, S>0, K>0")
    num = math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T
    den = sigma * math.sqrt(T)
    return num / den


def _d2(d1: float, sigma: float, T: float) -> float:
    return d1 - sigma * math.sqrt(T)


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """
    Black-Scholes-Merton price for European call/put with continuous dividend yield q.
    """
    d1 = _d1(S, K, T, r, q, sigma)
    d2 = _d2(d1, sigma, T)
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)

    if option_type == "call":
        return S * df_q * _norm_cdf(d1) - K * df_r * _norm_cdf(d2)
    elif option_type == "put":
        return K * df_r * _norm_cdf(-d2) - S * df_q * _norm_cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Vega = dPrice/dSigma (per 1.0 change in sigma). Scale by 0.01 for per-vol-point.
    """
    d1 = _d1(S, K, T, r, q, sigma)
    df_q = math.exp(-q * T)
    return S * df_q * _norm_pdf(d1) * math.sqrt(T)


__all__ = [
    "bs_price",
    "bs_vega",
]
