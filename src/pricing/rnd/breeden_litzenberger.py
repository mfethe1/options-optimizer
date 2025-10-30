"""
Risk-neutral density estimation from an arbitrage-aware IV surface via numerical differentiation
of call prices wrt strike (Breeden-Litzenberger). Surface fit is a later task; this provides API and
numerical core with smoothing.
"""
from __future__ import annotations
import numpy as np
from typing import Callable

# Type for an arbitrage-aware call price function C(K) at fixed maturity T and spot S
# Expects forward measure: we pass S, T, r, q for discounting in the surface function
CallPriceFn = Callable[[float], float]


def second_derivative_smooth(xs: np.ndarray, ys: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    """
    Smooth second derivative via Tikhonov regularization on a cubic spline-like finite difference.
    """
    n = len(xs)
    assert n == len(ys) and n >= 5
    h = np.diff(xs)
    # Construct tri-diagonal system for smoothing second derivative
    # Simple approach: penalize curvature of a piecewise-quadratic fit.
    A = np.zeros((n, n))
    b = ys.copy()
    # Identity plus curvature penalty (discrete Laplacian)
    for i in range(n):
        A[i, i] = 1.0 + lam * 2.0
        if i - 1 >= 0:
            A[i, i - 1] = -lam
        if i + 1 < n:
            A[i, i + 1] = -lam
    y_smooth = np.linalg.solve(A, b)
    # Central differences for second derivative
    d2 = np.zeros(n)
    for i in range(1, n - 1):
        d2[i] = 2.0 * (
            (y_smooth[i + 1] - y_smooth[i]) / (xs[i + 1] - xs[i])
            - (y_smooth[i] - y_smooth[i - 1]) / (xs[i] - xs[i - 1])
        ) / (xs[i + 1] - xs[i - 1])
    d2[0] = d2[1]
    d2[-1] = d2[-2]
    return d2


def risk_neutral_density_from_calls(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    discount_factor: float,
) -> np.ndarray:
    """
    f_T(s) = exp(rT) * d^2 C(K)/dK^2 evaluated at K=s when C is undiscounted. Here we pass discount_factor = exp(-rT)
    so density = (1/discount_factor) * d2C_dK2.
    Inputs must be strictly increasing strikes.
    """
    assert np.all(np.diff(strikes) > 0)
    d2 = second_derivative_smooth(strikes, call_prices)
    density = (1.0 / discount_factor) * np.maximum(d2, 0.0)
    # Normalize small numerical errors
    # Use np.trapz for compatibility across NumPy versions
    area = np.trapz(density, strikes)
    if area > 0:
        density /= area
    return density


__all__ = ["risk_neutral_density_from_calls", "second_derivative_smooth"]

