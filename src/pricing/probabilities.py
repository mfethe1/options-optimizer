"""
Probability calculators using risk-neutral density and basic Black-Scholes sanity checks.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple


def probability_itm_from_density(strikes: np.ndarray, density: np.ndarray, strike: float) -> float:
    """
    P(S_T >= K) where density is over strikes grid. Uses trapezoidal integration.
    """
    assert len(strikes) == len(density)
    mask = strikes >= strike
    if not np.any(mask):
        return 0.0
    return float(np.trapz(density[mask], strikes[mask]))


def cdf_from_density(strikes: np.ndarray, density: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (grid, CDF(grid)). Ensures last value is 1 via normalization.
    """
    cdf = np.cumsum(np.concatenate([[0.0], 0.5 * (density[1:] + density[:-1]) * (strikes[1:] - strikes[:-1])]))
    cdf = cdf[1:]
    if cdf[-1] > 0:
        cdf /= cdf[-1]
    return strikes, cdf

