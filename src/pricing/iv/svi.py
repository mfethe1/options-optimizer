from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float


def svi_total_var(k: np.ndarray, p: SVIParams) -> np.ndarray:
    # Raw SVI (Gatheral): w(k) = a + b*( rho*(k-m) + sqrt((k-m)^2 + sigma^2) )
    x = k - p.m
    return p.a + p.b * (p.rho * x + np.sqrt(x * x + p.sigma * p.sigma))


def _random_params(rng: np.random.RandomState) -> SVIParams:
    a = rng.uniform(1e-3, 0.3)
    b = rng.uniform(0.05, 2.0)
    rho = rng.uniform(-0.9, 0.9)
    m = rng.uniform(-0.5, 0.5)
    sigma = rng.uniform(0.05, 1.0)
    return SVIParams(a, b, rho, m, sigma)


def fit_svi_smile(k: np.ndarray, w: np.ndarray, n_starts: int = 128, iters: int = 400) -> SVIParams:
    """
    Simple randomized local search to fit SVI parameters to total variance data w(k).
    Avoids SciPy dependency for portability. Tightened priors and longer search for stability.
    """
    rng = np.random.RandomState(42)

    def loss(p: SVIParams) -> float:
        pred = svi_total_var(k, p)
        # L2 loss with small ridge to discourage extreme params
        return float(np.mean((pred - w) ** 2) + 1e-6 * (p.a**2 + p.b**2 + p.rho**2 + p.m**2 + p.sigma**2))

    best_p = None
    best_l = float("inf")

    for _ in range(n_starts):
        p = _random_params(rng)
        l = loss(p)
        if l < best_l:
            best_p, best_l = p, l

    # coordinate descent refinements
    for _ in range(iters):
        improved = False
        for name, scale in [("a", 0.01), ("b", 0.02), ("rho", 0.01), ("m", 0.01), ("sigma", 0.01)]:
            for d in (-1.0, 1.0):
                p = SVIParams(best_p.a, best_p.b, best_p.rho, best_p.m, best_p.sigma)
                val = getattr(p, name)
                new_val = val + d * scale
                # parameter bounds
                if name == "a":
                    new_val = min(max(1e-6, new_val), 1.0)
                elif name == "b":
                    new_val = min(max(1e-6, new_val), 5.0)
                elif name == "rho":
                    new_val = min(max(-0.999, new_val), 0.999)
                elif name == "m":
                    new_val = min(max(-2.0, new_val), 2.0)
                elif name == "sigma":
                    new_val = min(max(1e-5, new_val), 3.0)
                setattr(p, name, new_val)
                l = loss(p)
                if l + 1e-12 < best_l:
                    best_p, best_l = p, l
                    improved = True
        if not improved:
            break

    return best_p  # type: ignore


def basic_static_arbitrage_checks(k: np.ndarray, p: SVIParams) -> bool:
    # Basic parameter sanity: b>0, sigma>0, |rho|<1
    if not (p.b > 0 and p.sigma > 0 and abs(p.rho) < 1):
        return False
    # Ensure total variance is positive and reasonably convex numerically
    w = svi_total_var(k, p)
    if np.any(w <= 0):
        return False
    # Check discrete convexity in k (second finite diff >= 0) on central region
    if len(k) >= 5:
        d2 = w[:-2] - 2 * w[1:-1] + w[2:]
        if np.any(d2 < -1e-3):
            return False
    return True


def calendar_arbitrage_check(k: np.ndarray, p_short: SVIParams, p_long: SVIParams) -> bool:
    # For same k, total variance should increase with maturity (w_long >= w_short)
    w_s = svi_total_var(k, p_short)
    w_l = svi_total_var(k, p_long)
    return bool(np.all(w_l + 1e-8 >= w_s))


__all__ = ["SVIParams", "svi_total_var", "fit_svi_smile", "basic_static_arbitrage_checks", "calendar_arbitrage_check"]

