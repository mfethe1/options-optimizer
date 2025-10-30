from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import json
import os
import numpy as np

from .svi import SVIParams, svi_total_var, fit_svi_smile, basic_static_arbitrage_checks


@dataclass
class IVPoint:
    strike: float
    iv: float  # Black-Scholes implied volatility for this strike/expiry


def _log_moneyness(strike: float, forward: float) -> float:
    return math.log(strike / forward)


def fit_svi_from_iv_points(
    points: List[IVPoint],
    forward: float,
    T: float,
) -> Optional[SVIParams]:
    if T <= 0 or forward <= 0 or not points:
        return None
    # Build k and total variance
    k = np.array([_log_moneyness(p.strike, forward) for p in points], dtype=float)
    ivs = np.array([max(1e-6, float(p.iv)) for p in points], dtype=float)
    w = (ivs ** 2) * T
    # Sort by k (important for convexity checks)
    idx = np.argsort(k)
    k = k[idx]
    w = w[idx]
    if k.size < 5:
        return None
    # Special-case: near-constant total variance -> flat smile
    w_mean = float(np.mean(w))
    w_std = float(np.std(w))
    if w_mean > 0 and w_std / w_mean < 1e-6:
        return SVIParams(a=w_mean, b=1e-6, rho=0.0, m=0.0, sigma=0.1)

    params = fit_svi_smile(k, w)
    if not basic_static_arbitrage_checks(k, params):
        return None
    return params


def atm_iv_from_svi(params: SVIParams, T: float) -> Optional[float]:
    if T <= 0:
        return None
    w0 = float(svi_total_var(np.array([0.0]), params)[0])
    if w0 <= 0:
        return None
    return math.sqrt(w0 / T)


# Simple JSON cache for ATM IV history per symbol
class ATMIVHistory:
    def __init__(self, cache_dir: str = os.path.join("data", "cache", "atm_iv_history")):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, symbol: str) -> str:
        return os.path.join(self.cache_dir, f"{symbol.upper()}.json")

    def load(self, symbol: str) -> List[Tuple[str, float]]:
        path = self._path(symbol)
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [(str(d[0]), float(d[1])) for d in data]
        except Exception:
            return []

    def append(self, symbol: str, as_of: str, atm_iv: float, max_len: int = 300) -> None:
        items = self.load(symbol)
        items = [it for it in items if it[0] != as_of]
        items.append((as_of, float(atm_iv)))
        items = items[-max_len:]
        with open(self._path(symbol), "w", encoding="utf-8") as f:
            json.dump(items, f)

    def rank_percentile(self, symbol: str, current_iv: float, window: int = 252) -> Tuple[float, float]:
        items = self.load(symbol)
        if not items:
            return 0.0, 0.0
        vals = [v for _, v in items[-window:]]
        if not vals:
            return 0.0, 0.0
        low, high = float(min(vals)), float(max(vals))
        rank = 0.0 if high == low else 100.0 * (current_iv - low) / (high - low)
        pct = 100.0 * (sum(1 for x in vals if x <= current_iv) / len(vals))
        return rank, pct

