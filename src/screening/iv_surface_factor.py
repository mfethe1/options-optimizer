from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import datetime as dt
import math

from src.pricing.iv.surface_fit import IVPoint, fit_svi_from_iv_points, atm_iv_from_svi, ATMIVHistory
from src.pricing.iv.implied_vol import implied_vol
from src.data.providers.router import ProviderRouter


@dataclass
class ATMIVResult:
    atm_iv: Optional[float]
    iv_rank: float
    iv_percentile: float


def compute_atm_iv_and_rank(symbol: str, expiry: dt.date, as_of: dt.date | None = None) -> ATMIVResult:
    """
    Fetch a single expiry chain, compute per-strike IVs, fit SVI, and derive ATM IV and IV Rank/Percentile.
    Uses provider router (with fallback) for chain and underlying.
    """
    as_of = as_of or dt.date.today()
    pr = ProviderRouter(symbol)
    S = pr.underlying_price()
    if S is None or S <= 0:
        return ATMIVResult(None, 0.0, 0.0)
    # risk-free and q can be refined via Fred + parity later in this pipeline
    r, q = 0.0, 0.0
    # load calls near expiry
    calls = pr.options_chain(expiry, "call")
    if not calls:
        return ATMIVResult(None, 0.0, 0.0)
    # Build points for near-money strikes
    points: List[IVPoint] = []
    T = max(1e-6, (expiry - as_of).days / 365.0)
    forward = S * math.exp((r - q) * T)
    for c in calls:
        bid = getattr(c, "bid", None)
        ask = getattr(c, "ask", None)
        last_price = getattr(c, "last_price", None)
        strike = getattr(c, "strike", None)
        if strike is None:
            continue
        mid = None
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            mid = 0.5 * (bid + ask)
        elif last_price is not None and last_price > 0:
            mid = last_price
        if mid is None or mid <= 0:
            continue
        iv = implied_vol(mid, S, strike, T, r, q, "call")
        if iv is None:
            continue
        points.append(IVPoint(strike, iv))
    if len(points) < 5:
        return ATMIVResult(None, 0.0, 0.0)

    params = fit_svi_from_iv_points(points, forward, T)
    if not params:
        return ATMIVResult(None, 0.0, 0.0)
    atm = atm_iv_from_svi(params, T)
    if atm is None:
        return ATMIVResult(None, 0.0, 0.0)

    history = ATMIVHistory()
    history.append(symbol, as_of.isoformat(), atm)
    iv_rank, iv_pct = history.rank_percentile(symbol, atm)
    return ATMIVResult(atm, iv_rank, iv_pct)

