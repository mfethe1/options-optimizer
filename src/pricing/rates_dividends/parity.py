"""
Dividend yield inference from put-call parity per expiry.
C - P = S*e^{-qT} - K*e^{-rT}  =>  q = - (1/T) * ln( (C - P + K e^{-rT}) / S )
Includes guards and fallback behavior.
"""
from __future__ import annotations
import math
from typing import Optional


def infer_dividend_yield_from_parity(
    S: float,
    K: float,
    T: float,
    r: float,
    call_price: float,
    put_price: float,
) -> Optional[float]:
    if T <= 0 or S <= 0 or K <= 0:
        return None
    df_r = math.exp(-r * T)
    numerator = call_price - put_price + K * df_r
    if numerator <= 1e-12:
        return None
    ratio = numerator / S
    if ratio <= 0:
        return None
    q = -math.log(ratio) / T
    # Basic sanity bounds for q (e.g., -50%..+50% annualized)
    if not (-0.5 <= q <= 0.5):
        return None
    return q

