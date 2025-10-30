from __future__ import annotations
import math

from src.pricing.black_scholes import bs_price
from src.pricing.iv.implied_vol import implied_vol


def test_implied_vol_roundtrip_call():
    S, K, T, r, q, sigma = 100.0, 100.0, 0.5, 0.01, 0.0, 0.25
    price = bs_price(S, K, T, r, q, sigma, "call")
    iv = implied_vol(price, S, K, T, r, q, "call")
    assert iv is not None
    assert abs(iv - sigma) < 1e-4


def test_implied_vol_roundtrip_put():
    S, K, T, r, q, sigma = 100.0, 90.0, 1.0, 0.02, 0.01, 0.35
    price = bs_price(S, K, T, r, q, sigma, "put")
    iv = implied_vol(price, S, K, T, r, q, "put")
    assert iv is not None
    assert abs(iv - sigma) < 1e-4


def test_implied_vol_bounds_invalid_price():
    S, K, T, r, q = 100.0, 100.0, 0.5, 0.01, 0.0
    # absurd price above upper bound
    price = 1e6
    assert implied_vol(price, S, K, T, r, q, "call") is None

