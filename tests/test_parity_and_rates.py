from __future__ import annotations
import math

from src.pricing.black_scholes import bs_price
from src.pricing.rates_dividends.parity import infer_dividend_yield_from_parity


def test_parity_dividend_inference_roundtrip():
    # Synthetic BS world where we know q
    S, K, T, r, q, sigma = 100.0, 100.0, 0.5, 0.02, 0.03, 0.25
    C = bs_price(S, K, T, r, q, sigma, "call")
    P = bs_price(S, K, T, r, q, sigma, "put")
    q_est = infer_dividend_yield_from_parity(S, K, T, r, C, P)
    assert q_est is not None
    assert abs(q_est - q) < 1e-6


def test_parity_invalid_inputs():
    assert infer_dividend_yield_from_parity(0, 100, 0.5, 0.01, 10, 9) is None

