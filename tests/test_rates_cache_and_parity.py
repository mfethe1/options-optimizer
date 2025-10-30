from __future__ import annotations
import datetime as dt
from src.data.rates.base import RateCache, StaticRateProvider
from src.pricing.rates_dividends.parity import infer_dividend_yield_from_parity
from src.pricing.black_scholes import bs_price


def test_rate_cache_daily_keying():
    p = StaticRateProvider(0.03)
    cache = RateCache(p)
    d1 = dt.date(2025, 1, 2)
    d2 = dt.date(2025, 1, 3)
    r1 = cache.get(d1, 30)
    r2 = cache.get(d1, 30)
    r3 = cache.get(d2, 30)
    assert r1 == r2
    assert r1 == 0.03
    assert r3 == 0.03


def test_parity_inference_with_validation():
    S, K, T, r, q, sigma = 100.0, 100.0, 0.25, 0.02, 0.015, 0.3
    C = bs_price(S, K, T, r, q, sigma, "call")
    P = bs_price(S, K, T, r, q, sigma, "put")
    q_est = infer_dividend_yield_from_parity(S, K, T, r, C, P)
    assert q_est is not None and abs(q_est - q) < 1e-6

