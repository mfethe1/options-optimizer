from __future__ import annotations
import numpy as np
import math

from src.pricing.black_scholes import bs_price
from src.pricing.rnd.breeden_litzenberger import risk_neutral_density_from_calls
from src.pricing.probabilities import probability_itm_from_density, cdf_from_density


def test_density_integrates_to_one_under_flat_bs():
    # Construct call prices under flat IV and compute density; should integrate to ~1
    S, r, q, T, sigma = 100.0, 0.01, 0.0, 0.5, 0.2
    strikes = np.linspace(50, 150, 301)
    calls = np.array([bs_price(S, K, T, r, q, sigma, "call") for K in strikes])
    df = math.exp(-r * T)
    density = risk_neutral_density_from_calls(strikes, calls, df)
    area = np.trapz(density, strikes)
    assert abs(area - 1.0) < 1e-3


def test_pop_matches_bs_tail_prob_flat_iv():
    S, r, q, T, sigma = 100.0, 0.01, 0.0, 0.5, 0.2
    strikes = np.linspace(60, 140, 321)
    calls = np.array([bs_price(S, K, T, r, q, sigma, "call") for K in strikes])
    df = math.exp(-r * T)
    density = risk_neutral_density_from_calls(strikes, calls, df)

    # Compare P(S_T>=K) from density to Black-Scholes N(d2)
    def bs_tail_prob(K: float):
        # N(d2) with dividend yield q
        import math
        from src.pricing.black_scholes import _d1, _d2, _norm_cdf  # type: ignore
        d1 = _d1(S, K, T, r, q, sigma)
        d2 = _d2(d1, sigma, T)
        return _norm_cdf(d2)

    for K in [80, 90, 100, 110, 120]:
        p_est = probability_itm_from_density(strikes, density, K)
        p_bs = bs_tail_prob(K)
        assert abs(p_est - p_bs) < 2e-2

