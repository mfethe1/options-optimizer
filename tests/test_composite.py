from __future__ import annotations
from src.ranking.composite import composite_score, CompositeWeights


def test_composite_weighting_and_signs():
    w = CompositeWeights()
    f1 = dict(pop=0.6, expected_value=0.2, spread_penalty=0.1, illiq_penalty=0.05, iv_rank=0.4, theta_eff=0.1, delta_adj=0.0)
    f2 = dict(pop=0.55, expected_value=0.25, spread_penalty=0.02, illiq_penalty=0.02, iv_rank=0.1, theta_eff=0.05, delta_adj=0.0)
    s1 = composite_score(f1, w)
    s2 = composite_score(f2, w)
    # Lower spread/illiq and higher EV should help f2
    assert s2 > s1 or abs(s2 - s1) < 1e-9

