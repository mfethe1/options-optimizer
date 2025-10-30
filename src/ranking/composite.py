"""
Composite scoring for options ranking.

Score = w_pop*PoP + w_ev*EV - w_spread*spread_penalty - w_illiquidity*illiquidity_penalty
       + w_ivrank*iv_rank + w_theta*time_decay_efficiency + w_delta*delta_exposure_adj

Weights configurable via dict.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class CompositeWeights:
    # Adjusted defaults to prioritize EV and liquidity, per robust foundation goals
    w_pop: float = 0.30
    w_ev: float = 0.30
    w_spread: float = 0.20
    w_illiquidity: float = 0.12
    w_ivrank: float = 0.05
    w_theta: float = 0.02
    w_delta: float = 0.01


def composite_score(features: Dict[str, float], w: CompositeWeights) -> float:
    pop = features.get("pop", 0.0)
    ev = features.get("expected_value", 0.0)
    spread_penalty = features.get("spread_penalty", 0.0)
    illiq_penalty = features.get("illiq_penalty", 0.0)
    iv_rank = features.get("iv_rank", 0.0)
    theta_eff = features.get("theta_eff", 0.0)
    delta_adj = features.get("delta_adj", 0.0)
    return (
        w.w_pop * pop
        + w.w_ev * ev
        - w.w_spread * spread_penalty
        - w.w_illiquidity * illiq_penalty
        + w.w_ivrank * iv_rank
        + w.w_theta * theta_eff
        + w.w_delta * delta_adj
    )

