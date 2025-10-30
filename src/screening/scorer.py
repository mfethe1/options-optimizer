from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

from src.ranking.composite import composite_score, CompositeWeights


@dataclass
class ScreenWeights(CompositeWeights):
    # reuse CompositeWeights for now; can extend
    pass


def score_row(row: Dict[str, Any], w: ScreenWeights) -> float:
    adv = row.get("adv")
    try:
        adv_millions = float(adv) / 1e6 if adv is not None else 0.0
    except Exception:
        adv_millions = 0.0
    features = {
        "pop": row.get("iv_percentile", 0.0) / 100.0,
        "expected_value": row.get("expected_value", 0.0),
        "spread_penalty": max(0.0, row.get("spread_bps", 0.0) / 10000.0),
        "illiq_penalty": 1.0 / max(1.0, adv_millions),
        "iv_rank": row.get("iv_rank", 0.0) / 100.0,
        "theta_eff": row.get("theta_eff", 0.0),
        "delta_adj": row.get("delta_adj", 0.0),
    }
    return composite_score(features, w)

