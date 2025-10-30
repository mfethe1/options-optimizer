from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np

# Factors utilities and rationale builder

@dataclass
class FactorsConfig:
    iv_rank_window_days: int = 252
    hv_window_days: int = 30
    vol_anom_window_days: int = 30


def iv_rank_percentile(iv_series: List[float], current_iv: float) -> Tuple[float, float]:
    if not iv_series:
        return 0.0, 0.0
    low, high = float(min(iv_series)), float(max(iv_series))
    rank = 0.0 if high == low else 100.0 * (current_iv - low) / (high - low)
    pct = 100.0 * (sum(1 for x in iv_series if x <= current_iv) / len(iv_series))
    return rank, pct


def iv_hv_ratio(current_iv: float, hv: float) -> float:
    if hv <= 1e-9:
        return 0.0
    return float(current_iv / hv)


def anomaly_zscore(value: float, hist: List[float]) -> float:
    if not hist:
        return 0.0
    mu = float(np.mean(hist))
    sigma = float(np.std(hist))
    if sigma < 1e-9:
        return 0.0
    return float((value - mu) / sigma)


def build_rationale(row: Dict[str, Any]) -> str:
    parts = []
    # IV level and regime
    pct = row.get("iv_percentile")
    if isinstance(pct, (int, float)):
        if pct >= 85:
            parts.append(f"Very high IV percentile ({pct:.0f}%)")
        elif pct >= 75:
            parts.append(f"High IV percentile ({pct:.0f}%)")
        elif pct <= 25:
            parts.append(f"Low IV percentile ({pct:.0f}%)")
    # IV vs HV
    ivhv = row.get("iv_hv_ratio")
    if isinstance(ivhv, (int, float)):
        if ivhv >= 1.5:
            parts.append(f"IV rich vs HV ({ivhv:.2f}x)")
        elif ivhv <= 0.7:
            parts.append(f"IV cheap vs HV ({ivhv:.2f}x)")
    # Earnings proximity
    pe = row.get("pre_earnings_ratio")
    if isinstance(pe, (int, float)) and pe >= 1.2:
        parts.append(f"Earnings soon (pre-earnings IV {pe:.2f}x)")
    # Flow anomalies
    va = row.get("volume_anomaly")
    if isinstance(va, (int, float)) and va >= 2.0:
        parts.append(f"Volume spike +{(va-1)*100:.0f}%")
    oi = row.get("oi_anomaly")
    if isinstance(oi, (int, float)) and oi >= 1.5:
        parts.append(f"OI surge +{(oi-1)*100:.0f}%")
    # Momentum
    ms = row.get("momentum_score")
    if isinstance(ms, (int, float)):
        if ms >= 65:
            parts.append("Strong positive momentum")
        elif ms >= 55:
            parts.append("Positive momentum")
        elif ms <= 45:
            parts.append("Negative momentum")
    parts = [p for p in parts if p]
    return "; ".join(parts) if parts else "Meets liquidity and volatility criteria"
