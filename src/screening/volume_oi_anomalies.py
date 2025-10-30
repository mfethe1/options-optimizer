from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import statistics


def safe_mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def safe_stdev(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    try:
        return float(statistics.stdev(xs))
    except Exception:
        return None


@dataclass
class AnomalyResult:
    per_option: List[Dict[str, Any]]
    chain_volume_anomaly: Optional[float]
    chain_oi_anomaly: Optional[float]


def compute_anomalies_for_chain(chain_rows: List[Dict[str, Any]]) -> AnomalyResult:
    """
    chain_rows: list of dicts with keys: strike, option_type, volume, oi, hist_volume (list), hist_oi (list)
    Returns per-option z-scores and multiplicative ratios, plus chain-level aggregates.
    """
    per_opt: List[Dict[str, Any]] = []
    chain_vols: List[float] = []
    chain_hist_vols: List[float] = []
    chain_ois: List[float] = []
    chain_hist_ois: List[float] = []

    for row in chain_rows:
        vol = row.get("volume")
        oi = row.get("oi")
        hvol = row.get("hist_volume") or []
        hoi = row.get("hist_oi") or []
        mu_v = safe_mean(hvol)
        sd_v = safe_stdev(hvol)
        mu_oi = safe_mean(hoi)
        sd_oi = safe_stdev(hoi)

        z_vol = None if (sd_v is None or sd_v == 0) else (vol - mu_v) / sd_v if (vol is not None and mu_v is not None) else None
        z_oi = None if (sd_oi is None or sd_oi == 0) else (oi - mu_oi) / sd_oi if (oi is not None and mu_oi is not None) else None
        ratio_vol = None if (mu_v in (None, 0)) else (vol / mu_v) if vol is not None else None
        ratio_oi = None if (mu_oi in (None, 0)) else (oi / mu_oi) if oi is not None else None

        per_opt.append({
            "strike": row.get("strike"),
            "option_type": row.get("option_type"),
            "z_vol": z_vol,
            "z_oi": z_oi,
            "ratio_vol": ratio_vol,
            "ratio_oi": ratio_oi,
        })

        if vol is not None:
            chain_vols.append(vol)
        if oi is not None:
            chain_ois.append(oi)
        if mu_v is not None:
            chain_hist_vols.append(mu_v)
        if mu_oi is not None:
            chain_hist_ois.append(mu_oi)

    chain_vol_mu = safe_mean(chain_hist_vols)
    chain_oi_mu = safe_mean(chain_hist_ois)
    chain_vol = sum(chain_vols) if chain_vols else None
    chain_oi = sum(chain_ois) if chain_ois else None

    chain_ratio_vol = None if (chain_vol is None or chain_vol_mu in (None, 0)) else chain_vol / chain_vol_mu
    chain_ratio_oi = None if (chain_oi is None or chain_oi_mu in (None, 0)) else chain_oi / chain_oi_mu

    return AnomalyResult(per_option=per_opt, chain_volume_anomaly=chain_ratio_vol, chain_oi_anomaly=chain_ratio_oi)

