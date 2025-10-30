from __future__ import annotations
from src.screening.volume_oi_anomalies import compute_anomalies_for_chain


def test_anomalies_basic():
    chain = [
        {"strike": 100, "option_type": "call", "volume": 300, "oi": 1200, "hist_volume": [100, 120, 130], "hist_oi": [800, 900, 1000]},
        {"strike": 105, "option_type": "call", "volume": 50, "oi": 400, "hist_volume": [60, 55, 65], "hist_oi": [450, 430, 470]},
    ]
    res = compute_anomalies_for_chain(chain)
    assert res.chain_volume_anomaly is not None and res.chain_volume_anomaly > 1.0
    assert res.chain_oi_anomaly is not None
    assert len(res.per_option) == 2
    assert res.per_option[0]["ratio_vol"] > 2.0  # ~3x volume anomaly


def test_anomalies_edge_zero_variance():
    chain = [
        {"strike": 100, "option_type": "put", "volume": 200, "oi": 1000, "hist_volume": [200, 200, 200], "hist_oi": [1000, 1000, 1000]},
    ]
    res = compute_anomalies_for_chain(chain)
    # z-scores should be None when variance zero; ratios set
    assert res.per_option[0]["z_vol"] is None
    assert res.per_option[0]["z_oi"] is None
    assert res.per_option[0]["ratio_vol"] == 1.0
    assert res.per_option[0]["ratio_oi"] == 1.0

