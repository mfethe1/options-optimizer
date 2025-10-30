from __future__ import annotations
import datetime as dt

from src.screening.iv_hv_ratio import compute_iv_hv_ratio
from src.pricing.iv.surface_fit import IVPoint, fit_svi_from_iv_points, atm_iv_from_svi
from src.screening.hv import hv_annualized_from_prices


def test_hv_zero_edge_case():
    rows = [{"close": 100.0} for _ in range(40)]
    hv = hv_annualized_from_prices(rows, 30)
    assert hv == 0.0


def test_iv_hv_ratio_handles_missing_iv(monkeypatch):
    # Monkeypatch compute_atm_iv_and_rank to return None
    import src.screening.iv_hv_ratio as mod
    class Dummy:
        atm_iv = None
        iv_rank = 0.0
        iv_percentile = 0.0
    def fake_compute(*args, **kwargs):
        return Dummy()
    # Patch the imported symbol within iv_hv_ratio module scope
    monkeypatch.setattr("src.screening.iv_hv_ratio.compute_atm_iv_and_rank", fake_compute)

    res = compute_iv_hv_ratio("AAPL", dt.date.today())
    assert res.ratio30 is None and res.ratio60 is None


def test_iv_hv_ratio_with_synthetic_prices_and_iv(monkeypatch):
    # Build synthetic upward drift prices to produce non-zero HV
    rows = [{"close": 100.0 + i*0.5} for i in range(100)]

    import src.screening.iv_hv_ratio as mod
    from src.data.providers.router import ProviderRouter
    def fake_window(self, days):
        return rows
    monkeypatch.setattr(ProviderRouter, "daily_prices_window", fake_window)

    # Monkeypatch ATM IV to be fixed 0.2
    class Dummy:
        atm_iv = 0.2
        iv_rank = 0.0
        iv_percentile = 0.0
    def fake_compute(*args, **kwargs):
        return Dummy()
    # Patch the imported symbol within iv_hv_ratio module scope
    monkeypatch.setattr("src.screening.iv_hv_ratio.compute_atm_iv_and_rank", fake_compute)

    res = compute_iv_hv_ratio("AAPL", dt.date.today())
    assert res.hv30 >= 0.0 and res.hv60 >= 0.0
    if res.hv30 > 1e-9:
        assert res.ratio30 is not None and res.ratio30 > 0

