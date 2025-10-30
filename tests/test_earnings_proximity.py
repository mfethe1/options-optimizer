from __future__ import annotations
import datetime as dt

from src.screening.earnings_proximity import compute_earnings_proximity


def test_earnings_proximity_days_to(monkeypatch, tmp_path):
    sym = "AAPL"
    today = dt.date(2025, 1, 15)

    # Patch ProviderRouter.earnings_events
    from src.data.providers.router import ProviderRouter
    def fake_events(self, as_of, lookback_days=730, lookahead_days=365):
        return [dt.date(2024, 10, 30), dt.date(2025, 1, 20), dt.date(2025, 4, 20)]
    monkeypatch.setattr(ProviderRouter, "earnings_events", fake_events)

    # Patch ATMIVHistory.load to return simple near-daily entries
    from src.pricing.iv.surface_fit import ATMIVHistory
    def fake_load(self, symbol):
        # Build daily history around events
        days = [dt.date(2024, 10, d) for d in range(20, 31)] + [dt.date(2025, 1, d) for d in range(10, 21)]
        return [(d.isoformat(), 0.25 if d.day >= 25 else 0.20) for d in days]
    monkeypatch.setattr(ATMIVHistory, "load", fake_load)

    # Patch compute_atm_iv_and_rank to current ATM IV
    import src.screening.earnings_proximity as ep
    class Dummy:
        def __init__(self, iv): self.atm_iv = iv; self.iv_rank = 0.0; self.iv_percentile = 0.0
    def fake_compute(symbol, expiry, as_of=None):
        return Dummy(0.30)
    monkeypatch.setattr(ep, "compute_atm_iv_and_rank", fake_compute)

    res = compute_earnings_proximity(sym, dt.date(2025, 2, 28), as_of=today)
    assert res.days_to_earnings == 5
    assert res.pre_earnings_ratio is not None and res.pre_earnings_ratio > 1.0


def test_earnings_proximity_baseline_insufficient(monkeypatch):
    sym = "MSFT"
    today = dt.date(2025, 1, 15)
    from src.data.providers.router import ProviderRouter
    def fake_events(self, as_of, lookback_days=730, lookahead_days=365):
        return []
    monkeypatch.setattr(ProviderRouter, "earnings_events", fake_events)

    res = compute_earnings_proximity(sym, dt.date(2025, 2, 28), as_of=today)
    assert res.pre_earnings_ratio is None

