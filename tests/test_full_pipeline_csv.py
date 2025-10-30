from __future__ import annotations
import csv
import io
import datetime as dt

from src.screening.full_pipeline import build_rows_for_universe


def test_full_pipeline_csv_schema(monkeypatch):
    # Universe with two symbols
    universe = ["AAPL", "MSFT"]
    today = dt.date(2025, 1, 15)

    # Patch ProviderRouter to avoid network
    from src.data.providers.router import ProviderRouter
    def fake_expirations(self):
        return [dt.date(2025, 2, 21)]
    def fake_daily(self, days):
        return [{"date": str(today - dt.timedelta(days=i)), "close": 100 + i, "volume": 1000000} for i in range(30)]
    def fake_market_cap(self):
        return 1e12
    def fake_options_chain(self, expiry, side):
        # Minimal options with basic fields
        class Opt: pass
        out = []
        for k in [90, 95, 100, 105, 110]:
            o = Opt()
            o.strike = k
            o.option_type = side
            o.volume = 1000
            o.open_interest = 5000
            o.bid = 1.0
            o.ask = 1.2
            o.last_price = 1.1
            out.append(o)
        return out
    monkeypatch.setattr(ProviderRouter, "expirations", fake_expirations)
    monkeypatch.setattr(ProviderRouter, "daily_prices_window", fake_daily)
    monkeypatch.setattr(ProviderRouter, "market_cap", fake_market_cap)
    monkeypatch.setattr(ProviderRouter, "options_chain", fake_options_chain)

    # Patch ATM IV and ratios to deterministic
    import src.screening.iv_surface_factor as ivf
    class DummyIV:
        def __init__(self): self.atm_iv=0.3; self.iv_rank=70.0; self.iv_percentile=75.0
    monkeypatch.setattr(ivf, "compute_atm_iv_and_rank", lambda s,e,a=None: DummyIV())

    import src.screening.iv_hv_ratio as ivhv
    class DummyR:
        def __init__(self): self.hv30=0.2; self.hv60=0.18; self.atm_iv=0.3; self.ratio30=1.5; self.ratio60=1.67
    monkeypatch.setattr(ivhv, "compute_iv_hv_ratio", lambda s,e,a=None: DummyR())

    import src.screening.earnings_proximity as ep
    class DummyE:
        def __init__(self): self.days_to_earnings=10; self.pre_earnings_ratio=1.2
    monkeypatch.setattr(ep, "compute_earnings_proximity", lambda s,e,a=None: DummyE())

    # Patch momentum pipeline
    import src.screening.pipeline as pipe
    monkeypatch.setattr(pipe, "add_momentum_to_rows", lambda rows, interval="daily": [dict(r, momentum_score=60.0) for r in rows])

    rows = build_rows_for_universe(universe, as_of=today, top=10)

    # Validate schema and content
    cols = {"symbol","adv","market_cap","iv_rank","iv_percentile","iv_hv_ratio","volume_anomaly","oi_anomaly","pre_earnings_ratio","momentum_score","composite_score","rationale"}
    for r in rows:
        assert set(r.keys()) >= cols
        assert r["symbol"] in universe
        assert r["iv_hv_ratio"] == 1.5
        assert r["pre_earnings_ratio"] == 1.2
        assert r["momentum_score"] == 60.0
        assert isinstance(r["rationale"], str) and len(r["rationale"]) > 0

    # CSV writing sanity
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "symbol","adv","market_cap","iv_rank","iv_percentile","iv_hv_ratio","volume_anomaly","oi_anomaly","pre_earnings_ratio","momentum_score","composite_score","rationale"
    ])
    writer.writeheader()
    for r in rows:
        writer.writerow({
            "symbol": r["symbol"],
            "adv": r.get("adv"),
            "market_cap": r.get("market_cap"),
            "iv_rank": r.get("iv_rank"),
            "iv_percentile": r.get("iv_percentile"),
            "iv_hv_ratio": r.get("iv_hv_ratio"),
            "volume_anomaly": r.get("volume_anomaly"),
            "oi_anomaly": r.get("oi_anomaly"),
            "pre_earnings_ratio": r.get("pre_earnings_ratio"),
            "momentum_score": r.get("momentum_score"),
            "composite_score": r.get("composite_score"),
            "rationale": r.get("rationale"),
        })
    assert len(output.getvalue().splitlines()) == len(rows) + 1

