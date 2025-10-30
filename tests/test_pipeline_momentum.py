from __future__ import annotations
from src.screening.pipeline import add_momentum_to_rows


def test_add_momentum_with_mock(monkeypatch):
    rows = [{"symbol": "AAPL"}, {"symbol": "MSFT"}, {"symbol": None}]

    # Patch MomentumComputer to avoid network
    import src.screening.pipeline as pipe
    class DummyMC:
        def fetch_indicators(self, symbol, interval="daily"):
            if symbol == "AAPL":
                return {"rsi14": 60, "rsi30": 55, "roc10": 2.0, "roc20": 3.0, "macd": 0.5}
            if symbol == "MSFT":
                return {"rsi14": None, "rsi30": None, "roc10": None, "roc20": None, "macd": None}
            raise RuntimeError("unexpected symbol")
        def momentum_score(self, indicators, w):
            # return simple average of normalized placeholders to keep test deterministic
            from src.screening.momentum import MomentumComputer as MC
            return MC().momentum_score(indicators, w)
    monkeypatch.setattr(pipe, "MomentumComputer", lambda: DummyMC())

    out = add_momentum_to_rows(rows)
    assert out[0]["momentum_score"] is not None and 0.0 <= out[0]["momentum_score"] <= 100.0
    assert out[1]["momentum_score"] == 50.0  # neutral fallback when all None
    assert out[2]["momentum_score"] is None

