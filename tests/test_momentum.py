from __future__ import annotations
from src.screening.momentum import MomentumComputer, MomentumWeights


def test_normalization_bounds():
    mc = MomentumComputer()
    # No network call for normalization tests; call static normalize
    assert 0.0 <= mc.normalize_indicator("rsi14", 120.0) <= 100.0
    assert 0.0 <= mc.normalize_indicator("roc10", 50.0) <= 100.0
    assert 0.0 <= mc.normalize_indicator("macd", 10.0) <= 100.0


def test_momentum_score_neutral_when_missing():
    mc = MomentumComputer()
    indicators = {"rsi14": None, "rsi30": None, "roc10": None, "roc20": None, "macd": None}
    score = mc.momentum_score(indicators, MomentumWeights())
    assert 0.0 <= score <= 100.0

