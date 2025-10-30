from __future__ import annotations
import os
from src.config import load_config


def test_env_config_defaults_and_overrides(monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "TEST_ALPHA")
    monkeypatch.setenv("FINNHUB_API_KEY", "TEST_FINN")
    monkeypatch.setenv("MARKETSTACK_API_KEY", "TEST_MS")
    monkeypatch.setenv("FMP_API_KEY", "TEST_FMP")
    monkeypatch.setenv("ALPHA_VANTAGE_RPM", "10")
    cfg = load_config()
    assert cfg.alpha_vantage_key == "TEST_ALPHA"
    assert cfg.finnhub_key == "TEST_FINN"
    assert cfg.marketstack_key == "TEST_MS"
    assert cfg.fmp_key == "TEST_FMP"
    assert cfg.alpha_vantage_rpm == 10

