from __future__ import annotations
from src.screening.hv import hv_annualized_from_prices


def test_hv_with_insufficient_data():
    assert hv_annualized_from_prices([], 30) == 0.0


def test_hv_basic():
    rows = [{"close": 100.0 + i} for i in range(40)]
    hv = hv_annualized_from_prices(rows, 30)
    assert hv >= 0.0

