from __future__ import annotations
import datetime as dt
from src.pricing.iv.surface_fit import IVPoint, fit_svi_from_iv_points, atm_iv_from_svi
import math


def test_atm_iv_from_synthetic_points():
    S = 100.0
    T = 30/365
    # Synthetic: flat IV 20%
    points = [IVPoint(strike, 0.2) for strike in [80, 90, 95, 100, 105, 110, 120]]
    # forward ~ S if r=q=0
    params = fit_svi_from_iv_points(points, forward=S, T=T)
    assert params is not None
    atm = atm_iv_from_svi(params, T)
    assert atm is not None
    assert abs(atm - 0.2) < 1e-3

