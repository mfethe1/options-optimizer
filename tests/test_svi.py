from __future__ import annotations
import numpy as np
from src.pricing.iv.svi import SVIParams, svi_total_var, fit_svi_smile, basic_static_arbitrage_checks, calendar_arbitrage_check


def test_fit_svi_on_synthetic_smile():
    k = np.linspace(-1.0, 1.0, 41)
    true_p = SVIParams(a=0.04, b=0.4, rho=-0.3, m=0.0, sigma=0.5)
    w = svi_total_var(k, true_p)
    est = fit_svi_smile(k, w + 1e-4*np.random.RandomState(0).randn(*w.shape))
    assert basic_static_arbitrage_checks(k, est)
    w_est = svi_total_var(k, est)
    mse = float(np.mean((w_est - w)**2))
    assert mse < 1e-3


def test_calendar_arbitrage_check():
    k = np.linspace(-1.0, 1.0, 41)
    p_short = SVIParams(a=0.02, b=0.3, rho=-0.2, m=0.0, sigma=0.4)
    p_long = SVIParams(a=0.05, b=0.35, rho=-0.2, m=0.0, sigma=0.5)
    assert calendar_arbitrage_check(k, p_short, p_long)

