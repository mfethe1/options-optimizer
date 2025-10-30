from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import datetime as dt

from src.data.providers.router import ProviderRouter
from src.screening.hv import hv_annualized_from_prices
from src.screening.iv_surface_factor import compute_atm_iv_and_rank, ATMIVResult


@dataclass
class IVHVResult:
    hv30: float
    hv60: float
    atm_iv: Optional[float]
    ratio30: Optional[float]
    ratio60: Optional[float]


def compute_iv_hv_ratio(symbol: str, expiry: dt.date, as_of: dt.date | None = None) -> IVHVResult:
    as_of = as_of or dt.date.today()
    pr = ProviderRouter(symbol)
    prices_90 = pr.daily_prices_window(90)
    hv30 = hv_annualized_from_prices(prices_90, 30)
    hv60 = hv_annualized_from_prices(prices_90, 60)

    iv_res: ATMIVResult = compute_atm_iv_and_rank(symbol, expiry, as_of)
    atm_iv = iv_res.atm_iv

    ratio30 = None
    ratio60 = None
    if atm_iv is not None:
        if hv30 > 1e-9:
            ratio30 = atm_iv / hv30
        if hv60 > 1e-9:
            ratio60 = atm_iv / hv60

    return IVHVResult(hv30=hv30, hv60=hv60, atm_iv=atm_iv, ratio30=ratio30, ratio60=ratio60)

