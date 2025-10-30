"""
Rate provider interfaces and daily TTL cache.
"""
from __future__ import annotations
import datetime as dt
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple


class RateProvider:
    def get_risk_free_rate(self, as_of: dt.date, tenor_days: int) -> Optional[float]:
        """
        Annualized continuously compounded risk-free rate for the given tenor (in days),
        or None if unavailable.
        """
        raise NotImplementedError


@dataclass
class StaticRateProvider(RateProvider):
    r: float = 0.01

    def get_risk_free_rate(self, as_of: dt.date, tenor_days: int) -> Optional[float]:
        return self.r


@dataclass
class RateCache:
    provider: RateProvider
    store: Dict[Tuple[dt.date, int], float] = field(default_factory=dict)

    def get(self, as_of: dt.date, tenor_days: int) -> Optional[float]:
        key = (as_of, tenor_days)
        if key in self.store:
            return self.store[key]
        val = self.provider.get_risk_free_rate(as_of, tenor_days)
        if val is not None:
            self.store[key] = val
        return val

