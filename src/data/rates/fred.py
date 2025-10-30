from __future__ import annotations
import datetime as dt
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

from src.config import load_config
from src.data.http_client import CachingHttpClient, RateLimiter

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
# Series ID for 1-Month Treasury Bill: DGS1MO (note: percent per annum; convert to continuous)
SERIES_DGS1MO = "DGS1MO"


@dataclass
class FredRateProvider:
    api_key: Optional[str] = None

    def __post_init__(self):
        cfg = load_config()
        self.api_key = self.api_key or cfg.fred_key
        self.client = CachingHttpClient(cfg.cache_dir)
        self.rate_limiter = RateLimiter(cfg.fred_rpm)

    def get_risk_free_rate(self, as_of: dt.date, tenor_days: int) -> Optional[float]:
        # Fetch last available observation on/before as_of
        params = {
            "series_id": SERIES_DGS1MO,
            "api_key": self.api_key or "",
            "file_type": "json",
            "observation_start": (as_of - dt.timedelta(days=120)).isoformat(),
            "observation_end": as_of.isoformat(),
            "sort_order": "desc",
            "limit": 1,
        }
        data = self.client.get_json(FRED_URL, params, self.rate_limiter, ttl_seconds=24 * 3600)
        obs = data.get("observations", [])
        if not obs:
            return None
        value = obs[0].get("value")
        try:
            pct = float(value)
        except Exception:
            return None
        # Convert percent (simple annual) to continuous compounding
        r_simple = pct / 100.0
        # Map tenor: for simplicity use the 1M rate for all short tenors
        r_cont = float(os.getenv("RATE_CONTINUOUS_OVERRIDE", "0"))
        if r_cont:
            return r_cont
        # convert simple annual to continuous: r_cont = ln(1 + r_simple)
        return float(__import__("math").log1p(r_simple))

