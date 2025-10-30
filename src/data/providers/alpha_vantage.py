from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.config import load_config
from src.data.http_client import CachingHttpClient, RateLimiter

ALPHA_URL = "https://www.alphavantage.co/query"


@dataclass
class AlphaVantageProvider:
    api_key: Optional[str] = None

    def __post_init__(self):
        cfg = load_config()
        self.api_key = self.api_key or cfg.alpha_vantage_key
        self.client = CachingHttpClient(cfg.cache_dir)
        self.rate_limiter = RateLimiter(cfg.alpha_vantage_rpm)

    def _get(self, function: str, params: Dict[str, Any], ttl_seconds: int = 60) -> Dict[str, Any]:
        p = {"function": function, "apikey": self.api_key}
        p.update(params)
        return self.client.get_json(ALPHA_URL, p, self.rate_limiter, ttl_seconds=ttl_seconds)

    def daily_adjusted(self, symbol: str) -> List[Dict[str, Any]]:
        data = self._get("TIME_SERIES_DAILY_ADJUSTED", {"symbol": symbol, "outputsize": "compact"}, ttl_seconds=600)
        ts = data.get("Time Series (Daily)", {})
        out = []
        for d, row in ts.items():
            # Alpha Vantage keys: 5. adjusted close, 6. volume
            close = row.get("5. adjusted close") or row.get("4. close")
            vol = row.get("6. volume")
            try:
                close_f = float(close)
            except Exception:
                continue
            try:
                vol_i = int(vol)
            except Exception:
                vol_i = None
            out.append({"date": d, "close": close_f, "volume": vol_i})
        out.sort(key=lambda x: x["date"])
        return out

    # Placeholder for options chain endpoint mapping (Alpha Vantage options support varies)
    # Implement when stable endpoint is identified / mapped
    def earnings(self, symbol: str) -> Dict[str, Any]:
        data = self._get("EARNINGS", {"symbol": symbol}, ttl_seconds=3600)
        return data

