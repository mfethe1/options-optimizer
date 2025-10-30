from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.config import load_config
from src.data.http_client import CachingHttpClient, RateLimiter

MARKETSTACK_URL = "http://api.marketstack.com/v1"


@dataclass
class MarketstackProvider:
    api_key: Optional[str] = None

    def __post_init__(self):
        cfg = load_config()
        self.api_key = self.api_key or cfg.marketstack_key
        self.client = CachingHttpClient(cfg.cache_dir)
        self.rate_limiter = RateLimiter(cfg.marketstack_rpm)

    def _get(self, path: str, params: Dict[str, Any], ttl_seconds: int = 600) -> Dict[str, Any]:
        p = {"access_key": self.api_key}
        p.update(params)
        url = f"{MARKETSTACK_URL}/{path}"
        return self.client.get_json(url, p, self.rate_limiter, ttl_seconds=ttl_seconds)

    def eod(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        return self._get("eod", {"symbols": symbol, "limit": limit}, ttl_seconds=600)

