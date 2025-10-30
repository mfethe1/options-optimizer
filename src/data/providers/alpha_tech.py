from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.config import load_config
from src.data.http_client import CachingHttpClient, RateLimiter

ALPHA_URL = "https://www.alphavantage.co/query"


@dataclass
class AlphaTechProvider:
    api_key: Optional[str] = None

    def __post_init__(self):
        cfg = load_config()
        self.api_key = self.api_key or cfg.alpha_vantage_key
        self.client = CachingHttpClient(cfg.cache_dir)
        self.rate_limiter = RateLimiter(cfg.alpha_vantage_rpm)

    def _get(self, function: str, symbol: str, params: Dict[str, Any], ttl_seconds: int = 900) -> Dict[str, Any]:
        p = {"function": function, "symbol": symbol, "apikey": self.api_key}
        p.update(params)
        return self.client.get_json(ALPHA_URL, p, self.rate_limiter, ttl_seconds=ttl_seconds)

    def rsi(self, symbol: str, interval: str, time_period: int) -> Dict[str, Any]:
        return self._get("RSI", symbol, {"interval": interval, "time_period": time_period, "series_type": "close"})

    def roc(self, symbol: str, interval: str, time_period: int) -> Dict[str, Any]:
        return self._get("ROC", symbol, {"interval": interval, "time_period": time_period, "series_type": "close"})

    def macd(self, symbol: str, interval: str, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> Dict[str, Any]:
        return self._get("MACD", symbol, {"interval": interval, "series_type": "close", "fastperiod": fastperiod, "slowperiod": slowperiod, "signalperiod": signalperiod})

