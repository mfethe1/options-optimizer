from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from src.config import load_config
from src.data.http_client import CachingHttpClient, RateLimiter

FMP_URL = "https://financialmodelingprep.com/api/v3"


@dataclass
class FMPProvider:
    api_key: Optional[str] = None

    def __post_init__(self):
        cfg = load_config()
        self.api_key = self.api_key or cfg.fmp_key
        self.client = CachingHttpClient(cfg.cache_dir)
        self.rate_limiter = RateLimiter(cfg.fmp_rpm)

    def _get(self, path: str, params: Dict[str, Any], ttl_seconds: int = 300) -> Dict[str, Any]:
        p = {"apikey": self.api_key}
        p.update(params)
        url = f"{FMP_URL}/{path}"
        return self.client.get_json(url, p, self.rate_limiter, ttl_seconds=ttl_seconds)

    def company_key_metrics(self, symbol: str) -> Dict[str, Any]:
        return self._get(f"key-metrics-ttm/{symbol}", {}, ttl_seconds=86400)

    def earnings_transcripts(self, symbol: str) -> Dict[str, Any]:
        return self._get(f"earning_call_transcript/{symbol}", {}, ttl_seconds=86400)

    def profile(self, symbol: str) -> Dict[str, Any]:
        return self._get(f"profile/{symbol}", {}, ttl_seconds=86400)

    def market_cap(self, symbol: str) -> Optional[float]:
        try:
            data = self.profile(symbol)
            if isinstance(data, list) and data:
                mc = data[0].get("mktCap") or data[0].get("marketCap")
            else:
                mc = data.get("mktCap") if isinstance(data, dict) else None
            return float(mc) if mc is not None else None
        except Exception:
            return None

    def stock_list(self) -> List[Dict[str, Any]]:
        # Returns a large list of tickers with basic fields
        return self._get("stock/list", {}, ttl_seconds=24*3600)  # 1 day cache

    def stock_screener(self, marketCapMoreThan: float = 1e10, volumeMoreThan: int = 4_000_000, exchange: str | None = None, limit: int = 200) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "marketCapMoreThan": int(marketCapMoreThan),
            "volumeMoreThan": int(volumeMoreThan),
            "limit": int(limit),
        }
        if exchange:
            params["exchange"] = exchange
        return self._get("stock-screener", params, ttl_seconds=3600)

