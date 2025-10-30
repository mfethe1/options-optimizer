from __future__ import annotations
import os
from dataclasses import dataclass


def _load_dotenv_simple(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass


@dataclass
class ProviderConfig:
    alpha_vantage_key: str | None
    finnhub_key: str | None
    marketstack_key: str | None
    fmp_key: str | None
    fred_key: str | None

    # rate limits (calls per minute) - defaults for free tiers; override via env
    alpha_vantage_rpm: int
    finnhub_rpm: int
    marketstack_rpm: int
    fmp_rpm: int
    fred_rpm: int

    cache_dir: str


def load_config() -> ProviderConfig:
    _load_dotenv_simple()
    return ProviderConfig(
        alpha_vantage_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
        finnhub_key=os.getenv("FINNHUB_API_KEY"),
        marketstack_key=os.getenv("MARKETSTACK_API_KEY"),
        fmp_key=os.getenv("FMP_API_KEY"),
        fred_key=os.getenv("FRED_API_KEY"),
        alpha_vantage_rpm=int(os.getenv("ALPHA_VANTAGE_RPM", "5")),
        finnhub_rpm=int(os.getenv("FINNHUB_RPM", "60")),
        marketstack_rpm=int(os.getenv("MARKETSTACK_RPM", "100")),
        fmp_rpm=int(os.getenv("FMP_RPM", "30")),
        fred_rpm=int(os.getenv("FRED_RPM", "60")),
        cache_dir=os.getenv("CACHE_DIR", os.path.join("data", "cache")),
    )

