from __future__ import annotations
import datetime as dt
from typing import Any, Dict, Optional

from src.analytics.cache import load_json_cache, save_json_cache

# TTLs (seconds)
TTL_DAILY = 20 * 60 * 60  # 20 hours
TTL_INTRADAY = 15 * 60     # 15 minutes


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def compute_catalyst_score(symbol: str, as_of: dt.date) -> Dict[str, Any]:
    """
    Stub implementation using local earnings proximity already computed elsewhere.
    Future: augment with FDA/product/regulatory flags from external sources.
    """
    cache_key = f"{symbol}_{as_of.isoformat()}"
    cached = load_json_cache("catalyst", cache_key, TTL_DAILY)
    if cached:
        return cached

    # Neutral default; attempt to lean on existing earnings proximity if available later
    out = {
        "score": 50.0,
        "days_to_earnings": None,
        "events": [],
    }
    save_json_cache("catalyst", cache_key, out)
    return out


def compute_flow_sentiment_score(symbol: str, as_of: dt.date) -> Dict[str, Any]:
    """
    Stub: neutral flow sentiment. Future: integrate UOA provider(s) and compute z-scores.
    """
    cache_key = f"{symbol}_{as_of.isoformat()}"
    cached = load_json_cache("flow", cache_key, TTL_INTRADAY)
    if cached:
        return cached
    out = {
        "score": 50.0,
        "call_put_notional": None,
        "sweep_ratio": None,
        "dark_pool_intensity": None,
    }
    save_json_cache("flow", cache_key, out)
    return out


def compute_macro_risk_score(as_of: dt.date, sector: Optional[str] = None) -> Dict[str, Any]:
    """
    Stub: neutral macro risk. Future: integrate event calendars and sector beta adjustment.
    """
    cache_key = f"{as_of.isoformat()}_{sector or 'ALL'}"
    cached = load_json_cache("macro", cache_key, TTL_DAILY)
    if cached:
        return cached
    out = {
        "score": 50.0,
        "next_events": [],
    }
    save_json_cache("macro", cache_key, out)
    return out


def compute_sector_rotation_score(symbol: str, as_of: dt.date) -> Dict[str, Any]:
    """
    Stub: neutral sector rotation based on placeholder data.
    Future: compute RS vs SPY/sector ETF and map to percentile.
    """
    cache_key = f"{symbol}_{as_of.isoformat()}"
    cached = load_json_cache("sector", cache_key, TTL_DAILY)
    if cached:
        return cached
    out = {
        "score": 50.0,
        "relative_strength": None,
        "momentum_4w": None,
    }
    save_json_cache("sector", cache_key, out)
    return out

