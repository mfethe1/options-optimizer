from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any

from src.data.providers.router import ProviderRouter
from src.data.providers.fmp_provider import FMPProvider


@dataclass
class UniverseConfig:
    adv_min: int = 4_000_000
    market_cap_min: float = 10e9
    lookback_days: int = 30


def _adv(daily_rows: List[Dict[str, Any]], lookback: int) -> float:
    vols = [r["volume"] for r in daily_rows[-lookback:] if r.get("volume")]
    return float(sum(vols) / len(vols)) if vols else 0.0


def build_universe_dynamic_from_fmp(cfg: UniverseConfig, limit: int = 100) -> List[str]:
    # Prefer FMP stock screener for clean, liquid symbols to avoid delisted/SPAC artifacts in stock/list
    try:
        fmp = FMPProvider()
        screener = fmp.stock_screener(marketCapMoreThan=cfg.market_cap_min, volumeMoreThan=cfg.adv_min, limit=limit*3)
        symbols = [row.get("symbol") for row in screener if row.get("symbol")]
    except Exception:
        symbols = ProviderRouter.list_all_symbols_via_fmp()

    selected: List[tuple[str, float]] = []
    for sym in symbols:
        try:
            pr = ProviderRouter(sym)
            daily = pr.daily_prices()
            adv = _adv(daily, cfg.lookback_days)
            if adv < cfg.adv_min:
                continue
            mc = pr.market_cap() or 0.0
            if mc < cfg.market_cap_min:
                continue
            selected.append((sym, adv))
            if len(selected) >= limit:
                break
        except Exception:
            continue
    selected.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in selected[:limit]]

