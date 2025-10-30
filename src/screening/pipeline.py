from __future__ import annotations
from typing import List, Dict, Any

from src.screening.momentum import MomentumComputer, MomentumWeights


def add_momentum_to_rows(rows: List[Dict[str, Any]], interval: str = "daily") -> List[Dict[str, Any]]:
    """
    For each row with a 'symbol', fetch Alpha Vantage technical indicators via MomentumComputer
    and compute momentum_score (0-100). Adds 'momentum_score' to each row.

    Robust to provider/API errors: on failure or missing data, assigns a neutral score (50.0).
    Reuses a single MomentumComputer instance to share provider rate limiting/caching across tickers.
    """
    mc = MomentumComputer()
    w = MomentumWeights()  # equal weights (0.2 each)

    for row in rows:
        sym = row.get("symbol")
        if not sym:
            row["momentum_score"] = None
            continue
        try:
            inds = mc.fetch_indicators(sym, interval=interval)
            row["momentum_score"] = mc.momentum_score(inds, w)
        except Exception:
            # Graceful degradation on API or data issues
            row["momentum_score"] = 50.0
    return rows

