from __future__ import annotations
from typing import List, Dict
import math


def hv_annualized_from_prices(daily_rows: List[Dict[str, float]], window: int = 30) -> float:
    """
    Compute annualized historical volatility from a list of dict rows with 'close'.
    Uses log returns over the last `window` closes; annualization with 252.
    Returns 0.0 if insufficient data.
    """
    closes = [r["close"] for r in daily_rows if r.get("close")]
    if len(closes) < window + 1:
        return 0.0
    rets = []
    for i in range(-window, 0):
        try:
            r = math.log(closes[i] / closes[i - 1])
            rets.append(r)
        except Exception:
            return 0.0
    if not rets:
        return 0.0
    mu = sum(rets) / len(rets)
    var = sum((x - mu) ** 2 for x in rets) / max(1, len(rets) - 1)
    daily_vol = math.sqrt(var)
    return daily_vol * math.sqrt(252.0)

