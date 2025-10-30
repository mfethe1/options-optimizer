from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import datetime as dt

from src.data.providers.router import ProviderRouter
from src.pricing.iv.surface_fit import ATMIVHistory
from src.screening.iv_surface_factor import compute_atm_iv_and_rank


@dataclass
class EarningsProximityResult:
    days_to_earnings: Optional[int]
    pre_earnings_ratio: Optional[float]


def _previous_earnings_dates(dates: List[dt.date], as_of: dt.date, n: int = 3) -> List[dt.date]:
    prev = [d for d in dates if d <= as_of]
    prev.sort()
    return prev[-n:]


def compute_earnings_proximity(symbol: str, expiry: dt.date, as_of: dt.date | None = None) -> EarningsProximityResult:
    as_of = as_of or dt.date.today()
    pr = ProviderRouter(symbol)
    events = pr.earnings_events(as_of)

    # Days to next earnings
    next_events = [d for d in events if d >= as_of]
    next_events.sort()
    days_to = None
    if next_events:
        days_to = (next_events[0] - as_of).days

    # Pre-earnings IV ratio: current ATM IV vs. historical pre-earnings baseline
    # Get prior earnings dates and for each take 5-day pre-event ATM IV (mean)
    history = ATMIVHistory()
    hist_items = history.load(symbol)
    if not hist_items:
        return EarningsProximityResult(days_to, None)
    hist_map = {dt.date.fromisoformat(d): v for d, v in hist_items}

    prev_evts = _previous_earnings_dates(events, as_of, n=4)
    if not prev_evts:
        return EarningsProximityResult(days_to, None)

    baselines: List[float] = []
    for e in prev_evts:
        window = [hist_map.get(e - dt.timedelta(days=i)) for i in range(1, 6)]
        window = [x for x in window if x is not None]
        if len(window) >= 2:
            baselines.append(sum(window) / len(window))
    if not baselines:
        return EarningsProximityResult(days_to, None)

    baseline = sum(baselines) / len(baselines)

    cur = compute_atm_iv_and_rank(symbol, expiry, as_of).atm_iv
    if cur is None or baseline <= 0:
        return EarningsProximityResult(days_to, None)

    ratio = cur / baseline
    return EarningsProximityResult(days_to, ratio)

