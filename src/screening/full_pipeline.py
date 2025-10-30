from __future__ import annotations
import datetime as dt
from typing import List, Dict, Any

from src.data.providers.router import ProviderRouter
import src.screening.iv_surface_factor as ivf
import src.screening.iv_hv_ratio as ivhv
import src.screening.volume_oi_anomalies as voloi
import src.screening.earnings_proximity as ep
import src.screening.pipeline as pipe
from src.screening.factors import build_rationale


def build_rows_for_universe(universe: List[str], as_of: dt.date | None = None, top: int = 50, min_dte: int = 7, max_dte: int = 90, fresh_chains: bool = False) -> List[Dict[str, Any]]:
    as_of = as_of or dt.date.today()
    rows: List[Dict[str, Any]] = []

    for sym in universe:
        pr = ProviderRouter(sym, fresh_chains=fresh_chains)
        # Basic fields
        adv_rows = pr.daily_prices_window(30)
        adv = sum([r.get("volume", 0) for r in adv_rows]) / max(1, len(adv_rows)) if adv_rows else None
        mc = pr.market_cap()

        # Choose an expiry within DTE window if available
        expiries = pr.expirations()
        if not expiries:
            continue
        eligible = [d for d in expiries if min_dte <= (d - as_of).days <= max_dte]
        eligible.sort(key=lambda d: (d - as_of).days)
        if not eligible:
            continue
        expiry = eligible[0]

        # ATM IV and ranks
        iv_res = ivf.compute_atm_iv_and_rank(sym, expiry, as_of)
        # IV/HV ratios
        ivhv_res = ivhv.compute_iv_hv_ratio(sym, expiry, as_of)
        # Volume/OI anomalies (chain-level). Also cache chain snapshot to DataStore.
        chain_calls = pr.options_chain(expiry, "call")
        chain_puts = pr.options_chain(expiry, "put")
        # Provider-agnostic snapshot caching (Parquet-first, JSON fallback)
        try:
            from src.data.datastore import DataStore
            import logging
            logger = logging.getLogger("screen")
            ds = DataStore()
            def _opt_to_dict(opt) -> Dict[str, Any]:
                return {
                    "symbol": getattr(opt, "symbol", None),
                    "strike": getattr(opt, "strike", None),
                    "expiry": expiry.isoformat(),
                    "option_type": getattr(opt, "option_type", None),
                    "last_price": getattr(opt, "last_price", None),
                    "bid": getattr(opt, "bid", None),
                    "ask": getattr(opt, "ask", None),
                    "volume": getattr(opt, "volume", None),
                    "open_interest": getattr(opt, "open_interest", None),
                    "underlying_price": getattr(opt, "underlying_price", None),
                }
            snapshot_rows = [_opt_to_dict(o) for o in (chain_calls + chain_puts)]
            if snapshot_rows:
                ds.write_options_chain_snapshot(sym, as_of, snapshot_rows)
                logger.debug(f"{sym}: cached {len(snapshot_rows)} options to data/opt/chains/{sym}/{as_of.isoformat()}.parquet")
        except Exception as e:
            # Non-fatal; pipeline proceeds even if caching fails
            import logging
            logging.getLogger("screen").debug(f"{sym}: chain snapshot cache skipped: {e}")
        chain_rows: List[Dict[str, Any]] = []
        for opt in (chain_calls + chain_puts):
            hv_hist = [opt.volume or 0] * 30 if getattr(opt, 'volume', None) is not None else []
            hoi_hist = [opt.open_interest or 0] * 30 if getattr(opt, 'open_interest', None) is not None else []
            chain_rows.append({
                "strike": getattr(opt, 'strike', None),
                "option_type": getattr(opt, 'option_type', None),
                "volume": getattr(opt, 'volume', None),
                "oi": getattr(opt, 'open_interest', None),
                "hist_volume": hv_hist,
                "hist_oi": hoi_hist,
            })
        anom_res = voloi.compute_anomalies_for_chain(chain_rows) if chain_rows else None

        # Earnings proximity
        earn_res = ep.compute_earnings_proximity(sym, expiry, as_of)

        row = {
            "symbol": sym,
            "adv": adv,
            "market_cap": mc,
            "iv_rank": iv_res.iv_rank,
            "iv_percentile": iv_res.iv_percentile,
            "iv_hv_ratio": ivhv_res.ratio30 if ivhv_res.ratio30 is not None else ivhv_res.ratio60,
            "volume_anomaly": anom_res.chain_volume_anomaly if anom_res else None,
            "oi_anomaly": anom_res.chain_oi_anomaly if anom_res else None,
            "pre_earnings_ratio": earn_res.pre_earnings_ratio,
        }
        rows.append(row)

    # Momentum and composite scoring
    rows = pipe.add_momentum_to_rows(rows, interval="daily")
    # Composite score: reuse our composite system; populate inputs expected by scorer
    from src.screening.scorer import ScreenWeights, score_row
    w = ScreenWeights()
    for r in rows:
        r.setdefault("spread_bps", 10.0)
        r.setdefault("theta_eff", 0.0)
        r.setdefault("delta_adj", 0.0)
        r.setdefault("expected_value", 0.0)
        r["composite_score"] = score_row(r, w)
        r["rationale"] = build_rationale(r)

    rows.sort(key=lambda x: (x.get("composite_score") or 0.0), reverse=True)
    return rows[:top]

