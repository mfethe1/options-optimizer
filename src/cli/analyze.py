from __future__ import annotations
import argparse
import datetime as dt
import json
import math
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data.providers.router import ProviderRouter
from src.screening.iv_surface_factor import compute_atm_iv_and_rank


def _round2(x: Optional[float]) -> Optional[float]:
    return round(float(x), 2) if x is not None else None


def _mid_from_fields(bid: Optional[float], ask: Optional[float], last_price: Optional[float]) -> Optional[float]:
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return 0.5 * (bid + ask)
    if last_price is not None and last_price > 0:
        return float(last_price)
    return None


def _mid_of_option(opt) -> Optional[float]:
    return _mid_from_fields(getattr(opt, "bid", None), getattr(opt, "ask", None), getattr(opt, "last_price", None))


def _nearest_strike(strikes: List[float], target: float) -> Optional[float]:
    if not strikes:
        return None
    return min(strikes, key=lambda s: abs(s - target))


def _pick_expiration(pr: ProviderRouter, as_of: dt.date, dte_min: int, dte_max: int) -> Optional[dt.date]:
    exps = pr.expirations()
    if not exps:
        return None
    eligible = [d for d in exps if dte_min <= (d - as_of).days <= dte_max]
    if not eligible:
        eligible = [d for d in exps if (d - as_of).days >= dte_min]
    if not eligible:
        return None
    eligible.sort(key=lambda d: abs((d - as_of).days - int(0.5 * (dte_min + dte_max))))
    return eligible[0]


def _compute_em_and_atm(pr: ProviderRouter, expiry: dt.date, S: float) -> Dict[str, Any]:
    calls = pr.options_chain(expiry, "call")
    puts = pr.options_chain(expiry, "put")
    if not calls or not puts:
        return {"atm": None, "mid_call": None, "mid_put": None, "em_pct": None, "strikes": []}
    strikes = sorted({float(getattr(c, "strike", None)) for c in calls if getattr(c, "strike", None) is not None})
    if not strikes:
        return {"atm": None, "mid_call": None, "mid_put": None, "em_pct": None, "strikes": []}
    atm = _nearest_strike(strikes, S)
    call_atm = next((c for c in calls if getattr(c, "strike", None) == atm), None)
    put_atm = next((p for p in puts if getattr(p, "strike", None) == atm), None)
    m_call = _mid_of_option(call_atm) if call_atm else None
    m_put = _mid_of_option(put_atm) if put_atm else None
    em_pct = ((m_call or 0.0) + (m_put or 0.0)) / S if S else None
    return {"atm": atm, "mid_call": m_call, "mid_put": m_put, "em_pct": em_pct, "strikes": strikes, "calls": calls, "puts": puts}


def _mid_by_side_and_strike(calls, puts, strike: float, side: str) -> Optional[float]:
    if side == "call":
        opt = next((c for c in calls if getattr(c, "strike", None) == strike), None)
    else:
        opt = next((p for p in puts if getattr(p, "strike", None) == strike), None)
    return _mid_of_option(opt) if opt else None


def _scenario_prices(S: float, em_pct: float) -> Dict[str, float]:
    # Price scenarios at +/- EM, +/- 1.5x EM, +/- 2x EM and spot
    factors = [-2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0]
    out: Dict[str, float] = {}
    for f in factors:
        if f < 0:
            out[f"{f}xEM"] = S * (1.0 + f * em_pct)  # f is negative
        elif f > 0:
            out[f"+{f}xEM"] = S * (1.0 + f * em_pct)
        else:
            out["spot"] = S
    return out


def _straddle_pl_per_contract(S_T: float, K: float, debit: float) -> float:
    payoff_per_share = abs(S_T - K) - debit
    return round(payoff_per_share * 100.0, 2)


def _condor_pl_per_contract(S_T: float, sp: float, lp: float, sc: float, lc: float, credit: float) -> float:
    put_loss = max(0.0, sp - S_T) - max(0.0, lp - S_T)
    call_loss = max(0.0, S_T - sc) - max(0.0, S_T - lc)
    pl_per_share = credit - max(0.0, put_loss) - max(0.0, call_loss)
    return round(pl_per_share * 100.0, 2)


def _call_spread_pl_per_contract(S_T: float, long_k: float, short_k: float, debit: float) -> float:
    value_per_share = max(0.0, S_T - long_k) - max(0.0, S_T - short_k)
    value_per_share = max(0.0, min(value_per_share, short_k - long_k))
    return round((value_per_share - debit) * 100.0, 2)


def _put_spread_pl_per_contract(S_T: float, long_k: float, short_k: float, debit: float) -> float:
    value_per_share = max(0.0, long_k - S_T) - max(0.0, short_k - S_T)
    value_per_share = max(0.0, min(value_per_share, long_k - short_k))
    return round((value_per_share - debit) * 100.0, 2)


def _build_condor(S: float, strikes: List[float], calls, puts, em_pct: float) -> Dict[str, Any]:
    # Choose short strikes at +/- 1.2x EM and wings ~5 or 10 wide depending on S
    em_factor = 1.2
    width = 10.0 if S >= 500 else 5.0
    short_put_target = S * (1.0 - em_factor * em_pct)
    short_call_target = S * (1.0 + em_factor * em_pct)
    sp = _nearest_strike(strikes, short_put_target)
    sc = _nearest_strike(strikes, short_call_target)
    if sp is None or sc is None:
        return {"type": "iron_condor", "error": "no_strikes"}
    # Long wings: approximate width using available strikes
    put_wing_target = sp - width
    call_wing_target = sc + width
    lp = max([s for s in strikes if s <= put_wing_target], default=min(strikes))
    lc = min([s for s in strikes if s >= call_wing_target], default=max(strikes))
    # Estimate credit from mids
    short_put_mid = _mid_by_side_and_strike(calls, puts, sp, "put")
    long_put_mid = _mid_by_side_and_strike(calls, puts, lp, "put")
    short_call_mid = _mid_by_side_and_strike(calls, puts, sc, "call")
    long_call_mid = _mid_by_side_and_strike(calls, puts, lc, "call")
    credit_put = (short_put_mid or 0.0) - (long_put_mid or 0.0)
    credit_call = (short_call_mid or 0.0) - (long_call_mid or 0.0)
    credit_total = max(0.0, credit_put) + max(0.0, credit_call)
    width_put = abs(sp - lp)
    width_call = abs(lc - sc)
    target_min_credit = 0.33 * min(width_put, width_call)
    est_pop = [0.60, 0.75]  # heuristic band
    # Profit/loss and legs list (per contract; multiplier 100)
    legs = [
        {"action": "SELL", "type": "PUT", "strike": sp},
        {"action": "BUY",  "type": "PUT", "strike": lp},
        {"action": "SELL", "type": "CALL", "strike": sc},
        {"action": "BUY",  "type": "CALL", "strike": lc},
    ]
    out = {
        "type": "iron_condor",
        "legs": legs,
        "short_put": sp,
        "long_put": lp,
        "short_call": sc,
        "long_call": lc,
        "est_credit": _round2(credit_total) if credit_total else None,
        "target_min_credit": _round2(target_min_credit),
        "width_put": width_put,
        "width_call": width_call,
        "max_profit": _round2(credit_total * 100) if credit_total else None,
        "max_loss": _round2((min(width_put, width_call) - credit_total) * 100) if credit_total else None,
        "breakevens": [
            _round2(sp - credit_total),
            _round2(sc + credit_total)
        ],
        "est_pop_range": est_pop,
        "scenarios": {},
        "mgmt": {
            "take_profit": "50% credit",
            "stop": "1.5-2x credit",
            "notes": "Avoid holding through earnings/macro unless intended. Ensure credit >= 1/3 of min wing width."
        }
    }
    # Add scenario P/L at +/- EM, 1.5x, 2x
    try:
        prices = _scenario_prices(S, em_pct)
        scen = {}
        for label, st in prices.items():
            scen[label] = _condor_pl_per_contract(st, sp, lp, sc, lc, credit_total)
        out["scenarios"] = scen
    except Exception:
        pass
    return out


def _build_straddle(S: float, atm: float, mid_call: Optional[float], mid_put: Optional[float]) -> Dict[str, Any]:
    debit = (mid_call or 0.0) + (mid_put or 0.0)
    be_low = atm - debit
    be_high = atm + debit
    legs = [
        {"action": "BUY", "type": "CALL", "strike": atm},
        {"action": "BUY", "type": "PUT",  "strike": atm},
    ]
    out = {
        "type": "long_straddle",
        "legs": legs,
        "strike": atm,
        "est_debit": _round2(debit) if debit else None,
        "breakevens": [_round2(be_low), _round2(be_high)],
        "max_loss": _round2(debit * 100) if debit else None,
        "max_profit": "unlimited",
        "est_pop_range": [0.35, 0.45],
        "scenarios": {},
        "mgmt": {
            "take_profit": "25-50% gain",
            "stop": "40-50% loss on debit",
            "notes": "Exit 3-5 days before expiry unless actively managing delta/gamma."
        }
    }
    try:
        prices = _scenario_prices(S, (abs(S - atm) / S) if S else 0.0)
        # Use EM% for movement sizing; if unavailable debit/S is fallback for a baseline
        if prices and debit and S:
            scen = {}
            em_pct = debit / S
            for label, st in _scenario_prices(S, em_pct).items():
                scen[label] = _straddle_pl_per_contract(st, atm, debit)
            out["scenarios"] = scen
    except Exception:
        pass
    return out


def _build_directional(S: float, strikes: List[float], calls, puts, atm: float, bullish: bool) -> Dict[str, Any]:
    # Debit vertical: long near-ATM, short ~10% OTM
    up = _nearest_strike(strikes, S * 1.10)
    down = _nearest_strike(strikes, S * 0.90)
    long_k = atm
    if bullish:
        short_k = up
        long_mid = _mid_by_side_and_strike(calls, puts, long_k, "call")
        short_mid = _mid_by_side_and_strike(calls, puts, short_k, "call")
        debit = (long_mid or 0.0) - (short_mid or 0.0) if long_mid and short_mid else None
        max_val = (short_k - long_k) if (short_k and long_k) else None
        legs = [
            {"action": "BUY",  "type": "CALL", "strike": long_k},
            {"action": "SELL", "type": "CALL", "strike": short_k},
        ]
    else:
        short_k = down
        long_mid = _mid_by_side_and_strike(calls, puts, long_k, "put")
        short_mid = _mid_by_side_and_strike(calls, puts, short_k, "put")
        debit = (long_mid or 0.0) - (short_mid or 0.0) if long_mid and short_mid else None
        max_val = (long_k - short_k) if (short_k and long_k) else None
        legs = [
            {"action": "BUY",  "type": "PUT", "strike": long_k},
            {"action": "SELL", "type": "PUT", "strike": short_k},
        ]
    rr = (max_val / debit - 1.0) if (debit and max_val) else None
    out = {
        "type": "debit_call_spread" if bullish else "debit_put_spread",
        "legs": legs,
        "long": long_k,
        "short": short_k,
        "est_debit": _round2(debit) if debit else None,
        "max_profit": _round2((max_val - (debit or 0)) * 100) if (max_val and debit) else None,
        "max_loss": _round2((debit or 0) * 100) if debit else None,
        "est_rr": _round2(rr) if rr else None,
        "breakeven": _round2(long_k + (debit or 0)) if bullish else _round2(long_k - (debit or 0)),
        "scenarios": {},
        "mgmt": {
            "take_profit": "50-100% gain on debit",
            "stop": "40-50% loss on debit"
        }
    }
    try:
        if debit and S:
            scen = {}
            em_pct = debit / S  # rough movement proxy for scenario spacing
            for label, st in _scenario_prices(S, em_pct).items():
                if bullish:
                    scen[label] = _call_spread_pl_per_contract(st, long_k, short_k, debit)
                else:
                    scen[label] = _put_spread_pl_per_contract(st, long_k, short_k, debit)
            out["scenarios"] = scen
    except Exception:
        pass
    return out


def run():
    parser = argparse.ArgumentParser(description="Analyze a ticker's 30-day options profile and recommend a strategy")
    parser.add_argument("symbol", type=str)
    parser.add_argument("--dte-min", type=int, default=21)
    parser.add_argument("--dte-max", type=int, default=35)
    parser.add_argument("--fresh-chains", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()

    sym = args.symbol.upper()
    as_of = dt.date.today()

    pr = ProviderRouter(sym, fresh_chains=args.fresh_chains)
    S = pr.underlying_price()
    expiry = _pick_expiration(pr, as_of, args.dte_min, args.dte_max)
    if S is None or not expiry:
        print(json.dumps({"error": "missing_underlying_or_expiry", "symbol": sym}))
        return

    em = _compute_em_and_atm(pr, expiry, S)
    iv = compute_atm_iv_and_rank(sym, expiry, as_of)

    # Enriched signals (stubs currently with neutral values)
    from src.analytics.signals import (
        compute_catalyst_score,
        compute_flow_sentiment_score,
        compute_macro_risk_score,
        compute_sector_rotation_score,
    )
    catalyst = compute_catalyst_score(sym, as_of)
    flow = compute_flow_sentiment_score(sym, as_of)
    macro = compute_macro_risk_score(as_of)
    sector_sig = compute_sector_rotation_score(sym, as_of)

    # Optional momentum/composite from debug.csv if exists
    debug_row = None
    if os.path.exists("debug.csv"):
        try:
            df = pd.read_csv("debug.csv")
            r = df[df["symbol"].str.upper() == sym]
            if not r.empty:
                debug_row = r.iloc[0].to_dict()
        except Exception:
            pass

    # Decide strategy (LM-refined heuristic)
    iv_pct = iv.iv_percentile or 0.0
    em_pct = em.get("em_pct") or 0.0
    cat_score = float(catalyst.get("score") or 50.0)
    flow_score = float(flow.get("score") or 50.0)
    macro_score = float(macro.get("score") or 50.0)
    sector_score = float(sector_sig.get("score") or 50.0)

    strategy: Dict[str, Any]
    notes: List[str] = []

    if (iv_pct <= 25 and em_pct >= 0.045 and em.get("atm") is not None and (cat_score >= 70 or flow_score >= 70)):
        strategy = _build_straddle(S, em["atm"], em.get("mid_call"), em.get("mid_put"))
        notes.append("Low IV percentile with strong catalyst/flow and solid expected move: favor long volatility.")
    elif (iv_pct >= 75 and 0.025 <= em_pct <= 0.06 and em.get("strikes") and macro_score <= 60):
        strategy = _build_condor(S, em["strikes"], em.get("calls"), em.get("puts"), em_pct)
        notes.append("High IV percentile with moderate expected move and acceptable macro risk: favor short premium (risk-defined).")
    else:
        # Directional using momentum and sector tilt
        bullish = bool(debug_row and float(debug_row.get("momentum_score", 0)) > 0)
        if sector_score >= 70:
            notes.append("Sector rotation strong: tilt directional spreads in sector trend direction.")
        strategy = _build_directional(S, em.get("strikes", []), em.get("calls"), em.get("puts"), em.get("atm"), bullish)
        notes.append("Mid IV regime: use directional debit spread guided by momentum and sector tilt.")

    output = {
        "symbol": sym,
        "as_of": as_of.isoformat(),
        "S": S,
        "expiry": expiry.isoformat(),
        "iv": {"atm": iv.atm_iv, "rank": iv.iv_rank, "percentile": iv.iv_percentile},
        "em_pct": em_pct,
        "strategy": strategy,
        "signals": {
            "catalyst": catalyst,
            "flow_sentiment": flow,
            "macro_risk": macro,
            "sector_rotation": sector_sig,
        },
        "context": {
            "momentum_score": (float(debug_row.get("momentum_score")) if debug_row and not pd.isna(debug_row.get("momentum_score")) else None),
            "iv_hv_ratio": (float(debug_row.get("iv_hv_ratio")) if debug_row and not pd.isna(debug_row.get("iv_hv_ratio")) else None)
        },
        "notes": notes,
    }

    if args.print_json:
        print(json.dumps(output, indent=2))
    else:
        print(f"{sym} @ {S:.2f} exp {output['expiry']} | EM%={em_pct:.2%} | IV%ile={iv.iv_percentile:.0f}")
        print(f"Suggested: {strategy['type']} -> details in --print-json")


if __name__ == "__main__":
    run()

