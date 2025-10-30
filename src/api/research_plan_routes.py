"""
Research Plan API Routes
- Assemble a daily research plan using positions, news, social sentiment, and agent recommendations
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ..data.position_manager import PositionManager
from ..data.position_enrichment_service import PositionEnrichmentService
from ..services.research_service import ResearchService
from ..agents.options_research_agent import OptionsResearchAgent
from ..agents.position_context_service import PositionContextService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/research/plan", tags=["research-plan"])

# Local service instances (stateless / read persisted data)
_position_manager = PositionManager()
_enrichment = PositionEnrichmentService(_position_manager)
_research = ResearchService()
_context = PositionContextService(_position_manager, _enrichment)
_agent = OptionsResearchAgent(_position_manager, _enrichment, _context)


@router.get("/daily")
async def generate_daily_plan(
    schedule: str = Query("auto", description="auto|pre_market|market_open|mid_day|end_of_day"),
    max_age_hours: int = Query(6, ge=1, le=48),
    include_positions: bool = Query(True),
) -> Dict[str, Any]:
    """Generate a day-aware research plan across current positions.
    - Collect unique symbols from positions
    - Pull news + social + youtube research per symbol (cached)
    - Enrich option positions and run OptionsResearchAgent per position
    - Return aggregated plan with prioritized actions
    """
    try:
        # Determine schedule automatically if needed
        now = datetime.now()
        if schedule == "auto":
            hour = now.hour
            if hour < 9:
                schedule_effective = "pre_market"
            elif 9 <= hour < 11:
                schedule_effective = "market_open"
            elif 11 <= hour < 16:
                schedule_effective = "mid_day"
            else:
                schedule_effective = "end_of_day"
        else:
            schedule_effective = schedule

        # Gather symbols
        symbols = _position_manager.get_unique_symbols()
        if not symbols:
            return {
                "schedule": schedule_effective,
                "timestamp": now.isoformat(),
                "symbols": [],
                "positions": [],
                "research": {},
                "recommendations": [],
                "summary": {"message": "No positions found"},
            }

        # Research per symbol
        research_by_symbol: Dict[str, Any] = {}
        for sym in symbols:
            try:
                research_by_symbol[sym] = _research.research_symbol(sym, max_age_hours=max_age_hours)
            except Exception as e:
                logger.warning(f"Research failed for {sym}: {e}")
                research_by_symbol[sym] = {
                    "symbol": sym,
                    "timestamp": now.isoformat(),
                    "news": {"articles": [], "sentiment": "neutral", "score": 0.0},
                    "social": {"overall_sentiment": "neutral", "score": 0.0},
                    "youtube": {"sentiment": "neutral", "score": 0.0},
                    "summary": "Research unavailable",
                }

        # Optionally include enriched positions and recommendations
        positions_payload: List[Dict[str, Any]] = []
        recommendations: List[Dict[str, Any]] = []

        if include_positions:
            # Enrich all and run agent per option position
            option_positions = _position_manager.get_all_option_positions()
            stock_positions = _position_manager.get_all_stock_positions()

            # Enrich
            try:
                _enrichment.enrich_all_positions()
            except Exception as e:
                logger.warning(f"Enrichment failed: {e}")

            # Collect positions payload
            for sp in stock_positions:
                d = sp.to_dict()
                d["type"] = "stock"
                d["research"] = research_by_symbol.get(sp.symbol)
                positions_payload.append(d)

            for op in option_positions:
                d = op.to_dict()
                d["type"] = "option"
                d["research"] = research_by_symbol.get(op.symbol)
                positions_payload.append(d)

            # Agent recommendations for options
            for op in option_positions:
                try:
                    rec = _agent.analyze_position(op.position_id)
                    rec_out = {
                        "position_id": op.position_id,
                        "symbol": op.symbol,
                        "action": rec.get("action"),
                        "urgency": rec.get("urgency"),
                        "reasoning": rec.get("reasoning"),
                        "pnl_pct": rec.get("pnl_pct"),
                        "days_to_expiry": rec.get("days_to_expiry"),
                    }
                    recommendations.append(rec_out)
                except Exception as e:
                    logger.warning(f"Agent analysis failed for {op.position_id}: {e}")

        # Prioritize actions: CRITICAL/HIGH first, then by days_to_expiry ascending
        urgency_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        recommendations.sort(
            key=lambda r: (
                urgency_rank.get((r.get("urgency") or "LOW").upper(), 3),
                (r.get("days_to_expiry") or 9999),
            )
        )

        # Summary
        bullish = sum(1 for s in research_by_symbol.values() if (s.get("news", {}).get("sentiment") == "bullish" or s.get("social", {}).get("overall_sentiment") == "bullish" or s.get("youtube", {}).get("sentiment") == "bullish"))
        bearish = sum(1 for s in research_by_symbol.values() if (s.get("news", {}).get("sentiment") == "bearish" or s.get("social", {}).get("overall_sentiment") == "bearish" or s.get("youtube", {}).get("sentiment") == "bearish"))
        neutral = max(0, len(symbols) - bullish - bearish)

        plan = {
            "schedule": schedule_effective,
            "weekday": now.strftime("%A"),
            "timestamp": now.isoformat(),
            "symbols": list(symbols),
            "research": research_by_symbol,
            "positions": positions_payload,
            "recommendations": recommendations,
            "sentiment_overview": {
                "bullish": bullish,
                "bearish": bearish,
                "neutral": neutral,
            },
        }
        return plan
    except Exception as e:
        logger.error(f"Error generating daily plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

