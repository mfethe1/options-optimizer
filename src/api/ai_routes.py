"""
AI Trading Services API Routes

Exposes swarm analysis, risk management, and expert critique endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

from ..analytics.swarm_analysis_service import (
    SwarmAnalysisService,
    BacktestResult,
    SwarmConsensus
)
from ..analytics.risk_guardrails import (
    RiskGuardrailsService,
    RiskLevel,
    PortfolioState,
    Position,
    RiskCheckResult
)
from ..analytics.expert_critique import ExpertCritiqueService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai", tags=["AI Trading Services"])


# ============================================================================
# Request/Response Models
# ============================================================================

class BacktestResultRequest(BaseModel):
    """Backtest result for swarm analysis"""
    strategy_name: str
    symbol: str
    timeframe: str
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    kelly_criterion: float
    var_95: float
    expected_value: float


class PositionRequest(BaseModel):
    """Portfolio position"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    position_type: str
    expiration: Optional[str] = None
    sector: str = "Unknown"
    beta: float = 1.0


class PortfolioStateRequest(BaseModel):
    """Portfolio state for risk checks"""
    total_value: float
    cash: float
    positions: List[PositionRequest]
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_leverage: float = 1.0


class NewPositionRequest(BaseModel):
    """Request to check a new position"""
    symbol: str
    proposed_size: float
    position_type: str
    portfolio: PortfolioStateRequest
    market_data: Dict[str, Any] = {}


# ============================================================================
# Swarm Analysis Endpoints
# ============================================================================

@router.post("/swarm/analyze")
async def analyze_strategy(backtest: BacktestResultRequest) -> Dict[str, Any]:
    """
    Perform multi-agent swarm analysis on a strategy.

    Returns consensus recommendation with risk assessment and position sizing.
    """
    try:
        swarm_service = SwarmAnalysisService()

        # Convert request to BacktestResult
        backtest_result = BacktestResult(
            strategy_name=backtest.strategy_name,
            symbol=backtest.symbol,
            timeframe=backtest.timeframe,
            total_return=backtest.total_return,
            sharpe_ratio=backtest.sharpe_ratio,
            sortino_ratio=backtest.sortino_ratio,
            max_drawdown=backtest.max_drawdown,
            win_rate=backtest.win_rate,
            profit_factor=backtest.profit_factor,
            avg_win=backtest.avg_win,
            avg_loss=backtest.avg_loss,
            total_trades=backtest.total_trades,
            kelly_criterion=backtest.kelly_criterion,
            var_95=backtest.var_95,
            expected_value=backtest.expected_value
        )

        # Perform swarm analysis
        consensus = await swarm_service.analyze_strategy(backtest_result)

        # Convert to dict for JSON response
        return {
            "strategy": consensus.strategy,
            "symbol": consensus.symbol,
            "overall_score": consensus.overall_score,
            "consensus_recommendation": consensus.consensus_recommendation,
            "consensus_confidence": consensus.consensus_confidence,
            "expected_value": consensus.expected_value,
            "risk_adjusted_return": consensus.risk_adjusted_return,
            "suggested_position_size": consensus.suggested_position_size,
            "stop_loss": consensus.stop_loss,
            "take_profit": consensus.take_profit,
            "max_loss_per_trade": consensus.max_loss_per_trade,
            "agent_votes": consensus.agent_votes,
            "agent_analyses": [
                {
                    "agent_name": a.agent_name,
                    "agent_type": a.agent_type,
                    "score": a.score,
                    "recommendation": a.recommendation,
                    "confidence": a.confidence,
                    "reasoning": a.reasoning,
                    "risk_concerns": a.risk_concerns,
                    "opportunity_highlights": a.opportunity_highlights
                }
                for a in consensus.agent_analyses
            ],
            "risk_warnings": consensus.risk_warnings,
            "go_decision": consensus.go_decision,
            "reasoning_summary": consensus.reasoning_summary
        }

    except Exception as e:
        logger.error(f"Swarm analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/swarm/compare")
async def compare_strategies(backtests: List[BacktestResultRequest]) -> Dict[str, Any]:
    """
    Compare multiple strategies and rank by AI consensus.

    Returns ranked list with best strategy first.
    """
    try:
        swarm_service = SwarmAnalysisService()

        # Convert requests to BacktestResults
        backtest_results = [
            BacktestResult(
                strategy_name=bt.strategy_name,
                symbol=bt.symbol,
                timeframe=bt.timeframe,
                total_return=bt.total_return,
                sharpe_ratio=bt.sharpe_ratio,
                sortino_ratio=bt.sortino_ratio,
                max_drawdown=bt.max_drawdown,
                win_rate=bt.win_rate,
                profit_factor=bt.profit_factor,
                avg_win=bt.avg_win,
                avg_loss=bt.avg_loss,
                total_trades=bt.total_trades,
                kelly_criterion=bt.kelly_criterion,
                var_95=bt.var_95,
                expected_value=bt.expected_value
            )
            for bt in backtests
        ]

        # Compare strategies
        ranked_consensuses = await swarm_service.compare_strategies(backtest_results)

        # Return ranked results
        return {
            "total_strategies": len(ranked_consensuses),
            "best_strategy": ranked_consensuses[0].strategy if ranked_consensuses else None,
            "ranked_strategies": [
                {
                    "rank": idx + 1,
                    "strategy": c.strategy,
                    "symbol": c.symbol,
                    "score": c.overall_score,
                    "recommendation": c.consensus_recommendation,
                    "confidence": c.consensus_confidence,
                    "go_decision": c.go_decision,
                    "expected_value": c.expected_value,
                    "suggested_position_size": c.suggested_position_size
                }
                for idx, c in enumerate(ranked_consensuses)
            ]
        }

    except Exception as e:
        logger.error(f"Strategy comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Risk Management Endpoints
# ============================================================================

@router.post("/risk/check-position")
async def check_new_position(
    request: NewPositionRequest,
    risk_level: str = Query("moderate", description="Risk tolerance level")
) -> Dict[str, Any]:
    """
    Check if new position passes risk guardrails.

    Returns approval status, violations, and suggested position size.
    """
    try:
        # Parse risk level
        try:
            risk_level_enum = RiskLevel(risk_level.lower())
        except ValueError:
            risk_level_enum = RiskLevel.MODERATE

        risk_service = RiskGuardrailsService(risk_level_enum)

        # Convert request to internal models
        portfolio = PortfolioState(
            total_value=request.portfolio.total_value,
            cash=request.portfolio.cash,
            positions=[
                Position(
                    symbol=p.symbol,
                    quantity=p.quantity,
                    entry_price=p.entry_price,
                    current_price=p.current_price,
                    position_type=p.position_type,
                    expiration=p.expiration,
                    sector=p.sector,
                    beta=p.beta
                )
                for p in request.portfolio.positions
            ],
            daily_pnl=request.portfolio.daily_pnl,
            weekly_pnl=request.portfolio.weekly_pnl,
            monthly_pnl=request.portfolio.monthly_pnl,
            max_drawdown=request.portfolio.max_drawdown,
            current_leverage=request.portfolio.current_leverage
        )

        # Perform risk check
        result = risk_service.check_new_position(
            symbol=request.symbol,
            proposed_size=request.proposed_size,
            position_type=request.position_type,
            portfolio=portfolio,
            market_data=request.market_data
        )

        # Convert to response
        return {
            "approved": result.approved,
            "violations": [
                {
                    "severity": v.severity,
                    "rule_name": v.rule_name,
                    "current_value": v.current_value,
                    "limit_value": v.limit_value,
                    "message": v.message,
                    "blocking": v.blocking
                }
                for v in result.violations
            ],
            "warnings": [
                {
                    "severity": w.severity,
                    "rule_name": w.rule_name,
                    "current_value": w.current_value,
                    "limit_value": w.limit_value,
                    "message": w.message,
                    "blocking": w.blocking
                }
                for w in result.warnings
            ],
            "max_position_size": result.max_position_size,
            "suggested_position_size": result.suggested_position_size,
            "risk_score": result.risk_score,
            "risk_level": result.risk_level,
            "detailed_report": result.detailed_report
        }

    except Exception as e:
        logger.error(f"Risk check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk/limits/{risk_level}")
async def get_risk_limits(risk_level: str) -> Dict[str, Any]:
    """
    Get risk limits for a specific risk tolerance level.

    Returns complete risk configuration.
    """
    try:
        risk_level_enum = RiskLevel(risk_level.lower())
        risk_service = RiskGuardrailsService(risk_level_enum)

        limits = risk_service.limits

        return {
            "risk_level": risk_level,
            "limits": {
                "position_limits": {
                    "max_position_size_pct": limits.max_position_size_pct,
                    "max_sector_exposure_pct": limits.max_sector_exposure_pct,
                    "max_correlation_exposure": limits.max_correlation_exposure
                },
                "loss_limits": {
                    "max_daily_loss_pct": limits.max_daily_loss_pct,
                    "max_weekly_loss_pct": limits.max_weekly_loss_pct,
                    "max_monthly_loss_pct": limits.max_monthly_loss_pct,
                    "max_drawdown_pct": limits.max_drawdown_pct,
                    "max_loss_per_trade_pct": limits.max_loss_per_trade_pct
                },
                "leverage_limits": {
                    "max_leverage": limits.max_leverage,
                    "max_options_notional_pct": limits.max_options_notional_pct
                },
                "concentration_limits": {
                    "max_positions": limits.max_positions,
                    "min_positions": limits.min_positions,
                    "max_same_expiration_pct": limits.max_same_expiration_pct
                },
                "volatility_limits": {
                    "max_portfolio_volatility": limits.max_portfolio_volatility,
                    "max_position_beta": limits.max_position_beta
                },
                "liquidity_requirements": {
                    "min_daily_volume": limits.min_daily_volume,
                    "min_open_interest": limits.min_open_interest,
                    "max_bid_ask_spread_pct": limits.max_bid_ask_spread_pct
                },
                "capital_requirements": {
                    "min_cash_reserve_pct": limits.min_cash_reserve_pct,
                    "max_capital_deployment_pct": limits.max_capital_deployment_pct
                }
            }
        }

    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid risk level: {risk_level}")
    except Exception as e:
        logger.error(f"Failed to get risk limits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/risk/position-sizing")
async def get_position_sizing(
    kelly_fraction: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    portfolio_value: float,
    risk_level: str = Query("moderate")
) -> Dict[str, Any]:
    """
    Get position sizing recommendation based on Kelly Criterion and risk limits.

    Returns full Kelly, half Kelly, and recommended position sizes.
    """
    try:
        risk_level_enum = RiskLevel(risk_level.lower())
        risk_service = RiskGuardrailsService(risk_level_enum)

        sizing = risk_service.get_position_sizing_recommendation(
            kelly_fraction=kelly_fraction,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            portfolio_value=portfolio_value
        )

        return sizing

    except Exception as e:
        logger.error(f"Position sizing calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Expert Critique Endpoints
# ============================================================================

@router.get("/critique/platform")
async def get_platform_critique() -> Dict[str, Any]:
    """
    Get comprehensive expert critique of the entire platform.

    Returns institutional investor perspective analysis with recommendations.
    """
    try:
        critique_service = ExpertCritiqueService()
        report = await critique_service.generate_critique()

        return {
            "overall_rating": report.overall_rating,
            "overall_score": report.overall_score,
            "executive_summary": report.executive_summary,
            "competitive_positioning": report.competitive_positioning,
            "competitive_scores": {
                "vs_bloomberg": report.vs_bloomberg_score,
                "vs_refinitiv": report.vs_refinitiv_score,
                "vs_factset": report.vs_factset_score
            },
            "category_scores": {
                "data_quality": report.data_quality_score,
                "analytics": report.analytics_score,
                "execution": report.execution_score,
                "risk_management": report.risk_management_score,
                "user_experience": report.user_experience_score,
                "technology": report.technology_score
            },
            "strengths": report.strengths,
            "critical_gaps": report.critical_gaps,
            "recommendations": [
                {
                    "priority": r.priority,
                    "category": r.category,
                    "title": r.title,
                    "current_state": r.current_state,
                    "desired_state": r.desired_state,
                    "rationale": r.rationale,
                    "expected_impact": r.expected_impact,
                    "implementation_complexity": r.implementation_complexity,
                    "estimated_value": r.estimated_value
                }
                for r in report.recommendations
            ],
            "generated_at": report.generated_at
        }

    except Exception as e:
        logger.error(f"Platform critique failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health")
async def ai_services_health() -> Dict[str, str]:
    """Health check for AI services"""
    return {
        "status": "healthy",
        "services": "swarm_analysis, risk_management, expert_critique"
    }
