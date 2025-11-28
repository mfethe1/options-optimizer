"""
Swarm Analysis API Routes

Provides endpoints for multi-agent swarm portfolio analysis and recommendations.
"""

from fastapi import APIRouter, HTTPException, Depends, Request, UploadFile, File, Query
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import io

from src.agents.swarm import (
    SwarmCoordinator,
    MarketAnalystAgent,
    RiskManagerAgent,
    OptionsStrategistAgent,
    TechnicalAnalystAgent,
    SentimentAnalystAgent,
    PortfolioOptimizerAgent,
    TradeExecutorAgent,
    ComplianceOfficerAgent
)
# Import LLM-powered agents
from src.agents.swarm.agents.llm_market_analyst import LLMMarketAnalystAgent
from src.agents.swarm.agents.llm_risk_manager import LLMRiskManagerAgent
from src.agents.swarm.agents.llm_sentiment_analyst import LLMSentimentAnalystAgent
from src.agents.swarm.agents.llm_fundamental_analyst import LLMFundamentalAnalystAgent
from src.agents.swarm.agents.llm_macro_economist import LLMMacroEconomistAgent
from src.agents.swarm.agents.llm_volatility_specialist import LLMVolatilitySpecialistAgent
from src.agents.swarm.agents.llm_recommendation_agent import LLMRecommendationAgent
from src.agents.swarm.agents.swarm_overseer import SwarmOverseerAgent
from src.agents.swarm.consensus_engine import ConsensusMethod
from src.data.position_manager import PositionManager
from src.api.dependencies import require_trader, require_viewer, get_current_active_user
from src.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/swarm", tags=["swarm"])

# Import limiter for rate limiting
# Note: Rate limiting is now handled by middleware in rate_limiting.py
# The limiter decorator below is a compatibility shim (no-op)
# Actual rate limits are enforced by RateLimitMiddleware based on path patterns
try:
    from src.api.rate_limiting import limiter
    RATE_LIMITING_ENABLED = True
except ImportError:
    logger.warning("Rate limiting not available")
    RATE_LIMITING_ENABLED = False
    limiter = None

# Global swarm coordinator (singleton)
_swarm_coordinator: Optional[SwarmCoordinator] = None


def get_swarm_coordinator() -> SwarmCoordinator:
    """Get or create the swarm coordinator"""
    global _swarm_coordinator
    
    if _swarm_coordinator is None:
        logger.info("Initializing swarm coordinator...")
        
        # Create coordinator
        _swarm_coordinator = SwarmCoordinator(
            name="OptionsAnalysisSwarm",
            max_messages=1000,
            quorum_threshold=0.67
        )
        
        # Create and register all agents (17-AGENT INSTITUTIONAL-GRADE SWARM!)
        # Model distribution: ~70% LMStudio (local), ~30% Claude/GPT-4 (cloud)
        # NEW: Added LLMRecommendationAgent for intelligent position replacement suggestions
        agents = [
            # === TIER 1: OVERSIGHT & COORDINATION (Claude - Critical) ===
            SwarmOverseerAgent(
                agent_id="swarm_overseer",
                shared_context=_swarm_coordinator.shared_context,
                consensus_engine=_swarm_coordinator.consensus_engine,
                preferred_model="anthropic"  # Claude for oversight
            ),

            # === TIER 2: MARKET INTELLIGENCE (Mixed Models) ===
            # Market Analysis Team (3 agents - diverse perspectives)
            LLMMarketAnalystAgent(
                agent_id="market_analyst_claude",
                shared_context=_swarm_coordinator.shared_context,
                consensus_engine=_swarm_coordinator.consensus_engine,
                preferred_model="anthropic"  # Claude - 30%
            ),
            LLMMarketAnalystAgent(
                agent_id="market_analyst_local_1",
                shared_context=_swarm_coordinator.shared_context,
                consensus_engine=_swarm_coordinator.consensus_engine,
                preferred_model="lmstudio"  # LMStudio - 70%
            ),
            LLMMarketAnalystAgent(
                agent_id="market_analyst_local_2",
                shared_context=_swarm_coordinator.shared_context,
                consensus_engine=_swarm_coordinator.consensus_engine,
                preferred_model="lmstudio"  # LMStudio - 70%
            ),

            # === TIER 3: FUNDAMENTAL & MACRO ANALYSIS (LMStudio - Volume) ===
            LLMFundamentalAnalystAgent(
                agent_id="fundamental_analyst_1",
                shared_context=_swarm_coordinator.shared_context,
                consensus_engine=_swarm_coordinator.consensus_engine,
                preferred_model="lmstudio"  # LMStudio - 70%
            ),
            LLMFundamentalAnalystAgent(
                agent_id="fundamental_analyst_2",
                shared_context=_swarm_coordinator.shared_context,
                consensus_engine=_swarm_coordinator.consensus_engine,
                preferred_model="lmstudio"  # LMStudio - 70%
            ),
            LLMMacroEconomistAgent(
                agent_id="macro_economist_1",
                shared_context=_swarm_coordinator.shared_context,
                consensus_engine=_swarm_coordinator.consensus_engine,
                preferred_model="lmstudio"  # LMStudio - 70%
            ),
            LLMMacroEconomistAgent(
                agent_id="macro_economist_2",
                shared_context=_swarm_coordinator.shared_context,
                consensus_engine=_swarm_coordinator.consensus_engine,
                preferred_model="lmstudio"  # LMStudio - 70%
            ),

            # === TIER 4: RISK & SENTIMENT (Mixed) ===
            LLMRiskManagerAgent(
                agent_id="risk_manager_claude",
                shared_context=_swarm_coordinator.shared_context,
                consensus_engine=_swarm_coordinator.consensus_engine,
                max_portfolio_delta=100.0,
                max_position_size_pct=0.10,
                max_drawdown_pct=0.15,
                preferred_model="anthropic"  # Claude - 30%
            ),
            LLMRiskManagerAgent(
                agent_id="risk_manager_local",
                shared_context=_swarm_coordinator.shared_context,
                consensus_engine=_swarm_coordinator.consensus_engine,
                max_portfolio_delta=100.0,
                max_position_size_pct=0.10,
                max_drawdown_pct=0.15,
                preferred_model="lmstudio"  # LMStudio - 70%
            ),
            LLMSentimentAnalystAgent(
                agent_id="sentiment_analyst_local",
                shared_context=_swarm_coordinator.shared_context,
                consensus_engine=_swarm_coordinator.consensus_engine,
                preferred_model="lmstudio"  # LMStudio - 70%
            ),

            # === TIER 5: OPTIONS & VOLATILITY SPECIALISTS (LMStudio) ===
            LLMVolatilitySpecialistAgent(
                agent_id="volatility_specialist_1",
                shared_context=_swarm_coordinator.shared_context,
                consensus_engine=_swarm_coordinator.consensus_engine,
                preferred_model="lmstudio"  # LMStudio - 70%
            ),
            LLMVolatilitySpecialistAgent(
                agent_id="volatility_specialist_2",
                shared_context=_swarm_coordinator.shared_context,
                consensus_engine=_swarm_coordinator.consensus_engine,
                preferred_model="lmstudio"  # LMStudio - 70%
            ),
            OptionsStrategistAgent(
                agent_id="options_strategist_1",
                shared_context=_swarm_coordinator.shared_context,
                consensus_engine=_swarm_coordinator.consensus_engine
            ),

            # === TIER 6: EXECUTION & COMPLIANCE (Rule-based) ===
            TechnicalAnalystAgent(
                agent_id="technical_analyst_1",
                shared_context=_swarm_coordinator.shared_context,
                consensus_engine=_swarm_coordinator.consensus_engine
            ),
            PortfolioOptimizerAgent(
                agent_id="portfolio_optimizer_1",
                shared_context=_swarm_coordinator.shared_context,
                consensus_engine=_swarm_coordinator.consensus_engine
            ),

            # === TIER 7: RECOMMENDATION ENGINE (LLM-powered) ===
            # NEW: Intelligent replacement recommendations for underperforming positions
            LLMRecommendationAgent(
                agent_id="recommendation_agent_1",
                shared_context=_swarm_coordinator.shared_context,
                consensus_engine=_swarm_coordinator.consensus_engine,
                preferred_model="anthropic"  # Claude for intelligent recommendations
            )
        ]
        
        for agent in agents:
            _swarm_coordinator.register_agent(agent)
        
        # Start swarm
        _swarm_coordinator.start()
        
        logger.info(f"Swarm coordinator initialized with {len(agents)} agents")
    
    return _swarm_coordinator


class SwarmAnalysisRequest(BaseModel):
    """Request model for swarm analysis"""
    portfolio_data: Optional[Dict[str, Any]] = None
    market_data: Optional[Dict[str, Any]] = None
    consensus_method: str = "weighted"


class SwarmAnalysisResponse(BaseModel):
    """Response model for swarm analysis"""
    swarm_name: str
    timestamp: str
    analysis: Dict[str, Any]
    recommendations: Dict[str, Any]
    metrics: Dict[str, Any]


@router.post("/analyze")
@limiter.limit("5/minute")
async def analyze_portfolio_with_swarm(
    swarm_request: SwarmAnalysisRequest,
    request: Request,
    coordinator: SwarmCoordinator = Depends(get_swarm_coordinator),
    current_user: User = Depends(require_trader())
):
    """
    Analyze portfolio using multi-agent swarm.

    **Authentication Required**: Trader or Admin role
    **Rate Limited**: 5 requests per minute

    The swarm will:
    1. Analyze market conditions (MarketAnalyst)
    2. Assess portfolio risk (RiskManager)
    3. Recommend options strategies (OptionsStrategist)
    4. Analyze technical indicators (TechnicalAnalyst)
    5. Assess sentiment (SentimentAnalyst)
    6. Optimize portfolio allocation (PortfolioOptimizer)
    7. Simulate trade execution (TradeExecutor)
    8. Verify compliance (ComplianceOfficer)

    Then reach consensus on trading recommendations.
    """
    try:
        logger.info("Starting swarm analysis...")

        # Get portfolio data
        portfolio_data = swarm_request.portfolio_data
        if not portfolio_data:
            # Load from position manager
            position_manager = PositionManager()
            stock_positions = position_manager.get_all_stock_positions()
            option_positions = position_manager.get_all_option_positions()

            # Convert to dict format
            positions = []
            for pos in stock_positions:
                positions.append({
                    'symbol': pos.symbol,
                    'asset_type': 'stock',
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'market_value': pos.market_value if hasattr(pos, 'market_value') else (pos.current_price or 0) * pos.quantity,
                    'unrealized_pnl': pos.unrealized_pnl if hasattr(pos, 'unrealized_pnl') else 0
                })

            for pos in option_positions:
                positions.append({
                    'symbol': pos.symbol,
                    'asset_type': 'option',
                    'option_type': pos.option_type,
                    'strike': pos.strike,
                    'expiration_date': pos.expiration_date.isoformat() if pos.expiration_date else None,
                    'quantity': pos.quantity,
                    'premium_paid': pos.premium_paid,
                    'current_price': pos.current_price,
                    'underlying_price': pos.underlying_price,
                    'delta': pos.delta,
                    'gamma': pos.gamma,
                    'theta': pos.theta,
                    'vega': pos.vega,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl
                })

            total_value = sum(p.get('market_value', 0) for p in positions)
            unrealized_pnl = sum(p.get('unrealized_pnl', 0) for p in positions)

            portfolio_data = {
                'positions': positions,
                'total_value': total_value,
                'unrealized_pnl': unrealized_pnl,
                'initial_value': total_value - unrealized_pnl,
                'peak_value': total_value
            }
        
        # Get market data
        market_data = swarm_request.market_data or {}

        # Run swarm analysis
        analysis = coordinator.analyze_portfolio(portfolio_data, market_data)
        
        # Get consensus method
        consensus_method_map = {
            'majority': ConsensusMethod.MAJORITY,
            'weighted': ConsensusMethod.WEIGHTED,
            'unanimous': ConsensusMethod.UNANIMOUS,
            'quorum': ConsensusMethod.QUORUM,
            'entropy': ConsensusMethod.ENTROPY_BASED
        }
        consensus_method = consensus_method_map.get(
            swarm_request.consensus_method.lower(),
            ConsensusMethod.WEIGHTED
        )
        
        # Get recommendations
        recommendations = coordinator.make_recommendations(
            analysis,
            consensus_method=consensus_method
        )
        
        # Get metrics
        metrics = coordinator.get_swarm_metrics()
        
        logger.info("Swarm analysis complete")

        # Clean up NaN values and numpy types for JSON serialization
        import json
        import math
        import numpy as np

        def clean_nan(obj):
            """Recursively replace NaN/numpy types with JSON-serializable values"""
            if isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return clean_nan(obj.tolist())
            elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            else:
                return obj

        analysis = clean_nan(analysis)
        recommendations = clean_nan(recommendations)
        metrics = clean_nan(metrics)

        # Return as JSON directly to avoid Pydantic serialization issues
        from fastapi.responses import JSONResponse

        return JSONResponse(content={
            'swarm_name': coordinator.name,
            'timestamp': datetime.utcnow().isoformat(),
            'analysis': analysis,
            'recommendations': recommendations,
            'metrics': metrics
        })
        
    except Exception as e:
        logger.error(f"Error in swarm analysis: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_swarm_metrics(
    coordinator: SwarmCoordinator = Depends(get_swarm_coordinator)
):
    """Get comprehensive swarm metrics"""
    try:
        metrics = coordinator.get_swarm_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting swarm metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_swarm_status(
    coordinator: SwarmCoordinator = Depends(get_swarm_coordinator),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get swarm status.

    **Authentication Required**: Any authenticated user
    """
    try:
        return {
            'swarm_name': coordinator.name,
            'is_running': coordinator.is_running,
            'total_agents': len(coordinator.agents),
            'agent_types': {k: len(v) for k, v in coordinator.agent_types.items()},
            'start_time': coordinator.start_time.isoformat() if coordinator.start_time else None
        }
    except Exception as e:
        logger.error(f"Error getting swarm status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restart")
async def restart_swarm(
    coordinator: SwarmCoordinator = Depends(get_swarm_coordinator)
):
    """Restart the swarm"""
    try:
        logger.info("Restarting swarm...")
        coordinator.stop()
        coordinator.start()
        logger.info("Swarm restarted")
        
        return {
            'status': 'restarted',
            'swarm_name': coordinator.name,
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error restarting swarm: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents")
async def list_agents(
    coordinator: SwarmCoordinator = Depends(get_swarm_coordinator),
    current_user: User = Depends(get_current_active_user)
):
    """
    List all agents in the swarm.

    **Authentication Required**: Any authenticated user
    """
    try:
        agents_info = []
        
        for agent_id, agent in coordinator.agents.items():
            metrics = agent.get_metrics()
            agents_info.append({
                'agent_id': agent_id,
                'agent_type': agent.agent_type,
                'is_active': agent.is_active,
                'priority': agent.priority,
                'confidence_threshold': agent.confidence_threshold,
                'metrics': metrics
            })
        
        return {
            'total_agents': len(agents_info),
            'agents': agents_info
        }
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/messages")
async def get_swarm_messages(
    min_priority: int = 0,
    min_confidence: float = 0.0,
    max_age_seconds: Optional[int] = None,
    coordinator: SwarmCoordinator = Depends(get_swarm_coordinator)
):
    """Get messages from shared context"""
    try:
        messages = coordinator.shared_context.get_messages(
            min_priority=min_priority,
            min_confidence=min_confidence,
            max_age_seconds=max_age_seconds
        )
        
        return {
            'total_messages': len(messages),
            'messages': [
                {
                    'source': msg.source,
                    'content': msg.content,
                    'priority': msg.priority,
                    'confidence': msg.confidence,
                    'timestamp': msg.timestamp.isoformat()
                }
                for msg in messages
            ]
        }
    except Exception as e:
        logger.error(f"Error getting messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def build_institutional_response(
    consensus_decisions: Dict[str, Any],
    analysis: Dict[str, Any],
    recommendations: Dict[str, Any],
    portfolio_data: Dict[str, Any],
    positions: List[Dict[str, Any]],
    results: Dict[str, Any],
    metrics: Dict[str, Any],
    coordinator: SwarmCoordinator
) -> Dict[str, Any]:
    """
    Build institutional-grade portfolio analysis response with full agent insights.

    Returns comprehensive analysis including:
    - Full LLM responses from each agent
    - Position-by-position breakdown
    - Swarm health metrics
    - Enhanced consensus with vote breakdown
    - Agent-to-agent discussion logs
    """

    # 1. AGENT INSIGHTS - Full LLM outputs and detailed analysis
    agent_insights = []
    analyses_dict = analysis.get('analyses', {})
    recommendations_dict = recommendations.get('recommendations', {})

    for agent_id, agent_data in analyses_dict.items():
        agent_analysis = agent_data.get('analysis', {})
        agent_rec = recommendations_dict.get(agent_id, {}).get('recommendation', {})

        insight = {
            'agent_id': agent_id,
            'agent_type': agent_data.get('agent_type', 'unknown'),
            'timestamp': agent_data.get('timestamp'),

            # Full LLM response text (the key missing piece!)
            'llm_response_text': agent_analysis.get('llm_response', ''),

            # Structured analysis fields
            'analysis_fields': {
                'outlook': agent_analysis.get('outlook'),
                'valuation_level': agent_analysis.get('valuation_level'),
                'quality_score': agent_analysis.get('quality_score'),
                'key_insights': agent_analysis.get('key_insights', []),
                'symbols_analyzed': agent_analysis.get('symbols_analyzed', []),
                'risk_assessment': agent_analysis.get('risk_assessment'),
                'sentiment': agent_analysis.get('sentiment'),
                'volatility_regime': agent_analysis.get('volatility_regime'),
                'macro_outlook': agent_analysis.get('macro_outlook'),
                'cycle_phase': agent_analysis.get('cycle_phase'),
                'fed_stance': agent_analysis.get('fed_stance'),
                'recession_risk': agent_analysis.get('recession_risk')
            },

            # Individual agent recommendation
            'recommendation': {
                'overall_action': agent_rec.get('overall_action', {}),
                'risk_level': agent_rec.get('risk_level', {}),
                'market_outlook': agent_rec.get('market_outlook', {}),
                'confidence': agent_rec.get('overall_action', {}).get('confidence', 0),
                'reasoning': agent_rec.get('overall_action', {}).get('reasoning', '')
            },

            # Error info if any
            'error': agent_analysis.get('error')
        }

        agent_insights.append(insight)

    # 2. POSITION-BY-POSITION ANALYSIS (ENHANCED)
    position_analysis = []
    for pos in positions:
        # Extract agent insights for this position
        position_insights = _extract_position_insights(pos, agent_insights)

        # Build comprehensive stock report from all agent insights
        stock_report = _build_comprehensive_stock_report(
            symbol=pos.get('symbol'),
            position_insights=position_insights,
            agent_insights=agent_insights
        )

        pos_analysis = {
            'symbol': pos.get('symbol'),
            'asset_type': pos.get('asset_type'),
            'option_type': pos.get('option_type'),
            'strike': pos.get('strike'),
            'expiration_date': pos.get('expiration_date'),
            'quantity': pos.get('quantity'),

            # Current metrics
            'current_metrics': {
                'current_price': pos.get('current_price'),
                'underlying_price': pos.get('underlying_price'),
                'market_value': pos.get('market_value'),
                'unrealized_pnl': pos.get('unrealized_pnl'),
                'unrealized_pnl_pct': (pos.get('unrealized_pnl', 0) / pos.get('premium_paid', 1) * 100) if pos.get('premium_paid', 0) > 0 else 0,
                'days_to_expiry': pos.get('days_to_expiry'),
                'iv': pos.get('iv')
            },

            # Greeks
            'greeks': {
                'delta': pos.get('delta'),
                'gamma': pos.get('gamma'),
                'theta': pos.get('theta'),
                'vega': pos.get('vega')
            },

            # Agent-specific insights for this position
            'agent_insights_for_position': position_insights,

            # NEW: Comprehensive stock report assembled from all agents
            'comprehensive_stock_report': stock_report,

            # Risk warnings and opportunities
            'risk_warnings': _generate_risk_warnings(pos),
            'opportunities': _generate_opportunities(pos),

            # NEW: Replacement recommendations (if available from RecommendationAgent)
            'replacement_recommendations': _extract_replacement_recommendations(
                pos.get('symbol'),
                analysis
            )
        }

        position_analysis.append(pos_analysis)

    # 3. SWARM HEALTH METRICS
    contributed_agents = len([a for a in agent_insights if not a.get('error')])
    failed_agents = len([a for a in agent_insights if a.get('error')])

    # Get messages from shared context
    messages = coordinator.shared_context.get_messages()
    context_metrics = coordinator.shared_context.get_metrics()

    swarm_health = {
        'active_agents_count': len(coordinator.agents),
        'contributed_vs_failed': {
            'contributed': contributed_agents,
            'failed': failed_agents,
            'success_rate': (contributed_agents / len(agent_insights) * 100) if agent_insights else 0
        },
        'communication_stats': {
            'total_messages': len(messages),
            'total_state_updates': context_metrics.get('total_state_updates', 0),
            'average_message_priority': sum(m.priority for m in messages) / len(messages) if messages else 0,
            'average_confidence': sum(m.confidence for m in messages) / len(messages) if messages else 0
        },
        'consensus_strength': {
            'overall_action_confidence': consensus_decisions.get('overall_action', {}).get('confidence', 0),
            'risk_level_confidence': consensus_decisions.get('risk_level', {}).get('confidence', 0),
            'market_outlook_confidence': consensus_decisions.get('market_outlook', {}).get('confidence', 0),
            'average_confidence': sum([
                consensus_decisions.get('overall_action', {}).get('confidence', 0),
                consensus_decisions.get('risk_level', {}).get('confidence', 0),
                consensus_decisions.get('market_outlook', {}).get('confidence', 0)
            ]) / 3
        }
    }

    # 4. ENHANCED CONSENSUS with vote breakdown
    enhanced_consensus = _build_enhanced_consensus(
        recommendations_dict,
        consensus_decisions,
        agent_insights
    )

    # 5. DISCUSSION LOGS - Agent-to-agent messages
    discussion_logs = []
    for msg in messages[-50:]:  # Last 50 messages
        discussion_logs.append({
            'source_agent': msg.source,
            'content': msg.content,
            'priority': msg.priority,
            'confidence': msg.confidence,
            'timestamp': msg.timestamp.isoformat()
        })

    # Build final response
    response = {
        # Backward compatible - existing frontend format
        'consensus_decisions': consensus_decisions,

        # NEW: Full agent insights with LLM responses
        'agent_insights': agent_insights,

        # NEW: Position-by-position breakdown
        'position_analysis': position_analysis,

        # NEW: Swarm health metrics
        'swarm_health': swarm_health,

        # NEW: Enhanced consensus with vote breakdown
        'enhanced_consensus': enhanced_consensus,

        # NEW: Agent-to-agent discussion logs
        'discussion_logs': discussion_logs,

        # Existing fields (backward compatible)
        'portfolio_summary': {
            'total_value': portfolio_data['total_value'],
            'total_unrealized_pnl': portfolio_data['unrealized_pnl'],
            'total_unrealized_pnl_pct': (portfolio_data['unrealized_pnl'] / portfolio_data['initial_value'] * 100) if portfolio_data['initial_value'] > 0 else 0,
            'positions_count': len(positions)
        },
        'import_stats': {
            'positions_imported': results['success'],
            'positions_failed': results['failed'],
            'chase_conversion': results.get('chase_conversion'),
            'errors': results['errors']
        },
        'execution_time': metrics.get('total_execution_time', 0),
        'timestamp': datetime.utcnow().isoformat()
    }

    # Add investor report if available (from Distillation Agent)
    investor_report = analysis.get('investor_report')
    if investor_report:
        response['investor_report'] = investor_report

    return response


def _extract_position_insights(position: Dict[str, Any], agent_insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract agent insights specific to this position.

    Enhanced to parse full LLM responses and extract stock-specific analysis.
    """
    symbol = position.get('symbol', '')
    insights = []

    for agent in agent_insights:
        # Check if agent analyzed this symbol
        symbols_analyzed = agent.get('analysis_fields', {}).get('symbols_analyzed', [])
        if symbol not in symbols_analyzed:
            continue

        # Extract stock-specific text from LLM response
        llm_response = agent.get('llm_response_text', '')
        stock_specific_analysis = _extract_stock_specific_text(symbol, llm_response)

        insight = {
            'agent_id': agent.get('agent_id'),
            'agent_type': agent.get('agent_type'),
            'key_insights': agent.get('analysis_fields', {}).get('key_insights', []),
            'recommendation': agent.get('recommendation', {}).get('overall_action', {}).get('choice'),
            'confidence': agent.get('recommendation', {}).get('overall_action', {}).get('confidence', 0),

            # NEW: Full stock-specific analysis from LLM
            'stock_specific_analysis': stock_specific_analysis,

            # NEW: Structured fields specific to this stock
            'stock_metrics': _extract_stock_metrics(symbol, llm_response, agent.get('agent_type'))
        }

        insights.append(insight)

    return insights


def _extract_stock_specific_text(symbol: str, llm_response: str) -> str:
    """
    Extract stock-specific sections from LLM response.

    Looks for sections that mention the symbol and extracts surrounding context.
    """
    if not llm_response or not symbol:
        return ""

    lines = llm_response.split('\n')
    stock_lines = []
    in_stock_section = False
    section_buffer = []

    for i, line in enumerate(lines):
        # Check if line mentions the symbol
        if symbol.upper() in line.upper():
            in_stock_section = True
            # Include previous 2 lines for context
            if i > 0:
                section_buffer.append(lines[i-1])
            if i > 1:
                section_buffer.insert(0, lines[i-2])
            section_buffer.append(line)
        elif in_stock_section:
            # Continue collecting lines until we hit a new section or empty line
            if line.strip() == '' or line.strip().startswith('**') or line.strip().startswith('##'):
                # End of section
                stock_lines.extend(section_buffer)
                stock_lines.append('')  # Add separator
                section_buffer = []
                in_stock_section = False
            else:
                section_buffer.append(line)

    # Add any remaining buffer
    if section_buffer:
        stock_lines.extend(section_buffer)

    return '\n'.join(stock_lines).strip()


def _extract_stock_metrics(symbol: str, llm_response: str, agent_type: str) -> Dict[str, Any]:
    """
    Extract structured metrics for a specific stock from LLM response.

    Different agent types provide different metrics:
    - Fundamental: P/E, revenue growth, margins, moat
    - Market: momentum, technical levels, volume
    - Risk: volatility, correlation, max drawdown
    - Sentiment: sentiment score, news tone
    """
    metrics = {}

    if not llm_response or not symbol:
        return metrics

    # Find stock-specific section
    stock_text = _extract_stock_specific_text(symbol, llm_response)
    if not stock_text:
        return metrics

    stock_text_lower = stock_text.lower()

    # Extract metrics based on agent type
    if 'Fundamental' in agent_type:
        # Extract fundamental metrics
        metrics['pe_ratio'] = _extract_number(stock_text, ['p/e:', 'pe:', 'p/e ratio:'])
        metrics['revenue_growth'] = _extract_percentage(stock_text, ['revenue growth:', 'sales growth:'])
        metrics['market_cap'] = _extract_market_cap(stock_text)
        metrics['rating'] = _extract_rating(stock_text)
        metrics['valuation'] = _extract_valuation(stock_text)

    elif 'Market' in agent_type:
        # Extract market/technical metrics
        metrics['momentum'] = _extract_momentum(stock_text)
        metrics['support_level'] = _extract_number(stock_text, ['support:', 'support at'])
        metrics['resistance_level'] = _extract_number(stock_text, ['resistance:', 'resistance at'])
        metrics['rsi'] = _extract_number(stock_text, ['rsi:', 'rsi at'])

    elif 'Risk' in agent_type:
        # Extract risk metrics
        metrics['risk_level'] = _extract_risk_level(stock_text)
        metrics['volatility'] = _extract_percentage(stock_text, ['volatility:', 'vol:'])
        metrics['max_drawdown'] = _extract_percentage(stock_text, ['max drawdown:', 'drawdown:'])

    elif 'Sentiment' in agent_type:
        # Extract sentiment metrics
        metrics['sentiment'] = _extract_sentiment(stock_text)
        metrics['news_tone'] = _extract_news_tone(stock_text)

    # Remove None values
    metrics = {k: v for k, v in metrics.items() if v is not None}

    return metrics


def _extract_number(text: str, patterns: List[str]) -> Optional[float]:
    """Extract a number following specific patterns"""
    import re
    for pattern in patterns:
        # Look for pattern followed by number
        regex = rf'{re.escape(pattern)}\s*(\$)?(\d+\.?\d*)'
        match = re.search(regex, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(2))
            except:
                pass
    return None


def _extract_percentage(text: str, patterns: List[str]) -> Optional[float]:
    """Extract a percentage following specific patterns"""
    import re
    for pattern in patterns:
        regex = rf'{re.escape(pattern)}\s*(\+|-)?(\d+\.?\d*)%?'
        match = re.search(regex, text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(2))
                if match.group(1) == '-':
                    value = -value
                return value
            except:
                pass
    return None


def _extract_market_cap(text: str) -> Optional[str]:
    """Extract market cap (e.g., '$1.2T', '$500B')"""
    import re
    match = re.search(r'\$(\d+\.?\d*)\s*([TBM])', text, re.IGNORECASE)
    if match:
        return f"${match.group(1)}{match.group(2).upper()}"
    return None


def _extract_rating(text: str) -> Optional[str]:
    """Extract rating (e.g., 'STRONG BUY', 'HOLD', 'SELL')"""
    text_upper = text.upper()
    ratings = ['STRONG BUY', 'BUY', 'HOLD', 'SELL', 'STRONG SELL']
    for rating in ratings:
        if rating in text_upper:
            return rating
    return None


def _extract_valuation(text: str) -> Optional[str]:
    """Extract valuation assessment"""
    text_lower = text.lower()
    if 'undervalued' in text_lower:
        return 'undervalued'
    elif 'overvalued' in text_lower:
        return 'overvalued'
    elif 'fair' in text_lower or 'fairly valued' in text_lower:
        return 'fair'
    return None


def _extract_momentum(text: str) -> Optional[str]:
    """Extract momentum assessment"""
    text_lower = text.lower()
    if 'strong upward' in text_lower or 'bullish momentum' in text_lower:
        return 'strong_bullish'
    elif 'upward' in text_lower or 'positive momentum' in text_lower:
        return 'bullish'
    elif 'downward' in text_lower or 'bearish momentum' in text_lower:
        return 'bearish'
    elif 'sideways' in text_lower or 'neutral momentum' in text_lower:
        return 'neutral'
    return None


def _extract_risk_level(text: str) -> Optional[str]:
    """Extract risk level assessment"""
    text_lower = text.lower()
    if 'very high risk' in text_lower or 'extreme risk' in text_lower:
        return 'very_high'
    elif 'high risk' in text_lower:
        return 'high'
    elif 'moderate risk' in text_lower:
        return 'moderate'
    elif 'low risk' in text_lower:
        return 'low'
    return None


def _extract_sentiment(text: str) -> Optional[str]:
    """Extract sentiment assessment"""
    text_lower = text.lower()
    if 'very positive' in text_lower or 'extremely positive' in text_lower:
        return 'very_positive'
    elif 'positive' in text_lower:
        return 'positive'
    elif 'negative' in text_lower:
        return 'negative'
    elif 'neutral' in text_lower:
        return 'neutral'
    return None


def _extract_news_tone(text: str) -> Optional[str]:
    """Extract news tone assessment"""
    text_lower = text.lower()
    if 'positive news' in text_lower or 'favorable coverage' in text_lower:
        return 'positive'
    elif 'negative news' in text_lower or 'unfavorable coverage' in text_lower:
        return 'negative'
    elif 'mixed news' in text_lower:
        return 'mixed'
    return None


def _build_comprehensive_stock_report(
    symbol: str,
    position_insights: List[Dict[str, Any]],
    agent_insights: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build comprehensive stock report by assembling insights from all agents.

    This creates a unified view of what all agents think about this specific stock.
    """
    report = {
        'symbol': symbol,
        'analysis_by_category': {},
        'consensus_view': {},
        'key_metrics': {},
        'full_analysis_text': ""
    }

    # Organize insights by agent category
    categories = {
        'fundamental': [],
        'market_technical': [],
        'risk': [],
        'sentiment': [],
        'macro': [],
        'volatility': [],
        'other': []
    }

    for insight in position_insights:
        agent_type = insight.get('agent_type', '')
        category = 'other'

        if 'Fundamental' in agent_type:
            category = 'fundamental'
        elif 'Market' in agent_type or 'Technical' in agent_type:
            category = 'market_technical'
        elif 'Risk' in agent_type:
            category = 'risk'
        elif 'Sentiment' in agent_type:
            category = 'sentiment'
        elif 'Macro' in agent_type or 'Economist' in agent_type:
            category = 'macro'
        elif 'Volatility' in agent_type:
            category = 'volatility'

        categories[category].append(insight)

    # Build analysis by category
    full_text_parts = []

    for category, insights in categories.items():
        if not insights:
            continue

        category_analysis = {
            'agent_count': len(insights),
            'agents': [],
            'summary': ""
        }

        category_text = f"\n## {category.upper().replace('_', ' ')} ANALYSIS\n\n"

        for insight in insights:
            agent_info = {
                'agent_id': insight.get('agent_id'),
                'agent_type': insight.get('agent_type'),
                'recommendation': insight.get('recommendation'),
                'confidence': insight.get('confidence'),
                'key_insights': insight.get('key_insights', []),
                'stock_metrics': insight.get('stock_metrics', {})
            }
            category_analysis['agents'].append(agent_info)

            # Add to full text
            stock_analysis = insight.get('stock_specific_analysis', '')
            if stock_analysis:
                category_text += f"**{insight.get('agent_type')}:**\n{stock_analysis}\n\n"

        # Create category summary
        recommendations = [a['recommendation'] for a in category_analysis['agents'] if a.get('recommendation')]
        if recommendations:
            buy_count = recommendations.count('buy')
            hold_count = recommendations.count('hold')
            sell_count = recommendations.count('sell')

            if buy_count > hold_count and buy_count > sell_count:
                category_analysis['summary'] = f"Bullish ({buy_count}/{len(recommendations)} agents recommend BUY)"
            elif sell_count > buy_count and sell_count > hold_count:
                category_analysis['summary'] = f"Bearish ({sell_count}/{len(recommendations)} agents recommend SELL)"
            else:
                category_analysis['summary'] = f"Neutral ({hold_count}/{len(recommendations)} agents recommend HOLD)"

        report['analysis_by_category'][category] = category_analysis
        full_text_parts.append(category_text)

    # Build consensus view
    all_recommendations = [i.get('recommendation') for i in position_insights if i.get('recommendation')]
    if all_recommendations:
        buy_pct = (all_recommendations.count('buy') / len(all_recommendations)) * 100
        hold_pct = (all_recommendations.count('hold') / len(all_recommendations)) * 100
        sell_pct = (all_recommendations.count('sell') / len(all_recommendations)) * 100

        report['consensus_view'] = {
            'total_agents': len(all_recommendations),
            'buy_percentage': buy_pct,
            'hold_percentage': hold_pct,
            'sell_percentage': sell_pct,
            'consensus_recommendation': 'buy' if buy_pct > 50 else ('sell' if sell_pct > 50 else 'hold'),
            'confidence': max(buy_pct, hold_pct, sell_pct)
        }

    # Aggregate key metrics from all agents
    all_metrics = {}
    for insight in position_insights:
        stock_metrics = insight.get('stock_metrics', {})
        for key, value in stock_metrics.items():
            if key not in all_metrics:
                all_metrics[key] = []
            all_metrics[key].append(value)

    # Average numeric metrics, take mode for categorical
    for key, values in all_metrics.items():
        if values:
            # Try to average if numeric
            try:
                numeric_values = [float(v) for v in values if v is not None]
                if numeric_values:
                    all_metrics[key] = sum(numeric_values) / len(numeric_values)
            except:
                # Take most common value for categorical
                all_metrics[key] = max(set(values), key=values.count)

    report['key_metrics'] = all_metrics

    # Assemble full text
    report['full_analysis_text'] = f"# COMPREHENSIVE ANALYSIS: {symbol}\n" + '\n'.join(full_text_parts)

    return report


def _extract_replacement_recommendations(
    symbol: str,
    analysis: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Extract replacement recommendations from RecommendationAgent if available.
    """
    analyses_dict = analysis.get('analyses', {})

    # Find RecommendationAgent analysis
    for agent_id, agent_data in analyses_dict.items():
        agent_type = agent_data.get('agent_type', '')
        if 'Recommendation' in agent_type:
            agent_analysis = agent_data.get('analysis', {})
            recommendations = agent_analysis.get('recommendations', [])

            # Find recommendation for this symbol
            for rec in recommendations:
                if rec.get('symbol') == symbol:
                    return {
                        'assessment': rec.get('assessment'),
                        'action': rec.get('action'),
                        'stock_alternative': rec.get('stock_alternative'),
                        'option_alternative': rec.get('option_alternative'),
                        'agent_id': agent_id
                    }

    return None


def _generate_risk_warnings(position: Dict[str, Any]) -> List[str]:
    """Generate risk warnings for a position"""
    warnings = []

    # Time decay risk
    days_to_expiry = position.get('days_to_expiry', 0)
    if days_to_expiry < 30:
        warnings.append(f"⚠️ High time decay risk - only {days_to_expiry} days to expiration")

    # Underwater position
    unrealized_pnl = position.get('unrealized_pnl', 0)
    if unrealized_pnl < 0:
        pnl_pct = (unrealized_pnl / position.get('premium_paid', 1) * 100) if position.get('premium_paid', 0) > 0 else 0
        warnings.append(f"📉 Position underwater: {pnl_pct:.1f}% loss")

    # High theta (time decay)
    theta = position.get('theta', 0)
    if theta < -10:
        warnings.append(f"⏰ High daily time decay: ${abs(theta):.2f}/day")

    # Low delta (far OTM)
    delta = abs(position.get('delta', 0))
    if delta < 0.2:
        warnings.append(f"🎯 Far out-of-the-money (delta: {delta:.2f})")

    return warnings


def _generate_opportunities(position: Dict[str, Any]) -> List[str]:
    """Generate opportunities for a position"""
    opportunities = []

    # Profitable position
    unrealized_pnl = position.get('unrealized_pnl', 0)
    if unrealized_pnl > 0:
        pnl_pct = (unrealized_pnl / position.get('premium_paid', 1) * 100) if position.get('premium_paid', 0) > 0 else 0
        if pnl_pct > 50:
            opportunities.append(f"💰 Strong profit: {pnl_pct:.1f}% gain - consider taking profits")

    # High vega (volatility play)
    vega = position.get('vega', 0)
    if vega > 5:
        opportunities.append(f"📊 High vega exposure: ${vega:.2f} per 1% IV change")

    # Deep ITM (high delta)
    delta = abs(position.get('delta', 0))
    if delta > 0.7:
        opportunities.append(f"✅ Deep in-the-money (delta: {delta:.2f}) - high probability of profit")

    # Long time to expiration
    days_to_expiry = position.get('days_to_expiry', 0)
    if days_to_expiry > 90:
        opportunities.append(f"⏳ Long time horizon: {days_to_expiry} days - low time decay pressure")

    return opportunities


def _build_enhanced_consensus(
    recommendations_dict: Dict[str, Any],
    consensus_decisions: Dict[str, Any],
    agent_insights: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build enhanced consensus with vote breakdown"""

    # Vote breakdown by agent
    vote_breakdown = {
        'overall_action': {},
        'risk_level': {},
        'market_outlook': {}
    }

    # Confidence distribution
    confidence_distribution = []

    # Dissenting opinions
    dissenting_opinions = []

    # Top contributors
    top_contributors = []

    for agent_id, rec_data in recommendations_dict.items():
        rec = rec_data.get('recommendation', {})

        # Overall action votes
        overall_action = rec.get('overall_action', {})
        if overall_action:
            choice = overall_action.get('choice', 'hold')
            confidence = overall_action.get('confidence', 0)
            vote_breakdown['overall_action'][agent_id] = {
                'choice': choice,
                'confidence': confidence,
                'agent_type': rec_data.get('agent_type')
            }

            confidence_distribution.append({
                'agent_id': agent_id,
                'agent_type': rec_data.get('agent_type'),
                'confidence': confidence
            })

            # Check for dissent
            consensus_choice = consensus_decisions.get('overall_action', {}).get('choice', 'hold')
            if choice != consensus_choice and confidence > 0.6:
                dissenting_opinions.append({
                    'agent_id': agent_id,
                    'agent_type': rec_data.get('agent_type'),
                    'dissenting_choice': choice,
                    'consensus_choice': consensus_choice,
                    'confidence': confidence,
                    'reasoning': overall_action.get('reasoning', '')
                })

        # Risk level votes
        risk_level = rec.get('risk_level', {})
        if risk_level:
            vote_breakdown['risk_level'][agent_id] = {
                'choice': risk_level.get('choice', 'moderate'),
                'confidence': risk_level.get('confidence', 0),
                'agent_type': rec_data.get('agent_type')
            }

        # Market outlook votes
        market_outlook = rec.get('market_outlook', {})
        if market_outlook:
            vote_breakdown['market_outlook'][agent_id] = {
                'choice': market_outlook.get('choice', 'neutral'),
                'confidence': market_outlook.get('confidence', 0),
                'agent_type': rec_data.get('agent_type')
            }

    # Sort by confidence and get top 5 contributors
    confidence_distribution.sort(key=lambda x: x['confidence'], reverse=True)
    top_contributors = confidence_distribution[:5]

    # Get reasoning from top contributors
    top_reasoning = []
    for contributor in top_contributors:
        agent_id = contributor['agent_id']
        agent_insight = next((a for a in agent_insights if a['agent_id'] == agent_id), None)
        if agent_insight:
            top_reasoning.append({
                'agent_id': agent_id,
                'agent_type': contributor['agent_type'],
                'confidence': contributor['confidence'],
                'reasoning': agent_insight.get('recommendation', {}).get('reasoning', ''),
                'llm_response_excerpt': agent_insight.get('llm_response_text', '')[:500]  # First 500 chars
            })

    return {
        'vote_breakdown_by_agent': vote_breakdown,
        'confidence_distribution': confidence_distribution,
        'dissenting_opinions': dissenting_opinions,
        'top_contributors': top_contributors,
        'reasoning_from_top_contributors': top_reasoning
    }


@router.post("/analyze-csv")
async def analyze_portfolio_from_csv(
    file: UploadFile = File(...),
    chase_format: bool = Query(False, description="Is this a Chase.com CSV export?"),
    consensus_method: str = Query("weighted", description="Consensus method to use")
):
    """
    Upload a CSV file and run LLM-powered swarm analysis on the portfolio.

    This endpoint:
    1. Accepts a CSV file (standard format or Chase.com export)
    2. Imports positions into the system
    3. Runs the multi-agent swarm analysis with LLM-powered agents
    4. Returns comprehensive AI-generated recommendations

    Args:
        file: CSV file containing positions
        chase_format: Set to true if uploading Chase.com export
        consensus_method: Consensus method (weighted, majority, unanimous, quorum, entropy)

    Returns:
        Complete swarm analysis with AI-powered recommendations
    """
    try:
        from src.data.csv_position_service import CSVPositionService
        from src.data.position_manager import PositionManager
        from src.data.position_enrichment_service import PositionEnrichmentService

        logger.info(f"Received CSV upload for swarm analysis (chase_format={chase_format})")

        # Read file content
        content = await file.read()
        csv_content = content.decode('utf-8')

        # Initialize services
        position_manager = PositionManager()
        csv_service = CSVPositionService(position_manager)
        enrichment_service = PositionEnrichmentService(position_manager)

        # Import positions from CSV
        logger.info("Importing positions from CSV...")
        results = csv_service.import_option_positions(
            csv_content,
            replace_existing=False,  # Don't replace, just add for analysis
            chase_format=chase_format
        )

        if results['success'] == 0:
            raise HTTPException(
                status_code=400,
                detail=f"No positions imported. Errors: {results['errors']}"
            )

        logger.info(f"Imported {results['success']} positions successfully")

        # Enrich positions with real-time data
        logger.info("Enriching positions with real-time market data...")
        enrichment_service.enrich_all_positions()

        # Get portfolio summary
        portfolio_summary = position_manager.get_portfolio_summary()

        # Prepare portfolio data for swarm analysis
        positions = []
        for pos in position_manager.option_positions.values():
            pos_dict = pos.to_dict()
            positions.append({
                'symbol': pos_dict.get('symbol'),
                'asset_type': 'option',
                'option_type': pos_dict.get('option_type'),
                'strike': pos_dict.get('strike'),
                'expiration_date': pos_dict.get('expiration_date'),
                'quantity': pos_dict.get('quantity'),
                'premium_paid': pos_dict.get('cost_basis', 0),
                'current_price': pos_dict.get('current_price', 0),
                'underlying_price': pos_dict.get('underlying_price', 0),
                'delta': pos_dict.get('greeks', {}).get('delta', 0),
                'gamma': pos_dict.get('greeks', {}).get('gamma', 0),
                'theta': pos_dict.get('greeks', {}).get('theta', 0),
                'vega': pos_dict.get('greeks', {}).get('vega', 0),
                'market_value': pos_dict.get('current_value', 0),
                'unrealized_pnl': pos_dict.get('unrealized_pnl', 0),
                'days_to_expiry': pos_dict.get('days_to_expiry', 0),
                'iv': pos_dict.get('iv')
            })

        portfolio_data = {
            'positions': positions,
            'total_value': portfolio_summary.get('total_value', 0),
            'unrealized_pnl': portfolio_summary.get('total_unrealized_pnl', 0),
            'initial_value': portfolio_summary.get('total_value', 0) - portfolio_summary.get('total_unrealized_pnl', 0),
            'peak_value': portfolio_summary.get('total_value', 0)
        }

        # Market data (empty for now, agents will fetch)
        market_data = {}

        logger.info(f"Running swarm analysis on {len(positions)} positions...")

        # Get swarm coordinator
        coordinator = get_swarm_coordinator()

        # Run swarm analysis
        analysis = coordinator.analyze_portfolio(portfolio_data, market_data)

        # Get consensus method
        from src.agents.swarm.consensus_engine import ConsensusMethod
        consensus_method_map = {
            'majority': ConsensusMethod.MAJORITY,
            'weighted': ConsensusMethod.WEIGHTED,
            'unanimous': ConsensusMethod.UNANIMOUS,
            'quorum': ConsensusMethod.QUORUM,
            'entropy': ConsensusMethod.ENTROPY_BASED
        }
        consensus_method_enum = consensus_method_map.get(
            consensus_method.lower(),
            ConsensusMethod.WEIGHTED
        )

        # Get recommendations
        recommendations = coordinator.make_recommendations(
            analysis,
            consensus_method=consensus_method_enum
        )

        # Get metrics
        metrics = coordinator.get_swarm_metrics()

        logger.info("Swarm analysis complete!")

        # Build response
        # Map consensus_recommendations to consensus_decisions for frontend compatibility
        consensus_recs = recommendations.get('consensus_recommendations', {})

        # Helper function to transform consensus result to frontend format
        def transform_consensus_result(result_data: Dict[str, Any], default_choice: str) -> Dict[str, Any]:
            if not result_data:
                return {
                    'choice': default_choice,
                    'confidence': 0.5,
                    'reasoning': 'No consensus reached'
                }

            # Map 'result' to 'choice' for frontend compatibility
            choice = result_data.get('result', result_data.get('choice', default_choice))
            confidence = result_data.get('confidence', 0.5)

            # Generate reasoning from metadata if available
            metadata = result_data.get('metadata', {})
            weighted_votes = metadata.get('weighted_votes', {})

            if weighted_votes:
                top_votes = sorted(weighted_votes.items(), key=lambda x: x[1], reverse=True)[:3]
                vote_summary = ', '.join([f"{vote[0]}: {vote[1]:.1f}" for vote in top_votes])
                reasoning = f"Consensus reached with {confidence*100:.0f}% confidence. Weighted votes: {vote_summary}"
            else:
                reasoning = result_data.get('reasoning', f"Consensus: {choice} with {confidence*100:.0f}% confidence")

            return {
                'choice': choice,
                'confidence': confidence,
                'reasoning': reasoning
            }

        consensus_decisions = {
            'overall_action': transform_consensus_result(
                consensus_recs.get('overall_action', {}),
                'hold'
            ),
            'risk_level': transform_consensus_result(
                consensus_recs.get('risk_level', {}),
                'moderate'
            ),
            'market_outlook': transform_consensus_result(
                consensus_recs.get('market_outlook', {}),
                'neutral'
            )
        }

        # Build enhanced institutional-grade response
        response = build_institutional_response(
            consensus_decisions=consensus_decisions,
            analysis=analysis,
            recommendations=recommendations,
            portfolio_data=portfolio_data,
            positions=positions,
            results=results,
            metrics=metrics,
            coordinator=coordinator
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in CSV swarm analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


