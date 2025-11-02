"""Analytics module for options analysis and AI trading intelligence."""
from .ev_calculator import EVCalculator, EVResult
from .greeks_calculator import GreeksCalculator, GreeksResult, calculate_implied_volatility
from .black_scholes import (
    black_scholes_price,
    black_scholes_delta,
    calculate_probability_itm,
    calculate_breakeven
)
from .swarm_analysis_service import (
    SwarmAnalysisService,
    BacktestResult,
    SwarmConsensus,
    AgentAnalysis
)
from .risk_guardrails import (
    RiskGuardrailsService,
    RiskLevel,
    RiskLimits,
    RiskCheckResult,
    RiskViolation,
    PortfolioState,
    Position
)
from .expert_critique import (
    ExpertCritiqueService,
    ExpertCritiqueReport,
    Recommendation
)

__all__ = [
    # Original analytics
    'EVCalculator',
    'EVResult',
    'GreeksCalculator',
    'GreeksResult',
    'calculate_implied_volatility',
    'black_scholes_price',
    'black_scholes_delta',
    'calculate_probability_itm',
    'calculate_breakeven',
    # AI Trading Intelligence
    'SwarmAnalysisService',
    'BacktestResult',
    'SwarmConsensus',
    'AgentAnalysis',
    'RiskGuardrailsService',
    'RiskLevel',
    'RiskLimits',
    'RiskCheckResult',
    'RiskViolation',
    'PortfolioState',
    'Position',
    'ExpertCritiqueService',
    'ExpertCritiqueReport',
    'Recommendation',
]

