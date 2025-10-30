"""
Multi-Agent Swarm System for Options Portfolio Analysis

This module implements a sophisticated swarm intelligence system where multiple
specialized AI agents collaborate to analyze portfolios, identify opportunities,
and make trading recommendations while maximizing profit and minimizing risk.

Key Features:
- Decentralized decision-making with consensus mechanisms
- Entropy-based confidence levels for agent certainty
- Stigmergic communication through shared memory
- Emergent behavior through local agent interactions
- Self-organizing team composition
- Multi-layer risk validation
"""

from .swarm_coordinator import SwarmCoordinator
from .shared_context import SharedContext
from .consensus_engine import ConsensusEngine
from .agents import (
    MarketAnalystAgent,
    RiskManagerAgent,
    OptionsStrategistAgent,
    TechnicalAnalystAgent,
    SentimentAnalystAgent,
    PortfolioOptimizerAgent,
    TradeExecutorAgent,
    ComplianceOfficerAgent
)

__all__ = [
    'SwarmCoordinator',
    'SharedContext',
    'ConsensusEngine',
    'MarketAnalystAgent',
    'RiskManagerAgent',
    'OptionsStrategistAgent',
    'TechnicalAnalystAgent',
    'SentimentAnalystAgent',
    'PortfolioOptimizerAgent',
    'TradeExecutorAgent',
    'ComplianceOfficerAgent'
]

__version__ = '1.0.0'

