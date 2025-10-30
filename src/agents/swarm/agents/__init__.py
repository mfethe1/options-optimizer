"""
Specialized Swarm Agents

This module contains all specialized agents for the options portfolio analysis swarm.
Each agent has a specific role and expertise area.
"""

from .market_analyst import MarketAnalystAgent
from .risk_manager import RiskManagerAgent
from .options_strategist import OptionsStrategistAgent
from .technical_analyst import TechnicalAnalystAgent
from .sentiment_analyst import SentimentAnalystAgent
from .portfolio_optimizer import PortfolioOptimizerAgent
from .trade_executor import TradeExecutorAgent
from .compliance_officer import ComplianceOfficerAgent

# LLM-Powered Agents
from .llm_market_analyst import LLMMarketAnalystAgent
from .llm_risk_manager import LLMRiskManagerAgent
from .llm_sentiment_analyst import LLMSentimentAnalystAgent
from .llm_fundamental_analyst import LLMFundamentalAnalystAgent
from .llm_macro_economist import LLMMacroEconomistAgent
from .llm_volatility_specialist import LLMVolatilitySpecialistAgent
from .swarm_overseer import SwarmOverseerAgent

__all__ = [
    # Rule-based agents
    'MarketAnalystAgent',
    'RiskManagerAgent',
    'OptionsStrategistAgent',
    'TechnicalAnalystAgent',
    'SentimentAnalystAgent',
    'PortfolioOptimizerAgent',
    'TradeExecutorAgent',
    'ComplianceOfficerAgent',
    # LLM-powered agents
    'LLMMarketAnalystAgent',
    'LLMRiskManagerAgent',
    'LLMSentimentAnalystAgent',
    'LLMFundamentalAnalystAgent',
    'LLMMacroEconomistAgent',
    'LLMVolatilitySpecialistAgent',
    'SwarmOverseerAgent'
]

