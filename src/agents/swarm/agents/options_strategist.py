"""
Options Strategist Agent

Selects optimal options strategies, constructs spreads, and recommends
specific options trades based on market conditions and portfolio objectives.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from ..base_swarm_agent import BaseSwarmAgent
from ..shared_context import SharedContext
from ..consensus_engine import ConsensusEngine

logger = logging.getLogger(__name__)


class OptionsStrategistAgent(BaseSwarmAgent):
    """
    Options Strategist Agent - Strategy selection and spread construction expert.
    
    Responsibilities:
    - Select optimal options strategies (calls, puts, spreads, straddles, etc.)
    - Construct multi-leg option spreads
    - Recommend strike prices and expirations
    - Optimize strategy selection based on market regime
    - Calculate strategy payoff profiles
    """
    
    def __init__(
        self,
        agent_id: str,
        shared_context: SharedContext,
        consensus_engine: ConsensusEngine
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="OptionsStrategist",
            shared_context=shared_context,
            consensus_engine=consensus_engine,
            priority=7,
            confidence_threshold=0.65
        )
        
        # Strategy mappings
        self.strategies_by_regime = {
            'bull_market': ['long_call', 'bull_call_spread', 'covered_call'],
            'bear_market': ['long_put', 'bear_put_spread', 'protective_put'],
            'volatile_market': ['long_straddle', 'long_strangle', 'iron_condor'],
            'neutral_market': ['iron_condor', 'butterfly_spread', 'calendar_spread']
        }
    
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and select optimal options strategies"""
        try:
            logger.info(f"{self.agent_id}: Starting strategy analysis")
            
            # Get market regime from shared state
            market_regime = self.get_state('market_regime', 'neutral_market')
            market_trend = self.get_state('market_trend', 'neutral')
            
            # Select appropriate strategies
            recommended_strategies = self._select_strategies(market_regime, market_trend)
            
            # Analyze current positions
            portfolio_data = context.get('portfolio', {})
            position_analysis = self._analyze_current_positions(portfolio_data)
            
            analysis = {
                'market_regime': market_regime,
                'recommended_strategies': recommended_strategies,
                'position_analysis': position_analysis,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.send_message(
                content={'type': 'strategy_analysis', 'strategies': recommended_strategies},
                priority=7
            )
            
            self.record_action('strategy_analysis', {'regime': market_regime})
            
            return analysis
            
        except Exception as e:
            logger.error(f"{self.agent_id}: Error in analysis: {e}")
            self.record_error(e, context)
            return {'error': str(e)}
    
    def make_recommendation(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make strategy recommendations"""
        try:
            strategies = analysis.get('recommended_strategies', [])
            
            if not strategies:
                return {'error': 'No strategies recommended'}
            
            # Select top strategy
            top_strategy = strategies[0]
            
            recommendation = {
                'overall_action': {
                    'choice': 'buy' if 'long' in top_strategy else 'sell',
                    'confidence': 0.7,
                    'reasoning': f"Recommended strategy: {top_strategy}"
                },
                'primary_strategy': top_strategy,
                'alternative_strategies': strategies[1:3],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return recommendation
            
        except Exception as e:
            logger.error(f"{self.agent_id}: Error making recommendation: {e}")
            self.record_error(e, analysis)
            return {'error': str(e)}
    
    def _select_strategies(self, market_regime: str, market_trend: str) -> List[str]:
        """Select strategies based on market conditions"""
        strategies = self.strategies_by_regime.get(market_regime, ['long_call'])
        return strategies
    
    def _analyze_current_positions(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current option positions"""
        positions = portfolio_data.get('positions', [])
        option_positions = [p for p in positions if p.get('asset_type') == 'option']
        
        return {
            'total_options': len(option_positions),
            'calls': len([p for p in option_positions if p.get('option_type') == 'call']),
            'puts': len([p for p in option_positions if p.get('option_type') == 'put'])
        }

