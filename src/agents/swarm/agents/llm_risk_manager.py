"""
LLM-Powered Risk Manager Agent

Uses actual LLM calls for intelligent risk assessment and portfolio risk management.
"""

import logging
import json
from typing import Dict, Any
from datetime import datetime

from ..base_swarm_agent import BaseSwarmAgent
from ..shared_context import SharedContext
from ..consensus_engine import ConsensusEngine
from ..llm_agent_base import LLMAgentBase

logger = logging.getLogger(__name__)


class LLMRiskManagerAgent(BaseSwarmAgent, LLMAgentBase):
    """
    LLM-Powered Risk Manager - Uses AI for intelligent risk assessment.
    
    Analyzes portfolio risk using LLMs to provide nuanced, context-aware
    risk management recommendations.
    """
    
    def __init__(
        self,
        agent_id: str,
        shared_context: SharedContext,
        consensus_engine: ConsensusEngine,
        max_portfolio_delta: float = 100.0,
        max_position_size_pct: float = 0.10,
        max_drawdown_pct: float = 0.15,
        preferred_model: str = "anthropic"
    ):
        # Initialize both parent classes
        BaseSwarmAgent.__init__(
            self,
            agent_id=agent_id,
            agent_type="RiskManager",
            shared_context=shared_context,
            consensus_engine=consensus_engine,
            priority=9,
            confidence_threshold=0.7
        )
        
        LLMAgentBase.__init__(
            self,
            agent_id=agent_id,
            agent_type="RiskManager",
            preferred_model=preferred_model
        )
        
        # Risk parameters
        self.max_portfolio_delta = max_portfolio_delta
        self.max_position_size_pct = max_position_size_pct
        self.max_drawdown_pct = max_drawdown_pct
    
    def get_system_prompt(self) -> str:
        """Get specialized system prompt for risk management"""
        return f"""You are an expert Risk Manager specializing in options portfolio risk assessment.

Your responsibilities:
- Assess portfolio risk levels (conservative/moderate/aggressive)
- Identify concentration risks and position sizing issues
- Evaluate Greeks exposure (delta, gamma, theta, vega)
- Recommend hedging strategies
- Monitor drawdown and loss limits

Risk Parameters:
- Max Portfolio Delta: {self.max_portfolio_delta}
- Max Position Size: {self.max_position_size_pct * 100}%
- Max Drawdown: {self.max_drawdown_pct * 100}%

Provide specific, actionable risk management recommendations.
Format your response as JSON with clear structure."""
    
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze portfolio risk using LLM.
        
        Args:
            context: Current context including portfolio data
        
        Returns:
            LLM-powered risk analysis
        """
        try:
            logger.info(f"{self.agent_id}: Starting LLM-powered risk analysis")
            
            # Extract portfolio data
            portfolio = context.get('portfolio', {})
            
            # Build comprehensive prompt
            prompt = self._build_risk_analysis_prompt(portfolio, context)
            
            # Call LLM for intelligent risk analysis
            llm_response = self.call_llm(
                prompt=prompt,
                system_prompt=self.get_system_prompt(),
                temperature=0.6,  # Lower temperature for more consistent risk assessment
                max_tokens=2000
            )
            
            # Parse LLM response
            analysis = self._parse_llm_response(llm_response, portfolio)
            
            # Send message to swarm
            self.send_message(
                content={
                    'type': 'risk_analysis',
                    'risk_level': analysis.get('risk_level', 'moderate'),
                    'key_risks': analysis.get('key_risks', []),
                    'llm_powered': True
                },
                priority=9,
                confidence=analysis.get('confidence', 0.7)
            )
            
            # Update shared state
            self.update_state('portfolio_risk_level', analysis.get('risk_level'))
            self.update_state('risk_violations', analysis.get('violations', []))
            
            self.record_action('llm_risk_analysis', {
                'risk_level': analysis.get('risk_level'),
                'model': self.preferred_model
            })
            
            logger.info(f"{self.agent_id}: LLM risk analysis complete - Level: {analysis.get('risk_level')}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"{self.agent_id}: Error in LLM risk analysis: {e}")
            self.record_error(e, context)
            return {'error': str(e), 'llm_powered': False}
    
    def make_recommendation(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make risk management recommendations using LLM.
        
        Args:
            analysis: Risk analysis results
        
        Returns:
            LLM-powered risk recommendations
        """
        try:
            # Build prompt for recommendations
            prompt = f"""Based on this risk analysis, provide specific risk management recommendations:

Risk Analysis:
{json.dumps(analysis, indent=2)}

Provide recommendations in JSON format with:
1. risk_level: (conservative/moderate/aggressive)
2. confidence: (0.0 to 1.0)
3. reasoning: (brief explanation)
4. immediate_actions: (list of urgent actions needed)
5. hedging_recommendations: (specific hedging strategies)
6. position_adjustments: (which positions to reduce/close)
7. risk_score: (0-100, where 100 is highest risk)

Be specific and actionable."""
            
            llm_response = self.call_llm(
                prompt=prompt,
                system_prompt=self.get_system_prompt(),
                temperature=0.6,
                max_tokens=1500
            )
            
            # Parse recommendation
            recommendation = self._parse_recommendation(llm_response)
            recommendation['timestamp'] = datetime.utcnow().isoformat()
            recommendation['llm_powered'] = True
            
            logger.info(f"{self.agent_id}: LLM recommendation - Risk Level: {recommendation.get('risk_level', {}).get('choice')}")
            
            return recommendation
            
        except Exception as e:
            logger.error(f"{self.agent_id}: Error making LLM recommendation: {e}")
            self.record_error(e, analysis)
            return {'error': str(e), 'llm_powered': False}
    
    def _build_risk_analysis_prompt(self, portfolio: Dict, context: Dict) -> str:
        """Build comprehensive prompt for risk analysis"""
        
        # Calculate portfolio metrics
        total_value = portfolio.get('total_portfolio_value', 0)
        total_pnl = portfolio.get('total_unrealized_pnl', 0)
        total_pnl_pct = portfolio.get('total_unrealized_pnl_pct', 0)
        positions = portfolio.get('positions', [])
        
        # Position details
        position_details = []
        for pos in positions:
            position_details.append({
                'symbol': pos.get('symbol'),
                'type': pos.get('option_type'),
                'strike': pos.get('strike'),
                'expiry': pos.get('expiration'),
                'quantity': pos.get('quantity'),
                'current_value': pos.get('current_value'),
                'unrealized_pnl': pos.get('unrealized_pnl'),
                'unrealized_pnl_pct': pos.get('unrealized_pnl_pct'),
                'days_to_expiry': pos.get('days_to_expiry')
            })
        
        prompt = f"""Analyze the risk profile of this options portfolio:

PORTFOLIO SUMMARY:
- Total Value: ${total_value:,.2f}
- Unrealized P&L: ${total_pnl:,.2f} ({total_pnl_pct:.2f}%)
- Number of Positions: {len(positions)}

POSITIONS:
{json.dumps(position_details, indent=2)}

RISK PARAMETERS:
- Max Portfolio Delta: {self.max_portfolio_delta}
- Max Position Size: {self.max_position_size_pct * 100}%
- Max Drawdown: {self.max_drawdown_pct * 100}%

Provide a comprehensive risk analysis in JSON format with:
1. risk_level: (conservative/moderate/aggressive)
2. confidence: (0.0 to 1.0)
3. key_risks: (list of top 3-5 risks identified)
4. violations: (list of any parameter violations)
5. concentration_risks: (positions that are too large)
6. time_decay_risks: (positions with critical expiry dates)
7. delta_exposure: (assessment of directional risk)
8. risk_score: (0-100, where 100 is highest risk)
9. recommendations: (list of immediate actions)

Be specific and data-driven."""
        
        return prompt
    
    def _parse_llm_response(self, response: str, portfolio: Dict) -> Dict[str, Any]:
        """Parse LLM response into structured risk analysis"""
        try:
            # Try to parse as JSON
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                parsed = json.loads(json_str)
                
                # Add metadata
                parsed['portfolio_value'] = portfolio.get('total_portfolio_value', 0)
                parsed['timestamp'] = datetime.utcnow().isoformat()
                
                return parsed
        except:
            pass
        
        # Fallback: extract key information
        return {
            'risk_level': 'moderate',
            'confidence': 0.6,
            'key_risks': [response[:200]],
            'violations': [],
            'risk_score': 50,
            'timestamp': datetime.utcnow().isoformat(),
            'raw_response': response
        }
    
    def _parse_recommendation(self, response: str) -> Dict[str, Any]:
        """Parse LLM recommendation response"""
        try:
            # Try to parse as JSON
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback
        return {
            'risk_level': {'choice': 'moderate', 'confidence': 0.6, 'reasoning': response[:200]},
            'immediate_actions': [],
            'hedging_recommendations': [],
            'risk_score': 50
        }

