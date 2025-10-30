"""
LLM-Powered Volatility Specialist Agent

Expert in implied volatility, volatility surface analysis, VIX dynamics,
and volatility-based trading strategies for options.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from ..base_swarm_agent import BaseSwarmAgent
from ..llm_agent_base import LLMAgentBase
from ..shared_context import SharedContext
from ..consensus_engine import ConsensusEngine

logger = logging.getLogger(__name__)


class LLMVolatilitySpecialistAgent(BaseSwarmAgent, LLMAgentBase):
    """
    Volatility Specialist - Expert in IV, vol surface, and VIX analysis.
    
    Analyzes:
    - Implied volatility levels and skew
    - Volatility surface dynamics
    - VIX and volatility regime changes
    - Vol arbitrage opportunities
    - Optimal strike selection based on IV
    """
    
    def __init__(
        self,
        agent_id: str,
        shared_context: SharedContext,
        consensus_engine: ConsensusEngine,
        preferred_model: str = "lmstudio"
    ):
        BaseSwarmAgent.__init__(
            self,
            agent_id=agent_id,
            agent_type="VolatilitySpecialist",
            shared_context=shared_context,
            consensus_engine=consensus_engine,
            priority=8,
            confidence_threshold=0.7
        )
        
        LLMAgentBase.__init__(
            self,
            agent_id=agent_id,
            agent_type="VolatilitySpecialist",
            preferred_model=preferred_model
        )
        
        logger.info(f"{agent_id}: Volatility Specialist initialized with {preferred_model}")
    
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volatility environment"""
        try:
            logger.info(f"{self.agent_id}: Starting volatility analysis")
            
            portfolio_data = context.get('portfolio', {})
            positions = portfolio_data.get('positions', [])
            
            # Extract IV data from positions
            iv_data = self._extract_iv_data(positions)
            
            # Build volatility analysis prompt
            prompt = self._build_volatility_prompt(iv_data, positions)
            
            system_prompt = """You are a volatility trading expert specializing in:
- Implied volatility analysis and forecasting
- Volatility surface modeling and skew analysis
- VIX dynamics and volatility regime identification
- Volatility arbitrage and dispersion trading
- Optimal options strike selection based on IV

Provide actionable volatility insights for options trading."""
            
            # Call LLM
            llm_response = self.call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.6,
                max_tokens=2000
            )
            
            # Parse response
            analysis = self._parse_volatility_response(llm_response, iv_data)
            
            # Share volatility insights
            self.send_message({
                'type': 'volatility_analysis',
                'vol_regime': analysis.get('vol_regime', 'normal'),
                'iv_level': analysis.get('iv_level', 'normal'),
                'vix_outlook': analysis.get('vix_outlook', 'stable'),
                'timestamp': datetime.utcnow().isoformat()
            }, priority=8)
            
            # Update shared state
            self.update_state('volatility_regime', analysis.get('vol_regime', 'normal'))
            self.update_state('iv_environment', analysis.get('iv_level', 'normal'))
            
            logger.info(f"{self.agent_id}: Volatility analysis complete")
            
            return analysis
            
        except Exception as e:
            logger.error(f"{self.agent_id}: Error in volatility analysis: {e}")
            self.record_error(e, context)
            return {'error': str(e)}
    
    def make_recommendation(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make volatility-based recommendations"""
        try:
            vol_regime = analysis.get('vol_regime', 'normal')
            iv_level = analysis.get('iv_level', 'normal')
            
            # Determine strategy based on volatility
            if vol_regime == 'high' and iv_level == 'elevated':
                action = 'sell'  # Sell premium in high IV
                confidence = 0.75
                reasoning = "High volatility regime - favor premium selling strategies"
            elif vol_regime == 'low' and iv_level == 'depressed':
                action = 'buy'  # Buy options in low IV
                confidence = 0.70
                reasoning = "Low volatility regime - favor long volatility strategies"
            else:
                action = 'hedge'
                confidence = 0.65
                reasoning = "Normal volatility - maintain balanced exposure"
            
            recommendation = {
                'overall_action': {
                    'choice': action,
                    'confidence': confidence,
                    'reasoning': reasoning
                },
                'vol_regime': vol_regime,
                'iv_level': iv_level,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return recommendation
            
        except Exception as e:
            logger.error(f"{self.agent_id}: Error making recommendation: {e}")
            return {'error': str(e)}
    
    def _extract_iv_data(self, positions: list) -> Dict[str, Any]:
        """Extract IV data from positions"""
        iv_values = []
        
        for pos in positions:
            if pos.get('asset_type') == 'option':
                iv = pos.get('implied_volatility', 0)
                if iv > 0:
                    iv_values.append(iv)
        
        if iv_values:
            avg_iv = sum(iv_values) / len(iv_values)
        else:
            avg_iv = 0.3  # Default
        
        return {
            'average_iv': avg_iv,
            'iv_count': len(iv_values),
            'iv_range': (min(iv_values) if iv_values else 0, max(iv_values) if iv_values else 0)
        }
    
    def _build_volatility_prompt(self, iv_data: dict, positions: list) -> str:
        """Build volatility analysis prompt"""
        
        avg_iv = iv_data.get('average_iv', 0.3)
        
        prompt = f"""Analyze the current volatility environment for options trading.

Portfolio Volatility Metrics:
- Average Implied Volatility: {avg_iv:.1%}
- Number of option positions: {iv_data.get('iv_count', 0)}
- IV Range: {iv_data.get('iv_range', (0, 0))}

Provide:
1. **Volatility Regime**: low/normal/high/extreme
2. **IV Level**: depressed/normal/elevated/extreme
3. **VIX Outlook**: declining/stable/rising
4. **Trading Strategy**: Which vol strategies to favor (long vol, short vol, neutral)
5. **Risk Assessment**: Key volatility risks

Format:
VOL_REGIME: [regime]
IV_LEVEL: [level]
VIX_OUTLOOK: [outlook]
STRATEGY: [strategy recommendation]
RISKS:
- [Risk 1]
- [Risk 2]
"""
        
        return prompt
    
    def _parse_volatility_response(self, response: str, iv_data: dict) -> Dict[str, Any]:
        """Parse volatility analysis response"""
        
        vol_regime = 'normal'
        iv_level = 'normal'
        vix_outlook = 'stable'
        strategy = 'neutral'
        risks = []
        
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            
            if line_lower.startswith('vol_regime:'):
                if 'high' in line_lower or 'extreme' in line_lower:
                    vol_regime = 'high'
                elif 'low' in line_lower:
                    vol_regime = 'low'
            
            elif line_lower.startswith('iv_level:'):
                if 'elevated' in line_lower or 'extreme' in line_lower:
                    iv_level = 'elevated'
                elif 'depressed' in line_lower:
                    iv_level = 'depressed'
            
            elif line_lower.startswith('vix_outlook:'):
                if 'rising' in line_lower:
                    vix_outlook = 'rising'
                elif 'declining' in line_lower:
                    vix_outlook = 'declining'
            
            elif line_lower.startswith('strategy:'):
                if 'long vol' in line_lower:
                    strategy = 'long_volatility'
                elif 'short vol' in line_lower:
                    strategy = 'short_volatility'
            
            elif line.strip().startswith('-') and len(line.strip()) > 5:
                risks.append(line.strip()[1:].strip())
        
        return {
            'vol_regime': vol_regime,
            'iv_level': iv_level,
            'vix_outlook': vix_outlook,
            'strategy': strategy,
            'risks': risks[:3],
            'iv_data': iv_data,
            'llm_response': response
        }

