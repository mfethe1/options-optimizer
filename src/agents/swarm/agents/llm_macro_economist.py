"""
LLM-Powered Macro Economist Agent

Analyzes macroeconomic trends, Fed policy, inflation, GDP, and global markets
to provide top-down market outlook and sector rotation recommendations.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ..base_swarm_agent import BaseSwarmAgent
from ..llm_agent_base import LLMAgentBase
from ..shared_context import SharedContext
from ..consensus_engine import ConsensusEngine

logger = logging.getLogger(__name__)


class LLMMacroEconomistAgent(BaseSwarmAgent, LLMAgentBase):
    """
    LLM-Powered Macro Economist - Top-down economic analysis.
    
    Analyzes:
    - Federal Reserve policy and interest rates
    - Inflation trends (CPI, PCE)
    - GDP growth and recession indicators
    - Employment data and consumer spending
    - Global macro trends and currency movements
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
            agent_type="MacroEconomist",
            shared_context=shared_context,
            consensus_engine=consensus_engine,
            priority=9,
            confidence_threshold=0.65
        )
        
        LLMAgentBase.__init__(
            self,
            agent_id=agent_id,
            agent_type="MacroEconomist",
            preferred_model=preferred_model
        )
        
        logger.info(f"{agent_id}: LLM Macro Economist initialized with {preferred_model}")
    
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze macroeconomic environment using LLM"""
        try:
            logger.info(f"{self.agent_id}: Starting macro analysis")
            
            market_data = context.get('market', {})
            
            # Build macro analysis prompt
            prompt = self._build_macro_prompt(market_data)
            
            system_prompt = """You are a senior macroeconomist with expertise in:
- Federal Reserve monetary policy and interest rate cycles
- Inflation dynamics and central bank responses
- Business cycle analysis and recession forecasting
- Global macro trends and currency markets
- Sector rotation strategies based on economic cycles

Provide actionable macro insights for portfolio positioning."""
            
            # Call LLM
            llm_response = self.call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.5,
                max_tokens=2000
            )
            
            # Parse response
            analysis = self._parse_macro_response(llm_response)
            
            # Share macro insights with swarm
            self.send_message({
                'type': 'macro_analysis',
                'economic_cycle': analysis.get('cycle_phase', 'mid-cycle'),
                'fed_stance': analysis.get('fed_stance', 'neutral'),
                'recession_risk': analysis.get('recession_risk', 0.3),
                'key_risks': analysis.get('key_risks', []),
                'timestamp': datetime.utcnow().isoformat()
            }, priority=9)
            
            # Update shared state
            self.update_state('economic_cycle', analysis.get('cycle_phase', 'mid-cycle'))
            self.update_state('macro_outlook', analysis.get('outlook', 'neutral'))
            
            logger.info(f"{self.agent_id}: Macro analysis complete")
            
            return analysis
            
        except Exception as e:
            logger.error(f"{self.agent_id}: Error in macro analysis: {e}")
            self.record_error(e, context)
            return {'error': str(e)}
    
    def make_recommendation(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make macro-based recommendations"""
        try:
            outlook = analysis.get('outlook', 'neutral')
            cycle_phase = analysis.get('cycle_phase', 'mid-cycle')
            recession_risk = analysis.get('recession_risk', 0.3)
            
            # Determine market outlook based on macro
            if outlook == 'bullish' and recession_risk < 0.3:
                market_outlook = 'bullish'
                confidence = 0.75
            elif outlook == 'bearish' or recession_risk > 0.6:
                market_outlook = 'bearish'
                confidence = 0.70
            else:
                market_outlook = 'neutral'
                confidence = 0.60
            
            recommendation = {
                'market_outlook': {
                    'choice': market_outlook,
                    'confidence': confidence,
                    'reasoning': f"Economic cycle: {cycle_phase}, Recession risk: {recession_risk:.1%}, Outlook: {outlook}"
                },
                'cycle_phase': cycle_phase,
                'recession_risk': recession_risk,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return recommendation
            
        except Exception as e:
            logger.error(f"{self.agent_id}: Error making recommendation: {e}")
            return {'error': str(e)}
    
    def _build_macro_prompt(self, market_data: dict) -> str:
        """Build LLM prompt for macro analysis"""
        
        prompt = """Analyze the current macroeconomic environment and provide portfolio positioning guidance.

Consider:
1. **Fed Policy**: Current interest rate stance, future rate path, QT/QE
2. **Inflation**: CPI trends, core inflation, wage growth
3. **Growth**: GDP trajectory, employment data, consumer spending
4. **Cycle Phase**: Early/mid/late cycle, recession probability
5. **Global Factors**: International growth, currency trends, geopolitical risks

Provide:
- Economic Cycle Phase: early-cycle/mid-cycle/late-cycle/recession
- Fed Stance: dovish/neutral/hawkish
- Outlook: bullish/neutral/bearish
- Recession Risk: 0.0-1.0 probability
- Key Risks: Top 3 macro risks
- Sector Recommendations: Which sectors to favor/avoid

Format:
CYCLE_PHASE: [phase]
FED_STANCE: [stance]
OUTLOOK: [outlook]
RECESSION_RISK: [0.0-1.0]
KEY_RISKS:
- [Risk 1]
- [Risk 2]
- [Risk 3]
"""
        
        return prompt
    
    def _parse_macro_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM macro response"""
        
        cycle_phase = 'mid-cycle'
        fed_stance = 'neutral'
        outlook = 'neutral'
        recession_risk = 0.3
        key_risks = []
        
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            
            if line_lower.startswith('cycle_phase:'):
                if 'early' in line_lower:
                    cycle_phase = 'early-cycle'
                elif 'late' in line_lower:
                    cycle_phase = 'late-cycle'
                elif 'recession' in line_lower:
                    cycle_phase = 'recession'
            
            elif line_lower.startswith('fed_stance:'):
                if 'dovish' in line_lower:
                    fed_stance = 'dovish'
                elif 'hawkish' in line_lower:
                    fed_stance = 'hawkish'
            
            elif line_lower.startswith('outlook:'):
                if 'bullish' in line_lower:
                    outlook = 'bullish'
                elif 'bearish' in line_lower:
                    outlook = 'bearish'
            
            elif line_lower.startswith('recession_risk:'):
                try:
                    risk_str = line.split(':')[1].strip()
                    recession_risk = float(risk_str)
                except:
                    pass
            
            elif line.strip().startswith('-') and len(line.strip()) > 5:
                key_risks.append(line.strip()[1:].strip())
        
        return {
            'cycle_phase': cycle_phase,
            'fed_stance': fed_stance,
            'outlook': outlook,
            'recession_risk': recession_risk,
            'key_risks': key_risks[:3],
            'llm_response': response
        }

