"""
LLM-Powered Fundamental Analyst Agent

Analyzes company fundamentals, earnings, financial health, and intrinsic value
using AI-powered analysis of financial statements and market data.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ..base_swarm_agent import BaseSwarmAgent
from ..llm_agent_base import LLMAgentBase
from ..shared_context import SharedContext
from ..consensus_engine import ConsensusEngine

logger = logging.getLogger(__name__)


class LLMFundamentalAnalystAgent(BaseSwarmAgent, LLMAgentBase):
    """
    LLM-Powered Fundamental Analyst - Deep dive into company financials.
    
    Analyzes:
    - Financial statements (balance sheet, income, cash flow)
    - Earnings quality and growth
    - Valuation metrics (P/E, P/B, DCF)
    - Competitive positioning
    - Management quality
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
            agent_type="FundamentalAnalyst",
            shared_context=shared_context,
            consensus_engine=consensus_engine,
            priority=8,
            confidence_threshold=0.7
        )
        
        LLMAgentBase.__init__(
            self,
            agent_id=agent_id,
            agent_type="FundamentalAnalyst",
            preferred_model=preferred_model
        )
        
        logger.info(f"{agent_id}: LLM Fundamental Analyst initialized with {preferred_model}")
    
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze company fundamentals using LLM"""
        try:
            logger.info(f"{self.agent_id}: Starting fundamental analysis")
            
            portfolio_data = context.get('portfolio', {})
            market_data = context.get('market', {})
            
            # Extract unique symbols
            positions = portfolio_data.get('positions', [])
            symbols = list(set([p.get('symbol', '') for p in positions if p.get('symbol')]))
            
            if not symbols:
                return {'error': 'No symbols to analyze'}
            
            # Build analysis prompt
            prompt = self._build_analysis_prompt(symbols, positions, market_data)
            
            system_prompt = """You are an expert fundamental analyst with 20+ years of experience 
analyzing company financials, earnings quality, and intrinsic value. You specialize in:
- Financial statement analysis (balance sheet, income statement, cash flow)
- Earnings quality assessment
- Valuation modeling (DCF, multiples, comparable companies)
- Competitive moat analysis
- Management quality evaluation

Provide detailed, actionable insights based on fundamental data."""
            
            # Call LLM
            llm_response = self.call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.6,
                max_tokens=2500
            )
            
            # Parse response
            analysis = self._parse_llm_response(llm_response, symbols)
            
            # Share insights with swarm
            self.send_message({
                'type': 'fundamental_analysis',
                'symbols': symbols,
                'key_insights': analysis.get('key_insights', []),
                'valuation_summary': analysis.get('valuation', {}),
                'timestamp': datetime.utcnow().isoformat()
            }, priority=8)
            
            # Update shared state
            self.update_state('fundamental_outlook', analysis.get('outlook', 'neutral'))
            self.update_state('valuation_level', analysis.get('valuation_level', 'fair'))
            
            logger.info(f"{self.agent_id}: Fundamental analysis complete")
            
            return analysis
            
        except Exception as e:
            logger.error(f"{self.agent_id}: Error in fundamental analysis: {e}")
            self.record_error(e, context)
            return {'error': str(e)}
    
    def make_recommendation(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make fundamental-based recommendations"""
        try:
            outlook = analysis.get('outlook', 'neutral')
            valuation = analysis.get('valuation_level', 'fair')
            quality_score = analysis.get('quality_score', 0.5)
            
            # Determine action based on fundamentals
            if outlook == 'bullish' and valuation in ['undervalued', 'fair']:
                action = 'buy'
                confidence = 0.75 + (quality_score * 0.15)
            elif outlook == 'bearish' or valuation == 'overvalued':
                action = 'sell'
                confidence = 0.70 + (quality_score * 0.10)
            else:
                action = 'hold'
                confidence = 0.60
            
            recommendation = {
                'overall_action': {
                    'choice': action,
                    'confidence': min(confidence, 0.95),
                    'reasoning': f"Fundamental outlook: {outlook}, Valuation: {valuation}, Quality: {quality_score:.2f}"
                },
                'valuation_assessment': valuation,
                'quality_score': quality_score,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return recommendation
            
        except Exception as e:
            logger.error(f"{self.agent_id}: Error making recommendation: {e}")
            return {'error': str(e)}
    
    def _build_analysis_prompt(
        self,
        symbols: list,
        positions: list,
        market_data: dict
    ) -> str:
        """Build LLM prompt for fundamental analysis"""
        
        prompt = f"""Analyze the fundamental health and valuation of these companies: {', '.join(symbols[:5])}

Portfolio Context:
- Total positions: {len(positions)}
- Symbols: {', '.join(symbols[:10])}

Please provide:
1. **Financial Health**: Assess balance sheet strength, debt levels, cash flow quality
2. **Earnings Quality**: Evaluate earnings consistency, growth trajectory, and sustainability
3. **Valuation**: Determine if stocks are undervalued, fairly valued, or overvalued
4. **Competitive Position**: Analyze market share, competitive moats, industry trends
5. **Risk Factors**: Identify key fundamental risks (debt, competition, regulation)

For each company, provide:
- Outlook: bullish/neutral/bearish
- Valuation Level: undervalued/fair/overvalued
- Quality Score: 0-1 (financial health and earnings quality)
- Key Insights: 2-3 critical fundamental factors

Format your response as:
OUTLOOK: [bullish/neutral/bearish]
VALUATION: [undervalued/fair/overvalued]
QUALITY_SCORE: [0.0-1.0]
KEY_INSIGHTS:
- [Insight 1]
- [Insight 2]
- [Insight 3]
"""
        
        return prompt
    
    def _parse_llm_response(self, response: str, symbols: list) -> Dict[str, Any]:
        """Parse LLM response into structured data"""
        
        # Extract key fields
        outlook = 'neutral'
        valuation_level = 'fair'
        quality_score = 0.5
        key_insights = []
        
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            
            if line_lower.startswith('outlook:'):
                if 'bullish' in line_lower:
                    outlook = 'bullish'
                elif 'bearish' in line_lower:
                    outlook = 'bearish'
            
            elif line_lower.startswith('valuation:'):
                if 'undervalued' in line_lower:
                    valuation_level = 'undervalued'
                elif 'overvalued' in line_lower:
                    valuation_level = 'overvalued'
            
            elif line_lower.startswith('quality_score:'):
                try:
                    score_str = line.split(':')[1].strip()
                    quality_score = float(score_str)
                except:
                    pass
            
            elif line.strip().startswith('-') and len(line.strip()) > 5:
                key_insights.append(line.strip()[1:].strip())
        
        return {
            'outlook': outlook,
            'valuation_level': valuation_level,
            'quality_score': quality_score,
            'key_insights': key_insights[:5],
            'symbols_analyzed': symbols,
            'llm_response': response
        }

