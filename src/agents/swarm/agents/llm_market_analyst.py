"""
LLM-Powered Market Analyst Agent

Uses actual LLM calls (OpenAI/Anthropic/LMStudio) for intelligent market analysis
instead of hardcoded logic.
"""

import logging
import json
from typing import Dict, Any
from datetime import datetime
import yfinance as yf

from ..base_swarm_agent import BaseSwarmAgent
from ..shared_context import SharedContext
from ..consensus_engine import ConsensusEngine
from ..llm_agent_base import LLMAgentBase, FirecrawlMixin

logger = logging.getLogger(__name__)


class LLMMarketAnalystAgent(BaseSwarmAgent, LLMAgentBase, FirecrawlMixin):
    """
    LLM-Powered Market Analyst - Uses AI for intelligent market analysis.
    
    Combines:
    - Real market data from yfinance
    - Web research from Firecrawl
    - AI-powered analysis from LLMs
    """
    
    def __init__(
        self,
        agent_id: str,
        shared_context: SharedContext,
        consensus_engine: ConsensusEngine,
        preferred_model: str = "openai"
    ):
        # Initialize both parent classes
        BaseSwarmAgent.__init__(
            self,
            agent_id=agent_id,
            agent_type="MarketAnalyst",
            shared_context=shared_context,
            consensus_engine=consensus_engine,
            priority=8,
            confidence_threshold=0.6
        )
        
        LLMAgentBase.__init__(
            self,
            agent_id=agent_id,
            agent_type="MarketAnalyst",
            preferred_model=preferred_model
        )
        
        # Market indices to track
        self.indices = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ',
            'DIA': 'Dow Jones',
            'IWM': 'Russell 2000'
        }
        
        # Sector ETFs
        self.sectors = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLE': 'Energy',
            'XLV': 'Healthcare',
            'XLI': 'Industrials',
            'XLP': 'Consumer Staples',
            'XLY': 'Consumer Discretionary',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate'
        }
    
    def get_system_prompt(self) -> str:
        """Get specialized system prompt for market analysis"""
        return """You are an expert Market Analyst specializing in options and stock trading.

Your responsibilities:
- Analyze overall market trends (bull/bear/neutral)
- Identify sector rotation patterns
- Assess market breadth and momentum
- Evaluate macroeconomic factors
- Provide actionable market insights

Provide concise, data-driven analysis with specific recommendations.
Format your response as JSON with clear structure."""
    
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market conditions using LLM + real data.
        
        Args:
            context: Current context including portfolio and market data
        
        Returns:
            LLM-powered market analysis
        """
        try:
            logger.info(f"{self.agent_id}: Starting LLM-powered market analysis")
            
            # 1. Fetch real market data
            market_data = self._fetch_market_data()
            sector_data = self._fetch_sector_data()
            
            # 2. Get news via Firecrawl (placeholder for now)
            market_news = self.get_news("SPY market", days=3)
            
            # 3. Build comprehensive prompt for LLM
            prompt = self._build_analysis_prompt(market_data, sector_data, market_news, context)
            
            # 4. Call LLM for intelligent analysis
            llm_response = self.call_llm(
                prompt=prompt,
                system_prompt=self.get_system_prompt(),
                temperature=0.7,
                max_tokens=2000
            )
            
            # 5. Parse LLM response
            analysis = self._parse_llm_response(llm_response, market_data, sector_data)
            
            # 6. Send message to swarm
            self.send_message(
                content={
                    'type': 'market_analysis',
                    'market_regime': analysis.get('market_regime', 'neutral'),
                    'key_insights': analysis.get('key_insights', []),
                    'llm_powered': True
                },
                priority=8,
                confidence=analysis.get('confidence', 0.7)
            )
            
            # 7. Update shared state
            self.update_state('market_regime', analysis.get('market_regime'))
            self.update_state('market_trend', analysis.get('trend'))
            
            self.record_action('llm_market_analysis', {
                'regime': analysis.get('market_regime'),
                'model': self.preferred_model
            })
            
            logger.info(f"{self.agent_id}: LLM analysis complete - Regime: {analysis.get('market_regime')}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"{self.agent_id}: Error in LLM analysis: {e}")
            self.record_error(e, context)
            return {'error': str(e), 'llm_powered': False}
    
    def make_recommendation(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make recommendations using LLM analysis.
        
        Args:
            analysis: Market analysis results
        
        Returns:
            LLM-powered recommendations
        """
        try:
            # Build prompt for recommendations
            prompt = f"""Based on this market analysis, provide specific trading recommendations:

Analysis:
{json.dumps(analysis, indent=2)}

Provide recommendations in JSON format with:
1. overall_action: (buy/sell/hold/hedge)
2. confidence: (0.0 to 1.0)
3. reasoning: (brief explanation)
4. risk_level: (conservative/moderate/aggressive)
5. sector_recommendations: (list of top 3 sectors)
6. market_outlook: (bullish/bearish/neutral)

Be specific and actionable."""
            
            llm_response = self.call_llm(
                prompt=prompt,
                system_prompt=self.get_system_prompt(),
                temperature=0.6,
                max_tokens=1000
            )
            
            # Parse recommendation
            recommendation = self._parse_recommendation(llm_response)
            recommendation['timestamp'] = datetime.utcnow().isoformat()
            recommendation['llm_powered'] = True
            
            logger.info(f"{self.agent_id}: LLM recommendation - Action: {recommendation.get('overall_action', {}).get('choice')}")
            
            return recommendation
            
        except Exception as e:
            logger.error(f"{self.agent_id}: Error making LLM recommendation: {e}")
            self.record_error(e, analysis)
            return {'error': str(e), 'llm_powered': False}
    
    def _fetch_market_data(self) -> Dict[str, Any]:
        """Fetch current market data for indices"""
        data = {}
        
        for symbol, name in self.indices.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1mo')
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[0]
                    change_pct = ((current_price - prev_price) / prev_price) * 100
                    
                    data[symbol] = {
                        'name': name,
                        'current_price': float(current_price),
                        'change_pct_1m': float(change_pct),
                        'volume': int(hist['Volume'].iloc[-1]),
                        'avg_volume': float(hist['Volume'].mean())
                    }
            except Exception as e:
                logger.warning(f"Error fetching {symbol}: {e}")
        
        return data
    
    def _fetch_sector_data(self) -> Dict[str, Any]:
        """Fetch sector performance data"""
        data = {}
        
        for symbol, name in self.sectors.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1mo')
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[0]
                    change_pct = ((current_price - prev_price) / prev_price) * 100
                    
                    data[name] = {
                        'symbol': symbol,
                        'change_pct_1m': float(change_pct)
                    }
            except Exception as e:
                logger.warning(f"Error fetching {symbol}: {e}")
        
        return data
    
    def _build_analysis_prompt(
        self,
        market_data: Dict,
        sector_data: Dict,
        news: Dict,
        context: Dict
    ) -> str:
        """Build comprehensive prompt for LLM"""
        
        portfolio_summary = ""
        if 'portfolio' in context:
            portfolio = context['portfolio']
            portfolio_summary = f"""
Portfolio Context:
- Total Value: ${portfolio.get('total_portfolio_value', 0):,.2f}
- Positions: {len(portfolio.get('positions', []))}
- Unrealized P&L: ${portfolio.get('total_unrealized_pnl', 0):,.2f} ({portfolio.get('total_unrealized_pnl_pct', 0):.2f}%)
"""
        
        prompt = f"""Analyze the current market conditions and provide insights:

MARKET DATA (1-Month Performance):
{json.dumps(market_data, indent=2)}

SECTOR PERFORMANCE (1-Month):
{json.dumps(sector_data, indent=2)}

{portfolio_summary}

Provide a comprehensive market analysis in JSON format with:
1. market_regime: (bull_market/bear_market/neutral_market/volatile_market)
2. trend: (bullish/bearish/neutral)
3. confidence: (0.0 to 1.0)
4. key_insights: (list of 3-5 key observations)
5. top_sectors: (list of top 3 performing sectors)
6. bottom_sectors: (list of bottom 3 sectors)
7. volatility_assessment: (low/moderate/high)
8. risk_factors: (list of current market risks)

Be specific and data-driven."""
        
        return prompt
    
    def _parse_llm_response(self, response: str, market_data: Dict, sector_data: Dict) -> Dict[str, Any]:
        """Parse LLM response into structured analysis"""
        try:
            # Try to parse as JSON
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                parsed = json.loads(json_str)
                
                # Add raw data
                parsed['market_data'] = market_data
                parsed['sector_data'] = sector_data
                parsed['timestamp'] = datetime.utcnow().isoformat()
                
                return parsed
        except:
            pass
        
        # Fallback: extract key information
        return {
            'market_regime': 'neutral_market',
            'trend': 'neutral',
            'confidence': 0.5,
            'key_insights': [response[:200]],
            'market_data': market_data,
            'sector_data': sector_data,
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
            'overall_action': {'choice': 'hold', 'confidence': 0.5, 'reasoning': response[:200]},
            'risk_level': {'choice': 'moderate', 'confidence': 0.5},
            'market_outlook': {'choice': 'neutral', 'confidence': 0.5}
        }

