"""
LLM-Powered Sentiment Analyst Agent

Uses actual LLM calls + Firecrawl for intelligent sentiment analysis
based on real news and social media data.
"""

import logging
import json
from typing import Dict, Any, List
from datetime import datetime

from ..base_swarm_agent import BaseSwarmAgent
from ..shared_context import SharedContext
from ..consensus_engine import ConsensusEngine
from ..llm_agent_base import LLMAgentBase, FirecrawlMixin

logger = logging.getLogger(__name__)


class LLMSentimentAnalystAgent(BaseSwarmAgent, LLMAgentBase, FirecrawlMixin):
    """
    LLM-Powered Sentiment Analyst - Uses AI + web data for sentiment analysis.
    
    Combines:
    - Real news data from Firecrawl
    - Social media sentiment
    - AI-powered sentiment analysis from LLMs
    """
    
    def __init__(
        self,
        agent_id: str,
        shared_context: SharedContext,
        consensus_engine: ConsensusEngine,
        preferred_model: str = "anthropic"
    ):
        # Initialize all parent classes
        BaseSwarmAgent.__init__(
            self,
            agent_id=agent_id,
            agent_type="SentimentAnalyst",
            shared_context=shared_context,
            consensus_engine=consensus_engine,
            priority=6,
            confidence_threshold=0.6
        )
        
        LLMAgentBase.__init__(
            self,
            agent_id=agent_id,
            agent_type="SentimentAnalyst",
            preferred_model=preferred_model
        )
    
    def get_system_prompt(self) -> str:
        """Get specialized system prompt for sentiment analysis"""
        return """You are an expert Sentiment Analyst specializing in market sentiment and investor psychology.

Your responsibilities:
- Analyze news sentiment for stocks and sectors
- Assess social media sentiment (Twitter, Reddit, StockTwits)
- Identify sentiment trends and shifts
- Evaluate fear/greed indicators
- Provide sentiment-based trading insights

Provide concise, data-driven sentiment analysis.
Format your response as JSON with clear structure."""
    
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market sentiment using LLM + Firecrawl.
        
        Args:
            context: Current context including portfolio data
        
        Returns:
            LLM-powered sentiment analysis
        """
        try:
            logger.info(f"{self.agent_id}: Starting LLM-powered sentiment analysis")
            
            # Extract symbols from portfolio
            portfolio = context.get('portfolio', {})
            symbols = self._extract_symbols(portfolio)
            
            # Gather sentiment data using Firecrawl
            sentiment_data = self._gather_sentiment_data(symbols)
            
            # Build comprehensive prompt
            prompt = self._build_sentiment_analysis_prompt(sentiment_data, portfolio, context)
            
            # Call LLM for intelligent sentiment analysis
            llm_response = self.call_llm(
                prompt=prompt,
                system_prompt=self.get_system_prompt(),
                temperature=0.7,
                max_tokens=2000
            )
            
            # Parse LLM response
            analysis = self._parse_llm_response(llm_response, sentiment_data)
            
            # Send message to swarm
            self.send_message(
                content={
                    'type': 'sentiment_analysis',
                    'overall_sentiment': analysis.get('overall_sentiment', 'neutral'),
                    'key_insights': analysis.get('key_insights', []),
                    'llm_powered': True
                },
                priority=6,
                confidence=analysis.get('confidence', 0.6)
            )
            
            # Update shared state
            self.update_state('market_sentiment', analysis.get('overall_sentiment'))
            
            self.record_action('llm_sentiment_analysis', {
                'sentiment': analysis.get('overall_sentiment'),
                'model': self.preferred_model
            })
            
            logger.info(f"{self.agent_id}: LLM sentiment analysis complete - Sentiment: {analysis.get('overall_sentiment')}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"{self.agent_id}: Error in LLM sentiment analysis: {e}")
            self.record_error(e, context)
            return {'error': str(e), 'llm_powered': False}
    
    def make_recommendation(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make sentiment-based recommendations using LLM.
        
        Args:
            analysis: Sentiment analysis results
        
        Returns:
            LLM-powered sentiment recommendations
        """
        try:
            # Build prompt for recommendations
            prompt = f"""Based on this sentiment analysis, provide trading recommendations:

Sentiment Analysis:
{json.dumps(analysis, indent=2)}

Provide recommendations in JSON format with:
1. market_outlook: (bullish/bearish/neutral)
2. confidence: (0.0 to 1.0)
3. reasoning: (brief explanation based on sentiment)
4. sentiment_score: (-1.0 to 1.0, where -1 is very bearish, 1 is very bullish)
5. key_sentiment_drivers: (list of main factors driving sentiment)
6. contrarian_opportunities: (if sentiment is extreme, identify contrarian plays)

Be specific and actionable."""
            
            llm_response = self.call_llm(
                prompt=prompt,
                system_prompt=self.get_system_prompt(),
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse recommendation
            recommendation = self._parse_recommendation(llm_response)
            recommendation['timestamp'] = datetime.utcnow().isoformat()
            recommendation['llm_powered'] = True
            
            logger.info(f"{self.agent_id}: LLM recommendation - Outlook: {recommendation.get('market_outlook', {}).get('choice')}")
            
            return recommendation
            
        except Exception as e:
            logger.error(f"{self.agent_id}: Error making LLM recommendation: {e}")
            self.record_error(e, analysis)
            return {'error': str(e), 'llm_powered': False}
    
    def _extract_symbols(self, portfolio: Dict) -> List[str]:
        """Extract unique symbols from portfolio"""
        symbols = set()
        for pos in portfolio.get('positions', []):
            symbol = pos.get('symbol')
            if symbol:
                symbols.add(symbol)
        return list(symbols)
    
    def _gather_sentiment_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Gather sentiment data using Firecrawl"""
        sentiment_data = {
            'symbols': {},
            'market_news': {},
            'social_sentiment': {}
        }
        
        # Get news for each symbol (using Firecrawl)
        for symbol in symbols[:5]:  # Limit to top 5 to avoid rate limits
            try:
                # Get news via Firecrawl
                news = self.get_news(symbol, days=3)
                sentiment_data['symbols'][symbol] = {
                    'news': news,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Get social sentiment via Firecrawl
                social = self.get_social_sentiment(symbol)
                sentiment_data['social_sentiment'][symbol] = social
                
            except Exception as e:
                logger.warning(f"Error gathering sentiment for {symbol}: {e}")
        
        # Get general market news
        try:
            market_news = self.search_web("stock market news today", max_results=5)
            sentiment_data['market_news'] = market_news
        except Exception as e:
            logger.warning(f"Error gathering market news: {e}")
        
        return sentiment_data
    
    def _build_sentiment_analysis_prompt(
        self,
        sentiment_data: Dict,
        portfolio: Dict,
        context: Dict
    ) -> str:
        """Build comprehensive prompt for sentiment analysis"""
        
        symbols = list(sentiment_data.get('symbols', {}).keys())
        
        prompt = f"""Analyze market sentiment based on the following data:

PORTFOLIO SYMBOLS: {', '.join(symbols)}

SENTIMENT DATA GATHERED:
{json.dumps(sentiment_data, indent=2)}

Provide a comprehensive sentiment analysis in JSON format with:
1. overall_sentiment: (bullish/bearish/neutral)
2. confidence: (0.0 to 1.0)
3. sentiment_score: (-1.0 to 1.0)
4. key_insights: (list of 3-5 key sentiment observations)
5. symbol_sentiments: (sentiment for each symbol)
6. fear_greed_indicator: (extreme_fear/fear/neutral/greed/extreme_greed)
7. sentiment_trends: (improving/stable/deteriorating)
8. news_highlights: (top 3 news items affecting sentiment)

Be specific and data-driven. If data is limited, note that in your analysis."""
        
        return prompt
    
    def _parse_llm_response(self, response: str, sentiment_data: Dict) -> Dict[str, Any]:
        """Parse LLM response into structured sentiment analysis"""
        try:
            # Try to parse as JSON
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                parsed = json.loads(json_str)
                
                # Add raw data
                parsed['raw_sentiment_data'] = sentiment_data
                parsed['timestamp'] = datetime.utcnow().isoformat()
                
                return parsed
        except:
            pass
        
        # Fallback: extract key information
        return {
            'overall_sentiment': 'neutral',
            'confidence': 0.5,
            'sentiment_score': 0.0,
            'key_insights': [response[:200]],
            'raw_sentiment_data': sentiment_data,
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
            'market_outlook': {'choice': 'neutral', 'confidence': 0.5, 'reasoning': response[:200]},
            'sentiment_score': 0.0,
            'key_sentiment_drivers': []
        }

