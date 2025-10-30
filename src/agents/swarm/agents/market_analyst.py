"""
Market Analyst Agent

Analyzes macro trends, sector rotation, market conditions, and overall market sentiment.
Provides high-level market context for portfolio decisions.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
import yfinance as yf

from ..base_swarm_agent import BaseSwarmAgent
from ..shared_context import SharedContext
from ..consensus_engine import ConsensusEngine

logger = logging.getLogger(__name__)


class MarketAnalystAgent(BaseSwarmAgent):
    """
    Market Analyst Agent - Macro trends and market conditions expert.
    
    Responsibilities:
    - Analyze overall market trends (bull/bear/neutral)
    - Identify sector rotation patterns
    - Monitor market breadth and momentum
    - Track major indices and correlations
    - Assess macroeconomic factors
    """
    
    def __init__(
        self,
        agent_id: str,
        shared_context: SharedContext,
        consensus_engine: ConsensusEngine
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="MarketAnalyst",
            shared_context=shared_context,
            consensus_engine=consensus_engine,
            priority=8,  # High priority for market updates
            confidence_threshold=0.6
        )
        
        # Market indices to track
        self.indices = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ',
            'DIA': 'Dow Jones',
            'IWM': 'Russell 2000',
            'VIX': 'Volatility Index'
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
    
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market conditions and trends.
        
        Args:
            context: Current context including portfolio and market data
        
        Returns:
            Market analysis results
        """
        try:
            logger.info(f"{self.agent_id}: Starting market analysis")
            
            # Fetch market data
            market_data = self._fetch_market_data()
            
            # Analyze trends
            trend_analysis = self._analyze_trends(market_data)
            
            # Analyze sectors
            sector_analysis = self._analyze_sectors()
            
            # Analyze volatility
            volatility_analysis = self._analyze_volatility(market_data)
            
            # Determine market regime
            market_regime = self._determine_market_regime(
                trend_analysis,
                volatility_analysis
            )
            
            analysis = {
                'market_data': market_data,
                'trend_analysis': trend_analysis,
                'sector_analysis': sector_analysis,
                'volatility_analysis': volatility_analysis,
                'market_regime': market_regime,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Send message to swarm
            self.send_message(
                content={
                    'type': 'market_analysis',
                    'market_regime': market_regime,
                    'key_insights': self._extract_key_insights(analysis)
                },
                priority=8,
                confidence=self._calculate_confidence(analysis)
            )
            
            # Update shared state
            self.update_state('market_regime', market_regime)
            self.update_state('market_trend', trend_analysis.get('overall_trend'))
            
            self.record_action('market_analysis', {'regime': market_regime})
            
            logger.info(f"{self.agent_id}: Market analysis complete - Regime: {market_regime}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"{self.agent_id}: Error in analysis: {e}")
            self.record_error(e, context)
            return {'error': str(e)}
    
    def make_recommendation(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make recommendations based on market analysis.
        
        Args:
            analysis: Market analysis results
        
        Returns:
            Recommendations
        """
        try:
            market_regime = analysis.get('market_regime', 'neutral')
            trend = analysis.get('trend_analysis', {}).get('overall_trend', 'neutral')
            volatility = analysis.get('volatility_analysis', {}).get('level', 'moderate')
            
            # Determine overall action
            if market_regime == 'bull_market' and volatility == 'low':
                overall_action = 'buy'
                risk_level = 'moderate'
                confidence = 0.8
            elif market_regime == 'bear_market':
                overall_action = 'hedge'
                risk_level = 'conservative'
                confidence = 0.7
            elif volatility == 'high':
                overall_action = 'hold'
                risk_level = 'conservative'
                confidence = 0.6
            else:
                overall_action = 'hold'
                risk_level = 'moderate'
                confidence = 0.5
            
            # Market outlook
            if trend == 'bullish' and volatility != 'high':
                market_outlook = 'bullish'
            elif trend == 'bearish' or volatility == 'high':
                market_outlook = 'bearish'
            else:
                market_outlook = 'neutral'
            
            recommendation = {
                'overall_action': {
                    'choice': overall_action,
                    'confidence': confidence,
                    'reasoning': f"Market regime: {market_regime}, Trend: {trend}, Volatility: {volatility}"
                },
                'risk_level': {
                    'choice': risk_level,
                    'confidence': confidence,
                    'reasoning': f"Based on {volatility} volatility and {market_regime} regime"
                },
                'market_outlook': {
                    'choice': market_outlook,
                    'confidence': confidence,
                    'reasoning': f"Trend is {trend} with {volatility} volatility"
                },
                'sector_recommendations': analysis.get('sector_analysis', {}).get('top_sectors', []),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"{self.agent_id}: Recommendation - Action: {overall_action}, Outlook: {market_outlook}")
            
            return recommendation
            
        except Exception as e:
            logger.error(f"{self.agent_id}: Error making recommendation: {e}")
            self.record_error(e, analysis)
            return {'error': str(e)}
    
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
                        'current_price': current_price,
                        'change_pct_1m': change_pct,
                        'volume': hist['Volume'].iloc[-1],
                        'avg_volume': hist['Volume'].mean()
                    }
            except Exception as e:
                logger.warning(f"Error fetching {symbol}: {e}")
        
        return data
    
    def _analyze_trends(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market trends"""
        # Count bullish vs bearish indices
        bullish_count = sum(
            1 for data in market_data.values()
            if data.get('change_pct_1m', 0) > 0
        )
        total_count = len(market_data)
        
        if bullish_count / total_count > 0.7:
            overall_trend = 'bullish'
        elif bullish_count / total_count < 0.3:
            overall_trend = 'bearish'
        else:
            overall_trend = 'neutral'
        
        return {
            'overall_trend': overall_trend,
            'bullish_indices': bullish_count,
            'total_indices': total_count,
            'bullish_ratio': bullish_count / total_count if total_count > 0 else 0
        }
    
    def _analyze_sectors(self) -> Dict[str, Any]:
        """Analyze sector performance"""
        sector_performance = {}
        
        for symbol, name in self.sectors.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1mo')
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[0]
                    change_pct = ((current_price - prev_price) / prev_price) * 100
                    
                    sector_performance[name] = {
                        'symbol': symbol,
                        'change_pct_1m': change_pct,
                        'relative_strength': change_pct  # Simplified
                    }
            except Exception as e:
                logger.warning(f"Error analyzing sector {symbol}: {e}")
        
        # Sort by performance
        sorted_sectors = sorted(
            sector_performance.items(),
            key=lambda x: x[1]['change_pct_1m'],
            reverse=True
        )
        
        return {
            'sector_performance': sector_performance,
            'top_sectors': [s[0] for s in sorted_sectors[:3]],
            'bottom_sectors': [s[0] for s in sorted_sectors[-3:]]
        }
    
    def _analyze_volatility(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market volatility"""
        vix_data = market_data.get('VIX', {})
        vix_price = vix_data.get('current_price', 20)
        
        if vix_price < 15:
            level = 'low'
        elif vix_price < 25:
            level = 'moderate'
        else:
            level = 'high'
        
        return {
            'vix_level': vix_price,
            'level': level,
            'interpretation': f"VIX at {vix_price:.2f} indicates {level} volatility"
        }
    
    def _determine_market_regime(
        self,
        trend_analysis: Dict[str, Any],
        volatility_analysis: Dict[str, Any]
    ) -> str:
        """Determine overall market regime"""
        trend = trend_analysis.get('overall_trend', 'neutral')
        volatility = volatility_analysis.get('level', 'moderate')
        
        if trend == 'bullish' and volatility == 'low':
            return 'bull_market'
        elif trend == 'bearish' and volatility == 'high':
            return 'bear_market'
        elif volatility == 'high':
            return 'volatile_market'
        else:
            return 'neutral_market'
    
    def _extract_key_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract key insights from analysis"""
        insights = []
        
        regime = analysis.get('market_regime', 'neutral_market')
        insights.append(f"Market regime: {regime}")
        
        trend = analysis.get('trend_analysis', {}).get('overall_trend', 'neutral')
        insights.append(f"Overall trend: {trend}")
        
        top_sectors = analysis.get('sector_analysis', {}).get('top_sectors', [])
        if top_sectors:
            insights.append(f"Top performing sectors: {', '.join(top_sectors[:2])}")
        
        return insights

