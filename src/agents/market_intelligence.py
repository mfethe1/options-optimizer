"""
Market Intelligence Agent - Monitors real-time market data and identifies opportunities.
"""
from typing import Dict, Any
import logging
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class MarketIntelligenceAgent(BaseAgent):
    """
    Agent responsible for:
    - Monitoring real-time market data
    - Tracking IV changes and volume spikes
    - Identifying unusual options activity
    - Detecting gamma exposure shifts
    """
    
    def __init__(self):
        super().__init__(
            name="MarketIntelligenceAgent",
            role="Monitor market conditions and identify trading opportunities"
        )
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market conditions and update state with insights.
        
        Args:
            state: Current system state containing positions and market data
            
        Returns:
            Updated state with market intelligence insights
        """
        logger.info(f"{self.name}: Analyzing market conditions...")
        
        positions = state.get('positions', [])
        market_data = state.get('market_data', {})
        
        insights = {
            'timestamp': datetime.now().isoformat(),
            'iv_changes': self._analyze_iv_changes(positions, market_data),
            'volume_anomalies': self._detect_volume_anomalies(positions, market_data),
            'unusual_activity': self._identify_unusual_activity(positions, market_data),
            'gamma_exposure': self._calculate_gamma_exposure(positions, market_data)
        }
        
        # Add to short-term memory
        self.add_to_short_term_memory({
            'timestamp': datetime.now().isoformat(),
            'insights': insights
        })
        
        # Update state
        state['market_intelligence'] = insights
        
        logger.info(f"{self.name}: Market analysis complete")
        return state
    
    def _analyze_iv_changes(self, positions: list, market_data: dict) -> Dict[str, Any]:
        """Analyze implied volatility changes."""
        iv_changes = {}
        
        for position in positions:
            symbol = position.get('symbol')
            if symbol in market_data:
                current_iv = market_data[symbol].get('iv', 0)
                historical_iv = market_data[symbol].get('historical_iv', 0)
                
                if historical_iv > 0:
                    iv_change_pct = ((current_iv - historical_iv) / historical_iv) * 100
                    
                    if abs(iv_change_pct) > 10:  # Significant change threshold
                        iv_changes[symbol] = {
                            'current_iv': current_iv,
                            'historical_iv': historical_iv,
                            'change_pct': iv_change_pct,
                            'significance': 'high' if abs(iv_change_pct) > 20 else 'medium'
                        }
        
        return iv_changes
    
    def _detect_volume_anomalies(self, positions: list, market_data: dict) -> Dict[str, Any]:
        """Detect unusual volume patterns."""
        anomalies = {}
        
        for position in positions:
            symbol = position.get('symbol')
            if symbol in market_data:
                current_volume = market_data[symbol].get('volume', 0)
                avg_volume = market_data[symbol].get('avg_volume', 0)
                
                if avg_volume > 0:
                    volume_ratio = current_volume / avg_volume
                    
                    if volume_ratio > 2.0:  # 2x average volume
                        anomalies[symbol] = {
                            'current_volume': current_volume,
                            'avg_volume': avg_volume,
                            'ratio': volume_ratio,
                            'alert_level': 'high' if volume_ratio > 3.0 else 'medium'
                        }
        
        return anomalies
    
    def _identify_unusual_activity(self, positions: list, market_data: dict) -> Dict[str, Any]:
        """Identify unusual options activity."""
        unusual_activity = {}
        
        for position in positions:
            symbol = position.get('symbol')
            if symbol in market_data:
                put_call_ratio = market_data[symbol].get('put_call_ratio', 1.0)
                open_interest = market_data[symbol].get('open_interest', 0)
                
                # Unusual put/call ratio
                if put_call_ratio > 1.5 or put_call_ratio < 0.5:
                    unusual_activity[symbol] = {
                        'put_call_ratio': put_call_ratio,
                        'open_interest': open_interest,
                        'signal': 'bearish' if put_call_ratio > 1.5 else 'bullish'
                    }
        
        return unusual_activity
    
    def _calculate_gamma_exposure(self, positions: list, market_data: dict) -> Dict[str, Any]:
        """Calculate dealer gamma exposure."""
        gamma_exposure = {}
        
        for position in positions:
            symbol = position.get('symbol')
            if symbol in market_data:
                gamma = market_data[symbol].get('gamma', 0)
                delta = market_data[symbol].get('delta', 0)
                underlying_price = market_data[symbol].get('underlying_price', 0)
                
                # Simplified gamma exposure calculation
                exposure = gamma * delta * underlying_price * position.get('quantity', 0)
                
                gamma_exposure[symbol] = {
                    'gamma': gamma,
                    'delta': delta,
                    'exposure': exposure,
                    'underlying_price': underlying_price
                }
        
        return gamma_exposure

