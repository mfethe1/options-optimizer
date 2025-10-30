"""
Options Research Agent
Provides intelligent analysis and recommendations for options positions
Integrates with position context and real-time pricing
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import yfinance as yf

from ..data.position_manager import PositionManager, OptionPosition
from ..data.position_enrichment_service import PositionEnrichmentService
from .position_context_service import PositionContextService

logger = logging.getLogger(__name__)


class OptionsResearchAgent:
    """
    Intelligent agent for options position research and recommendations
    
    Capabilities:
    - Analyze individual option positions
    - Generate actionable recommendations
    - Provide real-time pricing updates
    - Assess risk levels
    - Suggest position adjustments
    """
    
    def __init__(
        self,
        position_manager: PositionManager,
        enrichment_service: PositionEnrichmentService,
        context_service: PositionContextService
    ):
        self.position_manager = position_manager
        self.enrichment_service = enrichment_service
        self.context_service = context_service
    
    def analyze_position(
        self,
        position_id: str,
        conversation_id: str = None
    ) -> Dict[str, Any]:
        """
        Analyze a single option position and generate recommendations
        
        Args:
            position_id: Position ID to analyze
            conversation_id: Optional conversation ID for context
        
        Returns:
            Dict with analysis and recommendations
        """
        # Get position
        position = self.position_manager.get_option_position(position_id)
        if not position:
            return {'error': f'Position {position_id} not found'}
        
        # Enrich with latest data
        self.enrichment_service.enrich_option_position(position)
        self.position_manager.save_positions()
        
        # Get market context
        market_context = self._get_market_context(position.symbol)
        
        # Analyze position
        analysis = {
            'position_id': position_id,
            'symbol': position.symbol,
            'option_type': position.option_type,
            'strike': position.strike,
            'expiration_date': position.expiration_date,
            'quantity': position.quantity,
            
            # Current metrics
            'current_price': position.current_price,
            'underlying_price': position.underlying_price,
            'premium_paid': position.premium_paid,
            'days_to_expiry': position.days_to_expiry(),
            
            # P&L
            'unrealized_pnl': position.unrealized_pnl,
            'unrealized_pnl_pct': position.unrealized_pnl_pct,
            
            # Greeks
            'delta': position.delta,
            'gamma': position.gamma,
            'theta': position.theta,
            'vega': position.vega,
            
            # Risk metrics
            'implied_volatility': position.implied_volatility,
            'probability_of_profit': position.probability_of_profit,
            'break_even_price': position.break_even_price,
            'risk_level': position.get_risk_level(),
            
            # Market context
            'market_context': market_context,
            
            # Timestamp
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Generate recommendation
        recommendation = self._generate_recommendation(position, market_context)
        analysis['recommendation'] = recommendation
        
        # Log to conversation if provided
        if conversation_id:
            self.context_service.log_agent_interaction(
                conversation_id=conversation_id,
                user_query=f"Analyze {position.symbol} {position.strike} {position.option_type}",
                agent_response=str(recommendation),
                positions_accessed=[position_id]
            )
        
        return analysis
    
    def analyze_portfolio(
        self,
        conversation_id: str = None
    ) -> Dict[str, Any]:
        """
        Analyze entire options portfolio
        
        Returns:
            Dict with portfolio-level analysis and recommendations
        """
        # Get all option positions
        positions = self.position_manager.get_all_option_positions()
        
        if not positions:
            return {'error': 'No option positions found'}
        
        # Enrich all positions
        self.enrichment_service.enrich_all_positions()
        
        # Analyze each position
        position_analyses = []
        for position in positions:
            analysis = self.analyze_position(position.position_id, conversation_id)
            position_analyses.append(analysis)
        
        # Calculate portfolio metrics
        total_value = sum(p.get('current_price', 0) * p.get('quantity', 0) * 100 
                         for p in position_analyses)
        total_pnl = sum(p.get('unrealized_pnl', 0) for p in position_analyses)
        total_pnl_pct = (total_pnl / total_value * 100) if total_value > 0 else 0
        
        # Count positions by status
        winning = sum(1 for p in position_analyses if p.get('unrealized_pnl', 0) > 0)
        losing = sum(1 for p in position_analyses if p.get('unrealized_pnl', 0) < 0)
        
        # Generate portfolio-level recommendations
        portfolio_recommendations = self._generate_portfolio_recommendations(
            position_analyses,
            total_pnl_pct
        )
        
        return {
            'total_positions': len(positions),
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'winning_positions': winning,
            'losing_positions': losing,
            'position_analyses': position_analyses,
            'recommendations': portfolio_recommendations,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def get_updated_pricing(
        self,
        position_id: str
    ) -> Dict[str, Any]:
        """
        Get real-time updated pricing for a position
        
        Args:
            position_id: Position ID
        
        Returns:
            Dict with updated pricing information
        """
        position = self.position_manager.get_option_position(position_id)
        if not position:
            return {'error': f'Position {position_id} not found'}
        
        # Refresh pricing
        self.enrichment_service.enrich_option_position(position)
        self.position_manager.save_positions()
        
        return {
            'position_id': position_id,
            'symbol': position.symbol,
            'current_price': position.current_price,
            'underlying_price': position.underlying_price,
            'unrealized_pnl': position.unrealized_pnl,
            'unrealized_pnl_pct': position.unrealized_pnl_pct,
            'delta': position.delta,
            'theta': position.theta,
            'implied_volatility': position.implied_volatility,
            'updated_at': datetime.now().isoformat()
        }
    
    def _get_market_context(self, symbol: str) -> Dict[str, Any]:
        """Get market context for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get recent price action
            hist = ticker.history(period='1mo')
            
            return {
                'current_price': info.get('currentPrice', 0),
                'day_change_pct': info.get('regularMarketChangePercent', 0),
                'month_high': hist['High'].max() if not hist.empty else 0,
                'month_low': hist['Low'].min() if not hist.empty else 0,
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'next_earnings': info.get('earningsDate', [None])[0] if info.get('earningsDate') else None
            }
        except Exception as e:
            logger.error(f"Error getting market context for {symbol}: {e}")
            return {}
    
    def _generate_recommendation(
        self,
        position: OptionPosition,
        market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate recommendation for a position"""
        days_to_expiry = position.days_to_expiry()
        pnl_pct = position.unrealized_pnl_pct or 0
        
        # Determine action based on multiple factors
        if pnl_pct > 50 and days_to_expiry < 30:
            action = "TAKE_PROFIT"
            reasoning = f"Strong profit ({pnl_pct:.1f}%) with limited time ({days_to_expiry}d). Lock in gains."
            urgency = "HIGH"
            adjustments = [
                "Close 50-75% of position immediately",
                "Set trailing stop on remainder",
                "Consider rolling to later expiration if still bullish"
            ]
        
        elif pnl_pct < -30 and days_to_expiry < 60:
            action = "CUT_LOSS"
            reasoning = f"Significant loss ({pnl_pct:.1f}%) with limited recovery time ({days_to_expiry}d)."
            urgency = "HIGH"
            adjustments = [
                "Close position to prevent further losses",
                "Review thesis - what changed?",
                "Consider tax loss harvesting"
            ]
        
        elif days_to_expiry < 7:
            action = "URGENT_DECISION"
            reasoning = f"Only {days_to_expiry} days left. Make final decision."
            urgency = "CRITICAL"
            adjustments = [
                "Close position if out of the money",
                "Exercise if deep in the money",
                "Roll to next month if thesis intact"
            ]
        
        elif pnl_pct > 20:
            action = "MONITOR_PROFIT"
            reasoning = f"Profitable position ({pnl_pct:.1f}%). Monitor for exit signals."
            urgency = "MEDIUM"
            adjustments = [
                "Set profit target at +50%",
                "Set stop loss at +10%",
                "Review weekly"
            ]
        
        elif pnl_pct < -15:
            action = "MONITOR_LOSS"
            reasoning = f"Losing position ({pnl_pct:.1f}%). Watch for reversal or cut."
            urgency = "MEDIUM"
            adjustments = [
                "Set stop loss at -25%",
                "Review thesis daily",
                "Consider averaging down if conviction high"
            ]
        
        else:
            action = "HOLD"
            reasoning = f"Position within normal range. {days_to_expiry}d to expiry."
            urgency = "LOW"
            adjustments = [
                "Monitor weekly",
                "Review at 30 days to expiry",
                "Watch for technical signals"
            ]
        
        # Add market context to reasoning
        if market_context.get('next_earnings'):
            reasoning += f" Earnings on {market_context['next_earnings']} - consider IV crush risk."
        
        return {
            'action': action,
            'reasoning': reasoning,
            'urgency': urgency,
            'suggested_adjustments': adjustments,
            'key_metrics': {
                'days_to_expiry': days_to_expiry,
                'pnl_pct': pnl_pct,
                'delta': position.delta,
                'theta': position.theta
            }
        }
    
    def _generate_portfolio_recommendations(
        self,
        position_analyses: List[Dict[str, Any]],
        total_pnl_pct: float
    ) -> List[str]:
        """Generate portfolio-level recommendations"""
        recommendations = []
        
        # Check overall P&L
        if total_pnl_pct < -20:
            recommendations.append(
                f"⚠️  Portfolio down {abs(total_pnl_pct):.1f}% - consider reducing options exposure"
            )
        elif total_pnl_pct > 30:
            recommendations.append(
                f"✅ Portfolio up {total_pnl_pct:.1f}% - consider taking profits on winners"
            )
        
        # Check for near-term expirations
        near_term = sum(1 for p in position_analyses if p.get('days_to_expiry', 999) < 30)
        if near_term > 0:
            recommendations.append(
                f"⏰ {near_term} position(s) expiring within 30 days - review urgently"
            )
        
        # Check for concentration
        symbols = [p.get('symbol') for p in position_analyses]
        if len(set(symbols)) < len(symbols):
            recommendations.append(
                "⚠️  Concentration risk detected - multiple positions in same underlying"
            )
        
        # Check for high-risk positions
        high_risk = sum(1 for p in position_analyses if p.get('risk_level') == 'HIGH')
        if high_risk > len(position_analyses) / 2:
            recommendations.append(
                f"⚠️  {high_risk} high-risk positions - consider hedging or reducing size"
            )
        
        return recommendations

