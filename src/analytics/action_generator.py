"""
Action Generator - Convert scores into specific, actionable recommendations
"""
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ActionGenerator:
    """
    Generate specific trade recommendations from scores
    
    Converts abstract recommendations (BUY/SELL/HOLD) into
    concrete actions (sell 51 shares, set stop at $175, etc.)
    """
    
    def __init__(self):
        pass
    
    def generate_actions(
        self,
        symbol: str,
        position: Optional[Dict[str, Any]],
        scores: Dict[str, Any],
        combined_score: float,
        recommendation: str,
        market_data: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate specific actions based on recommendation
        
        Args:
            symbol: Stock symbol
            position: Current position data
            scores: Individual scorer results
            combined_score: Overall score (0-100)
            recommendation: Recommendation string
            market_data: Market data for the symbol
            
        Returns:
            List of specific actions to take
        """
        logger.info(f"Generating actions for {symbol}: {recommendation}")
        
        actions = []
        
        try:
            # Get current price
            current_price = self._get_current_price(position, market_data)
            
            # Generate actions based on recommendation
            if recommendation == "STRONG_HOLD_ADD":
                actions.extend(self._generate_hold_add_actions(position, current_price, scores))
            
            elif recommendation == "HOLD":
                actions.extend(self._generate_hold_actions(position, current_price, scores))
            
            elif recommendation == "HOLD_TRIM":
                actions.extend(self._generate_trim_actions(position, current_price, scores, trim_pct=0.15))
            
            elif recommendation == "REDUCE":
                actions.extend(self._generate_trim_actions(position, current_price, scores, trim_pct=0.35))
            
            elif recommendation == "CLOSE":
                actions.extend(self._generate_close_actions(position, current_price))
            
            elif recommendation == "STRONG_BUY":
                actions.extend(self._generate_buy_actions(symbol, current_price, scores, strength='strong'))
            
            elif recommendation == "BUY":
                actions.extend(self._generate_buy_actions(symbol, current_price, scores, strength='moderate'))
            
            elif recommendation == "WATCH":
                actions.extend(self._generate_watch_actions(symbol, current_price, scores))
            
            # Always add risk management actions
            actions.extend(self._generate_risk_management_actions(position, current_price, scores))
            
            # Sort by priority
            actions.sort(key=lambda x: x.get('priority', 99))
            
            return actions
            
        except Exception as e:
            logger.error(f"Error generating actions: {e}")
            return []
    
    def _get_current_price(self, position: Optional[Dict], market_data: Optional[Dict]) -> float:
        """Get current price from available data"""
        price = None

        if market_data and 'current_price' in market_data:
            price = market_data['current_price']
        elif position and 'current_price' in position:
            price = position['current_price']
        elif position and 'entry_price' in position:
            price = position['entry_price']  # Fallback to entry price

        # Ensure we have a valid price (handle dict case)
        if isinstance(price, dict):
            logger.warning(f"Price is a dict, extracting value: {price}")
            price = None  # Reset and try to extract

        if price is None or (isinstance(price, (int, float)) and price <= 0):
            logger.warning("No valid price found, using default 100.0")
            return 100.0

        return float(price)
    
    def _generate_hold_add_actions(
        self,
        position: Optional[Dict],
        current_price: float,
        scores: Dict
    ) -> List[Dict]:
        """Generate actions for STRONG_HOLD_ADD"""
        actions = []
        
        if position:
            # Strong position - consider adding
            actions.append({
                'action': 'HOLD',
                'instrument': 'stock',
                'quantity': position.get('quantity', 0),
                'reason': 'Strong fundamentals and technicals support holding',
                'priority': 1
            })
            
            actions.append({
                'action': 'CONSIDER_ADDING',
                'instrument': 'stock',
                'suggested_quantity': int(position.get('quantity', 0) * 0.10),  # Add 10%
                'price_target': current_price * 0.98,  # On 2% dip
                'reason': 'Strong scores support adding on dips',
                'priority': 2
            })
        
        return actions
    
    def _generate_hold_actions(
        self,
        position: Optional[Dict],
        current_price: float,
        scores: Dict
    ) -> List[Dict]:
        """Generate actions for HOLD"""
        actions = []
        
        if position:
            actions.append({
                'action': 'HOLD',
                'instrument': 'stock',
                'quantity': position.get('quantity', 0),
                'reason': 'Balanced risk/reward supports holding current position',
                'priority': 1
            })
        
        return actions
    
    def _generate_trim_actions(
        self,
        position: Optional[Dict],
        current_price: float,
        scores: Dict,
        trim_pct: float = 0.25
    ) -> List[Dict]:
        """Generate actions for trimming position"""
        actions = []
        
        if not position:
            return actions
        
        quantity = position.get('quantity', 0)
        if quantity == 0:
            return actions
        
        # Calculate trim amount
        trim_qty = int(quantity * trim_pct)
        
        if trim_qty > 0:
            actions.append({
                'action': 'SELL',
                'instrument': 'stock',
                'quantity': trim_qty,
                'price_target': 'market',
                'reason': f'Take partial profits ({trim_pct*100:.0f}% of position)',
                'expected_proceeds': trim_qty * current_price,
                'priority': 1
            })
            
            # Hold remaining
            remaining = quantity - trim_qty
            actions.append({
                'action': 'HOLD',
                'instrument': 'stock',
                'quantity': remaining,
                'reason': 'Maintain core position for potential upside',
                'priority': 2
            })
        
        return actions
    
    def _generate_close_actions(
        self,
        position: Optional[Dict],
        current_price: float
    ) -> List[Dict]:
        """Generate actions for closing position"""
        actions = []
        
        if not position:
            return actions
        
        quantity = position.get('quantity', 0)
        if quantity > 0:
            actions.append({
                'action': 'SELL',
                'instrument': 'stock',
                'quantity': quantity,
                'price_target': 'market',
                'reason': 'Close position due to deteriorating fundamentals/technicals',
                'expected_proceeds': quantity * current_price,
                'priority': 1
            })
        
        return actions
    
    def _generate_buy_actions(
        self,
        symbol: str,
        current_price: float,
        scores: Dict,
        strength: str = 'moderate'
    ) -> List[Dict]:
        """Generate actions for buying"""
        actions = []
        
        # Suggest entry
        if strength == 'strong':
            reason = 'Strong buy signal - excellent fundamentals and technicals'
            priority = 1
        else:
            reason = 'Buy signal - favorable risk/reward'
            priority = 2
        
        actions.append({
            'action': 'BUY',
            'instrument': 'stock',
            'symbol': symbol,
            'suggested_entry': current_price,
            'suggested_stop': current_price * 0.92,  # 8% stop
            'suggested_target': current_price * 1.15,  # 15% target
            'reason': reason,
            'priority': priority
        })
        
        return actions
    
    def _generate_watch_actions(
        self,
        symbol: str,
        current_price: float,
        scores: Dict
    ) -> List[Dict]:
        """Generate actions for watching"""
        actions = []
        
        actions.append({
            'action': 'WATCH',
            'symbol': symbol,
            'watch_for': 'Improvement in technical or fundamental scores',
            'entry_trigger': current_price * 0.95,  # Buy on 5% dip
            'reason': 'Neutral signals - wait for better entry',
            'priority': 1
        })
        
        return actions
    
    def _generate_risk_management_actions(
        self,
        position: Optional[Dict],
        current_price: float,
        scores: Dict
    ) -> List[Dict]:
        """Generate risk management actions (stops, targets)"""
        actions = []
        
        if not position:
            return actions
        
        # Check if stop loss exists
        if not position.get('stop_loss') and current_price > 0:
            # Calculate stop loss based on risk score
            risk_score = scores.get('risk', {}).score if hasattr(scores.get('risk', {}), 'score') else 50
            
            # Higher risk = tighter stop
            if risk_score > 70:
                stop_pct = 0.92  # 8% stop
            elif risk_score > 50:
                stop_pct = 0.90  # 10% stop
            else:
                stop_pct = 0.88  # 12% stop
            
            entry_price = position.get('entry_price', current_price)
            stop_price = max(entry_price * stop_pct, current_price * stop_pct)
            
            actions.append({
                'action': 'SET_STOP',
                'instrument': 'stock',
                'quantity': position.get('quantity', 0),
                'stop_price': round(stop_price, 2),
                'reason': 'Protect downside with stop loss',
                'priority': 3
            })
        
        # Check if target price exists
        if not position.get('target_price') and current_price > 0:
            # Calculate target based on combined score
            combined_score = scores.get('combined_score', 50)
            
            if combined_score > 70:
                target_pct = 1.20  # 20% target
            elif combined_score > 55:
                target_pct = 1.15  # 15% target
            else:
                target_pct = 1.10  # 10% target
            
            target_price = current_price * target_pct
            
            actions.append({
                'action': 'SET_TARGET',
                'instrument': 'stock',
                'quantity': position.get('quantity', 0),
                'target_price': round(target_price, 2),
                'reason': 'Take profits at target price',
                'priority': 4
            })
        
        return actions

