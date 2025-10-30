"""
Quantitative Analysis Agent - Runs simulations and calculates expected values.
"""
from typing import Dict, Any, List
import logging
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class QuantAnalysisAgent(BaseAgent):
    """
    Agent responsible for:
    - Running probability simulations
    - Calculating expected values
    - Performing scenario analysis
    - Backtesting strategies
    - Identifying optimal entry/exit points
    """
    
    def __init__(self):
        super().__init__(
            name="QuantAnalysisAgent",
            role="Perform quantitative analysis and calculate expected values"
        )
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform quantitative analysis and update state.
        
        Args:
            state: Current system state
            
        Returns:
            Updated state with quant analysis
        """
        logger.info(f"{self.name}: Running quantitative analysis...")
        
        positions = state.get('positions', [])
        market_data = state.get('market_data', {})
        
        quant_analysis = {
            'timestamp': datetime.now().isoformat(),
            'ev_calculations': self._calculate_evs(positions, market_data),
            'probability_analysis': self._analyze_probabilities(positions, market_data),
            'scenario_analysis': self._run_scenarios(positions, market_data),
            'optimal_actions': self._identify_optimal_actions(positions, market_data)
        }
        
        # Add to memory
        self.add_to_short_term_memory({
            'timestamp': datetime.now().isoformat(),
            'positions_analyzed': len(positions),
            'avg_ev': self._calculate_avg_ev(quant_analysis['ev_calculations'])
        })
        
        # Update state
        state['quant_analysis'] = quant_analysis
        
        logger.info(f"{self.name}: Quantitative analysis complete")
        return state
    
    def _calculate_evs(
        self,
        positions: List[Dict[str, Any]],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate expected values for all positions."""
        from ..analytics import EVCalculator
        
        ev_calculator = EVCalculator()
        ev_results = {}
        
        for position in positions:
            symbol = position.get('symbol')
            if symbol not in market_data:
                logger.warning(f"No market data for {symbol}")
                continue
            
            try:
                ev_result = ev_calculator.calculate_ev(
                    position=position,
                    market_data=market_data[symbol]
                )
                
                ev_results[symbol] = {
                    'expected_value': ev_result.expected_value,
                    'expected_return_pct': ev_result.expected_return_pct,
                    'probability_profit': ev_result.probability_profit,
                    'confidence_interval': ev_result.confidence_interval,
                    'method_breakdown': ev_result.method_breakdown,
                    'rating': self._rate_ev(ev_result)
                }
            except Exception as e:
                logger.error(f"Error calculating EV for {symbol}: {e}")
                ev_results[symbol] = {'error': str(e)}
        
        return ev_results
    
    def _analyze_probabilities(
        self,
        positions: List[Dict[str, Any]],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze probability distributions for positions."""
        from ..analytics import calculate_probability_itm
        
        probability_analysis = {}
        
        for position in positions:
            symbol = position.get('symbol')
            if symbol not in market_data:
                continue
            
            md = market_data[symbol]
            probabilities = {}
            
            for leg in position.get('legs', []):
                leg_key = f"{leg['option_type']}_{leg['strike']}"
                
                prob_itm = calculate_probability_itm(
                    option_type=leg['option_type'],
                    underlying_price=md['underlying_price'],
                    strike=leg['strike'],
                    time_to_expiry=leg['time_to_expiry'],
                    volatility=md['iv']
                )
                
                probabilities[leg_key] = {
                    'prob_itm': prob_itm,
                    'prob_otm': 1 - prob_itm,
                    'strike': leg['strike'],
                    'current_price': md['underlying_price']
                }
            
            probability_analysis[symbol] = probabilities
        
        return probability_analysis
    
    def _run_scenarios(
        self,
        positions: List[Dict[str, Any]],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run scenario analysis (bull, bear, neutral, high vol, low vol)."""
        scenarios = {
            'bull_case': {'price_change': 0.10, 'iv_change': -0.05},
            'bear_case': {'price_change': -0.10, 'iv_change': 0.10},
            'neutral_case': {'price_change': 0.0, 'iv_change': -0.02},
            'high_vol': {'price_change': 0.0, 'iv_change': 0.20},
            'low_vol': {'price_change': 0.0, 'iv_change': -0.20}
        }
        
        scenario_results = {}
        
        for scenario_name, scenario_params in scenarios.items():
            scenario_pnl = {}
            
            for position in positions:
                symbol = position.get('symbol')
                if symbol not in market_data:
                    continue
                
                # Calculate P&L under scenario
                pnl = self._calculate_scenario_pnl(
                    position,
                    market_data[symbol],
                    scenario_params
                )
                
                scenario_pnl[symbol] = pnl
            
            total_pnl = sum(scenario_pnl.values())
            scenario_results[scenario_name] = {
                'total_pnl': total_pnl,
                'by_position': scenario_pnl,
                'parameters': scenario_params
            }
        
        return scenario_results
    
    def _identify_optimal_actions(
        self,
        positions: List[Dict[str, Any]],
        market_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify optimal actions for each position."""
        actions = []
        
        for position in positions:
            symbol = position.get('symbol')
            if symbol not in market_data:
                continue
            
            # Check if position should be closed
            days_to_expiry = position.get('days_to_expiry', 999)
            current_pnl_pct = position.get('pnl_pct', 0)
            
            # Close if near expiration with profit
            if days_to_expiry < 7 and current_pnl_pct > 50:
                actions.append({
                    'symbol': symbol,
                    'action': 'close',
                    'reason': 'Near expiration with 50%+ profit',
                    'priority': 'high',
                    'expected_benefit': 'Lock in profits, avoid gamma risk'
                })
            
            # Close if significant loss
            elif current_pnl_pct < -50:
                actions.append({
                    'symbol': symbol,
                    'action': 'close',
                    'reason': 'Position down 50%+',
                    'priority': 'medium',
                    'expected_benefit': 'Cut losses, free up capital'
                })
            
            # Roll if profitable but time remaining
            elif current_pnl_pct > 70 and days_to_expiry > 14:
                actions.append({
                    'symbol': symbol,
                    'action': 'roll',
                    'reason': '70%+ profit with time remaining',
                    'priority': 'medium',
                    'expected_benefit': 'Capture more premium'
                })
            
            # Adjust if delta has shifted significantly
            position_delta = position.get('delta', 0)
            if abs(position_delta) > 50:
                actions.append({
                    'symbol': symbol,
                    'action': 'adjust',
                    'reason': f'Delta at {position_delta:.0f}',
                    'priority': 'low',
                    'expected_benefit': 'Reduce directional risk'
                })
        
        return actions
    
    def _calculate_scenario_pnl(
        self,
        position: Dict[str, Any],
        market_data: Dict[str, Any],
        scenario: Dict[str, float]
    ) -> float:
        """Calculate P&L under a specific scenario."""
        from ..analytics import black_scholes_price
        
        # Adjust market data for scenario
        new_price = market_data['underlying_price'] * (1 + scenario['price_change'])
        new_iv = market_data['iv'] * (1 + scenario['iv_change'])
        
        total_pnl = 0.0
        
        for leg in position.get('legs', []):
            # Calculate new option price
            new_option_price = black_scholes_price(
                option_type=leg['option_type'],
                underlying_price=new_price,
                strike=leg['strike'],
                time_to_expiry=leg['time_to_expiry'],
                volatility=new_iv
            )
            
            # Calculate P&L
            entry_price = leg.get('entry_price', 0)
            quantity = leg['quantity']
            multiplier = leg.get('multiplier', 100)
            is_short = leg.get('is_short', False)
            
            if is_short:
                pnl = (entry_price - new_option_price) * quantity * multiplier
            else:
                pnl = (new_option_price - entry_price) * quantity * multiplier
            
            total_pnl += pnl
        
        return total_pnl
    
    def _rate_ev(self, ev_result) -> str:
        """Rate the EV result (excellent, good, fair, poor)."""
        ev = ev_result.expected_value
        prob_profit = ev_result.probability_profit
        
        if ev > 50 and prob_profit > 0.6:
            return 'excellent'
        elif ev > 20 and prob_profit > 0.5:
            return 'good'
        elif ev > 0 and prob_profit > 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _calculate_avg_ev(self, ev_calculations: Dict[str, Any]) -> float:
        """Calculate average EV across all positions."""
        evs = [
            calc['expected_value']
            for calc in ev_calculations.values()
            if 'expected_value' in calc
        ]
        return sum(evs) / len(evs) if evs else 0.0

