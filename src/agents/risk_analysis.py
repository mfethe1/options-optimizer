"""
Risk Analysis Agent - Calculates portfolio risk metrics and suggests hedges.
"""
from typing import Dict, Any, List
import logging
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class RiskAnalysisAgent(BaseAgent):
    """
    Agent responsible for:
    - Calculating portfolio Greeks
    - Monitoring position concentrations
    - Identifying tail risks
    - Suggesting hedging strategies
    - Tracking risk limits
    """
    
    def __init__(self):
        super().__init__(
            name="RiskAnalysisAgent",
            role="Analyze portfolio risk and suggest hedging strategies"
        )
        
        # Risk thresholds
        self.risk_limits = {
            'max_delta_exposure': 1000,
            'max_gamma_exposure': 500,
            'max_theta_decay': -500,
            'max_vega_exposure': 2000,
            'max_position_concentration': 0.25,  # 25% of portfolio
            'max_sector_concentration': 0.40,    # 40% of portfolio
        }
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze portfolio risk and update state with risk metrics.
        
        Args:
            state: Current system state
            
        Returns:
            Updated state with risk analysis
        """
        logger.info(f"{self.name}: Analyzing portfolio risk...")
        
        positions = state.get('positions', [])
        market_data = state.get('market_data', {})
        portfolio_greeks = state.get('portfolio_greeks', {})
        
        risk_analysis = {
            'timestamp': datetime.now().isoformat(),
            'greeks_analysis': self._analyze_greeks(portfolio_greeks),
            'concentration_risk': self._analyze_concentration(positions),
            'tail_risks': self._identify_tail_risks(positions, market_data),
            'hedge_suggestions': self._suggest_hedges(portfolio_greeks, positions),
            'risk_score': 0.0,
            'alerts': []
        }
        
        # Calculate overall risk score (0-100)
        risk_analysis['risk_score'] = self._calculate_risk_score(risk_analysis)
        
        # Generate alerts
        risk_analysis['alerts'] = self._generate_alerts(risk_analysis, portfolio_greeks)
        
        # Add to memory
        self.add_to_short_term_memory({
            'timestamp': datetime.now().isoformat(),
            'risk_score': risk_analysis['risk_score'],
            'alerts': risk_analysis['alerts']
        })
        
        # Update state
        state['risk_analysis'] = risk_analysis
        
        logger.info(f"{self.name}: Risk analysis complete. Risk score: {risk_analysis['risk_score']:.1f}")
        return state
    
    def _analyze_greeks(self, portfolio_greeks: Dict[str, float]) -> Dict[str, Any]:
        """Analyze portfolio Greeks against limits."""
        analysis = {}
        
        for greek, value in portfolio_greeks.items():
            limit_key = f'max_{greek}_exposure'
            if limit_key in self.risk_limits:
                limit = self.risk_limits[limit_key]
                utilization = abs(value) / abs(limit) if limit != 0 else 0
                
                analysis[greek] = {
                    'value': value,
                    'limit': limit,
                    'utilization_pct': utilization * 100,
                    'status': self._get_status(utilization)
                }
        
        return analysis
    
    def _analyze_concentration(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze position and sector concentration."""
        if not positions:
            return {'position_concentration': {}, 'sector_concentration': {}}
        
        # Calculate total portfolio value
        total_value = sum(
            abs(pos.get('market_value', 0)) for pos in positions
        )
        
        if total_value == 0:
            return {'position_concentration': {}, 'sector_concentration': {}}
        
        # Position concentration
        position_concentration = {}
        for pos in positions:
            symbol = pos.get('symbol')
            value = abs(pos.get('market_value', 0))
            concentration = value / total_value
            
            if concentration > 0.10:  # Report if > 10%
                position_concentration[symbol] = {
                    'value': value,
                    'concentration_pct': concentration * 100,
                    'status': self._get_status(concentration / self.risk_limits['max_position_concentration'])
                }
        
        # Sector concentration
        sector_values = {}
        for pos in positions:
            sector = pos.get('sector', 'Unknown')
            value = abs(pos.get('market_value', 0))
            sector_values[sector] = sector_values.get(sector, 0) + value
        
        sector_concentration = {}
        for sector, value in sector_values.items():
            concentration = value / total_value
            if concentration > 0.15:  # Report if > 15%
                sector_concentration[sector] = {
                    'value': value,
                    'concentration_pct': concentration * 100,
                    'status': self._get_status(concentration / self.risk_limits['max_sector_concentration'])
                }
        
        return {
            'position_concentration': position_concentration,
            'sector_concentration': sector_concentration
        }
    
    def _identify_tail_risks(
        self,
        positions: List[Dict[str, Any]],
        market_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify potential tail risk scenarios."""
        tail_risks = []
        
        for pos in positions:
            symbol = pos.get('symbol')
            if symbol not in market_data:
                continue
            
            # Check for earnings events
            days_to_earnings = market_data[symbol].get('days_to_earnings', 999)
            if days_to_earnings < 7:
                tail_risks.append({
                    'type': 'earnings_event',
                    'symbol': symbol,
                    'days_until': days_to_earnings,
                    'severity': 'high' if days_to_earnings < 3 else 'medium',
                    'description': f'Earnings in {days_to_earnings} days'
                })
            
            # Check for high IV (potential volatility crush)
            iv_rank = market_data[symbol].get('iv_rank', 50)
            if iv_rank > 80:
                tail_risks.append({
                    'type': 'high_iv',
                    'symbol': symbol,
                    'iv_rank': iv_rank,
                    'severity': 'medium',
                    'description': f'IV rank at {iv_rank}% - potential vol crush'
                })
            
            # Check for low liquidity
            avg_volume = market_data[symbol].get('avg_volume', 0)
            if avg_volume < 100000:
                tail_risks.append({
                    'type': 'low_liquidity',
                    'symbol': symbol,
                    'avg_volume': avg_volume,
                    'severity': 'medium',
                    'description': f'Low liquidity - avg volume {avg_volume:,.0f}'
                })
        
        return tail_risks
    
    def _suggest_hedges(
        self,
        portfolio_greeks: Dict[str, float],
        positions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Suggest hedging strategies based on portfolio Greeks."""
        suggestions = []
        
        # Delta hedge
        delta = portfolio_greeks.get('delta', 0)
        if abs(delta) > self.risk_limits['max_delta_exposure'] * 0.8:
            suggestions.append({
                'type': 'delta_hedge',
                'current_delta': delta,
                'action': 'sell' if delta > 0 else 'buy',
                'shares_needed': abs(delta),
                'priority': 'high' if abs(delta) > self.risk_limits['max_delta_exposure'] else 'medium',
                'description': f'Portfolio delta at {delta:.0f}, consider hedging with underlying'
            })
        
        # Gamma hedge
        gamma = portfolio_greeks.get('gamma', 0)
        if abs(gamma) > self.risk_limits['max_gamma_exposure'] * 0.8:
            suggestions.append({
                'type': 'gamma_hedge',
                'current_gamma': gamma,
                'action': 'reduce gamma exposure',
                'priority': 'medium',
                'description': f'High gamma exposure at {gamma:.0f}, consider reducing position sizes'
            })
        
        # Vega hedge
        vega = portfolio_greeks.get('vega', 0)
        if abs(vega) > self.risk_limits['max_vega_exposure'] * 0.8:
            suggestions.append({
                'type': 'vega_hedge',
                'current_vega': vega,
                'action': 'sell' if vega > 0 else 'buy',
                'priority': 'medium',
                'description': f'High vega exposure at {vega:.0f}, consider IV hedges'
            })
        
        return suggestions
    
    def _calculate_risk_score(self, risk_analysis: Dict[str, Any]) -> float:
        """
        Calculate overall risk score (0-100).
        Higher score = higher risk.
        """
        score = 0.0
        
        # Greeks utilization (40% weight)
        greeks_analysis = risk_analysis.get('greeks_analysis', {})
        if greeks_analysis:
            avg_utilization = sum(
                g['utilization_pct'] for g in greeks_analysis.values()
            ) / len(greeks_analysis)
            score += avg_utilization * 0.4
        
        # Concentration risk (30% weight)
        concentration = risk_analysis.get('concentration_risk', {})
        max_position_conc = max(
            (c['concentration_pct'] for c in concentration.get('position_concentration', {}).values()),
            default=0
        )
        score += max_position_conc * 0.3
        
        # Tail risks (30% weight)
        tail_risks = risk_analysis.get('tail_risks', [])
        high_severity_count = sum(1 for r in tail_risks if r.get('severity') == 'high')
        tail_risk_score = min(100, high_severity_count * 20)
        score += tail_risk_score * 0.3
        
        return min(100, score)
    
    def _generate_alerts(
        self,
        risk_analysis: Dict[str, Any],
        portfolio_greeks: Dict[str, float]
    ) -> List[Dict[str, str]]:
        """Generate risk alerts."""
        alerts = []
        
        # Greeks alerts
        for greek, data in risk_analysis.get('greeks_analysis', {}).items():
            if data['status'] == 'critical':
                alerts.append({
                    'level': 'critical',
                    'type': f'{greek}_limit',
                    'message': f'{greek.capitalize()} exposure at {data["utilization_pct"]:.0f}% of limit'
                })
        
        # Concentration alerts
        for symbol, data in risk_analysis.get('concentration_risk', {}).get('position_concentration', {}).items():
            if data['status'] in ['warning', 'critical']:
                alerts.append({
                    'level': data['status'],
                    'type': 'position_concentration',
                    'message': f'{symbol} represents {data["concentration_pct"]:.1f}% of portfolio'
                })
        
        # Tail risk alerts
        for risk in risk_analysis.get('tail_risks', []):
            if risk.get('severity') == 'high':
                alerts.append({
                    'level': 'warning',
                    'type': risk['type'],
                    'message': risk['description']
                })
        
        return alerts
    
    def _get_status(self, utilization: float) -> str:
        """Get status based on utilization."""
        if utilization >= 1.0:
            return 'critical'
        elif utilization >= 0.8:
            return 'warning'
        else:
            return 'normal'

