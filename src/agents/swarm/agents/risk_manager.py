"""
Risk Manager Agent

Monitors portfolio risk, calculates position sizing, tracks Greeks exposure,
and ensures risk limits are maintained.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

from ..base_swarm_agent import BaseSwarmAgent
from ..shared_context import SharedContext
from ..consensus_engine import ConsensusEngine

logger = logging.getLogger(__name__)


class RiskManagerAgent(BaseSwarmAgent):
    """
    Risk Manager Agent - Portfolio risk and position sizing expert.
    
    Responsibilities:
    - Monitor portfolio Greeks (Delta, Gamma, Theta, Vega)
    - Calculate position sizing using Kelly Criterion
    - Track portfolio-level risk metrics
    - Enforce risk limits and circuit breakers
    - Assess concentration risk
    - Monitor P&L and drawdowns
    """
    
    def __init__(
        self,
        agent_id: str,
        shared_context: SharedContext,
        consensus_engine: ConsensusEngine,
        max_portfolio_delta: float = 100.0,
        max_position_size_pct: float = 0.10,
        max_drawdown_pct: float = 0.15
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="RiskManager",
            shared_context=shared_context,
            consensus_engine=consensus_engine,
            priority=10,  # Highest priority for risk alerts
            confidence_threshold=0.7
        )
        
        # Risk limits
        self.max_portfolio_delta = max_portfolio_delta
        self.max_position_size_pct = max_position_size_pct
        self.max_drawdown_pct = max_drawdown_pct
        
        # Risk thresholds
        self.risk_thresholds = {
            'delta': {'low': 20, 'medium': 50, 'high': 80},
            'gamma': {'low': 5, 'medium': 15, 'high': 30},
            'theta': {'low': -50, 'medium': -150, 'high': -300},
            'vega': {'low': 20, 'medium': 50, 'high': 100}
        }
    
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze portfolio risk.
        
        Args:
            context: Current context including portfolio data
        
        Returns:
            Risk analysis results
        """
        try:
            logger.info(f"{self.agent_id}: Starting risk analysis")
            
            portfolio_data = context.get('portfolio', {})
            
            # Calculate portfolio Greeks
            greeks_analysis = self._analyze_greeks(portfolio_data)
            
            # Analyze position sizing
            position_analysis = self._analyze_positions(portfolio_data)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(portfolio_data)
            
            # Check risk limits
            limit_violations = self._check_risk_limits(
                greeks_analysis,
                position_analysis,
                risk_metrics
            )
            
            # Determine overall risk level
            risk_level = self._determine_risk_level(
                greeks_analysis,
                position_analysis,
                limit_violations
            )
            
            analysis = {
                'greeks_analysis': greeks_analysis,
                'position_analysis': position_analysis,
                'risk_metrics': risk_metrics,
                'limit_violations': limit_violations,
                'risk_level': risk_level,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Send high-priority message if violations detected
            if limit_violations:
                self.send_message(
                    content={
                        'type': 'risk_alert',
                        'risk_level': risk_level,
                        'violations': limit_violations
                    },
                    priority=10,
                    confidence=0.9
                )
            
            # Update shared state
            self.update_state('portfolio_risk_level', risk_level)
            self.update_state('portfolio_delta', greeks_analysis.get('total_delta', 0))
            
            self.record_action('risk_analysis', {
                'risk_level': risk_level,
                'violations': len(limit_violations)
            })
            
            logger.info(f"{self.agent_id}: Risk analysis complete - Level: {risk_level}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"{self.agent_id}: Error in analysis: {e}")
            self.record_error(e, context)
            return {'error': str(e)}
    
    def make_recommendation(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make risk management recommendations.
        
        Args:
            analysis: Risk analysis results
        
        Returns:
            Risk management recommendations
        """
        try:
            risk_level = analysis.get('risk_level', 'moderate')
            violations = analysis.get('limit_violations', [])
            greeks = analysis.get('greeks_analysis', {})
            
            # Determine actions based on risk level
            if risk_level == 'critical' or len(violations) > 2:
                overall_action = 'sell'  # Reduce exposure
                risk_recommendation = 'conservative'
                confidence = 0.9
                reasoning = f"Critical risk level with {len(violations)} violations"
            elif risk_level == 'high' or len(violations) > 0:
                overall_action = 'hedge'
                risk_recommendation = 'conservative'
                confidence = 0.8
                reasoning = f"High risk level with {len(violations)} violations"
            elif risk_level == 'moderate':
                overall_action = 'hold'
                risk_recommendation = 'moderate'
                confidence = 0.6
                reasoning = "Moderate risk level, maintain current positions"
            else:
                overall_action = 'buy'
                risk_recommendation = 'moderate'
                confidence = 0.7
                reasoning = "Low risk level, can increase exposure"
            
            # Position sizing recommendations
            position_sizing = self._calculate_position_sizing(analysis)
            
            # Hedging recommendations
            hedging_recs = self._generate_hedging_recommendations(greeks)
            
            recommendation = {
                'overall_action': {
                    'choice': overall_action,
                    'confidence': confidence,
                    'reasoning': reasoning
                },
                'risk_level': {
                    'choice': risk_recommendation,
                    'confidence': confidence,
                    'reasoning': f"Based on {risk_level} portfolio risk"
                },
                'position_sizing': position_sizing,
                'hedging_recommendations': hedging_recs,
                'violations': violations,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"{self.agent_id}: Recommendation - Action: {overall_action}, Risk: {risk_recommendation}")
            
            return recommendation
            
        except Exception as e:
            logger.error(f"{self.agent_id}: Error making recommendation: {e}")
            self.record_error(e, analysis)
            return {'error': str(e)}
    
    def _analyze_greeks(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze portfolio Greeks"""
        positions = portfolio_data.get('positions', [])
        
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        
        for position in positions:
            if position.get('asset_type') == 'option':
                quantity = position.get('quantity', 0)
                total_delta += position.get('delta', 0) * quantity
                total_gamma += position.get('gamma', 0) * quantity
                total_theta += position.get('theta', 0) * quantity
                total_vega += position.get('vega', 0) * quantity
        
        # Determine risk levels for each Greek
        delta_risk = self._classify_greek_risk('delta', abs(total_delta))
        gamma_risk = self._classify_greek_risk('gamma', abs(total_gamma))
        theta_risk = self._classify_greek_risk('theta', abs(total_theta))
        vega_risk = self._classify_greek_risk('vega', abs(total_vega))
        
        return {
            'total_delta': total_delta,
            'total_gamma': total_gamma,
            'total_theta': total_theta,
            'total_vega': total_vega,
            'delta_risk': delta_risk,
            'gamma_risk': gamma_risk,
            'theta_risk': theta_risk,
            'vega_risk': vega_risk
        }
    
    def _classify_greek_risk(self, greek_name: str, value: float) -> str:
        """Classify risk level for a Greek"""
        thresholds = self.risk_thresholds.get(greek_name, {})
        
        if value < thresholds.get('low', 0):
            return 'low'
        elif value < thresholds.get('medium', 0):
            return 'moderate'
        elif value < thresholds.get('high', 0):
            return 'high'
        else:
            return 'critical'
    
    def _analyze_positions(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze position sizing and concentration"""
        positions = portfolio_data.get('positions', [])
        total_value = portfolio_data.get('total_value', 0)
        
        if total_value == 0:
            return {'error': 'No portfolio value'}
        
        position_sizes = []
        concentration_risk = {}
        
        for position in positions:
            position_value = position.get('market_value', 0)
            size_pct = (position_value / total_value) * 100 if total_value > 0 else 0
            
            position_sizes.append({
                'symbol': position.get('symbol'),
                'value': position_value,
                'size_pct': size_pct
            })
            
            # Track by symbol
            symbol = position.get('symbol')
            if symbol:
                concentration_risk[symbol] = concentration_risk.get(symbol, 0) + size_pct
        
        # Find largest positions
        largest_positions = sorted(position_sizes, key=lambda x: x['size_pct'], reverse=True)[:5]
        
        # Calculate concentration metrics
        top_5_concentration = sum(p['size_pct'] for p in largest_positions)
        
        return {
            'position_count': len(positions),
            'largest_positions': largest_positions,
            'top_5_concentration': top_5_concentration,
            'concentration_risk': concentration_risk
        }
    
    def _calculate_risk_metrics(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate portfolio risk metrics"""
        total_value = portfolio_data.get('total_value', 0)
        unrealized_pnl = portfolio_data.get('unrealized_pnl', 0)
        
        # Calculate drawdown
        peak_value = portfolio_data.get('peak_value', total_value)
        current_drawdown = ((peak_value - total_value) / peak_value) * 100 if peak_value > 0 else 0
        
        # Calculate return
        initial_value = portfolio_data.get('initial_value', total_value)
        total_return = ((total_value - initial_value) / initial_value) * 100 if initial_value > 0 else 0
        
        return {
            'total_value': total_value,
            'unrealized_pnl': unrealized_pnl,
            'current_drawdown_pct': current_drawdown,
            'total_return_pct': total_return,
            'peak_value': peak_value
        }
    
    def _check_risk_limits(
        self,
        greeks: Dict[str, Any],
        positions: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check for risk limit violations"""
        violations = []
        
        # Check Delta limit
        total_delta = abs(greeks.get('total_delta', 0))
        if total_delta > self.max_portfolio_delta:
            violations.append({
                'type': 'delta_limit',
                'current': total_delta,
                'limit': self.max_portfolio_delta,
                'severity': 'high'
            })
        
        # Check position size limits
        for position in positions.get('largest_positions', []):
            if position['size_pct'] > self.max_position_size_pct * 100:
                violations.append({
                    'type': 'position_size',
                    'symbol': position['symbol'],
                    'current_pct': position['size_pct'],
                    'limit_pct': self.max_position_size_pct * 100,
                    'severity': 'medium'
                })
        
        # Check drawdown limit
        drawdown = metrics.get('current_drawdown_pct', 0)
        if drawdown > self.max_drawdown_pct * 100:
            violations.append({
                'type': 'drawdown',
                'current_pct': drawdown,
                'limit_pct': self.max_drawdown_pct * 100,
                'severity': 'critical'
            })
        
        return violations
    
    def _determine_risk_level(
        self,
        greeks: Dict[str, Any],
        positions: Dict[str, Any],
        violations: List[Dict[str, Any]]
    ) -> str:
        """Determine overall portfolio risk level"""
        # Check for critical violations
        if any(v['severity'] == 'critical' for v in violations):
            return 'critical'
        
        # Check Greeks risk
        greek_risks = [
            greeks.get('delta_risk'),
            greeks.get('gamma_risk'),
            greeks.get('vega_risk')
        ]
        
        if 'critical' in greek_risks or len(violations) > 2:
            return 'high'
        elif 'high' in greek_risks or len(violations) > 0:
            return 'moderate'
        else:
            return 'low'
    
    def _calculate_position_sizing(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate recommended position sizing using Kelly Criterion"""
        # Simplified Kelly Criterion
        # In production, would use more sophisticated calculations
        
        risk_level = analysis.get('risk_level', 'moderate')
        
        if risk_level == 'critical':
            max_position_pct = 2.0
            kelly_fraction = 0.1
        elif risk_level == 'high':
            max_position_pct = 5.0
            kelly_fraction = 0.25
        elif risk_level == 'moderate':
            max_position_pct = 10.0
            kelly_fraction = 0.5
        else:
            max_position_pct = 15.0
            kelly_fraction = 0.75
        
        return {
            'max_position_pct': max_position_pct,
            'kelly_fraction': kelly_fraction,
            'reasoning': f"Based on {risk_level} risk level"
        }
    
    def _generate_hedging_recommendations(self, greeks: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate hedging recommendations based on Greeks exposure"""
        recommendations = []
        
        total_delta = greeks.get('total_delta', 0)
        
        # Delta hedging
        if abs(total_delta) > 50:
            recommendations.append({
                'type': 'delta_hedge',
                'action': 'sell' if total_delta > 0 else 'buy',
                'quantity': abs(total_delta),
                'reasoning': f"Portfolio delta of {total_delta:.2f} exceeds threshold"
            })
        
        # Vega hedging
        total_vega = greeks.get('total_vega', 0)
        if abs(total_vega) > 50:
            recommendations.append({
                'type': 'vega_hedge',
                'action': 'reduce_vega_exposure',
                'reasoning': f"High vega exposure of {total_vega:.2f}"
            })
        
        return recommendations

