"""
Report Generation Agent - Creates natural language summaries and recommendations.
"""
from typing import Dict, Any, List
import logging
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ReportGenerationAgent(BaseAgent):
    """
    Agent responsible for:
    - Synthesizing insights from other agents
    - Creating natural language summaries
    - Generating actionable recommendations
    - Formatting reports per user preferences
    """
    
    def __init__(self):
        super().__init__(
            name="ReportGenerationAgent",
            role="Generate comprehensive reports and recommendations"
        )
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive report from all agent insights.
        
        Args:
            state: Current system state with all agent outputs
            
        Returns:
            Updated state with generated report
        """
        logger.info(f"{self.name}: Generating report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'report_type': state.get('report_type', 'daily'),
            'executive_summary': self._generate_executive_summary(state),
            'market_overview': self._generate_market_overview(state),
            'portfolio_analysis': self._generate_portfolio_analysis(state),
            'risk_assessment': self._generate_risk_assessment(state),
            'recommendations': self._generate_recommendations(state),
            'action_items': self._generate_action_items(state),
            'appendix': self._generate_appendix(state)
        }
        
        # Add to memory
        self.add_to_short_term_memory({
            'timestamp': datetime.now().isoformat(),
            'report_type': report['report_type'],
            'num_recommendations': len(report['recommendations'])
        })
        
        # Update state
        state['report'] = report
        
        logger.info(f"{self.name}: Report generation complete")
        return state
    
    def _generate_executive_summary(self, state: Dict[str, Any]) -> str:
        """Generate executive summary."""
        positions = state.get('positions', [])
        risk_analysis = state.get('risk_analysis', {})
        quant_analysis = state.get('quant_analysis', {})
        
        num_positions = len(positions)
        risk_score = risk_analysis.get('risk_score', 0)
        
        # Calculate portfolio metrics
        total_value = sum(abs(p.get('market_value', 0)) for p in positions)
        total_pnl = sum(p.get('pnl', 0) for p in positions)
        
        # Get average EV
        ev_calcs = quant_analysis.get('ev_calculations', {})
        avg_ev = sum(
            calc.get('expected_value', 0) for calc in ev_calcs.values()
        ) / len(ev_calcs) if ev_calcs else 0
        
        summary = f"""
**Portfolio Overview**
- Active Positions: {num_positions}
- Total Portfolio Value: ${total_value:,.2f}
- Total P&L: ${total_pnl:,.2f} ({(total_pnl/total_value*100) if total_value > 0 else 0:.1f}%)
- Average Expected Value: ${avg_ev:.2f}
- Risk Score: {risk_score:.1f}/100 ({self._risk_level(risk_score)})

**Key Highlights**
{self._generate_key_highlights(state)}
        """.strip()
        
        return summary
    
    def _generate_market_overview(self, state: Dict[str, Any]) -> str:
        """Generate market overview section."""
        market_intel = state.get('market_intelligence', {})
        
        iv_changes = market_intel.get('iv_changes', {})
        volume_anomalies = market_intel.get('volume_anomalies', {})
        unusual_activity = market_intel.get('unusual_activity', {})
        
        overview = "**Market Conditions**\n\n"
        
        if iv_changes:
            overview += "**Significant IV Changes:**\n"
            for symbol, data in list(iv_changes.items())[:5]:
                overview += f"- {symbol}: {data['change_pct']:+.1f}% (Current: {data['current_iv']:.1f}%)\n"
            overview += "\n"
        
        if volume_anomalies:
            overview += "**Volume Anomalies:**\n"
            for symbol, data in list(volume_anomalies.items())[:5]:
                overview += f"- {symbol}: {data['ratio']:.1f}x average volume\n"
            overview += "\n"
        
        if unusual_activity:
            overview += "**Unusual Options Activity:**\n"
            for symbol, data in list(unusual_activity.items())[:5]:
                overview += f"- {symbol}: Put/Call ratio {data['put_call_ratio']:.2f} ({data['signal']})\n"
        
        return overview.strip()
    
    def _generate_portfolio_analysis(self, state: Dict[str, Any]) -> str:
        """Generate portfolio analysis section."""
        portfolio_greeks = state.get('portfolio_greeks', {})
        quant_analysis = state.get('quant_analysis', {})
        
        analysis = "**Portfolio Greeks**\n\n"
        
        for greek, value in portfolio_greeks.items():
            analysis += f"- {greek.capitalize()}: {value:,.2f}\n"
        
        analysis += "\n**Expected Value Analysis**\n\n"
        
        ev_calcs = quant_analysis.get('ev_calculations', {})
        for symbol, ev_data in ev_calcs.items():
            if 'error' in ev_data:
                continue
            
            analysis += f"**{symbol}**\n"
            analysis += f"- Expected Value: ${ev_data['expected_value']:.2f}\n"
            analysis += f"- Expected Return: {ev_data['expected_return_pct']:.1f}%\n"
            analysis += f"- Probability of Profit: {ev_data['probability_profit']*100:.1f}%\n"
            analysis += f"- Rating: {ev_data['rating'].upper()}\n\n"
        
        return analysis.strip()
    
    def _generate_risk_assessment(self, state: Dict[str, Any]) -> str:
        """Generate risk assessment section."""
        risk_analysis = state.get('risk_analysis', {})
        
        assessment = "**Risk Assessment**\n\n"
        
        # Risk score
        risk_score = risk_analysis.get('risk_score', 0)
        assessment += f"Overall Risk Score: {risk_score:.1f}/100 ({self._risk_level(risk_score)})\n\n"
        
        # Alerts
        alerts = risk_analysis.get('alerts', [])
        if alerts:
            assessment += "**Active Alerts:**\n"
            for alert in alerts:
                assessment += f"- [{alert['level'].upper()}] {alert['message']}\n"
            assessment += "\n"
        
        # Concentration risks
        concentration = risk_analysis.get('concentration_risk', {})
        pos_conc = concentration.get('position_concentration', {})
        if pos_conc:
            assessment += "**Position Concentration:**\n"
            for symbol, data in pos_conc.items():
                assessment += f"- {symbol}: {data['concentration_pct']:.1f}% of portfolio\n"
            assessment += "\n"
        
        # Tail risks
        tail_risks = risk_analysis.get('tail_risks', [])
        if tail_risks:
            assessment += "**Tail Risks:**\n"
            for risk in tail_risks[:5]:
                assessment += f"- [{risk['severity'].upper()}] {risk['description']}\n"
        
        return assessment.strip()
    
    def _generate_recommendations(self, state: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # From risk analysis
        risk_analysis = state.get('risk_analysis', {})
        hedge_suggestions = risk_analysis.get('hedge_suggestions', [])
        
        for hedge in hedge_suggestions:
            recommendations.append({
                'type': 'risk_management',
                'priority': hedge['priority'],
                'action': hedge['action'],
                'description': hedge['description']
            })
        
        # From quant analysis
        quant_analysis = state.get('quant_analysis', {})
        optimal_actions = quant_analysis.get('optimal_actions', [])
        
        for action in optimal_actions:
            recommendations.append({
                'type': 'position_management',
                'priority': action['priority'],
                'action': f"{action['action']} {action['symbol']}",
                'description': f"{action['reason']} - {action['expected_benefit']}"
            })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return recommendations
    
    def _generate_action_items(self, state: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate specific action items for today."""
        action_items = []
        
        recommendations = self._generate_recommendations(state)
        
        # Convert high-priority recommendations to action items
        for rec in recommendations:
            if rec['priority'] == 'high':
                action_items.append({
                    'action': rec['action'],
                    'reason': rec['description'],
                    'deadline': 'Today'
                })
        
        # Add monitoring items
        risk_analysis = state.get('risk_analysis', {})
        tail_risks = risk_analysis.get('tail_risks', [])
        
        for risk in tail_risks:
            if risk.get('severity') == 'high':
                action_items.append({
                    'action': f"Monitor {risk.get('symbol', 'position')}",
                    'reason': risk['description'],
                    'deadline': 'Ongoing'
                })
        
        return action_items[:10]  # Limit to top 10
    
    def _generate_appendix(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appendix with detailed data."""
        return {
            'scenario_analysis': state.get('quant_analysis', {}).get('scenario_analysis', {}),
            'probability_distributions': state.get('quant_analysis', {}).get('probability_analysis', {}),
            'greeks_breakdown': state.get('risk_analysis', {}).get('greeks_analysis', {})
        }
    
    def _generate_key_highlights(self, state: Dict[str, Any]) -> str:
        """Generate key highlights."""
        highlights = []
        
        # Best performing position
        positions = state.get('positions', [])
        if positions:
            best_pos = max(positions, key=lambda p: p.get('pnl_pct', -999))
            highlights.append(
                f"- Best performer: {best_pos.get('symbol')} ({best_pos.get('pnl_pct', 0):+.1f}%)"
            )
        
        # Highest risk position
        risk_analysis = state.get('risk_analysis', {})
        alerts = risk_analysis.get('alerts', [])
        if alerts:
            critical_alerts = [a for a in alerts if a['level'] == 'critical']
            if critical_alerts:
                highlights.append(f"- {len(critical_alerts)} critical risk alert(s)")
        
        # Best EV opportunity
        quant_analysis = state.get('quant_analysis', {})
        ev_calcs = quant_analysis.get('ev_calculations', {})
        if ev_calcs:
            best_ev = max(
                ((k, v) for k, v in ev_calcs.items() if 'expected_value' in v),
                key=lambda x: x[1]['expected_value'],
                default=None
            )
            if best_ev:
                highlights.append(
                    f"- Best EV: {best_ev[0]} (${best_ev[1]['expected_value']:.2f})"
                )
        
        return "\n".join(highlights) if highlights else "- No significant highlights"
    
    def _risk_level(self, score: float) -> str:
        """Convert risk score to level."""
        if score >= 75:
            return "High Risk"
        elif score >= 50:
            return "Moderate Risk"
        elif score >= 25:
            return "Low-Moderate Risk"
        else:
            return "Low Risk"

