"""
Tests for Multi-Agent System
"""
import pytest
from src.agents.coordinator import CoordinatorAgent, AnalysisState
from src.agents.market_intelligence import MarketIntelligenceAgent
from src.agents.risk_analysis import RiskAnalysisAgent
from src.agents.quant_analysis import QuantAnalysisAgent
from src.agents.report_generation import ReportGenerationAgent


class TestMarketIntelligenceAgent:
    """Test Market Intelligence Agent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = MarketIntelligenceAgent()
        
        self.state = {
            'positions': [
                {
                    'symbol': 'AAPL',
                    'market_value': 1000.0
                }
            ],
            'market_data': {
                'AAPL': {
                    'iv': 0.35,
                    'historical_iv': 0.30,
                    'volume': 2000000,
                    'avg_volume': 1000000,
                    'put_call_ratio': 1.2,
                    'underlying_price': 150.0
                }
            }
        }
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        assert self.agent.name == "MarketIntelligenceAgent"
        assert self.agent.role is not None
    
    def test_process_state(self):
        """Test state processing."""
        result = self.agent.process(self.state)
        
        assert 'market_intelligence' in result
        assert 'iv_changes' in result['market_intelligence']
        assert 'volume_anomalies' in result['market_intelligence']
    
    def test_iv_change_detection(self):
        """Test IV change detection."""
        result = self.agent.process(self.state)
        
        iv_changes = result['market_intelligence']['iv_changes']
        assert 'AAPL' in iv_changes
        assert iv_changes['AAPL']['change_pct'] > 10  # Significant change


class TestRiskAnalysisAgent:
    """Test Risk Analysis Agent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = RiskAnalysisAgent()
        
        self.state = {
            'positions': [
                {
                    'symbol': 'AAPL',
                    'market_value': 5000.0,
                    'sector': 'Technology'
                },
                {
                    'symbol': 'MSFT',
                    'market_value': 3000.0,
                    'sector': 'Technology'
                }
            ],
            'market_data': {
                'AAPL': {'days_to_earnings': 5},
                'MSFT': {'days_to_earnings': 999}
            },
            'portfolio_greeks': {
                'delta': 500,
                'gamma': 200,
                'theta': -100,
                'vega': 1500,
                'rho': 50
            }
        }
    
    def test_risk_score_calculation(self):
        """Test risk score calculation."""
        result = self.agent.process(self.state)
        
        assert 'risk_analysis' in result
        assert 'risk_score' in result['risk_analysis']
        assert 0 <= result['risk_analysis']['risk_score'] <= 100
    
    def test_concentration_analysis(self):
        """Test concentration risk analysis."""
        result = self.agent.process(self.state)
        
        concentration = result['risk_analysis']['concentration_risk']
        assert 'sector_concentration' in concentration
        # Technology sector should be concentrated
        assert 'Technology' in concentration['sector_concentration']
    
    def test_tail_risk_identification(self):
        """Test tail risk identification."""
        result = self.agent.process(self.state)
        
        tail_risks = result['risk_analysis']['tail_risks']
        # Should identify earnings event for AAPL
        earnings_risks = [r for r in tail_risks if r['type'] == 'earnings_event']
        assert len(earnings_risks) > 0


class TestQuantAnalysisAgent:
    """Test Quant Analysis Agent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = QuantAnalysisAgent()
        
        self.state = {
            'positions': [
                {
                    'symbol': 'AAPL',
                    'total_premium': 500.0,
                    'days_to_expiry': 30,
                    'pnl_pct': 25,
                    'legs': [
                        {
                            'option_type': 'call',
                            'strike': 150.0,
                            'quantity': 1,
                            'is_short': False,
                            'entry_price': 5.0,
                            'time_to_expiry': 0.08,
                            'multiplier': 100
                        }
                    ]
                }
            ],
            'market_data': {
                'AAPL': {
                    'underlying_price': 155.0,
                    'iv': 0.30,
                    'time_to_expiry': 0.08
                }
            }
        }
    
    def test_ev_calculations(self):
        """Test EV calculations."""
        result = self.agent.process(self.state)
        
        assert 'quant_analysis' in result
        assert 'ev_calculations' in result['quant_analysis']
        assert 'AAPL' in result['quant_analysis']['ev_calculations']
    
    def test_scenario_analysis(self):
        """Test scenario analysis."""
        result = self.agent.process(self.state)
        
        scenarios = result['quant_analysis']['scenario_analysis']
        assert 'bull_case' in scenarios
        assert 'bear_case' in scenarios
        assert 'neutral_case' in scenarios


class TestReportGenerationAgent:
    """Test Report Generation Agent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = ReportGenerationAgent()
        
        self.state = {
            'positions': [
                {'symbol': 'AAPL', 'market_value': 1000.0, 'pnl': 100.0}
            ],
            'portfolio_greeks': {
                'delta': 50,
                'gamma': 10,
                'theta': -5,
                'vega': 100,
                'rho': 5
            },
            'risk_analysis': {
                'risk_score': 45.0,
                'alerts': [],
                'hedge_suggestions': []
            },
            'quant_analysis': {
                'ev_calculations': {
                    'AAPL': {
                        'expected_value': 50.0,
                        'expected_return_pct': 10.0,
                        'probability_profit': 0.65,
                        'rating': 'good'
                    }
                },
                'optimal_actions': []
            },
            'market_intelligence': {
                'iv_changes': {},
                'volume_anomalies': {}
            }
        }
    
    def test_report_generation(self):
        """Test report generation."""
        result = self.agent.process(self.state)
        
        assert 'report' in result
        report = result['report']
        
        assert 'executive_summary' in report
        assert 'market_overview' in report
        assert 'portfolio_analysis' in report
        assert 'risk_assessment' in report
        assert 'recommendations' in report
    
    def test_executive_summary_content(self):
        """Test executive summary contains key metrics."""
        result = self.agent.process(self.state)
        
        summary = result['report']['executive_summary']
        assert 'Active Positions' in summary
        assert 'Risk Score' in summary


class TestCoordinatorAgent:
    """Test Coordinator Agent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.coordinator = CoordinatorAgent()
        
        self.positions = [
            {
                'symbol': 'AAPL',
                'market_value': 1000.0,
                'legs': [
                    {
                        'option_type': 'call',
                        'strike': 150.0,
                        'quantity': 1,
                        'time_to_expiry': 0.08
                    }
                ]
            }
        ]
        
        self.market_data = {
            'AAPL': {
                'underlying_price': 155.0,
                'iv': 0.30,
                'time_to_expiry': 0.08
            }
        }
        
        self.portfolio_greeks = {
            'delta': 50,
            'gamma': 10,
            'theta': -5,
            'vega': 100,
            'rho': 5
        }
    
    def test_workflow_execution(self):
        """Test complete workflow execution."""
        result = self.coordinator.run_analysis(
            positions=self.positions,
            market_data=self.market_data,
            portfolio_greeks=self.portfolio_greeks
        )
        
        assert result['workflow_status'] == 'completed'
        assert 'market_intelligence' in result
        assert 'risk_analysis' in result
        assert 'quant_analysis' in result
        assert 'report' in result
    
    def test_scheduled_analysis(self):
        """Test scheduled analysis."""
        result = self.coordinator.run_scheduled_analysis(
            schedule_type='end_of_day',
            positions=self.positions,
            market_data=self.market_data,
            portfolio_greeks=self.portfolio_greeks
        )
        
        assert result['workflow_status'] == 'completed'
        assert result['report']['report_type'] == 'daily'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

