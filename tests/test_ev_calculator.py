"""
Tests for EV Calculator
"""
import pytest
import numpy as np
from src.analytics.ev_calculator import EVCalculator, ev_long_call, ev_long_put


class TestEVCalculator:
    """Test EV Calculator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = EVCalculator()
        
        self.market_data = {
            'underlying_price': 100.0,
            'iv': 0.30,
            'time_to_expiry': 0.25,  # 3 months
            'risk_free_rate': 0.05
        }
        
        self.long_call_position = {
            'symbol': 'AAPL',
            'strategy_type': 'long_call',
            'total_premium': 500.0,
            'legs': [
                {
                    'option_type': 'call',
                    'strike': 105.0,
                    'quantity': 1,
                    'is_short': False,
                    'entry_price': 5.0,
                    'multiplier': 100
                }
            ]
        }
    
    def test_calculate_ev_long_call(self):
        """Test EV calculation for long call."""
        result = self.calculator.calculate_ev(
            self.long_call_position,
            self.market_data
        )
        
        assert result is not None
        assert hasattr(result, 'expected_value')
        assert hasattr(result, 'probability_profit')
        assert 0 <= result.probability_profit <= 1
        assert len(result.price_range) > 0
        assert len(result.expected_payoff_distribution) > 0
    
    def test_probability_distributions_sum_to_one(self):
        """Test that probability distributions are normalized."""
        price_range = np.linspace(80, 120, 100)
        
        bs_dist = self.calculator._black_scholes_distribution(
            self.market_data,
            price_range
        )
        
        assert np.isclose(bs_dist.sum(), 1.0, atol=1e-6)
    
    def test_payoff_calculation(self):
        """Test payoff calculation for different strategies."""
        # Long call payoff
        payoff_100 = self.calculator._calculate_payoff(
            self.long_call_position,
            100.0
        )
        assert payoff_100 == 0  # OTM
        
        payoff_110 = self.calculator._calculate_payoff(
            self.long_call_position,
            110.0
        )
        assert payoff_110 == 500.0  # ITM by $5, 100 shares
    
    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        result = self.calculator.calculate_ev(
            self.long_call_position,
            self.market_data
        )
        
        lower, upper = result.confidence_interval
        assert lower < upper
        assert isinstance(lower, (int, float))
        assert isinstance(upper, (int, float))
    
    def test_method_breakdown(self):
        """Test that all three methods contribute to EV."""
        result = self.calculator.calculate_ev(
            self.long_call_position,
            self.market_data
        )
        
        assert 'black_scholes' in result.method_breakdown
        assert 'risk_neutral_density' in result.method_breakdown
        assert 'monte_carlo' in result.method_breakdown
    
    def test_vertical_spread(self):
        """Test EV for vertical spread."""
        vertical_spread = {
            'symbol': 'AAPL',
            'strategy_type': 'bull_call_spread',
            'total_premium': 200.0,
            'legs': [
                {
                    'option_type': 'call',
                    'strike': 100.0,
                    'quantity': 1,
                    'is_short': False,
                    'entry_price': 5.0,
                    'multiplier': 100
                },
                {
                    'option_type': 'call',
                    'strike': 105.0,
                    'quantity': 1,
                    'is_short': True,
                    'entry_price': 3.0,
                    'multiplier': 100
                }
            ]
        }
        
        result = self.calculator.calculate_ev(
            vertical_spread,
            self.market_data
        )
        
        assert result is not None
        # Max profit should be limited to spread width minus premium
        max_payoff = max(result.expected_payoff_distribution)
        assert max_payoff <= 500.0  # $5 spread * 100 shares


class TestStrategyEVFunctions:
    """Test strategy-specific EV functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.prices = np.linspace(80, 120, 100)
        # Simple uniform distribution for testing
        self.prob_dist = np.ones(100) / 100
    
    def test_ev_long_call(self):
        """Test long call EV calculation."""
        ev = ev_long_call(
            strike=100.0,
            premium=5.0,
            prob_dist=self.prob_dist,
            prices=self.prices
        )
        
        assert isinstance(ev, (int, float))
    
    def test_ev_long_put(self):
        """Test long put EV calculation."""
        ev = ev_long_put(
            strike=100.0,
            premium=5.0,
            prob_dist=self.prob_dist,
            prices=self.prices
        )
        
        assert isinstance(ev, (int, float))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

