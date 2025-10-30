"""
Expected Value (EV) Calculator
Implements three probability methods: Black-Scholes, Risk-Neutral Density, Monte Carlo
"""
import numpy as np
from scipy.stats import norm
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EVResult:
    """Expected Value calculation result."""
    expected_value: float
    expected_return_pct: float
    probability_profit: float
    expected_payoff_distribution: np.ndarray
    price_range: np.ndarray
    confidence_interval: Tuple[float, float]
    method_breakdown: Dict[str, float]


class EVCalculator:
    """
    Calculate Expected Value using multiple probability methods.
    
    Methods:
    1. Black-Scholes Distribution (30% weight)
    2. Risk-Neutral Density (40% weight)
    3. Monte Carlo Simulation (30% weight)
    """
    
    def __init__(self, weights: List[float] = None):
        """
        Initialize EV Calculator.
        
        Args:
            weights: Weights for [BS, RND, MC] methods. Default: [0.3, 0.4, 0.3]
        """
        self.weights = weights or [0.3, 0.4, 0.3]
        assert abs(sum(self.weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
    
    def calculate_ev(
        self,
        position: Dict[str, Any],
        market_data: Dict[str, Any],
        num_price_points: int = 1000
    ) -> EVResult:
        """
        Calculate expected value for a position.
        
        Args:
            position: Position details (legs, strikes, premiums, etc.)
            market_data: Market data (underlying price, IV, rates, etc.)
            num_price_points: Number of price points for distribution
            
        Returns:
            EVResult with complete EV analysis
        """
        logger.info(f"Calculating EV for position: {position.get('symbol')}")
        
        # Get price range
        price_range = self._get_price_range(market_data, num_price_points)
        
        # Calculate probability distributions
        bs_dist = self._black_scholes_distribution(market_data, price_range)
        rnd_dist = self._risk_neutral_density(market_data, price_range)
        mc_dist = self._monte_carlo_distribution(market_data, price_range)
        
        # Weighted combination
        combined_dist = (
            self.weights[0] * bs_dist +
            self.weights[1] * rnd_dist +
            self.weights[2] * mc_dist
        )
        
        # Normalize
        combined_dist = combined_dist / combined_dist.sum()
        
        # Calculate payoffs at each price point
        payoffs = np.array([
            self._calculate_payoff(position, price)
            for price in price_range
        ])
        
        # Calculate EV
        premium_paid = position.get('total_premium', 0)
        expected_payoff = np.sum(combined_dist * payoffs)
        net_ev = expected_payoff - premium_paid
        
        # Calculate probability of profit
        prob_profit = np.sum(combined_dist[payoffs > premium_paid])
        
        # Calculate confidence interval (95%)
        sorted_indices = np.argsort(payoffs)
        cumulative_prob = np.cumsum(combined_dist[sorted_indices])
        lower_idx = sorted_indices[np.searchsorted(cumulative_prob, 0.025)]
        upper_idx = sorted_indices[np.searchsorted(cumulative_prob, 0.975)]
        confidence_interval = (payoffs[lower_idx], payoffs[upper_idx])
        
        # Method breakdown
        method_breakdown = {
            'black_scholes': np.sum(bs_dist * payoffs) - premium_paid,
            'risk_neutral_density': np.sum(rnd_dist * payoffs) - premium_paid,
            'monte_carlo': np.sum(mc_dist * payoffs) - premium_paid
        }
        
        return EVResult(
            expected_value=net_ev,
            expected_return_pct=(net_ev / premium_paid * 100) if premium_paid > 0 else 0,
            probability_profit=prob_profit,
            expected_payoff_distribution=payoffs,
            price_range=price_range,
            confidence_interval=confidence_interval,
            method_breakdown=method_breakdown
        )
    
    def _get_price_range(
        self,
        market_data: Dict[str, Any],
        num_points: int
    ) -> np.ndarray:
        """Generate price range for analysis."""
        underlying_price = market_data['underlying_price']
        iv = market_data['iv']
        time_to_expiry = market_data['time_to_expiry']
        
        # 4 standard deviations
        std_dev = underlying_price * iv * np.sqrt(time_to_expiry)
        lower_bound = max(0.01, underlying_price - 4 * std_dev)
        upper_bound = underlying_price + 4 * std_dev
        
        return np.linspace(lower_bound, upper_bound, num_points)
    
    def _black_scholes_distribution(
        self,
        market_data: Dict[str, Any],
        price_range: np.ndarray
    ) -> np.ndarray:
        """Calculate probability distribution using Black-Scholes."""
        S = market_data['underlying_price']
        sigma = market_data['iv']
        T = market_data['time_to_expiry']
        r = market_data.get('risk_free_rate', 0.05)
        
        # Log-normal distribution
        mu = np.log(S) + (r - 0.5 * sigma**2) * T
        std = sigma * np.sqrt(T)
        
        # PDF at each price point
        log_prices = np.log(price_range)
        pdf = norm.pdf(log_prices, mu, std) / price_range
        
        # Normalize
        return pdf / pdf.sum()
    
    def _risk_neutral_density(
        self,
        market_data: Dict[str, Any],
        price_range: np.ndarray
    ) -> np.ndarray:
        """
        Calculate risk-neutral density from option prices.
        Uses Breeden-Litzenberger formula.
        """
        # For now, use Black-Scholes as approximation
        # In production, would extract from actual option chain
        return self._black_scholes_distribution(market_data, price_range)
    
    def _monte_carlo_distribution(
        self,
        market_data: Dict[str, Any],
        price_range: np.ndarray,
        num_simulations: int = 10000
    ) -> np.ndarray:
        """Calculate probability distribution using Monte Carlo simulation."""
        S = market_data['underlying_price']
        sigma = market_data['iv']
        T = market_data['time_to_expiry']
        r = market_data.get('risk_free_rate', 0.05)
        
        # Simulate price paths
        dt = T
        random_shocks = np.random.standard_normal(num_simulations)
        final_prices = S * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shocks
        )
        
        # Create histogram matching price_range
        hist, _ = np.histogram(final_prices, bins=len(price_range), 
                              range=(price_range[0], price_range[-1]))
        
        # Normalize
        return hist / hist.sum()
    
    def _calculate_payoff(
        self,
        position: Dict[str, Any],
        underlying_price: float
    ) -> float:
        """
        Calculate strategy payoff at given underlying price.
        
        Args:
            position: Position with legs
            underlying_price: Price to calculate payoff at
            
        Returns:
            Total payoff at this price
        """
        total_payoff = 0.0
        
        for leg in position.get('legs', []):
            option_type = leg['option_type']  # 'call' or 'put'
            strike = leg['strike']
            quantity = leg['quantity']
            multiplier = leg.get('multiplier', 100)
            is_short = leg.get('is_short', False)
            
            # Calculate intrinsic value
            if option_type == 'call':
                intrinsic = max(0, underlying_price - strike)
            else:  # put
                intrinsic = max(0, strike - underlying_price)
            
            # Calculate payoff
            payoff = intrinsic * quantity * multiplier
            
            # Adjust for short positions
            if is_short:
                payoff = -payoff
            
            total_payoff += payoff
        
        # Round to cents to avoid floating epsilon artifacts (e.g., 500.00000000000045)
        return round(float(total_payoff), 6)


# Strategy-specific EV calculators

def ev_long_call(
    strike: float,
    premium: float,
    prob_dist: np.ndarray,
    prices: np.ndarray
) -> float:
    """Calculate EV for long call."""
    payoffs = np.maximum(prices - strike, 0) - premium
    return np.sum(prob_dist * payoffs)


def ev_long_put(
    strike: float,
    premium: float,
    prob_dist: np.ndarray,
    prices: np.ndarray
) -> float:
    """Calculate EV for long put."""
    payoffs = np.maximum(strike - prices, 0) - premium
    return np.sum(prob_dist * payoffs)


def ev_vertical_spread(
    long_strike: float,
    short_strike: float,
    net_premium: float,
    prob_dist: np.ndarray,
    prices: np.ndarray,
    is_call: bool = True
) -> float:
    """Calculate EV for vertical spread."""
    if is_call:
        long_payoffs = np.maximum(prices - long_strike, 0)
        short_payoffs = np.maximum(prices - short_strike, 0)
    else:
        long_payoffs = np.maximum(long_strike - prices, 0)
        short_payoffs = np.maximum(short_strike - prices, 0)
    
    net_payoffs = long_payoffs - short_payoffs - net_premium
    return np.sum(prob_dist * net_payoffs)


def ev_iron_condor(
    strikes: List[float],  # [put_long, put_short, call_short, call_long]
    premiums: List[float],  # [put_long, put_short, call_short, call_long]
    prob_dist: np.ndarray,
    prices: np.ndarray
) -> float:
    """Calculate EV for iron condor."""
    put_spread_payoff = np.clip(strikes[1] - prices, 0, strikes[1] - strikes[0])
    call_spread_payoff = np.clip(prices - strikes[2], 0, strikes[3] - strikes[2])
    
    total_payoff = put_spread_payoff + call_spread_payoff
    net_premium = premiums[1] + premiums[2] - premiums[0] - premiums[3]
    
    net_payoffs = net_premium - total_payoff
    return np.sum(prob_dist * net_payoffs)

