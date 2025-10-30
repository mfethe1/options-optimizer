"""
Greeks Calculator
Calculates Delta, Gamma, Theta, Vega, and Rho for options positions
"""
import numpy as np
from scipy.stats import norm
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GreeksResult:
    """Greeks calculation result."""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    underlying_price: float
    strike: float
    time_to_expiry: float
    iv: float


class GreeksCalculator:
    """
    Calculate option Greeks using Black-Scholes model.
    
    Greeks:
    - Delta: Price sensitivity to $1 stock move
    - Gamma: Rate of delta change
    - Theta: Time decay
    - Vega: Sensitivity to IV changes
    - Rho: Sensitivity to interest rate changes
    """
    
    def __init__(self):
        """Initialize Greeks Calculator."""
        pass
    
    def calculate_greeks(
        self,
        option_type: str,
        underlying_price: float,
        strike: float,
        time_to_expiry: float,
        iv: float,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0
    ) -> GreeksResult:
        """
        Calculate all Greeks for an option.
        
        Args:
            option_type: 'call' or 'put'
            underlying_price: Current underlying price
            strike: Strike price
            time_to_expiry: Time to expiration in years
            iv: Implied volatility (annualized)
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield
            
        Returns:
            GreeksResult with all Greeks
        """
        logger.debug(f"Calculating Greeks for {option_type} option")
        
        # Calculate d1 and d2
        d1, d2 = self._calculate_d1_d2(
            underlying_price, strike, time_to_expiry,
            iv, risk_free_rate, dividend_yield
        )
        
        # Calculate Greeks
        delta = self._calculate_delta(option_type, d1, dividend_yield, time_to_expiry)
        gamma = self._calculate_gamma(underlying_price, d1, iv, time_to_expiry, dividend_yield)
        theta = self._calculate_theta(
            option_type, underlying_price, strike, d1, d2,
            iv, time_to_expiry, risk_free_rate, dividend_yield
        )
        vega = self._calculate_vega(underlying_price, d1, time_to_expiry, dividend_yield)
        rho = self._calculate_rho(option_type, strike, d2, time_to_expiry, risk_free_rate)
        
        return GreeksResult(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            underlying_price=underlying_price,
            strike=strike,
            time_to_expiry=time_to_expiry,
            iv=iv
        )
    
    def calculate_portfolio_greeks(
        self,
        positions: list,
        market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate aggregate Greeks for entire portfolio.
        
        Args:
            positions: List of positions
            market_data: Market data for all symbols
            
        Returns:
            Dictionary with total Greeks
        """
        total_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
        
        for position in positions:
            symbol = position['symbol']
            if symbol not in market_data:
                logger.warning(f"No market data for {symbol}")
                continue
            
            for leg in position.get('legs', []):
                greeks = self.calculate_greeks(
                    option_type=leg['option_type'],
                    underlying_price=market_data[symbol]['underlying_price'],
                    strike=leg['strike'],
                    time_to_expiry=leg['time_to_expiry'],
                    iv=market_data[symbol]['iv'],
                    risk_free_rate=market_data[symbol].get('risk_free_rate', 0.05)
                )
                
                # Adjust for quantity and short positions
                multiplier = leg['quantity'] * leg.get('multiplier', 100)
                if leg.get('is_short', False):
                    multiplier = -multiplier
                
                total_greeks['delta'] += greeks.delta * multiplier
                total_greeks['gamma'] += greeks.gamma * multiplier
                total_greeks['theta'] += greeks.theta * multiplier
                total_greeks['vega'] += greeks.vega * multiplier
                total_greeks['rho'] += greeks.rho * multiplier
        
        return total_greeks
    
    def _calculate_d1_d2(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        r: float,
        q: float
    ) -> tuple:
        """Calculate d1 and d2 for Black-Scholes."""
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    def _calculate_delta(
        self,
        option_type: str,
        d1: float,
        q: float,
        T: float
    ) -> float:
        """
        Calculate Delta.
        
        Delta measures the rate of change of option price with respect to
        changes in the underlying asset's price.
        """
        if option_type == 'call':
            return np.exp(-q * T) * norm.cdf(d1)
        else:  # put
            return np.exp(-q * T) * (norm.cdf(d1) - 1)
    
    def _calculate_gamma(
        self,
        S: float,
        d1: float,
        sigma: float,
        T: float,
        q: float
    ) -> float:
        """
        Calculate Gamma.
        
        Gamma measures the rate of change in delta with respect to changes
        in the underlying price.
        """
        return (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
    
    def _calculate_theta(
        self,
        option_type: str,
        S: float,
        K: float,
        d1: float,
        d2: float,
        sigma: float,
        T: float,
        r: float,
        q: float
    ) -> float:
        """
        Calculate Theta.
        
        Theta measures the rate of decline in the value of an option due to
        the passage of time (time decay).
        
        Returns theta per day (divide by 365).
        """
        term1 = -(S * norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))
        
        if option_type == 'call':
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            term3 = q * S * np.exp(-q * T) * norm.cdf(d1)
            theta = term1 + term2 + term3
        else:  # put
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            term3 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
            theta = term1 + term2 + term3
        
        # Convert to per-day theta
        return theta / 365.0
    
    def _calculate_vega(
        self,
        S: float,
        d1: float,
        T: float,
        q: float
    ) -> float:
        """
        Calculate Vega.
        
        Vega measures sensitivity to volatility.
        Returns vega per 1% change in IV.
        """
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        
        # Convert to per 1% change
        return vega / 100.0
    
    def _calculate_rho(
        self,
        option_type: str,
        K: float,
        d2: float,
        T: float,
        r: float
    ) -> float:
        """
        Calculate Rho.
        
        Rho measures sensitivity to interest rate changes.
        Returns rho per 1% change in interest rate.
        """
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        
        # Convert to per 1% change
        return rho / 100.0


def calculate_implied_volatility(
    option_price: float,
    option_type: str,
    underlying_price: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
    max_iterations: int = 100,
    tolerance: float = 1e-5
) -> Optional[float]:
    """
    Calculate implied volatility using Newton-Raphson method.
    
    Args:
        option_price: Market price of option
        option_type: 'call' or 'put'
        underlying_price: Current underlying price
        strike: Strike price
        time_to_expiry: Time to expiration in years
        risk_free_rate: Risk-free interest rate
        dividend_yield: Dividend yield
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        
    Returns:
        Implied volatility or None if not converged
    """
    from .black_scholes import black_scholes_price
    
    # Initial guess
    iv = 0.3
    
    for i in range(max_iterations):
        # Calculate option price with current IV
        calc_price = black_scholes_price(
            option_type, underlying_price, strike,
            time_to_expiry, iv, risk_free_rate, dividend_yield
        )
        
        # Calculate vega
        calculator = GreeksCalculator()
        greeks = calculator.calculate_greeks(
            option_type, underlying_price, strike,
            time_to_expiry, iv, risk_free_rate, dividend_yield
        )
        
        # Newton-Raphson update
        diff = calc_price - option_price
        
        if abs(diff) < tolerance:
            return iv
        
        if greeks.vega == 0:
            return None
        
        iv = iv - diff / (greeks.vega * 100)  # vega is per 1%
        
        # Keep IV positive
        if iv <= 0:
            iv = 0.01
    
    logger.warning(f"IV calculation did not converge after {max_iterations} iterations")
    return None

