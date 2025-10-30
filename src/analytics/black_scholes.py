"""
Black-Scholes Option Pricing Model
"""
import numpy as np
from scipy.stats import norm
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def black_scholes_price(
    option_type: str,
    underlying_price: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0
) -> float:
    """
    Calculate option price using Black-Scholes model.
    
    Args:
        option_type: 'call' or 'put'
        underlying_price: Current price of underlying
        strike: Strike price
        time_to_expiry: Time to expiration in years
        volatility: Implied volatility (annualized)
        risk_free_rate: Risk-free interest rate
        dividend_yield: Dividend yield
        
    Returns:
        Option price
    """
    S = underlying_price
    K = strike
    T = time_to_expiry
    sigma = volatility
    r = risk_free_rate
    q = dividend_yield
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = (S * np.exp(-q * T) * norm.cdf(d1) - 
                K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == 'put':
        price = (K * np.exp(-r * T) * norm.cdf(-d2) - 
                S * np.exp(-q * T) * norm.cdf(-d1))
    else:
        raise ValueError(f"Invalid option_type: {option_type}")
    
    return price


def black_scholes_delta(
    option_type: str,
    underlying_price: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0
) -> float:
    """Calculate Black-Scholes delta."""
    S = underlying_price
    K = strike
    T = time_to_expiry
    sigma = volatility
    r = risk_free_rate
    q = dividend_yield
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    if option_type == 'call':
        return np.exp(-q * T) * norm.cdf(d1)
    else:  # put
        return np.exp(-q * T) * (norm.cdf(d1) - 1)


def calculate_probability_itm(
    option_type: str,
    underlying_price: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0
) -> float:
    """
    Calculate probability of option finishing in-the-money.
    
    Args:
        option_type: 'call' or 'put'
        underlying_price: Current price of underlying
        strike: Strike price
        time_to_expiry: Time to expiration in years
        volatility: Implied volatility
        risk_free_rate: Risk-free interest rate
        dividend_yield: Dividend yield
        
    Returns:
        Probability (0 to 1)
    """
    S = underlying_price
    K = strike
    T = time_to_expiry
    sigma = volatility
    r = risk_free_rate
    q = dividend_yield
    
    # Calculate d2 (risk-neutral probability)
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    if option_type == 'call':
        return norm.cdf(d2)
    else:  # put
        return norm.cdf(-d2)


def calculate_breakeven(
    option_type: str,
    strike: float,
    premium: float,
    is_short: bool = False
) -> float:
    """
    Calculate breakeven price for option position.
    
    Args:
        option_type: 'call' or 'put'
        strike: Strike price
        premium: Option premium paid/received
        is_short: Whether position is short
        
    Returns:
        Breakeven price
    """
    if option_type == 'call':
        if is_short:
            return strike + premium
        else:
            return strike + premium
    else:  # put
        if is_short:
            return strike - premium
        else:
            return strike - premium

