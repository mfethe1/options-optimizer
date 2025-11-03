"""
Bio-Financial Crossover Module

This module implements breakthrough bio-financial applications:
1. Epidemic Volatility Forecasting - SIR/SEIR models for market fear contagion
2. Circadian Trading Patterns - Gene oscillations for intraday patterns
3. Protein Folding Option Pricing - Conformational dynamics for IV surfaces

These are cutting-edge, unexplored applications transferring biological
dynamics to financial markets for competitive advantage.
"""

from .epidemic_volatility import (
    EpidemicVolatilityModel,
    SIRModel,
    SEIRModel,
    EpidemicVolatilityPredictor,
    MarketRegime
)

__all__ = [
    'EpidemicVolatilityModel',
    'SIRModel',
    'SEIRModel',
    'EpidemicVolatilityPredictor',
    'MarketRegime'
]
