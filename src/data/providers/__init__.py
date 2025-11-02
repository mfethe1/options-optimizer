"""
Professional Data Providers Module

Provides unified interface to multiple institutional-grade data providers
with automatic fallback capability.
"""

from .base_provider import DataProvider, MarketData, OptionsData, QuoteData
from .polygon_provider import PolygonProvider
from .intrinio_provider import IntrinioProvider
from .yfinance_provider import YFinanceProvider
from .provider_manager import DataProviderManager, get_provider_manager

__all__ = [
    'DataProvider',
    'MarketData',
    'OptionsData',
    'QuoteData',
    'PolygonProvider',
    'IntrinioProvider',
    'YFinanceProvider',
    'DataProviderManager',
    'get_provider_manager',
]
