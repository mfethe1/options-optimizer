"""
Financial News Integration Module

Bloomberg NEWS equivalent - multi-source news aggregation for trading decisions.
"""

from .base_provider import NewsProvider, NewsArticle, NewsCategory, NewsSentiment, NewsProviderError
from .benzinga_provider import BenzingaProvider
from .newsapi_provider import NewsAPIProvider
from .fmp_provider import FMPProvider
from .news_aggregator import NewsAggregator, get_news_aggregator

__all__ = [
    'NewsProvider',
    'NewsArticle',
    'NewsCategory',
    'NewsSentiment',
    'NewsProviderError',
    'BenzingaProvider',
    'NewsAPIProvider',
    'FMPProvider',
    'NewsAggregator',
    'get_news_aggregator',
]
