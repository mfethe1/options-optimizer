"""
Financial News Integration Module

Bloomberg NEWS equivalent - multi-source news aggregation for trading decisions.
"""

from .base_provider import NewsProvider, NewsArticle, NewsCategory, NewsSentiment
from .benzinga_provider import BenzingaProvider
from .newsapi_provider import NewsAPIProvider
from .news_aggregator import NewsAggregator, get_news_aggregator

__all__ = [
    'NewsProvider',
    'NewsArticle',
    'NewsCategory',
    'NewsSentiment',
    'BenzingaProvider',
    'NewsAPIProvider',
    'NewsAggregator',
    'get_news_aggregator',
]
