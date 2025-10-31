"""
News Aggregator

Aggregates news from multiple providers with deduplication and ranking.
"""
import asyncio
import logging
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from collections import defaultdict

from .base_provider import (
    NewsProvider,
    NewsArticle,
    NewsCategory,
    NewsProviderError
)
from .benzinga_provider import BenzingaProvider
from .newsapi_provider import NewsAPIProvider
from .fmp_provider import FMPProvider

logger = logging.getLogger(__name__)


class NewsAggregator:
    """
    Aggregates news from multiple providers.

    Features:
    - Multi-provider fetching with parallel requests
    - Deduplication by URL and title similarity
    - Provider priority ordering
    - Caching for performance
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize news aggregator.

        Args:
            config: Configuration dict with API keys

        Config format:
            {
                'benzinga_api_key': 'your_key',
                'newsapi_key': 'your_key',
                'fmp_api_key': 'your_key',
                'preferred_providers': ['benzinga', 'newsapi', 'fmp'],
                'enable_deduplication': True,
            }
        """
        self.config = config or self._load_config_from_env()
        self.providers: List[NewsProvider] = []
        self._initialized = False

    def _load_config_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        return {
            'benzinga_api_key': os.getenv('BENZINGA_API_KEY'),
            'newsapi_key': os.getenv('NEWSAPI_KEY'),
            'fmp_api_key': os.getenv('FMP_API_KEY'),
            'preferred_providers': os.getenv('NEWS_PROVIDERS', 'benzinga,newsapi,fmp').split(','),
            'enable_deduplication': os.getenv('NEWS_DEDUPLICATION', 'true').lower() == 'true',
        }

    async def initialize(self) -> None:
        """Initialize all configured news providers"""
        if self._initialized:
            return

        logger.info("Initializing news aggregator...")

        providers_to_try = []

        # Benzinga (professional)
        if self.config.get('benzinga_api_key'):
            providers_to_try.append(
                BenzingaProvider(api_key=self.config['benzinga_api_key'])
            )
        else:
            logger.info("Benzinga API key not configured")

        # NewsAPI (broad coverage)
        if self.config.get('newsapi_key'):
            providers_to_try.append(
                NewsAPIProvider(api_key=self.config['newsapi_key'])
            )
        else:
            logger.info("NewsAPI key not configured")

        # FMP (free/affordable)
        if self.config.get('fmp_api_key'):
            providers_to_try.append(
                FMPProvider(api_key=self.config['fmp_api_key'])
            )
        else:
            logger.info("FMP API key not configured")

        # Check availability of each provider
        for provider in providers_to_try:
            try:
                is_available = await provider.check_availability()
                if is_available:
                    self.providers.append(provider)
                    logger.info(f"✓ {provider.provider_name} news provider available")
                else:
                    logger.warning(
                        f"✗ {provider.provider_name} not available: "
                        f"{provider.get_last_error()}"
                    )
            except Exception as e:
                logger.warning(f"Failed to initialize {provider.provider_name}: {e}")

        if not self.providers:
            logger.warning(
                "No news providers available! Please configure at least one: "
                "Benzinga (BENZINGA_API_KEY), NewsAPI (NEWSAPI_KEY), or FMP (FMP_API_KEY)"
            )

        logger.info(f"News aggregator initialized with {len(self.providers)} providers")
        self._initialized = True

    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """
        Calculate similarity between two titles (simple word overlap).

        Returns:
            Similarity score between 0 and 1
        """
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """
        Remove duplicate articles based on URL and title similarity.

        Args:
            articles: List of articles to deduplicate

        Returns:
            Deduplicated list of articles
        """
        if not self.config.get('enable_deduplication', True):
            return articles

        # Group by URL first (exact duplicates)
        url_map: Dict[str, NewsArticle] = {}
        for article in articles:
            if article.url not in url_map:
                url_map[article.url] = article
            else:
                # Keep the one from the preferred provider
                existing = url_map[article.url]
                provider_priority = {
                    'benzinga': 3,
                    'newsapi': 2,
                    'fmp': 1,
                }
                if provider_priority.get(article.provider, 0) > provider_priority.get(existing.provider, 0):
                    url_map[article.url] = article

        unique_articles = list(url_map.values())

        # Now check for title similarity (near-duplicates)
        final_articles = []
        for article in unique_articles:
            is_duplicate = False
            for existing in final_articles:
                similarity = self._calculate_title_similarity(article.title, existing.title)
                if similarity > 0.8:  # 80% similar titles are considered duplicates
                    is_duplicate = True
                    break

            if not is_duplicate:
                final_articles.append(article)

        logger.debug(f"Deduplicated {len(articles)} articles to {len(final_articles)}")
        return final_articles

    async def get_news(
        self,
        symbols: Optional[List[str]] = None,
        categories: Optional[List[NewsCategory]] = None,
        limit: int = 50,
        since: Optional[datetime] = None,
    ) -> List[NewsArticle]:
        """
        Get news from all available providers and merge results.

        Args:
            symbols: Filter by stock symbols
            categories: Filter by categories
            limit: Maximum number of articles to return
            since: Only return articles published after this time

        Returns:
            Merged and deduplicated list of articles
        """
        if not self._initialized:
            await self.initialize()

        if not self.providers:
            logger.warning("No news providers available, returning empty list")
            return []

        # Fetch from all providers in parallel
        tasks = []
        for provider in self.providers:
            task = provider.get_news(
                symbols=symbols,
                categories=categories,
                limit=limit,
                since=since
            )
            tasks.append(task)

        # Gather results, catching errors
        all_articles = []
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            provider = self.providers[i]
            if isinstance(result, Exception):
                logger.error(f"Error fetching news from {provider.provider_name}: {result}")
            else:
                all_articles.extend(result)
                logger.info(f"Fetched {len(result)} articles from {provider.provider_name}")

        # Deduplicate
        unique_articles = self._deduplicate_articles(all_articles)

        # Sort by published date (most recent first)
        unique_articles.sort(key=lambda x: x.published_at, reverse=True)

        return unique_articles[:limit]

    async def search_news(
        self,
        query: str,
        symbols: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[NewsArticle]:
        """
        Search news across all providers.

        Args:
            query: Search query string
            symbols: Filter by stock symbols
            limit: Maximum number of articles to return

        Returns:
            Merged and deduplicated search results
        """
        if not self._initialized:
            await self.initialize()

        if not self.providers:
            logger.warning("No news providers available, returning empty list")
            return []

        # Search all providers in parallel
        tasks = []
        for provider in self.providers:
            task = provider.search_news(
                query=query,
                symbols=symbols,
                limit=limit
            )
            tasks.append(task)

        # Gather results
        all_articles = []
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            provider = self.providers[i]
            if isinstance(result, Exception):
                logger.error(f"Error searching news on {provider.provider_name}: {result}")
            else:
                all_articles.extend(result)
                logger.info(f"Found {len(result)} articles on {provider.provider_name}")

        # Deduplicate
        unique_articles = self._deduplicate_articles(all_articles)

        # Sort by published date
        unique_articles.sort(key=lambda x: x.published_at, reverse=True)

        return unique_articles[:limit]

    def get_provider_status(self) -> Dict[str, Any]:
        """
        Get status of all news providers.

        Returns:
            Dict with provider status information
        """
        return {
            'total_providers': len(self.providers),
            'available_providers': [p.provider_name for p in self.providers],
            'provider_details': [
                {
                    'name': p.provider_name,
                    'available': p.is_available(),
                    'rate_limit': p.rate_limit,
                    'last_error': p.get_last_error(),
                }
                for p in self.providers
            ],
        }

    async def close(self):
        """Close all provider sessions"""
        for provider in self.providers:
            try:
                await provider.close()
            except Exception as e:
                logger.warning(f"Error closing {provider.provider_name}: {e}")


# Global singleton instance
_news_aggregator: Optional[NewsAggregator] = None


def get_news_aggregator(config: Optional[Dict[str, Any]] = None) -> NewsAggregator:
    """
    Get or create global news aggregator instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        NewsAggregator singleton instance
    """
    global _news_aggregator
    if _news_aggregator is None:
        _news_aggregator = NewsAggregator(config)
    return _news_aggregator


async def initialize_news_aggregator(config: Optional[Dict[str, Any]] = None):
    """
    Initialize the global news aggregator.

    Should be called during application startup.

    Args:
        config: Optional configuration
    """
    aggregator = get_news_aggregator(config)
    await aggregator.initialize()
    logger.info("News aggregator initialized")
