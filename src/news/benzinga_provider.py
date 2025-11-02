"""
Benzinga News Provider

Professional-grade financial news provider used by traders.
https://www.benzinga.com/apis/

Pricing:
- Essential: $199/month (news data, delayed)
- Professional: $499/month (real-time news, alerts)
- Enterprise: Custom pricing
"""
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import aiohttp

from .base_provider import (
    NewsProvider,
    NewsArticle,
    NewsCategory,
    NewsSentiment,
    NewsProviderError
)

logger = logging.getLogger(__name__)


class BenzingaProvider(NewsProvider):
    """Benzinga news provider - professional grade"""

    BASE_URL = "https://api.benzinga.com/api/v2"

    # Category mapping
    CATEGORY_MAP = {
        'earnings': NewsCategory.EARNINGS,
        'mergers-acquisitions': NewsCategory.MERGER_ACQUISITION,
        'ipo': NewsCategory.IPO,
        'guidance': NewsCategory.GUIDANCE,
        'analyst-ratings': NewsCategory.ANALYST_RATING,
        'fda': NewsCategory.FDA,
        'legal': NewsCategory.LEGAL,
        'buybacks': NewsCategory.BUYBACK,
        'dividends': NewsCategory.DIVIDEND,
    }

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(api_key, config)
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def provider_name(self) -> str:
        return "benzinga"

    @property
    def rate_limit(self) -> int:
        return self.config.get('rate_limit', 60)  # requests per minute

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make API request to Benzinga"""
        if not self.api_key:
            raise NewsProviderError(
                "Benzinga API key not configured",
                self.provider_name
            )

        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        params['token'] = self.api_key

        try:
            session = await self._get_session()
            async with session.get(url, params=params, timeout=15) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    raise NewsProviderError(
                        "Rate limit exceeded",
                        self.provider_name
                    )
                elif response.status == 401:
                    raise NewsProviderError(
                        "Invalid API key",
                        self.provider_name
                    )
                else:
                    text = await response.text()
                    raise NewsProviderError(
                        f"HTTP {response.status}: {text}",
                        self.provider_name
                    )

        except NewsProviderError:
            raise
        except Exception as e:
            logger.error(f"Benzinga request failed: {e}")
            raise NewsProviderError(
                f"Request failed: {str(e)}",
                self.provider_name,
                e
            )

    async def check_availability(self) -> bool:
        """Check if Benzinga is available"""
        try:
            # Test with a simple news request
            await self.get_news(limit=1)
            self._is_available = True
            self._last_error = None
            logger.info("Benzinga news provider available")
            return True
        except Exception as e:
            self._is_available = False
            self._last_error = str(e)
            logger.warning(f"Benzinga not available: {e}")
            return False

    async def get_news(
        self,
        symbols: Optional[List[str]] = None,
        categories: Optional[List[NewsCategory]] = None,
        limit: int = 50,
        since: Optional[datetime] = None,
    ) -> List[NewsArticle]:
        """Get news from Benzinga"""
        try:
            endpoint = "/news"
            params = {
                'pageSize': min(limit, 100),
                'displayOutput': 'full',
            }

            # Filter by symbols
            if symbols:
                params['tickers'] = ','.join(symbols)

            # Filter by date
            if since:
                params['dateFrom'] = since.strftime('%Y-%m-%d')
                params['dateTo'] = datetime.now().strftime('%Y-%m-%d')

            # Filter by categories
            if categories:
                # Map our categories to Benzinga channels
                channels = []
                for cat in categories:
                    if cat == NewsCategory.EARNINGS:
                        channels.append('Earnings')
                    elif cat == NewsCategory.ANALYST_RATING:
                        channels.append('Analyst Ratings')
                    elif cat == NewsCategory.FDA:
                        channels.append('FDA')
                    elif cat == NewsCategory.MERGER_ACQUISITION:
                        channels.append('M&A')
                if channels:
                    params['channels'] = ','.join(channels)

            data = await self._make_request(endpoint, params)

            articles = []
            for item in data:
                # Extract symbols
                article_symbols = []
                if 'stocks' in item:
                    article_symbols = [s['name'] for s in item['stocks']]

                # Parse categories
                article_categories = []
                if 'channels' in item:
                    for channel in item['channels']:
                        for bz_cat, our_cat in self.CATEGORY_MAP.items():
                            if bz_cat.lower() in channel.get('name', '').lower():
                                article_categories.append(our_cat)
                                break

                # Default to general if no specific category
                if not article_categories:
                    article_categories = [NewsCategory.GENERAL]

                articles.append(NewsArticle(
                    id=f"benzinga_{item['id']}",
                    title=item.get('title', ''),
                    summary=item.get('teaser'),
                    content=item.get('body'),
                    url=item.get('url', ''),
                    source='Benzinga',
                    author=item.get('author'),
                    published_at=datetime.fromisoformat(
                        item['created'].replace('Z', '+00:00')
                    ),
                    updated_at=datetime.fromisoformat(
                        item['updated'].replace('Z', '+00:00')
                    ) if item.get('updated') else None,
                    symbols=article_symbols,
                    categories=article_categories,
                    sentiment=None,  # Benzinga doesn't provide sentiment in basic API
                    sentiment_score=None,
                    image_url=item.get('image', {}).get('url') if isinstance(item.get('image'), dict) else None,
                    provider=self.provider_name,
                ))

            return articles[:limit]

        except NewsProviderError:
            raise
        except Exception as e:
            logger.error(f"Failed to get news from Benzinga: {e}")
            raise NewsProviderError(
                f"Failed to get news: {str(e)}",
                self.provider_name,
                e
            )

    async def search_news(
        self,
        query: str,
        symbols: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[NewsArticle]:
        """Search news on Benzinga"""
        try:
            # Benzinga uses the same endpoint with search parameter
            endpoint = "/news"
            params = {
                'pageSize': min(limit, 100),
                'displayOutput': 'full',
                'search': query,
            }

            if symbols:
                params['tickers'] = ','.join(symbols)

            data = await self._make_request(endpoint, params)

            articles = []
            for item in data:
                article_symbols = []
                if 'stocks' in item:
                    article_symbols = [s['name'] for s in item['stocks']]

                articles.append(NewsArticle(
                    id=f"benzinga_{item['id']}",
                    title=item.get('title', ''),
                    summary=item.get('teaser'),
                    content=item.get('body'),
                    url=item.get('url', ''),
                    source='Benzinga',
                    author=item.get('author'),
                    published_at=datetime.fromisoformat(
                        item['created'].replace('Z', '+00:00')
                    ),
                    updated_at=datetime.fromisoformat(
                        item['updated'].replace('Z', '+00:00')
                    ) if item.get('updated') else None,
                    symbols=article_symbols,
                    categories=[NewsCategory.GENERAL],
                    sentiment=None,
                    sentiment_score=None,
                    image_url=item.get('image', {}).get('url') if isinstance(item.get('image'), dict) else None,
                    provider=self.provider_name,
                ))

            return articles[:limit]

        except NewsProviderError:
            raise
        except Exception as e:
            logger.error(f"Failed to search news on Benzinga: {e}")
            raise NewsProviderError(
                f"Failed to search news: {str(e)}",
                self.provider_name,
                e
            )

    async def close(self):
        """Close aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
