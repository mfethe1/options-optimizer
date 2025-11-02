"""
Financial Modeling Prep News Provider

Free/affordable financial data provider with news.
https://financialmodelingprep.com/

Pricing:
- Free: 250 requests/day
- Starter: $14/month (300 requests/day)
- Professional: $59/month (unlimited)
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


class FMPProvider(NewsProvider):
    """Financial Modeling Prep provider - free/affordable option"""

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(api_key, config)
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def provider_name(self) -> str:
        return "fmp"

    @property
    def rate_limit(self) -> int:
        # Free tier: ~10 requests per minute
        return self.config.get('rate_limit', 10)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """Make API request to FMP"""
        if not self.api_key:
            raise NewsProviderError(
                "FMP API key not configured",
                self.provider_name
            )

        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        params['apikey'] = self.api_key

        try:
            session = await self._get_session()
            async with session.get(url, params=params, timeout=15) as response:
                if response.status == 200:
                    data = await response.json()
                    # FMP returns error as dict with 'Error Message'
                    if isinstance(data, dict) and 'Error Message' in data:
                        raise NewsProviderError(
                            data['Error Message'],
                            self.provider_name
                        )
                    return data
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
            logger.error(f"FMP request failed: {e}")
            raise NewsProviderError(
                f"Request failed: {str(e)}",
                self.provider_name,
                e
            )

    async def check_availability(self) -> bool:
        """Check if FMP is available"""
        try:
            # Test with a simple request
            await self.get_news(limit=1)
            self._is_available = True
            self._last_error = None
            logger.info("FMP news provider available")
            return True
        except Exception as e:
            self._is_available = False
            self._last_error = str(e)
            logger.warning(f"FMP not available: {e}")
            return False

    async def get_news(
        self,
        symbols: Optional[List[str]] = None,
        categories: Optional[List[NewsCategory]] = None,
        limit: int = 50,
        since: Optional[datetime] = None,
    ) -> List[NewsArticle]:
        """Get news from FMP"""
        try:
            articles = []

            if symbols:
                # FMP has symbol-specific news endpoint
                for symbol in symbols[:5]:  # Limit to avoid too many requests
                    try:
                        endpoint = f"/stock_news"
                        params = {
                            'tickers': symbol,
                            'limit': limit,
                        }

                        data = await self._make_request(endpoint, params)

                        if not isinstance(data, list):
                            continue

                        for item in data:
                            try:
                                published_at = datetime.fromisoformat(
                                    item['publishedDate'].replace('Z', '+00:00')
                                )
                            except:
                                published_at = datetime.now()

                            # Skip if too old
                            if since and published_at < since:
                                continue

                            articles.append(NewsArticle(
                                id=f"fmp_{hash(item.get('url', ''))}",
                                title=item.get('title', ''),
                                summary=item.get('text'),
                                content=None,  # FMP doesn't provide full content
                                url=item.get('url', ''),
                                source=item.get('site', 'Unknown'),
                                author=None,
                                published_at=published_at,
                                updated_at=None,
                                symbols=[symbol],
                                categories=[NewsCategory.GENERAL],
                                sentiment=None,
                                sentiment_score=None,
                                image_url=item.get('image'),
                                provider=self.provider_name,
                            ))

                    except Exception as e:
                        logger.warning(f"Failed to get news for {symbol}: {e}")
                        continue

            else:
                # General market news
                endpoint = "/fmp/articles"
                params = {
                    'page': 0,
                    'size': limit,
                }

                data = await self._make_request(endpoint, params)

                if isinstance(data, dict) and 'content' in data:
                    data = data['content']

                if isinstance(data, list):
                    for item in data:
                        try:
                            published_at = datetime.fromisoformat(
                                item['publishedDate'].replace('Z', '+00:00')
                            )
                        except:
                            published_at = datetime.now()

                        if since and published_at < since:
                            continue

                        articles.append(NewsArticle(
                            id=f"fmp_{item.get('id', hash(item.get('url', '')))}",
                            title=item.get('title', ''),
                            summary=item.get('content'),
                            content=None,
                            url=item.get('url', ''),
                            source=item.get('site', 'FMP'),
                            author=item.get('author'),
                            published_at=published_at,
                            updated_at=None,
                            symbols=[],
                            categories=[NewsCategory.GENERAL],
                            sentiment=None,
                            sentiment_score=None,
                            image_url=item.get('image'),
                            provider=self.provider_name,
                        ))

            # Remove duplicates based on URL
            seen_urls = set()
            unique_articles = []
            for article in articles:
                if article.url not in seen_urls:
                    seen_urls.add(article.url)
                    unique_articles.append(article)

            # Sort by published date, most recent first
            unique_articles.sort(key=lambda x: x.published_at, reverse=True)

            return unique_articles[:limit]

        except NewsProviderError:
            raise
        except Exception as e:
            logger.error(f"Failed to get news from FMP: {e}")
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
        """
        Search news on FMP.

        Note: FMP doesn't have a dedicated search endpoint,
        so we fetch general news and filter by keyword.
        """
        try:
            # Get news
            articles = await self.get_news(symbols=symbols, limit=limit * 2)

            # Filter by query (case-insensitive)
            query_lower = query.lower()
            filtered = [
                article for article in articles
                if query_lower in article.title.lower() or
                (article.summary and query_lower in article.summary.lower())
            ]

            return filtered[:limit]

        except NewsProviderError:
            raise
        except Exception as e:
            logger.error(f"Failed to search news on FMP: {e}")
            raise NewsProviderError(
                f"Failed to search news: {str(e)}",
                self.provider_name,
                e
            )

    async def close(self):
        """Close aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
