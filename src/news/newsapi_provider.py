"""
NewsAPI Provider

Aggregated news from multiple sources.
https://newsapi.org/

Pricing:
- Developer: FREE (100 requests/day, delayed)
- Business: $449/month (250k requests/month, real-time)
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


class NewsAPIProvider(NewsProvider):
    """NewsAPI provider - broad news coverage"""

    BASE_URL = "https://newsapi.org/v2"

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(api_key, config)
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def provider_name(self) -> str:
        return "newsapi"

    @property
    def rate_limit(self) -> int:
        # Free tier: ~2 requests per minute (100/day)
        # Business tier: Much higher
        return self.config.get('rate_limit', 2)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            headers = {
                'X-Api-Key': self.api_key if self.api_key else ''
            }
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make API request to NewsAPI"""
        if not self.api_key:
            raise NewsProviderError(
                "NewsAPI key not configured",
                self.provider_name
            )

        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}

        try:
            session = await self._get_session()
            async with session.get(url, params=params, timeout=15) as response:
                data = await response.json()

                if response.status == 200:
                    if data.get('status') == 'ok':
                        return data
                    else:
                        raise NewsProviderError(
                            f"API error: {data.get('message', 'Unknown error')}",
                            self.provider_name
                        )
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
                    raise NewsProviderError(
                        f"HTTP {response.status}: {data.get('message', 'Unknown error')}",
                        self.provider_name
                    )

        except NewsProviderError:
            raise
        except Exception as e:
            logger.error(f"NewsAPI request failed: {e}")
            raise NewsProviderError(
                f"Request failed: {str(e)}",
                self.provider_name,
                e
            )

    async def check_availability(self) -> bool:
        """Check if NewsAPI is available"""
        try:
            # Test with a simple request
            await self.search_news("stocks", limit=1)
            self._is_available = True
            self._last_error = None
            logger.info("NewsAPI provider available")
            return True
        except Exception as e:
            self._is_available = False
            self._last_error = str(e)
            logger.warning(f"NewsAPI not available: {e}")
            return False

    def _extract_symbols_from_text(self, text: str) -> List[str]:
        """
        Extract stock symbols from text (very basic).
        In production, use NER or dedicated symbol extraction.
        """
        import re
        # Look for $SYMBOL pattern
        symbols = re.findall(r'\$([A-Z]{1,5})', text)
        return list(set(symbols))

    async def get_news(
        self,
        symbols: Optional[List[str]] = None,
        categories: Optional[List[NewsCategory]] = None,
        limit: int = 50,
        since: Optional[datetime] = None,
    ) -> List[NewsArticle]:
        """
        Get news from NewsAPI.

        Note: NewsAPI doesn't have stock symbol filtering, so we search by symbol in query.
        """
        try:
            endpoint = "/everything"

            # Build search query
            query_parts = []
            if symbols:
                # Search for articles mentioning these symbols
                symbol_query = ' OR '.join([f'"{symbol}"' for symbol in symbols[:5]])  # Limit to 5 symbols
                query_parts.append(f"({symbol_query})")

            # Add business/finance keywords if no symbols specified
            if not symbols:
                query_parts.append("(stocks OR trading OR market OR finance)")

            params = {
                'q': ' AND '.join(query_parts) if query_parts else 'stock market',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': min(limit, 100),
                'domains': 'cnbc.com,bloomberg.com,reuters.com,marketwatch.com,seekingalpha.com,fool.com',
            }

            if since:
                params['from'] = since.isoformat()

            data = await self._make_request(endpoint, params)

            articles = []
            for item in data.get('articles', []):
                # Try to extract symbols from title and description
                text = f"{item.get('title', '')} {item.get('description', '')}"
                article_symbols = self._extract_symbols_from_text(text)

                # If we were searching for specific symbols, include them
                if symbols and not article_symbols:
                    article_symbols = symbols

                # Parse published date
                published_str = item.get('publishedAt', '')
                try:
                    published_at = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
                except:
                    published_at = datetime.now()

                articles.append(NewsArticle(
                    id=f"newsapi_{hash(item.get('url', ''))}",
                    title=item.get('title', ''),
                    summary=item.get('description'),
                    content=item.get('content'),
                    url=item.get('url', ''),
                    source=item.get('source', {}).get('name', 'Unknown'),
                    author=item.get('author'),
                    published_at=published_at,
                    updated_at=None,
                    symbols=article_symbols,
                    categories=[NewsCategory.GENERAL],  # NewsAPI doesn't categorize
                    sentiment=None,
                    sentiment_score=None,
                    image_url=item.get('urlToImage'),
                    provider=self.provider_name,
                ))

            return articles[:limit]

        except NewsProviderError:
            raise
        except Exception as e:
            logger.error(f"Failed to get news from NewsAPI: {e}")
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
        """Search news on NewsAPI"""
        try:
            endpoint = "/everything"

            # Build search query
            search_query = query
            if symbols:
                symbol_query = ' OR '.join(symbols[:5])
                search_query = f"({query}) AND ({symbol_query})"

            params = {
                'q': search_query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': min(limit, 100),
            }

            data = await self._make_request(endpoint, params)

            articles = []
            for item in data.get('articles', []):
                text = f"{item.get('title', '')} {item.get('description', '')}"
                article_symbols = self._extract_symbols_from_text(text)

                if symbols and not article_symbols:
                    article_symbols = symbols

                try:
                    published_at = datetime.fromisoformat(
                        item.get('publishedAt', '').replace('Z', '+00:00')
                    )
                except:
                    published_at = datetime.now()

                articles.append(NewsArticle(
                    id=f"newsapi_{hash(item.get('url', ''))}",
                    title=item.get('title', ''),
                    summary=item.get('description'),
                    content=item.get('content'),
                    url=item.get('url', ''),
                    source=item.get('source', {}).get('name', 'Unknown'),
                    author=item.get('author'),
                    published_at=published_at,
                    updated_at=None,
                    symbols=article_symbols,
                    categories=[NewsCategory.GENERAL],
                    sentiment=None,
                    sentiment_score=None,
                    image_url=item.get('urlToImage'),
                    provider=self.provider_name,
                ))

            return articles[:limit]

        except NewsProviderError:
            raise
        except Exception as e:
            logger.error(f"Failed to search news on NewsAPI: {e}")
            raise NewsProviderError(
                f"Failed to search news: {str(e)}",
                self.provider_name,
                e
            )

    async def close(self):
        """Close aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
