"""
Base News Provider Interface

Defines the contract for news providers to ensure consistency.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class NewsCategory(Enum):
    """News categories"""
    GENERAL = "general"
    EARNINGS = "earnings"
    MERGER_ACQUISITION = "merger_acquisition"
    IPO = "ipo"
    GUIDANCE = "guidance"
    ANALYST_RATING = "analyst_rating"
    FDA = "fda"
    LEGAL = "legal"
    BUYBACK = "buyback"
    DIVIDEND = "dividend"
    RESTRUCTURING = "restructuring"
    BANKRUPTCY = "bankruptcy"


class NewsSentiment(Enum):
    """News sentiment"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    BULLISH = "bullish"
    BEARISH = "bearish"


@dataclass
class NewsArticle:
    """News article data"""
    id: str
    title: str
    summary: Optional[str]
    content: Optional[str]
    url: str
    source: str
    author: Optional[str]
    published_at: datetime
    updated_at: Optional[datetime]
    symbols: List[str]  # Related stock symbols
    categories: List[NewsCategory]
    sentiment: Optional[NewsSentiment]
    sentiment_score: Optional[float]  # -1.0 to 1.0
    image_url: Optional[str]
    language: str = "en"
    provider: str = ""  # Which provider fetched this


class NewsProvider(ABC):
    """
    Abstract base class for news providers.

    All news providers must implement this interface.
    """

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize provider.

        Args:
            api_key: API key for the provider
            config: Additional configuration options
        """
        self.api_key = api_key
        self.config = config or {}
        self._is_available = False
        self._last_error: Optional[str] = None

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name"""
        pass

    @property
    @abstractmethod
    def rate_limit(self) -> int:
        """Return requests per minute limit"""
        pass

    @abstractmethod
    async def check_availability(self) -> bool:
        """
        Check if provider is available and API key is valid.

        Returns:
            True if provider is available and working
        """
        pass

    @abstractmethod
    async def get_news(
        self,
        symbols: Optional[List[str]] = None,
        categories: Optional[List[NewsCategory]] = None,
        limit: int = 50,
        since: Optional[datetime] = None,
    ) -> List[NewsArticle]:
        """
        Get news articles.

        Args:
            symbols: Filter by stock symbols (None = all news)
            categories: Filter by categories (None = all categories)
            limit: Maximum number of articles to return
            since: Only return articles published after this time

        Returns:
            List of NewsArticle

        Raises:
            NewsProviderError: If fetch fails
        """
        pass

    @abstractmethod
    async def search_news(
        self,
        query: str,
        symbols: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[NewsArticle]:
        """
        Search news articles by keyword.

        Args:
            query: Search query string
            symbols: Filter by stock symbols
            limit: Maximum number of articles to return

        Returns:
            List of NewsArticle matching query

        Raises:
            NewsProviderError: If search fails
        """
        pass

    def get_last_error(self) -> Optional[str]:
        """Get last error message"""
        return self._last_error

    def is_available(self) -> bool:
        """Check if provider is currently available"""
        return self._is_available


class NewsProviderError(Exception):
    """Exception raised by news providers"""

    def __init__(self, message: str, provider: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.provider = provider
        self.original_error = original_error
        self.timestamp = datetime.now()
