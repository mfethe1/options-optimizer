"""
Base Data Provider Interface

Defines the contract that all data providers must implement.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from enum import Enum


class DataProviderType(Enum):
    """Types of data providers"""
    POLYGON = "polygon"
    INTRINIO = "intrinio"
    ALPHA_VANTAGE = "alpha_vantage"
    YFINANCE = "yfinance"  # Fallback only


class DataQuality(Enum):
    """Data quality tiers"""
    INSTITUTIONAL = "institutional"  # Polygon, Intrinio
    PROFESSIONAL = "professional"    # Alpha Vantage
    RETAIL = "retail"                # yfinance (fallback)


@dataclass
class QuoteData:
    """Real-time quote data"""
    symbol: str
    bid: Optional[float]
    ask: Optional[float]
    last: Optional[float]
    volume: int
    timestamp: datetime
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    prev_close: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None


@dataclass
class OptionsData:
    """Options chain data for a specific contract"""
    symbol: str
    underlying_symbol: str
    strike: float
    expiration: date
    option_type: str  # 'call' or 'put'
    bid: Optional[float]
    ask: Optional[float]
    last: Optional[float]
    volume: int
    open_interest: int
    implied_volatility: Optional[float]
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    timestamp: datetime = None


@dataclass
class HistoricalBar:
    """Historical OHLCV bar"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None  # Volume-weighted average price


@dataclass
class MarketData:
    """Complete market data package"""
    symbol: str
    quote: QuoteData
    timestamp: datetime
    provider: str
    quality: DataQuality


class DataProvider(ABC):
    """
    Abstract base class for data providers.

    All data providers must implement this interface to ensure
    consistent behavior and enable automatic fallback.
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
    def provider_type(self) -> DataProviderType:
        """Return the provider type"""
        pass

    @property
    @abstractmethod
    def data_quality(self) -> DataQuality:
        """Return the data quality tier"""
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
    async def get_quote(self, symbol: str) -> QuoteData:
        """
        Get real-time quote for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            QuoteData with current quote

        Raises:
            ProviderError: If data fetch fails
        """
        pass

    @abstractmethod
    async def get_options_chain(
        self,
        symbol: str,
        expiration: Optional[date] = None
    ) -> List[OptionsData]:
        """
        Get options chain for a symbol.

        Args:
            symbol: Stock symbol
            expiration: Specific expiration date (None = all expirations)

        Returns:
            List of OptionsData for all strikes and expirations

        Raises:
            ProviderError: If data fetch fails
        """
        pass

    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval: str = '1d'
    ) -> List[HistoricalBar]:
        """
        Get historical OHLCV data.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Bar interval ('1m', '5m', '1h', '1d', etc.)

        Returns:
            List of HistoricalBar data

        Raises:
            ProviderError: If data fetch fails
        """
        pass

    @abstractmethod
    async def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get company information.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with company info (name, sector, industry, etc.)

        Raises:
            ProviderError: If data fetch fails
        """
        pass

    def get_last_error(self) -> Optional[str]:
        """Get last error message"""
        return self._last_error

    def is_available(self) -> bool:
        """Check if provider is currently available"""
        return self._is_available


class ProviderError(Exception):
    """Exception raised by data providers"""

    def __init__(self, message: str, provider: DataProviderType, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.provider = provider
        self.original_error = original_error
        self.timestamp = datetime.now()
