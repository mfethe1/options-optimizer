"""
Data Provider Manager

Manages multiple data providers with automatic fallback capability.
Ensures institutional-grade reliability by trying providers in priority order.
"""
import asyncio
import logging
import os
from typing import List, Optional, Dict, Any
from datetime import date

from .base_provider import (
    DataProvider,
    DataProviderType,
    DataQuality,
    QuoteData,
    OptionsData,
    HistoricalBar,
    ProviderError
)
from .polygon_provider import PolygonProvider
from .intrinio_provider import IntrinioProvider
from .yfinance_provider import YFinanceProvider

logger = logging.getLogger(__name__)


class DataProviderManager:
    """
    Manages multiple data providers with automatic fallback.

    Providers are tried in order of data quality:
    1. Polygon.io (institutional)
    2. Intrinio (institutional, excellent for options)
    3. YFinance (retail, FALLBACK ONLY)

    If a provider fails, automatically falls back to the next available provider.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize provider manager.

        Args:
            config: Configuration dict with API keys and preferences

        Config format:
            {
                'polygon_api_key': 'your_key',
                'intrinio_api_key': 'your_key',
                'preferred_provider': 'polygon',  # optional
                'enable_fallback': True,  # default True
            }
        """
        self.config = config or self._load_config_from_env()
        self.providers: List[DataProvider] = []
        self.active_provider: Optional[DataProvider] = None
        self._initialized = False

    def _load_config_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        return {
            'polygon_api_key': os.getenv('POLYGON_API_KEY'),
            'intrinio_api_key': os.getenv('INTRINIO_API_KEY'),
            'alpha_vantage_api_key': os.getenv('ALPHA_VANTAGE_API_KEY'),
            'preferred_provider': os.getenv('DATA_PROVIDER', 'polygon'),
            'enable_fallback': os.getenv('ENABLE_DATA_FALLBACK', 'true').lower() == 'true',
        }

    async def initialize(self) -> None:
        """
        Initialize all configured providers and check availability.

        Providers are added in priority order based on data quality.
        """
        if self._initialized:
            return

        logger.info("Initializing data provider manager...")

        # Try to initialize providers in priority order
        providers_to_try = []

        # 1. Polygon.io (institutional)
        if self.config.get('polygon_api_key'):
            providers_to_try.append(
                PolygonProvider(api_key=self.config['polygon_api_key'])
            )
        else:
            logger.warning("Polygon API key not configured")

        # 2. Intrinio (institutional, excellent for options)
        if self.config.get('intrinio_api_key'):
            providers_to_try.append(
                IntrinioProvider(api_key=self.config['intrinio_api_key'])
            )
        else:
            logger.warning("Intrinio API key not configured")

        # 3. YFinance (retail, fallback only)
        if self.config.get('enable_fallback', True):
            providers_to_try.append(
                YFinanceProvider()
            )
            logger.info("YFinance fallback provider enabled (RETAIL DATA QUALITY)")
        else:
            logger.warning("Fallback provider disabled - service may fail if primary providers are down")

        # Check availability of each provider
        for provider in providers_to_try:
            try:
                is_available = await provider.check_availability()
                if is_available:
                    self.providers.append(provider)
                    logger.info(
                        f"✓ {provider.provider_type.value} available "
                        f"(Quality: {provider.data_quality.value})"
                    )
                else:
                    logger.warning(
                        f"✗ {provider.provider_type.value} not available: "
                        f"{provider.get_last_error()}"
                    )
            except Exception as e:
                logger.warning(f"Failed to initialize {provider.provider_type.value}: {e}")

        if not self.providers:
            raise RuntimeError(
                "No data providers available! Please configure at least one provider: "
                "Polygon (POLYGON_API_KEY), Intrinio (INTRINIO_API_KEY), "
                "or enable fallback (yfinance)"
            )

        # Set active provider to first available
        self.active_provider = self.providers[0]
        logger.info(
            f"Active provider: {self.active_provider.provider_type.value} "
            f"({self.active_provider.data_quality.value} quality)"
        )

        if self.active_provider.data_quality == DataQuality.RETAIL:
            logger.warning(
                "⚠️  WARNING: Using RETAIL-GRADE data provider (yfinance). "
                "For production use, configure institutional providers: "
                "Polygon.io or Intrinio"
            )

        self._initialized = True

    async def _try_providers(self, operation: str, func_name: str, *args, **kwargs) -> Any:
        """
        Try an operation across all providers with automatic fallback.

        Args:
            operation: Operation description for logging
            func_name: Name of provider method to call
            *args, **kwargs: Arguments to pass to provider method

        Returns:
            Result from first successful provider

        Raises:
            ProviderError: If all providers fail
        """
        if not self._initialized:
            await self.initialize()

        errors = []

        for provider in self.providers:
            try:
                logger.debug(f"Trying {operation} with {provider.provider_type.value}...")

                # Get the method from provider
                method = getattr(provider, func_name)
                result = await method(*args, **kwargs)

                # Success! Update active provider
                if self.active_provider != provider:
                    logger.info(f"Switched active provider to {provider.provider_type.value}")
                    self.active_provider = provider

                return result

            except ProviderError as e:
                logger.warning(
                    f"{provider.provider_type.value} failed for {operation}: {e}"
                )
                errors.append((provider.provider_type.value, str(e)))
                continue

            except Exception as e:
                logger.error(
                    f"Unexpected error from {provider.provider_type.value}: {e}"
                )
                errors.append((provider.provider_type.value, str(e)))
                continue

        # All providers failed
        error_msg = f"All providers failed for {operation}:\n" + "\n".join(
            f"  - {provider}: {error}" for provider, error in errors
        )
        logger.error(error_msg)
        raise ProviderError(
            error_msg,
            DataProviderType.YFINANCE  # Arbitrary, represents "all failed"
        )

    async def get_quote(self, symbol: str) -> QuoteData:
        """
        Get real-time quote with automatic fallback.

        Args:
            symbol: Stock symbol

        Returns:
            QuoteData from first available provider

        Raises:
            ProviderError: If all providers fail
        """
        return await self._try_providers(
            f"quote for {symbol}",
            'get_quote',
            symbol
        )

    async def get_options_chain(
        self,
        symbol: str,
        expiration: Optional[date] = None
    ) -> List[OptionsData]:
        """
        Get options chain with automatic fallback.

        Args:
            symbol: Stock symbol
            expiration: Specific expiration date (None = all)

        Returns:
            List of OptionsData from first available provider

        Raises:
            ProviderError: If all providers fail
        """
        return await self._try_providers(
            f"options chain for {symbol}",
            'get_options_chain',
            symbol,
            expiration
        )

    async def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval: str = '1d'
    ) -> List[HistoricalBar]:
        """
        Get historical data with automatic fallback.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Bar interval

        Returns:
            List of HistoricalBar from first available provider

        Raises:
            ProviderError: If all providers fail
        """
        return await self._try_providers(
            f"historical data for {symbol}",
            'get_historical_data',
            symbol,
            start_date,
            end_date,
            interval
        )

    async def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get company information with automatic fallback.

        Args:
            symbol: Stock symbol

        Returns:
            Company info dict from first available provider

        Raises:
            ProviderError: If all providers fail
        """
        return await self._try_providers(
            f"company info for {symbol}",
            'get_company_info',
            symbol
        )

    def get_active_provider(self) -> Optional[DataProvider]:
        """Get currently active provider"""
        return self.active_provider

    def get_active_provider_name(self) -> str:
        """Get name of active provider"""
        if self.active_provider:
            return self.active_provider.provider_type.value
        return "none"

    def get_data_quality(self) -> Optional[DataQuality]:
        """Get data quality tier of active provider"""
        if self.active_provider:
            return self.active_provider.data_quality
        return None

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return [p.provider_type.value for p in self.providers]

    def get_provider_status(self) -> Dict[str, Any]:
        """
        Get detailed status of all providers.

        Returns:
            Dict with provider status information
        """
        return {
            'active_provider': self.get_active_provider_name(),
            'data_quality': self.get_data_quality().value if self.get_data_quality() else None,
            'available_providers': self.get_available_providers(),
            'provider_details': [
                {
                    'name': p.provider_type.value,
                    'quality': p.data_quality.value,
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
                logger.warning(f"Error closing {provider.provider_type.value}: {e}")


# Global singleton instance
_provider_manager: Optional[DataProviderManager] = None


def get_provider_manager(config: Optional[Dict[str, Any]] = None) -> DataProviderManager:
    """
    Get or create global provider manager instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        DataProviderManager singleton instance
    """
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = DataProviderManager(config)
    return _provider_manager


async def initialize_provider_manager(config: Optional[Dict[str, Any]] = None):
    """
    Initialize the global provider manager.

    Should be called during application startup.

    Args:
        config: Optional configuration
    """
    manager = get_provider_manager(config)
    await manager.initialize()
    logger.info("Data provider manager initialized")
