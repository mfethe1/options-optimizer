"""
Polygon.io Data Provider

Institutional-grade market data provider.
https://polygon.io/

Pricing:
- Starter: $99/month (5 requests/second, delayed data)
- Developer: $249/month (100 requests/second, 15-min delayed)
- Advanced: $399/month (unlimited, real-time)
"""
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
import aiohttp

from .base_provider import (
    DataProvider,
    DataProviderType,
    DataQuality,
    QuoteData,
    OptionsData,
    HistoricalBar,
    ProviderError
)

logger = logging.getLogger(__name__)


class PolygonProvider(DataProvider):
    """Polygon.io data provider - institutional grade"""

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(api_key, config)
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def provider_type(self) -> DataProviderType:
        return DataProviderType.POLYGON

    @property
    def data_quality(self) -> DataQuality:
        return DataQuality.INSTITUTIONAL

    @property
    def rate_limit(self) -> int:
        # Default to conservative limit for starter plan
        return self.config.get('rate_limit', 5)  # requests per second

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make API request to Polygon.io

        Args:
            endpoint: API endpoint (e.g., '/v2/aggs/ticker/AAPL/range/1/day/...')
            params: Query parameters

        Returns:
            Response JSON

        Raises:
            ProviderError: If request fails
        """
        if not self.api_key:
            raise ProviderError(
                "Polygon API key not configured",
                self.provider_type
            )

        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        params['apiKey'] = self.api_key

        try:
            session = await self._get_session()
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('status') == 'ERROR':
                        raise ProviderError(
                            f"Polygon API error: {data.get('error', 'Unknown error')}",
                            self.provider_type
                        )
                    return data
                elif response.status == 429:
                    raise ProviderError(
                        "Rate limit exceeded",
                        self.provider_type
                    )
                elif response.status == 401:
                    raise ProviderError(
                        "Invalid API key",
                        self.provider_type
                    )
                else:
                    text = await response.text()
                    raise ProviderError(
                        f"HTTP {response.status}: {text}",
                        self.provider_type
                    )

        except ProviderError:
            raise
        except Exception as e:
            logger.error(f"Polygon request failed: {e}")
            raise ProviderError(
                f"Request failed: {str(e)}",
                self.provider_type,
                e
            )

    async def check_availability(self) -> bool:
        """Check if Polygon.io is available"""
        try:
            # Test with a simple quote request
            await self.get_quote('AAPL')
            self._is_available = True
            self._last_error = None
            return True
        except Exception as e:
            self._is_available = False
            self._last_error = str(e)
            logger.warning(f"Polygon.io not available: {e}")
            return False

    async def get_quote(self, symbol: str) -> QuoteData:
        """
        Get real-time quote from Polygon.io

        Uses /v2/last/trade/ endpoint for latest trade data
        """
        try:
            # Get last trade
            endpoint = f"/v2/last/trade/{symbol}"
            data = await self._make_request(endpoint)

            if 'results' not in data:
                raise ProviderError(
                    f"No data returned for {symbol}",
                    self.provider_type
                )

            result = data['results']

            # Get previous close for change calculation
            prev_close_data = await self._make_request(f"/v2/aggs/ticker/{symbol}/prev")
            prev_close = None
            if 'results' in prev_close_data and len(prev_close_data['results']) > 0:
                prev_close = prev_close_data['results'][0].get('c')

            last_price = result.get('p')
            change = None
            change_pct = None
            if last_price and prev_close:
                change = last_price - prev_close
                change_pct = (change / prev_close) * 100

            return QuoteData(
                symbol=symbol,
                bid=None,  # Not available in last trade endpoint
                ask=None,
                last=last_price,
                volume=result.get('s', 0),
                timestamp=datetime.fromtimestamp(result.get('t', 0) / 1000),
                prev_close=prev_close,
                change=change,
                change_pct=change_pct,
            )

        except ProviderError:
            raise
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            raise ProviderError(
                f"Failed to get quote: {str(e)}",
                self.provider_type,
                e
            )

    async def get_options_chain(
        self,
        symbol: str,
        expiration: Optional[date] = None
    ) -> List[OptionsData]:
        """
        Get options chain from Polygon.io

        Uses /v3/reference/options/contracts endpoint
        """
        try:
            endpoint = f"/v3/reference/options/contracts"
            params = {
                'underlying_ticker': symbol,
                'limit': 1000,
            }

            if expiration:
                params['expiration_date'] = expiration.strftime('%Y-%m-%d')

            data = await self._make_request(endpoint, params)

            if 'results' not in data:
                logger.warning(f"No options data for {symbol}")
                return []

            options_list = []
            for contract in data['results']:
                # Parse contract symbol to get details
                contract_symbol = contract.get('ticker', '')
                strike = contract.get('strike_price')
                exp_date_str = contract.get('expiration_date')
                option_type = contract.get('contract_type', '').lower()

                if not all([strike, exp_date_str, option_type]):
                    continue

                exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d').date()

                # Get quote for this specific contract
                try:
                    quote = await self.get_quote(contract_symbol)

                    options_list.append(OptionsData(
                        symbol=contract_symbol,
                        underlying_symbol=symbol,
                        strike=strike,
                        expiration=exp_date,
                        option_type=option_type,
                        bid=quote.bid,
                        ask=quote.ask,
                        last=quote.last,
                        volume=quote.volume,
                        open_interest=contract.get('open_interest', 0),
                        implied_volatility=None,  # Would need separate Greeks API
                        timestamp=quote.timestamp,
                    ))
                except Exception as e:
                    logger.debug(f"Could not get quote for {contract_symbol}: {e}")
                    continue

            return options_list

        except ProviderError:
            raise
        except Exception as e:
            logger.error(f"Failed to get options chain for {symbol}: {e}")
            raise ProviderError(
                f"Failed to get options chain: {str(e)}",
                self.provider_type,
                e
            )

    async def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval: str = '1d'
    ) -> List[HistoricalBar]:
        """
        Get historical data from Polygon.io

        Uses /v2/aggs/ticker/{symbol}/range endpoint
        """
        try:
            # Convert interval to Polygon format
            interval_map = {
                '1m': ('1', 'minute'),
                '5m': ('5', 'minute'),
                '15m': ('15', 'minute'),
                '1h': ('1', 'hour'),
                '1d': ('1', 'day'),
            }

            if interval not in interval_map:
                raise ValueError(f"Unsupported interval: {interval}")

            multiplier, timespan = interval_map[interval]

            endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000,
            }

            data = await self._make_request(endpoint, params)

            if 'results' not in data:
                logger.warning(f"No historical data for {symbol}")
                return []

            bars = []
            for bar in data['results']:
                bars.append(HistoricalBar(
                    timestamp=datetime.fromtimestamp(bar['t'] / 1000),
                    open=bar['o'],
                    high=bar['h'],
                    low=bar['l'],
                    close=bar['c'],
                    volume=bar['v'],
                    vwap=bar.get('vw'),
                ))

            return bars

        except ProviderError:
            raise
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            raise ProviderError(
                f"Failed to get historical data: {str(e)}",
                self.provider_type,
                e
            )

    async def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get company information from Polygon.io

        Uses /v3/reference/tickers/{symbol} endpoint
        """
        try:
            endpoint = f"/v3/reference/tickers/{symbol}"
            data = await self._make_request(endpoint)

            if 'results' not in data:
                raise ProviderError(
                    f"No company info for {symbol}",
                    self.provider_type
                )

            result = data['results']

            return {
                'symbol': symbol,
                'name': result.get('name'),
                'market': result.get('market'),
                'locale': result.get('locale'),
                'primary_exchange': result.get('primary_exchange'),
                'type': result.get('type'),
                'currency_name': result.get('currency_name'),
                'cik': result.get('cik'),
                'composite_figi': result.get('composite_figi'),
                'share_class_figi': result.get('share_class_figi'),
                'market_cap': result.get('market_cap'),
                'description': result.get('description'),
                'homepage_url': result.get('homepage_url'),
                'total_employees': result.get('total_employees'),
                'list_date': result.get('list_date'),
            }

        except ProviderError:
            raise
        except Exception as e:
            logger.error(f"Failed to get company info for {symbol}: {e}")
            raise ProviderError(
                f"Failed to get company info: {str(e)}",
                self.provider_type,
                e
            )

    async def close(self):
        """Close aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
