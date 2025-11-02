"""
Intrinio Data Provider

Institutional-grade financial data provider specialized in options.
https://intrinio.com/

Pricing:
- Options Essential: $150/month (real-time options data)
- Options Pro: $300/month (enhanced data + analytics)
- Enterprise: Custom pricing
"""
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, date
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


class IntrinioProvider(DataProvider):
    """Intrinio data provider - institutional grade with strong options support"""

    BASE_URL = "https://api-v2.intrinio.com"

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(api_key, config)
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def provider_type(self) -> DataProviderType:
        return DataProviderType.INTRINIO

    @property
    def data_quality(self) -> DataQuality:
        return DataQuality.INSTITUTIONAL

    @property
    def rate_limit(self) -> int:
        return self.config.get('rate_limit', 60)  # requests per minute

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            headers = {
                'Authorization': f'Bearer {self.api_key}' if self.api_key else ''
            }
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make API request to Intrinio

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            Response JSON

        Raises:
            ProviderError: If request fails
        """
        if not self.api_key:
            raise ProviderError(
                "Intrinio API key not configured",
                self.provider_type
            )

        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}

        try:
            session = await self._get_session()
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    return await response.json()
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
            logger.error(f"Intrinio request failed: {e}")
            raise ProviderError(
                f"Request failed: {str(e)}",
                self.provider_type,
                e
            )

    async def check_availability(self) -> bool:
        """Check if Intrinio is available"""
        try:
            # Test with a simple quote request
            await self.get_quote('AAPL')
            self._is_available = True
            self._last_error = None
            return True
        except Exception as e:
            self._is_available = False
            self._last_error = str(e)
            logger.warning(f"Intrinio not available: {e}")
            return False

    async def get_quote(self, symbol: str) -> QuoteData:
        """
        Get real-time quote from Intrinio

        Uses /securities/{symbol}/prices/realtime endpoint
        """
        try:
            endpoint = f"/securities/{symbol}/prices/realtime"
            data = await self._make_request(endpoint)

            last_price = data.get('last_price')
            bid = data.get('bid_price')
            ask = data.get('ask_price')
            volume = data.get('total_volume', 0)

            # Calculate change
            prev_close = data.get('previous_close')
            change = None
            change_pct = None
            if last_price and prev_close:
                change = last_price - prev_close
                change_pct = (change / prev_close) * 100

            return QuoteData(
                symbol=symbol,
                bid=bid,
                ask=ask,
                last=last_price,
                volume=volume,
                timestamp=datetime.now(),  # Intrinio returns realtime
                bid_size=data.get('bid_size'),
                ask_size=data.get('ask_size'),
                high=data.get('high_price'),
                low=data.get('low_price'),
                open=data.get('open_price'),
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
        Get options chain from Intrinio

        Uses /options/chain/{symbol} endpoint
        Intrinio has excellent options data with Greeks included!
        """
        try:
            endpoint = f"/options/chain/{symbol}"
            params = {
                'page_size': 10000,
            }

            if expiration:
                params['expiration'] = expiration.strftime('%Y-%m-%d')

            data = await self._make_request(endpoint, params)

            if 'chain' not in data:
                logger.warning(f"No options data for {symbol}")
                return []

            options_list = []
            for contract in data['chain']:
                option_symbol = contract.get('code')
                strike = contract.get('strike')
                exp_date_str = contract.get('expiration')
                option_type = contract.get('type', '').lower()  # 'call' or 'put'

                if not all([option_symbol, strike, exp_date_str, option_type]):
                    continue

                exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d').date()

                # Intrinio provides Greeks! This is a major advantage
                prices = contract.get('prices', {})

                options_list.append(OptionsData(
                    symbol=option_symbol,
                    underlying_symbol=symbol,
                    strike=strike,
                    expiration=exp_date,
                    option_type=option_type,
                    bid=prices.get('bid'),
                    ask=prices.get('ask'),
                    last=prices.get('last'),
                    volume=prices.get('volume', 0),
                    open_interest=prices.get('open_interest', 0),
                    implied_volatility=prices.get('implied_volatility'),
                    delta=prices.get('delta'),
                    gamma=prices.get('gamma'),
                    theta=prices.get('theta'),
                    vega=prices.get('vega'),
                    rho=prices.get('rho'),
                    timestamp=datetime.now(),
                ))

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
        Get historical data from Intrinio

        Uses /securities/{symbol}/prices endpoint
        """
        try:
            endpoint = f"/securities/{symbol}/prices"
            params = {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'frequency': 'daily' if interval == '1d' else 'intraday',
                'page_size': 10000,
            }

            data = await self._make_request(endpoint, params)

            if 'prices' not in data:
                logger.warning(f"No historical data for {symbol}")
                return []

            bars = []
            for bar in data['prices']:
                bars.append(HistoricalBar(
                    timestamp=datetime.strptime(bar['date'], '%Y-%m-%d'),
                    open=bar['open'],
                    high=bar['high'],
                    low=bar['low'],
                    close=bar['close'],
                    volume=bar.get('volume', 0),
                    vwap=None,  # Not provided by Intrinio
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
        Get company information from Intrinio

        Uses /companies/{symbol} endpoint
        """
        try:
            endpoint = f"/companies/{symbol}"
            data = await self._make_request(endpoint)

            return {
                'symbol': symbol,
                'name': data.get('name'),
                'lei': data.get('lei'),
                'cik': data.get('cik'),
                'ticker': data.get('ticker'),
                'composite_ticker': data.get('composite_ticker'),
                'exchange': data.get('exchange'),
                'sector': data.get('sector'),
                'industry_category': data.get('industry_category'),
                'industry_group': data.get('industry_group'),
                'template': data.get('template'),
                'standardized_active': data.get('standardized_active'),
                'first_stock_price_date': data.get('first_stock_price_date'),
                'last_stock_price_date': data.get('last_stock_price_date'),
                'last_sec_filing_date': data.get('last_sec_filing_date'),
                'fax_number': data.get('fax_number'),
                'business_address': data.get('business_address'),
                'mailing_address': data.get('mailing_address'),
                'business_phone_no': data.get('business_phone_no'),
                'hq_address1': data.get('hq_address1'),
                'hq_address2': data.get('hq_address2'),
                'hq_address_city': data.get('hq_address_city'),
                'hq_address_postal_code': data.get('hq_address_postal_code'),
                'entity_status': data.get('entity_status'),
                'number_of_employees': data.get('number_of_employees'),
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
