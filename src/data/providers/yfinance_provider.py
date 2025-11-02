"""
yFinance Data Provider

FREE retail-grade data provider - FALLBACK ONLY.

WARNING: Not suitable for professional/production use:
- Unofficial Yahoo Finance API
- No guarantees of availability or accuracy
- Subject to rate limiting
- Can break without notice
- 15-20 minute data delays
- No support or SLA

Use only when institutional providers (Polygon, Intrinio) are unavailable.
"""
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
import yfinance as yf

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


class YFinanceProvider(DataProvider):
    """yFinance data provider - FALLBACK ONLY, retail grade"""

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(api_key, config)
        logger.warning(
            "YFinanceProvider initialized - THIS IS A FALLBACK PROVIDER. "
            "Data quality is RETAIL-GRADE. Configure Polygon or Intrinio for production use."
        )

    @property
    def provider_type(self) -> DataProviderType:
        return DataProviderType.YFINANCE

    @property
    def data_quality(self) -> DataQuality:
        return DataQuality.RETAIL

    @property
    def rate_limit(self) -> int:
        return 1  # Very conservative to avoid Yahoo blocking

    async def check_availability(self) -> bool:
        """Check if yfinance is available"""
        try:
            # Test with a simple quote request
            await self.get_quote('AAPL')
            self._is_available = True
            self._last_error = None
            logger.info("yFinance is available (FALLBACK MODE)")
            return True
        except Exception as e:
            self._is_available = False
            self._last_error = str(e)
            logger.error(f"yFinance not available: {e}")
            return False

    async def get_quote(self, symbol: str) -> QuoteData:
        """
        Get quote from yfinance

        Note: Data is delayed 15-20 minutes
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info:
                raise ProviderError(
                    f"No data returned for {symbol}",
                    self.provider_type
                )

            # yfinance returns current price in various fields
            last = info.get('currentPrice') or info.get('regularMarketPrice')
            prev_close = info.get('previousClose') or info.get('regularMarketPreviousClose')

            change = None
            change_pct = None
            if last and prev_close:
                change = last - prev_close
                change_pct = (change / prev_close) * 100

            return QuoteData(
                symbol=symbol,
                bid=info.get('bid'),
                ask=info.get('ask'),
                last=last,
                volume=info.get('volume', 0),
                timestamp=datetime.now(),  # yfinance doesn't provide exact timestamp
                bid_size=info.get('bidSize'),
                ask_size=info.get('askSize'),
                high=info.get('dayHigh') or info.get('regularMarketDayHigh'),
                low=info.get('dayLow') or info.get('regularMarketDayLow'),
                open=info.get('open') or info.get('regularMarketOpen'),
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
        Get options chain from yfinance

        Note: Data is delayed, Greeks not always available
        """
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options

            if not expirations:
                logger.warning(f"No options data for {symbol}")
                return []

            # If specific expiration requested, use that; otherwise use first available
            if expiration:
                exp_str = expiration.strftime('%Y-%m-%d')
                if exp_str not in expirations:
                    logger.warning(f"Expiration {exp_str} not available for {symbol}")
                    return []
                expirations_to_fetch = [exp_str]
            else:
                # Fetch all expirations (could be slow!)
                expirations_to_fetch = expirations[:5]  # Limit to first 5 to avoid slowness

            options_list = []

            for exp_str in expirations_to_fetch:
                try:
                    chain = ticker.option_chain(exp_str)
                    exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()

                    # Process calls
                    if hasattr(chain, 'calls') and not chain.calls.empty:
                        for _, row in chain.calls.iterrows():
                            options_list.append(OptionsData(
                                symbol=row['contractSymbol'],
                                underlying_symbol=symbol,
                                strike=row['strike'],
                                expiration=exp_date,
                                option_type='call',
                                bid=row.get('bid'),
                                ask=row.get('ask'),
                                last=row.get('lastPrice'),
                                volume=int(row.get('volume', 0)) if row.get('volume') else 0,
                                open_interest=int(row.get('openInterest', 0)) if row.get('openInterest') else 0,
                                implied_volatility=row.get('impliedVolatility'),
                                # yfinance doesn't provide other Greeks
                                timestamp=datetime.now(),
                            ))

                    # Process puts
                    if hasattr(chain, 'puts') and not chain.puts.empty:
                        for _, row in chain.puts.iterrows():
                            options_list.append(OptionsData(
                                symbol=row['contractSymbol'],
                                underlying_symbol=symbol,
                                strike=row['strike'],
                                expiration=exp_date,
                                option_type='put',
                                bid=row.get('bid'),
                                ask=row.get('ask'),
                                last=row.get('lastPrice'),
                                volume=int(row.get('volume', 0)) if row.get('volume') else 0,
                                open_interest=int(row.get('openInterest', 0)) if row.get('openInterest') else 0,
                                implied_volatility=row.get('impliedVolatility'),
                                timestamp=datetime.now(),
                            ))

                except Exception as e:
                    logger.warning(f"Could not fetch options for expiration {exp_str}: {e}")
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
        Get historical data from yfinance
        """
        try:
            ticker = yf.Ticker(symbol)

            # Convert interval to yfinance format
            interval_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '1h': '1h',
                '1d': '1d',
            }

            yf_interval = interval_map.get(interval, '1d')

            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=yf_interval
            )

            if df.empty:
                logger.warning(f"No historical data for {symbol}")
                return []

            bars = []
            for timestamp, row in df.iterrows():
                bars.append(HistoricalBar(
                    timestamp=timestamp.to_pydatetime(),
                    open=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    close=row['Close'],
                    volume=int(row['Volume']),
                    vwap=None,  # Not provided by yfinance
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
        Get company information from yfinance
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info:
                raise ProviderError(
                    f"No company info for {symbol}",
                    self.provider_type
                )

            return {
                'symbol': symbol,
                'name': info.get('longName') or info.get('shortName'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'country': info.get('country'),
                'website': info.get('website'),
                'description': info.get('longBusinessSummary'),
                'employees': info.get('fullTimeEmployees'),
                'exchange': info.get('exchange'),
                'currency': info.get('currency'),
                'market_cap': info.get('marketCap'),
                'shares_outstanding': info.get('sharesOutstanding'),
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
        """No session to close for yfinance"""
        pass
