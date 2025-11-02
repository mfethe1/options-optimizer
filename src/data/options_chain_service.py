"""
Professional Options Chain Data Service

Provides real-time options chain data with Greeks, volume, OI, and IV.
Designed for institutional-grade performance and accuracy.
"""
import asyncio
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from decimal import Decimal
import yfinance as yf
import numpy as np
from scipy.stats import norm
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OptionStrike:
    """Single option strike data"""
    strike: float
    expiration: date

    # Call data
    call_bid: Optional[float]
    call_ask: Optional[float]
    call_last: Optional[float]
    call_volume: int
    call_open_interest: int
    call_iv: Optional[float]

    # Put data
    put_bid: Optional[float]
    put_ask: Optional[float]
    put_last: Optional[float]
    put_volume: int
    put_open_interest: int
    put_iv: Optional[float]

    # Greeks (calculated)
    call_delta: Optional[float] = None
    call_gamma: Optional[float] = None
    call_theta: Optional[float] = None
    call_vega: Optional[float] = None

    put_delta: Optional[float] = None
    put_gamma: Optional[float] = None
    put_theta: Optional[float] = None
    put_vega: Optional[float] = None

    # Metadata
    in_the_money_call: bool = False
    in_the_money_put: bool = False
    unusual_volume_call: bool = False
    unusual_volume_put: bool = False


@dataclass
class OptionsChain:
    """Complete options chain for a symbol"""
    symbol: str
    current_price: float
    price_change: float
    price_change_pct: float

    # Volatility metrics
    iv_rank: Optional[float]  # 0-100, current IV vs 52-week range
    iv_percentile: Optional[float]  # 0-100, % of days IV was below current
    hv_20: Optional[float]  # 20-day historical volatility
    hv_30: Optional[float]  # 30-day historical volatility

    # Available expirations
    expirations: List[date]

    # Strikes by expiration
    strikes: Dict[date, List[OptionStrike]]

    # Max pain (highest OI)
    max_pain: Optional[float]

    # Put/Call ratio
    put_call_ratio_volume: Optional[float]
    put_call_ratio_oi: Optional[float]

    # Last updated
    last_updated: datetime
    data_source: str


class OptionsChainService:
    """
    Service for fetching and managing options chain data.

    Performance targets:
    - Initial load: < 500ms
    - Real-time updates: < 100ms
    - Greeks calculation: < 50ms
    """

    def __init__(self):
        self._cache: Dict[str, OptionsChain] = {}
        self._cache_ttl = 60  # 60 seconds cache
        self._last_fetch: Dict[str, datetime] = {}

    async def get_options_chain(
        self,
        symbol: str,
        expiration: Optional[date] = None,
        force_refresh: bool = False
    ) -> OptionsChain:
        """
        Get complete options chain for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            expiration: Specific expiration to focus on (optional)
            force_refresh: Force refresh from data source

        Returns:
            Complete options chain with Greeks
        """
        cache_key = f"{symbol}_{expiration}" if expiration else symbol

        # Check cache
        if not force_refresh and cache_key in self._cache:
            last_fetch = self._last_fetch.get(cache_key)
            if last_fetch and (datetime.now() - last_fetch).seconds < self._cache_ttl:
                logger.info(f"Returning cached options chain for {symbol}")
                return self._cache[cache_key]

        # Fetch fresh data
        logger.info(f"Fetching options chain for {symbol}")
        chain = await self._fetch_options_chain(symbol, expiration)

        # Calculate Greeks
        chain = await self._calculate_greeks(chain)

        # Identify unusual activity
        chain = await self._identify_unusual_activity(chain)

        # Calculate max pain
        chain = await self._calculate_max_pain(chain)

        # Cache result
        self._cache[cache_key] = chain
        self._last_fetch[cache_key] = datetime.now()

        return chain

    async def _fetch_options_chain(
        self,
        symbol: str,
        expiration: Optional[date] = None
    ) -> OptionsChain:
        """
        Fetch raw options chain data from data provider.

        TODO: Replace with professional data provider (Polygon.io, Intrinio)
        Currently using yfinance as placeholder.
        """
        try:
            ticker = yf.Ticker(symbol)

            # Get current stock info
            info = ticker.info
            current_price = info.get('regularMarketPrice', info.get('currentPrice', 0))
            prev_close = info.get('previousClose', current_price)
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close * 100) if prev_close else 0

            # Get available expirations
            expirations = ticker.options
            if not expirations:
                raise ValueError(f"No options data available for {symbol}")

            expiration_dates = [datetime.strptime(exp, '%Y-%m-%d').date() for exp in expirations]

            # Fetch options data for each expiration
            strikes_by_exp: Dict[date, List[OptionStrike]] = {}

            for exp_str, exp_date in zip(expirations, expiration_dates):
                # Skip if specific expiration requested and this isn't it
                if expiration and exp_date != expiration:
                    continue

                opt = ticker.option_chain(exp_str)
                calls = opt.calls
                puts = opt.puts

                # Merge calls and puts by strike
                strikes = []
                all_strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))

                for strike in all_strikes:
                    call_data = calls[calls['strike'] == strike]
                    put_data = puts[puts['strike'] == strike]

                    # Extract call data
                    call_bid = call_data['bid'].iloc[0] if len(call_data) > 0 else None
                    call_ask = call_data['ask'].iloc[0] if len(call_data) > 0 else None
                    call_last = call_data['lastPrice'].iloc[0] if len(call_data) > 0 else None
                    call_volume = int(call_data['volume'].iloc[0]) if len(call_data) > 0 and pd.notna(call_data['volume'].iloc[0]) else 0
                    call_oi = int(call_data['openInterest'].iloc[0]) if len(call_data) > 0 and pd.notna(call_data['openInterest'].iloc[0]) else 0
                    call_iv = call_data['impliedVolatility'].iloc[0] if len(call_data) > 0 and pd.notna(call_data['impliedVolatility'].iloc[0]) else None

                    # Extract put data
                    put_bid = put_data['bid'].iloc[0] if len(put_data) > 0 else None
                    put_ask = put_data['ask'].iloc[0] if len(put_data) > 0 else None
                    put_last = put_data['lastPrice'].iloc[0] if len(put_data) > 0 else None
                    put_volume = int(put_data['volume'].iloc[0]) if len(put_data) > 0 and pd.notna(put_data['volume'].iloc[0]) else 0
                    put_oi = int(put_data['openInterest'].iloc[0]) if len(put_data) > 0 and pd.notna(put_data['openInterest'].iloc[0]) else 0
                    put_iv = put_data['impliedVolatility'].iloc[0] if len(put_data) > 0 and pd.notna(put_data['impliedVolatility'].iloc[0]) else None

                    # Create strike object
                    strike_obj = OptionStrike(
                        strike=float(strike),
                        expiration=exp_date,
                        call_bid=float(call_bid) if call_bid and pd.notna(call_bid) else None,
                        call_ask=float(call_ask) if call_ask and pd.notna(call_ask) else None,
                        call_last=float(call_last) if call_last and pd.notna(call_last) else None,
                        call_volume=call_volume,
                        call_open_interest=call_oi,
                        call_iv=float(call_iv) if call_iv and pd.notna(call_iv) else None,
                        put_bid=float(put_bid) if put_bid and pd.notna(put_bid) else None,
                        put_ask=float(put_ask) if put_ask and pd.notna(put_ask) else None,
                        put_last=float(put_last) if put_last and pd.notna(put_last) else None,
                        put_volume=put_volume,
                        put_open_interest=put_oi,
                        put_iv=float(put_iv) if put_iv and pd.notna(put_iv) else None,
                        in_the_money_call=strike < current_price,
                        in_the_money_put=strike > current_price
                    )

                    strikes.append(strike_obj)

                strikes_by_exp[exp_date] = strikes

            # Calculate IV metrics
            iv_rank, iv_percentile = await self._calculate_iv_metrics(ticker)
            hv_20, hv_30 = await self._calculate_historical_volatility(ticker)

            # Create chain object
            chain = OptionsChain(
                symbol=symbol.upper(),
                current_price=current_price,
                price_change=price_change,
                price_change_pct=price_change_pct,
                iv_rank=iv_rank,
                iv_percentile=iv_percentile,
                hv_20=hv_20,
                hv_30=hv_30,
                expirations=expiration_dates if not expiration else [expiration],
                strikes=strikes_by_exp,
                max_pain=None,  # Calculated later
                put_call_ratio_volume=None,  # Calculated later
                put_call_ratio_oi=None,  # Calculated later
                last_updated=datetime.now(),
                data_source="yfinance"  # TODO: Replace with professional provider
            )

            return chain

        except Exception as e:
            logger.error(f"Error fetching options chain for {symbol}: {e}", exc_info=True)
            raise

    async def _calculate_greeks(self, chain: OptionsChain) -> OptionsChain:
        """Calculate Greeks for all strikes using Black-Scholes model"""
        risk_free_rate = 0.045  # Current risk-free rate (approximate)

        for exp_date, strikes in chain.strikes.items():
            days_to_exp = (exp_date - date.today()).days
            time_to_exp = max(days_to_exp / 365.0, 0.0001)  # Avoid division by zero

            for strike in strikes:
                S = chain.current_price
                K = strike.strike
                T = time_to_exp
                r = risk_free_rate

                # Call Greeks
                if strike.call_iv:
                    sigma = strike.call_iv
                    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                    d2 = d1 - sigma * np.sqrt(T)

                    strike.call_delta = norm.cdf(d1)
                    strike.call_gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                    strike.call_theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) -
                                       r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
                    strike.call_vega = S * norm.pdf(d1) * np.sqrt(T) / 100

                # Put Greeks
                if strike.put_iv:
                    sigma = strike.put_iv
                    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                    d2 = d1 - sigma * np.sqrt(T)

                    strike.put_delta = -norm.cdf(-d1)
                    strike.put_gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                    strike.put_theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) +
                                      r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
                    strike.put_vega = S * norm.pdf(d1) * np.sqrt(T) / 100

        return chain

    async def _calculate_iv_metrics(self, ticker) -> tuple:
        """Calculate IV Rank and IV Percentile"""
        try:
            # Get historical data
            hist = ticker.history(period="1y")

            # Calculate historical volatility for each day
            # (This is simplified - real implementation would use actual IV history)
            returns = hist['Close'].pct_change()
            rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)

            if len(rolling_vol) > 0:
                current_vol = rolling_vol.iloc[-1]
                min_vol = rolling_vol.min()
                max_vol = rolling_vol.max()

                # IV Rank
                iv_rank = ((current_vol - min_vol) / (max_vol - min_vol) * 100) if max_vol > min_vol else 50

                # IV Percentile
                iv_percentile = (rolling_vol < current_vol).sum() / len(rolling_vol) * 100

                return round(iv_rank, 2), round(iv_percentile, 2)

            return None, None

        except Exception as e:
            logger.warning(f"Could not calculate IV metrics: {e}")
            return None, None

    async def _calculate_historical_volatility(self, ticker) -> tuple:
        """Calculate 20-day and 30-day historical volatility"""
        try:
            hist = ticker.history(period="3mo")
            returns = hist['Close'].pct_change()

            hv_20 = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252) * 100
            hv_30 = returns.rolling(window=30).std().iloc[-1] * np.sqrt(252) * 100

            return round(hv_20, 2), round(hv_30, 2)
        except Exception as e:
            logger.warning(f"Could not calculate historical volatility: {e}")
            return None, None

    async def _identify_unusual_activity(self, chain: OptionsChain) -> OptionsChain:
        """Identify strikes with unusual volume or OI"""
        for exp_date, strikes in chain.strikes.items():
            # Calculate average volume and OI
            call_volumes = [s.call_volume for s in strikes if s.call_volume > 0]
            put_volumes = [s.put_volume for s in strikes if s.put_volume > 0]
            call_ois = [s.call_open_interest for s in strikes if s.call_open_interest > 0]
            put_ois = [s.put_open_interest for s in strikes if s.put_open_interest > 0]

            avg_call_vol = np.mean(call_volumes) if call_volumes else 0
            avg_put_vol = np.mean(put_volumes) if put_volumes else 0

            # Flag unusual activity (> 3x average)
            for strike in strikes:
                if avg_call_vol > 0 and strike.call_volume > avg_call_vol * 3:
                    strike.unusual_volume_call = True
                if avg_put_vol > 0 and strike.put_volume > avg_put_vol * 3:
                    strike.unusual_volume_put = True

        return chain

    async def _calculate_max_pain(self, chain: OptionsChain) -> OptionsChain:
        """Calculate max pain (strike with highest total OI)"""
        try:
            # Get most liquid expiration (usually nearest)
            if not chain.strikes:
                return chain

            nearest_exp = min(chain.strikes.keys())
            strikes = chain.strikes[nearest_exp]

            # Calculate max pain
            max_pain_strike = None
            max_oi = 0

            for strike in strikes:
                total_oi = strike.call_open_interest + strike.put_open_interest
                if total_oi > max_oi:
                    max_oi = total_oi
                    max_pain_strike = strike.strike

            chain.max_pain = max_pain_strike

            # Calculate put/call ratios
            total_call_vol = sum(s.call_volume for s in strikes)
            total_put_vol = sum(s.put_volume for s in strikes)
            total_call_oi = sum(s.call_open_interest for s in strikes)
            total_put_oi = sum(s.put_open_interest for s in strikes)

            chain.put_call_ratio_volume = round(total_put_vol / total_call_vol, 3) if total_call_vol > 0 else None
            chain.put_call_ratio_oi = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else None

        except Exception as e:
            logger.warning(f"Could not calculate max pain: {e}")

        return chain

    async def stream_options_updates(
        self,
        symbol: str,
        expiration: date,
        callback
    ):
        """
        Stream real-time options chain updates via WebSocket.

        TODO: Implement with professional data provider WebSocket feed
        """
        while True:
            try:
                chain = await self.get_options_chain(symbol, expiration, force_refresh=True)
                await callback(chain)
                await asyncio.sleep(5)  # Update every 5 seconds
            except Exception as e:
                logger.error(f"Error streaming options updates: {e}")
                await asyncio.sleep(10)


# Global instance
options_chain_service = OptionsChainService()
