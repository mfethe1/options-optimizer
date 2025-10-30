"""
Market Data Fetcher - Get real-time stock and option prices
"""
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import yfinance as yf
import numpy as np

logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """Fetch real-time market data for stocks and options"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 60  # seconds
    
    def get_stock_price(self, symbol: str) -> Dict[str, Any]:
        """Get current stock price and data"""
        try:
            ticker = yf.Ticker(symbol)

            # Try to get historical data first
            hist = ticker.history(period='1d')

            if hist.empty:
                logger.warning(f"No historical data for {symbol}, trying 5d period")
                hist = ticker.history(period='5d')

            if hist.empty:
                logger.warning(f"No data available for {symbol}")
                return None

            current_price = hist['Close'].iloc[-1]

            # Get info with error handling
            try:
                info = ticker.info
            except Exception as info_error:
                logger.warning(f"Could not fetch info for {symbol}: {info_error}")
                info = {}

            # Safe get with defaults
            previous_close = info.get('previousClose', current_price)
            if previous_close is None or previous_close == 0:
                previous_close = current_price

            change = float(current_price - previous_close)
            change_pct = float(change / previous_close * 100) if previous_close != 0 else 0.0

            return {
                'symbol': symbol,
                'current_price': float(current_price),
                'open': float(hist['Open'].iloc[-1]) if 'Open' in hist.columns else float(current_price),
                'high': float(hist['High'].iloc[-1]) if 'High' in hist.columns else float(current_price),
                'low': float(hist['Low'].iloc[-1]) if 'Low' in hist.columns else float(current_price),
                'volume': int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0,
                'previous_close': float(previous_close),
                'change': change,
                'change_pct': change_pct,
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.exception(f"Error fetching stock data for {symbol}: {e}")
            return None
    
    def get_option_chain(self, symbol: str, expiration_date: str = None) -> Dict[str, Any]:
        """Get option chain data"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            expirations = ticker.options
            if not expirations:
                logger.warning(f"No options available for {symbol}")
                return None
            
            # Use specified date or nearest
            if expiration_date:
                exp_date = expiration_date
            else:
                exp_date = expirations[0]
            
            # Get option chain
            opt_chain = ticker.option_chain(exp_date)
            
            return {
                'symbol': symbol,
                'expiration_date': exp_date,
                'calls': opt_chain.calls.to_dict('records'),
                'puts': opt_chain.puts.to_dict('records'),
                'available_expirations': list(expirations),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching option chain for {symbol}: {e}")
            return None
    
    def get_option_price(
        self,
        symbol: str,
        option_type: str,
        strike: float,
        expiration_date: str
    ) -> Dict[str, Any]:
        """Get specific option price"""
        try:
            chain = self.get_option_chain(symbol, expiration_date)
            if not chain:
                logger.warning(f"No option chain available for {symbol} {expiration_date}")
                return None

            # Get the right chain (calls or puts)
            options = chain['calls'] if option_type.lower() == 'call' else chain['puts']

            if not options:
                logger.warning(f"No {option_type} options available for {symbol}")
                return None

            # Find the option with matching strike
            for opt in options:
                try:
                    opt_strike = float(opt.get('strike', 0))
                    if abs(opt_strike - strike) < 0.01:
                        return {
                            'symbol': symbol,
                            'option_type': option_type,
                            'strike': strike,
                            'expiration_date': expiration_date,
                            'last_price': opt.get('lastPrice'),
                            'bid': opt.get('bid'),
                            'ask': opt.get('ask'),
                            'volume': opt.get('volume'),
                            'open_interest': opt.get('openInterest'),
                            'implied_volatility': opt.get('impliedVolatility'),
                            'delta': opt.get('delta'),
                            'gamma': opt.get('gamma'),
                            'theta': opt.get('theta'),
                            'vega': opt.get('vega'),
                            'in_the_money': opt.get('inTheMoney'),
                            'timestamp': datetime.now().isoformat()
                        }
                except (ValueError, TypeError) as opt_error:
                    logger.debug(f"Error processing option: {opt_error}")
                    continue

            logger.warning(f"Option not found: {symbol} {strike} {option_type} {expiration_date}")
            return None
        except Exception as e:
            logger.exception(f"Error fetching option price: {e}")
            return None
    
    def get_historical_volatility(self, symbol: str, days: int = 30) -> float:
        """Calculate historical volatility"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f'{days}d')
            
            if len(hist) < 2:
                return None
            
            # Calculate daily returns
            returns = hist['Close'].pct_change().dropna()
            
            # Annualized volatility
            volatility = returns.std() * np.sqrt(252)
            
            return float(volatility)
        except Exception as e:
            logger.error(f"Error calculating historical volatility for {symbol}: {e}")
            return None
    
    def get_batch_stock_prices(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get prices for multiple stocks"""
        results = {}
        for symbol in symbols:
            data = self.get_stock_price(symbol)
            if data:
                results[symbol] = data
        return results
    
    def get_implied_volatility(self, symbol: str, expiration_date: str = None) -> float:
        """Get average implied volatility from options"""
        try:
            chain = self.get_option_chain(symbol, expiration_date)
            if not chain:
                return None
            
            # Get IV from ATM options
            calls = chain['calls']
            
            if not calls:
                return None
            
            # Find ATM options (closest to current price)
            stock_price = self.get_stock_price(symbol)
            if not stock_price:
                return None
            
            current_price = stock_price['current_price']
            
            # Get IVs from options near the money
            ivs = []
            for opt in calls:
                if abs(opt['strike'] - current_price) / current_price < 0.1:  # Within 10%
                    if opt.get('impliedVolatility'):
                        ivs.append(opt['impliedVolatility'])
            
            if ivs:
                return float(np.mean(ivs))
            
            return None
        except Exception as e:
            logger.error(f"Error getting implied volatility for {symbol}: {e}")
            return None
    
    def get_market_data_for_position(
        self, 
        symbol: str, 
        option_type: str = None, 
        strike: float = None, 
        expiration_date: str = None
    ) -> Dict[str, Any]:
        """Get comprehensive market data for a position"""
        data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat()
        }
        
        # Get stock data
        stock_data = self.get_stock_price(symbol)
        if stock_data:
            data['stock'] = stock_data
        
        # Get option data if applicable
        if option_type and strike and expiration_date:
            option_data = self.get_option_price(symbol, option_type, strike, expiration_date)
            if option_data:
                data['option'] = option_data
        
        # Get volatility data
        hv = self.get_historical_volatility(symbol)
        if hv:
            data['historical_volatility'] = hv
        
        iv = self.get_implied_volatility(symbol, expiration_date)
        if iv:
            data['implied_volatility'] = iv
        
        return data

