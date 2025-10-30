"""
Position Enrichment Service
Calculates Greeks, metrics, and enriches positions with real-time data
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import yfinance as yf
import numpy as np

from .position_manager import PositionManager, StockPosition, OptionPosition
from ..analytics.greeks_calculator import GreeksCalculator
from ..analytics import black_scholes

logger = logging.getLogger(__name__)


class PositionEnrichmentService:
    """Service to enrich positions with real-time data and calculated metrics"""

    def __init__(self, position_manager: PositionManager):
        self.position_manager = position_manager
        self.greeks_calculator = GreeksCalculator()
    
    def get_risk_free_rate(self) -> float:
        """Get current risk-free rate (10-year Treasury)"""
        try:
            tnx = yf.Ticker("^TNX")
            rate = tnx.info.get('regularMarketPrice', 4.5) / 100  # Convert to decimal
            return rate
        except:
            return 0.045  # Default 4.5%
    
    def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch stock data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1mo")
            
            if hist.empty:
                logger.warning(f"No historical data for {symbol}")
                return {}
            
            current_price = hist['Close'].iloc[-1]
            
            # Calculate historical volatility (30-day)
            returns = np.log(hist['Close'] / hist['Close'].shift(1))
            hv = returns.std() * np.sqrt(252)  # Annualized
            
            return {
                'current_price': float(current_price),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'market_cap': info.get('marketCap'),
                'historical_volatility': float(hv) if not np.isnan(hv) else None,
                'beta': info.get('beta'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'average_volume': info.get('averageVolume'),
                'analyst_target_price': info.get('targetMeanPrice'),
                'analyst_count': info.get('numberOfAnalystOpinions'),
                'recommendation': info.get('recommendationKey')
            }
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            return {}
    
    def get_option_chain_data(self, symbol: str, expiration_date: str, strike: float, option_type: str) -> Dict[str, Any]:
        """Fetch option chain data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get option chain for the expiration date
            opt_chain = ticker.option_chain(expiration_date)
            
            # Select calls or puts
            if option_type.lower() == 'call':
                options = opt_chain.calls
            else:
                options = opt_chain.puts
            
            # Find the specific strike
            option_data = options[options['strike'] == strike]
            
            if option_data.empty:
                logger.warning(f"No option data for {symbol} {strike} {option_type} {expiration_date}")
                return {}
            
            row = option_data.iloc[0]
            
            return {
                'last_price': float(row['lastPrice']),
                'bid': float(row['bid']),
                'ask': float(row['ask']),
                'volume': int(row['volume']) if not np.isnan(row['volume']) else 0,
                'open_interest': int(row['openInterest']) if not np.isnan(row['openInterest']) else 0,
                'implied_volatility': float(row['impliedVolatility']) if not np.isnan(row['impliedVolatility']) else None
            }
        except Exception as e:
            logger.error(f"Error fetching option data for {symbol}: {e}")
            return {}
    
    def calculate_option_greeks(
        self,
        underlying_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        option_type: str,
        risk_free_rate: float = None
    ) -> Dict[str, float]:
        """Calculate option Greeks using Black-Scholes"""
        if risk_free_rate is None:
            risk_free_rate = self.get_risk_free_rate()
        
        try:
            # Calculate Greeks (map to calculator's signature)
            greeks = self.greeks_calculator.calculate_greeks(
                option_type=option_type,
                underlying_price=underlying_price,
                strike=strike,
                time_to_expiry=time_to_expiry,
                iv=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=0.0
            )

            # Convert dataclass to dict
            return {
                'delta': greeks.delta,
                'gamma': greeks.gamma,
                'theta': greeks.theta,
                'vega': greeks.vega,
                'rho': greeks.rho
            }
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return {}
    
    def enrich_stock_position(self, position: StockPosition) -> StockPosition:
        """Enrich stock position with real-time data"""
        try:
            # Get market data
            market_data = self.get_stock_data(position.symbol)
            
            if not market_data:
                logger.warning(f"No market data available for {position.symbol}")
                return position
            
            # Update position with market data
            position.calculate_metrics(market_data)
            
            # Add additional metrics
            position.pe_ratio = market_data.get('pe_ratio')
            position.dividend_yield = market_data.get('dividend_yield')
            position.market_cap = market_data.get('market_cap')
            position.analyst_target_price = market_data.get('analyst_target_price')
            position.analyst_count = market_data.get('analyst_count')
            position.analyst_consensus = market_data.get('recommendation')
            
            logger.info(f"Enriched stock position: {position.symbol} @ ${position.current_price}")
            
        except Exception as e:
            logger.error(f"Error enriching stock position {position.symbol}: {e}")
        
        return position
    
    def enrich_option_position(self, position: OptionPosition) -> OptionPosition:
        """Enrich option position with real-time data and Greeks"""
        try:
            # Get underlying stock data
            market_data = self.get_stock_data(position.symbol)
            
            if not market_data:
                logger.warning(f"No market data available for {position.symbol}")
                return position
            
            underlying_price = market_data.get('current_price')
            hv = market_data.get('historical_volatility', 0.3)  # Default 30%
            
            # Get option chain data
            option_data = self.get_option_chain_data(
                position.symbol,
                position.expiration_date,
                position.strike,
                position.option_type
            )
            
            # If no option data, use Black-Scholes to estimate
            if not option_data and underlying_price:
                time_to_expiry = position.time_to_expiry()
                risk_free_rate = self.get_risk_free_rate()

                # Estimate option price using Black-Scholes
                bs_price = black_scholes.black_scholes_price(
                    option_type=position.option_type,
                    underlying_price=underlying_price,
                    strike=position.strike,
                    time_to_expiry=time_to_expiry,
                    volatility=hv,
                    risk_free_rate=risk_free_rate
                )

                option_data = {
                    'last_price': bs_price,
                    'implied_volatility': hv
                }
            
            # Calculate metrics
            position.calculate_metrics(market_data, option_data)
            
            # Calculate Greeks if we have IV
            if position.implied_volatility and underlying_price:
                greeks = self.calculate_option_greeks(
                    underlying_price=underlying_price,
                    strike=position.strike,
                    time_to_expiry=position.time_to_expiry(),
                    volatility=position.implied_volatility,
                    option_type=position.option_type
                )
                
                position.delta = greeks.get('delta')
                position.gamma = greeks.get('gamma')
                position.theta = greeks.get('theta')
                position.vega = greeks.get('vega')
                position.rho = greeks.get('rho')
            
            # Calculate IV rank and percentile if we have HV
            if position.implied_volatility and hv:
                position.historical_volatility = hv
                # Simplified IV rank (would need historical IV data for accurate calculation)
                position.iv_rank = min(100, (position.implied_volatility / hv) * 50)
            
            # Calculate probability of profit (simplified)
            if position.delta:
                # For long calls/puts, rough estimate based on delta
                if position.option_type.lower() == 'call':
                    position.probability_of_profit = abs(position.delta) * 100
                else:
                    position.probability_of_profit = (1 - abs(position.delta)) * 100
            
            logger.info(f"Enriched option position: {position.symbol} {position.strike} {position.option_type} @ ${position.current_price}")
            
        except Exception as e:
            logger.error(f"Error enriching option position {position.symbol}: {e}")
        
        return position
    
    def enrich_all_positions(self) -> Dict[str, Any]:
        """Enrich all positions with real-time data"""
        results = {
            'stocks_enriched': 0,
            'options_enriched': 0,
            'errors': []
        }
        
        # Enrich stock positions
        for position in self.position_manager.get_all_stock_positions():
            try:
                self.enrich_stock_position(position)
                results['stocks_enriched'] += 1
            except Exception as e:
                results['errors'].append(f"Stock {position.symbol}: {str(e)}")
        
        # Enrich option positions
        for position in self.position_manager.get_all_option_positions():
            try:
                self.enrich_option_position(position)
                results['options_enriched'] += 1
            except Exception as e:
                results['errors'].append(f"Option {position.symbol}: {str(e)}")
        
        # Save enriched positions
        self.position_manager.save_positions()
        
        logger.info(f"Enriched {results['stocks_enriched']} stocks, {results['options_enriched']} options")
        return results
    
    def get_enriched_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary with enriched data"""
        # First enrich all positions
        self.enrich_all_positions()
        
        # Get base summary
        summary = self.position_manager.get_portfolio_summary()
        
        # Calculate total current values
        total_stock_current_value = sum(
            (p.current_price or p.entry_price) * p.quantity
            for p in self.position_manager.get_all_stock_positions()
        )
        
        total_option_current_value = sum(
            (p.current_price or p.premium_paid) * p.quantity * 100
            for p in self.position_manager.get_all_option_positions()
        )
        
        # Calculate total P&L
        total_stock_pnl = sum(
            p.unrealized_pnl or 0
            for p in self.position_manager.get_all_stock_positions()
        )
        
        total_option_pnl = sum(
            p.unrealized_pnl or 0
            for p in self.position_manager.get_all_option_positions()
        )
        
        # Add enriched metrics
        summary.update({
            'total_stock_current_value': total_stock_current_value,
            'total_option_current_value': total_option_current_value,
            'total_current_value': total_stock_current_value + total_option_current_value,
            'total_stock_pnl': total_stock_pnl,
            'total_option_pnl': total_option_pnl,
            'total_pnl': total_stock_pnl + total_option_pnl,
            'total_pnl_pct': ((total_stock_pnl + total_option_pnl) / (summary['total_portfolio_value'] or 1)) * 100
        })
        
        return summary

