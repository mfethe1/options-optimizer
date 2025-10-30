"""
Earnings Calendar Service - Track upcoming earnings and analyze releases
"""
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import requests
import pandas as pd

logger = logging.getLogger(__name__)


class EarningsService:
    """
    Earnings calendar and analysis service.
    Supports multiple providers: Finnhub, Polygon, FMP (FinancialModelingPrep)
    """
    
    def __init__(self):
        self.cache_dir = Path("data/research/earnings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # API keys
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        self.polygon_api_key = os.getenv('POLYGON_API_KEY')
        self.fmp_api_key = os.getenv('FMP_API_KEY')
        
        # Provider priority
        self.providers = []
        if self.finnhub_api_key:
            self.providers.append('finnhub')
        if self.polygon_api_key:
            self.providers.append('polygon')
        if self.fmp_api_key:
            self.providers.append('fmp')
        
        logger.info(f"Earnings service initialized with providers: {self.providers}")
    
    def get_earnings_calendar(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get earnings calendar for date range or specific symbol.
        
        Args:
            start_date: Start date (YYYY-MM-DD), defaults to today
            end_date: End date (YYYY-MM-DD), defaults to +30 days
            symbol: Optional symbol filter
            
        Returns:
            List of earnings events
        """
        if not start_date:
            start_date = datetime.now().strftime('%Y-%m-%d')
        if not end_date:
            end_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        
        # Check cache
        cache_file = self.cache_dir / f"calendar_{start_date}_{end_date}.json"
        if cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age < timedelta(hours=12):  # Cache for 12 hours
                logger.info(f"Using cached earnings calendar")
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                    if symbol:
                        return [e for e in cached if e['symbol'] == symbol.upper()]
                    return cached
        
        # Fetch from providers
        earnings = []
        for provider in self.providers:
            try:
                if provider == 'finnhub':
                    earnings = self._fetch_finnhub_calendar(start_date, end_date)
                elif provider == 'polygon':
                    earnings = self._fetch_polygon_calendar(start_date, end_date)
                elif provider == 'fmp':
                    earnings = self._fetch_fmp_calendar(start_date, end_date)
                
                if earnings:
                    logger.info(f"Fetched {len(earnings)} earnings from {provider}")
                    break
            except Exception as e:
                logger.warning(f"Error fetching from {provider}: {e}")
                continue
        
        # Cache results
        if earnings:
            with open(cache_file, 'w') as f:
                json.dump(earnings, f, indent=2)
        
        # Filter by symbol if requested
        if symbol:
            earnings = [e for e in earnings if e['symbol'] == symbol.upper()]
        
        return earnings
    
    def _fetch_finnhub_calendar(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Fetch from Finnhub"""
        url = "https://finnhub.io/api/v1/calendar/earnings"
        params = {
            'from': start_date,
            'to': end_date,
            'token': self.finnhub_api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        earnings = []
        for item in data.get('earningsCalendar', []):
            earnings.append({
                'symbol': item.get('symbol'),
                'date': item.get('date'),
                'eps_estimate': item.get('epsEstimate'),
                'eps_actual': item.get('epsActual'),
                'revenue_estimate': item.get('revenueEstimate'),
                'revenue_actual': item.get('revenueActual'),
                'quarter': item.get('quarter'),
                'year': item.get('year'),
                'source': 'finnhub'
            })
        
        return earnings
    
    def _fetch_polygon_calendar(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Fetch from Polygon"""
        # Polygon doesn't have a direct earnings calendar endpoint
        # Would need to use their reference data or other endpoints
        logger.warning("Polygon earnings calendar not yet implemented")
        return []
    
    def _fetch_fmp_calendar(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Fetch from Financial Modeling Prep"""
        url = "https://financialmodelingprep.com/api/v3/earning_calendar"
        params = {
            'from': start_date,
            'to': end_date,
            'apikey': self.fmp_api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        earnings = []
        for item in data:
            earnings.append({
                'symbol': item.get('symbol'),
                'date': item.get('date'),
                'eps_estimate': item.get('epsEstimated'),
                'eps_actual': item.get('eps'),
                'revenue_estimate': item.get('revenueEstimated'),
                'revenue_actual': item.get('revenue'),
                'time': item.get('time'),  # BMO, AMC, etc.
                'source': 'fmp'
            })
        
        return earnings
    
    def get_next_earnings(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get next earnings date for a symbol"""
        calendar = self.get_earnings_calendar(symbol=symbol)
        
        if not calendar:
            return None
        
        # Find next future earnings
        today = datetime.now().date()
        future_earnings = [
            e for e in calendar 
            if datetime.strptime(e['date'], '%Y-%m-%d').date() >= today
        ]
        
        if not future_earnings:
            return None
        
        # Sort by date and return earliest
        future_earnings.sort(key=lambda x: x['date'])
        return future_earnings[0]
    
    def calculate_implied_move(
        self, 
        symbol: str, 
        earnings_date: str,
        current_price: float,
        atm_straddle_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate implied earnings move from options pricing.
        
        Args:
            symbol: Stock symbol
            earnings_date: Earnings date (YYYY-MM-DD)
            current_price: Current stock price
            atm_straddle_price: Price of ATM straddle (call + put)
            
        Returns:
            Implied move analysis
        """
        if not atm_straddle_price:
            # Would need to fetch from options data
            logger.warning("ATM straddle price not provided, cannot calculate implied move")
            return {
                'symbol': symbol,
                'earnings_date': earnings_date,
                'implied_move_pct': None,
                'implied_move_dollars': None,
                'upper_range': None,
                'lower_range': None
            }
        
        # Implied move = straddle price / stock price
        implied_move_pct = (atm_straddle_price / current_price) * 100
        implied_move_dollars = atm_straddle_price
        
        return {
            'symbol': symbol,
            'earnings_date': earnings_date,
            'current_price': current_price,
            'atm_straddle_price': atm_straddle_price,
            'implied_move_pct': implied_move_pct,
            'implied_move_dollars': implied_move_dollars,
            'upper_range': current_price + implied_move_dollars,
            'lower_range': current_price - implied_move_dollars,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_historical_earnings_moves(self, symbol: str, quarters: int = 4) -> List[Dict[str, Any]]:
        """Get historical earnings moves for comparison"""
        # This would require historical price data around earnings dates
        # Placeholder for now
        logger.warning("Historical earnings moves not yet implemented")
        return []
    
    def analyze_earnings_risk(
        self, 
        symbol: str, 
        position_type: str,
        strike: Optional[float] = None,
        expiration: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze earnings risk for a position.
        
        Args:
            symbol: Stock symbol
            position_type: 'stock' or 'option'
            strike: Option strike (if option)
            expiration: Option expiration (if option)
            
        Returns:
            Risk analysis
        """
        next_earnings = self.get_next_earnings(symbol)
        
        if not next_earnings:
            return {
                'has_earnings_risk': False,
                'message': 'No upcoming earnings found'
            }
        
        earnings_date = datetime.strptime(next_earnings['date'], '%Y-%m-%d').date()
        today = datetime.now().date()
        days_to_earnings = (earnings_date - today).days
        
        risk_level = 'low'
        if days_to_earnings <= 7:
            risk_level = 'critical'
        elif days_to_earnings <= 14:
            risk_level = 'high'
        elif days_to_earnings <= 30:
            risk_level = 'medium'
        
        analysis = {
            'has_earnings_risk': True,
            'earnings_date': next_earnings['date'],
            'days_to_earnings': days_to_earnings,
            'risk_level': risk_level,
            'eps_estimate': next_earnings.get('eps_estimate'),
            'revenue_estimate': next_earnings.get('revenue_estimate')
        }
        
        # Additional analysis for options
        if position_type == 'option' and expiration:
            exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()
            expires_before_earnings = exp_date < earnings_date
            
            analysis['expires_before_earnings'] = expires_before_earnings
            analysis['recommendation'] = (
                'Position expires before earnings - lower IV crush risk' 
                if expires_before_earnings 
                else 'Position exposed to earnings - high IV crush risk'
            )
        
        return analysis
    
    def save_to_parquet(self, earnings: List[Dict[str, Any]], filename: str):
        """Save earnings data to parquet"""
        if not earnings:
            return
        
        df = pd.DataFrame(earnings)
        filepath = self.cache_dir / filename
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved {len(earnings)} earnings to {filepath}")

