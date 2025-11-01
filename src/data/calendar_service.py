"""
Economic Calendar Service

Provides earnings calendar, economic events, and volatility implications.
Bloomberg EVTS equivalent - institutional-grade event calendar.
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import date, datetime, timedelta
import aiohttp
from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class EarningsEvent:
    """Earnings announcement event"""
    symbol: str
    company_name: str
    date: date
    time: str  # 'bmo' (before market open), 'amc' (after market close), 'intraday'
    fiscal_quarter: str
    fiscal_year: int

    # Estimates
    eps_estimate: Optional[float] = None
    revenue_estimate: Optional[float] = None

    # Actuals (if already reported)
    eps_actual: Optional[float] = None
    revenue_actual: Optional[float] = None

    # Surprise metrics
    eps_surprise_pct: Optional[float] = None
    revenue_surprise_pct: Optional[float] = None

    # Volatility implications
    historical_move_avg: Optional[float] = None  # Average stock move on earnings (%)
    implied_move: Optional[float] = None  # IV-implied move (%)


@dataclass
class EconomicEvent:
    """Economic data release or Fed event"""
    event_type: str  # 'fed_meeting', 'cpi', 'gdp', 'jobs', 'ppi', 'pce', 'retail_sales', etc.
    name: str
    date: date
    time: str
    importance: str  # 'high', 'medium', 'low'

    # Estimate and actual
    estimate: Optional[str] = None
    actual: Optional[str] = None
    previous: Optional[str] = None

    # Market impact
    market_impact: Optional[str] = None  # 'bullish', 'bearish', 'neutral'
    volatility_expected: Optional[str] = None  # 'high', 'medium', 'low'


@dataclass
class CalendarDay:
    """All events for a specific day"""
    date: date
    earnings_events: List[EarningsEvent]
    economic_events: List[EconomicEvent]

    # Day-level metrics
    total_events: int
    high_importance_count: int
    major_earnings_count: int  # S&P 500 companies


class CalendarService:
    """Service for economic and earnings calendar"""

    def __init__(self):
        # API keys from environment
        self.fmp_api_key = settings.FMP_API_KEY if hasattr(settings, 'FMP_API_KEY') else None
        self.base_url_fmp = "https://financialmodelingprep.com/api/v3"

        # Major economic event schedule (Fed meetings are predictable)
        self.fed_meeting_dates_2025 = [
            date(2025, 1, 29),  # Jan 28-29
            date(2025, 3, 19),  # Mar 18-19
            date(2025, 5, 7),   # May 6-7
            date(2025, 6, 18),  # Jun 17-18
            date(2025, 7, 30),  # Jul 29-30
            date(2025, 9, 17),  # Sep 16-17
            date(2025, 11, 5),  # Nov 4-5
            date(2025, 12, 17), # Dec 16-17
        ]

    async def get_earnings_calendar(
        self,
        from_date: date,
        to_date: date,
        symbols: Optional[List[str]] = None
    ) -> List[EarningsEvent]:
        """
        Get earnings calendar for date range.

        Args:
            from_date: Start date
            to_date: End date
            symbols: Optional list of symbols to filter

        Returns:
            List of earnings events
        """
        if not self.fmp_api_key:
            logger.warning("FMP API key not configured, returning mock data")
            return self._get_mock_earnings_calendar(from_date, to_date, symbols)

        try:
            events = []

            # FMP earnings calendar endpoint
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url_fmp}/earning_calendar"
                params = {
                    'from': from_date.isoformat(),
                    'to': to_date.isoformat(),
                    'apikey': self.fmp_api_key
                }

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        for item in data:
                            symbol = item.get('symbol', '')

                            # Filter by symbols if provided
                            if symbols and symbol not in symbols:
                                continue

                            event = EarningsEvent(
                                symbol=symbol,
                                company_name=item.get('name', symbol),
                                date=datetime.strptime(item.get('date'), '%Y-%m-%d').date(),
                                time=item.get('time', 'amc'),  # 'bmo', 'amc', or time
                                fiscal_quarter=item.get('fiscalQuarter', ''),
                                fiscal_year=item.get('fiscalYear', 0),
                                eps_estimate=item.get('epsEstimate'),
                                revenue_estimate=item.get('revenueEstimate'),
                                eps_actual=item.get('eps'),
                                revenue_actual=item.get('revenue'),
                            )

                            # Calculate surprise if actual data available
                            if event.eps_actual and event.eps_estimate:
                                event.eps_surprise_pct = (
                                    (event.eps_actual - event.eps_estimate) / abs(event.eps_estimate) * 100
                                )
                            if event.revenue_actual and event.revenue_estimate:
                                event.revenue_surprise_pct = (
                                    (event.revenue_actual - event.revenue_estimate) / event.revenue_estimate * 100
                                )

                            events.append(event)

                    else:
                        logger.error(f"FMP API error: {response.status}")
                        return self._get_mock_earnings_calendar(from_date, to_date, symbols)

            return events

        except Exception as e:
            logger.error(f"Failed to fetch earnings calendar: {e}")
            return self._get_mock_earnings_calendar(from_date, to_date, symbols)

    async def get_economic_calendar(
        self,
        from_date: date,
        to_date: date,
        event_types: Optional[List[str]] = None
    ) -> List[EconomicEvent]:
        """
        Get economic events calendar.

        Args:
            from_date: Start date
            to_date: End date
            event_types: Optional list of event types to filter

        Returns:
            List of economic events
        """
        events = []

        # Add Fed meeting dates
        for fed_date in self.fed_meeting_dates_2025:
            if from_date <= fed_date <= to_date:
                if not event_types or 'fed_meeting' in event_types:
                    events.append(EconomicEvent(
                        event_type='fed_meeting',
                        name='FOMC Meeting - Interest Rate Decision',
                        date=fed_date,
                        time='14:00 EST',
                        importance='high',
                        volatility_expected='high',
                        market_impact='tbd'
                    ))

        # Try to fetch from FMP economic calendar
        if self.fmp_api_key:
            try:
                async with aiohttp.ClientSession() as session:
                    url = f"{self.base_url_fmp}/economic_calendar"
                    params = {
                        'from': from_date.isoformat(),
                        'to': to_date.isoformat(),
                        'apikey': self.fmp_api_key
                    }

                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()

                            for item in data:
                                event_type = self._categorize_economic_event(item.get('event', ''))

                                if event_types and event_type not in event_types:
                                    continue

                                event = EconomicEvent(
                                    event_type=event_type,
                                    name=item.get('event', ''),
                                    date=datetime.strptime(item.get('date'), '%Y-%m-%d').date(),
                                    time=item.get('time', 'TBD'),
                                    importance=item.get('impact', 'medium').lower(),
                                    estimate=item.get('estimate'),
                                    actual=item.get('actual'),
                                    previous=item.get('previous'),
                                )

                                # Determine market impact if actual data available
                                if event.actual and event.estimate:
                                    event.market_impact = self._calculate_market_impact(
                                        event_type, event.actual, event.estimate
                                    )

                                events.append(event)

            except Exception as e:
                logger.warning(f"Failed to fetch economic calendar: {e}")

        # Add standard recurring events if no API data
        if len(events) <= len([e for e in events if e.event_type == 'fed_meeting']):
            events.extend(self._get_standard_economic_events(from_date, to_date))

        return sorted(events, key=lambda x: x.date)

    async def get_calendar_by_day(
        self,
        from_date: date,
        to_date: date,
        symbols: Optional[List[str]] = None
    ) -> List[CalendarDay]:
        """
        Get complete calendar organized by day.

        Args:
            from_date: Start date
            to_date: End date
            symbols: Optional symbols filter for earnings

        Returns:
            List of calendar days with all events
        """
        # Fetch both calendars in parallel
        earnings_task = self.get_earnings_calendar(from_date, to_date, symbols)
        economic_task = self.get_economic_calendar(from_date, to_date)

        earnings_events, economic_events = await asyncio.gather(
            earnings_task,
            economic_task
        )

        # Group by date
        calendar_days: Dict[date, CalendarDay] = {}

        # Create empty days
        current = from_date
        while current <= to_date:
            calendar_days[current] = CalendarDay(
                date=current,
                earnings_events=[],
                economic_events=[],
                total_events=0,
                high_importance_count=0,
                major_earnings_count=0
            )
            current += timedelta(days=1)

        # Add earnings events
        for event in earnings_events:
            if event.date in calendar_days:
                calendar_days[event.date].earnings_events.append(event)

        # Add economic events
        for event in economic_events:
            if event.date in calendar_days:
                calendar_days[event.date].economic_events.append(event)

        # Calculate day-level metrics
        sp500_symbols = self._get_sp500_symbols()  # Top 50 for simplicity

        for day in calendar_days.values():
            day.total_events = len(day.earnings_events) + len(day.economic_events)
            day.high_importance_count = len([
                e for e in day.economic_events if e.importance == 'high'
            ])
            day.major_earnings_count = len([
                e for e in day.earnings_events if e.symbol in sp500_symbols
            ])

        return sorted(calendar_days.values(), key=lambda x: x.date)

    async def get_symbol_earnings_history(
        self,
        symbol: str,
        limit: int = 8
    ) -> List[EarningsEvent]:
        """
        Get historical earnings for a symbol.

        Useful for calculating average earnings move.
        """
        if not self.fmp_api_key:
            return []

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url_fmp}/historical/earning_calendar/{symbol}"
                params = {
                    'limit': limit,
                    'apikey': self.fmp_api_key
                }

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        events = []
                        for item in data[:limit]:
                            event = EarningsEvent(
                                symbol=symbol,
                                company_name=item.get('symbol', symbol),
                                date=datetime.strptime(item.get('date'), '%Y-%m-%d').date(),
                                time=item.get('time', 'amc'),
                                fiscal_quarter=item.get('fiscalQuarter', ''),
                                fiscal_year=item.get('fiscalYear', 0),
                                eps_estimate=item.get('epsEstimated'),
                                eps_actual=item.get('eps'),
                                revenue_estimate=item.get('revenueEstimated'),
                                revenue_actual=item.get('revenue'),
                            )

                            if event.eps_actual and event.eps_estimate:
                                event.eps_surprise_pct = (
                                    (event.eps_actual - event.eps_estimate) / abs(event.eps_estimate) * 100
                                )

                            events.append(event)

                        return events

        except Exception as e:
            logger.error(f"Failed to fetch earnings history for {symbol}: {e}")
            return []

    def _categorize_economic_event(self, event_name: str) -> str:
        """Categorize economic event by name"""
        event_lower = event_name.lower()

        if 'cpi' in event_lower or 'inflation' in event_lower:
            return 'cpi'
        elif 'gdp' in event_lower:
            return 'gdp'
        elif 'employment' in event_lower or 'jobs' in event_lower or 'nonfarm' in event_lower:
            return 'jobs'
        elif 'ppi' in event_lower:
            return 'ppi'
        elif 'pce' in event_lower:
            return 'pce'
        elif 'retail sales' in event_lower:
            return 'retail_sales'
        elif 'fed' in event_lower or 'fomc' in event_lower:
            return 'fed_meeting'
        else:
            return 'other'

    def _calculate_market_impact(
        self,
        event_type: str,
        actual: str,
        estimate: str
    ) -> str:
        """Calculate if event is bullish/bearish based on actual vs estimate"""
        # This is simplified - real implementation would be more nuanced
        try:
            actual_val = float(actual.strip('%').replace(',', ''))
            estimate_val = float(estimate.strip('%').replace(',', ''))

            diff = actual_val - estimate_val

            # CPI, PPI (lower is better for markets)
            if event_type in ['cpi', 'ppi', 'pce']:
                return 'bullish' if diff < 0 else 'bearish'

            # GDP, Jobs, Retail Sales (higher is better)
            elif event_type in ['gdp', 'jobs', 'retail_sales']:
                return 'bullish' if diff > 0 else 'bearish'

            else:
                return 'neutral'

        except:
            return 'neutral'

    def _get_standard_economic_events(
        self,
        from_date: date,
        to_date: date
    ) -> List[EconomicEvent]:
        """Get standard recurring economic events"""
        events = []

        # CPI is typically released mid-month
        current = from_date.replace(day=1)
        while current <= to_date:
            try:
                cpi_date = current.replace(day=13)
                if from_date <= cpi_date <= to_date:
                    events.append(EconomicEvent(
                        event_type='cpi',
                        name='Consumer Price Index (CPI)',
                        date=cpi_date,
                        time='08:30 EST',
                        importance='high',
                        volatility_expected='high'
                    ))

                # Jobs report - first Friday
                first_day = current.replace(day=1)
                days_ahead = 4 - first_day.weekday()  # Friday is 4
                if days_ahead < 0:
                    days_ahead += 7
                jobs_date = first_day + timedelta(days=days_ahead)

                if from_date <= jobs_date <= to_date:
                    events.append(EconomicEvent(
                        event_type='jobs',
                        name='Non-Farm Payrolls',
                        date=jobs_date,
                        time='08:30 EST',
                        importance='high',
                        volatility_expected='high'
                    ))

            except ValueError:
                pass

            # Next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        return events

    def _get_sp500_symbols(self) -> List[str]:
        """Get top S&P 500 symbols (simplified to top 50)"""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH', 'JNJ',
            'V', 'WMT', 'XOM', 'JPM', 'PG', 'MA', 'HD', 'CVX', 'LLY', 'ABBV',
            'MRK', 'KO', 'PEP', 'AVGO', 'COST', 'TMO', 'MCD', 'CSCO', 'ACN', 'DHR',
            'ABT', 'VZ', 'NKE', 'ADBE', 'CRM', 'TXN', 'NEE', 'PM', 'UNP', 'CMCSA',
            'RTX', 'ORCL', 'INTC', 'AMD', 'QCOM', 'HON', 'LOW', 'UPS', 'IBM', 'BA'
        ]

    def _get_mock_earnings_calendar(
        self,
        from_date: date,
        to_date: date,
        symbols: Optional[List[str]] = None
    ) -> List[EarningsEvent]:
        """Mock earnings calendar for testing"""
        mock_events = [
            EarningsEvent(
                symbol='AAPL',
                company_name='Apple Inc.',
                date=from_date + timedelta(days=3),
                time='amc',
                fiscal_quarter='Q1',
                fiscal_year=2025,
                eps_estimate=2.10,
                revenue_estimate=120000000000,
                historical_move_avg=4.5,
                implied_move=5.2
            ),
            EarningsEvent(
                symbol='MSFT',
                company_name='Microsoft Corporation',
                date=from_date + timedelta(days=5),
                time='amc',
                fiscal_quarter='Q2',
                fiscal_year=2025,
                eps_estimate=2.75,
                revenue_estimate=60000000000,
                historical_move_avg=3.8,
                implied_move=4.1
            ),
            EarningsEvent(
                symbol='GOOGL',
                company_name='Alphabet Inc.',
                date=from_date + timedelta(days=7),
                time='amc',
                fiscal_quarter='Q1',
                fiscal_year=2025,
                eps_estimate=1.55,
                revenue_estimate=85000000000,
                historical_move_avg=5.1,
                implied_move=5.8
            ),
        ]

        if symbols:
            mock_events = [e for e in mock_events if e.symbol in symbols]

        return [e for e in mock_events if from_date <= e.date <= to_date]
