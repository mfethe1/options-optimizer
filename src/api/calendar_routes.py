"""
API Routes for Economic Calendar

Earnings calendar, economic events, and volatility implications.
Bloomberg EVTS equivalent.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from datetime import date, datetime, timedelta
import logging

from ..data.calendar_service import CalendarService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/calendar", tags=["Calendar"])

# Initialize service
calendar_service = CalendarService()


@router.get("/earnings")
async def get_earnings_calendar(
    from_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    symbols: Optional[str] = Query(None, description="Comma-separated symbols"),
    days_ahead: Optional[int] = Query(30, description="Days ahead if dates not provided")
) -> dict:
    """
    Get earnings calendar.

    Args:
        from_date: Start date (defaults to today)
        to_date: End date (defaults to 30 days ahead)
        symbols: Optional comma-separated symbols
        days_ahead: Days to look ahead if dates not provided

    Returns:
        Earnings events with estimates and actuals
    """
    try:
        # Parse dates
        if from_date:
            start = datetime.strptime(from_date, '%Y-%m-%d').date()
        else:
            start = date.today()

        if to_date:
            end = datetime.strptime(to_date, '%Y-%m-%d').date()
        else:
            end = start + timedelta(days=days_ahead)

        # Parse symbols
        symbol_list = None
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(',')]

        # Fetch earnings
        events = await calendar_service.get_earnings_calendar(start, end, symbol_list)

        return {
            'from_date': start.isoformat(),
            'to_date': end.isoformat(),
            'count': len(events),
            'events': [
                {
                    'symbol': e.symbol,
                    'company_name': e.company_name,
                    'date': e.date.isoformat(),
                    'time': e.time,
                    'fiscal_quarter': e.fiscal_quarter,
                    'fiscal_year': e.fiscal_year,
                    'eps_estimate': e.eps_estimate,
                    'revenue_estimate': e.revenue_estimate,
                    'eps_actual': e.eps_actual,
                    'revenue_actual': e.revenue_actual,
                    'eps_surprise_pct': e.eps_surprise_pct,
                    'revenue_surprise_pct': e.revenue_surprise_pct,
                    'historical_move_avg': e.historical_move_avg,
                    'implied_move': e.implied_move,
                }
                for e in events
            ]
        }

    except Exception as e:
        logger.error(f"Failed to fetch earnings calendar: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch earnings calendar: {str(e)}")


@router.get("/economic")
async def get_economic_calendar(
    from_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    event_types: Optional[str] = Query(None, description="Comma-separated event types"),
    days_ahead: Optional[int] = Query(30, description="Days ahead if dates not provided")
) -> dict:
    """
    Get economic events calendar.

    Args:
        from_date: Start date (defaults to today)
        to_date: End date (defaults to 30 days ahead)
        event_types: Optional comma-separated event types
        days_ahead: Days to look ahead if dates not provided

    Returns:
        Economic events with estimates and actuals
    """
    try:
        # Parse dates
        if from_date:
            start = datetime.strptime(from_date, '%Y-%m-%d').date()
        else:
            start = date.today()

        if to_date:
            end = datetime.strptime(to_date, '%Y-%m-%d').date()
        else:
            end = start + timedelta(days=days_ahead)

        # Parse event types
        types_list = None
        if event_types:
            types_list = [t.strip().lower() for t in event_types.split(',')]

        # Fetch economic events
        events = await calendar_service.get_economic_calendar(start, end, types_list)

        return {
            'from_date': start.isoformat(),
            'to_date': end.isoformat(),
            'count': len(events),
            'events': [
                {
                    'event_type': e.event_type,
                    'name': e.name,
                    'date': e.date.isoformat(),
                    'time': e.time,
                    'importance': e.importance,
                    'estimate': e.estimate,
                    'actual': e.actual,
                    'previous': e.previous,
                    'market_impact': e.market_impact,
                    'volatility_expected': e.volatility_expected,
                }
                for e in events
            ]
        }

    except Exception as e:
        logger.error(f"Failed to fetch economic calendar: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch economic calendar: {str(e)}")


@router.get("/complete")
async def get_complete_calendar(
    from_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    symbols: Optional[str] = Query(None, description="Comma-separated symbols for earnings"),
    days_ahead: Optional[int] = Query(30, description="Days ahead if dates not provided")
) -> dict:
    """
    Get complete calendar organized by day.

    Includes both earnings and economic events.

    Args:
        from_date: Start date
        to_date: End date
        symbols: Optional symbols filter for earnings
        days_ahead: Days to look ahead

    Returns:
        Calendar days with all events
    """
    try:
        # Parse dates
        if from_date:
            start = datetime.strptime(from_date, '%Y-%m-%d').date()
        else:
            start = date.today()

        if to_date:
            end = datetime.strptime(to_date, '%Y-%m-%d').date()
        else:
            end = start + timedelta(days=days_ahead)

        # Parse symbols
        symbol_list = None
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(',')]

        # Fetch complete calendar
        calendar_days = await calendar_service.get_calendar_by_day(start, end, symbol_list)

        return {
            'from_date': start.isoformat(),
            'to_date': end.isoformat(),
            'total_days': len(calendar_days),
            'days': [
                {
                    'date': day.date.isoformat(),
                    'total_events': day.total_events,
                    'high_importance_count': day.high_importance_count,
                    'major_earnings_count': day.major_earnings_count,
                    'earnings_events': [
                        {
                            'symbol': e.symbol,
                            'company_name': e.company_name,
                            'time': e.time,
                            'fiscal_quarter': e.fiscal_quarter,
                            'eps_estimate': e.eps_estimate,
                            'eps_actual': e.eps_actual,
                            'eps_surprise_pct': e.eps_surprise_pct,
                            'implied_move': e.implied_move,
                        }
                        for e in day.earnings_events
                    ],
                    'economic_events': [
                        {
                            'event_type': e.event_type,
                            'name': e.name,
                            'time': e.time,
                            'importance': e.importance,
                            'estimate': e.estimate,
                            'actual': e.actual,
                            'volatility_expected': e.volatility_expected,
                        }
                        for e in day.economic_events
                    ]
                }
                for day in calendar_days
            ]
        }

    except Exception as e:
        logger.error(f"Failed to fetch complete calendar: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch complete calendar: {str(e)}")


@router.get("/earnings/{symbol}/history")
async def get_earnings_history(
    symbol: str,
    limit: int = Query(8, description="Number of historical earnings to fetch")
) -> dict:
    """
    Get earnings history for a symbol.

    Useful for calculating average earnings moves.

    Args:
        symbol: Stock symbol
        limit: Number of historical earnings

    Returns:
        Historical earnings with surprises
    """
    try:
        events = await calendar_service.get_symbol_earnings_history(symbol.upper(), limit)

        # Calculate average surprise
        surprises = [e.eps_surprise_pct for e in events if e.eps_surprise_pct is not None]
        avg_surprise = sum(surprises) / len(surprises) if surprises else None

        return {
            'symbol': symbol.upper(),
            'count': len(events),
            'average_eps_surprise_pct': avg_surprise,
            'events': [
                {
                    'date': e.date.isoformat(),
                    'time': e.time,
                    'fiscal_quarter': e.fiscal_quarter,
                    'fiscal_year': e.fiscal_year,
                    'eps_estimate': e.eps_estimate,
                    'eps_actual': e.eps_actual,
                    'revenue_estimate': e.revenue_estimate,
                    'revenue_actual': e.revenue_actual,
                    'eps_surprise_pct': e.eps_surprise_pct,
                    'revenue_surprise_pct': e.revenue_surprise_pct,
                }
                for e in events
            ]
        }

    except Exception as e:
        logger.error(f"Failed to fetch earnings history for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch earnings history: {str(e)}")


@router.get("/upcoming-week")
async def get_upcoming_week() -> dict:
    """
    Get calendar for the upcoming week.

    Quick endpoint for dashboard widgets.

    Returns:
        This week's key events
    """
    try:
        today = date.today()
        week_end = today + timedelta(days=7)

        calendar_days = await calendar_service.get_calendar_by_day(today, week_end)

        # Get high-impact events only
        high_impact_days = [
            day for day in calendar_days
            if day.high_importance_count > 0 or day.major_earnings_count > 0
        ]

        return {
            'from_date': today.isoformat(),
            'to_date': week_end.isoformat(),
            'high_impact_days': len(high_impact_days),
            'days': [
                {
                    'date': day.date.isoformat(),
                    'total_events': day.total_events,
                    'high_importance_count': day.high_importance_count,
                    'major_earnings_count': day.major_earnings_count,
                    'earnings_events': [
                        {
                            'symbol': e.symbol,
                            'company_name': e.company_name,
                            'time': e.time,
                        }
                        for e in day.earnings_events
                    ][:5],  # Limit to 5 per day
                    'economic_events': [
                        {
                            'event_type': e.event_type,
                            'name': e.name,
                            'time': e.time,
                            'importance': e.importance,
                        }
                        for e in day.economic_events
                        if e.importance == 'high'
                    ]
                }
                for day in high_impact_days
            ]
        }

    except Exception as e:
        logger.error(f"Failed to fetch upcoming week: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch upcoming week: {str(e)}")


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint for calendar service."""
    return {
        "status": "ok",
        "service": "calendar",
        "timestamp": datetime.now().isoformat(),
        "capabilities": [
            "Earnings Calendar",
            "Economic Events",
            "Fed Meeting Schedule",
            "Earnings History",
            "Volatility Implications"
        ]
    }
