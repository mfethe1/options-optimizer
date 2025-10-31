"""
API Routes for Options Analytics

Advanced options analytics: IV surface, skew, term structure.
Bloomberg OMON equivalent - institutional-grade analytics.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import datetime, date
import logging

from ..analytics.options_analytics_service import OptionsAnalyticsService
from ..data.options_chain_service import OptionsChainService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/options-analytics", tags=["Options Analytics"])

# Initialize services
analytics_service = OptionsAnalyticsService()
options_service = OptionsChainService()


@router.get("/{symbol}/iv-surface")
async def get_iv_surface(symbol: str) -> dict:
    """
    Get IV surface for a symbol.

    The IV surface shows implied volatility across strikes and expirations.

    Args:
        symbol: Stock symbol

    Returns:
        IV surface data with all surface points
    """
    try:
        # Get options chain
        chain = await options_service.get_options_chain(symbol.upper())

        # Convert to format for analytics
        options_data = []
        for exp_date, strikes in chain.strikes.items():
            for strike in strikes:
                # Add calls
                if strike.call_iv:
                    options_data.append({
                        'strike': strike.strike,
                        'expiration': exp_date,
                        'implied_volatility': strike.call_iv,
                        'delta': strike.call_delta,
                        'option_type': 'call'
                    })
                # Add puts
                if strike.put_iv:
                    options_data.append({
                        'strike': strike.strike,
                        'expiration': exp_date,
                        'implied_volatility': strike.put_iv,
                        'delta': strike.put_delta,
                        'option_type': 'put'
                    })

        # Calculate IV surface
        surface = await analytics_service.calculate_iv_surface(
            symbol.upper(),
            options_data,
            chain.current_price
        )

        return {
            'symbol': surface.symbol,
            'spot_price': surface.spot_price,
            'min_iv': surface.min_iv,
            'max_iv': surface.max_iv,
            'atm_iv': surface.atm_iv,
            'iv_range': surface.iv_range,
            'surface_points': [
                {
                    'strike': p.strike,
                    'expiration': p.expiration.isoformat(),
                    'days_to_expiry': p.days_to_expiry,
                    'moneyness': p.moneyness,
                    'delta': p.delta,
                    'implied_volatility': p.implied_volatility,
                    'option_type': p.option_type
                }
                for p in surface.surface_points
            ],
            'calculation_time': surface.calculation_time.isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to calculate IV surface for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate IV surface: {str(e)}")


@router.get("/{symbol}/skew")
async def get_skew(
    symbol: str,
    expiration: Optional[str] = Query(None, description="Expiration date (YYYY-MM-DD)")
) -> dict:
    """
    Get volatility skew for a symbol.

    Shows how IV varies across strikes for a given expiration.

    Args:
        symbol: Stock symbol
        expiration: Optional specific expiration (defaults to front month)

    Returns:
        Skew metrics including put/call skew, risk reversal, butterfly
    """
    try:
        # Get options chain
        chain = await options_service.get_options_chain(symbol.upper())

        # Convert to format for analytics
        options_data = []
        for exp_date, strikes in chain.strikes.items():
            for strike in strikes:
                if strike.call_iv:
                    options_data.append({
                        'strike': strike.strike,
                        'expiration': exp_date,
                        'implied_volatility': strike.call_iv,
                        'delta': strike.call_delta,
                        'option_type': 'call'
                    })
                if strike.put_iv:
                    options_data.append({
                        'strike': strike.strike,
                        'expiration': exp_date,
                        'implied_volatility': strike.put_iv,
                        'delta': strike.put_delta,
                        'option_type': 'put'
                    })

        # Determine expiration to use
        if expiration:
            target_exp = datetime.strptime(expiration, '%Y-%m-%d').date()
        else:
            # Use front month (nearest expiration)
            target_exp = min(chain.strikes.keys())

        # Calculate skew
        skew = await analytics_service.calculate_skew(
            symbol.upper(),
            target_exp,
            options_data,
            chain.current_price
        )

        return {
            'symbol': skew.symbol,
            'expiration': skew.expiration.isoformat(),
            'days_to_expiry': skew.days_to_expiry,
            'spot_price': skew.spot_price,
            'atm_strike': skew.atm_strike,
            'atm_iv': skew.atm_iv,
            'put_skew': skew.put_skew,
            'call_skew': skew.call_skew,
            'skew_slope': skew.skew_slope,
            'risk_reversal_25delta': skew.risk_reversal_25delta,
            'butterfly_25delta': skew.butterfly_25delta,
            'skew_by_strike': skew.skew_by_strike
        }

    except Exception as e:
        logger.error(f"Failed to calculate skew for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate skew: {str(e)}")


@router.get("/{symbol}/term-structure")
async def get_term_structure(
    symbol: str,
    structure_type: str = Query(default='atm', description="Type: atm, otm_put, or otm_call")
) -> dict:
    """
    Get volatility term structure for a symbol.

    Shows how IV varies across time for a given strike level.

    Args:
        symbol: Stock symbol
        structure_type: Type of structure ('atm', 'otm_put', 'otm_call')

    Returns:
        Term structure showing IV across expirations
    """
    try:
        # Get options chain
        chain = await options_service.get_options_chain(symbol.upper())

        # Convert to format for analytics
        options_data = []
        for exp_date, strikes in chain.strikes.items():
            for strike in strikes:
                if strike.call_iv:
                    options_data.append({
                        'strike': strike.strike,
                        'expiration': exp_date,
                        'implied_volatility': strike.call_iv,
                        'option_type': 'call'
                    })
                if strike.put_iv:
                    options_data.append({
                        'strike': strike.strike,
                        'expiration': exp_date,
                        'implied_volatility': strike.put_iv,
                        'option_type': 'put'
                    })

        # Calculate term structure
        term_struct = await analytics_service.calculate_term_structure(
            symbol.upper(),
            options_data,
            chain.current_price,
            structure_type
        )

        return {
            'symbol': term_struct.symbol,
            'spot_price': term_struct.spot_price,
            'structure_type': term_struct.structure_type,
            'expirations': [exp.isoformat() for exp in term_struct.expirations],
            'days_to_expiry': term_struct.days_to_expiry,
            'implied_volatilities': term_struct.implied_volatilities,
            'front_month_iv': term_struct.front_month_iv,
            'back_month_iv': term_struct.back_month_iv,
            'term_structure_slope': term_struct.term_structure_slope,
            'is_inverted': term_struct.is_inverted
        }

    except Exception as e:
        logger.error(f"Failed to calculate term structure for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate term structure: {str(e)}")


@router.get("/{symbol}/complete")
async def get_complete_analytics(symbol: str) -> dict:
    """
    Get complete options analytics package.

    Includes IV surface, skew for multiple expirations, and term structures.

    Args:
        symbol: Stock symbol

    Returns:
        Complete analytics package
    """
    try:
        # Get options chain
        chain = await options_service.get_options_chain(symbol.upper())

        # Convert to format for analytics
        options_data = []
        for exp_date, strikes in chain.strikes.items():
            for strike in strikes:
                if strike.call_iv:
                    options_data.append({
                        'strike': strike.strike,
                        'expiration': exp_date,
                        'implied_volatility': strike.call_iv,
                        'delta': strike.call_delta,
                        'option_type': 'call'
                    })
                if strike.put_iv:
                    options_data.append({
                        'strike': strike.strike,
                        'expiration': exp_date,
                        'implied_volatility': strike.put_iv,
                        'delta': strike.put_delta,
                        'option_type': 'put'
                    })

        # Calculate all analytics
        analytics = await analytics_service.calculate_all_analytics(
            symbol.upper(),
            options_data,
            chain.current_price
        )

        # Serialize
        result = {
            'symbol': analytics['symbol'],
            'spot_price': analytics['spot_price'],
            'iv_surface': {
                'min_iv': analytics['iv_surface'].min_iv,
                'max_iv': analytics['iv_surface'].max_iv,
                'atm_iv': analytics['iv_surface'].atm_iv,
                'iv_range': analytics['iv_surface'].iv_range,
                'point_count': len(analytics['iv_surface'].surface_points)
            },
            'skew_metrics': [
                {
                    'expiration': s.expiration.isoformat(),
                    'days_to_expiry': s.days_to_expiry,
                    'atm_iv': s.atm_iv,
                    'put_skew': s.put_skew,
                    'call_skew': s.call_skew,
                    'skew_slope': s.skew_slope,
                    'risk_reversal_25delta': s.risk_reversal_25delta,
                    'butterfly_25delta': s.butterfly_25delta
                }
                for s in analytics['skew_metrics']
            ],
            'term_structures': {
                struct_type: {
                    'front_month_iv': ts.front_month_iv,
                    'back_month_iv': ts.back_month_iv,
                    'term_structure_slope': ts.term_structure_slope,
                    'is_inverted': ts.is_inverted,
                    'point_count': len(ts.implied_volatilities)
                }
                for struct_type, ts in analytics['term_structures'].items()
            },
            'calculation_time': analytics['calculation_time']
        }

        return result

    except Exception as e:
        logger.error(f"Failed to calculate complete analytics for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate analytics: {str(e)}")


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint for options analytics service."""
    return {
        "status": "ok",
        "service": "options-analytics",
        "timestamp": datetime.now().isoformat(),
        "capabilities": [
            "IV Surface",
            "Volatility Skew",
            "Term Structure",
            "Risk Reversal",
            "Butterfly Spread"
        ]
    }
