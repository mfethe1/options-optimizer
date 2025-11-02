"""
Options Chain API Routes

Bloomberg OMON equivalent - real-time options chain with Greeks.
"""
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query
from typing import Optional, List
from datetime import date, datetime
import logging
import json

from ..data.options_chain_service import options_chain_service, OptionsChain

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/options-chain",
    tags=["Options Chain"]
)


@router.get("/{symbol}")
async def get_options_chain(
    symbol: str,
    expiration: Optional[str] = Query(None, description="Expiration date (YYYY-MM-DD)")
):
    """
    Get complete options chain for a symbol.

    **CRITICAL FEATURE**: Bloomberg OMON equivalent

    Returns:
    - All strikes for expiration(s)
    - Bid/ask/last for calls and puts
    - Volume and open interest
    - Implied volatility for each strike
    - Greeks (delta, gamma, theta, vega)
    - Unusual activity flags
    - Max pain level
    - Put/call ratios

    Performance target: < 500ms
    """
    try:
        exp_date = None
        if expiration:
            try:
                exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

        chain = await options_chain_service.get_options_chain(
            symbol=symbol.upper(),
            expiration=exp_date
        )

        # Convert to dict for JSON serialization
        return _serialize_chain(chain)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching options chain: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch options chain: {str(e)}")


@router.get("/{symbol}/expirations")
async def get_expirations(symbol: str):
    """
    Get available expiration dates for a symbol.

    Returns list of expiration dates in ascending order.
    """
    try:
        chain = await options_chain_service.get_options_chain(symbol.upper())

        return {
            "symbol": chain.symbol,
            "expirations": [exp.isoformat() for exp in chain.expirations],
            "count": len(chain.expirations)
        }

    except Exception as e:
        logger.error(f"Error fetching expirations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}/summary")
async def get_options_summary(symbol: str):
    """
    Get options summary metrics for a symbol.

    Includes:
    - Current stock price
    - IV Rank and IV Percentile
    - Historical volatility
    - Max pain
    - Put/call ratios
    """
    try:
        chain = await options_chain_service.get_options_chain(symbol.upper())

        return {
            "symbol": chain.symbol,
            "current_price": chain.current_price,
            "price_change": chain.price_change,
            "price_change_pct": chain.price_change_pct,
            "iv_rank": chain.iv_rank,
            "iv_percentile": chain.iv_percentile,
            "hv_20": chain.hv_20,
            "hv_30": chain.hv_30,
            "max_pain": chain.max_pain,
            "put_call_ratio_volume": chain.put_call_ratio_volume,
            "put_call_ratio_oi": chain.put_call_ratio_oi,
            "last_updated": chain.last_updated.isoformat(),
            "data_source": chain.data_source
        }

    except Exception as e:
        logger.error(f"Error fetching options summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}/unusual-activity")
async def get_unusual_activity(
    symbol: str,
    expiration: Optional[str] = None,
    min_volume: int = 100
):
    """
    Get strikes with unusual volume or open interest.

    Filters:
    - Volume > 3x average
    - Min volume threshold
    """
    try:
        exp_date = None
        if expiration:
            exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()

        chain = await options_chain_service.get_options_chain(symbol.upper(), exp_date)

        unusual = []
        for exp, strikes in chain.strikes.items():
            for strike in strikes:
                # Check for unusual call activity
                if strike.unusual_volume_call and strike.call_volume >= min_volume:
                    unusual.append({
                        "type": "call",
                        "strike": strike.strike,
                        "expiration": exp.isoformat(),
                        "volume": strike.call_volume,
                        "open_interest": strike.call_open_interest,
                        "iv": strike.call_iv,
                        "last_price": strike.call_last
                    })

                # Check for unusual put activity
                if strike.unusual_volume_put and strike.put_volume >= min_volume:
                    unusual.append({
                        "type": "put",
                        "strike": strike.strike,
                        "expiration": exp.isoformat(),
                        "volume": strike.put_volume,
                        "open_interest": strike.put_open_interest,
                        "iv": strike.put_iv,
                        "last_price": strike.put_last
                    })

        return {
            "symbol": chain.symbol,
            "unusual_activity": unusual,
            "count": len(unusual),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching unusual activity: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/{symbol}")
async def options_chain_websocket(websocket: WebSocket, symbol: str, expiration: Optional[str] = None):
    """
    WebSocket endpoint for real-time options chain updates.

    **Performance**: Sub-100ms updates

    Usage:
        ws://localhost:8000/api/options-chain/ws/AAPL?expiration=2024-12-20

    Sends updates every 5 seconds with:
    - Updated prices
    - Volume changes
    - OI changes
    - Greeks updates
    """
    await websocket.accept()

    try:
        exp_date = None
        if expiration:
            try:
                exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()
            except ValueError:
                await websocket.send_json({"error": "Invalid date format"})
                await websocket.close()
                return

        logger.info(f"WebSocket connected for {symbol} options chain")

        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "symbol": symbol.upper(),
            "expiration": expiration,
            "timestamp": datetime.now().isoformat()
        })

        # Stream updates
        async def send_update(chain: OptionsChain):
            try:
                await websocket.send_json({
                    "type": "update",
                    "data": _serialize_chain(chain),
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error sending WebSocket update: {e}")
                raise

        await options_chain_service.stream_options_updates(
            symbol=symbol.upper(),
            expiration=exp_date,
            callback=send_update
        )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {symbol}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
        except:
            pass


def _serialize_chain(chain: OptionsChain) -> dict:
    """Convert OptionsChain to JSON-serializable dict"""
    strikes_dict = {}

    for exp, strikes in chain.strikes.items():
        strikes_dict[exp.isoformat()] = [
            {
                "strike": s.strike,
                "expiration": s.expiration.isoformat(),

                # Call data
                "call": {
                    "bid": s.call_bid,
                    "ask": s.call_ask,
                    "last": s.call_last,
                    "volume": s.call_volume,
                    "open_interest": s.call_open_interest,
                    "iv": s.call_iv,
                    "delta": s.call_delta,
                    "gamma": s.call_gamma,
                    "theta": s.call_theta,
                    "vega": s.call_vega,
                    "in_the_money": s.in_the_money_call,
                    "unusual_volume": s.unusual_volume_call
                },

                # Put data
                "put": {
                    "bid": s.put_bid,
                    "ask": s.put_ask,
                    "last": s.put_last,
                    "volume": s.put_volume,
                    "open_interest": s.put_open_interest,
                    "iv": s.put_iv,
                    "delta": s.put_delta,
                    "gamma": s.put_gamma,
                    "theta": s.put_theta,
                    "vega": s.put_vega,
                    "in_the_money": s.in_the_money_put,
                    "unusual_volume": s.unusual_volume_put
                }
            }
            for s in strikes
        ]

    return {
        "symbol": chain.symbol,
        "current_price": chain.current_price,
        "price_change": chain.price_change,
        "price_change_pct": chain.price_change_pct,
        "iv_rank": chain.iv_rank,
        "iv_percentile": chain.iv_percentile,
        "hv_20": chain.hv_20,
        "hv_30": chain.hv_30,
        "expirations": [exp.isoformat() for exp in chain.expirations],
        "strikes": strikes_dict,
        "max_pain": chain.max_pain,
        "put_call_ratio_volume": chain.put_call_ratio_volume,
        "put_call_ratio_oi": chain.put_call_ratio_oi,
        "last_updated": chain.last_updated.isoformat(),
        "data_source": chain.data_source
    }
