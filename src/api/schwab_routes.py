"""
API Routes for Schwab Integration

Provides access to Schwab trading accounts for live execution.
"""
from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional, Dict, Any, List
import logging

from ..integrations.schwab_api import (
    SchwabAPIService,
    OrderType,
    OrderAction,
    OrderDuration
)
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/schwab", tags=["Schwab"])

# Initialize Schwab service (credentials from environment)
# Users must set SCHWAB_CLIENT_ID and SCHWAB_CLIENT_SECRET in .env
schwab_service: Optional[SchwabAPIService] = None

def get_schwab_service() -> SchwabAPIService:
    """Get or create Schwab API service"""
    global schwab_service

    if schwab_service is None:
        client_id = getattr(settings, 'SCHWAB_CLIENT_ID', None)
        client_secret = getattr(settings, 'SCHWAB_CLIENT_SECRET', None)

        if not client_id or not client_secret:
            raise HTTPException(
                status_code=500,
                detail="Schwab API credentials not configured. Set SCHWAB_CLIENT_ID and SCHWAB_CLIENT_SECRET in .env"
            )

        schwab_service = SchwabAPIService(
            client_id=client_id,
            client_secret=client_secret
        )

    return schwab_service


@router.get("/auth/url")
async def get_auth_url() -> Dict[str, str]:
    """
    Get Schwab OAuth authorization URL.

    User must visit this URL to grant access.

    Returns:
        Authorization URL and instructions
    """
    try:
        service = get_schwab_service()
        auth_url = service.get_authorization_url()

        return {
            'auth_url': auth_url,
            'instructions': 'Visit this URL to authorize access to your Schwab account. '
                          'After authorization, you will be redirected with a code parameter. '
                          'Copy the code and use POST /api/schwab/auth/token to complete setup.'
        }

    except Exception as e:
        logger.error(f"Failed to get auth URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auth/token")
async def exchange_token(
    authorization_code: str = Body(..., description="Authorization code from OAuth redirect")
) -> Dict[str, str]:
    """
    Exchange authorization code for access token.

    Args:
        authorization_code: Code from OAuth redirect URL

    Returns:
        Success status
    """
    try:
        service = get_schwab_service()
        success = await service.exchange_code_for_token(authorization_code)

        if success:
            return {
                'status': 'success',
                'message': 'Successfully authenticated with Schwab'
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to exchange code for token")

    except Exception as e:
        logger.error(f"Token exchange failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/accounts")
async def get_accounts() -> Dict[str, Any]:
    """
    Get all Schwab accounts.

    Returns:
        List of accounts with balances and positions
    """
    try:
        service = get_schwab_service()
        accounts = await service.get_accounts()

        return {
            'count': len(accounts),
            'accounts': [
                {
                    'account_id': acc.account_id,
                    'account_number': acc.account_number,
                    'account_type': acc.account_type,
                    'balances': {
                        'cash': acc.current_balances.get('cashBalance', 0),
                        'equity': acc.current_balances.get('equity', 0),
                        'buying_power': acc.current_balances.get('buyingPower', 0),
                        'total_value': acc.current_balances.get('liquidationValue', 0)
                    },
                    'position_count': len(acc.positions)
                }
                for acc in accounts
            ]
        }

    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get accounts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/accounts/{account_id}/positions")
async def get_positions(account_id: str) -> Dict[str, Any]:
    """
    Get positions for specific account.

    Args:
        account_id: Schwab account ID

    Returns:
        List of positions
    """
    try:
        service = get_schwab_service()
        positions = await service.get_account_positions(account_id)

        return {
            'account_id': account_id,
            'count': len(positions),
            'positions': [
                {
                    'symbol': pos.symbol,
                    'quantity': pos.quantity,
                    'average_price': pos.average_price,
                    'current_price': pos.current_price,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                    'instrument_type': pos.instrument_type
                }
                for pos in positions
            ]
        }

    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quote/{symbol}")
async def get_quote(symbol: str) -> Dict[str, Any]:
    """
    Get real-time quote from Schwab.

    Args:
        symbol: Stock or option symbol

    Returns:
        Real-time quote data
    """
    try:
        service = get_schwab_service()
        quote = await service.get_quote(symbol.upper())

        if not quote:
            raise HTTPException(status_code=404, detail=f"Quote not found for {symbol}")

        return {
            'symbol': quote.symbol,
            'bid': quote.bid,
            'ask': quote.ask,
            'last': quote.last,
            'bid_size': quote.bid_size,
            'ask_size': quote.ask_size,
            'volume': quote.volume,
            'timestamp': quote.timestamp.isoformat()
        }

    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get quote: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/options/{symbol}")
async def get_option_chain(
    symbol: str,
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """
    Get options chain from Schwab.

    Args:
        symbol: Underlying symbol
        from_date: Start expiration date (YYYY-MM-DD)
        to_date: End expiration date (YYYY-MM-DD)

    Returns:
        Options chain data
    """
    try:
        service = get_schwab_service()
        chain = await service.get_option_chain(symbol.upper(), from_date, to_date)

        if not chain:
            raise HTTPException(status_code=404, detail=f"Option chain not found for {symbol}")

        return {
            'symbol': chain.symbol,
            'underlying_price': chain.underlying_price,
            'expiration_dates': chain.expiration_dates,
            'call_count': sum(len(strikes) for strikes in chain.calls.values()),
            'put_count': sum(len(strikes) for strikes in chain.puts.values()),
            'calls': chain.calls,
            'puts': chain.puts
        }

    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get option chain: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/accounts/{account_id}/orders")
async def place_order(
    account_id: str,
    symbol: str = Body(...),
    quantity: int = Body(...),
    order_type: str = Body(...),
    order_action: str = Body(...),
    duration: str = Body("DAY"),
    price: Optional[float] = Body(None)
) -> Dict[str, Any]:
    """
    Place an order through Schwab.

    ⚠️ LIVE TRADING - This places real orders with real money!

    Args:
        account_id: Schwab account ID
        symbol: Symbol to trade
        quantity: Number of shares/contracts
        order_type: MARKET, LIMIT, STOP, STOP_LIMIT
        order_action: BUY, SELL, BUY_TO_OPEN, SELL_TO_CLOSE, etc.
        duration: DAY, GOOD_TILL_CANCEL, FILL_OR_KILL
        price: Limit/stop price (required for LIMIT and STOP orders)

    Returns:
        Order ID and confirmation
    """
    try:
        service = get_schwab_service()

        # Parse enums
        try:
            order_type_enum = OrderType[order_type.upper()]
            order_action_enum = OrderAction[order_action.upper()]
            duration_enum = OrderDuration[duration.upper()] if duration else OrderDuration.DAY
        except KeyError as e:
            raise HTTPException(status_code=400, detail=f"Invalid order parameter: {e}")

        # Validate price for limit/stop orders
        if order_type_enum in [OrderType.LIMIT, OrderType.STOP] and not price:
            raise HTTPException(status_code=400, detail=f"{order_type} orders require a price")

        # Place order
        order_id = await service.place_order(
            account_id=account_id,
            symbol=symbol.upper(),
            quantity=quantity,
            order_type=order_type_enum,
            order_action=order_action_enum,
            duration=duration_enum,
            price=price
        )

        if order_id:
            return {
                'status': 'success',
                'order_id': order_id,
                'message': f'Order placed: {order_action} {quantity} {symbol} @ {price or "MARKET"}'
            }
        else:
            raise HTTPException(status_code=500, detail="Order placement failed")

    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to place order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/accounts/{account_id}/orders/{order_id}")
async def get_order(account_id: str, order_id: str) -> Dict[str, Any]:
    """
    Get order details.

    Args:
        account_id: Schwab account ID
        order_id: Order ID

    Returns:
        Order details
    """
    try:
        service = get_schwab_service()
        order = await service.get_order(account_id, order_id)

        if not order:
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found")

        return order

    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/accounts/{account_id}/orders/{order_id}")
async def cancel_order(account_id: str, order_id: str) -> Dict[str, str]:
    """
    Cancel an order.

    Args:
        account_id: Schwab account ID
        order_id: Order ID to cancel

    Returns:
        Cancellation confirmation
    """
    try:
        service = get_schwab_service()
        success = await service.cancel_order(account_id, order_id)

        if success:
            return {
                'status': 'success',
                'message': f'Order {order_id} cancelled'
            }
        else:
            raise HTTPException(status_code=500, detail="Order cancellation failed")

    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to cancel order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint for Schwab integration."""
    try:
        service = get_schwab_service()
        is_authenticated = service.access_token is not None

        return {
            "status": "ok",
            "service": "schwab_integration",
            "authenticated": is_authenticated,
            "capabilities": [
                "OAuth 2.0 Authentication",
                "Account Data Retrieval",
                "Real-Time Quotes",
                "Options Chains",
                "Live Order Placement",
                "Position Tracking",
                "Order Management"
            ]
        }
    except Exception as e:
        return {
            "status": "not_configured",
            "error": str(e),
            "message": "Set SCHWAB_CLIENT_ID and SCHWAB_CLIENT_SECRET in .env to enable Schwab integration"
        }
