"""
Multi-Broker Connectivity API Routes

REST API endpoints for managing multiple broker connections.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/brokers", tags=["brokers"])

# Global broker manager (initialized on startup)
broker_manager = None


# ============================================================================
# Request/Response Models
# ============================================================================

class BrokerCredentialsRequest(BaseModel):
    """Broker credentials for connection"""
    broker_type: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    account_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None


class BrokerHealthResponse(BaseModel):
    """Broker health status"""
    broker_type: str
    status: str
    latency_ms: float
    last_check: str
    error_count: int
    error_message: Optional[str]
    uptime_pct: float


class QuoteResponse(BaseModel):
    """Market quote"""
    symbol: str
    bid: float
    ask: float
    last: float
    bid_size: int
    ask_size: int
    timestamp: str
    broker: str


class PositionResponse(BaseModel):
    """Position information"""
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    broker: str


class AccountResponse(BaseModel):
    """Account information"""
    account_id: str
    broker: str
    cash: float
    buying_power: float
    portfolio_value: float
    equity: float
    positions: List[PositionResponse]
    timestamp: str


class OrderRequest(BaseModel):
    """Order placement request"""
    symbol: str
    side: str  # "buy" or "sell"
    order_type: str  # "market" or "limit"
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    preferred_broker: Optional[str] = None


class OrderResponse(BaseModel):
    """Order information"""
    order_id: str
    broker: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    status: str
    filled_quantity: float
    average_fill_price: Optional[float]
    submitted_at: str
    filled_at: Optional[str]


# ============================================================================
# Startup/Shutdown
# ============================================================================

async def initialize_broker_manager():
    """Initialize broker manager"""
    global broker_manager
    try:
        from ..brokers.broker_manager import BrokerManager
        broker_manager = BrokerManager()
        logger.info("Broker manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize broker manager: {e}")
        raise


async def shutdown_broker_manager():
    """Shutdown broker manager"""
    global broker_manager
    if broker_manager:
        try:
            await broker_manager.shutdown()
            logger.info("Broker manager shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down broker manager: {e}")


# ============================================================================
# Broker Management Endpoints
# ============================================================================

@router.post("/connect")
async def connect_broker(credentials: BrokerCredentialsRequest):
    """
    Connect to a broker.

    Adds a new broker connection to the manager.
    First broker added becomes the primary broker.

    Example:
    ```json
    {
      "broker_type": "schwab",
      "client_id": "your_client_id",
      "client_secret": "your_client_secret",
      "account_id": "12345678"
    }
    ```
    """
    if not broker_manager:
        raise HTTPException(status_code=503, detail="Broker manager not initialized")

    try:
        from ..brokers.broker_adapters import BrokerCredentials, BrokerType

        creds = BrokerCredentials(
            broker_type=BrokerType(credentials.broker_type),
            api_key=credentials.api_key,
            api_secret=credentials.api_secret,
            account_id=credentials.account_id,
            client_id=credentials.client_id,
            client_secret=credentials.client_secret
        )

        success = await broker_manager.add_broker(creds)

        if success:
            return {
                "status": "connected",
                "broker_type": credentials.broker_type,
                "message": f"Successfully connected to {credentials.broker_type}"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to connect to broker")

    except Exception as e:
        logger.error(f"Error connecting broker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/disconnect/{broker_type}")
async def disconnect_broker(broker_type: str):
    """
    Disconnect from a broker.

    Removes broker connection from the manager.
    """
    if not broker_manager:
        raise HTTPException(status_code=503, detail="Broker manager not initialized")

    try:
        from ..brokers.broker_adapters import BrokerType

        success = await broker_manager.remove_broker(BrokerType(broker_type))

        if success:
            return {
                "status": "disconnected",
                "broker_type": broker_type,
                "message": f"Successfully disconnected from {broker_type}"
            }
        else:
            raise HTTPException(status_code=404, detail="Broker not found")

    except Exception as e:
        logger.error(f"Error disconnecting broker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=List[BrokerHealthResponse])
async def get_broker_health():
    """
    Get health status of all connected brokers.

    Returns:
    - Connection status (healthy, degraded, offline)
    - Latency metrics
    - Error counts
    - Uptime percentage
    """
    if not broker_manager:
        raise HTTPException(status_code=503, detail="Broker manager not initialized")

    try:
        health_list = await broker_manager.get_all_broker_health()

        return [
            BrokerHealthResponse(
                broker_type=health.broker_type.value,
                status=health.status.value,
                latency_ms=health.latency_ms,
                last_check=health.last_check.isoformat(),
                error_count=health.error_count,
                error_message=health.error_message,
                uptime_pct=health.uptime_pct
            ) for health in health_list
        ]

    except Exception as e:
        logger.error(f"Error getting broker health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_broker_status():
    """
    Get broker manager status.

    Returns:
    - Total brokers connected
    - Healthy brokers count
    - Primary broker
    """
    if not broker_manager:
        raise HTTPException(status_code=503, detail="Broker manager not initialized")

    return {
        "total_brokers": broker_manager.get_broker_count(),
        "healthy_brokers": broker_manager.get_healthy_broker_count(),
        "primary_broker": broker_manager.get_primary_broker().value if broker_manager.get_primary_broker() else None,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Trading Endpoints
# ============================================================================

@router.get("/quote/{symbol}", response_model=QuoteResponse)
async def get_best_quote(symbol: str):
    """
    Get best quote across all brokers.

    Aggregates quotes from all healthy brokers and returns best bid/ask.
    """
    if not broker_manager:
        raise HTTPException(status_code=503, detail="Broker manager not initialized")

    try:
        quote = await broker_manager.get_best_quote(symbol.upper())

        if not quote:
            raise HTTPException(status_code=404, detail=f"No quote available for {symbol}")

        return QuoteResponse(
            symbol=quote.symbol,
            bid=quote.bid,
            ask=quote.ask,
            last=quote.last,
            bid_size=quote.bid_size,
            ask_size=quote.ask_size,
            timestamp=quote.timestamp.isoformat(),
            broker=quote.broker.value
        )

    except Exception as e:
        logger.error(f"Error getting quote: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/account", response_model=AccountResponse)
async def get_consolidated_account():
    """
    Get consolidated account across all brokers.

    Combines positions and balances from all connected brokers.
    """
    if not broker_manager:
        raise HTTPException(status_code=503, detail="Broker manager not initialized")

    try:
        account = await broker_manager.get_consolidated_account()

        if not account:
            raise HTTPException(status_code=404, detail="No account data available")

        return AccountResponse(
            account_id=account.account_id,
            broker=account.broker.value,
            cash=account.cash,
            buying_power=account.buying_power,
            portfolio_value=account.portfolio_value,
            equity=account.equity,
            positions=[
                PositionResponse(
                    symbol=pos.symbol,
                    quantity=pos.quantity,
                    average_price=pos.average_price,
                    current_price=pos.current_price,
                    market_value=pos.market_value,
                    unrealized_pnl=pos.unrealized_pnl,
                    unrealized_pnl_pct=pos.unrealized_pnl_pct,
                    broker=pos.broker.value
                ) for pos in account.positions
            ],
            timestamp=account.timestamp.isoformat()
        )

    except Exception as e:
        logger.error(f"Error getting account: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/orders", response_model=OrderResponse)
async def place_order(request: OrderRequest):
    """
    Place order with intelligent broker selection.

    Tries preferred broker first, automatically fails over to other brokers if needed.

    Example:
    ```json
    {
      "symbol": "AAPL",
      "side": "buy",
      "order_type": "limit",
      "quantity": 100,
      "price": 180.50,
      "preferred_broker": "schwab"
    }
    ```
    """
    if not broker_manager:
        raise HTTPException(status_code=503, detail="Broker manager not initialized")

    try:
        from ..brokers.broker_adapters import BrokerType, OrderSide, OrderType

        preferred_broker = BrokerType(request.preferred_broker) if request.preferred_broker else None

        order = await broker_manager.place_order_smart(
            symbol=request.symbol.upper(),
            side=OrderSide(request.side.lower()),
            order_type=OrderType(request.order_type.lower()),
            quantity=request.quantity,
            price=request.price,
            stop_price=request.stop_price,
            preferred_broker=preferred_broker
        )

        if not order:
            raise HTTPException(status_code=500, detail="Failed to place order with any broker")

        return OrderResponse(
            order_id=order.order_id,
            broker=order.broker.value,
            symbol=order.symbol,
            side=order.side.value,
            order_type=order.order_type.value,
            quantity=order.quantity,
            price=order.price,
            stop_price=order.stop_price,
            status=order.status.value,
            filled_quantity=order.filled_quantity,
            average_fill_price=order.average_fill_price,
            submitted_at=order.submitted_at.isoformat(),
            filled_at=order.filled_at.isoformat() if order.filled_at else None
        )

    except Exception as e:
        logger.error(f"Error placing order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/orders/{broker_type}/{order_id}")
async def cancel_order(broker_type: str, order_id: str):
    """
    Cancel an order.

    Args:
        broker_type: Broker where order was placed
        order_id: Order ID to cancel
    """
    if not broker_manager:
        raise HTTPException(status_code=503, detail="Broker manager not initialized")

    try:
        from ..brokers.broker_adapters import BrokerType

        success = await broker_manager.cancel_order(BrokerType(broker_type), order_id)

        if success:
            return {
                "status": "cancelled",
                "order_id": order_id,
                "broker": broker_type
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to cancel order")

    except Exception as e:
        logger.error(f"Error cancelling order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orders/{broker_type}/{order_id}", response_model=OrderResponse)
async def get_order(broker_type: str, order_id: str):
    """
    Get order status.

    Args:
        broker_type: Broker where order was placed
        order_id: Order ID
    """
    if not broker_manager:
        raise HTTPException(status_code=503, detail="Broker manager not initialized")

    try:
        from ..brokers.broker_adapters import BrokerType

        order = await broker_manager.get_order(BrokerType(broker_type), order_id)

        if not order:
            raise HTTPException(status_code=404, detail="Order not found")

        return OrderResponse(
            order_id=order.order_id,
            broker=order.broker.value,
            symbol=order.symbol,
            side=order.side.value,
            order_type=order.order_type.value,
            quantity=order.quantity,
            price=order.price,
            stop_price=order.stop_price,
            status=order.status.value,
            filled_quantity=order.filled_quantity,
            average_fill_price=order.average_fill_price,
            submitted_at=order.submitted_at.isoformat(),
            filled_at=order.filled_at.isoformat() if order.filled_at else None
        )

    except Exception as e:
        logger.error(f"Error getting order: {e}")
        raise HTTPException(status_code=500, detail=str(e))
