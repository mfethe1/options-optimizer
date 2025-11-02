"""
Market Data API Routes

Real-time market data endpoints using institutional data aggregator.
Provides sub-200ms latency quotes aggregated from multiple providers.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from typing import List, Optional, Dict
import asyncio
import logging
import json
from datetime import datetime

from ..data.institutional_data_aggregator import (
    InstitutionalDataAggregator,
    AggregatedQuote,
    Level2OrderBook,
    DataProvider
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/market-data", tags=["market-data"])

# Global data aggregator instance (initialized on startup)
data_aggregator: Optional[InstitutionalDataAggregator] = None


# ============================================================================
# Startup/Shutdown
# ============================================================================

async def initialize_data_aggregator(api_keys: Dict[str, str]):
    """Initialize the data aggregator with API keys"""
    global data_aggregator
    try:
        data_aggregator = InstitutionalDataAggregator(api_keys)
        await data_aggregator.connect_all()
        logger.info("Data aggregator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize data aggregator: {e}")
        raise


async def shutdown_data_aggregator():
    """Shutdown the data aggregator"""
    global data_aggregator
    if data_aggregator:
        await data_aggregator.close()
        logger.info("Data aggregator shutdown successfully")


# ============================================================================
# REST Endpoints
# ============================================================================

@router.get("/quote/{symbol}")
async def get_real_time_quote(symbol: str) -> Dict:
    """
    Get real-time aggregated quote for a symbol.

    Returns best bid/ask across all connected providers.
    Typical latency: <200ms
    """
    if not data_aggregator:
        raise HTTPException(status_code=503, detail="Data aggregator not initialized")

    symbol = symbol.upper()
    quote = data_aggregator.get_quote(symbol)

    if not quote:
        raise HTTPException(status_code=404, detail=f"No quote available for {symbol}")

    return {
        "symbol": quote.symbol,
        "best_bid": quote.best_bid,
        "best_ask": quote.best_ask,
        "best_bid_provider": quote.best_bid_provider.value,
        "best_ask_provider": quote.best_ask_provider.value,
        "bid_size": quote.bid_size,
        "ask_size": quote.ask_size,
        "last": quote.last,
        "mid_price": quote.mid_price,
        "spread": quote.spread,
        "spread_bps": quote.spread_bps,
        "timestamp": quote.timestamp.isoformat(),
        "num_providers": quote.num_providers,
        "avg_latency_ms": quote.avg_latency_ms
    }


@router.get("/quotes/batch")
async def get_batch_quotes(symbols: str = Query(..., description="Comma-separated symbols")) -> Dict:
    """
    Get real-time quotes for multiple symbols.

    Example: /api/market-data/quotes/batch?symbols=AAPL,MSFT,GOOGL
    """
    if not data_aggregator:
        raise HTTPException(status_code=503, detail="Data aggregator not initialized")

    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    quotes = {}

    for symbol in symbol_list:
        quote = data_aggregator.get_quote(symbol)
        if quote:
            quotes[symbol] = {
                "best_bid": quote.best_bid,
                "best_ask": quote.best_ask,
                "mid_price": quote.mid_price,
                "spread_bps": quote.spread_bps,
                "last": quote.last,
                "timestamp": quote.timestamp.isoformat(),
                "num_providers": quote.num_providers,
                "avg_latency_ms": quote.avg_latency_ms
            }

    return {
        "quotes": quotes,
        "count": len(quotes),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/order-book/{symbol}")
async def get_order_book(symbol: str) -> Dict:
    """
    Get Level 2 order book for a symbol.

    Returns top 10 bid/ask levels with sizes.
    """
    if not data_aggregator:
        raise HTTPException(status_code=503, detail="Data aggregator not initialized")

    symbol = symbol.upper()
    order_book = data_aggregator.get_order_book(symbol)

    if not order_book:
        raise HTTPException(status_code=404, detail=f"No order book available for {symbol}")

    return {
        "symbol": order_book.symbol,
        "bids": [{"price": price, "size": size} for price, size in order_book.bids],
        "asks": [{"price": price, "size": size} for price, size in order_book.asks],
        "best_bid": order_book.best_bid,
        "best_ask": order_book.best_ask,
        "spread": order_book.spread,
        "spread_bps": order_book.spread_bps,
        "imbalance": order_book.imbalance,
        "timestamp": order_book.timestamp.isoformat()
    }


@router.get("/latency-stats")
async def get_latency_stats() -> Dict:
    """
    Get latency statistics for all data providers.

    Returns p50, p95, p99 latencies to monitor data quality.
    """
    if not data_aggregator:
        raise HTTPException(status_code=503, detail="Data aggregator not initialized")

    stats = data_aggregator.get_latency_stats()

    # Convert enum keys to strings for JSON serialization
    return {
        provider.value: {
            "avg_ms": round(values["avg_ms"], 2),
            "p50_ms": round(values["p50_ms"], 2),
            "p95_ms": round(values["p95_ms"], 2),
            "p99_ms": round(values["p99_ms"], 2),
            "min_ms": round(values["min_ms"], 2),
            "max_ms": round(values["max_ms"], 2)
        }
        for provider, values in stats.items()
    }


@router.get("/provider-status")
async def get_provider_status() -> Dict:
    """
    Get connection status for all data providers.

    Returns which providers are currently connected.
    """
    if not data_aggregator:
        raise HTTPException(status_code=503, detail="Data aggregator not initialized")

    status = data_aggregator.get_provider_status()

    return {
        "providers": {
            provider.value: is_connected
            for provider, is_connected in status.items()
        },
        "connected_count": sum(status.values()),
        "total_count": len(status),
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# WebSocket Streaming
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections for real-time streaming"""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, symbol: str):
        """Accept WebSocket connection and subscribe to symbol"""
        await websocket.accept()

        if symbol not in self.active_connections:
            self.active_connections[symbol] = []
        self.active_connections[symbol].append(websocket)

        logger.info(f"Client connected for {symbol} streaming")

    def disconnect(self, websocket: WebSocket, symbol: str):
        """Remove WebSocket connection"""
        if symbol in self.active_connections:
            self.active_connections[symbol].remove(websocket)
            if not self.active_connections[symbol]:
                del self.active_connections[symbol]

        logger.info(f"Client disconnected from {symbol} streaming")

    async def broadcast(self, symbol: str, message: Dict):
        """Broadcast message to all clients subscribed to symbol"""
        if symbol not in self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections[symbol]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection, symbol)


manager = ConnectionManager()


@router.websocket("/ws/stream/{symbol}")
async def market_data_stream(websocket: WebSocket, symbol: str):
    """
    WebSocket endpoint for real-time market data streaming.

    Streams aggregated quotes with <200ms latency.

    Example client code:
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/api/market-data/ws/stream/AAPL');
    ws.onmessage = (event) => {
        const quote = JSON.parse(event.data);
        console.log('Real-time quote:', quote);
    };
    ```
    """
    if not data_aggregator:
        await websocket.close(code=1011, reason="Data aggregator not initialized")
        return

    symbol = symbol.upper()
    await manager.connect(websocket, symbol)

    # Define callback for quote updates
    async def quote_callback(quote: AggregatedQuote):
        await manager.broadcast(symbol, {
            "type": "quote",
            "symbol": quote.symbol,
            "best_bid": quote.best_bid,
            "best_ask": quote.best_ask,
            "mid_price": quote.mid_price,
            "spread_bps": quote.spread_bps,
            "last": quote.last,
            "timestamp": quote.timestamp.isoformat(),
            "num_providers": quote.num_providers,
            "latency_ms": quote.avg_latency_ms
        })

    # Subscribe to data aggregator
    await data_aggregator.subscribe(symbol, quote_callback)

    try:
        # Keep connection alive
        while True:
            # Wait for client messages (e.g., ping/pong)
            data = await websocket.receive_text()

            if data == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})

    except WebSocketDisconnect:
        manager.disconnect(websocket, symbol)
        logger.info(f"Client disconnected from {symbol} stream")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, symbol)


@router.websocket("/ws/stream-multi")
async def multi_symbol_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming multiple symbols.

    Client sends: {"action": "subscribe", "symbols": ["AAPL", "MSFT"]}
    Server streams: Real-time quotes for all subscribed symbols

    Client sends: {"action": "unsubscribe", "symbols": ["AAPL"]}
    """
    if not data_aggregator:
        await websocket.close(code=1011, reason="Data aggregator not initialized")
        return

    await websocket.accept()
    subscribed_symbols: set = set()

    try:
        while True:
            # Receive subscription commands
            data = await websocket.receive_text()
            message = json.loads(data)

            action = message.get("action")
            symbols = message.get("symbols", [])

            if action == "subscribe":
                for symbol in symbols:
                    symbol = symbol.upper()
                    if symbol not in subscribed_symbols:
                        subscribed_symbols.add(symbol)

                        # Define callback
                        async def quote_callback(quote: AggregatedQuote, ws=websocket):
                            await ws.send_json({
                                "type": "quote",
                                "symbol": quote.symbol,
                                "best_bid": quote.best_bid,
                                "best_ask": quote.best_ask,
                                "mid_price": quote.mid_price,
                                "last": quote.last,
                                "timestamp": quote.timestamp.isoformat(),
                                "latency_ms": quote.avg_latency_ms
                            })

                        await data_aggregator.subscribe(symbol, quote_callback)

                await websocket.send_json({
                    "type": "subscribed",
                    "symbols": list(subscribed_symbols),
                    "count": len(subscribed_symbols)
                })

            elif action == "unsubscribe":
                for symbol in symbols:
                    symbol = symbol.upper()
                    subscribed_symbols.discard(symbol)

                await websocket.send_json({
                    "type": "unsubscribed",
                    "symbols": list(subscribed_symbols),
                    "count": len(subscribed_symbols)
                })

            elif action == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})

    except WebSocketDisconnect:
        logger.info(f"Multi-symbol client disconnected")

    except Exception as e:
        logger.error(f"Multi-symbol WebSocket error: {e}")
        await websocket.close()


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health")
async def health_check() -> Dict:
    """
    Health check endpoint for market data service.

    Returns service status and data quality metrics.
    """
    if not data_aggregator:
        return {
            "status": "unavailable",
            "message": "Data aggregator not initialized"
        }

    provider_status = data_aggregator.get_provider_status()
    connected_count = sum(provider_status.values())
    latency_stats = data_aggregator.get_latency_stats()

    # Calculate average p95 latency across all providers
    avg_p95_latency = 0
    if latency_stats:
        avg_p95_latency = sum(stats["p95_ms"] for stats in latency_stats.values()) / len(latency_stats)

    # Determine health status
    if connected_count == 0:
        status = "critical"
        message = "No data providers connected"
    elif connected_count < 2:
        status = "degraded"
        message = f"Only {connected_count} provider connected (redundancy at risk)"
    elif avg_p95_latency > 500:
        status = "degraded"
        message = f"High latency detected: {avg_p95_latency:.0f}ms p95"
    else:
        status = "healthy"
        message = f"{connected_count} providers connected, {avg_p95_latency:.0f}ms p95 latency"

    return {
        "status": status,
        "message": message,
        "providers_connected": connected_count,
        "providers_total": len(provider_status),
        "avg_p95_latency_ms": round(avg_p95_latency, 2) if avg_p95_latency else None,
        "target_latency_ms": 200,
        "timestamp": datetime.now().isoformat()
    }
