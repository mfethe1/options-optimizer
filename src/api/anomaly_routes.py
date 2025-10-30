"""
Real-Time Anomaly Detection API Routes

Detect unusual market activity that precedes major moves.
"""
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import asyncio
import json

from ..agents.swarm.agents.real_time_anomaly_agent import RealTimeAnomalyAgent
from ..agents.swarm.shared_context import SharedContext

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/anomalies",
    tags=["Anomaly Detection"]
)

# Initialize agent
shared_context = SharedContext()
anomaly_agent = RealTimeAnomalyAgent(agent_id="anomaly_detector", shared_context=shared_context)


# WebSocket connection manager for real-time alerts
class AnomalyAlertManager:
    """Manage WebSocket connections for real-time anomaly alerts"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_subscriptions: Dict[str, List[str]] = {}  # user_id -> list of symbols

    async def connect(self, websocket: WebSocket, user_id: str):
        """Connect a user's WebSocket"""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"User {user_id} connected to anomaly alerts")

    def disconnect(self, user_id: str):
        """Disconnect a user's WebSocket"""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        if user_id in self.user_subscriptions:
            del self.user_subscriptions[user_id]
        logger.info(f"User {user_id} disconnected from anomaly alerts")

    def subscribe(self, user_id: str, symbols: List[str]):
        """Subscribe user to anomaly alerts for specific symbols"""
        self.user_subscriptions[user_id] = symbols
        logger.info(f"User {user_id} subscribed to: {', '.join(symbols)}")

    async def send_alert(self, user_id: str, anomaly: Dict[str, Any]):
        """Send anomaly alert to specific user"""
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json({
                    "type": "anomaly_alert",
                    "data": anomaly,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error sending alert to user {user_id}: {e}")

    async def broadcast_alert(self, anomaly: Dict[str, Any]):
        """Broadcast anomaly alert to all subscribed users"""
        symbol = anomaly.get('symbol')

        for user_id, subscribed_symbols in self.user_subscriptions.items():
            if symbol in subscribed_symbols or '*' in subscribed_symbols:  # '*' = all symbols
                await self.send_alert(user_id, anomaly)


alert_manager = AnomalyAlertManager()


# Request/Response models
class AnomalyDetectionRequest(BaseModel):
    """Request for anomaly detection"""
    symbol: str = Field(..., description="Stock symbol to analyze")
    detection_types: Optional[List[str]] = Field(
        None,
        description="Types to detect: volume, price, iv, options_flow. If None, detects all."
    )


class AnomalyDetectionResponse(BaseModel):
    """Response with detected anomalies"""
    symbol: str
    anomalies: List[Dict[str, Any]] = Field(..., description="List of detected anomalies")
    count: int = Field(..., description="Number of anomalies detected")
    timestamp: str


class AnomalySubscriptionRequest(BaseModel):
    """Request to subscribe to anomaly alerts"""
    symbols: List[str] = Field(..., description="List of symbols to monitor (use ['*'] for all)")


# Routes

@router.post("/detect", response_model=AnomalyDetectionResponse)
async def detect_anomalies(request: AnomalyDetectionRequest):
    """
    Detect real-time anomalies for a symbol

    **COMPETITIVE ADVANTAGE**: Real-time statistical anomaly detection

    Detects:
    - **Volume spikes**: 3+ standard deviations above average
    - **Price anomalies**: Unusual intraday gaps and movements (2.5σ)
    - **IV expansion**: Rapid volatility increases (2σ)
    - **Options flow**: Block trades, unusual strikes, smart money activity

    Research shows anomalies often precede major moves:
    - Volume spikes → 60%+ chance of continuation
    - IV expansion → earnings or news catalyst likely
    - Block trades → institutional positioning

    Args:
        request: Symbol and detection types

    Returns:
        List of detected anomalies with severity, z-scores, and trading implications
    """
    try:
        logger.info(f"Detecting anomalies for {request.symbol}")

        # Fetch market data (in production, this would be real-time)
        from ..data.providers import get_market_data
        market_data = get_market_data(request.symbol)

        if not market_data:
            raise HTTPException(status_code=404, detail=f"No market data found for {request.symbol}")

        # Detect anomalies
        anomalies = []
        detection_types = request.detection_types or ['volume', 'price', 'iv', 'options_flow']

        if 'volume' in detection_types:
            volume_anomaly = anomaly_agent._detect_volume_anomaly(request.symbol, market_data)
            if volume_anomaly:
                anomalies.append(volume_anomaly)

        if 'price' in detection_types:
            price_anomaly = anomaly_agent._detect_price_anomaly(request.symbol, market_data)
            if price_anomaly:
                anomalies.append(price_anomaly)

        if 'iv' in detection_types:
            iv_anomaly = anomaly_agent._detect_iv_anomaly(request.symbol, market_data)
            if iv_anomaly:
                anomalies.append(iv_anomaly)

        if 'options_flow' in detection_types:
            flow_anomaly = anomaly_agent._detect_options_flow_anomaly(request.symbol, market_data)
            if flow_anomaly:
                anomalies.append(flow_anomaly)

        logger.info(f"Detected {len(anomalies)} anomalies for {request.symbol}")

        # Broadcast to WebSocket subscribers
        if anomalies:
            for anomaly in anomalies:
                await alert_manager.broadcast_alert({
                    'symbol': request.symbol,
                    'anomaly': anomaly
                })

        return AnomalyDetectionResponse(
            symbol=request.symbol,
            anomalies=anomalies,
            count=len(anomalies),
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to detect anomalies: {str(e)}")


@router.post("/scan")
async def scan_multiple_symbols(
    symbols: List[str],
    detection_types: Optional[List[str]] = None
):
    """
    Scan multiple symbols for anomalies

    Efficient batch detection across watchlist.

    Args:
        symbols: List of symbols to scan (max 50)
        detection_types: Optional filter for detection types

    Returns:
        Dictionary mapping symbols to their anomalies
    """
    try:
        if len(symbols) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 symbols per scan")

        logger.info(f"Scanning {len(symbols)} symbols for anomalies")

        results = {}

        # Process each symbol
        for symbol in symbols:
            try:
                detection_request = AnomalyDetectionRequest(
                    symbol=symbol,
                    detection_types=detection_types
                )
                result = await detect_anomalies(detection_request)

                if result.count > 0:
                    results[symbol] = result.anomalies

            except Exception as e:
                logger.warning(f"Error scanning {symbol}: {e}")
                continue

        return {
            "symbols_scanned": len(symbols),
            "symbols_with_anomalies": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scanning symbols: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to scan symbols: {str(e)}")


@router.get("/thresholds")
async def get_detection_thresholds():
    """
    Get current anomaly detection thresholds

    Returns:
        Dictionary of thresholds for each detection type
    """
    return {
        "thresholds": {
            "volume_spike": {
                "z_score": anomaly_agent.volume_threshold,
                "description": "Standard deviations above average volume"
            },
            "price_movement": {
                "z_score": anomaly_agent.price_threshold,
                "description": "Standard deviations from expected price movement"
            },
            "iv_expansion": {
                "z_score": anomaly_agent.iv_threshold,
                "description": "Standard deviations above average IV"
            },
            "options_flow": {
                "min_premium": anomaly_agent.flow_threshold,
                "description": "Minimum premium for block trade detection"
            }
        },
        "severity_levels": {
            "critical": "z_score > 5.0 or premium > $1M",
            "high": "z_score > 3.0 or premium > $500K",
            "medium": "z_score > 2.0 or premium > $100K"
        }
    }


# WebSocket endpoint for real-time alerts
@router.websocket("/ws/alerts/{user_id}")
async def anomaly_alerts_websocket(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for real-time anomaly alerts

    Connect to receive instant notifications when anomalies are detected.

    Usage:
        ws://localhost:8000/api/anomalies/ws/alerts/{user_id}

    After connecting:
        1. Send subscription message: {"action": "subscribe", "symbols": ["AAPL", "TSLA"]}
        2. Receive real-time alerts: {"type": "anomaly_alert", "data": {...}}

    Events sent to client:
        - anomaly_alert: New anomaly detected
        - heartbeat: Keep-alive ping every 30s
    """
    await alert_manager.connect(websocket, user_id)

    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to anomaly alerts. Send subscription message to start receiving alerts.",
            "timestamp": datetime.now().isoformat()
        })

        # Start heartbeat task
        async def send_heartbeat():
            while True:
                try:
                    await asyncio.sleep(30)
                    await websocket.send_json({
                        "type": "heartbeat",
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception:
                    break

        heartbeat_task = asyncio.create_task(send_heartbeat())

        # Listen for client messages
        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)

                action = data.get('action')

                if action == 'subscribe':
                    symbols = data.get('symbols', [])
                    alert_manager.subscribe(user_id, symbols)
                    await websocket.send_json({
                        "type": "subscribed",
                        "symbols": symbols,
                        "timestamp": datetime.now().isoformat()
                    })

                elif action == 'unsubscribe':
                    alert_manager.subscribe(user_id, [])
                    await websocket.send_json({
                        "type": "unsubscribed",
                        "timestamp": datetime.now().isoformat()
                    })

                elif action == 'ping':
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON",
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                break

    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
    finally:
        heartbeat_task.cancel()
        alert_manager.disconnect(user_id)
        logger.info(f"WebSocket closed for user {user_id}")
