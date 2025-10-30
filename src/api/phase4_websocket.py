"""
Phase 4 Metrics WebSocket Streaming

Implements WS /ws/phase4-metrics/{user_id} endpoint per frontend integration plan.
Streams real-time Phase 4 technical signals every 30 seconds.

Performance targets:
- Update interval: 30s
- Compute latency: <200ms per asset
- Smooth reconnection with heartbeat

Phase 4 Metrics (2x2 grid):
- options_flow_composite: PCR + IV skew + volume (-1 to +1)
- residual_momentum: Idiosyncratic momentum z-score
- seasonality_score: Calendar patterns (-1 to +1)
- breadth_liquidity: Market internals (-1 to +1)
"""

import logging
import asyncio
import os
from typing import Dict, Any
from datetime import datetime, timezone
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np

from ..agents.swarm.mcp_tools import JarvisMCPTools

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

# Configurable update interval (env override for testing)
WS_UPDATE_INTERVAL = float(os.getenv("PHASE4_WS_INTERVAL_SECONDS", "30"))


class Phase4StreamManager:
    """Manages Phase 4 metric streaming for connected clients"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.update_interval = WS_UPDATE_INTERVAL  # seconds (configurable)
        
    async def connect(self, user_id: str, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"✓ WebSocket connected: user_id={user_id}")
        
    def disconnect(self, user_id: str):
        """Remove WebSocket connection"""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            logger.info(f"✗ WebSocket disconnected: user_id={user_id}")
    
    async def send_phase4_update(self, user_id: str, data: Dict[str, Any]):
        """Send Phase 4 metrics update to client"""
        if user_id in self.active_connections:
            websocket = self.active_connections[user_id]
            try:
                await websocket.send_json({
                    "type": "phase4_update",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": data
                })
            except Exception as e:
                logger.error(f"❌ Error sending update to {user_id}: {e}")
                self.disconnect(user_id)
    
    async def send_heartbeat(self, user_id: str):
        """Send heartbeat to keep connection alive"""
        if user_id in self.active_connections:
            websocket = self.active_connections[user_id]
            try:
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            except Exception as e:
                logger.error(f"❌ Heartbeat failed for {user_id}: {e}")
                self.disconnect(user_id)
    
    async def compute_phase4_metrics(self, user_id: str) -> Dict[str, Any]:
        """
        Compute Phase 4 metrics for user's portfolio.
        
        In production, this would:
        1. Fetch user's portfolio positions
        2. Get latest market data
        3. Compute Phase 4 metrics via MCP tools
        4. Return structured data
        
        For now, returns synthetic data with realistic values.
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Generate synthetic returns (in production, fetch real data)
            asset_returns = np.random.normal(0.001, 0.02, 100)
            market_returns = np.random.normal(0.0008, 0.015, 100)
            
            # Compute Phase 4 metrics
            metrics = JarvisMCPTools.compute_phase4_metrics(
                asset_returns=asset_returns.tolist(),
                market_returns=market_returns.tolist()
            )
            
            # Add compute latency
            elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            metrics['compute_latency_ms'] = round(elapsed_ms, 2)
            
            logger.info(f"✓ Phase 4 computed in {elapsed_ms:.2f}ms for user {user_id}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error computing Phase 4 for {user_id}: {e}")
            return {
                'error': str(e),
                'options_flow_composite': None,
                'residual_momentum': None,
                'seasonality_score': None,
                'breadth_liquidity': None
            }


# Global stream manager
stream_manager = Phase4StreamManager()


@router.websocket("/ws/phase4-metrics/{user_id}")
async def phase4_metrics_stream(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for real-time Phase 4 metrics streaming.
    
    Protocol:
    1. Client connects to /ws/phase4-metrics/{user_id}
    2. Server sends initial phase4_update message
    3. Server sends phase4_update every 30 seconds
    4. Server sends heartbeat every 10 seconds between updates
    5. Client can send {"type": "ping"} to check connection
    6. Server responds with {"type": "pong"}
    
    Message format (phase4_update):
    {
        "type": "phase4_update",
        "timestamp": "2024-10-19T12:00:00+00:00",
        "data": {
            "options_flow_composite": 0.35,
            "residual_momentum": 1.2,
            "seasonality_score": -0.15,
            "breadth_liquidity": 0.6,
            "compute_latency_ms": 145.23,
            "as_of": "2024-10-19T12:00:00+00:00"
        }
    }
    """
    await stream_manager.connect(user_id, websocket)
    
    try:
        # Send initial update immediately
        metrics = await stream_manager.compute_phase4_metrics(user_id)
        await stream_manager.send_phase4_update(user_id, metrics)
        
        # Start streaming loop
        last_update = datetime.now(timezone.utc)
        
        while True:
            try:
                # Check for client messages (ping, etc.)
                message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=10.0  # 10s timeout for heartbeat
                )
                
                if message.get('type') == 'ping':
                    await websocket.send_json({
                        'type': 'pong',
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
                    
            except asyncio.TimeoutError:
                # No message received, send heartbeat or update
                now = datetime.now(timezone.utc)
                elapsed = (now - last_update).total_seconds()
                
                if elapsed >= stream_manager.update_interval:
                    # Time for Phase 4 update
                    metrics = await stream_manager.compute_phase4_metrics(user_id)
                    await stream_manager.send_phase4_update(user_id, metrics)
                    last_update = now
                else:
                    # Send heartbeat
                    await stream_manager.send_heartbeat(user_id)
                    
    except WebSocketDisconnect:
        stream_manager.disconnect(user_id)
        logger.info(f"Client disconnected: {user_id}")
        
    except Exception as e:
        logger.error(f"❌ WebSocket error for {user_id}: {e}")
        stream_manager.disconnect(user_id)

