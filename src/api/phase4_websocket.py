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

Memory Leak Prevention (added):
- Proper connection tracking with timeouts
- Per-user connection limit (max 3)
- Idle timeout (5 minutes)
- Max lifetime (30 minutes)
- Heartbeat with ping/pong
- Background cleanup task for stale connections
- Guaranteed cleanup in finally block
"""

import logging
import asyncio
import os
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np

from ..agents.swarm.mcp_tools import JarvisMCPTools
from .websocket_manager import (
    WebSocketConnectionManager,
    get_phase4_ws_manager,
    ConnectionInfo
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

# Configurable update interval (env override for testing)
WS_UPDATE_INTERVAL = float(os.getenv("PHASE4_WS_INTERVAL_SECONDS", "30"))


class Phase4StreamManager:
    """
    Manages Phase 4 metric streaming for connected clients.

    UPDATED: Now uses WebSocketConnectionManager for proper memory leak prevention:
    - Per-user connection limits (max 3 connections)
    - Idle timeout (5 minutes)
    - Max lifetime (30 minutes)
    - Heartbeat mechanism with ping/pong
    - Background cleanup of stale connections
    """

    def __init__(self):
        # Use the singleton WebSocket manager for proper connection tracking
        self.ws_manager = get_phase4_ws_manager()
        self.update_interval = WS_UPDATE_INTERVAL  # seconds (configurable)

    async def connect(self, user_id: str, websocket: WebSocket) -> ConnectionInfo:
        """
        Accept new WebSocket connection with proper tracking.

        Returns:
            ConnectionInfo object for the new connection
        """
        conn = await self.ws_manager.connect(user_id, websocket)
        logger.info(f"Phase4 WebSocket connected: user_id={user_id}, conn_id={conn.connection_id}")
        return conn

    async def disconnect(self, user_id: str, websocket: WebSocket):
        """Remove WebSocket connection with proper cleanup"""
        await self.ws_manager.disconnect(user_id, websocket)
        logger.info(f"Phase4 WebSocket disconnected: user_id={user_id}")

    async def send_phase4_update(self, websocket: WebSocket, user_id: str, data: Dict[str, Any]) -> bool:
        """
        Send Phase 4 metrics update to client.

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            await websocket.send_json({
                "type": "phase4_update",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": data
            })
            return True
        except Exception as e:
            logger.error(f"Error sending Phase4 update to {user_id}: {e}")
            return False

    async def send_heartbeat(self, websocket: WebSocket, user_id: str) -> bool:
        """
        Send heartbeat ping to keep connection alive.

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            await websocket.send_json({
                "type": "ping",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return True
        except Exception as e:
            logger.error(f"Heartbeat failed for {user_id}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return self.ws_manager.get_stats()

    async def compute_phase4_metrics(self, user_id: str) -> Dict[str, Any]:
        """
        Compute Phase 4 metrics for user's portfolio using REAL data.
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            from ..data.position_manager import PositionManager
            from ..agents.swarm.mcp_tools import JarvisMCPTools
            import pandas as pd
            
            # 1. Fetch real positions
            position_manager = PositionManager()
            positions = position_manager.get_all_stock_positions()
            
            if not positions:
                # Fallback to SPY if no positions
                positions = [{'symbol': 'SPY', 'quantity': 1}]
            
            # 2. Fetch market data (SPY)
            market_history = JarvisMCPTools.get_price_history('SPY', days=100)
            market_returns = market_history.get('returns', [])
            
            # 3. Compute portfolio returns (weighted)
            portfolio_returns = np.zeros(len(market_returns)) if market_returns else []
            total_value = 0
            
            has_data = False
            
            for pos in positions:
                symbol = pos.get('symbol') if isinstance(pos, dict) else pos.symbol
                quantity = pos.get('quantity', 1) if isinstance(pos, dict) else pos.quantity
                
                # Get history
                history = JarvisMCPTools.get_price_history(symbol, days=100)
                
                if history.get('success') and history.get('returns'):
                    returns = np.array(history['returns'])
                    current_price = history.get('current_price', 0)
                    market_value = current_price * quantity
                    
                    # Align lengths if needed (simple truncation for now)
                    min_len = min(len(returns), len(portfolio_returns))
                    if min_len > 0:
                        portfolio_returns = portfolio_returns[:min_len] + (returns[:min_len] * market_value)
                        total_value += market_value
                        has_data = True
            
            if has_data and total_value > 0:
                portfolio_returns = (portfolio_returns / total_value).tolist()
            else:
                # Fallback if data fetching failed
                portfolio_returns = market_returns
            
            # 4. Fetch options flow metrics for the primary asset (largest position)
            # Find largest position by market value (approx)
            primary_symbol = positions[0].get('symbol') if isinstance(positions[0], dict) else positions[0].symbol
            # (In a real implementation, we'd sort by value)
            
            flow_metrics = JarvisMCPTools.get_options_flow_metrics(primary_symbol)
            
            pcr = flow_metrics.get('pcr') if flow_metrics.get('success') else None
            iv_skew = flow_metrics.get('iv_skew') if flow_metrics.get('success') else None
            volume_ratio = flow_metrics.get('volume_ratio') if flow_metrics.get('success') else None
            
            # 5. Compute Phase 4 metrics
            metrics = JarvisMCPTools.compute_phase4_metrics(
                asset_returns=portfolio_returns,
                market_returns=market_returns,
                pcr=pcr,
                iv_skew=iv_skew,
                volume_ratio=volume_ratio
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


@router.get("/ws/phase4-metrics/stats")
async def get_phase4_ws_stats():
    """
    Get WebSocket connection statistics for Phase 4 metrics.

    Returns connection count, user count, and configuration.
    Useful for monitoring and debugging memory leaks.
    """
    return stream_manager.get_stats()


@router.on_event("startup")
async def start_phase4_cleanup_task():
    """Start the background cleanup task for stale connections"""
    await stream_manager.ws_manager.start_cleanup_task()
    logger.info("Phase4 WebSocket cleanup task started")


@router.on_event("shutdown")
async def stop_phase4_cleanup_task():
    """Stop the background cleanup task"""
    await stream_manager.ws_manager.stop_cleanup_task()
    logger.info("Phase4 WebSocket cleanup task stopped")


@router.websocket("/ws/phase4-metrics/{user_id}")
async def phase4_metrics_stream(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for real-time Phase 4 metrics streaming.

    Features (Memory Leak Prevention):
    - Proper connection tracking with timeouts
    - Per-user connection limit (max 3)
    - Idle timeout (5 minutes)
    - Max lifetime (30 minutes)
    - Heartbeat with ping/pong
    - Guaranteed cleanup in finally block

    Protocol:
    1. Client connects to /ws/phase4-metrics/{user_id}
    2. Server sends initial phase4_update message
    3. Server sends phase4_update every 30 seconds
    4. Server sends ping every 30 seconds (heartbeat)
    5. Client should respond with {"type": "pong"}
    6. Client can send {"type": "ping"} to check connection
    7. Server responds with {"type": "pong"}
    8. Connection closes on disconnect, error, or timeout

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
    logger.info(f"WebSocket connection attempt for user: {user_id}")

    # Connect with proper tracking (handles connection limits)
    conn = await stream_manager.connect(user_id, websocket)

    try:
        # Send initial update immediately
        metrics = await stream_manager.compute_phase4_metrics(user_id)
        if not await stream_manager.send_phase4_update(websocket, user_id, metrics):
            logger.warning(f"Failed to send initial metrics to {user_id}")
            return

        # Start streaming loop with proper timeout handling
        last_update = time.time()
        last_heartbeat = time.time()
        heartbeat_interval = stream_manager.ws_manager.heartbeat_interval_seconds

        while True:
            try:
                # Check for client messages (ping, etc.) with short timeout
                message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=1.0  # Short timeout for responsive loop
                )

                # Update activity on any message
                conn.update_activity()
                conn.messages_received += 1

                # Handle client messages
                if isinstance(message, dict):
                    msg_type = message.get('type', '')

                    if msg_type == 'ping':
                        # Client requesting ping - respond with pong
                        await websocket.send_json({
                            'type': 'pong',
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        })
                        conn.messages_sent += 1

                    elif msg_type == 'pong':
                        # Client responding to our heartbeat
                        stream_manager.ws_manager.handle_pong(conn)

                    elif msg_type == 'refresh':
                        # Client requesting immediate refresh
                        metrics = await stream_manager.compute_phase4_metrics(user_id)
                        await stream_manager.send_phase4_update(websocket, user_id, metrics)
                        conn.messages_sent += 1
                        last_update = time.time()

            except asyncio.TimeoutError:
                # No message from client - check if we need to send updates
                now = time.time()

                # Check if we need to send a heartbeat
                if now - last_heartbeat >= heartbeat_interval:
                    if not await stream_manager.send_heartbeat(websocket, user_id):
                        logger.warning(f"Failed to send heartbeat to {user_id}")
                        break
                    conn.messages_sent += 1
                    last_heartbeat = now
                    logger.debug(f"Sent heartbeat to {user_id}")

                # Check if we need to send Phase 4 update
                if now - last_update >= stream_manager.update_interval:
                    try:
                        metrics = await stream_manager.compute_phase4_metrics(user_id)
                        if not await stream_manager.send_phase4_update(websocket, user_id, metrics):
                            logger.warning(f"Failed to send Phase4 update to {user_id}")
                            break
                        conn.messages_sent += 1
                        last_update = now
                    except Exception as e:
                        logger.error(f"Failed to compute/send Phase4 metrics to {user_id}: {e}")
                        break

                # Check connection health (idle timeout, max lifetime)
                ws_manager = stream_manager.ws_manager
                if conn.is_stale(ws_manager.idle_timeout_seconds, ws_manager.max_lifetime_seconds):
                    logger.info(f"Connection timed out for {user_id}")
                    try:
                        await websocket.send_json({
                            "type": "timeout",
                            "message": "Connection timed out due to inactivity",
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                    except Exception:
                        pass
                    break

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {user_id}: {e}", exc_info=True)
    finally:
        # CRITICAL: Always clean up connection, even on error
        # This prevents memory leaks from abandoned connections
        await stream_manager.disconnect(user_id, websocket)
        logger.debug(f"Phase4 connection cleanup complete for {user_id}")

