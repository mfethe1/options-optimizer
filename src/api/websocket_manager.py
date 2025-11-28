"""
WebSocket Connection Manager with Memory Leak Prevention

This module provides a robust WebSocket connection manager that handles:
- Connection tracking with metadata (timestamps, client state)
- Connection timeouts (max lifetime and idle timeout)
- Per-user connection limits (automatically closes oldest connection)
- Periodic cleanup of stale/dead connections
- Heartbeat mechanism with ping/pong
- Proper error handling and guaranteed cleanup

Usage:
    manager = WebSocketConnectionManager(
        max_connections_per_user=3,
        idle_timeout_seconds=300,  # 5 minutes
        max_lifetime_seconds=1800,  # 30 minutes
        heartbeat_interval_seconds=30,
        heartbeat_timeout_seconds=10
    )

    # In your WebSocket endpoint:
    @router.websocket("/ws/example/{user_id}")
    async def websocket_endpoint(websocket: WebSocket, user_id: str):
        async with manager.connection_context(user_id, websocket) as conn:
            # Connection is automatically managed
            while True:
                data = await conn.receive_with_timeout()
                if data is None:
                    break  # Timeout or disconnect
                # Process data...
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, Awaitable
from contextlib import asynccontextmanager
from fastapi import WebSocket
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection state tracking"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    IDLE = "idle"
    WAITING_PONG = "waiting_pong"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"


@dataclass
class ConnectionInfo:
    """Metadata for a WebSocket connection"""
    websocket: WebSocket
    user_id: str
    connected_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    last_heartbeat_sent: float = 0.0
    last_pong_received: float = field(default_factory=time.time)
    state: ConnectionState = ConnectionState.CONNECTED
    messages_sent: int = 0
    messages_received: int = 0
    connection_id: str = ""

    def __post_init__(self):
        if not self.connection_id:
            self.connection_id = f"{self.user_id}_{id(self.websocket)}_{int(self.connected_at * 1000)}"

    def update_activity(self):
        """Mark connection as active"""
        self.last_activity = time.time()
        self.state = ConnectionState.CONNECTED

    def is_stale(self, idle_timeout: float, max_lifetime: float) -> bool:
        """Check if connection should be closed due to timeout"""
        now = time.time()

        # Check max lifetime
        if now - self.connected_at > max_lifetime:
            logger.info(f"Connection {self.connection_id} exceeded max lifetime ({max_lifetime}s)")
            return True

        # Check idle timeout
        if now - self.last_activity > idle_timeout:
            logger.info(f"Connection {self.connection_id} exceeded idle timeout ({idle_timeout}s)")
            return True

        return False

    def is_websocket_closed(self) -> bool:
        """Check if the underlying WebSocket is closed"""
        try:
            return self.websocket.client_state == WebSocketState.DISCONNECTED
        except Exception:
            return True


class WebSocketConnectionManager:
    """
    Thread-safe WebSocket connection manager with automatic cleanup.

    Features:
    - Per-user connection limits with LRU eviction
    - Idle and lifetime timeout enforcement
    - Background cleanup task
    - Heartbeat mechanism
    - Proper resource cleanup on errors
    """

    def __init__(
        self,
        max_connections_per_user: int = 3,
        idle_timeout_seconds: float = 300.0,  # 5 minutes
        max_lifetime_seconds: float = 1800.0,  # 30 minutes
        heartbeat_interval_seconds: float = 30.0,
        heartbeat_timeout_seconds: float = 10.0,
        cleanup_interval_seconds: float = 60.0,
        name: str = "default"
    ):
        self.max_connections_per_user = max_connections_per_user
        self.idle_timeout_seconds = idle_timeout_seconds
        self.max_lifetime_seconds = max_lifetime_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.name = name

        # Connection storage: user_id -> list of ConnectionInfo
        self._connections: Dict[str, List[ConnectionInfo]] = {}
        self._lock = asyncio.Lock()

        # Cleanup task handle
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        logger.info(
            f"[{self.name}] WebSocketConnectionManager initialized: "
            f"max_per_user={max_connections_per_user}, "
            f"idle_timeout={idle_timeout_seconds}s, "
            f"max_lifetime={max_lifetime_seconds}s"
        )

    async def start_cleanup_task(self):
        """Start the background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._shutdown_event.clear()
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info(f"[{self.name}] Started cleanup task")

    async def stop_cleanup_task(self):
        """Stop the background cleanup task"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._shutdown_event.set()
            try:
                await asyncio.wait_for(self._cleanup_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            logger.info(f"[{self.name}] Stopped cleanup task")

    async def _cleanup_loop(self):
        """Background task that periodically cleans up stale connections"""
        logger.info(f"[{self.name}] Cleanup loop started (interval: {self.cleanup_interval_seconds}s)")

        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.cleanup_interval_seconds
                )
                # If we reach here, shutdown was requested
                break
            except asyncio.TimeoutError:
                # Normal timeout - time to clean up
                await self._cleanup_stale_connections()

    async def _cleanup_stale_connections(self):
        """Remove connections that are closed, timed out, or stale"""
        connections_removed = 0

        async with self._lock:
            users_to_remove: List[str] = []

            for user_id, connections in self._connections.items():
                connections_to_remove: List[ConnectionInfo] = []

                for conn in connections:
                    should_remove = False
                    reason = ""

                    # Check if WebSocket is closed
                    if conn.is_websocket_closed():
                        should_remove = True
                        reason = "websocket_closed"

                    # Check timeouts
                    elif conn.is_stale(self.idle_timeout_seconds, self.max_lifetime_seconds):
                        should_remove = True
                        reason = "timeout"

                    # Check heartbeat timeout (waiting for pong too long)
                    elif conn.state == ConnectionState.WAITING_PONG:
                        if time.time() - conn.last_heartbeat_sent > self.heartbeat_timeout_seconds:
                            should_remove = True
                            reason = "heartbeat_timeout"

                    if should_remove:
                        connections_to_remove.append(conn)
                        logger.info(
                            f"[{self.name}] Marking connection for cleanup: "
                            f"user={user_id}, conn_id={conn.connection_id}, reason={reason}"
                        )

                # Close and remove stale connections
                for conn in connections_to_remove:
                    await self._close_connection_unsafe(conn)
                    connections.remove(conn)
                    connections_removed += 1

                # Mark user for removal if no connections left
                if not connections:
                    users_to_remove.append(user_id)

            # Clean up empty user entries
            for user_id in users_to_remove:
                del self._connections[user_id]

        if connections_removed > 0:
            logger.info(f"[{self.name}] Cleanup removed {connections_removed} stale connection(s)")

    async def _close_connection_unsafe(self, conn: ConnectionInfo):
        """Close a WebSocket connection (caller must hold lock)"""
        if conn.state == ConnectionState.DISCONNECTED:
            return

        conn.state = ConnectionState.DISCONNECTING

        try:
            if not conn.is_websocket_closed():
                await asyncio.wait_for(
                    conn.websocket.close(code=1000, reason="Connection closed by server"),
                    timeout=2.0
                )
        except asyncio.TimeoutError:
            logger.warning(f"[{self.name}] Timeout closing connection {conn.connection_id}")
        except Exception as e:
            logger.debug(f"[{self.name}] Error closing connection {conn.connection_id}: {e}")
        finally:
            conn.state = ConnectionState.DISCONNECTED

    async def connect(self, user_id: str, websocket: WebSocket) -> ConnectionInfo:
        """
        Register a new WebSocket connection for a user.

        If the user has reached max connections, the oldest connection is closed.

        Returns:
            ConnectionInfo for the new connection
        """
        await websocket.accept()

        async with self._lock:
            # Initialize user's connection list if needed
            if user_id not in self._connections:
                self._connections[user_id] = []

            user_connections = self._connections[user_id]

            # Enforce connection limit - close oldest if at limit
            while len(user_connections) >= self.max_connections_per_user:
                oldest = user_connections.pop(0)
                logger.info(
                    f"[{self.name}] Closing oldest connection for {user_id} "
                    f"(limit: {self.max_connections_per_user})"
                )
                await self._close_connection_unsafe(oldest)

            # Create new connection info
            conn_info = ConnectionInfo(
                websocket=websocket,
                user_id=user_id
            )
            user_connections.append(conn_info)

            logger.info(
                f"[{self.name}] Connected: user={user_id}, "
                f"conn_id={conn_info.connection_id}, "
                f"active_connections={len(user_connections)}"
            )

            return conn_info

    async def disconnect(self, user_id: str, websocket: WebSocket):
        """
        Remove a WebSocket connection for a user.

        This should be called in finally blocks to ensure cleanup.
        """
        async with self._lock:
            if user_id not in self._connections:
                return

            user_connections = self._connections[user_id]
            conn_to_remove = None

            for conn in user_connections:
                if conn.websocket is websocket:
                    conn_to_remove = conn
                    break

            if conn_to_remove:
                await self._close_connection_unsafe(conn_to_remove)
                user_connections.remove(conn_to_remove)

                logger.info(
                    f"[{self.name}] Disconnected: user={user_id}, "
                    f"conn_id={conn_to_remove.connection_id}, "
                    f"remaining={len(user_connections)}"
                )

                # Clean up empty user entry
                if not user_connections:
                    del self._connections[user_id]

    async def send_json(self, user_id: str, websocket: WebSocket, data: Any) -> bool:
        """
        Send JSON data to a specific connection.

        Returns:
            True if sent successfully, False otherwise
        """
        async with self._lock:
            conn = self._find_connection(user_id, websocket)
            if not conn:
                return False

        try:
            await websocket.send_json(data)
            async with self._lock:
                conn = self._find_connection(user_id, websocket)
                if conn:
                    conn.messages_sent += 1
                    conn.update_activity()
            return True
        except Exception as e:
            logger.error(f"[{self.name}] Error sending to {user_id}: {e}")
            return False

    async def broadcast_to_user(self, user_id: str, data: Any) -> int:
        """
        Send data to all connections for a specific user.

        Returns:
            Number of successful sends
        """
        async with self._lock:
            if user_id not in self._connections:
                return 0
            connections = list(self._connections[user_id])

        success_count = 0
        for conn in connections:
            try:
                await conn.websocket.send_json(data)
                conn.messages_sent += 1
                conn.update_activity()
                success_count += 1
            except Exception as e:
                logger.warning(f"[{self.name}] Broadcast failed to {conn.connection_id}: {e}")

        return success_count

    async def broadcast_all(self, data: Any) -> int:
        """
        Send data to all connected clients.

        Returns:
            Number of successful sends
        """
        async with self._lock:
            all_connections = [
                conn
                for connections in self._connections.values()
                for conn in connections
            ]

        success_count = 0
        for conn in all_connections:
            try:
                await conn.websocket.send_json(data)
                conn.messages_sent += 1
                conn.update_activity()
                success_count += 1
            except Exception:
                pass

        return success_count

    async def send_heartbeat(self, conn: ConnectionInfo) -> bool:
        """
        Send a heartbeat ping to a connection.

        Returns:
            True if sent successfully
        """
        try:
            await conn.websocket.send_json({
                "type": "ping",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            conn.last_heartbeat_sent = time.time()
            conn.state = ConnectionState.WAITING_PONG
            return True
        except Exception as e:
            logger.warning(f"[{self.name}] Heartbeat send failed for {conn.connection_id}: {e}")
            return False

    def handle_pong(self, conn: ConnectionInfo):
        """Handle pong response from client"""
        conn.last_pong_received = time.time()
        conn.state = ConnectionState.CONNECTED
        conn.update_activity()

    def _find_connection(self, user_id: str, websocket: WebSocket) -> Optional[ConnectionInfo]:
        """Find connection info (caller must hold lock)"""
        if user_id not in self._connections:
            return None

        for conn in self._connections[user_id]:
            if conn.websocket is websocket:
                return conn
        return None

    def get_connection_count(self, user_id: Optional[str] = None) -> int:
        """Get number of active connections"""
        if user_id:
            return len(self._connections.get(user_id, []))
        return sum(len(conns) for conns in self._connections.values())

    def get_all_users(self) -> List[str]:
        """Get list of all connected users"""
        return list(self._connections.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        total_connections = 0
        total_messages_sent = 0
        total_messages_received = 0
        oldest_connection_age = 0

        now = time.time()

        for connections in self._connections.values():
            for conn in connections:
                total_connections += 1
                total_messages_sent += conn.messages_sent
                total_messages_received += conn.messages_received
                age = now - conn.connected_at
                if age > oldest_connection_age:
                    oldest_connection_age = age

        return {
            "manager_name": self.name,
            "total_connections": total_connections,
            "unique_users": len(self._connections),
            "total_messages_sent": total_messages_sent,
            "total_messages_received": total_messages_received,
            "oldest_connection_age_seconds": round(oldest_connection_age, 1),
            "config": {
                "max_connections_per_user": self.max_connections_per_user,
                "idle_timeout_seconds": self.idle_timeout_seconds,
                "max_lifetime_seconds": self.max_lifetime_seconds,
                "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            }
        }

    @asynccontextmanager
    async def connection_context(self, user_id: str, websocket: WebSocket):
        """
        Context manager for WebSocket connections with guaranteed cleanup.

        Usage:
            async with manager.connection_context(user_id, websocket) as conn:
                # Use connection
                pass
            # Connection is automatically cleaned up
        """
        conn = await self.connect(user_id, websocket)
        try:
            yield conn
        finally:
            await self.disconnect(user_id, websocket)


class ManagedWebSocketConnection:
    """
    Helper class for managing a single WebSocket connection with
    timeout handling, heartbeat, and proper cleanup.
    """

    def __init__(
        self,
        conn: ConnectionInfo,
        manager: WebSocketConnectionManager,
        receive_timeout: float = 300.0  # 5 minutes default
    ):
        self.conn = conn
        self.manager = manager
        self.receive_timeout = receive_timeout
        self._last_heartbeat_check = time.time()

    async def receive_with_timeout(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Receive a message with timeout handling.

        Automatically sends heartbeat pings during long waits.
        Returns None on timeout or disconnect.
        """
        timeout = timeout or self.receive_timeout

        try:
            # Wait for message with timeout
            data = await asyncio.wait_for(
                self.conn.websocket.receive_json(),
                timeout=timeout
            )

            self.conn.messages_received += 1
            self.conn.update_activity()

            # Handle pong responses
            if isinstance(data, dict) and data.get("type") == "pong":
                self.manager.handle_pong(self.conn)
                # Continue waiting for actual message
                return await self.receive_with_timeout(timeout)

            return data

        except asyncio.TimeoutError:
            # Check if we should send a heartbeat
            now = time.time()
            if now - self._last_heartbeat_check >= self.manager.heartbeat_interval_seconds:
                self._last_heartbeat_check = now
                if await self.manager.send_heartbeat(self.conn):
                    # Heartbeat sent, wait for pong
                    try:
                        pong = await asyncio.wait_for(
                            self.conn.websocket.receive_json(),
                            timeout=self.manager.heartbeat_timeout_seconds
                        )
                        if isinstance(pong, dict) and pong.get("type") == "pong":
                            self.manager.handle_pong(self.conn)
                            return await self.receive_with_timeout(timeout)
                    except asyncio.TimeoutError:
                        logger.warning(f"Heartbeat timeout for {self.conn.connection_id}")
                        return None

            # Return None to signal timeout (caller can decide to retry or close)
            return None

        except Exception as e:
            logger.debug(f"Receive error for {self.conn.connection_id}: {e}")
            return None

    async def send_json(self, data: Any) -> bool:
        """Send JSON data"""
        return await self.manager.send_json(
            self.conn.user_id,
            self.conn.websocket,
            data
        )


# Singleton managers for common use cases
_unified_manager: Optional[WebSocketConnectionManager] = None
_phase4_manager: Optional[WebSocketConnectionManager] = None


def get_unified_ws_manager() -> WebSocketConnectionManager:
    """Get the singleton WebSocket manager for unified predictions"""
    global _unified_manager
    if _unified_manager is None:
        _unified_manager = WebSocketConnectionManager(
            max_connections_per_user=3,
            idle_timeout_seconds=300.0,
            max_lifetime_seconds=1800.0,
            heartbeat_interval_seconds=30.0,
            heartbeat_timeout_seconds=10.0,
            cleanup_interval_seconds=60.0,
            name="unified_predictions"
        )
    return _unified_manager


def get_phase4_ws_manager() -> WebSocketConnectionManager:
    """Get the singleton WebSocket manager for Phase 4 metrics"""
    global _phase4_manager
    if _phase4_manager is None:
        _phase4_manager = WebSocketConnectionManager(
            max_connections_per_user=3,
            idle_timeout_seconds=300.0,
            max_lifetime_seconds=1800.0,
            heartbeat_interval_seconds=30.0,
            heartbeat_timeout_seconds=10.0,
            cleanup_interval_seconds=60.0,
            name="phase4_metrics"
        )
    return _phase4_manager
