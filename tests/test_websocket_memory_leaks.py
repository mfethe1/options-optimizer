"""
Tests for WebSocket memory leak prevention fixes.

Tests the WebSocketConnectionManager class and ensures:
1. Connection limits are enforced
2. Idle timeout works correctly
3. Max lifetime timeout works correctly
4. Cleanup task removes stale connections
5. Proper cleanup on disconnect/error
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from starlette.websockets import WebSocketState

from src.api.websocket_manager import (
    WebSocketConnectionManager,
    ConnectionInfo,
    ConnectionState,
    get_unified_ws_manager,
    get_phase4_ws_manager,
)


class MockWebSocket:
    """Mock WebSocket for testing"""

    def __init__(self, state=WebSocketState.CONNECTED):
        self._client_state = state
        self.accepted = False
        self.closed = False
        self.close_code = None
        self.close_reason = None
        self.sent_messages = []

    @property
    def client_state(self):
        return self._client_state

    async def accept(self):
        self.accepted = True

    async def close(self, code=1000, reason=""):
        self.closed = True
        self.close_code = code
        self.close_reason = reason
        self._client_state = WebSocketState.DISCONNECTED

    async def send_json(self, data):
        self.sent_messages.append(data)

    async def receive_json(self):
        await asyncio.sleep(10)  # Simulate waiting for message


@pytest.fixture
def ws_manager():
    """Create a fresh WebSocket manager for testing"""
    return WebSocketConnectionManager(
        max_connections_per_user=3,
        idle_timeout_seconds=60.0,
        max_lifetime_seconds=300.0,
        heartbeat_interval_seconds=30.0,
        heartbeat_timeout_seconds=10.0,
        cleanup_interval_seconds=5.0,
        name="test_manager"
    )


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket"""
    return MockWebSocket()


class TestConnectionInfo:
    """Tests for ConnectionInfo dataclass"""

    def test_connection_info_creation(self, mock_websocket):
        """Test ConnectionInfo is created with correct defaults"""
        conn = ConnectionInfo(
            websocket=mock_websocket,
            user_id="test_user"
        )

        assert conn.user_id == "test_user"
        assert conn.state == ConnectionState.CONNECTED
        assert conn.messages_sent == 0
        assert conn.messages_received == 0
        assert conn.connection_id.startswith("test_user_")

    def test_update_activity(self, mock_websocket):
        """Test activity update resets last_activity timestamp"""
        conn = ConnectionInfo(
            websocket=mock_websocket,
            user_id="test_user"
        )

        initial_activity = conn.last_activity
        time.sleep(0.01)
        conn.update_activity()

        assert conn.last_activity > initial_activity
        assert conn.state == ConnectionState.CONNECTED

    def test_is_stale_idle_timeout(self, mock_websocket):
        """Test stale detection for idle timeout"""
        conn = ConnectionInfo(
            websocket=mock_websocket,
            user_id="test_user"
        )

        # Not stale immediately
        assert not conn.is_stale(idle_timeout=60.0, max_lifetime=300.0)

        # Simulate idle time by modifying last_activity
        conn.last_activity = time.time() - 120  # 2 minutes ago

        # Should be stale after 60 second idle timeout
        assert conn.is_stale(idle_timeout=60.0, max_lifetime=300.0)

    def test_is_stale_max_lifetime(self, mock_websocket):
        """Test stale detection for max lifetime"""
        conn = ConnectionInfo(
            websocket=mock_websocket,
            user_id="test_user"
        )

        # Not stale immediately
        assert not conn.is_stale(idle_timeout=60.0, max_lifetime=300.0)

        # Simulate old connection by modifying connected_at
        conn.connected_at = time.time() - 600  # 10 minutes ago

        # Should be stale after 300 second lifetime
        assert conn.is_stale(idle_timeout=60.0, max_lifetime=300.0)

    def test_is_websocket_closed(self, mock_websocket):
        """Test WebSocket closed detection"""
        conn = ConnectionInfo(
            websocket=mock_websocket,
            user_id="test_user"
        )

        # Not closed initially
        assert not conn.is_websocket_closed()

        # Mark as disconnected
        mock_websocket._client_state = WebSocketState.DISCONNECTED

        assert conn.is_websocket_closed()


class TestWebSocketConnectionManager:
    """Tests for WebSocketConnectionManager"""

    @pytest.mark.asyncio
    async def test_connect_basic(self, ws_manager, mock_websocket):
        """Test basic connection"""
        conn = await ws_manager.connect("user1", mock_websocket)

        assert mock_websocket.accepted
        assert conn.user_id == "user1"
        assert ws_manager.get_connection_count("user1") == 1
        assert ws_manager.get_connection_count() == 1

    @pytest.mark.asyncio
    async def test_connection_limit_enforced(self, ws_manager):
        """Test that connection limit is enforced"""
        websockets = [MockWebSocket() for _ in range(5)]

        # Connect up to limit
        for i, ws in enumerate(websockets[:3]):
            await ws_manager.connect("user1", ws)

        assert ws_manager.get_connection_count("user1") == 3

        # Connect 4th - should close oldest
        await ws_manager.connect("user1", websockets[3])

        assert ws_manager.get_connection_count("user1") == 3
        assert websockets[0].closed  # Oldest should be closed

    @pytest.mark.asyncio
    async def test_disconnect(self, ws_manager, mock_websocket):
        """Test disconnect properly removes connection"""
        await ws_manager.connect("user1", mock_websocket)
        assert ws_manager.get_connection_count("user1") == 1

        await ws_manager.disconnect("user1", mock_websocket)

        assert ws_manager.get_connection_count("user1") == 0
        assert mock_websocket.closed

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent(self, ws_manager, mock_websocket):
        """Test disconnect handles non-existent connection gracefully"""
        # Should not raise
        await ws_manager.disconnect("nonexistent_user", mock_websocket)

    @pytest.mark.asyncio
    async def test_send_json(self, ws_manager, mock_websocket):
        """Test sending JSON updates connection stats"""
        await ws_manager.connect("user1", mock_websocket)

        success = await ws_manager.send_json("user1", mock_websocket, {"test": "data"})

        assert success
        assert len(mock_websocket.sent_messages) == 1
        assert mock_websocket.sent_messages[0] == {"test": "data"}

    @pytest.mark.asyncio
    async def test_broadcast_to_user(self, ws_manager):
        """Test broadcast to all user connections"""
        websockets = [MockWebSocket() for _ in range(3)]
        for ws in websockets:
            await ws_manager.connect("user1", ws)

        count = await ws_manager.broadcast_to_user("user1", {"broadcast": True})

        assert count == 3
        for ws in websockets:
            assert {"broadcast": True} in ws.sent_messages

    @pytest.mark.asyncio
    async def test_get_stats(self, ws_manager, mock_websocket):
        """Test stats reporting"""
        await ws_manager.connect("user1", mock_websocket)

        stats = ws_manager.get_stats()

        assert stats["manager_name"] == "test_manager"
        assert stats["total_connections"] == 1
        assert stats["unique_users"] == 1
        assert "config" in stats

    @pytest.mark.asyncio
    async def test_cleanup_removes_stale_connections(self, ws_manager):
        """Test cleanup task removes stale connections"""
        mock_ws = MockWebSocket()
        conn = await ws_manager.connect("user1", mock_ws)

        # Make connection stale
        conn.last_activity = time.time() - 120  # 2 minutes ago

        # Run cleanup
        await ws_manager._cleanup_stale_connections()

        assert ws_manager.get_connection_count("user1") == 0
        assert mock_ws.closed

    @pytest.mark.asyncio
    async def test_cleanup_removes_closed_websockets(self, ws_manager):
        """Test cleanup removes connections with closed WebSocket"""
        mock_ws = MockWebSocket()
        await ws_manager.connect("user1", mock_ws)

        # Simulate WebSocket being closed
        mock_ws._client_state = WebSocketState.DISCONNECTED

        # Run cleanup
        await ws_manager._cleanup_stale_connections()

        assert ws_manager.get_connection_count("user1") == 0

    @pytest.mark.asyncio
    async def test_connection_context_manager(self, ws_manager, mock_websocket):
        """Test context manager ensures cleanup"""
        async with ws_manager.connection_context("user1", mock_websocket) as conn:
            assert ws_manager.get_connection_count("user1") == 1
            assert conn.user_id == "user1"

        # Should be cleaned up after context
        assert ws_manager.get_connection_count("user1") == 0

    @pytest.mark.asyncio
    async def test_connection_context_manager_on_error(self, ws_manager, mock_websocket):
        """Test context manager cleans up on error"""
        try:
            async with ws_manager.connection_context("user1", mock_websocket):
                assert ws_manager.get_connection_count("user1") == 1
                raise ValueError("Test error")
        except ValueError:
            pass

        # Should still be cleaned up
        assert ws_manager.get_connection_count("user1") == 0

    @pytest.mark.asyncio
    async def test_heartbeat_mechanism(self, ws_manager, mock_websocket):
        """Test heartbeat sending and pong handling"""
        conn = await ws_manager.connect("user1", mock_websocket)

        # Send heartbeat
        success = await ws_manager.send_heartbeat(conn)

        assert success
        assert conn.state == ConnectionState.WAITING_PONG

        # Handle pong
        ws_manager.handle_pong(conn)

        assert conn.state == ConnectionState.CONNECTED

    @pytest.mark.asyncio
    async def test_cleanup_task_lifecycle(self, ws_manager):
        """Test cleanup task can be started and stopped"""
        await ws_manager.start_cleanup_task()
        assert ws_manager._cleanup_task is not None
        assert not ws_manager._cleanup_task.done()

        await ws_manager.stop_cleanup_task()
        assert ws_manager._cleanup_task.done() or ws_manager._cleanup_task.cancelled()


class TestSingletonManagers:
    """Test singleton manager instances"""

    def test_unified_manager_singleton(self):
        """Test unified manager is a singleton"""
        manager1 = get_unified_ws_manager()
        manager2 = get_unified_ws_manager()

        assert manager1 is manager2
        assert manager1.name == "unified_predictions"

    def test_phase4_manager_singleton(self):
        """Test Phase4 manager is a singleton"""
        manager1 = get_phase4_ws_manager()
        manager2 = get_phase4_ws_manager()

        assert manager1 is manager2
        assert manager1.name == "phase4_metrics"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
