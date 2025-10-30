"""
Agent Stream WebSocket - Real-time LLM Agent Transparency

Streams agent thoughts, tool calls, and progress updates to the frontend
for institutional-grade transparency during long-running analysis.

Features:
- Real-time agent event streaming via WebSocket
- Progress tracking (time elapsed, percentage complete)
- Agent conversation display (thinking, tool calls, results, errors)
- Bloomberg Terminal-style UX with clear status indicators
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AgentEventType(str, Enum):
    """Types of agent events that can be streamed"""
    STARTED = "started"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    PROGRESS = "progress"
    ERROR = "error"
    COMPLETED = "completed"
    HEARTBEAT = "heartbeat"


class AgentEvent(BaseModel):
    """Agent event data structure"""
    timestamp: datetime
    agent_id: str
    event_type: AgentEventType
    content: str
    metadata: Optional[Dict] = None
    error_flag: bool = False
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentProgress(BaseModel):
    """Agent progress tracking"""
    task_id: str
    agent_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress_pct: float  # 0.0 to 100.0
    time_elapsed_sec: float
    estimated_time_remaining_sec: Optional[float] = None
    current_step: str
    total_steps: int
    completed_steps: int


class AgentStreamManager:
    """
    Manages WebSocket connections for agent event streaming.
    
    Handles:
    - Connection lifecycle (connect, disconnect, broadcast)
    - Event queuing and delivery
    - Heartbeat to keep connections alive
    - Error handling and reconnection
    """
    
    def __init__(self):
        # Active WebSocket connections by user_id
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        
        # Event queues by user_id (for buffering when no connection)
        self.event_queues: Dict[str, List[AgentEvent]] = {}
        
        # Max events to buffer per user
        self.max_queue_size = 100
        
        # Heartbeat interval (seconds)
        self.heartbeat_interval = 30
        
        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept WebSocket connection and add to active connections"""
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        
        self.active_connections[user_id].add(websocket)
        
        logger.info(f"Agent stream connected for user {user_id}. "
                   f"Total connections: {len(self.active_connections[user_id])}")
        
        # Send any buffered events
        if user_id in self.event_queues and self.event_queues[user_id]:
            for event in self.event_queues[user_id]:
                await self._send_event(websocket, event)
            self.event_queues[user_id].clear()
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        """Remove WebSocket connection"""
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
            
            logger.info(f"Agent stream disconnected for user {user_id}. "
                       f"Remaining connections: {len(self.active_connections.get(user_id, []))}")
    
    async def emit_event(self, user_id: str, event: AgentEvent):
        """
        Emit an agent event to all connected clients for a user.
        If no connections, buffer the event.
        """
        if user_id in self.active_connections and self.active_connections[user_id]:
            # Send to all active connections
            disconnected = set()
            for websocket in self.active_connections[user_id]:
                try:
                    await self._send_event(websocket, event)
                except Exception as e:
                    logger.error(f"Error sending event to websocket: {e}")
                    disconnected.add(websocket)
            
            # Clean up disconnected websockets
            for ws in disconnected:
                self.disconnect(ws, user_id)
        else:
            # Buffer event for later delivery
            if user_id not in self.event_queues:
                self.event_queues[user_id] = []
            
            self.event_queues[user_id].append(event)
            
            # Trim queue if too large (keep most recent events)
            if len(self.event_queues[user_id]) > self.max_queue_size:
                self.event_queues[user_id] = self.event_queues[user_id][-self.max_queue_size:]
    
    async def emit_progress(self, user_id: str, progress: AgentProgress):
        """Emit a progress update"""
        event = AgentEvent(
            timestamp=datetime.now(timezone.utc),
            agent_id=progress.agent_id,
            event_type=AgentEventType.PROGRESS,
            content=f"Step {progress.completed_steps}/{progress.total_steps}: {progress.current_step}",
            metadata={
                "task_id": progress.task_id,
                "status": progress.status,
                "progress_pct": progress.progress_pct,
                "time_elapsed_sec": progress.time_elapsed_sec,
                "estimated_time_remaining_sec": progress.estimated_time_remaining_sec,
                "current_step": progress.current_step,
                "total_steps": progress.total_steps,
                "completed_steps": progress.completed_steps,
            }
        )
        await self.emit_event(user_id, event)
    
    async def _send_event(self, websocket: WebSocket, event: AgentEvent):
        """Send a single event to a websocket"""
        await websocket.send_json({
            "type": "agent_event",
            "data": event.dict()
        })
    
    async def _send_heartbeat(self, websocket: WebSocket):
        """Send heartbeat to keep connection alive"""
        await websocket.send_json({
            "type": "heartbeat",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def start_heartbeat(self):
        """Start background heartbeat task"""
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
    
    async def stop_heartbeat(self):
        """Stop background heartbeat task"""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
    
    async def _heartbeat_loop(self):
        """Background task to send heartbeats to all connections"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                for user_id, connections in list(self.active_connections.items()):
                    disconnected = set()
                    for websocket in connections:
                        try:
                            await self._send_heartbeat(websocket)
                        except Exception as e:
                            logger.error(f"Heartbeat failed for user {user_id}: {e}")
                            disconnected.add(websocket)
                    
                    # Clean up disconnected websockets
                    for ws in disconnected:
                        self.disconnect(ws, user_id)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")


# Global manager instance
agent_stream_manager = AgentStreamManager()


async def agent_stream_websocket(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for agent event streaming.
    
    Usage:
        ws://localhost:8000/ws/agent-stream/{user_id}
    
    Events sent to client:
        - agent_event: Real-time agent thoughts, tool calls, results
        - heartbeat: Keep-alive ping every 30s
    """
    await agent_stream_manager.connect(websocket, user_id)
    
    try:
        while True:
            # Wait for client messages (mostly for keep-alive)
            data = await websocket.receive_text()
            
            # Echo pong for ping messages
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        agent_stream_manager.disconnect(websocket, user_id)
    except Exception as e:
        logger.error(f"Error in agent stream websocket: {e}")
        agent_stream_manager.disconnect(websocket, user_id)

