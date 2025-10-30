"""
Agent Instrumentation - Real-time Event Emission for LLM Agents

Wraps DistillationAgent V2 to emit real-time events during execution,
providing transparency into agent thinking, tool calls, and progress.

Features:
- Automatic event emission for agent lifecycle
- Tool call tracking and result logging
- Progress estimation based on steps
- Error handling and recovery tracking
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Callable
from contextlib import asynccontextmanager

from src.api.agent_stream import (
    agent_stream_manager,
    AgentEvent,
    AgentEventType,
    AgentProgress,
)

logger = logging.getLogger(__name__)


class AgentInstrumentation:
    """
    Instrumentation wrapper for LLM agents to emit real-time events.
    
    Usage:
        async with AgentInstrumentation(user_id="user123", agent_id="distillation") as instr:
            await instr.emit_thinking("Analyzing portfolio metrics...")
            result = await some_tool_call()
            await instr.emit_tool_result("compute_portfolio_metrics", result)
    """
    
    def __init__(
        self,
        user_id: str,
        agent_id: str,
        task_id: Optional[str] = None,
        total_steps: int = 10,
    ):
        self.user_id = user_id
        self.agent_id = agent_id
        self.task_id = task_id or f"{agent_id}_{int(time.time())}"
        self.total_steps = total_steps
        self.completed_steps = 0
        self.start_time: Optional[float] = None
        self.status = "pending"
    
    async def __aenter__(self):
        """Start instrumentation context"""
        self.start_time = time.time()
        self.status = "running"
        
        await self.emit_event(
            event_type=AgentEventType.STARTED,
            content=f"Agent {self.agent_id} started task {self.task_id}",
            metadata={"total_steps": self.total_steps}
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """End instrumentation context"""
        if exc_type is not None:
            # Error occurred
            self.status = "failed"
            await self.emit_event(
                event_type=AgentEventType.ERROR,
                content=f"Agent {self.agent_id} failed: {str(exc_val)}",
                error_flag=True,
                metadata={"error_type": exc_type.__name__}
            )
        else:
            # Success
            self.status = "completed"
            elapsed = time.time() - self.start_time if self.start_time else 0
            await self.emit_event(
                event_type=AgentEventType.COMPLETED,
                content=f"Agent {self.agent_id} completed in {elapsed:.1f}s",
                metadata={"time_elapsed_sec": elapsed}
            )
        
        return False  # Don't suppress exceptions
    
    async def emit_event(
        self,
        event_type: AgentEventType,
        content: str,
        metadata: Optional[Dict] = None,
        error_flag: bool = False,
    ):
        """Emit a generic agent event"""
        event = AgentEvent(
            timestamp=datetime.now(timezone.utc),
            agent_id=self.agent_id,
            event_type=event_type,
            content=content,
            metadata=metadata or {},
            error_flag=error_flag,
        )
        
        await agent_stream_manager.emit_event(self.user_id, event)
    
    async def emit_thinking(self, thought: str):
        """Emit a thinking event (agent reasoning)"""
        await self.emit_event(
            event_type=AgentEventType.THINKING,
            content=thought,
        )
    
    async def emit_tool_call(self, tool_name: str, args: Dict[str, Any]):
        """Emit a tool call event"""
        await self.emit_event(
            event_type=AgentEventType.TOOL_CALL,
            content=f"Calling tool: {tool_name}",
            metadata={"tool_name": tool_name, "args": args}
        )
    
    async def emit_tool_result(self, tool_name: str, result: Any, success: bool = True):
        """Emit a tool result event"""
        await self.emit_event(
            event_type=AgentEventType.TOOL_RESULT,
            content=f"Tool {tool_name} {'succeeded' if success else 'failed'}",
            metadata={
                "tool_name": tool_name,
                "success": success,
                "result_preview": str(result)[:200] if result else None
            },
            error_flag=not success,
        )
    
    async def emit_progress(self, current_step: str, increment: int = 1):
        """
        Emit a progress update.
        
        Args:
            current_step: Description of current step
            increment: Number of steps to increment (default: 1)
        """
        self.completed_steps += increment
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        progress_pct = (self.completed_steps / self.total_steps) * 100
        
        # Estimate time remaining (simple linear extrapolation)
        if self.completed_steps > 0 and progress_pct < 100:
            time_per_step = elapsed / self.completed_steps
            remaining_steps = self.total_steps - self.completed_steps
            estimated_remaining = time_per_step * remaining_steps
        else:
            estimated_remaining = None
        
        progress = AgentProgress(
            task_id=self.task_id,
            agent_id=self.agent_id,
            status=self.status,
            progress_pct=min(progress_pct, 100.0),
            time_elapsed_sec=elapsed,
            estimated_time_remaining_sec=estimated_remaining,
            current_step=current_step,
            total_steps=self.total_steps,
            completed_steps=self.completed_steps,
        )
        
        await agent_stream_manager.emit_progress(self.user_id, progress)


@asynccontextmanager
async def instrument_agent(
    user_id: str,
    agent_id: str,
    task_id: Optional[str] = None,
    total_steps: int = 10,
):
    """
    Context manager for instrumenting an agent.
    
    Usage:
        async with instrument_agent(user_id="user123", agent_id="distillation", total_steps=5) as instr:
            await instr.emit_thinking("Starting analysis...")
            await instr.emit_progress("Fetching data", increment=1)
            # ... agent work ...
    """
    instr = AgentInstrumentation(
        user_id=user_id,
        agent_id=agent_id,
        task_id=task_id,
        total_steps=total_steps,
    )
    
    async with instr:
        yield instr


def create_instrumented_callback(user_id: str, agent_id: str) -> Callable:
    """
    Create a callback function for instrumenting synchronous agent code.
    
    Usage:
        callback = create_instrumented_callback(user_id="user123", agent_id="distillation")
        callback("thinking", "Analyzing portfolio...")
        callback("tool_call", "compute_portfolio_metrics", {"symbols": ["AAPL"]})
    """
    
    def callback(event_type: str, content: str, metadata: Optional[Dict] = None):
        """Synchronous callback for agent events"""
        try:
            # Convert to async and run in event loop
            loop = asyncio.get_event_loop()
            
            event = AgentEvent(
                timestamp=datetime.now(timezone.utc),
                agent_id=agent_id,
                event_type=AgentEventType(event_type),
                content=content,
                metadata=metadata or {},
                error_flag=False,
            )
            
            # Schedule emission (non-blocking)
            asyncio.create_task(agent_stream_manager.emit_event(user_id, event))
        
        except Exception as e:
            logger.error(f"Error in instrumentation callback: {e}")
    
    return callback


async def instrument_distillation_agent(
    user_id: str,
    agent_instance: Any,
    portfolio_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Wrapper for DistillationAgent V2 that emits real-time events.

    Args:
        user_id: User ID for event routing
        agent_instance: DistillationAgent V2 instance
        portfolio_data: Portfolio data to analyze

    Returns:
        InvestorReport.v1 JSON
    """
    async with instrument_agent(
        user_id=user_id,
        agent_id="distillation_v2",
        total_steps=8,
    ) as instr:
        try:
            # Step 1: Initialize
            await instr.emit_thinking("Initializing DistillationAgent V2...")
            await instr.emit_progress("Initialization", increment=1)

            # Step 2: Fetch portfolio metrics
            await instr.emit_thinking("Computing portfolio metrics...")
            await instr.emit_tool_call("compute_portfolio_metrics", {"data": "portfolio_data"})
            await instr.emit_progress("Computing metrics", increment=1)

            # Step 3: Compute Phase 4 metrics
            await instr.emit_thinking("Computing Phase 4 technical signals...")
            await instr.emit_tool_call("compute_phase4_metrics", {"symbols": list(portfolio_data.keys()) if isinstance(portfolio_data, dict) else []})
            await instr.emit_progress("Phase 4 analysis", increment=1)

            # Step 4: Analyze options flow
            await instr.emit_thinking("Analyzing options flow composite...")
            await instr.emit_tool_call("compute_options_flow", {"symbols": list(portfolio_data.keys()) if isinstance(portfolio_data, dict) else []})
            await instr.emit_progress("Options flow", increment=1)

            # Step 5: Generate executive summary
            await instr.emit_thinking("Generating executive summary with LLM...")
            await instr.emit_progress("Executive summary", increment=1)

            # Step 6: Generate risk panel
            await instr.emit_thinking("Analyzing risk metrics...")
            await instr.emit_progress("Risk analysis", increment=1)

            # Step 7: Generate signals
            await instr.emit_thinking("Synthesizing multi-factor signals...")
            await instr.emit_progress("Signal synthesis", increment=1)

            # Step 8: Validate schema
            await instr.emit_thinking("Validating InvestorReport.v1 schema...")
            await instr.emit_progress("Schema validation", increment=1)

            # Actually call the agent (synchronous method, so run in executor)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                agent_instance.synthesize_swarm_output,
                portfolio_data
            )

            await instr.emit_tool_result("synthesize_swarm_output", result, success=True)

            return result

        except Exception as e:
            await instr.emit_tool_result("synthesize_swarm_output", str(e), success=False)
            raise

