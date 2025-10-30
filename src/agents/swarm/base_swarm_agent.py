"""
Base Swarm Agent

Abstract base class for all agents in the swarm system.
Provides common functionality for communication, decision-making, and coordination.
"""

import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np

from .shared_context import SharedContext, Message
from .consensus_engine import ConsensusEngine, Decision

logger = logging.getLogger(__name__)


# Tier-specific temperature profiles for agent diversity
TIER_TEMPERATURES = {
    1: 0.3,  # Oversight & Coordination (focused, deterministic)
    2: 0.5,  # Market Intelligence (balanced)
    3: 0.4,  # Fundamental & Macro (analytical)
    4: 0.6,  # Risk & Sentiment (exploratory)
    5: 0.5,  # Options & Volatility (balanced)
    6: 0.2,  # Execution & Compliance (highly deterministic)
    7: 0.7,  # Recommendation Engine (creative synthesis)
    8: 0.7   # Distillation (creative synthesis) - NEW TIER
}


class BaseSwarmAgent(ABC):
    """
    Abstract base class for swarm agents.
    
    All specialized agents (Market Analyst, Risk Manager, etc.) inherit from this class.
    Provides common functionality for:
    - Communication through shared context
    - Voting in consensus decisions
    - Calculating confidence levels
    - Monitoring and metrics
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        shared_context: SharedContext,
        consensus_engine: ConsensusEngine,
        priority: int = 5,
        confidence_threshold: float = 0.6,
        tier: int = 1,
        temperature: Optional[float] = None
    ):
        """
        Initialize base swarm agent.

        Args:
            agent_id: Unique identifier for this agent
            agent_type: Type of agent (e.g., "MarketAnalyst", "RiskManager")
            shared_context: Shared context for communication
            consensus_engine: Consensus engine for voting
            priority: Default message priority (1-10, higher = more important)
            confidence_threshold: Minimum confidence to act on decisions
            tier: Agent tier (1-8) for temperature configuration
            temperature: Override temperature (uses tier default if None)
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.shared_context = shared_context
        self.consensus_engine = consensus_engine
        self.priority = priority
        self.confidence_threshold = confidence_threshold
        self.tier = tier
        self.temperature = temperature or TIER_TEMPERATURES.get(tier, 0.5)

        # Agent state
        self.is_active = True
        self.last_action_time: Optional[datetime] = None
        self.action_count = 0

        # Metrics
        self._metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'decisions_participated': 0,
            'actions_taken': 0,
            'errors': 0
        }

        logger.info(f"{self.agent_type} agent initialized: {self.agent_id} (tier={self.tier}, temperature={self.temperature})")
    
    @abstractmethod
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform agent-specific analysis.
        
        Args:
            context: Current context/state to analyze
        
        Returns:
            Analysis results
        """
        pass
    
    @abstractmethod
    def make_recommendation(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make recommendations based on analysis.

        Args:
            analysis: Analysis results

        Returns:
            Recommendations with confidence levels
        """
        pass

    def get_context_summary(self, min_priority: int = 6, max_age_seconds: int = 3600) -> Dict[str, Any]:
        """
        Get summary of what other agents have already analyzed.

        Args:
            min_priority: Minimum priority for insights
            max_age_seconds: Maximum age of messages

        Returns:
            Dictionary with topics_covered, key_insights, uncovered_topics
        """
        return self.shared_context.get_context_summary(
            agent_id=self.agent_id,
            min_priority=min_priority,
            max_age_seconds=max_age_seconds
        )

    def send_message(
        self,
        content: Dict[str, Any],
        priority: Optional[int] = None,
        confidence: Optional[float] = None,
        ttl_seconds: int = 3600
    ) -> None:
        """
        Send a message to the swarm.
        
        Args:
            content: Message content
            priority: Message priority (uses default if None)
            confidence: Confidence level (calculated if None)
            ttl_seconds: Time-to-live in seconds
        """
        if priority is None:
            priority = self.priority
        
        if confidence is None:
            confidence = self._calculate_confidence(content)
        
        message = Message(
            source=self.agent_id,
            content=content,
            priority=priority,
            confidence=confidence,
            ttl_seconds=ttl_seconds
        )
        
        self.shared_context.send_message(message)
        self._metrics['messages_sent'] += 1
        
        logger.debug(f"{self.agent_id} sent message with priority {priority}")
    
    def get_messages(
        self,
        min_priority: int = 0,
        min_confidence: float = 0.0,
        max_age_seconds: Optional[int] = None
    ) -> List[Message]:
        """
        Retrieve messages from the swarm.
        
        Args:
            min_priority: Minimum priority level
            min_confidence: Minimum confidence level
            max_age_seconds: Maximum age in seconds
        
        Returns:
            List of messages
        """
        messages = self.shared_context.get_messages(
            source=None,  # Get from all sources
            min_priority=min_priority,
            min_confidence=min_confidence,
            max_age_seconds=max_age_seconds
        )
        
        self._metrics['messages_received'] += len(messages)
        return messages
    
    def update_state(self, key: str, value: Any) -> None:
        """Update shared state"""
        self.shared_context.update_state(key, value, source=self.agent_id)
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get value from shared state"""
        return self.shared_context.get_state(key, default)
    
    def vote(
        self,
        decision_id: str,
        choice: str,
        confidence: Optional[float] = None,
        reasoning: str = ""
    ) -> None:
        """
        Vote on a decision.
        
        Args:
            decision_id: ID of the decision
            choice: The agent's choice
            confidence: Confidence in the choice (calculated if None)
            reasoning: Reasoning for the choice
        """
        if confidence is None:
            confidence = self._calculate_confidence({'choice': choice})
        
        self.consensus_engine.add_vote(
            decision_id=decision_id,
            agent_id=self.agent_id,
            choice=choice,
            confidence=confidence,
            reasoning=reasoning
        )
        
        self._metrics['decisions_participated'] += 1
        logger.debug(f"{self.agent_id} voted: {choice} (confidence: {confidence:.2f})")
    
    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """
        Calculate confidence level using entropy-based approach.
        
        Lower entropy in the data = higher confidence
        
        Args:
            data: Data to calculate confidence from
        
        Returns:
            Confidence level (0-1)
        """
        # This is a simplified entropy calculation
        # Subclasses can override for more sophisticated approaches
        
        # Count unique values
        values = []
        for v in data.values():
            if isinstance(v, (list, tuple)):
                values.extend(v)
            else:
                values.append(v)
        
        if not values:
            return 0.5  # Neutral confidence
        
        # Calculate value distribution
        unique_values = len(set(str(v) for v in values))
        total_values = len(values)
        
        # Entropy-based confidence
        # More unique values = higher entropy = lower confidence
        if unique_values == 1:
            return 1.0  # Perfect confidence
        
        # Normalized entropy
        max_entropy = np.log2(total_values) if total_values > 1 else 1.0
        actual_entropy = np.log2(unique_values)
        normalized_entropy = actual_entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Confidence is inverse of entropy
        confidence = 1.0 - normalized_entropy
        
        # Clamp to reasonable range
        return max(0.1, min(0.9, confidence))
    
    def record_action(self, action_type: str, details: Dict[str, Any]) -> None:
        """
        Record an action taken by the agent.
        
        Args:
            action_type: Type of action
            details: Action details
        """
        self.last_action_time = datetime.utcnow()
        self.action_count += 1
        self._metrics['actions_taken'] += 1
        
        # Store in shared context for audit trail
        self.update_state(
            f"action_{self.agent_id}_{self.action_count}",
            {
                'agent_id': self.agent_id,
                'agent_type': self.agent_type,
                'action_type': action_type,
                'details': details,
                'timestamp': self.last_action_time.isoformat()
            }
        )
        
        logger.info(f"{self.agent_id} performed action: {action_type}")
    
    def record_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """
        Record an error.
        
        Args:
            error: The exception
            context: Error context
        """
        self._metrics['errors'] += 1
        
        error_data = {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Send high-priority error message
        self.send_message(
            content={'error': error_data},
            priority=10,
            confidence=1.0
        )
        
        logger.error(f"{self.agent_id} error: {error}", exc_info=True)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'is_active': self.is_active,
            'last_action_time': self.last_action_time.isoformat() if self.last_action_time else None,
            'action_count': self.action_count,
            **self._metrics
        }
    
    def shutdown(self) -> None:
        """Shutdown the agent"""
        self.is_active = False
        logger.info(f"{self.agent_id} shutting down")

