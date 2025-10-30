"""
Shared Context for Agent Swarm Communication

Implements stigmergic communication where agents leave traces in a shared
environment that other agents can read and respond to. This enables
decentralized coordination without direct agent-to-agent communication.
"""

import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import json
import hashlib

logger = logging.getLogger(__name__)


class Message:
    """Represents a message in the swarm communication system"""
    
    def __init__(
        self,
        source: str,
        content: Dict[str, Any],
        priority: int = 5,
        confidence: float = 0.5,
        ttl_seconds: int = 3600
    ):
        self.source = source
        self.content = content
        self.priority = priority  # 1-10, higher = more important
        self.confidence = confidence  # 0-1, entropy-based certainty
        self.timestamp = datetime.utcnow()
        self.ttl = timedelta(seconds=ttl_seconds)
        self.id = f"{source}_{self.timestamp.isoformat()}"
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        return datetime.utcnow() > (self.timestamp + self.ttl)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            'id': self.id,
            'source': self.source,
            'content': self.content,
            'priority': self.priority,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'ttl_seconds': self.ttl.total_seconds()
        }


class SharedContext:
    """
    Shared memory space for agent swarm communication.
    
    Implements stigmergic communication where agents deposit and read
    information from a shared environment. Supports:
    - Message passing with priorities
    - State sharing across agents
    - Event-driven notifications
    - Automatic cleanup of expired data
    """
    
    def __init__(self, max_messages: int = 1000):
        self.max_messages = max_messages
        self._lock = threading.RLock()

        # Message storage
        self._messages: List[Message] = []
        self._messages_by_source: Dict[str, List[Message]] = defaultdict(list)

        # Deduplication tracking
        self._message_hashes: Set[str] = set()
        self._topic_coverage: Dict[str, List[str]] = defaultdict(list)  # topic -> [agent_ids]

        # Shared state
        self._state: Dict[str, Any] = {}
        self._state_history: Dict[str, List[tuple]] = defaultdict(list)

        # Event subscribers
        self._subscribers: Dict[str, List[callable]] = defaultdict(list)

        # Metrics
        self._metrics = {
            'total_messages': 0,
            'total_state_updates': 0,
            'total_events': 0,
            'duplicate_messages': 0
        }

        logger.info("SharedContext initialized")
    
    def _hash_content(self, content: Dict[str, Any]) -> str:
        """
        Generate MD5 hash of message content for deduplication.

        Args:
            content: Message content dictionary

        Returns:
            MD5 hash string
        """
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()

    def send_message(self, message: Message) -> bool:
        """
        Send a message to the swarm with deduplication check.

        Args:
            message: Message to send

        Returns:
            True if message was added, False if duplicate was rejected
        """
        with self._lock:
            # Check for duplicate content
            content_hash = self._hash_content(message.content)
            if content_hash in self._message_hashes:
                logger.info(f"🔄 Duplicate message from {message.source} - skipping")
                self._metrics['duplicate_messages'] += 1
                return False

            # Add message
            self._message_hashes.add(content_hash)
            self._messages.append(message)
            self._messages_by_source[message.source].append(message)

            # Track topic coverage
            topic = message.content.get('topic', 'general')
            if message.source not in self._topic_coverage[topic]:
                self._topic_coverage[topic].append(message.source)

            # Cleanup if needed
            if len(self._messages) > self.max_messages:
                self._cleanup_messages()

            # Update metrics
            self._metrics['total_messages'] += 1

            # Trigger event
            self._trigger_event('message_received', message.to_dict())

            logger.debug(f"✅ Message sent from {message.source} with priority {message.priority}")
            return True
    
    def get_messages(
        self,
        source: Optional[str] = None,
        min_priority: int = 0,
        min_confidence: float = 0.0,
        max_age_seconds: Optional[int] = None
    ) -> List[Message]:
        """
        Retrieve messages from the swarm.
        
        Args:
            source: Filter by source agent (None = all sources)
            min_priority: Minimum priority level
            min_confidence: Minimum confidence level
            max_age_seconds: Maximum age in seconds (None = no limit)
        
        Returns:
            List of matching messages
        """
        with self._lock:
            # Start with all messages or source-specific
            if source:
                messages = self._messages_by_source.get(source, [])
            else:
                messages = self._messages
            
            # Apply filters
            filtered = []
            cutoff_time = datetime.utcnow() - timedelta(seconds=max_age_seconds) if max_age_seconds else None
            
            for msg in messages:
                # Skip expired
                if msg.is_expired():
                    continue
                
                # Check priority
                if msg.priority < min_priority:
                    continue
                
                # Check confidence
                if msg.confidence < min_confidence:
                    continue
                
                # Check age
                if cutoff_time and msg.timestamp < cutoff_time:
                    continue
                
                filtered.append(msg)
            
            # Sort by priority (descending) then timestamp (descending)
            filtered.sort(key=lambda m: (-m.priority, -m.timestamp.timestamp()))
            
            return filtered
    
    def update_state(self, key: str, value: Any, source: str = "system") -> None:
        """
        Update shared state.
        
        Args:
            key: State key
            value: State value
            source: Source of update
        """
        with self._lock:
            old_value = self._state.get(key)
            self._state[key] = value
            
            # Record history
            self._state_history[key].append((
                datetime.utcnow(),
                source,
                old_value,
                value
            ))
            
            # Limit history
            if len(self._state_history[key]) > 100:
                self._state_history[key] = self._state_history[key][-100:]
            
            # Update metrics
            self._metrics['total_state_updates'] += 1
            
            # Trigger event
            self._trigger_event('state_updated', {
                'key': key,
                'value': value,
                'source': source,
                'old_value': old_value
            })
            
            logger.debug(f"State updated: {key} = {value} (source: {source})")
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get value from shared state"""
        with self._lock:
            return self._state.get(key, default)
    
    def get_all_state(self) -> Dict[str, Any]:
        """Get all shared state"""
        with self._lock:
            return self._state.copy()
    
    def subscribe(self, event_type: str, callback: callable) -> None:
        """
        Subscribe to events.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event occurs
        """
        with self._lock:
            self._subscribers[event_type].append(callback)
            logger.debug(f"Subscribed to {event_type}")
    
    def _trigger_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Trigger an event to all subscribers"""
        self._metrics['total_events'] += 1
        
        for callback in self._subscribers.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
    
    def _cleanup_messages(self) -> None:
        """Remove expired and old messages"""
        # Remove expired messages
        self._messages = [m for m in self._messages if not m.is_expired()]
        
        # Rebuild source index
        self._messages_by_source.clear()
        for msg in self._messages:
            self._messages_by_source[msg.source].append(msg)
        
        # If still too many, remove oldest
        if len(self._messages) > self.max_messages:
            self._messages.sort(key=lambda m: m.timestamp)
            self._messages = self._messages[-self.max_messages:]
            
            # Rebuild source index again
            self._messages_by_source.clear()
            for msg in self._messages:
                self._messages_by_source[msg.source].append(msg)
    
    def get_uncovered_topics(self, agent_id: str) -> List[str]:
        """
        Get topics that this agent hasn't analyzed yet.

        Args:
            agent_id: Agent requesting uncovered topics

        Returns:
            List of topic names not yet covered by this agent
        """
        with self._lock:
            all_topics = set(self._topic_coverage.keys())
            agent_topics = {topic for topic, agents in self._topic_coverage.items() if agent_id in agents}
            uncovered = list(all_topics - agent_topics)
            return uncovered

    def get_context_summary(self, agent_id: str, min_priority: int = 6, max_age_seconds: int = 3600) -> Dict[str, Any]:
        """
        Get summary of what other agents have already analyzed.

        Args:
            agent_id: Agent requesting the summary
            min_priority: Minimum priority for insights
            max_age_seconds: Maximum age of messages to include

        Returns:
            Dictionary with topics_covered, key_insights, uncovered_topics
        """
        messages = self.get_messages(min_priority=min_priority, max_age_seconds=max_age_seconds)

        # Filter out messages from requesting agent
        other_messages = [m for m in messages if m.source != agent_id]

        return {
            'topics_covered': list(self._topic_coverage.keys()),
            'key_insights': [m.content for m in other_messages[:10]],  # Top 10 insights
            'uncovered_topics': self.get_uncovered_topics(agent_id),
            'total_insights': len(other_messages)
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get context metrics"""
        with self._lock:
            return {
                **self._metrics,
                'active_messages': len(self._messages),
                'state_keys': len(self._state),
                'subscribers': sum(len(subs) for subs in self._subscribers.values()),
                'unique_topics': len(self._topic_coverage),
                'deduplication_rate': self._metrics['duplicate_messages'] / max(self._metrics['total_messages'] + self._metrics['duplicate_messages'], 1)
            }
    
    def clear(self) -> None:
        """Clear all messages and state (for testing)"""
        with self._lock:
            self._messages.clear()
            self._messages_by_source.clear()
            self._state.clear()
            self._state_history.clear()
            logger.info("SharedContext cleared")

