"""
Base Agent class for all specialized agents in the system.
"""
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent system.
    
    Each agent has:
    - A specific role and responsibility
    - Access to tools/functions
    - Memory (short-term and long-term)
    - Ability to communicate with other agents
    """
    
    def __init__(self, name: str, role: str, tools: Optional[List] = None):
        """
        Initialize the base agent.
        
        Args:
            name: Agent's name
            role: Agent's role/responsibility
            tools: List of tools the agent can use
        """
        self.name = name
        self.role = role
        self.tools = tools or []
        self.short_term_memory: List[Dict[str, Any]] = []
        self.long_term_memory: Dict[str, Any] = {}
        
        logger.info(f"Initialized {self.name} with role: {self.role}")
    
    def add_to_memory(self, key: str, value: Any):
        """
        Add information to agent's long-term memory.

        Args:
            key: Memory key
            value: Value to store
        """
        self.long_term_memory[key] = value
        logger.debug(f"{self.name}: Added to memory - {key}")

    def get_from_memory(self, key: str, default: Any = None) -> Any:
        """
        Retrieve information from agent's long-term memory.

        Args:
            key: Memory key
            default: Default value if key not found

        Returns:
            Stored value or default
        """
        return self.long_term_memory.get(key, default)

    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state and return updated state.

        Args:
            state: Current system state

        Returns:
            Updated state after agent processing
        """
        pass
    
    def add_to_short_term_memory(self, item: Dict[str, Any]):
        """Add item to short-term memory."""
        self.short_term_memory.append(item)
        
        # Keep only last 100 items
        if len(self.short_term_memory) > 100:
            self.short_term_memory = self.short_term_memory[-100:]
    
    def update_long_term_memory(self, key: str, value: Any):
        """Update long-term memory."""
        self.long_term_memory[key] = value
    
    def get_context(self) -> Dict[str, Any]:
        """Get agent's current context."""
        return {
            'name': self.name,
            'role': self.role,
            'short_term_memory': self.short_term_memory[-10:],  # Last 10 items
            'long_term_memory': self.long_term_memory
        }

