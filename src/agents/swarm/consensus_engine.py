"""
Consensus Engine for Multi-Agent Decision Making

Implements various consensus mechanisms for the swarm to reach agreement
on decisions, using entropy-based confidence levels and weighted voting.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import numpy as np
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)


class ConsensusMethod(Enum):
    """Available consensus methods"""
    MAJORITY = "majority"  # Simple majority vote
    WEIGHTED = "weighted"  # Weighted by confidence
    UNANIMOUS = "unanimous"  # All agents must agree
    QUORUM = "quorum"  # Minimum percentage must agree
    ENTROPY_BASED = "entropy_based"  # Based on information entropy


class Decision:
    """Represents a decision made by the swarm"""
    
    def __init__(
        self,
        decision_id: str,
        question: str,
        options: List[str],
        method: ConsensusMethod = ConsensusMethod.WEIGHTED
    ):
        self.decision_id = decision_id
        self.question = question
        self.options = options
        self.method = method
        self.votes: Dict[str, Dict[str, Any]] = {}  # agent_id -> vote_data
        self.result: Optional[str] = None
        self.confidence: float = 0.0
        self.timestamp = datetime.utcnow()
        self.metadata: Dict[str, Any] = {}
    
    def add_vote(
        self,
        agent_id: str,
        choice: str,
        confidence: float = 1.0,
        reasoning: str = ""
    ) -> None:
        """
        Add a vote from an agent.
        
        Args:
            agent_id: ID of the voting agent
            choice: The agent's choice
            confidence: Agent's confidence in their choice (0-1)
            reasoning: Optional reasoning for the choice
        """
        if choice not in self.options:
            raise ValueError(f"Invalid choice: {choice}. Must be one of {self.options}")
        
        if not 0 <= confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")
        
        self.votes[agent_id] = {
            'choice': choice,
            'confidence': confidence,
            'reasoning': reasoning,
            'timestamp': datetime.utcnow()
        }
        
        logger.debug(f"Vote recorded: {agent_id} -> {choice} (confidence: {confidence:.2f})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary"""
        return {
            'decision_id': self.decision_id,
            'question': self.question,
            'options': self.options,
            'method': self.method.value,
            'votes': self.votes,
            'result': self.result,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class ConsensusEngine:
    """
    Engine for reaching consensus among swarm agents.
    
    Supports multiple consensus mechanisms:
    - Majority voting
    - Weighted voting (by confidence)
    - Unanimous agreement
    - Quorum-based decisions
    - Entropy-based consensus
    """
    
    def __init__(self, quorum_threshold: float = 0.67):
        """
        Initialize consensus engine.
        
        Args:
            quorum_threshold: Minimum fraction of agents needed for quorum (0-1)
        """
        self.quorum_threshold = quorum_threshold
        self.decisions: Dict[str, Decision] = {}
        self._metrics = {
            'total_decisions': 0,
            'consensus_reached': 0,
            'consensus_failed': 0
        }
        
        logger.info(f"ConsensusEngine initialized with quorum threshold: {quorum_threshold}")
    
    def create_decision(
        self,
        decision_id: str,
        question: str,
        options: List[str],
        method: ConsensusMethod = ConsensusMethod.WEIGHTED
    ) -> Decision:
        """
        Create a new decision for the swarm to vote on.
        
        Args:
            decision_id: Unique identifier for the decision
            question: The question being decided
            options: List of possible choices
            method: Consensus method to use
        
        Returns:
            Decision object
        """
        decision = Decision(decision_id, question, options, method)
        self.decisions[decision_id] = decision
        self._metrics['total_decisions'] += 1
        
        logger.info(f"Decision created: {decision_id} - {question}")
        return decision
    
    def add_vote(
        self,
        decision_id: str,
        agent_id: str,
        choice: str,
        confidence: float = 1.0,
        reasoning: str = ""
    ) -> None:
        """Add a vote to a decision"""
        if decision_id not in self.decisions:
            raise ValueError(f"Decision not found: {decision_id}")
        
        self.decisions[decision_id].add_vote(agent_id, choice, confidence, reasoning)
    
    def reach_consensus(self, decision_id: str) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """
        Attempt to reach consensus on a decision.
        
        Args:
            decision_id: ID of the decision
        
        Returns:
            Tuple of (result, confidence, metadata)
            - result: The consensus choice (None if no consensus)
            - confidence: Overall confidence in the decision (0-1)
            - metadata: Additional information about the consensus process
        """
        if decision_id not in self.decisions:
            raise ValueError(f"Decision not found: {decision_id}")
        
        decision = self.decisions[decision_id]
        
        if not decision.votes:
            logger.warning(f"No votes for decision: {decision_id}")
            return None, 0.0, {'error': 'No votes'}
        
        # Route to appropriate consensus method
        if decision.method == ConsensusMethod.MAJORITY:
            result, confidence, metadata = self._majority_consensus(decision)
        elif decision.method == ConsensusMethod.WEIGHTED:
            result, confidence, metadata = self._weighted_consensus(decision)
        elif decision.method == ConsensusMethod.UNANIMOUS:
            result, confidence, metadata = self._unanimous_consensus(decision)
        elif decision.method == ConsensusMethod.QUORUM:
            result, confidence, metadata = self._quorum_consensus(decision)
        elif decision.method == ConsensusMethod.ENTROPY_BASED:
            result, confidence, metadata = self._entropy_consensus(decision)
        else:
            raise ValueError(f"Unknown consensus method: {decision.method}")
        
        # Update decision
        decision.result = result
        decision.confidence = confidence
        decision.metadata = metadata
        
        # Update metrics
        if result is not None:
            self._metrics['consensus_reached'] += 1
        else:
            self._metrics['consensus_failed'] += 1
        
        logger.info(f"Consensus reached for {decision_id}: {result} (confidence: {confidence:.2f})")
        return result, confidence, metadata
    
    def _majority_consensus(self, decision: Decision) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """Simple majority voting"""
        votes = [v['choice'] for v in decision.votes.values()]
        vote_counts = Counter(votes)
        
        if not vote_counts:
            return None, 0.0, {'error': 'No votes'}
        
        # Get most common choice
        result, count = vote_counts.most_common(1)[0]
        total_votes = len(votes)
        confidence = count / total_votes
        
        metadata = {
            'vote_counts': dict(vote_counts),
            'total_votes': total_votes,
            'winning_votes': count
        }
        
        return result, confidence, metadata
    
    def _weighted_consensus(self, decision: Decision) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """Weighted voting by agent confidence"""
        weighted_votes: Dict[str, float] = {}
        
        for vote_data in decision.votes.values():
            choice = vote_data['choice']
            confidence = vote_data['confidence']
            weighted_votes[choice] = weighted_votes.get(choice, 0.0) + confidence
        
        if not weighted_votes:
            return None, 0.0, {'error': 'No votes'}
        
        # Get choice with highest weighted score
        result = max(weighted_votes, key=weighted_votes.get)
        total_weight = sum(weighted_votes.values())
        confidence = weighted_votes[result] / total_weight if total_weight > 0 else 0.0
        
        metadata = {
            'weighted_votes': weighted_votes,
            'total_weight': total_weight,
            'winning_weight': weighted_votes[result]
        }
        
        return result, confidence, metadata
    
    def _unanimous_consensus(self, decision: Decision) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """Unanimous agreement required"""
        votes = [v['choice'] for v in decision.votes.values()]
        unique_votes = set(votes)
        
        if len(unique_votes) == 1:
            result = votes[0]
            # Average confidence of all agents
            confidences = [v['confidence'] for v in decision.votes.values()]
            confidence = np.mean(confidences)
            
            metadata = {
                'unanimous': True,
                'total_votes': len(votes),
                'avg_confidence': confidence
            }
            
            return result, confidence, metadata
        else:
            metadata = {
                'unanimous': False,
                'vote_distribution': dict(Counter(votes))
            }
            return None, 0.0, metadata
    
    def _quorum_consensus(self, decision: Decision) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """Quorum-based consensus"""
        votes = [v['choice'] for v in decision.votes.values()]
        vote_counts = Counter(votes)
        total_votes = len(votes)
        
        # Check if any option meets quorum
        for choice, count in vote_counts.items():
            if count / total_votes >= self.quorum_threshold:
                # Calculate confidence as weighted average
                choice_confidences = [
                    v['confidence'] for v in decision.votes.values()
                    if v['choice'] == choice
                ]
                confidence = np.mean(choice_confidences)
                
                metadata = {
                    'quorum_met': True,
                    'quorum_threshold': self.quorum_threshold,
                    'vote_fraction': count / total_votes,
                    'vote_counts': dict(vote_counts)
                }
                
                return choice, confidence, metadata
        
        # No quorum reached
        metadata = {
            'quorum_met': False,
            'quorum_threshold': self.quorum_threshold,
            'vote_counts': dict(vote_counts)
        }
        return None, 0.0, metadata
    
    def _entropy_consensus(self, decision: Decision) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """
        Entropy-based consensus using information theory.
        
        Lower entropy = higher certainty = better consensus
        """
        # Calculate weighted votes
        weighted_votes: Dict[str, float] = {}
        for vote_data in decision.votes.values():
            choice = vote_data['choice']
            confidence = vote_data['confidence']
            weighted_votes[choice] = weighted_votes.get(choice, 0.0) + confidence
        
        total_weight = sum(weighted_votes.values())
        if total_weight == 0:
            return None, 0.0, {'error': 'No votes'}
        
        # Calculate probabilities
        probabilities = {k: v / total_weight for k, v in weighted_votes.items()}
        
        # Calculate Shannon entropy
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities.values())
        max_entropy = np.log2(len(decision.options)) if len(decision.options) > 1 else 1.0
        
        # Normalize entropy to 0-1 (0 = perfect consensus, 1 = maximum uncertainty)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Confidence is inverse of normalized entropy
        confidence = 1.0 - normalized_entropy
        
        # Get choice with highest probability
        result = max(probabilities, key=probabilities.get)
        
        metadata = {
            'entropy': entropy,
            'max_entropy': max_entropy,
            'normalized_entropy': normalized_entropy,
            'probabilities': probabilities,
            'weighted_votes': weighted_votes
        }
        
        return result, confidence, metadata
    
    def get_decision(self, decision_id: str) -> Optional[Decision]:
        """Get a decision by ID"""
        return self.decisions.get(decision_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get consensus engine metrics"""
        return {
            **self._metrics,
            'active_decisions': len(self.decisions),
            'success_rate': (
                self._metrics['consensus_reached'] / self._metrics['total_decisions']
                if self._metrics['total_decisions'] > 0 else 0.0
            )
        }

