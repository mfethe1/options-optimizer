"""
Swarm Overseer Agent

Monitors agent activity, ensures all agents are contributing, identifies gaps
in analysis, and coordinates multi-round discussions between agents.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict

from ..base_swarm_agent import BaseSwarmAgent
from ..llm_agent_base import LLMAgentBase
from ..shared_context import SharedContext
from ..consensus_engine import ConsensusEngine

logger = logging.getLogger(__name__)


class SwarmOverseerAgent(BaseSwarmAgent, LLMAgentBase):
    """
    Swarm Overseer - Ensures optimal swarm performance.
    
    Responsibilities:
    - Monitor agent activity and contributions
    - Identify analysis gaps and missing perspectives
    - Coordinate multi-round agent discussions
    - Ensure diverse viewpoints are considered
    - Flag low-confidence or conflicting analyses
    """
    
    def __init__(
        self,
        agent_id: str,
        shared_context: SharedContext,
        consensus_engine: ConsensusEngine,
        preferred_model: str = "anthropic"
    ):
        BaseSwarmAgent.__init__(
            self,
            agent_id=agent_id,
            agent_type="SwarmOverseer",
            shared_context=shared_context,
            consensus_engine=consensus_engine,
            priority=10,  # Highest priority
            confidence_threshold=0.8
        )
        
        LLMAgentBase.__init__(
            self,
            agent_id=agent_id,
            agent_type="SwarmOverseer",
            preferred_model=preferred_model
        )
        
        self.agent_activity = defaultdict(int)
        self.analysis_gaps = []
        self.discussion_rounds = 0
        
        logger.info(f"{agent_id}: Swarm Overseer initialized with {preferred_model}")
    
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor swarm activity and identify gaps"""
        try:
            logger.info(f"{self.agent_id}: Monitoring swarm activity")

            # Get all messages from shared context (no limit parameter)
            all_messages = self.get_messages(min_priority=0)
            
            # Analyze agent contributions
            agent_contributions = self._analyze_contributions(all_messages)
            
            # Identify analysis gaps
            gaps = self._identify_gaps(all_messages, context)
            
            # Check for conflicts
            conflicts = self._detect_conflicts(all_messages)
            
            # Determine if additional discussion rounds needed
            needs_discussion = len(gaps) > 0 or len(conflicts) > 2
            
            analysis = {
                'agent_contributions': agent_contributions,
                'analysis_gaps': gaps,
                'conflicts_detected': conflicts,
                'needs_additional_discussion': needs_discussion,
                'swarm_health_score': self._calculate_health_score(agent_contributions, gaps),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Share oversight insights
            self.send_message({
                'type': 'oversight_report',
                'gaps': gaps,
                'conflicts': len(conflicts),
                'health_score': analysis['swarm_health_score'],
                'timestamp': datetime.utcnow().isoformat()
            }, priority=10)
            
            logger.info(f"{self.agent_id}: Oversight analysis complete - Health: {analysis['swarm_health_score']:.2f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"{self.agent_id}: Error in oversight: {e}")
            self.record_error(e, context)
            return {'error': str(e)}
    
    def make_recommendation(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make oversight recommendations"""
        try:
            gaps = analysis.get('analysis_gaps', [])
            health_score = analysis.get('swarm_health_score', 0.7)
            needs_discussion = analysis.get('needs_additional_discussion', False)
            
            recommendation = {
                'swarm_status': {
                    'choice': 'healthy' if health_score > 0.7 else 'needs_attention',
                    'confidence': health_score,
                    'reasoning': f"Swarm health: {health_score:.2f}, Gaps: {len(gaps)}, Discussion needed: {needs_discussion}"
                },
                'action_items': self._generate_action_items(gaps, needs_discussion),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return recommendation
            
        except Exception as e:
            logger.error(f"{self.agent_id}: Error making recommendation: {e}")
            return {'error': str(e)}
    
    def coordinate_discussion_round(self, topic: str, participants: List[str]) -> Dict[str, Any]:
        """Coordinate a discussion round between specific agents"""
        try:
            logger.info(f"{self.agent_id}: Coordinating discussion on '{topic}' with {len(participants)} agents")
            
            self.discussion_rounds += 1
            
            # Send discussion prompt to participants
            self.send_message({
                'type': 'discussion_prompt',
                'topic': topic,
                'participants': participants,
                'round': self.discussion_rounds,
                'timestamp': datetime.utcnow().isoformat()
            }, priority=9)
            
            return {
                'discussion_initiated': True,
                'topic': topic,
                'participants': participants,
                'round': self.discussion_rounds
            }
            
        except Exception as e:
            logger.error(f"{self.agent_id}: Error coordinating discussion: {e}")
            return {'error': str(e)}
    
    def _analyze_contributions(self, messages: List[Dict]) -> Dict[str, int]:
        """Analyze agent contribution levels"""
        contributions = defaultdict(int)
        
        for msg in messages:
            source = msg.get('source', 'unknown')
            contributions[source] += 1
        
        return dict(contributions)
    
    def _identify_gaps(self, messages: List[Dict], context: Dict) -> List[str]:
        """Identify missing analysis perspectives"""
        gaps = []
        
        # Check for key analysis types
        message_types = set([msg.get('content', {}).get('type', '') for msg in messages])
        
        required_analyses = [
            'fundamental_analysis',
            'technical_analysis',
            'sentiment_analysis',
            'risk_analysis',
            'macro_analysis',
            'options_strategy'
        ]
        
        for analysis_type in required_analyses:
            if analysis_type not in message_types:
                gaps.append(f"Missing {analysis_type}")
        
        return gaps
    
    def _detect_conflicts(self, messages: List[Dict]) -> List[Dict]:
        """Detect conflicting recommendations"""
        conflicts = []
        
        # Group messages by type
        by_type = defaultdict(list)
        for msg in messages:
            msg_type = msg.get('content', {}).get('type', '')
            if msg_type:
                by_type[msg_type].append(msg)
        
        # Check for conflicts within each type
        for msg_type, msgs in by_type.items():
            if len(msgs) > 1:
                # Check if recommendations conflict
                recommendations = [
                    msg.get('content', {}).get('recommendation', '')
                    for msg in msgs
                ]
                unique_recs = set([r for r in recommendations if r])
                
                if len(unique_recs) > 1:
                    conflicts.append({
                        'type': msg_type,
                        'conflicting_views': list(unique_recs),
                        'count': len(msgs)
                    })
        
        return conflicts
    
    def _calculate_health_score(self, contributions: Dict, gaps: List) -> float:
        """Calculate overall swarm health score"""
        
        # Base score
        score = 0.5
        
        # Bonus for active agents
        if len(contributions) >= 10:
            score += 0.2
        elif len(contributions) >= 5:
            score += 0.1
        
        # Penalty for gaps
        score -= (len(gaps) * 0.05)
        
        # Bonus for balanced contributions
        if contributions:
            contribution_values = list(contributions.values())
            avg_contrib = sum(contribution_values) / len(contribution_values)
            variance = sum((c - avg_contrib) ** 2 for c in contribution_values) / len(contribution_values)
            
            if variance < 2.0:  # Low variance = balanced
                score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _generate_action_items(self, gaps: List, needs_discussion: bool) -> List[str]:
        """Generate action items for swarm improvement"""
        actions = []
        
        if gaps:
            actions.append(f"Address {len(gaps)} analysis gaps: {', '.join(gaps[:3])}")
        
        if needs_discussion:
            actions.append("Initiate multi-round discussion to resolve conflicts")
        
        if not actions:
            actions.append("Continue monitoring - swarm operating optimally")
        
        return actions

