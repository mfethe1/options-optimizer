"""
Swarm Coordinator

Orchestrates the multi-agent swarm system, managing agent lifecycle,
coordinating decisions, and ensuring optimal swarm performance.
"""

import logging
from typing import Dict, List, Any, Optional, Type
from datetime import datetime
import threading
import time
import pandas as pd
import numpy as np

from .shared_context import SharedContext
from .consensus_engine import ConsensusEngine, ConsensusMethod
from .base_swarm_agent import BaseSwarmAgent

# Import analytics modules
try:
    from src.analytics.portfolio_metrics import PortfolioAnalytics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logging.warning("Portfolio metrics module not available")

logger = logging.getLogger(__name__)


class SwarmCoordinator:
    """
    Coordinates the multi-agent swarm system.
    
    Responsibilities:
    - Manage agent lifecycle (creation, activation, shutdown)
    - Coordinate swarm-wide decisions
    - Monitor swarm health and performance
    - Facilitate agent communication
    - Optimize swarm composition
    """
    
    def __init__(
        self,
        name: str = "OptionsAnalysisSwarm",
        max_messages: int = 1000,
        quorum_threshold: float = 0.67
    ):
        """
        Initialize swarm coordinator.
        
        Args:
            name: Name of the swarm
            max_messages: Maximum messages in shared context
            quorum_threshold: Quorum threshold for consensus
        """
        self.name = name
        self.shared_context = SharedContext(max_messages=max_messages)
        self.consensus_engine = ConsensusEngine(quorum_threshold=quorum_threshold)

        # Agent management
        self.agents: Dict[str, BaseSwarmAgent] = {}
        self.agent_types: Dict[str, List[str]] = {}  # type -> [agent_ids]

        # Tier 8: Distillation Agent
        self.distillation_agent: Optional[DistillationAgent] = None
        self._initialize_distillation_agent()

        # Swarm state
        self.is_running = False
        self.start_time: Optional[datetime] = None

        # Metrics
        self._metrics = {
            'total_agents_created': 0,
            'total_decisions': 0,
            'total_recommendations': 0,
            'total_errors': 0
        }

        logger.info(f"SwarmCoordinator initialized: {name}")

    def _initialize_distillation_agent(self) -> None:
        """Initialize the Tier 8 Distillation Agent."""
        try:
            # Lazy import to avoid circular dependency
            from .agents.distillation_agent import DistillationAgent

            self.distillation_agent = DistillationAgent(
                shared_context=self.shared_context,
                consensus_engine=self.consensus_engine
            )
            logger.info("🎨 Distillation Agent initialized")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize Distillation Agent: {e}")
            self.distillation_agent = None

    def register_agent(self, agent: BaseSwarmAgent) -> None:
        """
        Register an agent with the swarm.
        
        Args:
            agent: Agent to register
        """
        if agent.agent_id in self.agents:
            raise ValueError(f"Agent already registered: {agent.agent_id}")
        
        self.agents[agent.agent_id] = agent
        
        # Track by type
        if agent.agent_type not in self.agent_types:
            self.agent_types[agent.agent_type] = []
        self.agent_types[agent.agent_type].append(agent.agent_id)
        
        self._metrics['total_agents_created'] += 1
        
        logger.info(f"Agent registered: {agent.agent_id} ({agent.agent_type})")
    
    def get_agent(self, agent_id: str) -> Optional[BaseSwarmAgent]:
        """Get an agent by ID"""
        return self.agents.get(agent_id)
    
    def get_agents_by_type(self, agent_type: str) -> List[BaseSwarmAgent]:
        """Get all agents of a specific type"""
        agent_ids = self.agent_types.get(agent_type, [])
        return [self.agents[aid] for aid in agent_ids if aid in self.agents]
    
    def start(self) -> None:
        """Start the swarm"""
        if self.is_running:
            logger.warning("Swarm already running")
            return
        
        self.is_running = True
        self.start_time = datetime.utcnow()
        
        logger.info(f"Swarm started: {self.name} with {len(self.agents)} agents")
    
    def stop(self) -> None:
        """Stop the swarm"""
        if not self.is_running:
            logger.warning("Swarm not running")
            return
        
        self.is_running = False
        
        # Shutdown all agents
        for agent in self.agents.values():
            agent.shutdown()
        
        logger.info(f"Swarm stopped: {self.name}")
    
    def analyze_portfolio(
        self,
        portfolio_data: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive portfolio analysis using all agents.
        
        Args:
            portfolio_data: Current portfolio positions and metrics
            market_data: Current market data
        
        Returns:
            Comprehensive analysis from all agents
        """
        if not self.is_running:
            raise RuntimeError("Swarm not running. Call start() first.")
        
        logger.info("Starting portfolio analysis")
        
        # Prepare context
        context = {
            'portfolio': portfolio_data,
            'market': market_data,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Update shared state
        self.shared_context.update_state('portfolio_data', portfolio_data, source='coordinator')
        self.shared_context.update_state('market_data', market_data, source='coordinator')
        
        # Collect analyses from all agents
        analyses = {}
        
        for agent_id, agent in self.agents.items():
            try:
                logger.debug(f"Running analysis: {agent_id}")
                analysis = agent.analyze(context)
                analyses[agent_id] = {
                    'agent_type': agent.agent_type,
                    'analysis': analysis,
                    'timestamp': datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.error(f"Error in agent {agent_id}: {e}")
                agent.record_error(e, context)
                self._metrics['total_errors'] += 1
        
        logger.info(f"Portfolio analysis complete: {len(analyses)} agents contributed")

        # Calculate institutional-grade portfolio metrics
        portfolio_metrics = None
        try:
            logger.info("📊 Calculating institutional-grade portfolio metrics...")
            portfolio_metrics = self._calculate_portfolio_metrics(portfolio_data, market_data)
            if portfolio_metrics:
                logger.info(f"✅ Metrics calculated: Sharpe={portfolio_metrics.get('sharpe_ratio', 0):.2f}, Omega={portfolio_metrics.get('omega_ratio', 0):.2f}")
        except Exception as e:
            logger.warning(f"⚠️ Could not calculate portfolio metrics: {e}")

        # Generate investor-friendly narrative via Distillation Agent
        investor_report = None
        if self.distillation_agent:
            try:
                logger.info("🎨 Synthesizing investor narrative...")
                investor_report = self.distillation_agent.synthesize_swarm_output({
                    'portfolio': portfolio_data,
                    'market': market_data,
                    'analyses': analyses,
                    'metrics': portfolio_metrics  # Pass metrics to distillation agent
                })
                logger.info("✅ Investor narrative generated")
            except Exception as e:
                logger.error(f"❌ Error generating investor narrative: {e}")

        result = {
            'swarm_name': self.name,
            'timestamp': datetime.utcnow().isoformat(),
            'context': context,
            'analyses': analyses,
            'agent_count': len(analyses)
        }

        # Add investor report if available
        if investor_report:
            result['investor_report'] = investor_report

        return result
    
    def make_recommendations(
        self,
        analysis_results: Dict[str, Any],
        consensus_method: ConsensusMethod = ConsensusMethod.WEIGHTED
    ) -> Dict[str, Any]:
        """
        Generate trading recommendations using swarm consensus.
        
        Args:
            analysis_results: Results from analyze_portfolio()
            consensus_method: Method for reaching consensus
        
        Returns:
            Swarm recommendations with confidence levels
        """
        logger.info("Generating swarm recommendations")
        
        # Collect recommendations from all agents
        recommendations = {}
        
        for agent_id, result in analysis_results.get('analyses', {}).items():
            agent = self.agents.get(agent_id)
            if not agent:
                continue
            
            try:
                logger.debug(f"Getting recommendations: {agent_id}")
                rec = agent.make_recommendation(result['analysis'])
                recommendations[agent_id] = {
                    'agent_type': agent.agent_type,
                    'recommendation': rec,
                    'timestamp': datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.error(f"Error getting recommendations from {agent_id}: {e}")
                agent.record_error(e, result)
                self._metrics['total_errors'] += 1
        
        # Create consensus decisions for key recommendations
        consensus_results = self._build_consensus(recommendations, consensus_method)
        
        self._metrics['total_recommendations'] += 1
        
        logger.info(f"Recommendations generated: {len(consensus_results)} consensus decisions")
        
        return {
            'swarm_name': self.name,
            'timestamp': datetime.utcnow().isoformat(),
            'individual_recommendations': recommendations,
            'consensus_recommendations': consensus_results,
            'consensus_method': consensus_method.value
        }
    
    def _build_consensus(
        self,
        recommendations: Dict[str, Any],
        method: ConsensusMethod
    ) -> Dict[str, Any]:
        """
        Build consensus from individual agent recommendations.
        
        Args:
            recommendations: Individual agent recommendations
            method: Consensus method
        
        Returns:
            Consensus results
        """
        consensus_results = {}
        
        # Extract common decision points
        decision_points = self._extract_decision_points(recommendations)
        
        for decision_id, decision_data in decision_points.items():
            # Create decision
            decision = self.consensus_engine.create_decision(
                decision_id=decision_id,
                question=decision_data['question'],
                options=decision_data['options'],
                method=method
            )
            
            # Collect votes from agents
            for agent_id, rec_data in recommendations.items():
                agent = self.agents.get(agent_id)
                if not agent:
                    continue
                
                # Extract agent's choice for this decision
                choice, confidence, reasoning = self._extract_agent_choice(
                    rec_data['recommendation'],
                    decision_id,
                    decision_data
                )
                
                if choice:
                    agent.vote(decision_id, choice, confidence, reasoning)
            
            # Reach consensus
            result, confidence, metadata = self.consensus_engine.reach_consensus(decision_id)
            
            consensus_results[decision_id] = {
                'question': decision_data['question'],
                'result': result,
                'confidence': confidence,
                'metadata': metadata,
                'method': method.value
            }
            
            self._metrics['total_decisions'] += 1
        
        return consensus_results
    
    def _extract_decision_points(self, recommendations: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Extract common decision points from recommendations.
        
        This is a simplified version - in production, this would use
        more sophisticated NLP/analysis to identify common themes.
        """
        decision_points = {}
        
        # Common decision types for options trading
        decision_points['overall_action'] = {
            'question': 'What should be the overall portfolio action?',
            'options': ['buy', 'sell', 'hold', 'hedge', 'rebalance']
        }
        
        decision_points['risk_level'] = {
            'question': 'What is the appropriate risk level?',
            'options': ['conservative', 'moderate', 'aggressive']
        }
        
        decision_points['market_outlook'] = {
            'question': 'What is the market outlook?',
            'options': ['bullish', 'bearish', 'neutral', 'volatile']
        }
        
        return decision_points
    
    def _extract_agent_choice(
        self,
        recommendation: Dict[str, Any],
        decision_id: str,
        decision_data: Dict[str, Any]
    ) -> tuple:
        """
        Extract an agent's choice for a specific decision.
        
        Returns:
            Tuple of (choice, confidence, reasoning)
        """
        # This is simplified - in production, would use more sophisticated extraction
        
        # Try to find matching field in recommendation
        if decision_id in recommendation:
            choice_data = recommendation[decision_id]
            if isinstance(choice_data, dict):
                return (
                    choice_data.get('choice'),
                    choice_data.get('confidence', 0.5),
                    choice_data.get('reasoning', '')
                )
            elif isinstance(choice_data, str):
                return choice_data, 0.5, ''
        
        # Default to first option with low confidence
        return decision_data['options'][0], 0.3, 'Default choice'
    
    def get_swarm_metrics(self) -> Dict[str, Any]:
        """Get comprehensive swarm metrics"""
        agent_metrics = {
            agent_id: agent.get_metrics()
            for agent_id, agent in self.agents.items()
        }

        return {
            'swarm_name': self.name,
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime_seconds': (
                (datetime.utcnow() - self.start_time).total_seconds()
                if self.start_time else 0
            ),
            'total_agents': len(self.agents),
            'agent_types': {k: len(v) for k, v in self.agent_types.items()},
            'swarm_metrics': self._metrics,
            'context_metrics': self.shared_context.get_metrics(),
            'consensus_metrics': self.consensus_engine.get_metrics(),
            'agent_metrics': agent_metrics
        }

    def _calculate_portfolio_metrics(
        self,
        portfolio_data: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate institutional-grade portfolio metrics.

        Args:
            portfolio_data: Portfolio positions and values
            market_data: Market data for benchmarking

        Returns:
            Dict with portfolio metrics or None if calculation fails
        """
        if not METRICS_AVAILABLE:
            logger.warning("Portfolio metrics module not available")
            return None

        try:
            positions = portfolio_data.get('positions', [])
            if not positions:
                return None

            # Extract position data
            position_values = []
            position_returns = []

            for pos in positions:
                market_value = pos.get('market_value', 0)
                unrealized_pnl = pos.get('unrealized_pnl', 0)
                initial_value = market_value - unrealized_pnl

                if initial_value > 0:
                    position_values.append(market_value)
                    position_returns.append(unrealized_pnl / initial_value)

            if not position_values:
                return None

            # Calculate portfolio-level metrics
            total_value = sum(position_values)
            weights = pd.Series([v / total_value for v in position_values])

            # Create synthetic return series (simplified for now)
            # In production, this would use historical price data
            portfolio_return = sum(r * w for r, w in zip(position_returns, weights))

            # Create minimal metrics dict
            metrics = {
                'total_return': portfolio_return,
                'position_count': len(positions),
                'total_value': total_value,
                'concentration_risk': float((weights ** 2).sum()),  # HHI
                'effective_n': float(1.0 / ((weights ** 2).sum())) if (weights ** 2).sum() > 0 else 0.0,
                'avg_position_return': np.mean(position_returns) if position_returns else 0.0,
                'win_rate': sum(1 for r in position_returns if r > 0) / len(position_returns) if position_returns else 0.0
            }

            # If we have enough data, calculate advanced metrics
            if len(position_returns) >= 5:
                try:
                    # Create synthetic time series for demonstration
                    # In production, use actual historical data
                    returns_series = pd.Series(position_returns)

                    # Calculate volatility
                    volatility = returns_series.std() * np.sqrt(252)  # Annualized

                    # Calculate Sharpe (simplified)
                    risk_free_rate = 0.04
                    sharpe_ratio = (portfolio_return - risk_free_rate) / volatility if volatility > 0 else 0.0

                    # Calculate Omega ratio (simplified)
                    threshold = 0.0
                    gains = returns_series[returns_series > threshold] - threshold
                    losses = threshold - returns_series[returns_series <= threshold]
                    omega_ratio = gains.sum() / losses.sum() if losses.sum() > 0 else float('inf')

                    metrics.update({
                        'volatility': float(volatility),
                        'sharpe_ratio': float(sharpe_ratio),
                        'omega_ratio': float(omega_ratio) if omega_ratio != float('inf') else 999.0,
                        'max_drawdown': float(min(position_returns)) if position_returns else 0.0
                    })
                except Exception as e:
                    logger.warning(f"Could not calculate advanced metrics: {e}")

            return metrics

        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return None

