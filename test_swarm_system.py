"""
Test and Demo Script for Multi-Agent Swarm System

This script demonstrates the complete swarm system in action,
showing how multiple agents collaborate to analyze a portfolio
and make trading recommendations.
"""

import logging
from datetime import datetime
from src.agents.swarm import (
    SwarmCoordinator,
    SharedContext,
    ConsensusEngine,
    MarketAnalystAgent,
    RiskManagerAgent,
    OptionsStrategistAgent,
    TechnicalAnalystAgent,
    SentimentAnalystAgent,
    PortfolioOptimizerAgent,
    TradeExecutorAgent,
    ComplianceOfficerAgent
)
from src.agents.swarm.consensus_engine import ConsensusMethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_swarm():
    """Create and configure the swarm with all agents"""
    logger.info("Creating swarm coordinator...")
    
    # Create coordinator
    coordinator = SwarmCoordinator(
        name="OptionsAnalysisSwarm",
        max_messages=1000,
        quorum_threshold=0.67
    )
    
    # Create all agents
    agents = [
        MarketAnalystAgent(
            agent_id="market_analyst_1",
            shared_context=coordinator.shared_context,
            consensus_engine=coordinator.consensus_engine
        ),
        RiskManagerAgent(
            agent_id="risk_manager_1",
            shared_context=coordinator.shared_context,
            consensus_engine=coordinator.consensus_engine,
            max_portfolio_delta=100.0,
            max_position_size_pct=0.10,
            max_drawdown_pct=0.15
        ),
        OptionsStrategistAgent(
            agent_id="options_strategist_1",
            shared_context=coordinator.shared_context,
            consensus_engine=coordinator.consensus_engine
        ),
        TechnicalAnalystAgent(
            agent_id="technical_analyst_1",
            shared_context=coordinator.shared_context,
            consensus_engine=coordinator.consensus_engine
        ),
        SentimentAnalystAgent(
            agent_id="sentiment_analyst_1",
            shared_context=coordinator.shared_context,
            consensus_engine=coordinator.consensus_engine
        ),
        PortfolioOptimizerAgent(
            agent_id="portfolio_optimizer_1",
            shared_context=coordinator.shared_context,
            consensus_engine=coordinator.consensus_engine
        ),
        TradeExecutorAgent(
            agent_id="trade_executor_1",
            shared_context=coordinator.shared_context,
            consensus_engine=coordinator.consensus_engine
        ),
        ComplianceOfficerAgent(
            agent_id="compliance_officer_1",
            shared_context=coordinator.shared_context,
            consensus_engine=coordinator.consensus_engine
        )
    ]
    
    # Register all agents
    for agent in agents:
        coordinator.register_agent(agent)
        logger.info(f"Registered agent: {agent.agent_id} ({agent.agent_type})")
    
    return coordinator


def create_sample_portfolio():
    """Create sample portfolio data for testing"""
    return {
        'positions': [
            {
                'symbol': 'AAPL',
                'asset_type': 'option',
                'option_type': 'call',
                'strike': 180.0,
                'expiration_date': '2025-03-21',
                'quantity': 10,
                'premium_paid': 5.50,
                'current_price': 6.20,
                'underlying_price': 185.0,
                'delta': 0.65,
                'gamma': 0.05,
                'theta': -0.15,
                'vega': 0.25,
                'market_value': 6200
            },
            {
                'symbol': 'SPY',
                'asset_type': 'option',
                'option_type': 'put',
                'strike': 450.0,
                'expiration_date': '2025-02-21',
                'quantity': 5,
                'premium_paid': 8.00,
                'current_price': 7.50,
                'underlying_price': 455.0,
                'delta': -0.35,
                'gamma': 0.03,
                'theta': -0.20,
                'vega': 0.30,
                'market_value': 3750
            },
            {
                'symbol': 'TSLA',
                'asset_type': 'stock',
                'quantity': 50,
                'purchase_price': 200.0,
                'current_price': 215.0,
                'market_value': 10750
            }
        ],
        'total_value': 20700,
        'initial_value': 19000,
        'unrealized_pnl': 1700,
        'peak_value': 21500
    }


def create_sample_market_data():
    """Create sample market data for testing"""
    return {
        'SPY': {
            'price': 455.0,
            'change_pct': 1.2,
            'volume': 85000000
        },
        'QQQ': {
            'price': 385.0,
            'change_pct': 1.5,
            'volume': 45000000
        },
        'VIX': {
            'price': 18.5,
            'change_pct': -2.3
        }
    }


def print_analysis_results(analysis):
    """Print analysis results in a readable format"""
    print("\n" + "="*80)
    print("SWARM ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nSwarm: {analysis['swarm_name']}")
    print(f"Timestamp: {analysis['timestamp']}")
    print(f"Agents Participated: {analysis['agent_count']}")
    
    print("\n" + "-"*80)
    print("INDIVIDUAL AGENT ANALYSES")
    print("-"*80)
    
    for agent_id, result in analysis['analyses'].items():
        print(f"\n{agent_id} ({result['agent_type']}):")
        
        # Print key findings
        agent_analysis = result['analysis']
        if 'market_regime' in agent_analysis:
            print(f"  Market Regime: {agent_analysis['market_regime']}")
        if 'risk_level' in agent_analysis:
            print(f"  Risk Level: {agent_analysis['risk_level']}")
        if 'recommended_strategies' in agent_analysis:
            print(f"  Strategies: {agent_analysis['recommended_strategies']}")


def print_recommendations(recommendations):
    """Print recommendations in a readable format"""
    print("\n" + "="*80)
    print("SWARM RECOMMENDATIONS")
    print("="*80)
    
    print(f"\nSwarm: {recommendations['swarm_name']}")
    print(f"Consensus Method: {recommendations['consensus_method']}")
    
    print("\n" + "-"*80)
    print("CONSENSUS DECISIONS")
    print("-"*80)
    
    for decision_id, result in recommendations['consensus_recommendations'].items():
        print(f"\n{result['question']}")
        print(f"  Decision: {result['result']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Method: {result['method']}")


def print_metrics(metrics):
    """Print swarm metrics"""
    print("\n" + "="*80)
    print("SWARM METRICS")
    print("="*80)
    
    print(f"\nSwarm: {metrics['swarm_name']}")
    print(f"Status: {'Running' if metrics['is_running'] else 'Stopped'}")
    print(f"Uptime: {metrics['uptime_seconds']:.1f} seconds")
    print(f"Total Agents: {metrics['total_agents']}")
    
    print("\nAgent Types:")
    for agent_type, count in metrics['agent_types'].items():
        print(f"  {agent_type}: {count}")
    
    print("\nSwarm Metrics:")
    for key, value in metrics['swarm_metrics'].items():
        print(f"  {key}: {value}")
    
    print("\nContext Metrics:")
    for key, value in metrics['context_metrics'].items():
        print(f"  {key}: {value}")
    
    print("\nConsensus Metrics:")
    for key, value in metrics['consensus_metrics'].items():
        print(f"  {key}: {value}")


def main():
    """Main test function"""
    print("\n" + "="*80)
    print("MULTI-AGENT SWARM SYSTEM - DEMONSTRATION")
    print("="*80)
    
    # Create swarm
    print("\n1. Creating swarm with 8 specialized agents...")
    coordinator = create_swarm()
    
    # Start swarm
    print("\n2. Starting swarm...")
    coordinator.start()
    
    # Create test data
    print("\n3. Preparing test data...")
    portfolio_data = create_sample_portfolio()
    market_data = create_sample_market_data()
    
    print(f"   Portfolio Value: ${portfolio_data['total_value']:,.2f}")
    print(f"   Unrealized P&L: ${portfolio_data['unrealized_pnl']:,.2f}")
    print(f"   Positions: {len(portfolio_data['positions'])}")
    
    # Run analysis
    print("\n4. Running swarm analysis...")
    analysis = coordinator.analyze_portfolio(portfolio_data, market_data)
    print_analysis_results(analysis)
    
    # Get recommendations
    print("\n5. Generating swarm recommendations...")
    recommendations = coordinator.make_recommendations(
        analysis,
        consensus_method=ConsensusMethod.WEIGHTED
    )
    print_recommendations(recommendations)
    
    # Show metrics
    print("\n6. Collecting swarm metrics...")
    metrics = coordinator.get_swarm_metrics()
    print_metrics(metrics)
    
    # Stop swarm
    print("\n7. Stopping swarm...")
    coordinator.stop()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nThe swarm successfully:")
    print("  ✓ Analyzed the portfolio with 8 specialized agents")
    print("  ✓ Reached consensus on trading decisions")
    print("  ✓ Provided risk-managed recommendations")
    print("  ✓ Maintained comprehensive metrics")
    print("\nAll agents worked together using swarm intelligence principles:")
    print("  • Decentralized decision-making")
    print("  • Stigmergic communication")
    print("  • Entropy-based confidence")
    print("  • Emergent behavior")
    print("\n")


if __name__ == "__main__":
    main()

