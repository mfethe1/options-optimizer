"""
Debug script to identify the exact error in swarm analyze endpoint
"""

import sys
import traceback
from datetime import datetime

# Test the swarm analysis directly without API
def test_swarm_directly():
    """Test swarm analysis directly to see the actual error"""
    print("="*80)
    print("DIRECT SWARM ANALYSIS TEST")
    print("="*80)
    
    try:
        from src.agents.swarm import (
            SwarmCoordinator,
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
        
        print("\n✓ Imports successful")
        
        # Create coordinator
        coordinator = SwarmCoordinator(
            name="TestSwarm",
            max_messages=1000,
            quorum_threshold=0.67
        )
        print("✓ Coordinator created")
        
        # Create agents
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
        ]
        
        for agent in agents:
            coordinator.register_agent(agent)
        
        print(f"✓ Registered {len(agents)} agents")
        
        # Start swarm
        coordinator.start()
        print("✓ Swarm started")
        
        # Create test portfolio
        portfolio_data = {
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
                }
            ],
            'total_value': 6200,
            'unrealized_pnl': 700,
            'initial_value': 5500,
            'peak_value': 6500
        }
        
        market_data = {
            'SPY': {'price': 455.0, 'change_pct': 1.2}
        }
        
        print("\n✓ Test data created")
        print(f"  Portfolio value: ${portfolio_data['total_value']}")
        print(f"  Positions: {len(portfolio_data['positions'])}")
        
        # Run analysis
        print("\nRunning swarm analysis...")
        analysis = coordinator.analyze_portfolio(portfolio_data, market_data)
        print(f"✓ Analysis complete: {len(analysis)} agent results")
        
        # Get recommendations
        print("\nGenerating recommendations...")
        recommendations = coordinator.make_recommendations(
            analysis,
            consensus_method=ConsensusMethod.WEIGHTED
        )
        print(f"✓ Recommendations generated")
        
        # Get metrics
        metrics = coordinator.get_swarm_metrics()
        print(f"✓ Metrics collected")
        
        # Try to create response
        print("\nCreating response object...")
        response_data = {
            'swarm_name': coordinator.name,
            'timestamp': datetime.utcnow().isoformat(),
            'analysis': analysis,
            'recommendations': recommendations,
            'metrics': metrics
        }
        
        print(f"✓ Response data created")
        print(f"  Keys: {list(response_data.keys())}")
        
        # Try Pydantic validation
        print("\nTesting Pydantic validation...")
        from pydantic import BaseModel
        from typing import Dict, Any
        
        class SwarmAnalysisResponse(BaseModel):
            swarm_name: str
            timestamp: str
            analysis: Dict[str, Any]
            recommendations: Dict[str, Any]
            metrics: Dict[str, Any]
        
        try:
            response = SwarmAnalysisResponse(**response_data)
            print("✓ Pydantic validation successful!")
            print(f"  Response: {response.swarm_name}")
        except Exception as e:
            print(f"✗ Pydantic validation FAILED!")
            print(f"  Error: {e}")
            print(f"\nDetailed error:")
            traceback.print_exc()
            
            # Print problematic data
            print(f"\nAnalyzing response data structure:")
            for key, value in response_data.items():
                print(f"\n  {key}: {type(value)}")
                if isinstance(value, dict):
                    print(f"    Keys: {list(value.keys())[:10]}")
                    # Check for non-serializable objects
                    for k, v in list(value.items())[:5]:
                        print(f"      {k}: {type(v)}")
        
        # Stop swarm
        coordinator.stop()
        print("\n✓ Swarm stopped")
        
        print("\n" + "="*80)
        print("TEST COMPLETE - Check output above for errors")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = test_swarm_directly()
    sys.exit(0 if success else 1)

