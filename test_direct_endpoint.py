"""
Test the endpoint function directly to see the actual error
"""

import sys
import traceback
import asyncio

async def test_endpoint():
    """Test the endpoint directly"""
    try:
        from src.api.swarm_routes import analyze_portfolio_with_swarm, SwarmAnalysisRequest, get_swarm_coordinator
        
        # Create request
        request = SwarmAnalysisRequest(
            portfolio_data={
                'positions': [{
                    'symbol': 'AAPL',
                    'asset_type': 'option',
                    'quantity': 10,
                    'market_value': 1000
                }],
                'total_value': 1000
            },
            market_data={},
            consensus_method='weighted'
        )
        
        # Get coordinator
        coordinator = get_swarm_coordinator()
        
        print("Calling analyze_portfolio_with_swarm...")
        
        # Call endpoint
        result = await analyze_portfolio_with_swarm(request, coordinator)
        
        print(f"\n✓ SUCCESS!")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_endpoint())
    sys.exit(0 if success else 1)

