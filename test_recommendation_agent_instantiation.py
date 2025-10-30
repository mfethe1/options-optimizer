"""
Test if LLMRecommendationAgent can be instantiated successfully.
"""
import sys
import traceback

print("=" * 60)
print("TESTING LLMRecommendationAgent INSTANTIATION")
print("=" * 60)

# Test 1: Import the agent
print("\n[1] Importing LLMRecommendationAgent...")
try:
    from src.agents.swarm.agents.llm_recommendation_agent import LLMRecommendationAgent
    from src.agents.swarm.shared_context import SharedContext
    from src.agents.swarm.consensus_engine import ConsensusEngine
    print("    ✓ Imports successful")
except Exception as e:
    print(f"    ✗ Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 2: Create dependencies
print("\n[2] Creating dependencies (SharedContext, ConsensusEngine)...")
try:
    shared_context = SharedContext()
    consensus_engine = ConsensusEngine()
    print("    ✓ Dependencies created")
except Exception as e:
    print(f"    ✗ Failed to create dependencies: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 3: Instantiate the agent
print("\n[3] Instantiating LLMRecommendationAgent...")
try:
    agent = LLMRecommendationAgent(
        agent_id="test_recommendation_agent",
        shared_context=shared_context,
        consensus_engine=consensus_engine,
        preferred_model="anthropic"
    )
    print("    ✓ Agent instantiated successfully")
    print(f"    → Agent ID: {agent.agent_id}")
    print(f"    → Agent Type: {agent.agent_type}")
    print(f"    → Preferred Model: {agent.preferred_model}")
except TypeError as e:
    if "abstract" in str(e).lower():
        print(f"    ✗ ABSTRACT METHOD ERROR: {e}")
        print("\n    This means the agent is missing required abstract methods.")
        print("    Check BaseSwarmAgent for all @abstractmethod definitions.")
    else:
        print(f"    ✗ Type error: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"    ✗ Instantiation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check required methods exist
print("\n[4] Checking required methods...")
required_methods = ['analyze', 'make_recommendation']
all_methods_exist = True

for method_name in required_methods:
    if hasattr(agent, method_name):
        method = getattr(agent, method_name)
        if callable(method):
            print(f"    ✓ {method_name}() exists and is callable")
        else:
            print(f"    ✗ {method_name} exists but is not callable")
            all_methods_exist = False
    else:
        print(f"    ✗ {method_name}() is MISSING")
        all_methods_exist = False

if not all_methods_exist:
    print("\n    ⚠ Some required methods are missing!")
    sys.exit(1)

# Test 5: Test make_recommendation with mock data
print("\n[5] Testing make_recommendation() with mock data...")
try:
    mock_analysis = {
        'recommendations': [
            {
                'symbol': 'NVDA',
                'assessment': 'Strong position',
                'action': 'HOLD',
                'stock_alternative': None,
                'option_alternative': None
            },
            {
                'symbol': 'TSLA',
                'assessment': 'Underperforming',
                'action': 'REPLACE',
                'stock_alternative': {'symbol': 'MSFT'},
                'option_alternative': {'symbol': 'MSFT', 'type': 'CALL'}
            }
        ],
        'total_positions_analyzed': 2,
        'replacements_recommended': 1
    }
    
    recommendation = agent.make_recommendation(mock_analysis)
    
    if 'error' in recommendation:
        print(f"    ✗ make_recommendation() returned error: {recommendation['error']}")
        sys.exit(1)
    
    print("    ✓ make_recommendation() executed successfully")
    print(f"    → Overall Action: {recommendation.get('overall_action', {}).get('choice', 'N/A')}")
    print(f"    → Confidence: {recommendation.get('overall_action', {}).get('confidence', 0):.2f}")
    print(f"    → Reasoning: {recommendation.get('overall_action', {}).get('reasoning', 'N/A')}")
    print(f"    → Positions Analyzed: {recommendation.get('positions_analyzed', 0)}")
    print(f"    → Replacements Recommended: {recommendation.get('replacements_recommended', 0)}")
    
except Exception as e:
    print(f"    ✗ make_recommendation() failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
print("\nLLMRecommendationAgent can be instantiated and used successfully!")

