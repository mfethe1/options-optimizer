"""
Test if swarm routes can be imported successfully.
"""
import sys
import traceback

print("=" * 60)
print("TESTING SWARM ROUTES IMPORT")
print("=" * 60)

# Test 1: Import swarm_routes module
print("\n[1] Attempting to import swarm_routes module...")
try:
    from src.api import swarm_routes
    print("    ✓ swarm_routes module imported successfully")
except Exception as e:
    print(f"    ✗ Failed to import swarm_routes module")
    print(f"    → Error: {e}")
    print("\n    Full traceback:")
    traceback.print_exc()
    sys.exit(1)

# Test 2: Check if router exists
print("\n[2] Checking if router object exists...")
try:
    router = swarm_routes.router
    print(f"    ✓ Router object exists")
    print(f"    → Prefix: {router.prefix}")
    print(f"    → Tags: {router.tags}")
except Exception as e:
    print(f"    ✗ Router object not found: {e}")
    sys.exit(1)

# Test 3: Check routes
print("\n[3] Checking registered routes...")
try:
    routes = router.routes
    print(f"    ✓ Found {len(routes)} routes")
    
    for route in routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            methods = ', '.join(route.methods) if route.methods else 'N/A'
            print(f"      - {route.path} [{methods}]")
    
    # Check for analyze-csv specifically
    analyze_csv_found = any(
        hasattr(route, 'path') and 'analyze-csv' in route.path 
        for route in routes
    )
    
    if analyze_csv_found:
        print("\n    ✓ analyze-csv route found in router")
    else:
        print("\n    ✗ analyze-csv route NOT found in router")
        
except Exception as e:
    print(f"    ✗ Error checking routes: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check if swarm coordinator can be initialized
print("\n[4] Testing swarm coordinator initialization...")
try:
    coordinator = swarm_routes._swarm_coordinator
    if coordinator:
        print(f"    ✓ Swarm coordinator exists")
        print(f"    → Agents: {len(coordinator.agents) if hasattr(coordinator, 'agents') else 'unknown'}")
    else:
        print("    ⚠ Swarm coordinator is None (will be initialized on first use)")
except Exception as e:
    print(f"    ✗ Error accessing swarm coordinator: {e}")
    traceback.print_exc()

# Test 5: Check LLMRecommendationAgent import
print("\n[5] Testing LLMRecommendationAgent import...")
try:
    from src.agents.swarm.agents.llm_recommendation_agent import LLMRecommendationAgent
    print("    ✓ LLMRecommendationAgent imported successfully")
except Exception as e:
    print(f"    ✗ Failed to import LLMRecommendationAgent")
    print(f"    → Error: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("IMPORT TEST COMPLETE")
print("=" * 60)

