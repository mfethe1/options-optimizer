"""
Test if the swarm analyze-csv endpoint exists and is accessible.
"""
import requests
import sys

def test_endpoint():
    """Test if the endpoint exists and is accessible."""
    
    print("=" * 60)
    print("TESTING SWARM ANALYZE-CSV ENDPOINT")
    print("=" * 60)
    
    # Test 1: Server running?
    print("\n[1] Checking if server is running...")
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        print(f"    ✓ Server is running (status: {response.status_code})")
    except requests.exceptions.ConnectionError:
        print("    ✗ Server is NOT running on port 8000")
        print("    → Start it with: python -m uvicorn src.api.main:app --port 8000 --reload")
        return False
    except Exception as e:
        print(f"    ✗ Error connecting to server: {e}")
        return False
    
    # Test 2: Get OpenAPI spec
    print("\n[2] Fetching OpenAPI specification...")
    try:
        spec_response = requests.get('http://localhost:8000/openapi.json', timeout=5)
        spec = spec_response.json()
        print(f"    ✓ OpenAPI spec retrieved ({len(spec.get('paths', {}))} endpoints)")
    except Exception as e:
        print(f"    ✗ Error fetching OpenAPI spec: {e}")
        return False
    
    # Test 3: Check if endpoint exists
    print("\n[3] Checking if /api/swarm/analyze-csv exists...")
    target_endpoint = '/api/swarm/analyze-csv'
    
    if target_endpoint in spec['paths']:
        print(f"    ✓ Endpoint {target_endpoint} EXISTS")
        methods = list(spec['paths'][target_endpoint].keys())
        print(f"    → Methods: {methods}")
        
        if 'post' in methods:
            post_spec = spec['paths'][target_endpoint]['post']
            print(f"    → Summary: {post_spec.get('summary', 'N/A')}")
            print(f"    → Parameters: {len(post_spec.get('parameters', []))}")
            
            # Show parameters
            for param in post_spec.get('parameters', []):
                print(f"      - {param.get('name')} ({param.get('in')}): {param.get('schema', {}).get('type', 'unknown')}")
    else:
        print(f"    ✗ Endpoint {target_endpoint} NOT FOUND")
        
        # Show available swarm endpoints
        print("\n    Available swarm endpoints:")
        swarm_endpoints = [path for path in spec['paths'] if 'swarm' in path.lower()]
        if swarm_endpoints:
            for path in swarm_endpoints:
                methods = list(spec['paths'][path].keys())
                print(f"      - {path} [{', '.join(methods).upper()}]")
        else:
            print("      (none found - swarm routes may not be registered)")
        
        # Show all endpoints
        print("\n    All available endpoints:")
        for path in sorted(spec['paths'].keys())[:20]:  # First 20
            methods = list(spec['paths'][path].keys())
            print(f"      - {path} [{', '.join(methods).upper()}]")
        
        if len(spec['paths']) > 20:
            print(f"      ... and {len(spec['paths']) - 20} more")
        
        return False
    
    # Test 4: Try to access the endpoint
    print("\n[4] Testing endpoint accessibility...")
    try:
        # Try OPTIONS request (CORS preflight)
        options_response = requests.options(f'http://localhost:8000{target_endpoint}', timeout=5)
        print(f"    ✓ OPTIONS request successful (status: {options_response.status_code})")
        print(f"    → Allowed methods: {options_response.headers.get('Allow', 'N/A')}")
    except Exception as e:
        print(f"    ⚠ OPTIONS request failed: {e}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_endpoint()
    sys.exit(0 if success else 1)

