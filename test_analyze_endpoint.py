"""
Test the analyze endpoint with detailed error capture
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_analyze_simple():
    """Test with minimal data"""
    print("\n" + "="*80)
    print("TEST: Simple Portfolio Analysis")
    print("="*80)
    
    payload = {
        'portfolio_data': {
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
        },
        'market_data': {
            'SPY': {'price': 455.0, 'change_pct': 1.2}
        },
        'consensus_method': 'weighted'
    }
    
    print(f"\nSending request to {BASE_URL}/api/swarm/analyze")
    print(f"Payload size: {len(json.dumps(payload))} bytes")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/swarm/analyze",
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        
        print(f"\nResponse Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ SUCCESS!")
            print(f"  Swarm: {data.get('swarm_name')}")
            print(f"  Timestamp: {data.get('timestamp')}")
            print(f"  Analysis keys: {list(data.get('analysis', {}).keys())}")
            print(f"  Recommendations keys: {list(data.get('recommendations', {}).keys())}")
            print(f"  Metrics keys: {list(data.get('metrics', {}).keys())}")
            return True
        else:
            print(f"\n✗ FAILED!")
            print(f"  Status: {response.status_code}")
            print(f"  Response: {response.text}")
            
            # Try to get more details
            try:
                error_data = response.json()
                print(f"  Error detail: {error_data}")
            except:
                pass
            
            return False
            
    except requests.exceptions.Timeout:
        print(f"\n✗ TIMEOUT after 60 seconds")
        return False
    except Exception as e:
        print(f"\n✗ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_analyze_empty():
    """Test with empty portfolio"""
    print("\n" + "="*80)
    print("TEST: Empty Portfolio Analysis")
    print("="*80)
    
    payload = {
        'portfolio_data': {
            'positions': [],
            'total_value': 0,
            'unrealized_pnl': 0,
            'initial_value': 0,
            'peak_value': 0
        },
        'market_data': {},
        'consensus_method': 'weighted'
    }
    
    print(f"\nSending request to {BASE_URL}/api/swarm/analyze")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/swarm/analyze",
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        
        print(f"\nResponse Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ SUCCESS with empty portfolio!")
            return True
        else:
            print(f"✗ FAILED: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ EXCEPTION: {e}")
        return False


def test_analyze_no_data():
    """Test with no portfolio data (should load from position manager)"""
    print("\n" + "="*80)
    print("TEST: No Portfolio Data (Load from Position Manager)")
    print("="*80)
    
    payload = {
        'consensus_method': 'weighted'
    }
    
    print(f"\nSending request to {BASE_URL}/api/swarm/analyze")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/swarm/analyze",
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        
        print(f"\nResponse Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ SUCCESS loading from position manager!")
            return True
        else:
            print(f"✗ FAILED: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ EXCEPTION: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SWARM ANALYZE ENDPOINT TESTS")
    print("="*80)
    
    results = []
    
    # Test 1: Simple portfolio
    results.append(("Simple Portfolio", test_analyze_simple()))
    
    # Test 2: Empty portfolio
    results.append(("Empty Portfolio", test_analyze_empty()))
    
    # Test 3: No data (load from position manager)
    results.append(("Load from Position Manager", test_analyze_no_data()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

