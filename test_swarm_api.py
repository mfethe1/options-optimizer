"""
Test Swarm API Integration

Tests the swarm analysis endpoints integrated with the FastAPI backend.
"""

import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"


def test_swarm_status():
    """Test swarm status endpoint"""
    print("\n" + "="*80)
    print("TEST 1: Swarm Status")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/api/swarm/status")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Swarm Name: {data['swarm_name']}")
        print(f"✓ Running: {data['is_running']}")
        print(f"✓ Total Agents: {data['total_agents']}")
        print(f"✓ Agent Types: {data['agent_types']}")
        return True
    else:
        print(f"✗ Failed: {response.status_code}")
        print(response.text)
        return False


def test_list_agents():
    """Test list agents endpoint"""
    print("\n" + "="*80)
    print("TEST 2: List Agents")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/api/swarm/agents")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Total Agents: {data['total_agents']}")
        
        for agent in data['agents']:
            print(f"\n  Agent: {agent['agent_id']}")
            print(f"    Type: {agent['agent_type']}")
            print(f"    Active: {agent['is_active']}")
            print(f"    Priority: {agent['priority']}")
            print(f"    Actions: {agent['metrics']['action_count']}")
        
        return True
    else:
        print(f"✗ Failed: {response.status_code}")
        print(response.text)
        return False


def test_swarm_metrics():
    """Test swarm metrics endpoint"""
    print("\n" + "="*80)
    print("TEST 3: Swarm Metrics")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/api/swarm/metrics")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Swarm: {data['swarm_name']}")
        print(f"✓ Uptime: {data['uptime_seconds']:.1f} seconds")
        
        print("\n  Swarm Metrics:")
        for key, value in data['swarm_metrics'].items():
            print(f"    {key}: {value}")
        
        print("\n  Context Metrics:")
        for key, value in data['context_metrics'].items():
            print(f"    {key}: {value}")
        
        print("\n  Consensus Metrics:")
        for key, value in data['consensus_metrics'].items():
            print(f"    {key}: {value}")
        
        return True
    else:
        print(f"✗ Failed: {response.status_code}")
        print(response.text)
        return False


def test_swarm_analysis():
    """Test swarm portfolio analysis"""
    print("\n" + "="*80)
    print("TEST 4: Swarm Portfolio Analysis")
    print("="*80)
    
    # Sample portfolio data
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
            }
        ],
        'total_value': 9950,
        'unrealized_pnl': 200,
        'initial_value': 9750,
        'peak_value': 10000
    }
    
    market_data = {
        'SPY': {'price': 455.0, 'change_pct': 1.2},
        'QQQ': {'price': 385.0, 'change_pct': 1.5}
    }
    
    request_data = {
        'portfolio_data': portfolio_data,
        'market_data': market_data,
        'consensus_method': 'weighted'
    }
    
    print("Sending analysis request...")
    response = requests.post(
        f"{BASE_URL}/api/swarm/analyze",
        json=request_data,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code == 200:
        data = response.json()
        
        print(f"\n✓ Analysis Complete")
        print(f"  Swarm: {data['swarm_name']}")
        print(f"  Timestamp: {data['timestamp']}")
        
        # Show analysis summary
        analysis = data['analysis']
        print(f"\n  Agents Participated: {analysis['agent_count']}")
        
        # Show recommendations
        recommendations = data['recommendations']
        print(f"\n  Consensus Recommendations:")
        
        for decision_id, result in recommendations['consensus_recommendations'].items():
            print(f"\n    {result['question']}")
            print(f"      Decision: {result['result']}")
            print(f"      Confidence: {result['confidence']:.2%}")
            print(f"      Method: {result['method']}")
        
        # Show metrics
        metrics = data['metrics']
        print(f"\n  Swarm Performance:")
        print(f"    Total Decisions: {metrics['swarm_metrics']['total_decisions']}")
        print(f"    Total Recommendations: {metrics['swarm_metrics']['total_recommendations']}")
        print(f"    Errors: {metrics['swarm_metrics']['total_errors']}")
        print(f"    Consensus Success Rate: {metrics['consensus_metrics']['success_rate']:.1%}")
        
        return True
    else:
        print(f"✗ Failed: {response.status_code}")
        print(response.text)
        return False


def test_swarm_messages():
    """Test swarm messages endpoint"""
    print("\n" + "="*80)
    print("TEST 5: Swarm Messages")
    print("="*80)
    
    response = requests.get(
        f"{BASE_URL}/api/swarm/messages",
        params={'min_priority': 5, 'min_confidence': 0.5}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Total Messages: {data['total_messages']}")
        
        for msg in data['messages'][:5]:  # Show first 5
            print(f"\n  From: {msg['source']}")
            print(f"    Priority: {msg['priority']}")
            print(f"    Confidence: {msg['confidence']:.2f}")
            print(f"    Content: {msg['content']}")
        
        return True
    else:
        print(f"✗ Failed: {response.status_code}")
        print(response.text)
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("SWARM API INTEGRATION TESTS")
    print("="*80)
    print(f"\nTesting API at: {BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Swarm Status", test_swarm_status),
        ("List Agents", test_list_agents),
        ("Swarm Metrics", test_swarm_metrics),
        ("Swarm Analysis", test_swarm_analysis),
        ("Swarm Messages", test_swarm_messages)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Swarm API is fully operational.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the output above.")


if __name__ == "__main__":
    main()

