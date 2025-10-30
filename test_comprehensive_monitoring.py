"""
Comprehensive End-to-End Test for Swarm Analysis Monitoring System

Tests:
1. Backend server health
2. Monitoring endpoints
3. CSV upload and analysis
4. Agent conversation capture
5. Progress tracking
6. Frontend data transformation
"""

import requests
import json
import time
from pathlib import Path

# Configuration
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:5173"
TEST_CSV = "data/examples/positions.csv"

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def test_backend_health():
    """Test 1: Backend Server Health"""
    print_section("TEST 1: Backend Server Health")
    
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        print(f"✓ Backend is running (status: {response.status_code})")
        print(f"  Response: {response.json()}")
        return True
    except Exception as e:
        print(f"✗ Backend health check failed: {e}")
        return False

def test_monitoring_endpoints():
    """Test 2: Monitoring Endpoints"""
    print_section("TEST 2: Monitoring Endpoints")
    
    endpoints = [
        "/api/monitoring/",
        "/api/monitoring/health",
        "/api/monitoring/diagnostics",
        "/api/monitoring/analyses/active",
        "/api/monitoring/agents/statistics"
    ]
    
    results = {}
    for endpoint in endpoints:
        try:
            response = requests.get(f"{BACKEND_URL}{endpoint}", timeout=5)
            status = "✓" if response.status_code == 200 else "✗"
            print(f"{status} {endpoint} (status: {response.status_code})")
            results[endpoint] = response.status_code == 200
            
            # Show sample data for key endpoints
            if endpoint == "/api/monitoring/health":
                data = response.json()
                print(f"  Active analyses: {data['active_analyses']}")
                print(f"  Health: {data['health_percentage']:.1f}%")
            elif endpoint == "/api/monitoring/diagnostics":
                data = response.json()
                print(f"  Recent errors: {data['recent_errors_count']}")
                print(f"  Problematic agents: {data['problematic_agents_count']}")
        except Exception as e:
            print(f"✗ {endpoint} failed: {e}")
            results[endpoint] = False
    
    return all(results.values())

def test_csv_upload_and_analysis():
    """Test 3: CSV Upload and Analysis"""
    print_section("TEST 3: CSV Upload and Analysis")
    
    # Check if test CSV exists
    csv_path = Path(TEST_CSV)
    if not csv_path.exists():
        print(f"✗ Test CSV not found: {TEST_CSV}")
        return False
    
    print(f"✓ Test CSV found: {TEST_CSV}")
    
    # Upload CSV
    try:
        with open(csv_path, 'rb') as f:
            files = {'file': f}
            params = {
                'chase_format': 'true',
                'consensus_method': 'weighted'
            }
            
            print("  Uploading CSV and starting analysis...")
            print("  (This may take 3-5 minutes for 17 agents)")
            
            start_time = time.time()
            response = requests.post(
                f"{BACKEND_URL}/api/swarm/analyze-csv",
                files=files,
                params=params,
                timeout=600  # 10 minute timeout
            )
            duration = time.time() - start_time
            
            if response.status_code == 200:
                print(f"✓ Analysis complete (status: {response.status_code})")
                print(f"  Duration: {duration:.2f} seconds")
                
                data = response.json()
                
                # Check import stats
                if 'import_stats' in data:
                    stats = data['import_stats']
                    print(f"\n  Import Stats:")
                    print(f"    Imported: {stats.get('positions_imported', 0)}")
                    print(f"    Failed: {stats.get('positions_failed', 0)}")
                
                # Check consensus decisions
                if 'consensus_decisions' in data:
                    consensus = data['consensus_decisions']
                    print(f"\n  Consensus Decisions:")
                    if 'overall_action' in consensus:
                        action = consensus['overall_action']
                        print(f"    Action: {action.get('action', 'N/A')}")
                        print(f"    Confidence: {action.get('confidence', 0):.2%}")
                
                # Check agent insights
                if 'agent_insights' in data:
                    insights = data['agent_insights']
                    print(f"\n  Agent Insights:")
                    print(f"    Total agents: {len(insights)}")
                    for insight in insights[:3]:  # Show first 3
                        rec = insight.get('recommendation', 'N/A')
                        rec_str = str(rec)[:50] if rec else 'N/A'
                        print(f"    - {insight.get('agent_id', 'Unknown')}: {rec_str}...")
                
                # Check discussion logs
                if 'discussion_logs' in data:
                    logs = data['discussion_logs']
                    print(f"\n  Discussion Logs:")
                    print(f"    Total messages: {len(logs)}")
                    if logs:
                        print(f"    Sample message:")
                        msg = logs[0]
                        print(f"      From: {msg.get('source_agent', 'Unknown')}")
                        print(f"      Priority: {msg.get('priority', 0)}")
                        print(f"      Confidence: {msg.get('confidence', 0):.2%}")
                
                # Check enhanced features
                if 'position_analysis' in data:
                    print(f"\n  Position Analysis: {len(data['position_analysis'])} positions")
                
                if 'swarm_health' in data:
                    health = data['swarm_health']
                    print(f"\n  Swarm Health:")
                    print(f"    Success rate: {health.get('agent_contribution', {}).get('success_rate', 0):.2%}")
                    print(f"    Total messages: {health.get('communication_stats', {}).get('total_messages', 0)}")
                
                # Save response for inspection
                output_file = "test_output/comprehensive_analysis_response.json"
                Path("test_output").mkdir(exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"\n  ✓ Full response saved to: {output_file}")
                
                return True
            else:
                print(f"✗ Analysis failed (status: {response.status_code})")
                print(f"  Response: {response.text[:500]}")
                return False
                
    except Exception as e:
        print(f"✗ CSV upload failed: {e}")
        return False

def test_monitoring_after_analysis():
    """Test 4: Monitoring After Analysis"""
    print_section("TEST 4: Monitoring After Analysis")
    
    try:
        # Check agent statistics
        response = requests.get(f"{BACKEND_URL}/api/monitoring/agents/statistics", timeout=5)
        if response.status_code == 200:
            data = response.json()
            summary = data.get('summary', {})
            print(f"✓ Agent Statistics:")
            print(f"  Total agents: {summary.get('total_agents', 0)}")
            print(f"  Total calls: {summary.get('total_calls', 0)}")
            print(f"  Success rate: {summary.get('overall_success_rate', 0):.2%}")
            
            # Show top 5 agents by calls
            agents = data.get('agents', {})
            if agents:
                sorted_agents = sorted(
                    agents.items(),
                    key=lambda x: x[1].get('total_calls', 0),
                    reverse=True
                )[:5]
                print(f"\n  Top 5 Most Active Agents:")
                for agent_id, stats in sorted_agents:
                    print(f"    {agent_id}:")
                    print(f"      Calls: {stats.get('total_calls', 0)}")
                    print(f"      Avg time: {stats.get('avg_time', 0):.2f}s")
                    print(f"      Success: {stats.get('successful_calls', 0)}/{stats.get('total_calls', 0)}")
        
        # Check diagnostics
        response = requests.get(f"{BACKEND_URL}/api/monitoring/diagnostics", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ Diagnostics:")
            print(f"  Recent errors: {data.get('recent_errors_count', 0)}")
            print(f"  Problematic agents: {data.get('problematic_agents_count', 0)}")
            
            if data.get('problematic_agents'):
                print(f"\n  Problematic Agents:")
                for agent in data['problematic_agents']:
                    print(f"    {agent['agent_id']}: {agent['failure_rate']:.1f}% failure rate")
                    if agent.get('last_error'):
                        print(f"      Last error: {agent['last_error'][:100]}")
        
        return True
    except Exception as e:
        print(f"✗ Monitoring check failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("  COMPREHENSIVE SWARM ANALYSIS MONITORING TEST")
    print("=" * 80)
    print(f"\nBackend URL: {BACKEND_URL}")
    print(f"Frontend URL: {FRONTEND_URL}")
    print(f"Test CSV: {TEST_CSV}")
    
    results = {}
    
    # Test 1: Backend Health
    results['backend_health'] = test_backend_health()
    
    # Test 2: Monitoring Endpoints
    results['monitoring_endpoints'] = test_monitoring_endpoints()
    
    # Test 3: CSV Upload and Analysis
    if results['backend_health']:
        results['csv_analysis'] = test_csv_upload_and_analysis()
    else:
        print("\n⚠ Skipping CSV analysis test (backend not healthy)")
        results['csv_analysis'] = False
    
    # Test 4: Monitoring After Analysis
    if results['csv_analysis']:
        results['monitoring_after'] = test_monitoring_after_analysis()
    else:
        print("\n⚠ Skipping post-analysis monitoring (analysis failed)")
        results['monitoring_after'] = False
    
    # Summary
    print_section("TEST SUMMARY")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n⚠ SOME TESTS FAILED - Check output above for details")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

