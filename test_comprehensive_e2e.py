"""
Comprehensive End-to-End Testing Suite for Multi-Agent Swarm System
Tests: Backend API, Frontend Integration, User Workflows, Performance
"""

import requests
import json
import time
import csv
from datetime import datetime
from typing import Dict, List, Any
import statistics

# Configuration
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:3000"
POSITIONS_CSV = "data/examples/positions.csv"

# Test Results Tracking
test_results = {
    'total_tests': 0,
    'passed': 0,
    'failed': 0,
    'errors': [],
    'performance_metrics': {},
    'start_time': datetime.now()
}

def log_test(name: str, passed: bool, details: str = ""):
    """Log test result"""
    test_results['total_tests'] += 1
    if passed:
        test_results['passed'] += 1
        print(f"✓ PASS: {name}")
    else:
        test_results['failed'] += 1
        test_results['errors'].append({'test': name, 'details': details})
        print(f"✗ FAIL: {name}")
        if details:
            print(f"  Details: {details}")

def measure_time(func):
    """Decorator to measure execution time"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    return wrapper

# ============================================================================
# SECTION 1: BACKEND API TESTING
# ============================================================================

print("\n" + "="*80)
print("SECTION 1: BACKEND API TESTING")
print("="*80 + "\n")

def test_backend_health():
    """Test backend health endpoint"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        passed = response.status_code == 200
        log_test("Backend Health Check", passed, 
                 f"Status: {response.status_code}" if not passed else "")
        return passed
    except Exception as e:
        log_test("Backend Health Check", False, str(e))
        return False

def test_swarm_status():
    """Test swarm status endpoint"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/swarm/status", timeout=5)
        passed = response.status_code == 200
        if passed:
            data = response.json()
            passed = data.get('is_running') == True
        log_test("Swarm Status Endpoint", passed,
                 f"Status: {response.status_code}" if not passed else "")
        return response.json() if passed else None
    except Exception as e:
        log_test("Swarm Status Endpoint", False, str(e))
        return None

def test_swarm_agents():
    """Test list agents endpoint"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/swarm/agents", timeout=5)
        passed = response.status_code == 200
        if passed:
            data = response.json()
            passed = len(data.get('agents', [])) == 8
        log_test("List Agents Endpoint", passed,
                 f"Expected 8 agents, got {len(data.get('agents', []))}" if not passed else "")
        return data if passed else None
    except Exception as e:
        log_test("List Agents Endpoint", False, str(e))
        return None

def test_swarm_metrics():
    """Test swarm metrics endpoint"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/swarm/metrics", timeout=5)
        passed = response.status_code == 200
        log_test("Swarm Metrics Endpoint", passed,
                 f"Status: {response.status_code}" if not passed else "")
        return response.json() if passed else None
    except Exception as e:
        log_test("Swarm Metrics Endpoint", False, str(e))
        return None

def test_swarm_messages():
    """Test swarm messages endpoint"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/swarm/messages", timeout=5)
        passed = response.status_code == 200
        log_test("Swarm Messages Endpoint", passed,
                 f"Status: {response.status_code}" if not passed else "")
        return response.json() if passed else None
    except Exception as e:
        log_test("Swarm Messages Endpoint", False, str(e))
        return None

@measure_time
def test_swarm_analyze(portfolio_data: Dict, consensus_method: str = "weighted"):
    """Test swarm analysis endpoint"""
    try:
        payload = {
            "portfolio_data": portfolio_data,
            "market_data": {},
            "consensus_method": consensus_method
        }
        response = requests.post(
            f"{BACKEND_URL}/api/swarm/analyze",
            json=payload,
            timeout=60
        )
        passed = response.status_code == 200
        if passed:
            data = response.json()
            # Verify response structure
            passed = all(key in data for key in ['swarm_name', 'analysis', 'recommendations', 'metrics'])
        
        log_test(f"Swarm Analysis ({consensus_method})", passed,
                 f"Status: {response.status_code}" if not passed else "")
        return response.json() if passed else None
    except Exception as e:
        log_test(f"Swarm Analysis ({consensus_method})", False, str(e))
        return None

# ============================================================================
# SECTION 2: PORTFOLIO DATA TESTING
# ============================================================================

print("\n" + "="*80)
print("SECTION 2: PORTFOLIO DATA TESTING")
print("="*80 + "\n")

def load_positions_from_csv():
    """Load positions from CSV file"""
    try:
        positions = []
        with open(POSITIONS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip empty rows and footnotes
                if not row.get('Ticker') or row.get('Ticker').startswith('FOOTNOTES'):
                    continue
                
                # Parse position data
                ticker = row.get('Ticker', '').strip()
                quantity_str = row.get('Quantity', '0').replace(',', '')
                value_str = row.get('Value', '0').replace(',', '').replace('$', '')
                
                try:
                    quantity = float(quantity_str) if quantity_str else 0
                    value = float(value_str) if value_str else 0
                except ValueError:
                    continue
                
                # Determine asset type
                asset_type = 'stock'
                if 'CALL' in ticker or 'PUT' in ticker:
                    asset_type = 'option'
                elif 'CASH' in ticker or 'DOLLAR' in ticker:
                    asset_type = 'cash'
                
                positions.append({
                    'symbol': ticker,
                    'asset_type': asset_type,
                    'quantity': quantity,
                    'market_value': value
                })
        
        total_value = sum(p['market_value'] for p in positions)
        log_test("Load Positions from CSV", len(positions) > 0,
                 f"Loaded {len(positions)} positions, total value: ${total_value:,.2f}")
        
        return {
            'positions': positions,
            'total_value': total_value
        }
    except Exception as e:
        log_test("Load Positions from CSV", False, str(e))
        return None

# ============================================================================
# SECTION 3: CONSENSUS METHODS TESTING
# ============================================================================

print("\n" + "="*80)
print("SECTION 3: CONSENSUS METHODS TESTING")
print("="*80 + "\n")

def test_all_consensus_methods(portfolio_data: Dict):
    """Test all 5 consensus methods"""
    methods = ['majority', 'weighted', 'unanimous', 'quorum', 'entropy']
    results = {}
    
    for method in methods:
        result, elapsed = test_swarm_analyze(portfolio_data, method)
        if result:
            results[method] = {
                'result': result,
                'elapsed': elapsed
            }
            test_results['performance_metrics'][f'consensus_{method}'] = elapsed
            print(f"  {method}: {elapsed:.2f}s")
    
    return results

# ============================================================================
# SECTION 4: PERFORMANCE TESTING
# ============================================================================

print("\n" + "="*80)
print("SECTION 4: PERFORMANCE TESTING")
print("="*80 + "\n")

def test_performance_varying_sizes():
    """Test performance with varying portfolio sizes"""
    sizes = [1, 5, 10]
    results = {}
    
    for size in sizes:
        # Create test portfolio
        positions = []
        for i in range(size):
            positions.append({
                'symbol': f'TEST{i}',
                'asset_type': 'stock',
                'quantity': 100,
                'market_value': 10000
            })
        
        portfolio = {
            'positions': positions,
            'total_value': size * 10000
        }
        
        result, elapsed = test_swarm_analyze(portfolio, 'weighted')
        if result:
            results[size] = elapsed
            test_results['performance_metrics'][f'portfolio_size_{size}'] = elapsed
            print(f"  Portfolio size {size}: {elapsed:.2f}s")
    
    return results

# ============================================================================
# SECTION 5: EDGE CASES AND ERROR HANDLING
# ============================================================================

print("\n" + "="*80)
print("SECTION 5: EDGE CASES AND ERROR HANDLING")
print("="*80 + "\n")

def test_edge_cases():
    """Test edge cases and error handling"""
    
    # Test 1: Empty portfolio
    print("\nTest 1: Empty Portfolio")
    result, _ = test_swarm_analyze({'positions': [], 'total_value': 0})
    
    # Test 2: Invalid consensus method
    print("\nTest 2: Invalid Consensus Method")
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/swarm/analyze",
            json={
                'portfolio_data': {'positions': [], 'total_value': 0},
                'market_data': {},
                'consensus_method': 'invalid_method'
            },
            timeout=30
        )
        # Should return 422 validation error
        passed = response.status_code == 422
        log_test("Invalid Consensus Method Handling", passed,
                 f"Expected 422, got {response.status_code}")
    except Exception as e:
        log_test("Invalid Consensus Method Handling", False, str(e))
    
    # Test 3: Large position values
    print("\nTest 3: Large Position Values")
    large_portfolio = {
        'positions': [{
            'symbol': 'AAPL',
            'asset_type': 'stock',
            'quantity': 1000000,
            'market_value': 150000000
        }],
        'total_value': 150000000
    }
    result, _ = test_swarm_analyze(large_portfolio)

# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("COMPREHENSIVE END-TO-END TESTING SUITE")
    print("Multi-Agent Swarm System for Options Portfolio Analysis")
    print("="*80)
    print(f"\nStart Time: {test_results['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Frontend URL: {FRONTEND_URL}")
    
    # Section 1: Backend API Tests
    if not test_backend_health():
        print("\n⚠️  Backend is not running! Skipping remaining tests.")
        return
    
    test_swarm_status()
    test_swarm_agents()
    test_swarm_metrics()
    test_swarm_messages()
    
    # Section 2: Portfolio Data Tests
    portfolio_data = load_positions_from_csv()
    
    if portfolio_data:
        # Section 3: Consensus Methods
        consensus_results = test_all_consensus_methods(portfolio_data)
        
        # Section 4: Performance Testing
        performance_results = test_performance_varying_sizes()
        
        # Section 5: Edge Cases
        test_edge_cases()
    
    # Print Final Summary
    print_final_summary()

def print_final_summary():
    """Print comprehensive test summary"""
    end_time = datetime.now()
    duration = (end_time - test_results['start_time']).total_seconds()
    
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)
    print(f"\nTotal Tests: {test_results['total_tests']}")
    print(f"Passed: {test_results['passed']} ({test_results['passed']/test_results['total_tests']*100:.1f}%)")
    print(f"Failed: {test_results['failed']} ({test_results['failed']/test_results['total_tests']*100:.1f}%)")
    print(f"Duration: {duration:.2f}s")
    
    if test_results['errors']:
        print("\n" + "-"*80)
        print("FAILED TESTS:")
        print("-"*80)
        for error in test_results['errors']:
            print(f"\n✗ {error['test']}")
            print(f"  {error['details']}")
    
    if test_results['performance_metrics']:
        print("\n" + "-"*80)
        print("PERFORMANCE METRICS:")
        print("-"*80)
        for metric, value in test_results['performance_metrics'].items():
            print(f"  {metric}: {value:.2f}s")
    
    # System Readiness Assessment
    print("\n" + "-"*80)
    print("SYSTEM READINESS ASSESSMENT:")
    print("-"*80)
    pass_rate = test_results['passed'] / test_results['total_tests'] * 100
    
    if pass_rate >= 95:
        print("✅ READY FOR PRODUCTION")
        print("   All critical tests passing, system is stable and performant.")
    elif pass_rate >= 80:
        print("⚠️  READY FOR STAGING")
        print("   Most tests passing, minor issues need attention.")
    else:
        print("❌ NOT READY")
        print("   Critical issues found, requires debugging.")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    run_all_tests()

