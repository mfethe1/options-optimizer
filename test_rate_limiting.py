"""
Test Rate Limiting Implementation
Tests that rate limits are properly enforced on different endpoints
"""

import requests
import time
from datetime import datetime

# Configuration
BACKEND_URL = "http://localhost:8000"

# Test results
results = {
    'tests': [],
    'start_time': datetime.now()
}

def log_result(test_name: str, passed: bool, details: str = ""):
    """Log test result"""
    result = {
        'name': test_name,
        'passed': passed,
        'details': details,
        'timestamp': datetime.now().isoformat()
    }
    results['tests'].append(result)
    
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {test_name}")
    if details:
        print(f"  {details}")

print("\n" + "="*80)
print("RATE LIMITING TESTS")
print("="*80)
print(f"\nBackend: {BACKEND_URL}")
print(f"Start Time: {results['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# TEST 1: Health Check - Should have very high limit (1000/minute)
# ============================================================================

print("\n" + "-"*80)
print("TEST 1: Health Check Endpoint (1000/minute limit)")
print("-"*80)

success_count = 0
rate_limited = False

for i in range(20):
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            success_count += 1
        elif response.status_code == 429:
            rate_limited = True
            break
    except Exception as e:
        print(f"  Request {i+1} failed: {e}")
        break

log_result(
    "Health Check - High Limit",
    success_count >= 20 and not rate_limited,
    f"Successful requests: {success_count}/20, Rate limited: {rate_limited}"
)

# ============================================================================
# TEST 2: Read Endpoint - Should have 100/minute limit
# ============================================================================

print("\n" + "-"*80)
print("TEST 2: Read Endpoint (100/minute limit)")
print("-"*80)

# Test with positions endpoint (read operation)
success_count = 0
rate_limited = False
first_429 = None

for i in range(15):
    try:
        response = requests.get(
            f"{BACKEND_URL}/api/positions",
            params={'user_id': 'test_user', 'status': 'open'},
            timeout=5
        )
        if response.status_code in [200, 404]:  # 404 is OK (no positions)
            success_count += 1
        elif response.status_code == 429:
            rate_limited = True
            first_429 = i + 1
            break
    except Exception as e:
        print(f"  Request {i+1} failed: {e}")
        break
    
    time.sleep(0.1)  # Small delay between requests

log_result(
    "Read Endpoint - Moderate Limit",
    success_count >= 15 and not rate_limited,
    f"Successful requests: {success_count}/15, Rate limited at: {first_429 if first_429 else 'Never'}"
)

# ============================================================================
# TEST 3: Swarm Analysis - Should have 5/minute limit
# ============================================================================

print("\n" + "-"*80)
print("TEST 3: Swarm Analysis Endpoint (5/minute limit)")
print("-"*80)

success_count = 0
rate_limited = False
first_429 = None

# Create test portfolio
test_portfolio = {
    'portfolio_data': {
        'positions': [
            {
                'symbol': 'AAPL',
                'asset_type': 'stock',
                'quantity': 100,
                'market_value': 15000
            }
        ],
        'total_value': 15000
    },
    'market_data': {},
    'consensus_method': 'weighted'
}

for i in range(10):
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/swarm/analyze",
            json=test_portfolio,
            timeout=30
        )
        if response.status_code == 200:
            success_count += 1
            print(f"  Request {i+1}: Success (200)")
        elif response.status_code == 429:
            rate_limited = True
            first_429 = i + 1
            print(f"  Request {i+1}: Rate Limited (429)")
            
            # Check for rate limit headers
            retry_after = response.headers.get('Retry-After')
            remaining = response.headers.get('X-RateLimit-Remaining')
            print(f"    Retry-After: {retry_after}, Remaining: {remaining}")
            break
        else:
            print(f"  Request {i+1}: Error ({response.status_code})")
    except Exception as e:
        print(f"  Request {i+1} failed: {e}")
        break
    
    time.sleep(0.5)  # Small delay between requests

log_result(
    "Swarm Analysis - Strict Limit",
    rate_limited and first_429 <= 6,  # Should be rate limited within 6 requests
    f"Successful requests: {success_count}, Rate limited at request: {first_429}"
)

# ============================================================================
# TEST 4: Rate Limit Headers
# ============================================================================

print("\n" + "-"*80)
print("TEST 4: Rate Limit Headers")
print("-"*80)

try:
    response = requests.get(f"{BACKEND_URL}/health", timeout=5)
    
    # Check for rate limit headers
    has_headers = False
    headers_found = []
    
    if 'X-RateLimit-Limit' in response.headers:
        headers_found.append(f"X-RateLimit-Limit: {response.headers['X-RateLimit-Limit']}")
        has_headers = True
    
    if 'X-RateLimit-Remaining' in response.headers:
        headers_found.append(f"X-RateLimit-Remaining: {response.headers['X-RateLimit-Remaining']}")
        has_headers = True
    
    if 'X-RateLimit-Reset' in response.headers:
        headers_found.append(f"X-RateLimit-Reset: {response.headers['X-RateLimit-Reset']}")
        has_headers = True
    
    log_result(
        "Rate Limit Headers Present",
        has_headers,
        f"Headers: {', '.join(headers_found) if headers_found else 'None found'}"
    )
except Exception as e:
    log_result("Rate Limit Headers Present", False, str(e))

# ============================================================================
# TEST 5: 429 Response Format
# ============================================================================

print("\n" + "-"*80)
print("TEST 5: 429 Response Format")
print("-"*80)

# Make rapid requests to trigger rate limit
for i in range(15):
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/swarm/analyze",
            json=test_portfolio,
            timeout=30
        )
        
        if response.status_code == 429:
            # Check response format
            has_retry_after = 'Retry-After' in response.headers
            has_content = len(response.text) > 0
            
            log_result(
                "429 Response Format",
                has_retry_after,
                f"Retry-After header: {has_retry_after}, Content: {response.text[:100]}"
            )
            break
    except Exception as e:
        continue
    
    time.sleep(0.2)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

total_tests = len(results['tests'])
passed_tests = sum(1 for t in results['tests'] if t['passed'])
failed_tests = total_tests - passed_tests
pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

end_time = datetime.now()
total_duration = (end_time - results['start_time']).total_seconds()

print(f"\nTotal Tests: {total_tests}")
print(f"Passed: {passed_tests} ({pass_rate:.1f}%)")
print(f"Failed: {failed_tests}")
print(f"Total Duration: {total_duration:.2f}s")

if failed_tests > 0:
    print(f"\n❌ Failed Tests:")
    for test in results['tests']:
        if not test['passed']:
            print(f"  - {test['name']}: {test['details']}")

# System Readiness
print(f"\n🎯 Rate Limiting Status:")
if pass_rate >= 80:
    print("  ✅ WORKING CORRECTLY")
elif pass_rate >= 60:
    print("  ⚠️  PARTIALLY WORKING")
else:
    print("  ❌ NOT WORKING")

print("\n" + "="*80)

