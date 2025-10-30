"""
Quick test for swarm endpoint rate limiting
"""

import requests
import time

BACKEND_URL = "http://localhost:8000"

# Test portfolio
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

print("\n" + "="*80)
print("SWARM ENDPOINT RATE LIMITING TEST")
print("="*80)
print(f"\nEndpoint: {BACKEND_URL}/api/swarm/analyze")
print("Expected Limit: 5 requests per minute")
print("\nMaking 10 rapid requests...")
print("-"*80)

success_count = 0
rate_limited_at = None

for i in range(10):
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/swarm/analyze",
            json=test_portfolio,
            timeout=30
        )
        
        # Check headers
        limit = response.headers.get('X-RateLimit-Limit')
        remaining = response.headers.get('X-RateLimit-Remaining')
        reset = response.headers.get('X-RateLimit-Reset')
        
        if response.status_code == 200:
            success_count += 1
            print(f"Request {i+1}: ✓ SUCCESS (200) - Remaining: {remaining}/{limit}")
        elif response.status_code == 429:
            rate_limited_at = i + 1
            retry_after = response.headers.get('Retry-After')
            print(f"Request {i+1}: ✗ RATE LIMITED (429) - Retry-After: {retry_after}s")
            print(f"\n🎯 Rate limit enforced after {success_count} successful requests!")
            break
        else:
            print(f"Request {i+1}: ⚠️  ERROR ({response.status_code})")
    except Exception as e:
        print(f"Request {i+1}: ❌ FAILED - {str(e)[:100]}")
        break
    
    time.sleep(0.5)  # Small delay between requests

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Successful requests: {success_count}")
print(f"Rate limited at request: {rate_limited_at if rate_limited_at else 'Never'}")

if rate_limited_at and rate_limited_at <= 6:
    print("\n✅ PASS: Rate limiting working correctly (limited within 6 requests)")
elif rate_limited_at:
    print(f"\n⚠️  PARTIAL: Rate limited at request {rate_limited_at} (expected ≤6)")
else:
    print("\n❌ FAIL: No rate limiting detected")

print("="*80 + "\n")

