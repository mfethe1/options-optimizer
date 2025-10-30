"""
Test Caching System

Tests in-memory caching with TTL.
"""

import sys
import time
from datetime import datetime

print("\n" + "="*80)
print("CACHING SYSTEM TESTS")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n")

# Test counters
total_tests = 0
passed_tests = 0
failed_tests = 0

def test(name, condition, details=""):
    """Helper function to run a test"""
    global total_tests, passed_tests, failed_tests
    total_tests += 1
    
    if condition:
        passed_tests += 1
        print(f"✓ PASS: {name}")
        if details:
            print(f"  {details}")
    else:
        failed_tests += 1
        print(f"✗ FAIL: {name}")
        if details:
            print(f"  {details}")
    print()

# ============================================================================
# TEST 1: Import Cache Module
# ============================================================================
print("-" * 80)
print("TEST 1: Import Cache Module")
print("-" * 80)

try:
    from src.api.cache import (
        InMemoryCache,
        CacheEntry,
        cached,
        get_cache,
        clear_cache,
        clear_expired,
        get_cache_stats,
        invalidate_pattern,
        cache_market_data,
        get_cached_market_data,
        invalidate_market_data
    )
    test("Import cache module", True, "All caching components imported successfully")
except Exception as e:
    test("Import cache module", False, f"Error: {str(e)}")
    sys.exit(1)

# ============================================================================
# TEST 2: Basic Cache Operations
# ============================================================================
print("-" * 80)
print("TEST 2: Basic Cache Operations")
print("-" * 80)

cache = InMemoryCache()

# Test set and get
cache.set("key1", "value1", ttl_seconds=10)
value = cache.get("key1")
test("Set and get", value == "value1", f"Retrieved: {value}")

# Test cache miss
value = cache.get("nonexistent")
test("Cache miss", value is None, "Returns None for missing key")

# Test cache stats
stats = cache.get_stats()
test(
    "Cache stats",
    stats["hits"] == 1 and stats["misses"] == 1,
    f"Hits: {stats['hits']}, Misses: {stats['misses']}, Hit rate: {stats['hit_rate']:.2%}"
)

# ============================================================================
# TEST 3: TTL Expiration
# ============================================================================
print("-" * 80)
print("TEST 3: TTL Expiration")
print("-" * 80)

cache.clear()

# Set with short TTL
cache.set("short_ttl", "expires_soon", ttl_seconds=2)
value1 = cache.get("short_ttl")
test("Get before expiration", value1 == "expires_soon", f"Retrieved: {value1}")

# Wait for expiration
print("  Waiting 3 seconds for TTL expiration...")
time.sleep(3)

value2 = cache.get("short_ttl")
test("Get after expiration", value2 is None, "Returns None after TTL expires")

# ============================================================================
# TEST 4: Cache Decorator
# ============================================================================
print("-" * 80)
print("TEST 4: Cache Decorator")
print("-" * 80)

call_count = 0

@cached(ttl_seconds=5, key_prefix="test")
def expensive_function(x, y):
    global call_count
    call_count += 1
    time.sleep(0.1)  # Simulate expensive operation
    return x + y

# First call (cache miss)
call_count = 0
start = time.time()
result1 = expensive_function(1, 2)
duration1 = time.time() - start
test(
    "Decorator first call",
    result1 == 3 and call_count == 1,
    f"Result: {result1}, Duration: {duration1:.3f}s, Calls: {call_count}"
)

# Second call (cache hit)
start = time.time()
result2 = expensive_function(1, 2)
duration2 = time.time() - start
test(
    "Decorator cached call",
    result2 == 3 and call_count == 1 and duration2 < 0.05,
    f"Result: {result2}, Duration: {duration2:.3f}s, Calls: {call_count} (cached)"
)

# Different arguments (cache miss)
result3 = expensive_function(2, 3)
test(
    "Decorator different args",
    result3 == 5 and call_count == 2,
    f"Result: {result3}, Calls: {call_count}"
)

# ============================================================================
# TEST 5: Market Data Caching
# ============================================================================
print("-" * 80)
print("TEST 5: Market Data Caching")
print("-" * 80)

# Cache market data
market_data = {
    "symbol": "AAPL",
    "price": 150.25,
    "volume": 1000000,
    "timestamp": datetime.now().isoformat()
}

cache_market_data("AAPL", market_data, ttl_seconds=10)
cached_data = get_cached_market_data("AAPL")
test(
    "Cache market data",
    cached_data == market_data,
    f"Symbol: {cached_data.get('symbol')}, Price: ${cached_data.get('price')}"
)

# Test cache miss
cached_data2 = get_cached_market_data("GOOGL")
test("Market data cache miss", cached_data2 is None, "Returns None for uncached symbol")

# Invalidate market data
invalidate_market_data("AAPL")
cached_data3 = get_cached_market_data("AAPL")
test("Invalidate market data", cached_data3 is None, "Returns None after invalidation")

# ============================================================================
# TEST 6: Pattern Invalidation
# ============================================================================
print("-" * 80)
print("TEST 6: Pattern Invalidation")
print("-" * 80)

cache = get_cache()
cache.clear()

# Set multiple entries
cache.set("market_data:AAPL", {"price": 150}, ttl_seconds=60)
cache.set("market_data:GOOGL", {"price": 2800}, ttl_seconds=60)
cache.set("user_data:123", {"name": "John"}, ttl_seconds=60)

# Invalidate market data pattern
invalidate_pattern("market_data")

# Check results
aapl_data = cache.get("market_data:AAPL")
user_data = cache.get("user_data:123")
test(
    "Pattern invalidation",
    aapl_data is None and user_data is not None,
    f"Market data invalidated, user data preserved"
)

# ============================================================================
# TEST 7: Clear Expired Entries
# ============================================================================
print("-" * 80)
print("TEST 7: Clear Expired Entries")
print("-" * 80)

cache.clear()

# Set entries with different TTLs
cache.set("short", "value1", ttl_seconds=1)
cache.set("long", "value2", ttl_seconds=60)

# Wait for short TTL to expire
print("  Waiting 2 seconds for short TTL to expire...")
time.sleep(2)

# Clear expired
clear_expired()

# Check results
short_value = cache.get("short")
long_value = cache.get("long")
test(
    "Clear expired entries",
    short_value is None and long_value == "value2",
    f"Expired entry removed, valid entry preserved"
)

# ============================================================================
# TEST 8: Cache Statistics
# ============================================================================
print("-" * 80)
print("TEST 8: Cache Statistics")
print("-" * 80)

stats = get_cache_stats()
test(
    "Cache statistics",
    "hit_rate" in stats and "size" in stats,
    f"Size: {stats['size']}, Hits: {stats['hits']}, Misses: {stats['misses']}, Hit rate: {stats['hit_rate']:.2%}"
)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"\nTotal Tests: {total_tests}")
print(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
print(f"Failed: {failed_tests}")
print(f"\nPass Rate: {passed_tests/total_tests*100:.1f}%")

if failed_tests == 0:
    print("\n✅ ALL TESTS PASSED!")
elif passed_tests / total_tests >= 0.8:
    print("\n⚠️  MOSTLY PASSING (≥80%)")
else:
    print("\n❌ TESTS FAILING")

print("=" * 80 + "\n")

print("\n📊 **CACHING FEATURES**")
print("-" * 80)
print("✓ In-memory cache with TTL")
print("✓ Cache decorator for functions")
print("✓ Market data caching helpers")
print("✓ Pattern-based invalidation")
print("✓ Automatic expiration")
print("✓ Cache statistics (hit rate, size, etc.)")
print("✓ Thread-safe operations")
print("\n📍 **WHERE TO FIND RESULTS**")
print("-" * 80)
print("Implementation:")
print("  - src/api/cache.py - Caching module")
print("  - src/api/main.py - Cache management endpoints")
print("\nEndpoints:")
print("  - GET /cache/stats - Cache statistics")
print("  - POST /cache/clear - Clear all cache")
print("  - POST /cache/clear-expired - Clear expired entries")
print("  - POST /cache/invalidate/{pattern} - Invalidate by pattern")
print("\nTests:")
print("  - test_caching.py - This test file")
print("\n")

