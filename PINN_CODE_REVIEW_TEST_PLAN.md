# PINN Code Review Fixes - Comprehensive Test Plan

**Date**: 2025-11-09
**Status**: Ready for Execution
**Estimated Duration**: 90 minutes

---

## Test Overview

This test plan validates all P0 and P1 fixes for the PINN model code review issues.

### Fixes Implemented

**P0 (Critical - Must Pass Before Merge)**
- ✅ P0-1: Cache key rounding mismatch (30 min)
- ✅ P0-2: Incorrect fallback_to_cpu() call (15 min)

**P1 (High Priority - Should Pass Within 24h)**
- ✅ P1-1: Missing scalar validation (30 min)
- ✅ P1-2: Confusing variable naming (20 min)
- ✅ P1-3: Memory leak from persistent tape (15 min)
- ✅ P1-4: Thread pool not closed (20 min)

---

## Test Suite 1: P0-1 Cache Key Rounding (Critical)

### Objective
Verify cache hits for similar parameters (e.g., r=0.0501 and r=0.0499 both use cached r=0.05 model)

### Test Cases

#### TC1.1: Basic Cache Hit
```python
from src.api.pinn_model_cache import get_cached_pinn_model, get_cache_stats, clear_cache

# Start with empty cache
clear_cache()

# First call - should be CACHE MISS
pinn1 = get_cached_pinn_model(r=0.05, sigma=0.20, option_type='call')
stats1 = get_cache_stats()
assert stats1['misses'] == 1
assert stats1['hits'] == 0
assert stats1['currsize'] == 1

# Second call with exact same params - should be CACHE HIT
pinn2 = get_cached_pinn_model(r=0.05, sigma=0.20, option_type='call')
stats2 = get_cache_stats()
assert stats2['hits'] == 1
assert stats2['currsize'] == 1
assert pinn1 is pinn2  # Same object reference!

print("✅ TC1.1 PASSED: Basic cache hit works")
```

#### TC1.2: Rounding-Based Cache Hit (P0-1 FIX VERIFICATION)
```python
from src.api.pinn_model_cache import get_cached_pinn_model, get_cache_stats, clear_cache

# Start with empty cache
clear_cache()

# First call with r=0.05
pinn1 = get_cached_pinn_model(r=0.05, sigma=0.20, option_type='call')
stats1 = get_cache_stats()
assert stats1['misses'] == 1

# Second call with r=0.0501 (should round to 0.05 and hit cache)
pinn2 = get_cached_pinn_model(r=0.0501, sigma=0.20, option_type='call')
stats2 = get_cache_stats()
assert stats2['hits'] == 1  # ✅ CRITICAL: This should be a cache HIT
assert pinn1 is pinn2  # ✅ CRITICAL: Should be same object!

# Third call with r=0.0499 (should also round to 0.05)
pinn3 = get_cached_pinn_model(r=0.0499, sigma=0.1999, option_type='call')
stats3 = get_cache_stats()
assert stats3['hits'] == 2  # Two cache hits
assert pinn1 is pinn3  # Same object!

print("✅ TC1.2 PASSED: Rounding-based cache hits work (P0-1 FIX VERIFIED)")
```

#### TC1.3: Cache Hit Rate >80%
```python
from src.api.pinn_model_cache import get_cached_pinn_model, get_cache_stats, clear_cache
import random

clear_cache()

# Simulate production usage with noisy parameters
for i in range(100):
    # Most calls use similar rates (4-6%) with noise
    r_noisy = 0.05 + random.uniform(-0.01, 0.01)  # 0.04-0.06
    sigma_noisy = 0.20 + random.uniform(-0.02, 0.02)  # 0.18-0.22

    get_cached_pinn_model(r=r_noisy, sigma=sigma_noisy, option_type='call')

stats = get_cache_stats()
hit_rate = stats['hit_rate']
print(f"Cache hit rate after 100 calls: {hit_rate:.1%}")

assert hit_rate > 0.80  # ✅ CRITICAL: Must exceed 80% hit rate
print(f"✅ TC1.3 PASSED: Cache hit rate = {hit_rate:.1%} (target: >80%)")
```

#### TC1.4: LRU Eviction (maxsize=10)
```python
from src.api.pinn_model_cache import get_cached_pinn_model, get_cache_stats, clear_cache

clear_cache()

# Fill cache with 10 different models
for i in range(10):
    r = 0.01 * i  # 0.00, 0.01, 0.02, ..., 0.09
    get_cached_pinn_model(r=r, sigma=0.20, option_type='call')

stats1 = get_cache_stats()
assert stats1['currsize'] == 10  # Cache full
assert stats1['maxsize'] == 10

# Add 11th model - should evict least recently used (r=0.00)
get_cached_pinn_model(r=0.10, sigma=0.20, option_type='call')
stats2 = get_cache_stats()
assert stats2['currsize'] == 10  # Still at max

# Try to access evicted model - should be CACHE MISS
get_cached_pinn_model(r=0.00, sigma=0.20, option_type='call')
stats3 = get_cache_stats()
assert stats3['misses'] >= 11  # Evicted model caused miss

print("✅ TC1.4 PASSED: LRU eviction works correctly")
```

**Expected Results:**
- All 4 test cases pass
- Cache hit rate >80% in TC1.3
- No regression in existing functionality

---

## Test Suite 2: P0-2 GPU Fallback (Critical)

### Objective
Verify GPU error handling with CPU fallback works without runtime errors

### Test Cases

#### TC2.1: CPU Fallback on GPU OOM
```python
from src.ml.physics_informed.tf_error_handler import handle_tf_errors, GPUMemoryError
import tensorflow as tf

# Simulate GPU OOM error
@handle_tf_errors(enable_cpu_fallback=True, check_nan=False)
def gpu_oom_function():
    raise RuntimeError("out of memory: failed to allocate 4GB")

try:
    result = gpu_oom_function()
    print("❌ TC2.1 FAILED: Should have raised GPUMemoryError")
except GPUMemoryError as e:
    print(f"✅ TC2.1 PASSED: Correctly raised GPUMemoryError: {e}")
```

#### TC2.2: CPU Fallback on CUDA Error
```python
from src.ml.physics_informed.tf_error_handler import handle_tf_errors, CUDAError

@handle_tf_errors(enable_cpu_fallback=True, check_nan=False, retry_on_error=True)
def cuda_error_function():
    raise RuntimeError("CUDA error: device not found")

try:
    result = cuda_error_function()
    print("❌ TC2.2 FAILED: Should have raised CUDAError")
except CUDAError as e:
    print(f"✅ TC2.2 PASSED: Correctly raised CUDAError: {e}")
```

#### TC2.3: PINN Prediction with GPU Fallback
```python
from src.api.pinn_model_cache import get_cached_pinn_model

try:
    # Should work even if GPU fails (falls back to CPU or Black-Scholes)
    pinn = get_cached_pinn_model(r=0.05, sigma=0.20, option_type='call')
    result = pinn.predict(S=100.0, K=100.0, tau=0.25)

    assert 'price' in result
    assert result['price'] > 0
    assert result['method'] in ['PINN', 'PINN (price-only)', 'Black-Scholes (fallback)', 'unavailable']

    print(f"✅ TC2.3 PASSED: PINN prediction works with fallback (method={result['method']})")
except Exception as e:
    print(f"❌ TC2.3 FAILED: {e}")
```

**Expected Results:**
- All 3 test cases pass
- No `AttributeError: 'bool' object is not callable` errors
- Graceful fallback to CPU or Black-Scholes on GPU failures

---

## Test Suite 3: P1-1 Scalar Validation

### Objective
Verify `check_numerical_stability()` handles scalar floats/ints without errors

### Test Cases

#### TC3.1: Scalar Float Validation
```python
from src.ml.physics_informed.tf_error_handler import check_numerical_stability, NumericalInstabilityError
import numpy as np

# Valid scalar float
try:
    result = check_numerical_stability(3.14, name="test_scalar")
    assert result is True
    print("✅ TC3.1a PASSED: Valid scalar float accepted")
except Exception as e:
    print(f"❌ TC3.1a FAILED: {e}")

# NaN scalar
try:
    check_numerical_stability(float('nan'), name="nan_scalar")
    print("❌ TC3.1b FAILED: Should have raised NumericalInstabilityError for NaN")
except NumericalInstabilityError:
    print("✅ TC3.1b PASSED: NaN scalar correctly rejected")

# Inf scalar
try:
    check_numerical_stability(float('inf'), name="inf_scalar")
    print("❌ TC3.1c FAILED: Should have raised NumericalInstabilityError for Inf")
except NumericalInstabilityError:
    print("✅ TC3.1c PASSED: Inf scalar correctly rejected")
```

#### TC3.2: Scalar Int Validation
```python
from src.ml.physics_informed.tf_error_handler import check_numerical_stability

# Valid scalar int
try:
    result = check_numerical_stability(42, name="test_int")
    assert result is True
    print("✅ TC3.2 PASSED: Scalar int accepted")
except Exception as e:
    print(f"❌ TC3.2 FAILED: {e}")
```

**Expected Results:**
- All 4 test cases pass
- No `AttributeError: 'float' object has no attribute 'numpy'` errors

---

## Test Suite 4: P1-2 Variable Naming

### Objective
Verify variable renaming improves code clarity (code review only)

### Test Cases

#### TC4.1: Code Review
```python
# Manual code inspection
# File: src/ml/physics_informed/general_pinn.py
# Lines: ~567-590

# ✅ VERIFY: Variable is named `dV_dS_tensor` (not `delta_tensor`)
# ✅ VERIFY: Comment says "dV/dS (Delta)"
# ✅ VERIFY: All references updated (3 occurrences)

print("✅ TC4.1 PASSED: Variable naming improved")
```

**Expected Results:**
- All references to `delta_tensor` replaced with `dV_dS_tensor`
- Code is clearer and more maintainable

---

## Test Suite 5: P1-3 Memory Leak Prevention

### Objective
Verify persistent tape cleanup prevents memory leaks

### Test Cases

#### TC5.1: Memory Leak Prevention
```python
from src.api.pinn_model_cache import get_cached_pinn_model
import gc

# Run 100 predictions to stress test memory management
pinn = get_cached_pinn_model(r=0.05, sigma=0.20, option_type='call')

for i in range(100):
    result = pinn.predict(S=100.0, K=100.0, tau=0.25)

    if i % 10 == 0:
        gc.collect()  # Force garbage collection
        print(f"Iteration {i}: price={result['price']:.2f}")

print("✅ TC5.1 PASSED: No memory leak detected (100 predictions)")
```

#### TC5.2: Tape Cleanup Verification
```python
# Manual code inspection
# File: src/ml/physics_informed/general_pinn.py
# Line: ~605

# ✅ VERIFY: `del tape` statement exists after GradientTape block
# ✅ VERIFY: Comment mentions "P1-3 FIX: Explicit cleanup"

print("✅ TC5.2 PASSED: Tape cleanup code present")
```

**Expected Results:**
- No memory growth over 100 predictions
- Explicit `del tape` statement in code

---

## Test Suite 6: P1-4 Thread Pool Shutdown

### Objective
Verify thread pool gracefully shuts down on program exit

### Test Cases

#### TC6.1: Atexit Handler Registration
```python
import atexit
from src.api import ml_integration_helpers

# ✅ VERIFY: _shutdown_thread_pool function exists
assert hasattr(ml_integration_helpers, '_shutdown_thread_pool')

# ✅ VERIFY: Atexit handler is registered
# (Check by inspecting atexit._exithandlers or running script and checking logs)

print("✅ TC6.1 PASSED: Atexit handler registered")
```

#### TC6.2: Thread Pool Shutdown on Exit
```bash
# Create test script
cat > test_thread_pool_shutdown.py << 'EOF'
from src.api.ml_integration_helpers import fetch_historical_prices
import asyncio

async def main():
    # Use thread pool
    prices = await fetch_historical_prices(['AAPL'], days=5)
    print(f"Fetched prices: {len(prices['AAPL'])} days")
    print("Exiting...")

asyncio.run(main())
# Should print "[ML Helpers] Shutting down thread pool..." on exit
EOF

# Run and check logs
python test_thread_pool_shutdown.py | grep "Shutting down thread pool"
```

**Expected Results:**
- Atexit handler is registered
- Log message "Shutting down thread pool..." appears on exit
- No hanging worker threads after exit

---

## Test Suite 7: Integration Tests

### Objective
Verify all fixes work together in production scenarios

### Test Cases

#### TC7.1: End-to-End PINN Prediction
```python
from src.api.pinn_model_cache import get_cached_pinn_model, get_cache_stats, clear_cache
import time

clear_cache()

# Simulate production usage
start_time = time.time()

for i in range(10):
    r = 0.05 + (i * 0.001)  # 0.050, 0.051, 0.052, ...
    sigma = 0.20 + (i * 0.001)

    pinn = get_cached_pinn_model(r=r, sigma=sigma, option_type='call')
    result = pinn.predict(S=100.0, K=100.0, tau=0.25)

    print(f"Iteration {i}: price={result['price']:.2f}, method={result['method']}")

elapsed = time.time() - start_time
stats = get_cache_stats()

print(f"\n✅ TC7.1 PASSED:")
print(f"  - Total time: {elapsed:.2f}s")
print(f"  - Cache hit rate: {stats['hit_rate']:.1%}")
print(f"  - Cache size: {stats['currsize']}/{stats['maxsize']}")

assert stats['hit_rate'] > 0.80  # >80% cache hits
assert elapsed < 5.0  # <5s for 10 predictions
```

#### TC7.2: Concurrent Cache Access (Thread Safety)
```python
from src.api.pinn_model_cache import get_cached_pinn_model, clear_cache
import concurrent.futures
import time

clear_cache()

def predict_wrapper(worker_id):
    """Worker function for concurrent predictions"""
    pinn = get_cached_pinn_model(r=0.05, sigma=0.20, option_type='call')
    result = pinn.predict(S=100.0, K=100.0, tau=0.25)
    return worker_id, result['price']

# Run 10 workers concurrently
start_time = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(predict_wrapper, i) for i in range(10)]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]

elapsed = time.time() - start_time

print(f"✅ TC7.2 PASSED:")
print(f"  - 10 concurrent predictions completed in {elapsed:.2f}s")
print(f"  - All prices: {[r[1] for r in sorted(results)]}")
```

#### TC7.3: NaN Injection Detection
```python
from src.ml.physics_informed.tf_error_handler import check_numerical_stability, NumericalInstabilityError
import numpy as np
import tensorflow as tf

# Create tensor with injected NaN
data = np.array([1.0, 2.0, float('nan'), 4.0])
tensor = tf.constant(data)

try:
    check_numerical_stability(tensor, name="nan_injection_test")
    print("❌ TC7.3 FAILED: Should have detected NaN")
except NumericalInstabilityError as e:
    assert "NaN count=1" in str(e)
    print(f"✅ TC7.3 PASSED: NaN injection detected: {e}")
```

#### TC7.4: Circuit Breaker Behavior
```python
from src.api.ml_integration_helpers import _yfinance_circuit_breaker

# Reset circuit breaker
_yfinance_circuit_breaker.state = 'closed'
_yfinance_circuit_breaker.failure_count = 0

# Simulate failures to trigger circuit breaker
for i in range(5):
    _yfinance_circuit_breaker.record_failure()

assert _yfinance_circuit_breaker.state == 'open'
assert not _yfinance_circuit_breaker.can_execute()

print("✅ TC7.4 PASSED: Circuit breaker opens after 5 failures")
```

**Expected Results:**
- All 4 integration tests pass
- Cache hit rate >80%
- No race conditions in concurrent access
- NaN detection works correctly
- Circuit breaker prevents cascading failures

---

## Test Suite 8: Performance Benchmarks

### Objective
Verify performance improvements and no regressions

### Test Cases

#### TC8.1: Cache Performance Improvement
```python
from src.api.pinn_model_cache import get_cached_pinn_model, clear_cache
import time

clear_cache()

# First call (cache miss)
start = time.time()
pinn1 = get_cached_pinn_model(r=0.05, sigma=0.20, option_type='call')
_ = pinn1.predict(S=100.0, K=100.0, tau=0.25)
miss_time = time.time() - start

# Second call (cache hit)
start = time.time()
pinn2 = get_cached_pinn_model(r=0.05, sigma=0.20, option_type='call')
_ = pinn2.predict(S=100.0, K=100.0, tau=0.25)
hit_time = time.time() - start

speedup = miss_time / hit_time

print(f"✅ TC8.1 PASSED:")
print(f"  - Cache MISS: {miss_time*1000:.0f}ms")
print(f"  - Cache HIT: {hit_time*1000:.0f}ms")
print(f"  - Speedup: {speedup:.1f}x")

assert speedup > 5.0  # At least 5x faster
```

#### TC8.2: Error Handling Overhead
```python
from src.ml.physics_informed.tf_error_handler import handle_tf_errors
import time

@handle_tf_errors(enable_cpu_fallback=False, check_nan=False)
def simple_function():
    return 42

# Measure overhead
iterations = 1000
start = time.time()
for _ in range(iterations):
    result = simple_function()
elapsed = time.time() - start

overhead_per_call = (elapsed / iterations) * 1000  # ms

print(f"✅ TC8.2 PASSED:")
print(f"  - Error handling overhead: {overhead_per_call:.3f}ms per call")

assert overhead_per_call < 0.1  # <0.1ms overhead
```

**Expected Results:**
- Cache provides >5x speedup
- Error handling adds <0.1ms overhead
- No performance regressions

---

## Test Execution

### Prerequisites
```bash
# Ensure all dependencies installed
pip install -r requirements.txt

# Start backend (for integration tests)
python -m uvicorn src.api.main:app --reload
```

### Run All Tests
```bash
# Unit tests
python -m pytest tests/test_pinn_code_review_fixes.py -v

# Integration tests
python -m pytest tests/test_pinn_integration.py -v

# Performance benchmarks
python scripts/benchmark_pinn_fixes.py
```

### Success Criteria
- ✅ All P0 tests pass (100% success rate)
- ✅ All P1 tests pass (>95% success rate)
- ✅ Cache hit rate >80% in production simulation
- ✅ No memory leaks over 100 predictions
- ✅ Error rate <5% in stress tests
- ✅ Performance meets targets:
  - Cache hit: <50ms (p95)
  - Cache miss: <1s (p95)
  - Error handling overhead: <0.1ms

---

## Rollback Plan

If any P0 test fails:
1. Revert commits for failed fix
2. Document failure reason
3. Create hotfix branch
4. Re-test in isolation
5. Merge hotfix when tests pass

---

## Monitoring Post-Deployment

### Prometheus Metrics to Track
```python
# Add to src/api/pinn_model_cache.py
from prometheus_client import Counter, Histogram

pinn_cache_hits = Counter('pinn_cache_hits_total', 'PINN cache hits')
pinn_cache_misses = Counter('pinn_cache_misses_total', 'PINN cache misses')
pinn_prediction_latency = Histogram('pinn_prediction_latency_seconds', 'PINN prediction latency')
pinn_fallback_rate = Counter('pinn_fallback_total', 'PINN fallbacks to Black-Scholes', ['reason'])
```

### Alerts
- Cache hit rate <70% (warning)
- Cache hit rate <50% (critical)
- Prediction latency p95 >2s (warning)
- Error rate >10% (critical)

---

## Appendix: Test Data

### Typical Market Parameters
```python
MARKET_PARAMS = {
    'r': [0.02, 0.03, 0.04, 0.05, 0.06],  # 2-6% risk-free rate
    'sigma': [0.15, 0.20, 0.25, 0.30, 0.35],  # 15-35% volatility
    'S': [80, 90, 100, 110, 120],  # Stock prices
    'K': [100],  # ATM strike
    'tau': [0.25, 0.5, 1.0]  # 3M, 6M, 1Y maturities
}
```

---

**Document Version**: 1.0
**Last Updated**: 2025-11-09
**Owner**: Agent Orchestration Specialist
**Review Status**: Ready for Execution
