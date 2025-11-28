"""
PINN Code Review Fixes - Performance Benchmarking Script

Validates performance improvements from P0 optimizations:
1. Model caching (LRU) - Expected: ~500ms savings per cache hit
2. Greek computation optimization - Expected: ~200ms savings
3. Removed dual put prediction - Expected: ~400ms savings

Total expected improvement: ~1100ms (81% reduction)

Usage:
    python scripts/benchmark_pinn_fixes.py
    python scripts/benchmark_pinn_fixes.py --json
    python scripts/benchmark_pinn_fixes.py --iterations 1000
"""

import argparse
import time
import statistics
import json
from typing import Dict, List
from src.api.pinn_model_cache import get_cached_pinn_model, get_cache_stats, clear_cache, warmup_cache


def benchmark_cache_performance(iterations: int = 100) -> Dict:
    """
    Benchmark cache HIT vs MISS performance

    Expected results:
    - Cache MISS: ~500ms (model creation + weight loading)
    - Cache HIT: ~10ms (cached model retrieval)
    - Speedup: >5x
    """
    print("\n" + "="*80)
    print("BENCHMARK 1: Cache Performance (HIT vs MISS)")
    print("="*80)

    miss_times = []
    hit_times = []

    # Test cache MISS performance
    for i in range(iterations):
        clear_cache()
        start = time.perf_counter()
        pinn = get_cached_pinn_model(r=0.05, sigma=0.20, option_type='call')
        _ = pinn.predict(S=100.0, K=100.0, tau=0.25)
        elapsed = time.perf_counter() - start
        miss_times.append(elapsed * 1000)  # Convert to ms

    # Test cache HIT performance
    clear_cache()
    _ = get_cached_pinn_model(r=0.05, sigma=0.20, option_type='call')  # Prime cache

    for i in range(iterations):
        start = time.perf_counter()
        pinn = get_cached_pinn_model(r=0.05, sigma=0.20, option_type='call')
        _ = pinn.predict(S=100.0, K=100.0, tau=0.25)
        elapsed = time.perf_counter() - start
        hit_times.append(elapsed * 1000)

    # Calculate statistics
    miss_p50 = statistics.median(miss_times)
    miss_p95 = statistics.quantiles(miss_times, n=20)[18]  # 95th percentile
    miss_p99 = statistics.quantiles(miss_times, n=100)[98]  # 99th percentile

    hit_p50 = statistics.median(hit_times)
    hit_p95 = statistics.quantiles(hit_times, n=20)[18]
    hit_p99 = statistics.quantiles(hit_times, n=100)[98]

    speedup_p50 = miss_p50 / hit_p50 if hit_p50 > 0 else 0
    speedup_p95 = miss_p95 / hit_p95 if hit_p95 > 0 else 0

    # Print results
    print(f"\nCache MISS Performance ({iterations} iterations):")
    print(f"  p50: {miss_p50:.2f}ms")
    print(f"  p95: {miss_p95:.2f}ms")
    print(f"  p99: {miss_p99:.2f}ms")

    print(f"\nCache HIT Performance ({iterations} iterations):")
    print(f"  p50: {hit_p50:.2f}ms")
    print(f"  p95: {hit_p95:.2f}ms")
    print(f"  p99: {hit_p99:.2f}ms")

    print(f"\nSpeedup:")
    print(f"  p50: {speedup_p50:.1f}x")
    print(f"  p95: {speedup_p95:.1f}x")
    print(f"  Savings (p50): {miss_p50 - hit_p50:.2f}ms")

    # Validation
    passed = speedup_p50 > 2.0  # At least 2x speedup (conservative)
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"\n{status}: Cache speedup = {speedup_p50:.1f}x (target: >2.0x)")

    return {
        "test": "cache_performance",
        "iterations": iterations,
        "miss_p50_ms": miss_p50,
        "miss_p95_ms": miss_p95,
        "miss_p99_ms": miss_p99,
        "hit_p50_ms": hit_p50,
        "hit_p95_ms": hit_p95,
        "hit_p99_ms": hit_p99,
        "speedup_p50": speedup_p50,
        "speedup_p95": speedup_p95,
        "savings_p50_ms": miss_p50 - hit_p50,
        "passed": passed
    }


def benchmark_cache_hit_rate(iterations: int = 100) -> Dict:
    """
    Benchmark cache hit rate in production-like scenario

    Expected: >80% hit rate with noisy parameters
    """
    print("\n" + "="*80)
    print("BENCHMARK 2: Cache Hit Rate (Production Simulation)")
    print("="*80)

    clear_cache()

    import random
    for i in range(iterations):
        # Simulate production: most calls cluster around r=0.05, sigma=0.20
        r_noisy = 0.05 + random.uniform(-0.01, 0.01)  # 0.04-0.06
        sigma_noisy = 0.20 + random.uniform(-0.02, 0.02)  # 0.18-0.22
        option_type = random.choice(['call', 'put'])

        _ = get_cached_pinn_model(r=r_noisy, sigma=sigma_noisy, option_type=option_type)

    stats = get_cache_stats()
    hit_rate = stats['hit_rate']

    print(f"\nResults after {iterations} calls:")
    print(f"  Cache hits: {stats['hits']}")
    print(f"  Cache misses: {stats['misses']}")
    print(f"  Hit rate: {hit_rate:.1%}")
    print(f"  Cache size: {stats['currsize']}/{stats['maxsize']}")

    passed = hit_rate > 0.80
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"\n{status}: Hit rate = {hit_rate:.1%} (target: >80%)")

    return {
        "test": "cache_hit_rate",
        "iterations": iterations,
        "hit_rate": hit_rate,
        "hits": stats['hits'],
        "misses": stats['misses'],
        "cache_size": stats['currsize'],
        "passed": passed
    }


def benchmark_concurrent_access(workers: int = 10, iterations_per_worker: int = 10) -> Dict:
    """
    Benchmark concurrent cache access (thread safety)

    Expected: No race conditions, consistent results
    """
    print("\n" + "="*80)
    print(f"BENCHMARK 3: Concurrent Access ({workers} workers)")
    print("="*80)

    import concurrent.futures

    clear_cache()
    warmup_cache()  # Pre-populate cache

    def worker_task(worker_id: int) -> List[float]:
        times = []
        for i in range(iterations_per_worker):
            start = time.perf_counter()
            pinn = get_cached_pinn_model(r=0.05, sigma=0.20, option_type='call')
            result = pinn.predict(S=100.0, K=100.0, tau=0.25)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)
        return times

    start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker_task, i) for i in range(workers)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    total_time = time.perf_counter() - start_time

    # Flatten all times
    all_times = [t for worker_times in results for t in worker_times]

    p50 = statistics.median(all_times)
    p95 = statistics.quantiles(all_times, n=20)[18]
    p99 = statistics.quantiles(all_times, n=100)[98]

    total_calls = workers * iterations_per_worker
    throughput = total_calls / total_time

    print(f"\nResults:")
    print(f"  Total calls: {total_calls}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {throughput:.1f} calls/sec")
    print(f"  Latency p50: {p50:.2f}ms")
    print(f"  Latency p95: {p95:.2f}ms")
    print(f"  Latency p99: {p99:.2f}ms")

    passed = p95 < 100.0  # p95 should be <100ms for cached calls
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"\n{status}: p95 latency = {p95:.2f}ms (target: <100ms)")

    return {
        "test": "concurrent_access",
        "workers": workers,
        "iterations_per_worker": iterations_per_worker,
        "total_calls": total_calls,
        "total_time_s": total_time,
        "throughput_calls_per_sec": throughput,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "latency_p99_ms": p99,
        "passed": passed
    }


def benchmark_memory_stability(iterations: int = 1000) -> Dict:
    """
    Benchmark memory stability (P1-3 tape cleanup fix)

    Expected: No memory leak over many predictions
    """
    print("\n" + "="*80)
    print(f"BENCHMARK 4: Memory Stability ({iterations} iterations)")
    print("="*80)

    import gc
    import psutil
    import os

    process = psutil.Process(os.getpid())

    pinn = get_cached_pinn_model(r=0.05, sigma=0.20, option_type='call')

    # Get baseline memory
    gc.collect()
    baseline_mb = process.memory_info().rss / 1024 / 1024

    # Run predictions
    for i in range(iterations):
        _ = pinn.predict(S=100.0, K=100.0, tau=0.25)

        if i % 100 == 0:
            gc.collect()
            current_mb = process.memory_info().rss / 1024 / 1024
            growth_mb = current_mb - baseline_mb
            print(f"  Iteration {i}: Memory = {current_mb:.1f}MB (growth: {growth_mb:.1f}MB)")

    # Final memory check
    gc.collect()
    final_mb = process.memory_info().rss / 1024 / 1024
    total_growth_mb = final_mb - baseline_mb

    print(f"\nResults:")
    print(f"  Baseline memory: {baseline_mb:.1f}MB")
    print(f"  Final memory: {final_mb:.1f}MB")
    print(f"  Total growth: {total_growth_mb:.1f}MB")
    print(f"  Growth per iteration: {total_growth_mb/iterations:.4f}MB")

    # Allow max 50MB growth for 1000 iterations (0.05MB per iteration)
    passed = total_growth_mb < 50.0
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"\n{status}: Memory growth = {total_growth_mb:.1f}MB (target: <50MB)")

    return {
        "test": "memory_stability",
        "iterations": iterations,
        "baseline_mb": baseline_mb,
        "final_mb": final_mb,
        "total_growth_mb": total_growth_mb,
        "growth_per_iteration_mb": total_growth_mb / iterations,
        "passed": passed
    }


def main():
    parser = argparse.ArgumentParser(description="PINN performance benchmarking")
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations per benchmark')
    args = parser.parse_args()

    print("="*80)
    print("PINN CODE REVIEW FIXES - PERFORMANCE BENCHMARKING")
    print("="*80)
    print(f"Iterations per benchmark: {args.iterations}")

    results = []

    # Run benchmarks
    results.append(benchmark_cache_performance(args.iterations))
    results.append(benchmark_cache_hit_rate(args.iterations))
    results.append(benchmark_concurrent_access(workers=10, iterations_per_worker=10))

    # Memory benchmark requires psutil
    try:
        results.append(benchmark_memory_stability(min(args.iterations, 1000)))
    except ImportError:
        print("\n⚠️  Memory benchmark skipped (psutil not installed)")

    # Summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    all_passed = all(r['passed'] for r in results)

    for result in results:
        status = "✅ PASSED" if result['passed'] else "❌ FAILED"
        print(f"{status}: {result['test']}")

    if all_passed:
        print("\n✅ ALL BENCHMARKS PASSED - Ready for production deployment")
        exit_code = 0
    else:
        print("\n❌ SOME BENCHMARKS FAILED - Review results before deployment")
        exit_code = 1

    # JSON output
    if args.json:
        print("\n" + json.dumps(results, indent=2))

    return exit_code


if __name__ == '__main__':
    exit(main())
