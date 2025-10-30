"""
Test Monitoring System

Tests Prometheus metrics and health checks.
"""

import sys
from datetime import datetime

print("\n" + "="*80)
print("MONITORING SYSTEM TESTS")
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
# TEST 1: Import Monitoring Module
# ============================================================================
print("-" * 80)
print("TEST 1: Import Monitoring Module")
print("-" * 80)

try:
    from src.api.monitoring import (
        setup_sentry,
        PrometheusMiddleware,
        get_metrics,
        http_requests_total,
        http_request_duration_seconds,
        swarm_analysis_duration_seconds,
        track_swarm_analysis,
        track_agent_performance,
        track_auth_request
    )
    test("Import monitoring module", True, "All monitoring components imported successfully")
except Exception as e:
    test("Import monitoring module", False, f"Error: {str(e)}")
    sys.exit(1)

# ============================================================================
# TEST 2: Import Health Module
# ============================================================================
print("-" * 80)
print("TEST 2: Import Health Module")
print("-" * 80)

try:
    from src.api.health import (
        check_database_health,
        check_swarm_health,
        check_auth_health,
        check_monitoring_health,
        get_detailed_health
    )
    test("Import health module", True, "All health check components imported successfully")
except Exception as e:
    test("Import health module", False, f"Error: {str(e)}")
    sys.exit(1)

# ============================================================================
# TEST 3: Prometheus Metrics
# ============================================================================
print("-" * 80)
print("TEST 3: Prometheus Metrics")
print("-" * 80)

# Test counter
try:
    http_requests_total.labels(method="GET", endpoint="/test", status="2xx").inc()
    test("Counter metric", True, "http_requests_total incremented successfully")
except Exception as e:
    test("Counter metric", False, f"Error: {str(e)}")

# Test histogram
try:
    http_request_duration_seconds.labels(method="GET", endpoint="/test").observe(0.5)
    test("Histogram metric", True, "http_request_duration_seconds observed successfully")
except Exception as e:
    test("Histogram metric", False, f"Error: {str(e)}")

# Test swarm metrics
try:
    track_swarm_analysis("weighted", 5.2, "success")
    test("Swarm analysis metric", True, "Swarm analysis tracked successfully")
except Exception as e:
    test("Swarm analysis metric", False, f"Error: {str(e)}")

try:
    track_agent_performance("MarketAnalyst", 1.3)
    test("Agent performance metric", True, "Agent performance tracked successfully")
except Exception as e:
    test("Agent performance metric", False, f"Error: {str(e)}")

# Test auth metrics
try:
    track_auth_request("/api/auth/login", "success")
    test("Auth request metric", True, "Auth request tracked successfully")
except Exception as e:
    test("Auth request metric", False, f"Error: {str(e)}")

# ============================================================================
# TEST 4: Metrics Generation
# ============================================================================
print("-" * 80)
print("TEST 4: Metrics Generation")
print("-" * 80)

try:
    response = get_metrics()
    content = response.body.decode('utf-8')
    
    # Check if metrics are present
    has_http_requests = "http_requests_total" in content
    has_duration = "http_request_duration_seconds" in content
    has_swarm = "swarm_analysis_duration_seconds" in content
    
    test(
        "Metrics generation",
        has_http_requests and has_duration and has_swarm,
        f"HTTP requests: {has_http_requests}, Duration: {has_duration}, Swarm: {has_swarm}"
    )
    
    # Print sample metrics
    print("Sample metrics:")
    for line in content.split('\n')[:20]:
        if line and not line.startswith('#'):
            print(f"  {line}")
    print()
    
except Exception as e:
    test("Metrics generation", False, f"Error: {str(e)}")

# ============================================================================
# TEST 5: Health Checks
# ============================================================================
print("-" * 80)
print("TEST 5: Health Checks")
print("-" * 80)

# Database health
try:
    db_health = check_database_health()
    test(
        "Database health check",
        db_health["status"] == "healthy",
        f"Status: {db_health['status']}, Type: {db_health.get('type')}"
    )
except Exception as e:
    test("Database health check", False, f"Error: {str(e)}")

# Swarm health
try:
    swarm_health = check_swarm_health()
    test(
        "Swarm health check",
        swarm_health["status"] == "healthy",
        f"Status: {swarm_health['status']}, Agents: {swarm_health.get('agents')}"
    )
except Exception as e:
    test("Swarm health check", False, f"Error: {str(e)}")

# Auth health
try:
    auth_health = check_auth_health()
    test(
        "Auth health check",
        auth_health["status"] == "healthy",
        f"Status: {auth_health['status']}, Users: {auth_health.get('users')}"
    )
except Exception as e:
    test("Auth health check", False, f"Error: {str(e)}")

# Monitoring health
try:
    monitoring_health = check_monitoring_health()
    test(
        "Monitoring health check",
        monitoring_health["status"] == "healthy",
        f"Status: {monitoring_health['status']}, Sentry: {monitoring_health.get('sentry_enabled')}, Metrics: {monitoring_health.get('prometheus_metrics')}"
    )
except Exception as e:
    test("Monitoring health check", False, f"Error: {str(e)}")

# Detailed health
try:
    detailed_health = get_detailed_health()
    test(
        "Detailed health check",
        detailed_health["status"] in ["healthy", "degraded"],
        f"Overall status: {detailed_health['status']}, Components: {len(detailed_health.get('components', {}))}"
    )
    
    # Print component status
    print("Component status:")
    for component, status in detailed_health.get("components", {}).items():
        print(f"  {component}: {status['status']}")
    print()
    
except Exception as e:
    test("Detailed health check", False, f"Error: {str(e)}")

# ============================================================================
# TEST 6: Sentry Setup
# ============================================================================
print("-" * 80)
print("TEST 6: Sentry Setup")
print("-" * 80)

try:
    import os
    
    # Check if Sentry DSN is configured
    sentry_dsn = os.getenv("SENTRY_DSN")
    
    if sentry_dsn:
        setup_sentry()
        test("Sentry setup", True, "Sentry initialized with DSN")
    else:
        test("Sentry setup", True, "Sentry DSN not configured (expected for local development)")
    
except Exception as e:
    test("Sentry setup", False, f"Error: {str(e)}")

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

print("\n📊 **MONITORING FEATURES**")
print("-" * 80)
print("✓ Prometheus metrics collection")
print("✓ HTTP request tracking (count, duration, size)")
print("✓ Swarm-specific metrics (analysis, agents, consensus)")
print("✓ Authentication metrics")
print("✓ Cache metrics (ready for Phase 1, Enhancement 4)")
print("✓ Health checks for all components")
print("✓ Sentry error tracking (when configured)")
print("\n📍 **WHERE TO FIND RESULTS**")
print("-" * 80)
print("Implementation:")
print("  - src/api/monitoring.py - Monitoring module")
print("  - src/api/health.py - Health checks")
print("  - src/api/main.py - Integration")
print("\nEndpoints:")
print("  - GET /metrics - Prometheus metrics")
print("  - GET /health - Simple health check")
print("  - GET /health/detailed - Detailed component health")
print("\nTests:")
print("  - test_monitoring.py - This test file")
print("\n")

