#!/bin/bash
set -euo pipefail

# PINN Load Testing Script
# Tests latency improvements from ~1350ms to ~260ms target
# Validates performance under realistic traffic patterns

BASE_URL="${1:-http://localhost:8001}"
TARGET_LATENCY_MS=260
BASELINE_LATENCY_MS=1350
CONCURRENT_USERS="${CONCURRENT_USERS:-50}"
TEST_DURATION="${TEST_DURATION:-300}"  # 5 minutes
RAMP_UP_TIME="${RAMP_UP_TIME:-60}"     # 1 minute

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# Check dependencies
check_dependencies() {
    local missing_deps=()
    
    if ! command -v curl &>/dev/null; then
        missing_deps+=("curl")
    fi
    
    if ! command -v jq &>/dev/null; then
        missing_deps+=("jq")
    fi
    
    if ! command -v ab &>/dev/null && ! command -v wrk &>/dev/null; then
        missing_deps+=("apache2-utils (ab) or wrk")
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        error "Missing dependencies: ${missing_deps[*]}"
    fi
}

# Test basic connectivity
test_connectivity() {
    log "Testing connectivity to $BASE_URL..."
    
    if ! curl -s --max-time 10 "$BASE_URL/health" >/dev/null; then
        error "Cannot connect to $BASE_URL/health"
    fi
    
    if ! curl -s --max-time 10 "$BASE_URL/health/pinn/quick" >/dev/null; then
        error "PINN health endpoint not responding"
    fi
    
    success "Connectivity test passed"
}

# Warmup PINN cache
warmup_cache() {
    log "Warming up PINN cache..."
    
    if curl -s -X POST "$BASE_URL/health/pinn/warmup" | grep -q "success"; then
        success "PINN cache warmed up"
    else
        warn "Cache warmup failed or not available"
    fi
}

# Single request latency test
test_single_request_latency() {
    log "Testing single request latency..."
    
    local total_time=0
    local successful_requests=0
    local failed_requests=0
    
    for i in {1..10}; do
        local start_time=$(date +%s.%3N)
        
        if curl -s --max-time 5 "$BASE_URL/api/pinn/predict?symbol=AAPL&current_price=150" >/dev/null; then
            local end_time=$(date +%s.%3N)
            local request_time=$(echo "($end_time - $start_time) * 1000" | bc -l)
            total_time=$(echo "$total_time + $request_time" | bc -l)
            ((successful_requests++))
            log "Request $i: ${request_time}ms"
        else
            ((failed_requests++))
            warn "Request $i failed"
        fi
    done
    
    if [[ $successful_requests -gt 0 ]]; then
        local avg_latency=$(echo "scale=2; $total_time / $successful_requests" | bc -l)
        log "Average latency: ${avg_latency}ms (target: <${TARGET_LATENCY_MS}ms)"
        
        if (( $(echo "$avg_latency < $TARGET_LATENCY_MS" | bc -l) )); then
            success "Single request latency test PASSED"
            return 0
        else
            warn "Single request latency test FAILED - exceeds target"
            return 1
        fi
    else
        error "All single requests failed"
    fi
}

# Load test with Apache Bench
run_ab_load_test() {
    log "Running Apache Bench load test..."
    log "Concurrent users: $CONCURRENT_USERS"
    log "Test duration: ${TEST_DURATION}s"
    
    local total_requests=$((CONCURRENT_USERS * TEST_DURATION / 5))  # ~5s per request estimate
    local test_url="$BASE_URL/api/pinn/predict?symbol=AAPL&current_price=150"
    
    log "Total requests: $total_requests"
    log "Starting load test..."
    
    if command -v ab &>/dev/null; then
        ab -n "$total_requests" -c "$CONCURRENT_USERS" -g ab_results.tsv "$test_url" > ab_results.txt 2>&1
        
        # Parse results
        local avg_latency=$(grep "Time per request:" ab_results.txt | head -1 | awk '{print $4}')
        local p95_latency=$(grep "95%" ab_results.txt | awk '{print $2}')
        local success_rate=$(grep "Non-2xx responses:" ab_results.txt | awk '{print $3}' || echo "0")
        local total_requests_completed=$(grep "Complete requests:" ab_results.txt | awk '{print $3}')
        
        log "Results:"
        log "  Total requests completed: $total_requests_completed"
        log "  Average latency: ${avg_latency}ms"
        log "  95th percentile: ${p95_latency}ms"
        log "  Failed requests: $success_rate"
        
        # Validate results
        if (( $(echo "$p95_latency < $TARGET_LATENCY_MS" | bc -l) )); then
            success "Load test PASSED - P95 latency within target"
        else
            warn "Load test FAILED - P95 latency exceeds target"
        fi
    else
        warn "Apache Bench not available, skipping AB load test"
    fi
}

# Load test with wrk
run_wrk_load_test() {
    if ! command -v wrk &>/dev/null; then
        warn "wrk not available, skipping wrk load test"
        return 0
    fi
    
    log "Running wrk load test..."
    
    local test_url="$BASE_URL/api/pinn/predict?symbol=AAPL&current_price=150"
    
    wrk -t4 -c"$CONCURRENT_USERS" -d"${TEST_DURATION}s" \
        --latency \
        --script=<(cat <<'EOF'
wrk.method = "GET"
wrk.headers["Content-Type"] = "application/json"

request = function()
    local symbols = {"AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"}
    local prices = {150, 2800, 300, 200, 3200}
    local idx = math.random(1, #symbols)
    local path = "/api/pinn/predict?symbol=" .. symbols[idx] .. "&current_price=" .. prices[idx]
    return wrk.format("GET", path)
end
EOF
        ) "$BASE_URL" > wrk_results.txt 2>&1
    
    # Parse wrk results
    local avg_latency=$(grep "Latency" wrk_results.txt | awk '{print $2}' | sed 's/ms//')
    local p99_latency=$(grep "99%" wrk_results.txt | awk '{print $2}' | sed 's/ms//')
    local requests_per_sec=$(grep "Requests/sec:" wrk_results.txt | awk '{print $2}')
    
    log "wrk Results:"
    log "  Average latency: ${avg_latency}ms"
    log "  99th percentile: ${p99_latency}ms"
    log "  Requests/sec: $requests_per_sec"
    
    cat wrk_results.txt
}

# Stress test - gradual load increase
run_stress_test() {
    log "Running stress test with gradual load increase..."

    local max_users=200
    local step_size=25
    local step_duration=60  # seconds per step

    for users in $(seq $step_size $step_size $max_users); do
        log "Testing with $users concurrent users..."

        local test_url="$BASE_URL/api/pinn/predict?symbol=AAPL&current_price=150"
        local requests_per_step=$((users * step_duration / 3))  # ~3s per request

        if command -v ab &>/dev/null; then
            timeout $step_duration ab -n "$requests_per_step" -c "$users" "$test_url" > "stress_${users}_users.txt" 2>&1 &
            local ab_pid=$!

            # Monitor during test
            sleep 30
            if curl -s "$BASE_URL/health/pinn" | jq -r '.overall_status' | grep -q "unhealthy"; then
                warn "System became unhealthy at $users users - stopping stress test"
                kill $ab_pid 2>/dev/null || true
                break
            fi

            wait $ab_pid

            # Check results
            local avg_latency=$(grep "Time per request:" "stress_${users}_users.txt" | head -1 | awk '{print $4}' || echo "999999")
            log "  $users users: ${avg_latency}ms average latency"

            if (( $(echo "$avg_latency > $(($TARGET_LATENCY_MS * 2))" | bc -l) )); then
                warn "Latency degraded significantly at $users users"
                break
            fi
        fi

        sleep 10  # Brief pause between steps
    done

    success "Stress test completed"
}

# Cache performance test
test_cache_performance() {
    log "Testing PINN cache performance..."

    # Test same request multiple times to measure cache hit improvement
    local test_url="$BASE_URL/api/pinn/predict?symbol=AAPL&current_price=150"
    local cache_miss_time=0
    local cache_hit_time=0

    # First request (cache miss)
    log "Testing cache miss performance..."
    local start_time=$(date +%s.%3N)
    curl -s "$test_url" >/dev/null
    local end_time=$(date +%s.%3N)
    cache_miss_time=$(echo "($end_time - $start_time) * 1000" | bc -l)
    log "Cache miss latency: ${cache_miss_time}ms"

    # Subsequent requests (cache hits)
    log "Testing cache hit performance..."
    local total_hit_time=0
    for i in {1..5}; do
        start_time=$(date +%s.%3N)
        curl -s "$test_url" >/dev/null
        end_time=$(date +%s.%3N)
        local hit_time=$(echo "($end_time - $start_time) * 1000" | bc -l)
        total_hit_time=$(echo "$total_hit_time + $hit_time" | bc -l)
    done
    cache_hit_time=$(echo "scale=2; $total_hit_time / 5" | bc -l)
    log "Average cache hit latency: ${cache_hit_time}ms"

    # Calculate improvement
    local improvement=$(echo "scale=1; ($cache_miss_time - $cache_hit_time) / $cache_miss_time * 100" | bc -l)
    log "Cache performance improvement: ${improvement}%"

    if (( $(echo "$improvement > 50" | bc -l) )); then
        success "Cache performance test PASSED"
    else
        warn "Cache performance test FAILED - insufficient improvement"
    fi
}

# Generate performance report
generate_report() {
    log "Generating performance report..."

    local report_file="pinn_load_test_report_$(date +%Y%m%d_%H%M%S).md"

    cat > "$report_file" <<EOF
# PINN Load Test Report

**Date**: $(date)
**Target URL**: $BASE_URL
**Target Latency**: ${TARGET_LATENCY_MS}ms
**Baseline Latency**: ${BASELINE_LATENCY_MS}ms
**Expected Improvement**: 80.7%

## Test Configuration
- Concurrent Users: $CONCURRENT_USERS
- Test Duration: ${TEST_DURATION}s
- Ramp-up Time: ${RAMP_UP_TIME}s

## Results Summary

### Single Request Test
$(if [[ -f "single_request_results.txt" ]]; then cat single_request_results.txt; else echo "Not available"; fi)

### Load Test Results
$(if [[ -f "ab_results.txt" ]]; then grep -A 10 "Percentage of the requests served within" ab_results.txt; else echo "Not available"; fi)

### Stress Test Results
$(if ls stress_*_users.txt 1> /dev/null 2>&1; then echo "Stress test files generated"; else echo "Not available"; fi)

## Recommendations
- Monitor P95 latency < ${TARGET_LATENCY_MS}ms
- Ensure cache hit rate > 80%
- Watch for memory usage spikes
- Validate fallback mechanisms under load

EOF

    success "Report generated: $report_file"
}

# Main execution
main() {
    log "Starting PINN load testing"
    log "Target URL: $BASE_URL"
    log "Target latency: ${TARGET_LATENCY_MS}ms"

    check_dependencies
    test_connectivity
    warmup_cache

    # Run tests
    test_single_request_latency
    test_cache_performance
    run_ab_load_test
    run_wrk_load_test
    run_stress_test

    generate_report

    success "PINN load testing completed"
}

# Execute main function
main "$@"
