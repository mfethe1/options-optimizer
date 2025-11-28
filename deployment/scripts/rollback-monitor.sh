#!/bin/bash
set -euo pipefail

# Automated Rollback Monitor
# Continuously monitors PINN deployment health and triggers automatic rollback

NAMESPACE="options-production"
HEALTH_ENDPOINT="http://api.options-prod.com/health/pinn"
PROMETHEUS_ENDPOINT="http://prometheus:9090"
CHECK_INTERVAL=30  # seconds
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Thresholds for automatic rollback
LATENCY_THRESHOLD_MS=260
ERROR_RATE_THRESHOLD=0.05  # 5%
FALLBACK_RATE_THRESHOLD=0.10  # 10%
MEMORY_THRESHOLD_GB=1.8
CONSECUTIVE_FAILURES_THRESHOLD=3

# Counters
consecutive_failures=0
rollback_triggered=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Check if deployment is in progress
is_deployment_active() {
    local canary_status=$(kubectl argo rollouts get rollout options-api-pinn-fixes -n "$NAMESPACE" -o json 2>/dev/null | jq -r '.status.phase' 2>/dev/null || echo "NotFound")
    local bg_status=$(kubectl argo rollouts get rollout options-api-blue-green -n "$NAMESPACE" -o json 2>/dev/null | jq -r '.status.phase' 2>/dev/null || echo "NotFound")
    
    if [[ "$canary_status" == "Progressing" || "$bg_status" == "Progressing" ]]; then
        return 0  # Active deployment
    else
        return 1  # No active deployment
    fi
}

# Get current deployment strategy
get_deployment_strategy() {
    if kubectl get rollout options-api-pinn-fixes -n "$NAMESPACE" &>/dev/null; then
        echo "canary"
    elif kubectl get rollout options-api-blue-green -n "$NAMESPACE" &>/dev/null; then
        echo "blue-green"
    else
        echo "unknown"
    fi
}

# Check PINN health metrics
check_pinn_health() {
    local health_status="unknown"
    local latency_ms=0
    local error_rate=0
    local fallback_rate=0
    local memory_gb=0
    
    # Get health status
    if HEALTH_RESPONSE=$(curl -s --max-time 10 "$HEALTH_ENDPOINT" 2>/dev/null); then
        health_status=$(echo "$HEALTH_RESPONSE" | jq -r '.overall_status' 2>/dev/null || echo "unknown")
        latency_ms=$(echo "$HEALTH_RESPONSE" | jq -r '.details.prediction_latency_ms // 0' 2>/dev/null || echo "0")
    fi
    
    # Get Prometheus metrics (if available)
    if command -v curl &>/dev/null && curl -s --max-time 5 "$PROMETHEUS_ENDPOINT/api/v1/query?query=pinn_prediction_latency_seconds" &>/dev/null; then
        # Get P95 latency from Prometheus
        local prom_latency=$(curl -s "$PROMETHEUS_ENDPOINT/api/v1/query?query=histogram_quantile(0.95,sum(rate(pinn_prediction_latency_seconds_bucket[5m]))by(le))*1000" | jq -r '.data.result[0].value[1] // "0"' 2>/dev/null || echo "0")
        if (( $(echo "$prom_latency > 0" | bc -l 2>/dev/null || echo "0") )); then
            latency_ms=$prom_latency
        fi
        
        # Get error rate from Prometheus
        error_rate=$(curl -s "$PROMETHEUS_ENDPOINT/api/v1/query?query=sum(rate(pinn_prediction_errors_total[5m]))/sum(rate(http_requests_total[5m]))" | jq -r '.data.result[0].value[1] // "0"' 2>/dev/null || echo "0")
        
        # Get fallback rate from Prometheus
        fallback_rate=$(curl -s "$PROMETHEUS_ENDPOINT/api/v1/query?query=sum(rate(pinn_fallback_total[5m]))/sum(rate(http_requests_total[5m]))" | jq -r '.data.result[0].value[1] // "0"' 2>/dev/null || echo "0")
    fi
    
    # Return metrics as JSON
    cat <<EOF
{
    "health_status": "$health_status",
    "latency_ms": $latency_ms,
    "error_rate": $error_rate,
    "fallback_rate": $fallback_rate,
    "memory_gb": $memory_gb
}
EOF
}

# Evaluate if rollback is needed
should_rollback() {
    local metrics="$1"
    local health_status=$(echo "$metrics" | jq -r '.health_status')
    local latency_ms=$(echo "$metrics" | jq -r '.latency_ms')
    local error_rate=$(echo "$metrics" | jq -r '.error_rate')
    local fallback_rate=$(echo "$metrics" | jq -r '.fallback_rate')
    
    # Check health status
    if [[ "$health_status" == "unhealthy" ]]; then
        log "Rollback trigger: Health status is unhealthy"
        return 0
    fi
    
    # Check latency threshold
    if (( $(echo "$latency_ms > $LATENCY_THRESHOLD_MS" | bc -l 2>/dev/null || echo "0") )); then
        log "Rollback trigger: Latency ${latency_ms}ms exceeds threshold ${LATENCY_THRESHOLD_MS}ms"
        return 0
    fi
    
    # Check error rate threshold
    if (( $(echo "$error_rate > $ERROR_RATE_THRESHOLD" | bc -l 2>/dev/null || echo "0") )); then
        log "Rollback trigger: Error rate ${error_rate} exceeds threshold ${ERROR_RATE_THRESHOLD}"
        return 0
    fi
    
    # Check fallback rate threshold
    if (( $(echo "$fallback_rate > $FALLBACK_RATE_THRESHOLD" | bc -l 2>/dev/null || echo "0") )); then
        log "Rollback trigger: Fallback rate ${fallback_rate} exceeds threshold ${FALLBACK_RATE_THRESHOLD}"
        return 0
    fi
    
    return 1  # No rollback needed
}

# Execute rollback
execute_rollback() {
    local strategy="$1"
    
    log "🚨 AUTOMATIC ROLLBACK TRIGGERED"
    log "Strategy: $strategy"
    log "Reason: Health check failures exceeded threshold"
    
    case "$strategy" in
        "canary")
            log "Executing canary rollback..."
            if "$SCRIPT_DIR/emergency-rollback-canary.sh"; then
                success "Canary rollback completed successfully"
                return 0
            else
                error "Canary rollback failed"
                return 1
            fi
            ;;
        "blue-green")
            log "Executing blue-green rollback..."
            if "$SCRIPT_DIR/emergency-rollback-blue-green.sh"; then
                success "Blue-green rollback completed successfully"
                return 0
            else
                error "Blue-green rollback failed"
                return 1
            fi
            ;;
        *)
            error "Unknown deployment strategy: $strategy"
            return 1
            ;;
    esac
}

# Main monitoring loop
main() {
    log "Starting PINN deployment monitor"
    log "Check interval: ${CHECK_INTERVAL}s"
    log "Latency threshold: ${LATENCY_THRESHOLD_MS}ms"
    log "Error rate threshold: ${ERROR_RATE_THRESHOLD}"
    log "Fallback rate threshold: ${FALLBACK_RATE_THRESHOLD}"

    while true; do
        if ! is_deployment_active; then
            log "No active deployment detected, monitoring paused"
            sleep $CHECK_INTERVAL
            continue
        fi

        local strategy=$(get_deployment_strategy)
        log "Monitoring active deployment (strategy: $strategy)"

        # Check PINN health
        local metrics=$(check_pinn_health)
        local health_status=$(echo "$metrics" | jq -r '.health_status')
        local latency_ms=$(echo "$metrics" | jq -r '.latency_ms')
        local error_rate=$(echo "$metrics" | jq -r '.error_rate')
        local fallback_rate=$(echo "$metrics" | jq -r '.fallback_rate')

        log "Health: $health_status, Latency: ${latency_ms}ms, Error Rate: $error_rate, Fallback Rate: $fallback_rate"

        if should_rollback "$metrics"; then
            ((consecutive_failures++))
            warn "Health check failure $consecutive_failures/$CONSECUTIVE_FAILURES_THRESHOLD"

            if [[ $consecutive_failures -ge $CONSECUTIVE_FAILURES_THRESHOLD ]]; then
                if ! $rollback_triggered; then
                    rollback_triggered=true
                    if execute_rollback "$strategy"; then
                        success "Automatic rollback completed successfully"
                        break
                    else
                        error "Automatic rollback failed - manual intervention required"
                        break
                    fi
                fi
            fi
        else
            if [[ $consecutive_failures -gt 0 ]]; then
                log "Health check passed, resetting failure counter"
                consecutive_failures=0
            fi
            success "Health check passed"
        fi

        sleep $CHECK_INTERVAL
    done
}

# Handle script termination
cleanup() {
    log "Rollback monitor shutting down"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Check dependencies
if ! command -v kubectl &>/dev/null; then
    error "kubectl not found in PATH"
fi

if ! command -v jq &>/dev/null; then
    error "jq not found in PATH"
fi

if ! command -v bc &>/dev/null; then
    warn "bc not found - some numeric comparisons may fail"
fi

# Start monitoring
main "$@"
