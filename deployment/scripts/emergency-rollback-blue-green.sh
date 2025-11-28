#!/bin/bash
set -euo pipefail

# Emergency Blue-Green Rollback Script
# Target Recovery Time: 1 minute
# Use this for immediate rollback of blue-green deployment

NAMESPACE="options-production"
ROLLOUT_NAME="options-api-blue-green"
HEALTH_ENDPOINT="http://api.options-prod.com/health/pinn"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Start rollback timer
START_TIME=$(date +%s)

echo "🚨 EMERGENCY BLUE-GREEN ROLLBACK INITIATED"
echo "Timestamp: $(date)"
echo "Target: <60 seconds recovery"
echo "=========================================="

# Step 1: Abort promotion (if in progress)
log "Step 1: Aborting blue-green promotion..."
if kubectl argo rollouts abort "$ROLLOUT_NAME" -n "$NAMESPACE"; then
    success "Blue-green promotion aborted"
else
    warn "Rollout abort failed or not needed"
fi

# Step 2: Ensure traffic is on stable (blue) version
log "Step 2: Ensuring traffic on stable version..."
if kubectl patch service options-api-active -n "$NAMESPACE" \
    -p '{"spec":{"selector":{"app":"options-api","version":"stable"}}}'; then
    success "Traffic routed to stable version"
else
    error "Failed to route traffic to stable version"
fi

# Step 3: Scale down green (problematic) version
log "Step 3: Scaling down green version..."
GREEN_DEPLOYMENTS=$(kubectl get deployments -n "$NAMESPACE" -l version=pinn-fixes -o name 2>/dev/null || echo "")
if [[ -n "$GREEN_DEPLOYMENTS" ]]; then
    for deployment in $GREEN_DEPLOYMENTS; do
        log "Scaling down $deployment..."
        kubectl scale "$deployment" -n "$NAMESPACE" --replicas=0
    done
    success "Green version scaled down"
else
    warn "No green deployments found to scale down"
fi

# Step 4: Verify stable version is running
log "Step 4: Verifying stable version pods..."
STABLE_READY=$(kubectl get pods -n "$NAMESPACE" -l app=options-api,version=stable \
    --field-selector=status.phase=Running -o name | wc -l)
log "Stable pods running: $STABLE_READY"

if [[ $STABLE_READY -gt 0 ]]; then
    success "Stable version pods are running"
else
    error "No stable version pods running - critical issue!"
fi

# Step 5: Health check with retries
log "Step 5: Verifying stable version health..."
HEALTH_CHECK_ATTEMPTS=5
HEALTH_STATUS="unknown"

for attempt in $(seq 1 $HEALTH_CHECK_ATTEMPTS); do
    sleep 2  # Brief wait between attempts
    
    if HEALTH_RESPONSE=$(curl -s --max-time 5 "$HEALTH_ENDPOINT" 2>/dev/null); then
        HEALTH_STATUS=$(echo "$HEALTH_RESPONSE" | jq -r '.overall_status' 2>/dev/null || echo "unknown")
        if [[ "$HEALTH_STATUS" == "healthy" ]]; then
            success "Stable version health check passed"
            break
        fi
    fi
    
    if [[ $attempt -eq $HEALTH_CHECK_ATTEMPTS ]]; then
        warn "Health check failed after $HEALTH_CHECK_ATTEMPTS attempts. Status: $HEALTH_STATUS"
    else
        log "Health check attempt $attempt/$HEALTH_CHECK_ATTEMPTS failed, retrying..."
    fi
done

# Step 6: Verify PINN cache is working
log "Step 6: Checking PINN cache status..."
if CACHE_RESPONSE=$(curl -s --max-time 5 "$HEALTH_ENDPOINT/metrics" 2>/dev/null); then
    CACHE_HIT_RATE=$(echo "$CACHE_RESPONSE" | jq -r '.cache.hit_rate' 2>/dev/null || echo "0")
    log "PINN cache hit rate: $CACHE_HIT_RATE"
    if (( $(echo "$CACHE_HIT_RATE > 0.5" | bc -l) )); then
        success "PINN cache is functioning"
    else
        warn "PINN cache hit rate is low: $CACHE_HIT_RATE"
    fi
else
    warn "Could not check PINN cache status"
fi

# Calculate recovery time
END_TIME=$(date +%s)
RECOVERY_TIME=$((END_TIME - START_TIME))

echo "=========================================="
echo "✅ BLUE-GREEN ROLLBACK COMPLETED"
echo "Recovery time: ${RECOVERY_TIME} seconds"
echo "Target met: $(if [[ $RECOVERY_TIME -lt 60 ]]; then echo "✅ YES"; else echo "❌ NO"; fi)"
echo "Stable version health: $HEALTH_STATUS"
echo "Stable pods running: $STABLE_READY"
echo "Timestamp: $(date)"

# Log rollback event
log "Logging rollback event..."
kubectl annotate rollout "$ROLLOUT_NAME" -n "$NAMESPACE" \
    "rollback.timestamp=$(date -Iseconds)" \
    "rollback.reason=emergency" \
    "rollback.recovery-time=${RECOVERY_TIME}s" \
    "rollback.stable-pods=$STABLE_READY" \
    --overwrite || warn "Failed to log rollback annotation"

success "Emergency blue-green rollback procedure completed"

if [[ $RECOVERY_TIME -gt 60 ]]; then
    warn "Recovery time exceeded 60-second target. Review procedure for optimization."
fi
