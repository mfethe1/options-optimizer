#!/bin/bash
set -euo pipefail

# Emergency Canary Rollback Script
# Target Recovery Time: 30 seconds
# Use this for immediate rollback of canary deployment

NAMESPACE="options-production"
ROLLOUT_NAME="options-api-pinn-fixes"
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

echo "🚨 EMERGENCY CANARY ROLLBACK INITIATED"
echo "Timestamp: $(date)"
echo "Target: <30 seconds recovery"
echo "=========================================="

# Step 1: Abort current rollout (immediate traffic shift back)
log "Step 1: Aborting canary rollout..."
if kubectl argo rollouts abort "$ROLLOUT_NAME" -n "$NAMESPACE"; then
    success "Canary rollout aborted successfully"
else
    error "Failed to abort canary rollout"
fi

# Step 2: Verify traffic is back to stable
log "Step 2: Verifying traffic routing..."
sleep 3

# Check rollout status
ROLLOUT_STATUS=$(kubectl argo rollouts get rollout "$ROLLOUT_NAME" -n "$NAMESPACE" -o json | jq -r '.status.phase')
log "Rollout status: $ROLLOUT_STATUS"

if [[ "$ROLLOUT_STATUS" == "Degraded" || "$ROLLOUT_STATUS" == "Aborted" ]]; then
    success "Traffic successfully routed back to stable version"
else
    warn "Rollout status unclear: $ROLLOUT_STATUS"
fi

# Step 3: Check health of stable version
log "Step 3: Checking stable version health..."
HEALTH_CHECK_ATTEMPTS=3
HEALTH_STATUS="unknown"

for attempt in $(seq 1 $HEALTH_CHECK_ATTEMPTS); do
    if HEALTH_RESPONSE=$(curl -s --max-time 5 "$HEALTH_ENDPOINT" 2>/dev/null); then
        HEALTH_STATUS=$(echo "$HEALTH_RESPONSE" | jq -r '.overall_status' 2>/dev/null || echo "unknown")
        if [[ "$HEALTH_STATUS" == "healthy" ]]; then
            success "Stable version health check passed"
            break
        fi
    fi
    
    if [[ $attempt -eq $HEALTH_CHECK_ATTEMPTS ]]; then
        warn "Health check failed after $HEALTH_CHECK_ATTEMPTS attempts. Status: $HEALTH_STATUS"
        warn "Manual verification recommended"
    else
        log "Health check attempt $attempt failed, retrying..."
        sleep 2
    fi
done

# Step 4: Verify PINN functionality
log "Step 4: Quick PINN functionality test..."
if PINN_RESPONSE=$(curl -s --max-time 5 "$HEALTH_ENDPOINT/quick" 2>/dev/null); then
    PINN_STATUS=$(echo "$PINN_RESPONSE" | jq -r '.pinn_available' 2>/dev/null || echo "false")
    if [[ "$PINN_STATUS" == "true" ]]; then
        success "PINN functionality verified"
    else
        warn "PINN functionality check failed"
    fi
else
    warn "Could not verify PINN functionality"
fi

# Calculate recovery time
END_TIME=$(date +%s)
RECOVERY_TIME=$((END_TIME - START_TIME))

echo "=========================================="
echo "✅ CANARY ROLLBACK COMPLETED"
echo "Recovery time: ${RECOVERY_TIME} seconds"
echo "Target met: $(if [[ $RECOVERY_TIME -lt 30 ]]; then echo "✅ YES"; else echo "❌ NO"; fi)"
echo "Stable version health: $HEALTH_STATUS"
echo "PINN available: $PINN_STATUS"
echo "Timestamp: $(date)"

# Log rollback event
log "Logging rollback event..."
kubectl annotate rollout "$ROLLOUT_NAME" -n "$NAMESPACE" \
    "rollback.timestamp=$(date -Iseconds)" \
    "rollback.reason=emergency" \
    "rollback.recovery-time=${RECOVERY_TIME}s" \
    --overwrite || warn "Failed to log rollback annotation"

# Notify monitoring systems
log "Sending rollback notification..."
curl -X POST -H "Content-Type: application/json" \
    -d "{\"text\":\"🚨 PINN Canary Rollback Completed\\nRecovery Time: ${RECOVERY_TIME}s\\nHealth: $HEALTH_STATUS\"}" \
    "${SLACK_WEBHOOK_URL:-}" 2>/dev/null || warn "Failed to send Slack notification"

success "Emergency canary rollback procedure completed"

if [[ $RECOVERY_TIME -gt 30 ]]; then
    warn "Recovery time exceeded 30-second target. Review procedure for optimization."
fi
