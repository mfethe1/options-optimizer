# PINN Fixes Rollback Runbook

**Target Recovery Time: <5 minutes**

## Quick Reference

### Emergency Rollback Commands

```bash
# IMMEDIATE ROLLBACK (Use in emergency)
kubectl argo rollouts abort options-api-pinn-fixes -n options-production
kubectl argo rollouts undo options-api-pinn-fixes -n options-production

# OR for Blue-Green
kubectl argo rollouts abort options-api-blue-green -n options-production
kubectl argo rollouts undo options-api-blue-green -n options-production
```

### Health Check Commands

```bash
# Check PINN health
curl -s http://api.options-prod.com/health/pinn | jq '.overall_status'

# Check deployment status
kubectl argo rollouts get rollout options-api-pinn-fixes -n options-production

# Check error rates
kubectl logs -l app=options-api -n options-production --tail=100 | grep ERROR
```

## Rollback Triggers

### Automatic Rollback Conditions

1. **Latency Degradation**: P95 latency > 260ms for 3 consecutive measurements
2. **Error Rate Spike**: Error rate > 5% for 2 minutes
3. **PINN Fallback Rate**: Fallback rate > 10% for 5 minutes
4. **Memory Issues**: Memory usage > 1.8GB consistently
5. **Health Check Failures**: Health endpoint failures > 3 in 1 minute

### Manual Rollback Indicators

- Customer complaints about slow responses
- Unusual PINN prediction errors in logs
- Cache hit rate drops below 70%
- TensorFlow errors in application logs
- Prometheus alerts firing

## Rollback Procedures

### 1. Canary Rollback (Fastest - 30 seconds)

```bash
#!/bin/bash
# File: deployment/scripts/emergency-rollback-canary.sh

set -euo pipefail

NAMESPACE="options-production"
ROLLOUT_NAME="options-api-pinn-fixes"

echo "🚨 EMERGENCY CANARY ROLLBACK INITIATED"
echo "Timestamp: $(date)"

# Step 1: Abort current rollout (immediate traffic shift back)
echo "Step 1: Aborting canary rollout..."
kubectl argo rollouts abort "$ROLLOUT_NAME" -n "$NAMESPACE"

# Step 2: Verify traffic is back to stable
echo "Step 2: Verifying traffic routing..."
sleep 5
kubectl argo rollouts get rollout "$ROLLOUT_NAME" -n "$NAMESPACE"

# Step 3: Check health of stable version
echo "Step 3: Checking stable version health..."
STABLE_HEALTH=$(curl -s http://api.options-prod.com/health/pinn | jq -r '.overall_status')
if [[ "$STABLE_HEALTH" != "healthy" ]]; then
    echo "❌ WARNING: Stable version health check failed: $STABLE_HEALTH"
    exit 1
fi

echo "✅ CANARY ROLLBACK COMPLETED SUCCESSFULLY"
echo "All traffic routed back to stable version"
echo "Recovery time: ~30 seconds"
```

### 2. Blue-Green Rollback (Fast - 1 minute)

```bash
#!/bin/bash
# File: deployment/scripts/emergency-rollback-blue-green.sh

set -euo pipefail

NAMESPACE="options-production"
ROLLOUT_NAME="options-api-blue-green"

echo "🚨 EMERGENCY BLUE-GREEN ROLLBACK INITIATED"
echo "Timestamp: $(date)"

# Step 1: Abort promotion (if in progress)
echo "Step 1: Aborting blue-green promotion..."
kubectl argo rollouts abort "$ROLLOUT_NAME" -n "$NAMESPACE"

# Step 2: Ensure traffic is on stable (blue) version
echo "Step 2: Ensuring traffic on stable version..."
kubectl patch service options-api-active -n "$NAMESPACE" -p '{"spec":{"selector":{"app":"options-api","version":"stable"}}}'

# Step 3: Scale down green (problematic) version
echo "Step 3: Scaling down green version..."
kubectl scale deployment options-api-green -n "$NAMESPACE" --replicas=0

# Step 4: Verify stable version health
echo "Step 4: Verifying stable version health..."
sleep 10
STABLE_HEALTH=$(curl -s http://api.options-prod.com/health/pinn | jq -r '.overall_status')
if [[ "$STABLE_HEALTH" != "healthy" ]]; then
    echo "❌ WARNING: Stable version health check failed: $STABLE_HEALTH"
    exit 1
fi

echo "✅ BLUE-GREEN ROLLBACK COMPLETED SUCCESSFULLY"
echo "All traffic on stable (blue) version"
echo "Recovery time: ~1 minute"
```

### 3. Database/Cache Rollback (2-3 minutes)

```bash
#!/bin/bash
# File: deployment/scripts/rollback-with-cache-reset.sh

set -euo pipefail

NAMESPACE="options-production"

echo "🔄 ROLLBACK WITH CACHE RESET INITIATED"
echo "Timestamp: $(date)"

# Step 1: Rollback application
echo "Step 1: Rolling back application..."
kubectl argo rollouts undo options-api-pinn-fixes -n "$NAMESPACE"

# Step 2: Clear PINN model cache
echo "Step 2: Clearing PINN model cache..."
kubectl exec -n "$NAMESPACE" deployment/redis -- redis-cli FLUSHDB

# Step 3: Restart all API pods to clear in-memory cache
echo "Step 3: Restarting API pods..."
kubectl rollout restart deployment/options-api-stable -n "$NAMESPACE"
kubectl rollout status deployment/options-api-stable -n "$NAMESPACE" --timeout=120s

# Step 4: Warmup cache with stable version
echo "Step 4: Warming up cache..."
sleep 30
curl -X POST http://api.options-prod.com/health/pinn/warmup

echo "✅ ROLLBACK WITH CACHE RESET COMPLETED"
echo "Recovery time: ~3 minutes"
```

### 4. Complete Infrastructure Rollback (4-5 minutes)

```bash
#!/bin/bash
# File: deployment/scripts/complete-rollback.sh

set -euo pipefail

NAMESPACE="options-production"
BACKUP_VERSION="stable-v0.9.0"

echo "🚨 COMPLETE INFRASTRUCTURE ROLLBACK INITIATED"
echo "Timestamp: $(date)"
echo "Rolling back to: $BACKUP_VERSION"

# Step 1: Scale down all new versions
echo "Step 1: Scaling down all new deployments..."
kubectl scale deployment options-api-pinn-fixes -n "$NAMESPACE" --replicas=0 || true
kubectl scale deployment options-api-canary -n "$NAMESPACE" --replicas=0 || true
kubectl scale deployment options-api-green -n "$NAMESPACE" --replicas=0 || true

# Step 2: Deploy known good version
echo "Step 2: Deploying known good version..."
kubectl set image deployment/options-api-stable -n "$NAMESPACE" \
    options-api="options-optimizer:$BACKUP_VERSION"

# Step 3: Wait for rollout
echo "Step 3: Waiting for stable deployment..."
kubectl rollout status deployment/options-api-stable -n "$NAMESPACE" --timeout=180s

# Step 4: Reset all services to stable
echo "Step 4: Resetting service routing..."
kubectl patch service options-api-active -n "$NAMESPACE" \
    -p '{"spec":{"selector":{"app":"options-api","version":"stable"}}}'

# Step 5: Clear all caches
echo "Step 5: Clearing all caches..."
kubectl exec -n "$NAMESPACE" deployment/redis -- redis-cli FLUSHALL

# Step 6: Health check
echo "Step 6: Final health check..."
sleep 30
HEALTH=$(curl -s http://api.options-prod.com/health | jq -r '.status')
if [[ "$HEALTH" != "healthy" ]]; then
    echo "❌ CRITICAL: Health check failed after complete rollback"
    exit 1
fi

echo "✅ COMPLETE ROLLBACK SUCCESSFUL"
echo "System restored to: $BACKUP_VERSION"
echo "Recovery time: ~5 minutes"
```

## Rollback Decision Matrix

| Issue Severity | Rollback Type | Recovery Time | Command |
|---------------|---------------|---------------|---------|
| **P0 - Critical** | Emergency Canary | 30 seconds | `./emergency-rollback-canary.sh` |
| **P1 - High** | Blue-Green Abort | 1 minute | `./emergency-rollback-blue-green.sh` |
| **P2 - Medium** | Cache Reset | 3 minutes | `./rollback-with-cache-reset.sh` |
| **P3 - Low** | Complete Rollback | 5 minutes | `./complete-rollback.sh` |

## Post-Rollback Actions

### Immediate (0-5 minutes)
1. ✅ Verify system health: `curl http://api.options-prod.com/health`
2. ✅ Check error rates in logs
3. ✅ Notify stakeholders via Slack/email
4. ✅ Update incident tracking system

### Short-term (5-30 minutes)
1. 📊 Analyze rollback metrics
2. 🔍 Root cause analysis of failure
3. 📝 Document lessons learned
4. 🧪 Plan fix validation in staging

### Long-term (30+ minutes)
1. 🔧 Implement fixes based on root cause
2. 🧪 Re-test in staging environment
3. 📋 Update deployment procedures
4. 🎯 Schedule next deployment attempt
