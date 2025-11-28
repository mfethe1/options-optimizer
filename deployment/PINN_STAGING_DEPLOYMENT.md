# PINN Code Review Fixes - Staging Deployment Guide

**Status**: Ready for Execution
**Date**: 2025-11-10
**Target Environment**: Staging → Production
**Deployment Strategy**: Blue-Green with gradual rollout

---

## Pre-Deployment Checklist

### 1. Test Validation ✅
- [ ] All P0 tests pass (100% success rate)
- [ ] Cache hit rate >80% validated
- [ ] Performance benchmarks meet targets
- [ ] Memory leak tests pass (1000 iterations)
- [ ] Concurrent access tests pass

### 2. Code Review ✅
- [ ] All fixes peer-reviewed
- [ ] Documentation updated (CLAUDE.md, README.md)
- [ ] Prometheus metrics validated
- [ ] Error handling tested

### 3. Infrastructure Readiness
- [ ] Staging environment provisioned
- [ ] Prometheus/Grafana configured
- [ ] Load balancer configured
- [ ] Rollback plan reviewed

---

## Staging Environment Setup

### Docker Configuration

**Dockerfile** (already exists):
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Deploy

```bash
# Build staging image
docker build -t pinn-service:staging-v1.1.0 .

# Run staging container
docker run -d \
  --name pinn-staging \
  -p 8001:8000 \
  -e ENVIRONMENT=staging \
  -e TF_CPP_MIN_LOG_LEVEL=2 \
  -v $(pwd)/models:/app/models:ro \
  pinn-service:staging-v1.1.0

# Verify health
curl http://localhost:8001/health/detailed
```

### Kubernetes Configuration (Production)

**deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pinn-service
  namespace: production
  labels:
    app: pinn-service
    version: v1.1.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: pinn-service
  template:
    metadata:
      labels:
        app: pinn-service
        version: v1.1.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: pinn-service
        image: gcr.io/your-project/pinn-service:v1.1.0
        ports:
        - containerPort: 8000
          protocol: TCP
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: TF_CPP_MIN_LOG_LEVEL
          value: "2"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/detailed
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
        volumeMounts:
        - name: pinn-models
          mountPath: /app/models
          readOnly: true
      volumes:
      - name: pinn-models
        persistentVolumeClaim:
          claimName: pinn-models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: pinn-service
  namespace: production
spec:
  type: LoadBalancer
  selector:
    app: pinn-service
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

---

## Health Check Endpoints

### Basic Health Check
```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy"}
```

### Detailed Health Check (with PINN validation)
```bash
curl http://localhost:8000/health/detailed

# Expected:
{
  "status": "healthy",
  "timestamp": "2025-11-10T12:00:00Z",
  "components": {
    "pinn_cache": {
      "status": "healthy",
      "cache_size": 5,
      "hit_rate": 0.87
    },
    "pinn_model": {
      "status": "healthy",
      "prediction_test": "passed",
      "latency_ms": 42
    },
    "tensorflow": {
      "status": "healthy",
      "version": "2.16.1",
      "gpu_available": false
    }
  }
}
```

### Custom PINN Health Check Script

```python
# scripts/health_check_pinn.py
import requests
import sys

def check_pinn_health(base_url="http://localhost:8000"):
    try:
        # 1. Basic health
        resp = requests.get(f"{base_url}/health", timeout=5)
        assert resp.status_code == 200, f"Basic health failed: {resp.status_code}"

        # 2. Detailed health
        resp = requests.get(f"{base_url}/health/detailed", timeout=10)
        assert resp.status_code == 200, f"Detailed health failed: {resp.status_code}"
        data = resp.json()

        # 3. Validate PINN components
        assert data['components']['pinn_cache']['status'] == 'healthy'
        assert data['components']['pinn_model']['status'] == 'healthy'
        assert data['components']['pinn_cache']['hit_rate'] > 0.5  # At least 50% hit rate

        # 4. Test prediction endpoint
        test_payload = {
            "symbols": ["SPY"],
            "user_id": "health-check"
        }
        resp = requests.get(f"{base_url}/api/investor-report", params=test_payload, timeout=30)
        assert resp.status_code == 200, f"Prediction test failed: {resp.status_code}"

        print("✅ PINN Health Check: PASSED")
        return 0

    except Exception as e:
        print(f"❌ PINN Health Check: FAILED - {e}")
        return 1

if __name__ == '__main__':
    sys.exit(check_pinn_health())
```

---

## Load Testing Strategy

### Apache Bench (Quick Test)
```bash
# Test cache hit rate under load
ab -n 1000 -c 10 -p test_payload.json -T application/json \
  http://localhost:8000/api/investor-report?symbols=SPY

# Expected:
# - Requests per second: >50 RPS
# - Mean latency: <200ms (cache hits)
# - 99th percentile: <1000ms
```

### Locust Load Test (Production Simulation)

**locustfile.py**:
```python
from locust import HttpUser, task, between

class PINNUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def get_investor_report(self):
        """Simulate investor report requests (most common)"""
        self.client.get("/api/investor-report?symbols=SPY,QQQ&user_id=load-test")

    @task(1)
    def get_forecast_all(self):
        """Simulate unified forecast requests"""
        self.client.get("/api/forecast/all?symbols=AAPL")

    @task(1)
    def health_check(self):
        """Background health checks"""
        self.client.get("/health/detailed")
```

**Run load test**:
```bash
# Staging test: 10 users, ramp up over 30s
locust -f locustfile.py --host=http://localhost:8001 \
  --users 10 --spawn-rate 1 --run-time 5m --headless

# Production test: 100 users, ramp up over 2 minutes
locust -f locustfile.py --host=http://production-lb-url \
  --users 100 --spawn-rate 1 --run-time 30m --headless
```

**Success Criteria**:
- 95th percentile latency <500ms
- Error rate <1%
- Cache hit rate >80%
- No memory leaks (memory stable over 30 minutes)

---

## Gradual Rollout Strategy

### Phase 1: Canary Deployment (10% traffic)

```bash
# Deploy canary pod
kubectl apply -f deployment-canary.yaml

# Route 10% traffic to canary
kubectl apply -f service-canary-10pct.yaml

# Monitor for 30 minutes
kubectl logs -f deployment/pinn-service-canary --tail=100

# Check metrics
curl http://prometheus:9090/api/v1/query?query=pinn_cache_hit_rate
```

**Validation Checklist** (30 minutes):
- [ ] Error rate <1%
- [ ] Cache hit rate >80%
- [ ] p95 latency <500ms
- [ ] No memory growth
- [ ] No fallback rate increase

### Phase 2: Expand to 50% Traffic

```bash
# Update traffic split
kubectl apply -f service-canary-50pct.yaml

# Monitor for 1 hour
```

**Validation Checklist** (1 hour):
- [ ] Same success criteria as Phase 1
- [ ] No user complaints
- [ ] Grafana dashboards show green

### Phase 3: Full Rollout (100% traffic)

```bash
# Promote canary to production
kubectl set image deployment/pinn-service \
  pinn-service=gcr.io/your-project/pinn-service:v1.1.0

# Delete canary deployment
kubectl delete deployment pinn-service-canary
```

**Validation Checklist** (24 hours):
- [ ] All metrics stable
- [ ] No increase in support tickets
- [ ] Cache statistics healthy

---

## Rollback Procedure

### Emergency Rollback (<5 minutes)

```bash
# Option 1: Kubernetes rollback
kubectl rollout undo deployment/pinn-service

# Option 2: Revert image tag
kubectl set image deployment/pinn-service \
  pinn-service=gcr.io/your-project/pinn-service:v1.0.0

# Option 3: Scale down new pods, scale up old
kubectl scale deployment/pinn-service-old --replicas=3
kubectl scale deployment/pinn-service --replicas=0
```

### Rollback Triggers
- Error rate >5% for >5 minutes
- p95 latency >2s for >10 minutes
- Cache hit rate <50% for >15 minutes
- Memory leak detected (>100MB growth in 1 hour)
- Critical user-facing bug reported

### Post-Rollback Actions
1. Capture metrics snapshot
2. Save logs for analysis
3. Create incident report
4. Schedule postmortem
5. Fix root cause before retry

---

## Monitoring Dashboard

### Grafana Dashboard Panels

**Panel 1: Cache Performance**
```promql
# Cache hit rate (%)
100 * rate(pinn_cache_hits_total[5m]) / (rate(pinn_cache_hits_total[5m]) + rate(pinn_cache_misses_total[5m]))

# Cache size
pinn_cache_size
```

**Panel 2: Prediction Latency**
```promql
# p50, p95, p99 latency by method
histogram_quantile(0.50, sum(rate(pinn_prediction_latency_seconds_bucket[5m])) by (le, method))
histogram_quantile(0.95, sum(rate(pinn_prediction_latency_seconds_bucket[5m])) by (le, method))
histogram_quantile(0.99, sum(rate(pinn_prediction_latency_seconds_bucket[5m])) by (le, method))
```

**Panel 3: Error Rates**
```promql
# Fallback rate (%)
100 * rate(pinn_fallback_total[5m]) / rate(pinn_prediction_latency_seconds_count[5m])

# Error rate (%)
100 * rate(pinn_prediction_errors_total[5m]) / rate(pinn_prediction_latency_seconds_count[5m])
```

**Panel 4: Memory Usage**
```promql
# Container memory usage
container_memory_usage_bytes{pod=~"pinn-service.*"}

# Memory growth rate
rate(container_memory_usage_bytes{pod=~"pinn-service.*"}[1h])
```

---

## Alerting Rules

**Prometheus Alerts** (prometheus-alerts.yaml):
```yaml
groups:
- name: pinn_service
  interval: 30s
  rules:
  - alert: PINNCacheHitRateLow
    expr: |
      100 * rate(pinn_cache_hits_total[5m]) / (rate(pinn_cache_hits_total[5m]) + rate(pinn_cache_misses_total[5m])) < 70
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "PINN cache hit rate below 70%"
      description: "Hit rate: {{ $value | humanizePercentage }}"

  - alert: PINNLatencyHigh
    expr: |
      histogram_quantile(0.95, sum(rate(pinn_prediction_latency_seconds_bucket[5m])) by (le)) > 2.0
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "PINN p95 latency exceeds 2 seconds"
      description: "p95 latency: {{ $value }}s"

  - alert: PINNErrorRateHigh
    expr: |
      100 * rate(pinn_prediction_errors_total[5m]) / rate(pinn_prediction_latency_seconds_count[5m]) > 10
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "PINN error rate exceeds 10%"
      description: "Error rate: {{ $value | humanizePercentage }}"

  - alert: PINNMemoryLeak
    expr: |
      rate(container_memory_usage_bytes{pod=~"pinn-service.*"}[1h]) > 10485760  # 10MB/hour
    for: 1h
    labels:
      severity: critical
    annotations:
      summary: "PINN service memory leak detected"
      description: "Memory growth: {{ $value | humanize }}B/hour"

  - alert: PINNFallbackRateHigh
    expr: |
      100 * rate(pinn_fallback_total[5m]) / rate(pinn_prediction_latency_seconds_count[5m]) > 20
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "PINN fallback rate exceeds 20%"
      description: "Fallback rate: {{ $value | humanizePercentage }}"
```

---

## Post-Deployment Validation

### 24-Hour Checklist

**Hour 1** (Immediate):
- [ ] All health checks passing
- [ ] Cache hit rate >80%
- [ ] No error spikes
- [ ] Grafana dashboards green

**Hour 4** (Short-term):
- [ ] Memory usage stable
- [ ] Cache statistics healthy
- [ ] Latency within targets
- [ ] No user complaints

**Hour 24** (Long-term):
- [ ] Cost metrics reviewed
- [ ] Performance targets met
- [ ] Support tickets normal
- [ ] Team debrief completed

### Performance Targets

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Cache hit rate | >80% | ___ | ⏳ |
| p50 latency | <200ms | ___ | ⏳ |
| p95 latency | <500ms | ___ | ⏳ |
| p99 latency | <1000ms | ___ | ⏳ |
| Error rate | <1% | ___ | ⏳ |
| Fallback rate | <10% | ___ | ⏳ |
| Memory growth | <10MB/hour | ___ | ⏳ |

---

## Troubleshooting Guide

### Issue: Low Cache Hit Rate (<50%)

**Symptoms**: Cache hit rate below expectations

**Diagnosis**:
```bash
# Check cache stats
curl http://localhost:8000/cache/stats

# Check parameter distribution
grep "PINN Cache" logs/app.log | awk '{print $5, $6}' | sort | uniq -c
```

**Solution**:
1. Warmup cache on startup
2. Increase cache size (maxsize=10 → 20)
3. Review parameter rounding precision

### Issue: High Fallback Rate (>20%)

**Symptoms**: Many predictions using Black-Scholes fallback

**Diagnosis**:
```bash
# Check fallback reasons
curl http://prometheus:9090/api/v1/query?query=pinn_fallback_total

# Check error logs
kubectl logs deployment/pinn-service | grep "fallback"
```

**Solution**:
1. Validate model weights: `python scripts/validate_pinn_weights.py`
2. Check TensorFlow GPU availability
3. Review recent training quality

### Issue: Memory Leak

**Symptoms**: Memory growing >10MB/hour

**Diagnosis**:
```bash
# Monitor memory over time
watch -n 60 'kubectl top pod -l app=pinn-service'

# Check for tape cleanup
grep "P1-3 FIX" src/ml/physics_informed/general_pinn.py
```

**Solution**:
1. Verify tape cleanup (line 605 in general_pinn.py)
2. Check thread pool shutdown
3. Force garbage collection intervals

---

**Document Version**: 1.0
**Last Updated**: 2025-11-10
**Owner**: DevOps & SRE Team
**Status**: Ready for Execution
