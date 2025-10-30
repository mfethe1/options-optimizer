# Redis Deployment Guide

**Purpose**: Deploy Redis for L2 caching to achieve <500ms API response times.

**Performance Targets**:
- L1 cache hit: <10ms (in-memory TTLCache)
- L2 cache hit: <50ms (Redis)
- Cache hit rate: >80% after warmup
- P50 latency: <100ms (cached)
- P95 latency: <500ms (cached)

---

## Quick Start (Docker Compose)

### 1. Start Redis Container

```bash
# Start Redis only
docker-compose up -d redis

# Verify Redis is running
docker-compose ps
docker-compose logs redis

# Test Redis connection
docker exec -it options_redis redis-cli ping
# Should return: PONG
```

### 2. Configure Environment

Create `.env` file (or copy from `.env.example`):

```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379

# WebSocket Configuration
PHASE4_WS_INTERVAL_SECONDS=30

# API Configuration
LOG_LEVEL=INFO
```

### 3. Start API Server

```bash
# Option A: Local development
uvicorn src.api.main:app --reload --port 8000

# Option B: Docker Compose (full stack)
docker-compose up -d

# Verify API is running
curl http://localhost:8000/
curl http://localhost:8000/api/health
```

### 4. Test Caching Performance

```bash
# Run caching performance tests
python -m pytest tests/test_caching_performance.py -v -s

# Expected output:
# - L1 cache avg latency: <10ms
# - Cache hit rate: >80%
# - P50 latency: <100ms
# - P95 latency: <500ms
```

---

## Manual Redis Installation (Without Docker)

### Windows

```powershell
# Install via Chocolatey
choco install redis-64

# Start Redis
redis-server

# Test connection
redis-cli ping
```

### macOS

```bash
# Install via Homebrew
brew install redis

# Start Redis
brew services start redis

# Test connection
redis-cli ping
```

### Linux (Ubuntu/Debian)

```bash
# Install Redis
sudo apt-get update
sudo apt-get install redis-server

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Test connection
redis-cli ping
```

---

## Redis Configuration

### Memory Management

Redis is configured with:
- **Max Memory**: 512MB (adjust based on your needs)
- **Eviction Policy**: `allkeys-lru` (Least Recently Used)
- **Persistence**: AOF (Append-Only File) enabled

To adjust memory limit:

```bash
# Edit docker-compose.yml
command: redis-server --appendonly yes --maxmemory 1gb --maxmemory-policy allkeys-lru

# Or edit redis.conf
maxmemory 1gb
maxmemory-policy allkeys-lru
```

### TTL Configuration

Cache TTL is set to 15 minutes (900 seconds) in `src/api/investor_report_routes.py`:

```python
L1_CACHE_TTL = 900  # 15 minutes
L2_CACHE_TTL = 900  # 15 minutes
```

To adjust TTL, modify these constants or set environment variables.

---

## Monitoring Redis

### Check Cache Stats

```bash
# Connect to Redis CLI
redis-cli

# Get cache info
INFO stats
INFO memory

# Check number of keys
DBSIZE

# List all keys (use with caution in production)
KEYS ir_v1:*

# Get specific key
GET ir_v1:test_user:AAPL,MSFT

# Check TTL for a key
TTL ir_v1:test_user:AAPL,MSFT
```

### Monitor Cache Hit Rate

```bash
# Watch cache stats in real-time
redis-cli --stat

# Monitor commands
redis-cli MONITOR
```

### Clear Cache

```bash
# Clear all keys
redis-cli FLUSHALL

# Clear specific pattern
redis-cli --scan --pattern "ir_v1:*" | xargs redis-cli DEL
```

---

## Performance Validation

### 1. Measure L1 Cache Performance

```python
import time
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

# First request (cold)
response1 = client.get("/api/investor-report?user_id=test&symbols=AAPL")

# Second request (L1 cache hit)
t0 = time.perf_counter()
response2 = client.get("/api/investor-report?user_id=test&symbols=AAPL")
latency_ms = (time.perf_counter() - t0) * 1000

print(f"L1 Cache Latency: {latency_ms:.2f}ms")
# Expected: <10ms
```

### 2. Measure L2 Cache Performance

```python
# Clear L1 cache (restart server or wait for TTL)
# Then make request (should hit L2)

t0 = time.perf_counter()
response = client.get("/api/investor-report?user_id=test&symbols=AAPL")
latency_ms = (time.perf_counter() - t0) * 1000

print(f"L2 Cache Latency: {latency_ms:.2f}ms")
# Expected: <50ms
```

### 3. Measure Cache Hit Rate

```bash
# Run performance test suite
python -m pytest tests/test_caching_performance.py::TestCachingPerformance::test_cache_hit_rate -v -s

# Expected: >80% hit rate after warmup
```

---

## Troubleshooting

### Redis Connection Failed

**Error**: `ConnectionRefusedError: [Errno 111] Connection refused`

**Solution**:
1. Check Redis is running: `docker-compose ps` or `redis-cli ping`
2. Verify REDIS_URL in `.env`: `REDIS_URL=redis://localhost:6379`
3. Check firewall/network settings

### Graceful Degradation (L1-Only Mode)

If Redis is unavailable, the system automatically falls back to L1-only mode:

```python
# src/api/investor_report_routes.py
async def _get_redis():
    try:
        redis_client = await aioredis.from_url(REDIS_URL)
        await redis_client.ping()
        return redis_client
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}. Using L1-only mode.")
        return False  # Sentinel value
```

### High Memory Usage

**Solution**:
1. Reduce max memory: `maxmemory 256mb`
2. Adjust TTL: Reduce cache TTL to 5 minutes
3. Enable eviction: `maxmemory-policy allkeys-lru`

### Slow Cache Performance

**Symptoms**: L2 cache hits >100ms

**Solutions**:
1. Check Redis memory: `redis-cli INFO memory`
2. Check network latency: `redis-cli --latency`
3. Optimize serialization: Use msgpack instead of JSON
4. Enable Redis pipelining for batch operations

---

## Production Deployment

### Security

1. **Enable Authentication**:
```bash
# docker-compose.yml
command: redis-server --requirepass your_secure_password

# .env
REDIS_URL=redis://:your_secure_password@localhost:6379
```

2. **Network Isolation**:
```yaml
# docker-compose.yml
networks:
  options_network:
    driver: bridge
    internal: true  # Isolate from external network
```

3. **TLS/SSL** (for production):
```bash
REDIS_URL=rediss://username:password@redis-host:6380
```

### High Availability

For production, consider:
- **Redis Sentinel**: Automatic failover
- **Redis Cluster**: Horizontal scaling
- **Redis Enterprise**: Managed service with HA

### Monitoring

Integrate with:
- **Prometheus**: Redis exporter for metrics
- **Grafana**: Dashboards for cache performance
- **Sentry**: Error tracking for cache failures

---

## Where to Find Results

- **Docker Compose**: `docker-compose.yml`
- **Dockerfile**: `Dockerfile`
- **Environment Config**: `.env.example`
- **Performance Tests**: `tests/test_caching_performance.py`
- **API Implementation**: `src/api/investor_report_routes.py`

**Start Redis**: `docker-compose up -d redis`  
**Test Caching**: `python -m pytest tests/test_caching_performance.py -v -s`  
**Monitor**: `redis-cli --stat`

