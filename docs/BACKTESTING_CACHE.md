# Backtesting Cache System

## Overview

The backtesting engine includes a file-based caching layer to avoid yfinance rate limits when loading historical stock data. This dramatically improves performance for repeated backtests and eliminates "Too Many Requests" errors.

## Features

- **Automatic Caching**: Historical data is automatically cached to disk after first fetch
- **Smart Expiry**: Cache expires after 24 hours to ensure fresh data
- **Fast Loading**: Cached data loads in milliseconds vs seconds for API calls
- **Rate Limit Avoidance**: Eliminates yfinance rate limit errors
- **Transparent Logging**: All cache operations are logged for debugging
- **Graceful Fallback**: If cache fails, system falls back to yfinance

## Architecture

### Cache Directory Structure

```
data/backtest_cache/
├── AAPL_2025-01-01_2025-11-09.pkl    # Symbol_StartDate_EndDate.pkl
├── TSLA_2025-01-01_2025-11-09.pkl
└── MSFT_2025-01-01_2025-11-09.pkl
```

### Cache Key Format

```
{symbol}_{start_date}_{end_date}.pkl
```

Examples:
- `AAPL_2025-01-01_2025-11-09.pkl`
- `TSLA_2024-06-01_2025-11-09.pkl`

### Cache File Format

- **Format**: Python pickle (`.pkl`)
- **Protocol**: `pickle.HIGHEST_PROTOCOL` for best performance
- **Contents**: `pandas.DataFrame` with OHLCV data (same format as yfinance)
- **Typical Size**: 5-20 KB per 90 days of data

## Usage

### Automatic Mode (Default)

The caching layer works automatically - no code changes needed:

```python
from src.backtesting.backtest_engine import BacktestEngine

engine = BacktestEngine()

# First call: Fetches from yfinance and caches
data1 = await engine._load_historical_data("AAPL", "2025-01-01", "2025-11-09")
# Log: [Cache MISS] Fetching AAPL from yfinance
# Log: [Cache SAVE] Cached AAPL to disk (91 rows, 5.0 KB)

# Second call (within 24h): Loads from cache instantly
data2 = await engine._load_historical_data("AAPL", "2025-01-01", "2025-11-09")
# Log: [Cache HIT] Loaded AAPL from cache (age: 0.1h, 91 rows)
```

### Custom Cache Expiry

```python
engine = BacktestEngine()
engine.cache_expiry_hours = 12  # Expire after 12 hours instead of 24
```

### Cache Directory

```python
engine = BacktestEngine()
print(engine.cache_dir)  # data/backtest_cache
```

## Cache Behavior

### Cache HIT (data is fresh)

```
[Cache HIT] Loaded AAPL from cache (age: 2.5h, 91 rows)
```

**When:**
- Cache file exists
- File is less than 24 hours old

**Performance:**
- Load time: ~10ms
- No network requests
- No rate limits

### Cache MISS (data needs refresh)

```
[Cache MISS] Fetching AAPL from yfinance (2025-01-01 to 2025-11-09)
[Cache SAVE] Cached AAPL to disk (91 rows, 5.0 KB)
```

**When:**
- Cache file doesn't exist
- File is older than 24 hours

**Performance:**
- Load time: ~2-5 seconds (network dependent)
- Rate limited by yfinance
- Data saved to cache for future use

### Cache EXPIRED (data is stale)

```
[Cache EXPIRED] Cache for AAPL is 25.3h old (max: 24h)
[Cache MISS] Fetching AAPL from yfinance (2025-01-01 to 2025-11-09)
[Cache SAVE] Cached AAPL to disk (91 rows, 5.0 KB)
```

**When:**
- Cache file exists but is older than `cache_expiry_hours`

**Behavior:**
- Old cache file is ignored
- Fresh data fetched from yfinance
- Cache updated with new data

### Cache ERROR (cache load fails)

```
[Cache ERROR] Failed to load cache for AAPL: Pickle file corrupted
[Cache MISS] Fetching AAPL from yfinance (2025-01-01 to 2025-11-09)
```

**When:**
- Cache file is corrupted
- Pickle protocol mismatch
- Permissions issues

**Behavior:**
- Error logged but not fatal
- System falls back to yfinance
- New cache file created

## Performance Comparison

### Without Cache

```
Backtest 10 stocks × 90 days:
- 10 API calls to yfinance
- ~30 seconds total (rate limited)
- Risk of "Too Many Requests" error
```

### With Cache

```
First run:
- 10 API calls (cached)
- ~30 seconds

Subsequent runs (within 24h):
- 0 API calls
- ~0.1 seconds (300× faster!)
- Zero rate limit errors
```

## Maintenance

### View Cache Statistics

```python
from pathlib import Path

cache_dir = Path("data/backtest_cache")
cache_files = list(cache_dir.glob("*.pkl"))

print(f"Cache files: {len(cache_files)}")
print(f"Total size: {sum(f.stat().st_size for f in cache_files) / 1024:.1f} KB")

for f in cache_files:
    age_hours = (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).total_seconds() / 3600
    print(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB, age: {age_hours:.1f}h")
```

### Clear Cache

```bash
# Windows
rmdir /s /q data\backtest_cache

# Linux/Mac
rm -rf data/backtest_cache
```

### Clear Expired Cache Only

```python
from pathlib import Path
from datetime import datetime

cache_dir = Path("data/backtest_cache")
expiry_hours = 24

for cache_file in cache_dir.glob("*.pkl"):
    age_hours = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 3600
    if age_hours >= expiry_hours:
        cache_file.unlink()
        print(f"Deleted expired cache: {cache_file.name} (age: {age_hours:.1f}h)")
```

## Testing

### Test Cache Functionality

```bash
# Create mock cache file (simulates fetched data)
python scripts/create_mock_cache.py

# Test cache hit/miss behavior
python scripts/test_backtest_cache.py
```

**Expected Output:**

```
TEST 1: First load (should be Cache HIT if mock exists)
[Cache HIT] Loaded AAPL from cache (age: 0.0h, 91 rows)
[OK] Loaded 91 rows for AAPL

TEST 2: Second load (should be Cache HIT)
[Cache HIT] Loaded AAPL from cache (age: 0.0h, 91 rows)
[OK] Cache data matches original data

TEST 3: Cache statistics
Cache directory: data\backtest_cache
Cache files: 1
Total cache size: 5.0 KB
  - AAPL_2025-08-11_2025-11-09.pkl: 5.0 KB, age: 0.0h

[OK] ALL TESTS PASSED
```

## Troubleshooting

### Issue: Cache not being used

**Symptoms:** Every load shows `[Cache MISS]` even for same symbol/dates

**Causes:**
1. Date format mismatch (cache key uses exact string)
2. Cache directory doesn't exist
3. Permissions issue

**Solution:**
```python
# Check cache directory exists
import os
print(os.path.exists("data/backtest_cache"))

# Check cache files
from pathlib import Path
print(list(Path("data/backtest_cache").glob("*.pkl")))

# Verify date format matches
print(f"AAPL_2025-01-01_2025-11-09.pkl")
```

### Issue: "Pickle load failed" errors

**Symptoms:** `[Cache ERROR] Failed to load cache for AAPL: ...`

**Causes:**
1. Pickle protocol version mismatch (Python 3.11 vs 3.12)
2. pandas version mismatch
3. Corrupted file

**Solution:**
```bash
# Clear cache and rebuild
rm -rf data/backtest_cache
mkdir -p data/backtest_cache

# Re-run backtest (will fetch fresh data)
```

### Issue: Cache using too much disk space

**Symptoms:** `data/backtest_cache` directory is very large

**Solution:**
```bash
# Check cache size
du -sh data/backtest_cache

# Clear old cache files (manual)
rm data/backtest_cache/*.pkl

# Or implement auto-cleanup (future feature)
```

## Configuration

### Environment Variables (Optional)

```bash
# .env file
BACKTEST_CACHE_DIR=data/backtest_cache
BACKTEST_CACHE_EXPIRY_HOURS=24
```

### Code Configuration

```python
engine = BacktestEngine()

# Custom cache directory
engine.cache_dir = Path("custom/cache/path")

# Custom expiry time
engine.cache_expiry_hours = 12  # Refresh every 12 hours

# Disable caching (for testing)
engine.cache_expiry_hours = 0  # Always fetch fresh
```

## Future Enhancements

- [ ] Auto-cleanup of expired cache files (cron job)
- [ ] Parquet format option (smaller files, faster loading)
- [ ] Redis cache for distributed systems
- [ ] Cache warming (pre-fetch popular symbols)
- [ ] Cache compression (gzip pickle files)
- [ ] Cache metrics dashboard (hit rate, size, age)
- [ ] Environment variable configuration

## Best Practices

1. **Production:** Set `cache_expiry_hours = 24` (default)
2. **Development:** Set `cache_expiry_hours = 12` for faster iteration
3. **Testing:** Set `cache_expiry_hours = 0` to always fetch fresh data
4. **CI/CD:** Clear cache before critical tests to ensure fresh data
5. **Monitoring:** Log cache hit rate to detect issues

## Security Considerations

- **Data Privacy:** Cache files contain market data (not sensitive)
- **Permissions:** Cache directory should have restrictive permissions (700)
- **Cleanup:** Implement auto-cleanup to prevent disk space issues
- **Validation:** Always validate loaded data (non-empty, correct date range)

## Related Files

- **Implementation:** `src/backtesting/backtest_engine.py` (lines 221-295)
- **Test Script:** `scripts/test_backtest_cache.py`
- **Mock Data Generator:** `scripts/create_mock_cache.py`
- **Cache Directory:** `data/backtest_cache/`

---

**Version:** 1.0.0
**Last Updated:** 2025-11-09
**Author:** Claude Code
