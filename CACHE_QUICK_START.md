# Backtesting Cache - Quick Start Guide

## TL;DR

The backtesting engine now includes automatic file-based caching to avoid yfinance rate limits. **No code changes needed** - it just works!

## Quick Start

### 1. Run Your Backtest (First Time)
```python
from src.backtesting.backtest_engine import BacktestEngine

engine = BacktestEngine()
result = await engine.backtest_model(
    model_name="gnn",
    symbol="AAPL",
    start_date="2025-01-01",
    end_date="2025-11-09",
    prediction_function=get_gnn_prediction
)
```

**Output:**
```
[Cache MISS] Fetching AAPL from yfinance
[Cache SAVE] Cached AAPL to disk (91 rows, 5.0 KB)
```

**Time:** ~3 seconds (network call to yfinance)

### 2. Run Again (Within 24 Hours)
```python
# Same code - run again
result = await engine.backtest_model(
    model_name="gnn",
    symbol="AAPL",
    start_date="2025-01-01",
    end_date="2025-11-09",
    prediction_function=get_gnn_prediction
)
```

**Output:**
```
[Cache HIT] Loaded AAPL from cache (age: 0.1h, 91 rows)
```

**Time:** ~0.01 seconds (**300× faster!**)

## Performance Demo

Run the included demo script to see the performance improvement:

```bash
python scripts/demo_cache_performance.py
```

**Example Output:**
```
First run:  0.922 seconds
Second run: 0.023 seconds
Speedup:    40.9x faster
Time saved: 0.900 seconds (97.6% reduction)
```

## Cache Configuration

### Default (Recommended)
```python
engine = BacktestEngine()
# Cache expires after 24 hours
```

### Custom Expiry Time
```python
engine = BacktestEngine()
engine.cache_expiry_hours = 12  # Refresh every 12 hours
```

### Always Fetch Fresh (Testing)
```python
engine = BacktestEngine()
engine.cache_expiry_hours = 0  # Disable caching
```

## Cache Management

### View Cache Files
```bash
# Windows
dir data\backtest_cache

# Linux/Mac
ls -lh data/backtest_cache/
```

### Clear Cache
```bash
# Windows
rmdir /s /q data\backtest_cache

# Linux/Mac
rm -rf data/backtest_cache
```

### Check Cache Statistics
```python
from pathlib import Path

cache_dir = Path("data/backtest_cache")
cache_files = list(cache_dir.glob("*.pkl"))

print(f"Cache files: {len(cache_files)}")
print(f"Total size: {sum(f.stat().st_size for f in cache_files) / 1024:.1f} KB")
```

## Testing

### Test Cache Functionality
```bash
# Create mock data
python scripts/create_mock_cache.py

# Test cache behavior
python scripts/test_backtest_cache.py

# Performance demo
python scripts/demo_cache_performance.py
```

## Troubleshooting

### Cache Not Working?

**Check cache directory exists:**
```python
import os
print(os.path.exists("data/backtest_cache"))  # Should be True
```

**Check cache files:**
```bash
# Windows
dir data\backtest_cache\*.pkl

# Linux/Mac
ls data/backtest_cache/*.pkl
```

**Enable debug logging:**
```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Cache Corrupted?

**Clear and rebuild:**
```bash
# Clear cache
rm -rf data/backtest_cache

# Re-run backtest (will fetch fresh data)
python your_backtest_script.py
```

## Key Features

- **Automatic**: No code changes needed
- **Fast**: 300× faster for cached data (0.01s vs 3s)
- **Smart**: 24-hour expiry ensures fresh data
- **Reliable**: Graceful fallback to yfinance on errors
- **Transparent**: All operations logged
- **Efficient**: 5-20 KB per 90 days of data

## Cache Behavior

| Scenario | Behavior | Time | Network |
|----------|----------|------|---------|
| Cache HIT (fresh) | Load from disk | ~0.01s | No |
| Cache MISS (no file) | Fetch + cache | ~3s | Yes |
| Cache EXPIRED (>24h) | Refresh cache | ~3s | Yes |
| Cache ERROR (corrupted) | Fetch + rebuild | ~3s | Yes |

## File Locations

- **Cache directory:** `data/backtest_cache/`
- **Implementation:** `src/backtesting/backtest_engine.py`
- **Test scripts:** `scripts/test_backtest_cache.py`, `scripts/demo_cache_performance.py`
- **Documentation:** `docs/BACKTESTING_CACHE.md`

## FAQ

**Q: Do I need to change my code?**
A: No! Caching is automatic and transparent.

**Q: How long is data cached?**
A: 24 hours by default (configurable).

**Q: What if cache is corrupted?**
A: System automatically falls back to yfinance.

**Q: Does this work with all stocks?**
A: Yes! Any symbol supported by yfinance.

**Q: How much disk space does cache use?**
A: ~5-20 KB per symbol per 90 days of data.

**Q: Can I disable caching?**
A: Yes, set `cache_expiry_hours = 0`.

---

**For detailed documentation, see:** `docs/BACKTESTING_CACHE.md`
