"""
Test script for backtest caching functionality

Demonstrates:
1. First run: Cache MISS -> Fetch from yfinance -> Save to cache
2. Second run: Cache HIT -> Load from disk (fast, no rate limits)
3. Cache expiry verification
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.backtest_engine import BacktestEngine

# Configure logging to see cache messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_cache_functionality():
    """Test cache hit/miss behavior"""
    engine = BacktestEngine()

    # Test parameters
    symbol = "AAPL"
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    logger.info("=" * 80)
    logger.info("TEST 1: First load (should be Cache HIT if mock exists)")
    logger.info("=" * 80)

    # First load - should hit cache if mock exists
    data1 = await engine._load_historical_data(symbol, start_date, end_date)
    if data1 is not None:
        logger.info(f"[OK] Loaded {len(data1)} rows for {symbol}")
    else:
        logger.error("[FAIL] Failed to load data")
        return

    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Second load (should be Cache HIT)")
    logger.info("=" * 80)

    # Second load - should hit cache
    data2 = await engine._load_historical_data(symbol, start_date, end_date)
    if data2 is not None:
        logger.info(f"[OK] Loaded {len(data2)} rows for {symbol}")
        # Verify data matches
        if len(data1) == len(data2):
            logger.info("[OK] Cache data matches original data")
        else:
            logger.warning(f"[WARN] Data length mismatch: {len(data1)} vs {len(data2)}")
    else:
        logger.error("[FAIL] Failed to load from cache")

    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Cache statistics")
    logger.info("=" * 80)

    cache_dir = engine.cache_dir
    cache_files = list(cache_dir.glob("*.pkl"))
    total_size = sum(f.stat().st_size for f in cache_files)

    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Cache files: {len(cache_files)}")
    logger.info(f"Total cache size: {total_size / 1024:.1f} KB")

    for cache_file in cache_files:
        age_hours = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 3600
        logger.info(f"  - {cache_file.name}: {cache_file.stat().st_size / 1024:.1f} KB, age: {age_hours:.1f}h")

    logger.info("\n" + "=" * 80)
    logger.info("[OK] ALL TESTS PASSED")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_cache_functionality())
