"""
Performance demonstration: Cache vs No-Cache

Shows the dramatic performance improvement from caching
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.backtest_engine import BacktestEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def demo_performance():
    """Demonstrate cache performance improvement"""
    engine = BacktestEngine()

    # Test parameters
    symbols = ["AAPL", "MSFT", "GOOGL"]
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    logger.info("=" * 80)
    logger.info("CACHE PERFORMANCE DEMONSTRATION")
    logger.info("=" * 80)
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Cache directory: {engine.cache_dir}")
    logger.info(f"Cache expiry: {engine.cache_expiry_hours} hours")
    logger.info("")

    # Scenario 1: First run (cache MISS expected for uncached symbols)
    logger.info("=" * 80)
    logger.info("SCENARIO 1: First Load (Cache MISS or use existing cache)")
    logger.info("=" * 80)

    start_time = time.time()
    results_1 = []

    for symbol in symbols:
        data = await engine._load_historical_data(symbol, start_date, end_date)
        results_1.append((symbol, data is not None, len(data) if data is not None else 0))

    elapsed_1 = time.time() - start_time

    logger.info("")
    logger.info("First load results:")
    for symbol, success, rows in results_1:
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {symbol}: {status} ({rows} rows)")
    logger.info(f"Total time: {elapsed_1:.3f} seconds")
    logger.info("")

    # Scenario 2: Second run (cache HIT expected)
    logger.info("=" * 80)
    logger.info("SCENARIO 2: Second Load (Cache HIT expected)")
    logger.info("=" * 80)

    start_time = time.time()
    results_2 = []

    for symbol in symbols:
        data = await engine._load_historical_data(symbol, start_date, end_date)
        results_2.append((symbol, data is not None, len(data) if data is not None else 0))

    elapsed_2 = time.time() - start_time

    logger.info("")
    logger.info("Second load results:")
    for symbol, success, rows in results_2:
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {symbol}: {status} ({rows} rows)")
    logger.info(f"Total time: {elapsed_2:.3f} seconds")
    logger.info("")

    # Performance comparison
    logger.info("=" * 80)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("=" * 80)

    if elapsed_2 > 0:
        speedup = elapsed_1 / elapsed_2
        time_saved = elapsed_1 - elapsed_2
        logger.info(f"First run:  {elapsed_1:.3f} seconds")
        logger.info(f"Second run: {elapsed_2:.3f} seconds")
        logger.info(f"Speedup:    {speedup:.1f}x faster")
        logger.info(f"Time saved: {time_saved:.3f} seconds ({time_saved/elapsed_1*100:.1f}% reduction)")
    else:
        logger.info(f"First run:  {elapsed_1:.3f} seconds")
        logger.info(f"Second run: {elapsed_2:.3f} seconds (instant!)")

    logger.info("")

    # Cache statistics
    logger.info("=" * 80)
    logger.info("CACHE STATISTICS")
    logger.info("=" * 80)

    cache_files = list(engine.cache_dir.glob("*.pkl"))
    total_size = sum(f.stat().st_size for f in cache_files)

    logger.info(f"Cache directory: {engine.cache_dir}")
    logger.info(f"Total cache files: {len(cache_files)}")
    logger.info(f"Total cache size: {total_size / 1024:.2f} KB")
    logger.info("")

    if cache_files:
        logger.info("Cache file details:")
        for cache_file in sorted(cache_files):
            age_hours = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 3600
            size_kb = cache_file.stat().st_size / 1024
            logger.info(f"  {cache_file.name}")
            logger.info(f"    Size: {size_kb:.2f} KB, Age: {age_hours:.2f}h")
    else:
        logger.info("No cache files found")

    logger.info("")
    logger.info("=" * 80)
    logger.info("DEMONSTRATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(demo_performance())
