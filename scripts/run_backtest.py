"""
CLI Script for Running ML Model Backtests

Usage:
    # Quick test (3 stocks, 1 year)
    python scripts/run_backtest.py --symbols AAPL MSFT GOOGL --years 1

    # Full backtest (all 46 trained stocks, 3 years)
    python scripts/run_backtest.py --symbols TIER_1 --years 3

    # Single stock
    python scripts/run_backtest.py --symbol AAPL --years 2
"""

import asyncio
import argparse
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# CRITICAL: Import TensorFlow first (Windows DLL fix)
# This must be imported before any other ML libraries to avoid DLL initialization errors
# See CLAUDE.md for details
import tensorflow as tf

from src.backtesting.backtest_engine import BacktestEngine, BacktestResult
from src.backtesting.metrics import PerformanceMetrics
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 46 stocks with pre-trained GNN models
TIER_1_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMZN', 'TSLA', 'NFLX', 'AMD', 'INTC',
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'XOM', 'CVX', 'UNH', 'JNJ', 'PFE',
    'V', 'MA', 'PYPL', 'DIS', 'CMCSA', 'T', 'VZ', 'PG', 'KO', 'PEP',
    'WMT', 'TGT', 'COST', 'HD', 'LOW', 'NKE', 'MCD', 'SBUX', 'F', 'GM',
    'BA', 'CAT', 'GE', 'LMT', 'MMM', 'HON'
]


async def run_backtest_single_stock(
    engine: BacktestEngine,
    symbol: str,
    start_date: str,
    end_date: str
) -> dict:
    """Run backtest for all models on one stock"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Backtesting {symbol} ({start_date} to {end_date})")
    logger.info(f"{'='*60}\n")

    results = await engine.backtest_all_models(symbol, start_date, end_date)

    if not results:
        logger.error(f"❌ No results for {symbol}")
        return None

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS FOR {symbol}")
    logger.info(f"{'='*60}\n")

    for model_name, result in results.items():
        m = result.metrics
        logger.info(f"{model_name.upper():10} | RMSE: {m.rmse:6.3f} | Accuracy: {m.directional_accuracy:5.1%} | Sharpe: {m.sharpe_ratio:5.2f} | Grade: {result.grade}")

    # Generate report
    report = engine.generate_summary_report(results)
    logger.info(f"\n{report}")

    return {
        'symbol': symbol,
        'results': {k: {
            'rmse': v.metrics.rmse,
            'mae': v.metrics.mae,
            'directional_accuracy': v.metrics.directional_accuracy,
            'sharpe_ratio': v.metrics.sharpe_ratio,
            'grade': v.grade
        } for k, v in results.items()}
    }


async def main():
    parser = argparse.ArgumentParser(description='Run ML model backtests')
    parser.add_argument('--symbol', type=str, help='Single stock symbol')
    parser.add_argument('--symbols', nargs='+', help='Multiple stock symbols or TIER_1')
    parser.add_argument('--years', type=int, default=1, help='Years of historical data (default: 1)')

    args = parser.parse_args()

    # Determine symbols to test
    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        if 'TIER_1' in args.symbols:
            symbols = TIER_1_STOCKS
        else:
            symbols = args.symbols
    else:
        # Default: quick test with 3 stocks
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        logger.info("No symbols specified, using default: AAPL, MSFT, GOOGL")

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.years * 365)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    logger.info(f"\n{'='*60}")
    logger.info(f"BACKTEST CONFIGURATION")
    logger.info(f"{'='*60}")
    logger.info(f"Symbols: {len(symbols)} stocks")
    logger.info(f"Period: {start_str} to {end_str} ({args.years} year{'s' if args.years > 1 else ''})")
    logger.info(f"{'='*60}\n")

    # Run backtests
    engine = BacktestEngine()
    all_results = []

    for symbol in symbols:
        try:
            result = await run_backtest_single_stock(engine, symbol, start_str, end_str)
            if result:
                all_results.append(result)
        except Exception as e:
            logger.error(f"❌ Error backtesting {symbol}: {e}")
            continue

    # Save results to JSON
    output_file = project_root / 'BACKTEST_RESULTS.json'
    with open(output_file, 'w') as f:
        json.dump({
            'backtest_date': datetime.now().isoformat(),
            'period': {
                'start': start_str,
                'end': end_str,
                'years': args.years
            },
            'symbols_tested': len(all_results),
            'results': all_results
        }, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Backtest complete!")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Stocks tested: {len(all_results)}/{len(symbols)}")
    logger.info(f"{'='*60}\n")

    # Summary statistics
    if all_results:
        logger.info("SUMMARY STATISTICS")
        logger.info("=" * 60)

        for model in ['gnn', 'mamba', 'pinn']:
            rmse_values = [r['results'][model]['rmse'] for r in all_results if model in r['results']]
            acc_values = [r['results'][model]['directional_accuracy'] for r in all_results if model in r['results']]

            if rmse_values:
                import numpy as np
                logger.info(f"{model.upper():10} | Avg RMSE: {np.mean(rmse_values):.3f} | Avg Accuracy: {np.mean(acc_values):.1%}")

    return 0


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
