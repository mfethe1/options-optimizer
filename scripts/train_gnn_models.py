#!/usr/bin/env python3
"""
GNN Pre-Training Script

Trains GNN models for top 50 stocks with correlation-based architecture.
Saves weights and metadata for production serving.

Usage:
    python scripts/train_gnn_models.py --symbols TIER_1
    python scripts/train_gnn_models.py --symbols AAPL,MSFT,GOOGL
    python scripts/train_gnn_models.py --test  # Quick test with 3 symbols

Performance Targets:
    - Per-symbol training: ~35-65s
    - Batch training (50 symbols, 10 parallel): ~4-5 minutes
    - Prediction latency after pre-training: <2s (vs 5-8s without)
"""

import sys
import os
import argparse
import asyncio
import logging
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import TensorFlow first (Windows DLL fix)
try:
    import tensorflow as tf
except ImportError:
    tf = None

import numpy as np

# Import from project
from src.ml.graph_neural_network.stock_gnn import GNNPredictor, CorrelationGraphBuilder
from src.api.ml_integration_helpers import (
    fetch_historical_prices,
    get_correlated_stocks,
    build_node_features
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Top 50 high-volume stocks (Tier 1)
TIER_1_STOCKS = [
    # Mega-cap tech (highest volume, highest priority)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',

    # Financial sector leaders
    'JPM', 'BAC', 'GS', 'MS', 'BLK', 'C', 'WFC',

    # Healthcare leaders
    'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY',

    # Consumer/Retail
    'WMT', 'HD', 'DIS', 'NKE', 'MCD', 'COST',

    # Energy
    'XOM', 'CVX', 'COP',

    # Indices/ETFs
    'SPY', 'QQQ', 'IWM', 'DIA',

    # Other high-volume
    'NFLX', 'AMD', 'INTC', 'CSCO', 'ADBE', 'CRM', 'ORCL',
    'V', 'MA', 'PYPL', 'SQ', 'UBER', 'ABNB'
]

# Test subset for quick validation
TEST_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL']


async def train_single_symbol(
    symbol: str,
    save_dir: str = 'models/gnn',
    historical_days: int = 1000,
    epochs: int = 20
) -> Tuple[str, Dict[str, Any]]:
    """
    Train GNN for single symbol

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        save_dir: Base directory for saving weights/metadata
        historical_days: Days of historical data for training
        epochs: Training epochs

    Returns:
        (symbol, result_dict)
    """
    start_time = time.time()
    logger.info(f"[{symbol}] Starting training...")

    try:
        # 1. Get correlated stocks
        logger.info(f"[{symbol}] Fetching correlated stocks...")
        correlated = await get_correlated_stocks(symbol, top_n=10)
        all_symbols = [symbol] + correlated
        logger.info(f"[{symbol}] Correlated stocks: {', '.join(correlated[:3])}...")

        # 2. Fetch historical data
        logger.info(f"[{symbol}] Fetching {historical_days} days of price data...")
        price_data = await fetch_historical_prices(all_symbols, days=historical_days)

        # Verify data quality
        if symbol not in price_data or len(price_data[symbol]) < 60:
            raise ValueError(f"Insufficient price data for {symbol}: {len(price_data.get(symbol, []))} days")

        # 3. Build features
        logger.info(f"[{symbol}] Building node features...")
        features = await build_node_features(all_symbols, price_data)

        # 4. Initialize and train GNN
        logger.info(f"[{symbol}] Initializing GNN predictor...")
        predictor = GNNPredictor(symbols=all_symbols)

        logger.info(f"[{symbol}] Training GNN ({epochs} epochs)...")
        train_result = await predictor.train(
            price_data, features, epochs=epochs
        )

        # 5. Weights already saved by predictor.train() to models/gnn/weights.weights.h5
        # But we want per-symbol weights, so save again with custom path
        weights_path = os.path.join(save_dir, 'weights', f'{symbol}.weights.h5')
        predictor.gnn.save_weights(weights_path)
        logger.info(f"[{symbol}] Saved weights to {weights_path}")

        # 6. Save correlation matrix
        logger.info(f"[{symbol}] Computing and caching correlation matrix...")
        graph_builder = CorrelationGraphBuilder()
        graph = graph_builder.build_graph(price_data, features)

        corr_path = os.path.join(save_dir, 'correlations', f'{symbol}_corr.npy')
        np.save(corr_path, graph.correlation_matrix)

        # Calculate correlation stats
        avg_correlation = float(np.mean(np.abs(graph.correlation_matrix)))
        max_correlation = float(np.max(graph.correlation_matrix[graph.correlation_matrix < 1.0]))
        num_edges = len(graph.edge_index[0]) if len(graph.edge_index) > 0 else 0

        # 7. Save metadata
        logger.info(f"[{symbol}] Saving metadata...")
        metadata = {
            'symbol': symbol,
            'training_date': datetime.now().isoformat(),
            'data_version': '1.0.0',
            'training_config': {
                'epochs': epochs,
                'correlated_symbols': correlated,
                'num_stocks': len(all_symbols),
                'historical_days': historical_days,
                'hidden_dim': predictor.gnn.hidden_dim,
                'num_gcn_layers': predictor.gnn.num_gcn_layers,
                'num_gat_heads': predictor.gnn.num_gat_heads,
            },
            'performance_metrics': {
                'final_loss': train_result.get('final_loss', 0.0),
                'training_time_seconds': time.time() - start_time,
                'num_params': train_result.get('num_nodes', 0) * 64 * 128,  # Approximate
            },
            'data_stats': {
                'historical_days_actual': len(price_data[symbol]),
                'avg_correlation': avg_correlation,
                'max_correlation': max_correlation,
                'num_edges': num_edges,
            },
            'files': {
                'weights': weights_path,
                'correlation_matrix': corr_path,
            }
        }

        metadata_path = os.path.join(save_dir, 'metadata', f'{symbol}_metadata.json')
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        elapsed = time.time() - start_time
        logger.info(
            f"[{symbol}] ✓ Training complete! "
            f"Loss: {train_result['final_loss']:.4f}, "
            f"Time: {elapsed:.1f}s, "
            f"Avg Corr: {avg_correlation:.3f}"
        )

        return symbol, {
            'success': True,
            'elapsed_seconds': elapsed,
            'final_loss': train_result['final_loss'],
            'avg_correlation': avg_correlation,
            'metadata': metadata
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[{symbol}] ✗ Training failed after {elapsed:.1f}s: {e}", exc_info=True)
        return symbol, {
            'success': False,
            'error': str(e),
            'elapsed_seconds': elapsed
        }


async def train_batch(
    symbols: List[str],
    save_dir: str = 'models/gnn',
    parallel: int = 10,
    historical_days: int = 1000,
    epochs: int = 20
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Train GNN models for multiple symbols in parallel batches

    Args:
        symbols: List of stock symbols
        save_dir: Base directory for saving
        parallel: Number of concurrent training jobs
        historical_days: Days of historical data
        epochs: Training epochs

    Returns:
        List of (symbol, result) tuples
    """
    logger.info("=" * 70)
    logger.info(f"GNN Pre-Training: {len(symbols)} symbols ({parallel} parallel workers)")
    logger.info("=" * 70)

    # Create directories
    for subdir in ['weights', 'metadata', 'correlations']:
        os.makedirs(os.path.join(save_dir, subdir), exist_ok=True)
        logger.info(f"Created directory: {save_dir}/{subdir}")

    # Train in parallel batches
    results = []
    total_start = time.time()

    for i in range(0, len(symbols), parallel):
        batch = symbols[i:i+parallel]
        batch_num = (i // parallel) + 1
        total_batches = (len(symbols) + parallel - 1) // parallel

        logger.info("")
        logger.info(f"[Batch {batch_num}/{total_batches}] Training: {', '.join(batch)}")
        logger.info("-" * 70)

        batch_start = time.time()

        # Run batch in parallel
        batch_results = await asyncio.gather(
            *[train_single_symbol(sym, save_dir, historical_days, epochs) for sym in batch],
            return_exceptions=True  # Don't fail entire batch if one symbol fails
        )

        # Handle exceptions from gather
        for sym, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                logger.error(f"[{sym}] Exception during training: {result}")
                results.append((sym, {'success': False, 'error': str(result)}))
            else:
                results.append(result)

        batch_elapsed = time.time() - batch_start
        success_count = sum(1 for _, r in batch_results if not isinstance(r, Exception) and r['success'])

        logger.info(f"[Batch {batch_num}/{total_batches}] Complete: {success_count}/{len(batch)} successful, {batch_elapsed:.1f}s")

    total_elapsed = time.time() - total_start

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("Training Summary")
    logger.info("=" * 70)

    successful = [s for s, r in results if r['success']]
    failed = [s for s, r in results if not r['success']]

    logger.info(f"Total symbols: {len(symbols)}")
    logger.info(f"Successful:    {len(successful)} ({len(successful)/len(symbols)*100:.1f}%)")
    logger.info(f"Failed:        {len(failed)} ({len(failed)/len(symbols)*100:.1f}%)")
    logger.info(f"Total time:    {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    logger.info(f"Avg per symbol: {total_elapsed/len(symbols):.1f}s")

    if successful:
        avg_loss = np.mean([r['final_loss'] for s, r in results if r['success']])
        avg_corr = np.mean([r['avg_correlation'] for s, r in results if r['success']])
        logger.info(f"Avg loss:      {avg_loss:.4f}")
        logger.info(f"Avg correlation: {avg_corr:.3f}")

    if failed:
        logger.info("")
        logger.info("Failed symbols:")
        for symbol in failed:
            error = next(r['error'] for s, r in results if s == symbol and not r['success'])
            logger.info(f"  - {symbol}: {error}")

    logger.info("")
    logger.info("✓ GNN pre-training complete!")
    logger.info(f"Models saved to: {save_dir}/weights/")
    logger.info("=" * 70)

    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Train GNN models for stock prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all Tier 1 stocks (50 symbols, ~4-5 minutes)
  python scripts/train_gnn_models.py --symbols TIER_1

  # Train specific symbols
  python scripts/train_gnn_models.py --symbols AAPL,MSFT,GOOGL

  # Quick test (3 symbols)
  python scripts/train_gnn_models.py --test

  # Custom settings
  python scripts/train_gnn_models.py --symbols TIER_1 --parallel 5 --epochs 30
        """
    )

    parser.add_argument(
        '--symbols',
        default='TEST',
        help='Comma-separated symbols or TIER_1 (default: TEST for quick validation)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Quick test with 3 symbols (AAPL, MSFT, GOOGL)'
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=10,
        help='Number of parallel training workers (default: 10)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Training epochs per model (default: 20)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=1000,
        help='Historical days for training data (default: 1000)'
    )
    parser.add_argument(
        '--save-dir',
        default='models/gnn',
        help='Base directory for saving models (default: models/gnn)'
    )

    args = parser.parse_args()

    # Determine symbol list
    if args.test:
        symbols = TEST_SYMBOLS
        logger.info("Test mode: Using 3 symbols for quick validation")
    elif args.symbols == 'TIER_1':
        symbols = TIER_1_STOCKS
    elif args.symbols == 'TEST':
        symbols = TEST_SYMBOLS
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]

    logger.info(f"Training {len(symbols)} GNN models...")
    logger.info(f"Symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
    logger.info(f"Config: {args.epochs} epochs, {args.days} days history, {args.parallel} parallel workers")

    # Check TensorFlow availability
    if tf is None:
        logger.error("TensorFlow not available! Cannot train GNN models.")
        logger.error("Install with: pip install tensorflow")
        sys.exit(1)

    # GPU configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid OOM errors
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU acceleration enabled: {len(gpus)} GPU(s) detected")
            logger.info(f"GPU devices: {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            logger.warning(f"GPU configuration failed: {e}")
            logger.info("Falling back to CPU training")
    else:
        logger.warning("No GPU detected - training on CPU (slower but functional)")
        logger.info("For GPU acceleration on Windows, install: pip install tensorflow[and-cuda]")

    # Run training
    try:
        results = asyncio.run(
            train_batch(
                symbols=symbols,
                save_dir=args.save_dir,
                parallel=args.parallel,
                historical_days=args.days,
                epochs=args.epochs
            )
        )

        # Exit code based on success rate
        successful = sum(1 for _, r in results if r['success'])
        success_rate = successful / len(results)

        if success_rate == 1.0:
            logger.info("✓ All models trained successfully!")
            sys.exit(0)
        elif success_rate >= 0.8:
            logger.warning(f"⚠ {successful}/{len(results)} models trained (some failures)")
            sys.exit(0)
        else:
            logger.error(f"✗ Only {successful}/{len(results)} models trained successfully")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("\n\n⚠ Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"✗ Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
