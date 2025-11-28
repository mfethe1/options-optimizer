#!/usr/bin/env python3
"""
GNN Training Diagnostic Script

Tests GNN model training pipeline end-to-end:
1. GPU detection and configuration
2. Single model training
3. Model loading and prediction
4. Data pipeline validation

Usage:
    python scripts/test_gnn_training.py
"""

import sys
import os
import logging
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import TensorFlow first (Windows DLL fix)
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None

import numpy as np

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


def test_gpu_detection():
    """Test 1: GPU Detection"""
    logger.info("=" * 70)
    logger.info("Test 1: GPU Detection")
    logger.info("=" * 70)

    if not TENSORFLOW_AVAILABLE:
        logger.error("FAIL: TensorFlow not available")
        return False

    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"GPU detected: {len(gpus)} device(s)")
        for gpu in gpus:
            logger.info(f"  - {gpu.name}")
        return True
    else:
        logger.warning("No GPU detected - CPU training will be used (slower but functional)")
        logger.info("For GPU acceleration: pip install tensorflow[and-cuda]")
        return True  # CPU is acceptable


async def test_data_pipeline():
    """Test 2: Data Pipeline"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Test 2: Data Pipeline Validation")
    logger.info("=" * 70)

    symbol = 'AAPL'

    # 2.1: Fetch historical prices
    logger.info(f"Fetching historical prices for {symbol}...")
    price_data = await fetch_historical_prices([symbol], days=100)

    if symbol not in price_data:
        logger.error(f"FAIL: No price data for {symbol}")
        return False

    prices = price_data[symbol]
    logger.info(f"PASS: Fetched {len(prices)} days of price data")
    logger.info(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")

    # 2.2: Get correlated stocks
    logger.info(f"Fetching correlated stocks for {symbol}...")
    correlated = await get_correlated_stocks(symbol, top_n=5)
    logger.info(f"PASS: Found {len(correlated)} correlated stocks: {', '.join(correlated)}")

    # 2.3: Build node features
    all_symbols = [symbol] + correlated
    logger.info(f"Fetching price data for {len(all_symbols)} symbols...")
    all_prices = await fetch_historical_prices(all_symbols, days=100)

    logger.info(f"Building node features...")
    features = await build_node_features(all_symbols, all_prices)

    if symbol not in features:
        logger.error(f"FAIL: No features for {symbol}")
        return False

    feature_dim = len(features[symbol])
    logger.info(f"PASS: Built {feature_dim}-dimensional features for {len(features)} stocks")
    logger.info(f"  Feature sample ({symbol}): {features[symbol]}")

    return True


async def test_single_model_training():
    """Test 3: Single Model Training"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Test 3: Single Model Training")
    logger.info("=" * 70)

    if not TENSORFLOW_AVAILABLE:
        logger.error("FAIL: TensorFlow not available")
        return False

    symbol = 'AAPL'

    # 3.1: Prepare data
    logger.info(f"Preparing data for {symbol}...")
    correlated = await get_correlated_stocks(symbol, top_n=5)
    all_symbols = [symbol] + correlated

    price_data = await fetch_historical_prices(all_symbols, days=100)
    features = await build_node_features(all_symbols, price_data)

    # 3.2: Initialize GNN
    logger.info(f"Initializing GNN predictor...")
    predictor = GNNPredictor(symbols=all_symbols)

    # 3.3: Train model
    logger.info(f"Training GNN (5 epochs)...")
    train_result = await predictor.train(price_data, features, epochs=5)

    logger.info(f"PASS: Training complete!")
    logger.info(f"  Final loss: {train_result['final_loss']:.6f}")
    logger.info(f"  Weights saved: {train_result['weights_path']}")
    logger.info(f"  Number of nodes: {train_result['num_nodes']}")

    # 3.4: Test prediction
    logger.info(f"Testing prediction...")
    predictions = await predictor.predict(price_data, features)

    if symbol not in predictions:
        logger.error(f"FAIL: No prediction for {symbol}")
        return False

    logger.info(f"PASS: Prediction successful!")
    logger.info(f"  {symbol} predicted return: {predictions[symbol]:.6f}")

    return True


async def test_model_loading():
    """Test 4: Model Weight Loading"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Test 4: Model Weight Loading")
    logger.info("=" * 70)

    if not TENSORFLOW_AVAILABLE:
        logger.error("FAIL: TensorFlow not available")
        return False

    # Check if pre-trained models exist
    weights_dir = Path('models/gnn/weights')
    if not weights_dir.exists():
        logger.warning("No pre-trained models found - run train_gnn_models.py first")
        return True

    model_files = list(weights_dir.glob('*.h5'))
    logger.info(f"Found {len(model_files)} pre-trained models")

    if not model_files:
        logger.warning("No model files found - run train_gnn_models.py first")
        return True

    # Test loading a random model
    test_file = model_files[0]
    symbol = test_file.stem.replace('.weights', '')

    logger.info(f"Testing load of {symbol} model...")

    # Initialize predictor for this symbol
    correlated = await get_correlated_stocks(symbol, top_n=5)
    all_symbols = [symbol] + correlated

    predictor = GNNPredictor(symbols=all_symbols)

    # Load weights
    success = predictor.gnn.load_weights(str(test_file))

    if success:
        logger.info(f"PASS: Successfully loaded weights from {test_file.name}")
    else:
        logger.error(f"FAIL: Could not load weights from {test_file.name}")
        return False

    # Test prediction with loaded model
    price_data = await fetch_historical_prices(all_symbols, days=100)
    features = await build_node_features(all_symbols, price_data)
    predictions = await predictor.predict(price_data, features)

    logger.info(f"PASS: Prediction with loaded model successful!")
    logger.info(f"  {symbol} predicted return: {predictions[symbol]:.6f}")

    return True


async def main():
    """Run all tests"""
    logger.info("GNN Training Diagnostic Script")
    logger.info("Testing GNN training pipeline end-to-end")
    logger.info("")

    results = {
        'GPU Detection': test_gpu_detection(),
        'Data Pipeline': await test_data_pipeline(),
        'Single Model Training': await test_single_model_training(),
        'Model Loading': await test_model_loading()
    }

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("Test Summary")
    logger.info("=" * 70)

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"{test_name}: {status}")

    all_passed = all(results.values())

    logger.info("")
    if all_passed:
        logger.info("All tests PASSED! GNN training pipeline is working correctly.")
        return 0
    else:
        logger.error("Some tests FAILED. Check logs above for details.")
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
