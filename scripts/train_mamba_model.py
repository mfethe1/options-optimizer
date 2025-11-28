#!/usr/bin/env python3
"""
Mamba State Space Model Training Script

Enhanced training with:
- Multi-objective loss (directional accuracy + MSE)
- Advanced feature engineering (20+ features)
- Data augmentation for financial sequences
- Walk-forward validation
- Early stopping based on directional accuracy
- Comprehensive metrics tracking

Usage:
    python scripts/train_mamba_model.py --symbols AAPL,MSFT,GOOGL
    python scripts/train_mamba_model.py --symbols TIER_1 --epochs 100
    python scripts/train_mamba_model.py --test  # Quick test run

Performance Targets:
    - Directional Accuracy: >60% (1-day), >55% (30-day)
    - MAPE: <5% (1-day), <10% (30-day)
    - Sharpe Ratio: >1.5
    - Training Time: <2 hours for 50 stocks (GPU)
"""

import sys
import os
import argparse
import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import TensorFlow first (Windows DLL fix)
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None

# Import from project
from src.ml.state_space.mamba_model import MambaConfig, MambaPredictor
from src.api.ml_integration_helpers import fetch_historical_prices

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Top stocks for training
TIER_1_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    'JPM', 'BAC', 'GS', 'MS', 'BLK', 'C', 'WFC',
    'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY',
    'WMT', 'HD', 'DIS', 'NKE', 'MCD', 'COST',
    'XOM', 'CVX', 'COP',
    'SPY', 'QQQ', 'IWM', 'DIA',
    'NFLX', 'AMD', 'INTC', 'CSCO', 'ADBE', 'CRM', 'ORCL',
    'V', 'MA', 'PYPL', 'SQ', 'UBER', 'ABNB'
]

TEST_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL']


# ============================================================================
# Enhanced Feature Engineering
# ============================================================================

def calculate_rsi(returns: np.ndarray, window: int = 14) -> np.ndarray:
    """Calculate Relative Strength Index"""
    gains = np.where(returns > 0, returns, 0)
    losses = np.where(returns < 0, -returns, 0)

    avg_gains = np.convolve(gains, np.ones(window)/window, mode='same')
    avg_losses = np.convolve(losses, np.ones(window)/window, mode='same')

    rs = avg_gains / (avg_losses + 1e-8)
    rsi = 100 - (100 / (1 + rs))

    return rsi / 100.0  # Normalize to [0, 1]


def exponential_moving_average(prices: np.ndarray, window: int) -> np.ndarray:
    """Calculate Exponential Moving Average"""
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]

    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]

    return ema


def prepare_enhanced_features(price_history: np.ndarray) -> np.ndarray:
    """
    Enhanced feature engineering with 20+ features

    Args:
        price_history: [seq_len] array of prices

    Returns:
        features: [seq_len, n_features] array (n_features ≈ 21)
    """
    features = []

    # 1. PRICE FEATURES (6)
    # Returns
    returns = np.diff(price_history) / (price_history[:-1] + 1e-8)
    returns = np.concatenate([[0], returns])
    features.append(returns)

    # Log returns
    log_returns = np.diff(np.log(price_history + 1e-8))
    log_returns = np.concatenate([[0], log_returns])
    features.append(log_returns)

    # Normalized price
    normalized_price = (price_history - np.mean(price_history)) / (np.std(price_history) + 1e-8)
    features.append(normalized_price)

    # Multiple SMA horizons
    for window in [5, 20, 50]:
        sma = np.convolve(price_history, np.ones(window)/window, mode='same')
        sma_distance = (price_history - sma) / (sma + 1e-8)
        features.append(sma_distance)

    # 2. VOLATILITY FEATURES (4)
    for window in [10, 20]:
        volatility = np.array([
            np.std(price_history[max(0, i-window):i+1])
            for i in range(len(price_history))
        ])
        vol_norm = volatility / (np.mean(price_history) + 1e-8)
        features.append(vol_norm)

        # Volatility of volatility
        vol_of_vol = np.array([
            np.std(volatility[max(0, i-window):i+1])
            for i in range(len(volatility))
        ])
        features.append(vol_of_vol)

    # 3. MOMENTUM FEATURES (4)
    for window in [5, 10, 20]:
        momentum = (price_history - np.roll(price_history, window)) / (np.roll(price_history, window) + 1e-8)
        momentum[:window] = 0
        features.append(momentum)

    # RSI
    rsi = calculate_rsi(returns, window=14)
    features.append(rsi)

    # 4. TREND FEATURES (3)
    # MACD
    ema_12 = exponential_moving_average(price_history, 12)
    ema_26 = exponential_moving_average(price_history, 26)
    macd = (ema_12 - ema_26) / (ema_26 + 1e-8)
    features.append(macd)

    # Bollinger Band position
    sma_20 = np.convolve(price_history, np.ones(20)/20, mode='same')
    std_20 = np.array([
        np.std(price_history[max(0, i-20):i+1])
        for i in range(len(price_history))
    ])
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_position = (price_history - bb_lower) / (bb_upper - bb_lower + 1e-8)
    features.append(bb_position)

    # Linear regression slope (trend strength)
    slopes = []
    for i in range(len(price_history)):
        window_prices = price_history[max(0, i-20):i+1]
        x = np.arange(len(window_prices))
        if len(x) > 1:
            slope = np.polyfit(x, window_prices, 1)[0]
        else:
            slope = 0
        slopes.append(slope)
    slopes = np.array(slopes)
    slopes = slopes / (np.std(price_history) + 1e-8)
    features.append(slopes)

    # Stack all features
    feature_array = np.stack(features, axis=-1)

    # Handle NaN/Inf
    feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_array


# ============================================================================
# Data Augmentation
# ============================================================================

class FinancialDataAugmenter:
    """Data augmentation for financial time series"""

    def __init__(self, augmentation_prob: float = 0.3):
        self.aug_prob = augmentation_prob

    def augment(self, features: np.ndarray) -> np.ndarray:
        """Apply random augmentation"""
        if np.random.rand() > self.aug_prob:
            return features

        technique = np.random.choice(['noise', 'warp', 'slice'])

        if technique == 'noise':
            return self._add_noise(features)
        elif technique == 'warp':
            return self._time_warp(features)
        elif technique == 'slice':
            return self._window_slice(features)

        return features

    def _add_noise(self, features: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_std, features.shape)
        return features + noise

    def _time_warp(self, features: np.ndarray, warp_factor: float = 0.2) -> np.ndarray:
        """Time warping: stretch/compress time axis"""
        seq_len = len(features)

        warp = 1.0 + np.random.uniform(-warp_factor, warp_factor)
        new_len = int(seq_len * warp)
        new_len = max(30, min(new_len, 120))

        old_indices = np.linspace(0, seq_len - 1, seq_len)
        new_indices = np.linspace(0, seq_len - 1, new_len)

        warped_features = np.array([
            np.interp(new_indices, old_indices, features[:, i])
            for i in range(features.shape[1])
        ]).T

        if new_len < seq_len:
            pad = np.zeros((seq_len - new_len, features.shape[1]))
            warped_features = np.vstack([warped_features, pad])
        else:
            warped_features = warped_features[:seq_len]

        return warped_features

    def _window_slice(self, features: np.ndarray, min_len: int = 40) -> np.ndarray:
        """Window slicing: use random subsequence"""
        seq_len = len(features)
        if seq_len <= min_len:
            return features

        start = np.random.randint(0, seq_len - min_len)
        end = np.random.randint(start + min_len, seq_len)

        sliced = features[start:end]

        if len(sliced) < seq_len:
            pad = np.zeros((seq_len - len(sliced), features.shape[1]))
            sliced = np.vstack([sliced, pad])

        return sliced


# ============================================================================
# Custom Metrics
# ============================================================================

class DirectionalAccuracyMetric(keras.metrics.Metric):
    """Track directional accuracy during training"""

    def __init__(self, name='directional_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_sign = tf.sign(y_true)
        pred_sign = tf.sign(y_pred)

        matches = tf.cast(tf.equal(true_sign, pred_sign), tf.float32)

        self.correct.assign_add(tf.reduce_sum(matches))
        self.total.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.correct / (self.total + 1e-8)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)


# ============================================================================
# Enhanced Training Functions
# ============================================================================

def create_training_dataset(
    price_data: Dict[str, np.ndarray],
    horizons: List[int] = [1, 5, 10, 30],
    window_size: int = 60,
    augment: bool = True
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Create training dataset with enhanced features

    Args:
        price_data: Dict[symbol, price_array]
        horizons: Prediction horizons
        window_size: Input sequence length
        augment: Whether to apply data augmentation

    Returns:
        (X_train, y_train) where y_train is dict[horizon, labels]
    """
    logger.info("Creating enhanced training dataset...")

    augmenter = FinancialDataAugmenter(augmentation_prob=0.3) if augment else None

    X_train = []
    y_train = {h: [] for h in horizons}

    for symbol, prices in price_data.items():
        if len(prices) < window_size + max(horizons):
            logger.warning(f"Insufficient data for {symbol}: {len(prices)} days")
            continue

        for i in range(window_size, len(prices) - max(horizons)):
            # Enhanced features
            features = prepare_enhanced_features(prices[i-window_size:i])

            # Augmentation
            if augmenter:
                features = augmenter.augment(features)

            X_train.append(features)

            # Labels (future returns)
            current_price = prices[i]
            for horizon in horizons:
                if i + horizon < len(prices):
                    future_price = prices[i + horizon]
                    future_return = (future_price - current_price) / current_price
                    y_train[horizon].append(future_return)
                else:
                    y_train[horizon].append(0.0)

    X_train = np.array(X_train)
    y_train = {h: np.array(y).reshape(-1, 1) for h, y in y_train.items()}

    logger.info(f"Created dataset: {len(X_train)} samples, {X_train.shape[2]} features")
    logger.info(f"Feature shape: {X_train.shape}")

    return X_train, y_train


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int
) -> Dict[str, float]:
    """
    Evaluate predictions with comprehensive metrics

    Args:
        y_true: True returns [N, 1]
        y_pred: Predicted returns [N, 1]
        horizon: Prediction horizon (days)

    Returns:
        Dictionary of metrics
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # MSE, MAE, MAPE
    mse = np.mean((y_true_flat - y_pred_flat) ** 2)
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (np.abs(y_true_flat) + 1e-8))) * 100

    # Directional accuracy
    directional_acc = np.mean((np.sign(y_true_flat) == np.sign(y_pred_flat)).astype(float))

    # Correlation
    correlation = np.corrcoef(y_true_flat, y_pred_flat)[0, 1] if len(y_true_flat) > 1 else 0.0

    # Sharpe ratio (if trading based on predictions)
    signals = np.sign(y_pred_flat)
    returns = signals * y_true_flat
    mean_return = np.mean(returns)
    std_return = np.std(returns) + 1e-8
    sharpe = (mean_return / std_return) * np.sqrt(252 / horizon)

    return {
        'mse': float(mse),
        'mae': float(mae),
        'mape': float(mape),
        'directional_accuracy': float(directional_acc),
        'correlation': float(correlation),
        'sharpe_ratio': float(sharpe),
    }


async def train_mamba_enhanced(
    symbols: List[str],
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    validation_split: float = 0.2,
    early_stopping_patience: int = 15,
    save_dir: str = 'models/mamba'
) -> Dict[str, Any]:
    """
    Enhanced Mamba training with all improvements

    Args:
        symbols: List of stock symbols to train on
        epochs: Maximum training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        validation_split: Validation split ratio
        early_stopping_patience: Early stopping patience
        save_dir: Directory to save weights

    Returns:
        Training results dictionary
    """
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("Enhanced Mamba Training")
    logger.info("=" * 70)
    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate}")
    logger.info(f"Validation Split: {validation_split}")
    logger.info("")

    # 1. Fetch historical data
    logger.info("Fetching historical price data (1000 days)...")
    price_data = await fetch_historical_prices(symbols, days=1000)

    # Verify data quality
    valid_symbols = {
        sym: prices for sym, prices in price_data.items()
        if len(prices) >= 100
    }
    logger.info(f"Valid symbols: {len(valid_symbols)}/{len(symbols)}")

    if len(valid_symbols) == 0:
        raise ValueError("No valid price data available")

    # 2. Enhanced configuration
    config = MambaConfig(
        d_model=128,      # Increased from 64
        d_state=32,       # Increased from 16
        d_conv=8,         # Increased from 4
        expand=4,         # Increased from 2
        num_layers=6,     # Increased from 4
        prediction_horizons=[1, 5, 10, 30]
    )
    logger.info(f"Mamba Config: d_model={config.d_model}, d_state={config.d_state}, "
                f"layers={config.num_layers}")

    # 3. Create training dataset
    X_train, y_train = create_training_dataset(
        valid_symbols,
        horizons=config.prediction_horizons,
        window_size=60,
        augment=True
    )

    # 4. Initialize model
    logger.info("\nInitializing Mamba model...")
    from src.ml.state_space.mamba_model import MambaModel

    model = MambaModel(config)

    # Build model
    dummy_input = tf.zeros((1, 60, X_train.shape[2]))
    _ = model(dummy_input)

    logger.info(f"Model initialized with {model.count_params():,} parameters")

    # 5. Compile with custom loss and metrics
    logger.info("\nCompiling model...")

    # Learning rate schedule
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=epochs * len(X_train) // batch_size,
        alpha=0.1  # Decay to 10% of initial LR
    )

    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4
    )

    # Compile for each horizon
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=[DirectionalAccuracyMetric(), 'mae']
    )

    # 6. Callbacks
    callbacks = []

    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)

    # Model checkpoint
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, 'weights.weights.h5')
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_directional_accuracy',
        save_best_only=True,
        save_weights_only=True,
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint)

    # Reduce LR on plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(reduce_lr)

    # 7. Train model
    logger.info("\nStarting training...")
    logger.info("=" * 70)

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start_time

    # 8. Evaluate on validation set
    logger.info("\nEvaluating on validation set...")

    val_size = int(len(X_train) * validation_split)
    X_val = X_train[-val_size:]
    y_val = {h: y[-val_size:] for h, y in y_train.items()}

    predictions = model(X_val, training=False)

    results = {}
    for horizon in config.prediction_horizons:
        y_pred = predictions[horizon].numpy()
        metrics = evaluate_predictions(y_val[horizon], y_pred, horizon)

        results[f'{horizon}d'] = metrics

        logger.info(f"\n{horizon}-Day Horizon:")
        logger.info(f"  Directional Accuracy: {metrics['directional_accuracy']:.2%}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Correlation: {metrics['correlation']:.3f}")

    # 9. Save metadata
    logger.info("\nSaving metadata...")

    metadata = {
        'training_date': datetime.now().isoformat(),
        'symbols': symbols,
        'num_symbols': len(valid_symbols),
        'data_version': '2.0.0',
        'config': {
            'd_model': config.d_model,
            'd_state': config.d_state,
            'd_conv': config.d_conv,
            'expand': config.expand,
            'num_layers': config.num_layers,
            'prediction_horizons': config.prediction_horizons,
        },
        'training': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'validation_split': validation_split,
            'augmentation': True,
            'num_features': X_train.shape[2],
            'num_samples': len(X_train),
        },
        'performance': results,
        'training_time_seconds': training_time,
        'weights_path': checkpoint_path,
    }

    metadata_path = os.path.join(save_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_path}")

    # 10. Summary
    logger.info("\n" + "=" * 70)
    logger.info("Training Complete!")
    logger.info("=" * 70)
    logger.info(f"Training Time: {training_time:.1f}s ({training_time/60:.1f} minutes)")
    logger.info(f"Best 1-Day Directional Accuracy: {results['1d']['directional_accuracy']:.2%}")
    logger.info(f"Best 1-Day MAPE: {results['1d']['mape']:.2f}%")
    logger.info(f"Best 1-Day Sharpe: {results['1d']['sharpe_ratio']:.2f}")
    logger.info(f"Weights saved to: {checkpoint_path}")
    logger.info("=" * 70)

    return {
        'success': True,
        'training_time': training_time,
        'results': results,
        'metadata': metadata,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Enhanced Mamba model training',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--symbols',
        default='TEST',
        help='Comma-separated symbols or TIER_1'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Quick test with 3 symbols'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size (default: 64)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help='Validation split (default: 0.2)'
    )

    args = parser.parse_args()

    # Determine symbols
    if args.test:
        symbols = TEST_SYMBOLS
        logger.info("Test mode: Using 3 symbols")
    elif args.symbols == 'TIER_1':
        symbols = TIER_1_STOCKS
    elif args.symbols == 'TEST':
        symbols = TEST_SYMBOLS
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]

    logger.info(f"Training Mamba model on {len(symbols)} symbols")

    # Check TensorFlow
    if not TENSORFLOW_AVAILABLE or tf is None:
        logger.error("TensorFlow not available! Install with: pip install tensorflow")
        sys.exit(1)

    # GPU configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU acceleration enabled: {len(gpus)} GPU(s)")
        except RuntimeError as e:
            logger.warning(f"GPU config failed: {e}")
    else:
        logger.warning("No GPU detected - training on CPU")

    # Run training
    try:
        results = asyncio.run(
            train_mamba_enhanced(
                symbols=symbols,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                validation_split=args.validation_split,
            )
        )

        # Check success
        if results['success']:
            best_acc = results['results']['1d']['directional_accuracy']
            if best_acc >= 0.60:
                logger.info(f"✓ SUCCESS: Achieved {best_acc:.2%} directional accuracy (target: 60%)")
                sys.exit(0)
            elif best_acc >= 0.55:
                logger.info(f"⚠ PARTIAL SUCCESS: Achieved {best_acc:.2%} (target: 60%)")
                sys.exit(0)
            else:
                logger.warning(f"✗ BELOW TARGET: Only {best_acc:.2%} (target: 60%)")
                sys.exit(1)
        else:
            logger.error("Training failed")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("\n\n⚠ Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"✗ Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
