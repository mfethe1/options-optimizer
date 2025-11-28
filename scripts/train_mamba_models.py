#!/usr/bin/env python3
"""
Mamba State Space Model Training Script

Trains Mamba models for multi-horizon stock price forecasting with:
- Advanced data preprocessing and augmentation
- Early stopping and checkpoint management
- TensorBoard integration for monitoring
- Learning rate scheduling
- Comprehensive metrics tracking

Usage:
    python scripts/train_mamba_models.py --symbols TIER_1
    python scripts/train_mamba_models.py --symbols AAPL,MSFT,GOOGL
    python scripts/train_mamba_models.py --test  # Quick test with 3 symbols

Performance Targets:
    - Model inference: <1s per prediction
    - Training time: ~30-50s per symbol (20 epochs)
    - Linear O(N) complexity for long sequences
"""

import sys
import os
import argparse
import asyncio
import logging
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import TensorFlow first (Windows DLL fix)
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    tf = None
    keras = None
    TENSORFLOW_AVAILABLE = False

import numpy as np

# Import from project
from src.ml.state_space.mamba_model import (
    MambaPredictor,
    MambaConfig,
    MambaModel
)
from src.api.ml_integration_helpers import (
    fetch_historical_prices,
)
from src.utils.file_locking import (
    file_lock,
    atomic_json_write,
    atomic_model_save,
    FileLockTimeoutError
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
    # Mega-cap tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    # Financials
    'JPM', 'BAC', 'GS', 'MS', 'BLK', 'C', 'WFC',
    # Healthcare
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

# Test subset
TEST_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL']


class DataPreprocessor:
    """
    Advanced data preprocessing for Mamba training

    Features:
    - Multi-scale feature engineering
    - Data augmentation
    - Sequence windowing
    - Normalization strategies
    """

    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizons: List[int] = None,
        augment: bool = True
    ):
        """
        Args:
            sequence_length: Input sequence length
            prediction_horizons: Forecast horizons
            augment: Enable data augmentation
        """
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons or [1, 5, 10, 30]
        self.augment = augment

    def prepare_features(self, price_history: np.ndarray) -> np.ndarray:
        """
        Prepare advanced features from price history

        Args:
            price_history: [seq_len] array of prices
        Returns:
            features: [seq_len, n_features] array
        """
        # Calculate returns
        returns = np.diff(price_history) / (price_history[:-1] + 1e-8)
        returns = np.concatenate([[0], returns])

        # Multi-scale moving averages
        sma_5 = self._moving_average(price_history, 5)
        sma_20 = self._moving_average(price_history, 20)
        sma_60 = self._moving_average(price_history, 60)

        # Volatility (rolling std)
        vol_5 = self._rolling_std(returns, 5)
        vol_20 = self._rolling_std(returns, 20)

        # Momentum indicators
        momentum_5 = self._momentum(price_history, 5)
        momentum_20 = self._momentum(price_history, 20)

        # Price position (where price is in recent range)
        price_position = self._price_position(price_history, 20)

        # Normalized price
        price_norm = self._normalize(price_history)

        # Combine features
        features = np.stack([
            price_norm,
            returns,
            sma_5 / (np.mean(price_history) + 1e-8),
            sma_20 / (np.mean(price_history) + 1e-8),
            sma_60 / (np.mean(price_history) + 1e-8),
            vol_5,
            vol_20,
            momentum_5,
            momentum_20,
            price_position
        ], axis=-1)

        return features

    def create_sequences(
        self,
        price_history: np.ndarray
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Create training sequences with sliding window

        Args:
            price_history: Historical prices
        Returns:
            (X, y_dict) where:
                X: [n_samples, seq_len, n_features]
                y_dict: {horizon: [n_samples] targets}
        """
        features = self.prepare_features(price_history)

        X = []
        y = {h: [] for h in self.prediction_horizons}

        max_horizon = max(self.prediction_horizons)

        for i in range(self.sequence_length, len(price_history) - max_horizon):
            # Input sequence
            X.append(features[i-self.sequence_length:i])

            # Target returns for each horizon
            current_price = price_history[i]
            for horizon in self.prediction_horizons:
                if i + horizon < len(price_history):
                    future_price = price_history[i + horizon]
                    future_return = (future_price - current_price) / current_price
                    y[horizon].append(future_return)
                else:
                    y[horizon].append(0.0)

        X = np.array(X)
        y = {h: np.array(vals).reshape(-1, 1) for h, vals in y.items()}

        return X, y

    def augment_data(
        self,
        X: np.ndarray,
        y: Dict[int, np.ndarray],
        augment_factor: float = 0.2
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Data augmentation with noise injection

        Args:
            X: Input sequences
            y: Target dict
            augment_factor: Augmentation strength
        Returns:
            Augmented (X, y)
        """
        if not self.augment:
            return X, y

        # Add Gaussian noise to features
        noise = np.random.normal(0, augment_factor, X.shape)
        X_aug = X + noise * np.std(X, axis=(0, 1), keepdims=True)

        # Clip to reasonable range
        X_aug = np.clip(X_aug, -10, 10)

        return X_aug, y

    @staticmethod
    def _moving_average(data: np.ndarray, window: int) -> np.ndarray:
        """Calculate moving average"""
        if len(data) < window:
            return np.ones_like(data) * np.mean(data)
        return np.convolve(data, np.ones(window)/window, mode='same')

    @staticmethod
    def _rolling_std(data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling standard deviation"""
        result = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - window)
            result[i] = np.std(data[start:i+1])
        return result

    @staticmethod
    def _momentum(data: np.ndarray, window: int) -> np.ndarray:
        """Calculate momentum (rate of change)"""
        result = np.zeros_like(data)
        for i in range(len(data)):
            if i >= window:
                result[i] = (data[i] - data[i-window]) / (data[i-window] + 1e-8)
        return result

    @staticmethod
    def _price_position(data: np.ndarray, window: int) -> np.ndarray:
        """Calculate where price is in recent range (0-1)"""
        result = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - window)
            window_data = data[start:i+1]
            min_val = np.min(window_data)
            max_val = np.max(window_data)
            if max_val > min_val:
                result[i] = (data[i] - min_val) / (max_val - min_val)
            else:
                result[i] = 0.5
        return result

    @staticmethod
    def _normalize(data: np.ndarray) -> np.ndarray:
        """Z-score normalization"""
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / (std + 1e-8)


class TrainingCallbacks:
    """
    Custom callbacks for training monitoring and control
    """

    def __init__(
        self,
        save_dir: str,
        symbol: str,
        patience: int = 5,
        min_delta: float = 0.0001
    ):
        """
        Args:
            save_dir: Directory for saving checkpoints
            symbol: Stock symbol
            patience: Early stopping patience
            min_delta: Minimum improvement to count as progress
        """
        self.save_dir = Path(save_dir)
        self.symbol = symbol
        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float('inf')
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0

        # Create directories
        self.checkpoint_dir = self.save_dir / 'checkpoints' / symbol
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_keras_callbacks(self) -> List:
        """Get Keras callbacks"""
        if not TENSORFLOW_AVAILABLE:
            return []

        callbacks = []

        # Model checkpoint
        checkpoint_path = str(self.checkpoint_dir / 'best_model.weights.h5')
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                mode='min',
                verbose=0
            )
        )

        # Early stopping
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                min_delta=self.min_delta,
                restore_best_weights=True,
                verbose=0
            )
        )

        # Learning rate reduction
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=0
            )
        )

        # TensorBoard (optional - can be memory intensive)
        # Uncomment if needed:
        # log_dir = self.save_dir / 'logs' / self.symbol / datetime.now().strftime('%Y%m%d-%H%M%S')
        # callbacks.append(
        #     keras.callbacks.TensorBoard(
        #         log_dir=str(log_dir),
        #         histogram_freq=1,
        #         write_graph=False
        #     )
        # )

        return callbacks


class MetricsTracker:
    """
    Track and persist training metrics
    """

    def __init__(self, save_dir: str, symbol: str):
        """
        Args:
            save_dir: Directory for saving metrics
            symbol: Stock symbol
        """
        self.save_dir = Path(save_dir)
        self.symbol = symbol
        self.metrics = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'learning_rate': [],
            'epoch_time': []
        }

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_mae: float,
        val_mae: float,
        lr: float,
        epoch_time: float
    ):
        """Update metrics for epoch"""
        self.metrics['epochs'].append(epoch)
        self.metrics['train_loss'].append(float(train_loss))
        self.metrics['val_loss'].append(float(val_loss))
        self.metrics['train_mae'].append(float(train_mae))
        self.metrics['val_mae'].append(float(val_mae))
        self.metrics['learning_rate'].append(float(lr))
        self.metrics['epoch_time'].append(float(epoch_time))

    def save(self):
        """Save metrics to JSON with atomic write"""
        metrics_dir = self.save_dir / 'metrics'
        metrics_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = metrics_dir / f'{self.symbol}_metrics.json'
        
        # Use atomic write to prevent corruption
        atomic_json_write(str(metrics_path), self.metrics, indent=2)
        
        logger.info(f"[{self.symbol}] Saved training metrics to {metrics_path}")

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        if not self.metrics['epochs']:
            return {}

        return {
            'total_epochs': len(self.metrics['epochs']),
            'best_val_loss': float(np.min(self.metrics['val_loss'])),
            'best_epoch': int(np.argmin(self.metrics['val_loss'])) + 1,
            'final_train_loss': self.metrics['train_loss'][-1],
            'final_val_loss': self.metrics['val_loss'][-1],
            'avg_epoch_time': float(np.mean(self.metrics['epoch_time'])),
            'total_time': float(np.sum(self.metrics['epoch_time']))
        }



def save_model_artifacts_atomic(
    symbol: str,
    model,
    metrics_tracker,
    summary: Dict[str, Any],
    prices: np.ndarray,
    config: MambaConfig,
    X_train: np.ndarray,
    X_val: np.ndarray,
    save_dir: str,
    epochs: int,
    batch_size: int,
    validation_split: float,
    historical_days: int
) -> Dict[str, str]:
    """
    Save model artifacts with atomic file locking to prevent race conditions
    
    Args:
        symbol: Stock symbol
        model: Trained Mamba model
        metrics_tracker: MetricsTracker instance
        summary: Training summary dict
        prices: Historical price data
        config: MambaConfig
        X_train: Training features
        X_val: Validation features
        save_dir: Base directory for saving
        epochs: Number of training epochs
        batch_size: Batch size used
        validation_split: Validation split fraction
        historical_days: Historical days used
        
    Returns:
        Dict with file paths of saved artifacts
        
    Notes:
        - Uses file locking to prevent concurrent writes
        - All saves are atomic (temp file + rename)
        - Safe for parallel training with multiple workers
    """
    base_path = Path(save_dir) / symbol
    
    try:
        # Use file lock for entire save operation (60s timeout for large models)
        with file_lock(str(base_path / 'save.lock'), timeout=60):
            logger.debug(f"[{symbol}] Acquired lock for atomic save")
            
            # 1. Save model weights atomically
            weights_path = Path(save_dir) / 'weights' / f'{symbol}.weights.h5'
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use atomic_model_save with model.save_weights
            atomic_model_save(
                model,
                str(weights_path),
                save_func=lambda m, p: m.save_weights(p),
                lock_timeout=30  # Inner lock timeout (should acquire immediately since we have outer lock)
            )
            logger.info(f"[{symbol}] Saved weights: {weights_path}")
            
            # 2. Save metrics atomically
            metrics_tracker.save()  # This already uses atomic_json_write internally
            metrics_path = Path(save_dir) / 'metrics' / f'{symbol}_metrics.json'
            logger.info(f"[{symbol}] Saved metrics: {metrics_path}")
            
            # 3. Save metadata atomically
            metadata = {
                'symbol': symbol,
                'training_date': datetime.now().isoformat(),
                'data_version': '1.0.0',
                'model_config': {
                    'd_model': config.d_model,
                    'd_state': config.d_state,
                    'd_conv': config.d_conv,
                    'expand': config.expand,
                    'num_layers': config.num_layers,
                    'prediction_horizons': config.prediction_horizons
                },
                'training_config': {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'validation_split': validation_split,
                    'historical_days': historical_days,
                    'sequence_length': 60,
                    'num_features': X_train.shape[2],
                    'num_training_samples': len(X_train),
                    'num_validation_samples': len(X_val)
                },
                'performance_metrics': summary,
                'data_stats': {
                    'num_samples': len(X_train) + len(X_val),
                    'price_range': [float(np.min(prices)), float(np.max(prices))],
                    'avg_return': float(np.mean(np.diff(prices) / prices[:-1]))
                },
                'files': {
                    'weights': str(weights_path),
                    'metrics': str(metrics_path)
                }
            }
            
            metadata_path = Path(save_dir) / 'metadata' / f'{symbol}_metadata.json'
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            atomic_json_write(str(metadata_path), metadata, indent=2, lock_timeout=30)
            logger.info(f"[{symbol}] Saved metadata: {metadata_path}")
            
            logger.debug(f"[{symbol}] Released lock after successful atomic save")
            
            return {
                'weights': str(weights_path),
                'metrics': str(metrics_path),
                'metadata': str(metadata_path)
            }
            
    except FileLockTimeoutError as e:
        logger.error(f"[{symbol}] Lock timeout during save: {e}")
        raise
    except Exception as e:
        logger.error(f"[{symbol}] Error during atomic save: {e}")
        raise


async def train_single_symbol(
    symbol: str,
    save_dir: str = 'models/mamba',
    historical_days: int = 1000,
    epochs: int = 50,
    batch_size: int = 32,
    validation_split: float = 0.2,
    config: Optional[MambaConfig] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Train Mamba model for single symbol with advanced features

    Args:
        symbol: Stock symbol
        save_dir: Base directory for saving
        historical_days: Days of historical data
        epochs: Training epochs
        batch_size: Batch size
        validation_split: Validation data fraction
        config: Mamba configuration

    Returns:
        (symbol, result_dict)
    """
    start_time = time.time()
    logger.info(f"[{symbol}] Starting Mamba training...")

    try:
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available")

        # 1. Fetch historical data
        logger.info(f"[{symbol}] Fetching {historical_days} days of price data...")
        price_data = await fetch_historical_prices([symbol], days=historical_days)

        if symbol not in price_data or len(price_data[symbol]) < 60:
            raise ValueError(f"Insufficient price data for {symbol}: {len(price_data.get(symbol, []))} days")

        prices = price_data[symbol]
        logger.info(f"[{symbol}] Loaded {len(prices)} days of price data")

        # 2. Initialize components
        config = config or MambaConfig()
        preprocessor = DataPreprocessor(
            sequence_length=60,
            prediction_horizons=config.prediction_horizons,
            augment=True
        )

        # 3. Create training sequences
        logger.info(f"[{symbol}] Creating training sequences...")
        X, y = preprocessor.create_sequences(prices)
        logger.info(f"[{symbol}] Created {len(X)} training samples")

        # 4. Split train/validation BEFORE augmentation (CRITICAL FIX for data leakage)
        # Previously augmentation happened before split, causing validation contamination
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train = {h: v[:split_idx] for h, v in y.items()}
        y_val = {h: v[split_idx:] for h, v in y.items()}

        logger.info(f"[{symbol}] Split: {len(X_train)} train, {len(X_val)} validation samples")

        # 5. Data augmentation ONLY on training set (prevent validation leakage)
        X_train, y_train = preprocessor.augment_data(X_train, y_train, augment_factor=0.1)
        logger.info(f"[{symbol}] After augmentation: {len(X_train)} training samples")

        # 6. Build model
        logger.info(f"[{symbol}] Building Mamba model...")
        model = MambaModel(config)

        # Build with input shape
        model.build((None, X_train.shape[1], X_train.shape[2]))

        # Compile with custom optimizer
        optimizer = keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )

        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )

        logger.info(f"[{symbol}] Model has {model.count_params():,} parameters")

        # 7. Setup callbacks and metrics
        callbacks_manager = TrainingCallbacks(
            save_dir=save_dir,
            symbol=symbol,
            patience=5
        )

        metrics_tracker = MetricsTracker(
            save_dir=save_dir,
            symbol=symbol
        )

        # 8. Train model with explicit validation data (NO validation_split!)
        logger.info(f"[{symbol}] Training for {epochs} epochs...")

        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),  # FIXED: explicit validation data
            callbacks=callbacks_manager.get_keras_callbacks(),
            verbose=0
        )

        # 8. Track metrics
        for epoch in range(len(history.history['loss'])):
            metrics_tracker.update(
                epoch=epoch + 1,
                train_loss=history.history['loss'][epoch],
                val_loss=history.history['val_loss'][epoch],
                train_mae=history.history.get('mae', [0])[epoch],
                val_mae=history.history.get('val_mae', [0])[epoch],
                lr=float(keras.backend.get_value(model.optimizer.learning_rate)),
                epoch_time=0  # Would need custom callback for per-epoch timing
            )

        summary = metrics_tracker.get_summary()

        # 9-10. Save all artifacts atomically with file locking
        # CRITICAL: Prevents race conditions when parallel workers write to same files
        saved_files = save_model_artifacts_atomic(
            symbol=symbol,
            model=model,
            metrics_tracker=metrics_tracker,
            summary=summary,
            prices=prices,
            config=config,
            X_train=X_train,
            X_val=X_val,
            save_dir=save_dir,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            historical_days=historical_days
        )
        
        # Construct metadata for return value
        metadata = {
            'symbol': symbol,
            'training_date': datetime.now().isoformat(),
            'data_version': '1.0.0',
            'files': saved_files
        }

        elapsed = time.time() - start_time
        logger.info(
            f"[{symbol}] ✓ Training complete! "
            f"Best Val Loss: {summary['best_val_loss']:.4f}, "
            f"Time: {elapsed:.1f}s"
        )

        return symbol, {
            'success': True,
            'elapsed_seconds': elapsed,
            'metadata': metadata,
            **summary
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
    save_dir: str = 'models/mamba',
    parallel: int = 5,
    **kwargs
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Train Mamba models for multiple symbols in parallel

    Args:
        symbols: List of stock symbols
        save_dir: Base directory for saving
        parallel: Number of concurrent training jobs
        **kwargs: Additional arguments for train_single_symbol

    Returns:
        List of (symbol, result) tuples
    """
    logger.info("=" * 70)
    logger.info(f"Mamba Training: {len(symbols)} symbols ({parallel} parallel workers)")
    logger.info("=" * 70)

    # Create directories
    for subdir in ['weights', 'metadata', 'metrics', 'checkpoints']:
        os.makedirs(os.path.join(save_dir, subdir), exist_ok=True)

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
            *[train_single_symbol(sym, save_dir, **kwargs) for sym in batch],
            return_exceptions=True
        )

        # Handle exceptions
        for sym, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                logger.error(f"[{sym}] Exception: {result}")
                results.append((sym, {'success': False, 'error': str(result)}))
            else:
                results.append(result)

        batch_elapsed = time.time() - batch_start
        success_count = sum(1 for _, r in batch_results if not isinstance(r, Exception) and r['success'])
        logger.info(f"[Batch {batch_num}/{total_batches}] Complete: {success_count}/{len(batch)} successful, {batch_elapsed:.1f}s")

    # Summary
    total_elapsed = time.time() - total_start
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

    if successful:
        avg_val_loss = np.mean([r['best_val_loss'] for s, r in results if r['success']])
        logger.info(f"Avg best val loss: {avg_val_loss:.4f}")

    if failed:
        logger.info("")
        logger.info("Failed symbols:")
        for symbol in failed:
            error = next(r['error'] for s, r in results if s == symbol and not r['success'])
            logger.info(f"  - {symbol}: {error}")

    logger.info("")
    logger.info("✓ Mamba training complete!")
    logger.info("=" * 70)

    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Train Mamba models for stock prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all Tier 1 stocks
  python scripts/train_mamba_models.py --symbols TIER_1

  # Train specific symbols
  python scripts/train_mamba_models.py --symbols AAPL,MSFT,GOOGL

  # Quick test
  python scripts/train_mamba_models.py --test

  # Custom settings
  python scripts/train_mamba_models.py --symbols TIER_1 --epochs 100 --batch-size 64
        """
    )

    parser.add_argument('--symbols', default='TEST', help='Comma-separated symbols or TIER_1')
    parser.add_argument('--test', action='store_true', help='Quick test with 3 symbols')
    parser.add_argument('--parallel', type=int, default=5, help='Parallel workers')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--days', type=int, default=1000, help='Historical days')
    parser.add_argument('--save-dir', default='models/mamba', help='Save directory')

    args = parser.parse_args()

    # Determine symbols
    if args.test:
        symbols = TEST_SYMBOLS
    elif args.symbols == 'TIER_1':
        symbols = TIER_1_STOCKS
    elif args.symbols == 'TEST':
        symbols = TEST_SYMBOLS
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]

    logger.info(f"Training {len(symbols)} Mamba models...")
    logger.info(f"Symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")

    # Check TensorFlow
    if not TENSORFLOW_AVAILABLE:
        logger.error("TensorFlow not available! Install with: pip install tensorflow")
        sys.exit(1)

    # GPU configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU enabled: {len(gpus)} GPU(s)")
        except RuntimeError as e:
            logger.warning(f"GPU config failed: {e}")
    else:
        logger.warning("No GPU - using CPU")

    # Run training
    try:
        results = asyncio.run(
            train_batch(
                symbols=symbols,
                save_dir=args.save_dir,
                parallel=args.parallel,
                epochs=args.epochs,
                batch_size=args.batch_size,
                historical_days=args.days
            )
        )

        successful = sum(1 for _, r in results if r['success'])
        success_rate = successful / len(results)

        if success_rate >= 0.8:
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("\n\n⚠ Training interrupted")
        sys.exit(130)
    except Exception as e:
        logger.error(f"✗ Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
