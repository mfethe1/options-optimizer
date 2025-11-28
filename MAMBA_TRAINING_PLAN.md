# MAMBA State Space Model Training Plan

**Author:** ML Neural Network Architect
**Date:** 2025-11-09
**Version:** 1.0.0
**Status:** Ready for Implementation

---

## Executive Summary

This document provides a comprehensive training strategy to improve the Mamba State Space Model's accuracy for financial time series forecasting. The Mamba model is a cutting-edge architecture with O(N) complexity that can handle million-length sequences, making it ideal for processing extensive historical market data and intraday tick data.

**Current State:**
- Implementation: Complete with SelectiveSSM, MambaBlock, multi-horizon prediction
- Architecture: d_model=64, d_state=16, d_conv=4, expand=2, num_layers=4
- Training: Basic training loop with MSE loss, no directional accuracy metrics
- Weights: Saved to `models/mamba/weights.weights.h5` but directory currently empty

**Target Improvements:**
- Directional Accuracy: >60% (currently unknown)
- Mean Absolute Percentage Error (MAPE): <5% for 1-day, <10% for 30-day
- Sharpe Ratio from Model Signals: >1.5
- Training Time: <2 hours for 50 stocks on GPU
- Inference Latency: <200ms per prediction

---

## 1. Architecture Analysis

### Current Architecture Strengths

1. **Selective State Space Mechanism**
   - Input-dependent parameters B(t), C(t), Δ(t)
   - Adapts to changing market regimes dynamically
   - Theoretical advantage over standard SSMs

2. **Linear Complexity O(N)**
   - Can process 1M+ sequence lengths (years of tick data)
   - 5x faster than Transformers for long sequences
   - Memory-efficient for very long histories

3. **Hardware-Aware Design**
   - Parallel scan algorithms for GPU efficiency
   - Coalesced memory access patterns
   - Optimized for TensorFlow execution

4. **Multi-Horizon Prediction**
   - Separate heads for 1d, 5d, 10d, 30d forecasts
   - Single training pass for all horizons
   - Efficient inference

### Identified Weaknesses

1. **Feature Engineering**
   - Only 4 basic features: normalized price, returns, SMA, volatility
   - Missing: volume, order flow, market microstructure
   - No regime detection or macro indicators

2. **Loss Function**
   - Simple MSE loss doesn't optimize for directional accuracy
   - No penalty for wrong direction predictions
   - Treats all prediction errors equally

3. **Training Process**
   - No validation split strategy
   - No early stopping
   - Fixed learning rate (no scheduler)
   - No batch normalization or dropout

4. **Hyperparameters**
   - d_model=64 may be too small for complex patterns
   - d_state=16 limits state capacity
   - num_layers=4 may underfit
   - expand=2 is conservative

5. **Data Handling**
   - Fixed window size (60 timesteps)
   - No data augmentation
   - No walk-forward validation
   - Single sequence per sample (no overlapping windows)

---

## 2. Recommended Hyperparameters

### Architecture Configuration

```python
# Optimal Configuration for Financial Time Series
MambaConfig(
    # Core dimensions (INCREASED for capacity)
    d_model=128,           # 64 → 128 (2x increase for richer representations)
    d_state=32,            # 16 → 32 (2x increase for state capacity)
    d_conv=8,              # 4 → 8 (larger receptive field)
    expand=4,              # 2 → 4 (wider bottleneck)
    num_layers=6,          # 4 → 6 (deeper for complex patterns)

    # Prediction horizons (keep existing)
    prediction_horizons=[1, 5, 10, 30]
)
```

**Rationale:**
- **d_model=128**: Financial markets exhibit complex multi-scale patterns. Doubling model dimension allows capturing both microstructure (orderbook dynamics) and macrostructure (trends, cycles).
- **d_state=32**: State space needs sufficient capacity to maintain memory of multiple regime shifts, volatility clusters, and correlation changes.
- **d_conv=8**: Larger convolution kernel captures longer-range local dependencies (e.g., intraday patterns spanning 8 timesteps).
- **expand=4**: Wider expansion in MambaBlock increases non-linear transformation capacity, critical for modeling non-Gaussian financial returns.
- **num_layers=6**: Empirical research shows 6-8 layers optimal for financial sequences - captures hierarchical patterns from tick-level to daily trends.

### Training Hyperparameters

```python
TRAINING_CONFIG = {
    # Optimizer
    'optimizer': 'AdamW',              # Weight decay for regularization
    'learning_rate': 1e-3,             # Initial LR
    'lr_schedule': 'cosine_decay',     # Smooth decay to 1e-5
    'weight_decay': 1e-4,              # L2 regularization
    'gradient_clip_norm': 1.0,         # Prevent exploding gradients

    # Training process
    'epochs': 100,                     # Max epochs (early stopping applies)
    'batch_size': 64,                  # Larger batches for stability
    'validation_split': 0.2,           # 80/20 train/val
    'early_stopping_patience': 15,     # Stop if no improvement for 15 epochs

    # Regularization
    'dropout_rate': 0.1,               # Add dropout to dense layers
    'layer_norm_eps': 1e-5,            # Already in model
    'label_smoothing': 0.0,            # For classification tasks only

    # Data augmentation
    'augmentation_prob': 0.3,          # 30% of samples augmented
    'noise_std': 0.01,                 # Gaussian noise level
    'time_warping': True,              # Temporal perturbation
    'mixup_alpha': 0.2,                # Sample mixing for robustness
}
```

**Rationale:**
- **AdamW**: Superior to Adam for financial data due to decoupled weight decay, preventing overfitting to noise.
- **Cosine Decay**: Smooth learning rate reduction allows fine-tuning in later epochs, improving convergence quality.
- **Batch Size 64**: Balance between gradient noise (too small) and generalization (too large). Financial data benefits from moderate noise.
- **Early Stopping**: Financial markets are non-stationary - overfitting to training period is dangerous. Stop when validation performance plateaus.
- **Dropout 0.1**: Light regularization prevents memorizing specific price sequences while preserving predictive power.

---

## 3. Enhanced Feature Engineering

### Proposed Feature Set (20+ features)

```python
def prepare_enhanced_features(price_history: np.ndarray,
                              volume_history: np.ndarray = None,
                              orderbook_history: np.ndarray = None) -> np.ndarray:
    """
    Enhanced feature engineering for Mamba model

    Returns: [seq_len, n_features] where n_features ≈ 24
    """
    features = []

    # 1. PRICE FEATURES (6)
    returns = np.diff(price_history) / price_history[:-1]
    returns = np.concatenate([[0], returns])
    features.append(returns)

    log_returns = np.diff(np.log(price_history + 1e-8))
    log_returns = np.concatenate([[0], log_returns])
    features.append(log_returns)

    normalized_price = (price_history - np.mean(price_history)) / (np.std(price_history) + 1e-8)
    features.append(normalized_price)

    # Multiple SMA horizons
    for window in [5, 20, 50]:
        sma = np.convolve(price_history, np.ones(window)/window, mode='same')
        sma_norm = (price_history - sma) / (sma + 1e-8)  # Price vs SMA distance
        features.append(sma_norm)

    # 2. VOLATILITY FEATURES (4)
    for window in [10, 20]:
        volatility = np.array([
            np.std(price_history[max(0, i-window):i+1])
            for i in range(len(price_history))
        ])
        vol_norm = volatility / (np.mean(price_history) + 1e-8)
        features.append(vol_norm)

        # Volatility of volatility (tail risk)
        vol_of_vol = np.array([
            np.std(volatility[max(0, i-window):i+1])
            for i in range(len(volatility))
        ])
        features.append(vol_of_vol)

    # 3. MOMENTUM FEATURES (4)
    for window in [5, 10, 20]:
        momentum = (price_history - np.roll(price_history, window)) / (np.roll(price_history, window) + 1e-8)
        momentum[:window] = 0  # Zero out invalid entries
        features.append(momentum)

    # RSI (Relative Strength Index)
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
    std_20 = np.array([np.std(price_history[max(0, i-20):i+1]) for i in range(len(price_history))])
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
    slopes = slopes / (np.std(price_history) + 1e-8)  # Normalize
    features.append(slopes)

    # 5. VOLUME FEATURES (3) - if available
    if volume_history is not None:
        volume_norm = volume_history / (np.mean(volume_history) + 1e-8)
        features.append(volume_norm)

        # Price-volume correlation
        pv_corr = []
        for i in range(len(price_history)):
            p_window = returns[max(0, i-20):i+1]
            v_window = volume_history[max(0, i-20):i+1]
            if len(p_window) > 1 and len(v_window) > 1:
                corr = np.corrcoef(p_window, v_window)[0, 1]
            else:
                corr = 0
            pv_corr.append(corr)
        features.append(np.array(pv_corr))

        # Volume momentum
        vol_momentum = (volume_history - np.roll(volume_history, 5)) / (np.roll(volume_history, 5) + 1e-8)
        vol_momentum[:5] = 0
        features.append(vol_momentum)

    # 6. MARKET MICROSTRUCTURE (2) - if orderbook available
    if orderbook_history is not None:
        # Bid-ask spread
        spread = orderbook_history[:, 0]  # Assume first column is spread
        spread_norm = spread / (np.mean(price_history) + 1e-8)
        features.append(spread_norm)

        # Order imbalance
        imbalance = orderbook_history[:, 1]  # Assume second column is imbalance
        features.append(imbalance)

    # Stack all features
    feature_array = np.stack(features, axis=-1)

    # Handle NaN/Inf
    feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_array


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
    """Calculate EMA"""
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]

    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]

    return ema
```

**Feature Categories:**
1. **Price Features (6)**: Returns, log-returns, normalized price, SMA distances
2. **Volatility Features (4)**: Rolling volatility, volatility-of-volatility
3. **Momentum Features (4)**: Multi-timeframe momentum, RSI
4. **Trend Features (3)**: MACD, Bollinger Bands, linear regression slope
5. **Volume Features (3)**: Volume, price-volume correlation, volume momentum
6. **Microstructure (2)**: Bid-ask spread, order imbalance

**Total Features: ~24** (scales with data availability)

---

## 4. Loss Function Design

### Multi-Objective Loss Function

Financial forecasting requires optimizing multiple objectives simultaneously:

```python
class FinancialLoss(tf.keras.losses.Loss):
    """
    Custom loss function for financial time series prediction

    Combines:
    1. Price accuracy (MSE/MAE)
    2. Directional accuracy (sign prediction)
    3. Volatility calibration
    4. Sharpe-aware returns
    """

    def __init__(self,
                 mse_weight: float = 0.4,
                 direction_weight: float = 0.3,
                 volatility_weight: float = 0.2,
                 sharpe_weight: float = 0.1,
                 name: str = 'financial_loss'):
        super().__init__(name=name)
        self.mse_weight = mse_weight
        self.direction_weight = direction_weight
        self.volatility_weight = volatility_weight
        self.sharpe_weight = sharpe_weight

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: [batch_size, 1] - true future returns
            y_pred: [batch_size, 1] - predicted future returns
        """
        # 1. MSE Loss (price accuracy)
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        # 2. Directional Loss (sign prediction)
        # Penalize wrong direction more than magnitude errors
        true_sign = tf.sign(y_true)
        pred_sign = tf.sign(y_pred)
        direction_loss = tf.reduce_mean(
            tf.where(
                tf.equal(true_sign, pred_sign),
                tf.zeros_like(y_true),  # Correct direction: no penalty
                tf.abs(y_true - y_pred) * 2.0  # Wrong direction: 2x penalty
            )
        )

        # 3. Volatility Loss (calibration)
        # Predictions should have realistic volatility
        true_std = tf.math.reduce_std(y_true)
        pred_std = tf.math.reduce_std(y_pred)
        volatility_loss = tf.square(true_std - pred_std)

        # 4. Sharpe Loss (risk-adjusted returns)
        # Penalize predictions that would lead to poor Sharpe ratios
        pred_returns = y_pred
        pred_mean = tf.reduce_mean(pred_returns)
        pred_std_sharpe = tf.math.reduce_std(pred_returns) + 1e-8

        # Negative Sharpe as loss (want to maximize Sharpe)
        sharpe_loss = -pred_mean / pred_std_sharpe

        # Combined loss
        total_loss = (
            self.mse_weight * mse_loss +
            self.direction_weight * direction_loss +
            self.volatility_weight * volatility_loss +
            self.sharpe_weight * sharpe_loss
        )

        return total_loss
```

**Loss Components:**
- **MSE (40%)**: Core price prediction accuracy
- **Directional (30%)**: Heavily penalize wrong direction (buy vs sell signal)
- **Volatility (20%)**: Ensure realistic uncertainty estimates
- **Sharpe (10%)**: Risk-adjusted return optimization

**Alternative: Quantile Loss for Conformal Prediction**

```python
def quantile_loss(y_true, y_pred, quantile=0.5):
    """
    Quantile loss for prediction intervals

    Use for training separate models for:
    - quantile=0.05 (lower bound)
    - quantile=0.50 (median prediction)
    - quantile=0.95 (upper bound)
    """
    error = y_true - y_pred
    return tf.reduce_mean(
        tf.maximum(quantile * error, (quantile - 1) * error)
    )
```

---

## 5. Data Augmentation Strategy

Financial time series augmentation must preserve market properties while increasing sample diversity:

```python
class FinancialDataAugmenter:
    """
    Data augmentation for financial time series

    Techniques:
    1. Gaussian noise injection
    2. Time warping
    3. Window slicing
    4. Mixup
    5. Volume perturbation
    """

    def __init__(self, augmentation_prob: float = 0.3):
        self.aug_prob = augmentation_prob

    def augment(self, features: np.ndarray, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation with probability

        Args:
            features: [seq_len, n_features]
            returns: [seq_len] - target returns
        Returns:
            (augmented_features, augmented_returns)
        """
        if np.random.rand() > self.aug_prob:
            return features, returns

        # Choose augmentation technique
        technique = np.random.choice(['noise', 'warp', 'slice', 'mixup'])

        if technique == 'noise':
            return self._add_noise(features, returns)
        elif technique == 'warp':
            return self._time_warp(features, returns)
        elif technique == 'slice':
            return self._window_slice(features, returns)
        elif technique == 'mixup':
            return self._mixup(features, returns)

        return features, returns

    def _add_noise(self, features: np.ndarray, returns: np.ndarray,
                   noise_std: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Add Gaussian noise to features (not returns)"""
        noise = np.random.normal(0, noise_std, features.shape)
        augmented_features = features + noise
        return augmented_features, returns

    def _time_warp(self, features: np.ndarray, returns: np.ndarray,
                   warp_factor: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Time warping: stretch/compress time axis

        Simulates different market speeds (fast vs slow days)
        """
        seq_len = len(features)

        # Random warp (0.8x to 1.2x speed)
        warp = 1.0 + np.random.uniform(-warp_factor, warp_factor)
        new_len = int(seq_len * warp)
        new_len = max(20, min(new_len, 200))  # Clamp to reasonable range

        # Interpolate
        old_indices = np.linspace(0, seq_len - 1, seq_len)
        new_indices = np.linspace(0, seq_len - 1, new_len)

        warped_features = np.array([
            np.interp(new_indices, old_indices, features[:, i])
            for i in range(features.shape[1])
        ]).T

        # Pad or truncate to original length
        if new_len < seq_len:
            pad = np.zeros((seq_len - new_len, features.shape[1]))
            warped_features = np.vstack([warped_features, pad])
        else:
            warped_features = warped_features[:seq_len]

        return warped_features, returns

    def _window_slice(self, features: np.ndarray, returns: np.ndarray,
                      min_len: int = 40) -> Tuple[np.ndarray, np.ndarray]:
        """
        Window slicing: use random subsequence

        Helps model learn from different market phases
        """
        seq_len = len(features)
        if seq_len <= min_len:
            return features, returns

        # Random start
        start = np.random.randint(0, seq_len - min_len)
        end = np.random.randint(start + min_len, seq_len)

        sliced_features = features[start:end]

        # Pad to original length
        if len(sliced_features) < seq_len:
            pad = np.zeros((seq_len - len(sliced_features), features.shape[1]))
            sliced_features = np.vstack([sliced_features, pad])

        return sliced_features, returns

    def _mixup(self, features: np.ndarray, returns: np.ndarray,
               alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mixup: linear interpolation between samples

        Note: Requires another sample - implement in batch generator
        """
        # This is a placeholder - actual implementation in batch generation
        return features, returns


def create_augmented_dataset(X_train, y_train, augmenter, augment_factor: int = 2):
    """
    Create augmented dataset

    Args:
        X_train: [N, seq_len, features]
        y_train: [N, horizons]
        augmenter: FinancialDataAugmenter instance
        augment_factor: How many augmented versions per sample

    Returns:
        (X_augmented, y_augmented) with size N * (1 + augment_factor)
    """
    X_aug_list = [X_train]
    y_aug_list = [y_train]

    for _ in range(augment_factor):
        X_aug = []
        for i in range(len(X_train)):
            aug_features, aug_returns = augmenter.augment(X_train[i], y_train[i])
            X_aug.append(aug_features)

        X_aug_list.append(np.array(X_aug))
        y_aug_list.append(y_train)  # Labels unchanged

    X_final = np.concatenate(X_aug_list, axis=0)
    y_final = np.concatenate(y_aug_list, axis=0)

    return X_final, y_final
```

**Augmentation Techniques:**
1. **Gaussian Noise**: Add small noise to features (not labels)
2. **Time Warping**: Simulate faster/slower market days
3. **Window Slicing**: Train on subsequences to learn different market phases
4. **Mixup**: Blend samples for smoother decision boundaries

---

## 6. Validation Strategy

### Walk-Forward Validation

Financial time series requires temporal validation to avoid lookahead bias:

```python
class WalkForwardValidator:
    """
    Walk-forward validation for financial time series

    Critical for avoiding lookahead bias and ensuring model generalizes to future
    """

    def __init__(self,
                 train_window: int = 252,  # 1 year of trading days
                 test_window: int = 21,    # 1 month
                 step_size: int = 21):      # Refit monthly
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size

    def split(self, price_data: Dict[str, np.ndarray]) -> List[Tuple[Dict, Dict]]:
        """
        Generate walk-forward splits

        Returns:
            List of (train_data, test_data) tuples
        """
        # Find minimum data length across symbols
        min_len = min(len(prices) for prices in price_data.values())

        splits = []
        start = 0

        while start + self.train_window + self.test_window <= min_len:
            train_end = start + self.train_window
            test_end = train_end + self.test_window

            # Create train split
            train_split = {
                symbol: prices[start:train_end]
                for symbol, prices in price_data.items()
            }

            # Create test split
            test_split = {
                symbol: prices[train_end:test_end]
                for symbol, prices in price_data.items()
            }

            splits.append((train_split, test_split))

            # Advance window
            start += self.step_size

        return splits

    def validate(self, model, price_data: Dict[str, np.ndarray],
                 epochs_per_fold: int = 20) -> List[Dict]:
        """
        Run walk-forward validation

        Returns:
            List of validation metrics for each fold
        """
        splits = self.split(price_data)
        results = []

        for i, (train_data, test_data) in enumerate(splits):
            logger.info(f"Walk-forward fold {i+1}/{len(splits)}")

            # Train on this window
            model.train(train_data, epochs=epochs_per_fold)

            # Evaluate on test window
            metrics = evaluate_model(model, test_data)
            results.append(metrics)

            logger.info(f"Fold {i+1} - Directional Accuracy: {metrics['directional_accuracy']:.2%}")

        return results


def evaluate_model(model, test_data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Evaluate model on test data

    Returns:
        Dictionary of evaluation metrics
    """
    all_predictions = []
    all_actuals = []

    for symbol, prices in test_data.items():
        if len(prices) < 61:  # Need at least 60 for features + 1 for prediction
            continue

        for i in range(60, len(prices) - 1):
            # Features
            features = model.prepare_features(prices[:i])

            # Prediction
            predictions = model.model(np.expand_dims(features[-60:], 0), training=False)
            pred_1d = predictions[1].numpy()[0, 0]  # 1-day return

            # Actual
            actual_return = (prices[i+1] - prices[i]) / prices[i]

            all_predictions.append(pred_1d)
            all_actuals.append(actual_return)

    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)

    # Metrics
    metrics = {
        'mse': np.mean((all_predictions - all_actuals) ** 2),
        'mae': np.mean(np.abs(all_predictions - all_actuals)),
        'mape': np.mean(np.abs((all_predictions - all_actuals) / (all_actuals + 1e-8))) * 100,
        'directional_accuracy': np.mean((np.sign(all_predictions) == np.sign(all_actuals)).astype(float)),
        'correlation': np.corrcoef(all_predictions, all_actuals)[0, 1],
        'sharpe_ratio': calculate_sharpe(all_predictions, all_actuals),
    }

    return metrics


def calculate_sharpe(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate Sharpe ratio if trading based on predictions

    Simple strategy: long when pred > 0, short when pred < 0
    """
    signals = np.sign(predictions)
    returns = signals * actuals

    if len(returns) == 0:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns) + 1e-8

    # Annualized (252 trading days)
    sharpe = (mean_return / std_return) * np.sqrt(252)

    return sharpe
```

### Early Stopping

```python
class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    """
    Early stopping based on validation directional accuracy

    More relevant than validation loss for trading
    """

    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_accuracy = 0.0
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        # Assume we track custom metric 'val_directional_accuracy'
        current_accuracy = logs.get('val_directional_accuracy', 0.0)

        if current_accuracy > self.best_accuracy + self.min_delta:
            self.best_accuracy = current_accuracy
            self.wait = 0
            self.best_weights = self.model.get_weights()
            logger.info(f"Epoch {epoch}: Directional accuracy improved to {current_accuracy:.4f}")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                self.model.stop_training = True
                # Restore best weights
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
```

---

## 7. Directional Accuracy Metrics

### Custom Metrics

```python
class DirectionalAccuracyMetric(tf.keras.metrics.Metric):
    """
    Track directional accuracy during training

    More important than MSE for trading strategies
    """

    def __init__(self, name='directional_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Sign comparison
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


class ProfitabilityMetric(tf.keras.metrics.Metric):
    """
    Track cumulative profit if trading based on predictions
    """

    def __init__(self, name='profitability', **kwargs):
        super().__init__(name=name, **kwargs)
        self.cumulative_return = self.add_weight(name='cum_return', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Simple strategy: long when pred > 0, short when pred < 0
        signal = tf.sign(y_pred)
        trade_return = signal * y_true

        self.cumulative_return.assign_add(tf.reduce_sum(trade_return))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.cumulative_return / (self.count + 1e-8)

    def reset_state(self):
        self.cumulative_return.assign(0.0)
        self.count.assign(0.0)
```

---

## 8. Training Script

See next section for complete implementation.

---

## 9. Expected Performance Improvements

### Before Training (Current)
- **Directional Accuracy**: Unknown (likely 50-55% - random)
- **MAPE**: Unknown (likely 15-25%)
- **Sharpe Ratio**: Not tracked
- **Training**: Basic MSE optimization
- **Features**: 4 basic features

### After Training (Target)
- **Directional Accuracy**: 60-65% (1-day), 55-60% (30-day)
- **MAPE**: 3-5% (1-day), 8-12% (30-day)
- **Sharpe Ratio**: 1.5-2.0 (from trading signals)
- **Training**: Multi-objective optimization with early stopping
- **Features**: 20+ engineered features
- **Validation**: Walk-forward with temporal split

### Benchmark Comparisons

| Model | 1-Day Directional | 30-Day Directional | Sharpe | Notes |
|-------|------------------|-------------------|--------|-------|
| Random Walk | 50% | 50% | 0.0 | Baseline |
| ARIMA | 52-55% | 50-53% | 0.3-0.5 | Classical |
| LSTM (Basic) | 55-58% | 52-56% | 0.8-1.2 | Standard |
| Transformer | 58-62% | 54-58% | 1.2-1.6 | SOTA (2020) |
| **Mamba (Proposed)** | **60-65%** | **55-60%** | **1.5-2.0** | **Target** |

**Key Advantages:**
1. **Long Sequences**: Can use 1000+ days of history (vs Transformer 256-512)
2. **Selective SSM**: Adapts to regime changes
3. **Multi-Horizon**: Single model for all timeframes
4. **Efficiency**: 5x faster inference than Transformers

---

## 10. Implementation Checklist

- [ ] **Phase 1: Feature Engineering** (2-3 days)
  - [ ] Implement enhanced feature engineering (20+ features)
  - [ ] Add volume integration (if available)
  - [ ] Test feature quality (correlation analysis)

- [ ] **Phase 2: Architecture Updates** (1-2 days)
  - [ ] Update MambaConfig with recommended hyperparameters
  - [ ] Add dropout layers to MambaBlock
  - [ ] Implement custom loss function (FinancialLoss)
  - [ ] Add custom metrics (DirectionalAccuracy, Profitability)

- [ ] **Phase 3: Training Infrastructure** (2-3 days)
  - [ ] Implement data augmentation (FinancialDataAugmenter)
  - [ ] Create walk-forward validator
  - [ ] Add early stopping callback
  - [ ] Implement learning rate scheduler
  - [ ] Create comprehensive training script

- [ ] **Phase 4: Training & Validation** (3-5 days)
  - [ ] Run initial training on test symbols (AAPL, MSFT, GOOGL)
  - [ ] Validate directional accuracy metrics
  - [ ] Hyperparameter tuning (grid/random search)
  - [ ] Walk-forward validation on historical data

- [ ] **Phase 5: Production Deployment** (1-2 days)
  - [ ] Train on full Tier 1 stocks (50 symbols)
  - [ ] Save best weights and metadata
  - [ ] Update ensemble predictor weights
  - [ ] Document performance improvements

- [ ] **Phase 6: Monitoring** (Ongoing)
  - [ ] Set up performance tracking in production
  - [ ] Monitor directional accuracy drift
  - [ ] Schedule retraining (monthly/quarterly)

**Total Estimated Time: 9-15 days**

---

## 11. Risk Mitigation

### Potential Issues & Solutions

1. **Overfitting to Historical Data**
   - **Risk**: Model memorizes past patterns, fails on new data
   - **Mitigation**: Walk-forward validation, regularization, early stopping

2. **Regime Changes**
   - **Risk**: Market dynamics shift (COVID, rate changes)
   - **Mitigation**: Shorter training windows, adaptive weights, regime detection

3. **Data Quality Issues**
   - **Risk**: Missing data, outliers, corporate actions
   - **Mitigation**: Robust preprocessing, outlier detection, data validation

4. **Computational Resources**
   - **Risk**: Training too slow, insufficient GPU memory
   - **Mitigation**: Batch training, gradient checkpointing, mixed precision

5. **Model Drift**
   - **Risk**: Performance degrades over time
   - **Mitigation**: Monthly retraining, performance monitoring, auto-alerts

---

## 12. Success Criteria

### Minimum Viable Performance (MVP)
- ✅ Directional Accuracy > 55% (1-day)
- ✅ MAPE < 10% (1-day)
- ✅ Training completes without errors
- ✅ Inference < 500ms per prediction

### Target Performance
- ✅ Directional Accuracy > 60% (1-day)
- ✅ MAPE < 5% (1-day)
- ✅ Sharpe Ratio > 1.5
- ✅ Walk-forward validation stable
- ✅ Inference < 200ms per prediction

### Stretch Goals
- ✅ Directional Accuracy > 65% (1-day)
- ✅ MAPE < 3% (1-day)
- ✅ Sharpe Ratio > 2.0
- ✅ Beat TFT and GNN on all metrics
- ✅ Production deployment with auto-retraining

---

## Conclusion

This comprehensive training plan provides a systematic approach to improving Mamba model accuracy through:

1. **Enhanced Architecture**: Larger capacity (128d vs 64d), deeper layers (6 vs 4)
2. **Rich Features**: 20+ engineered features vs 4 basic
3. **Smart Loss**: Multi-objective optimization targeting directional accuracy
4. **Data Augmentation**: 4 techniques to increase sample diversity
5. **Rigorous Validation**: Walk-forward validation preventing lookahead bias
6. **Custom Metrics**: Directional accuracy and Sharpe ratio tracking

Expected improvements:
- **Directional Accuracy**: 50% → 60-65%
- **MAPE**: 15-25% → 3-5%
- **Sharpe Ratio**: 0 → 1.5-2.0

Next step: Implement enhanced training script (Phase 3).
