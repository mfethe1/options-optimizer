# Mamba State Space Model Training Guide

**Comprehensive guide for training Mamba models for multi-horizon stock price forecasting**

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Training Pipeline](#training-pipeline)
- [Advanced Features](#advanced-features)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

---

## Overview

The Mamba State Space Model is a cutting-edge architecture for time series forecasting with **linear O(N) complexity**, making it ideal for processing very long sequences (years of tick data) that would be infeasible for traditional Transformers.

### Key Advantages

- **Linear Complexity**: O(N) vs Transformers O(N²)
- **Long Sequences**: Handle million+ length sequences
- **Selective Mechanism**: Input-dependent parameters for adaptive modeling
- **Hardware-Aware**: Optimized for GPU parallel processing
- **Multi-Horizon**: Single model predicts multiple time horizons

### Research Foundation

Based on "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)

---

## Architecture

### Model Components

```
Input Sequence [batch, seq_len, features]
    ↓
Embedding Layer [d_model=64]
    ↓
MambaBlock × N [num_layers=4]
    ├─ LayerNorm
    ├─ Conv1D (causal, depthwise)
    ├─ SelectiveSSM (d_state=16)
    │   ├─ B(t) = Linear(x(t))  [input-dependent]
    │   ├─ C(t) = Linear(x(t))  [input-dependent]
    │   ├─ Δ(t) = softplus(Linear(x(t)))  [adaptive time step]
    │   └─ h(t) = A·h(t-1) + B(t)·x(t)
    └─ Residual Connection
    ↓
Final LayerNorm
    ↓
Multi-Head Predictions [1d, 5d, 10d, 30d]
```

### Configuration

```python
@dataclass
class MambaConfig:
    d_model: int = 64          # Model dimension
    d_state: int = 16          # SSM state dimension
    d_conv: int = 4            # Convolution kernel size
    expand: int = 2            # Expansion factor
    num_layers: int = 4        # Number of Mamba blocks
    prediction_horizons: List[int] = [1, 5, 10, 30]
```

---

## Quick Start

### Basic Training

```bash
# Train on test symbols (AAPL, MSFT, GOOGL)
python scripts/train_mamba_models.py --test

# Train on all Tier 1 stocks (50 symbols)
python scripts/train_mamba_models.py --symbols TIER_1

# Train specific symbols
python scripts/train_mamba_models.py --symbols AAPL,MSFT,NVDA,TSLA
```

### Custom Configuration

```bash
# Extended training with more epochs
python scripts/train_mamba_models.py \
    --symbols TIER_1 \
    --epochs 100 \
    --batch-size 64 \
    --days 2000

# Faster training with fewer workers
python scripts/train_mamba_models.py \
    --symbols AAPL,MSFT \
    --parallel 2 \
    --epochs 20
```

### Python API

```python
from src.ml.state_space.mamba_model import MambaPredictor, MambaConfig

# Initialize predictor
config = MambaConfig(
    d_model=128,
    num_layers=6,
    prediction_horizons=[1, 5, 10, 30]
)

predictor = MambaPredictor(
    symbols=['AAPL'],
    config=config
)

# Make predictions
predictions = await predictor.predict(
    symbol='AAPL',
    price_history=price_data,
    current_price=175.50
)

# Returns: {'1d': 176.2, '5d': 178.5, '10d': 180.1, '30d': 185.0}
```

---

## Training Pipeline

### 1. Data Preprocessing

**Feature Engineering** (10+ features per timestep):

```python
from src.ml.state_space.data_preprocessing import TimeSeriesFeatureEngineer

engineer = TimeSeriesFeatureEngineer(
    windows=[5, 10, 20, 60],
    include_volume=False
)

features = engineer.extract_features(prices)
```

**Features extracted:**
- Normalized price
- Simple returns
- Log returns
- Multi-scale SMA (5, 10, 20, 60 day)
- Multi-scale EMA (5, 10, 20, 60 day)
- Rolling volatility
- Momentum indicators
- Rate of change
- RSI (14-day)
- Bollinger Band position
- MACD (line, signal, histogram)
- Price position in range

### 2. Sequence Generation

```python
from src.ml.state_space.data_preprocessing import SequenceGenerator

generator = SequenceGenerator(
    sequence_length=60,        # 60-day lookback
    prediction_horizons=[1, 5, 10, 30],
    stride=1                   # Sliding window step
)

X, y = generator.generate_sequences(features, prices)
# X: [n_samples, 60, n_features]
# y: {1: [n_samples, 1], 5: [n_samples, 1], ...}
```

### 3. Data Augmentation

```python
from src.ml.state_space.data_preprocessing import DataAugmentor

augmentor = DataAugmentor(augmentation_rate=0.5)

# Augmentation techniques:
# - Gaussian noise injection
# - Magnitude warping
# - Feature dropout

X_aug, y_aug = augmentor.augment(X, y)
```

### 4. Model Training

```python
from src.ml.state_space.mamba_model import MambaModel
import tensorflow as tf

# Build model
model = MambaModel(config)
model.build((None, sequence_length, n_features))

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='mse',
    metrics=['mae']
)

# Train with callbacks
history = model.fit(
    X, y,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint('best_model.weights.h5')
    ]
)
```

### 5. Model Persistence

**Weights saved to:**
```
models/mamba/
├── weights/
│   ├── AAPL.weights.h5
│   ├── MSFT.weights.h5
│   └── ...
├── metadata/
│   ├── AAPL_metadata.json
│   └── ...
├── metrics/
│   ├── AAPL_metrics.json
│   └── ...
└── checkpoints/
    ├── AAPL/
    │   └── best_model.weights.h5
    └── ...
```

**Metadata example:**
```json
{
  "symbol": "AAPL",
  "training_date": "2025-01-09T10:30:00",
  "model_config": {
    "d_model": 64,
    "d_state": 16,
    "num_layers": 4,
    "prediction_horizons": [1, 5, 10, 30]
  },
  "performance_metrics": {
    "best_val_loss": 0.0234,
    "best_epoch": 38,
    "total_epochs": 50
  },
  "data_stats": {
    "num_samples": 850,
    "avg_return": 0.0015
  }
}
```

---

## Advanced Features

### Custom Training Callbacks

```python
from scripts.train_mamba_models import TrainingCallbacks

callbacks_mgr = TrainingCallbacks(
    save_dir='models/mamba',
    symbol='AAPL',
    patience=5,
    min_delta=0.0001
)

keras_callbacks = callbacks_mgr.get_keras_callbacks()
# Includes:
# - ModelCheckpoint (save best weights)
# - EarlyStopping (prevent overfitting)
# - ReduceLROnPlateau (adaptive learning rate)
```

### Metrics Tracking

```python
from scripts.train_mamba_models import MetricsTracker

tracker = MetricsTracker(
    save_dir='models/mamba',
    symbol='AAPL'
)

for epoch in range(epochs):
    # ... training ...
    tracker.update(
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        train_mae=train_mae,
        val_mae=val_mae,
        lr=learning_rate,
        epoch_time=elapsed
    )

tracker.save()  # Saves to metrics/AAPL_metrics.json
summary = tracker.get_summary()
```

### Data Quality Validation

```python
from src.ml.state_space.data_preprocessing import validate_data_quality

is_valid, message = validate_data_quality(
    prices,
    min_samples=100,
    max_missing_ratio=0.05
)

if not is_valid:
    print(f"Data quality issue: {message}")
```

Checks:
- ✓ Minimum sample count
- ✓ Missing/NaN values
- ✓ Infinite values
- ✓ Non-positive prices
- ✓ Sufficient variance

---

## Performance Tuning

### Hyperparameter Optimization

| Parameter | Default | Small Dataset | Large Dataset | High Variance |
|-----------|---------|---------------|---------------|---------------|
| `d_model` | 64 | 32 | 128 | 96 |
| `d_state` | 16 | 8 | 32 | 24 |
| `num_layers` | 4 | 2 | 6 | 5 |
| `epochs` | 50 | 20 | 100 | 75 |
| `batch_size` | 32 | 16 | 64 | 32 |
| `learning_rate` | 0.001 | 0.002 | 0.0005 | 0.001 |

### GPU Optimization

```python
import tensorflow as tf

# Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Mixed precision (faster training)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Multi-GPU training
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = MambaModel(config)
    model.compile(...)
```

### Training Time Estimates

| Configuration | GPU | CPU | Notes |
|--------------|-----|-----|-------|
| 1 symbol, 50 epochs | 30-50s | 2-4 min | NVIDIA RTX 3090 |
| 10 symbols, 50 epochs | 5-8 min | 30-45 min | Parallel=5 |
| 50 symbols (Tier 1) | 20-30 min | 3-4 hours | Parallel=10 |

### Memory Requirements

- **Model**: ~2-10 MB (depending on config)
- **Training data**: ~50-200 MB per symbol (1000 days)
- **GPU VRAM**: 2-4 GB recommended
- **System RAM**: 8 GB minimum, 16 GB recommended

---

## Troubleshooting

### Common Issues

#### 1. TensorFlow Import Error

```
ImportError: DLL load failed while importing _pywrap_tensorflow_internal
```

**Solution:**
- Ensure TensorFlow is imported first in scripts
- Install: `pip install tensorflow==2.16.1`
- Windows: May need Visual C++ Redistributable

#### 2. Out of Memory (OOM)

```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**
- Reduce `batch_size` (try 16 or 8)
- Reduce `d_model` or `num_layers`
- Enable GPU memory growth
- Use gradient accumulation

#### 3. NaN Loss

```
Loss becomes NaN during training
```

**Solutions:**
- Check data quality (no inf/nan values)
- Reduce learning rate (try 0.0005)
- Add gradient clipping: `clipnorm=1.0`
- Check feature normalization

#### 4. Slow Training

**Solutions:**
- Use GPU if available
- Increase `batch_size` (if memory allows)
- Reduce `sequence_length`
- Decrease number of features
- Use mixed precision training

#### 5. Poor Predictions

**Solutions:**
- Increase training data (try 2000+ days)
- Add more features
- Tune hyperparameters
- Check for data leakage
- Validate train/val split

### Validation Checklist

Before training:
- [ ] Data has >100 samples
- [ ] No missing/NaN values
- [ ] All prices positive
- [ ] Sufficient price variance
- [ ] Features normalized
- [ ] Sequences generated correctly
- [ ] Train/val split makes sense

After training:
- [ ] Validation loss converges
- [ ] No severe overfitting
- [ ] Predictions reasonable
- [ ] Weights saved successfully
- [ ] Metadata complete

---

## API Reference

### MambaPredictor

```python
class MambaPredictor:
    """High-level interface for Mamba predictions"""

    def __init__(
        self,
        symbols: List[str],
        config: Optional[MambaConfig] = None
    )

    async def predict(
        self,
        symbol: str,
        price_history: np.ndarray,
        current_price: float
    ) -> Dict[str, float]

    def train(
        self,
        training_data: Dict[str, np.ndarray],
        epochs: int = 50
    )
```

### TimeSeriesFeatureEngineer

```python
class TimeSeriesFeatureEngineer:
    """Advanced feature engineering"""

    def __init__(
        self,
        windows: List[int] = [5, 10, 20, 60],
        include_volume: bool = False
    )

    def extract_features(
        self,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None
    ) -> np.ndarray
```

### SequenceGenerator

```python
class SequenceGenerator:
    """Generate training sequences"""

    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizons: List[int] = [1, 5, 10, 30],
        stride: int = 1
    )

    def generate_sequences(
        self,
        features: np.ndarray,
        prices: np.ndarray
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]
```

### DataAugmentor

```python
class DataAugmentor:
    """Data augmentation"""

    def __init__(self, augmentation_rate: float = 0.5)

    def augment(
        self,
        X: np.ndarray,
        y: Dict[int, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]
```

---

## Best Practices

### Data Preparation

1. **Use sufficient history**: 500-1000+ days recommended
2. **Validate data quality**: Run validation before training
3. **Feature engineering**: Start with default features, add domain-specific ones
4. **Train/val split**: Use 80/20 temporal split (no shuffling!)

### Model Configuration

1. **Start simple**: Use default config first
2. **Incremental scaling**: Increase model size only if needed
3. **Monitor overfitting**: Watch train vs val loss
4. **Early stopping**: Use patience=5 to prevent overfitting

### Training Strategy

1. **Batch training**: Train multiple symbols in parallel
2. **Checkpointing**: Always save best weights
3. **Metrics tracking**: Monitor loss, MAE, learning rate
4. **Validation**: Test predictions on held-out data

### Production Deployment

1. **Model versioning**: Track metadata with each training run
2. **A/B testing**: Compare new models against baseline
3. **Fallback**: Always have a simple momentum-based fallback
4. **Monitoring**: Track prediction accuracy in production

---

## Examples

### Example 1: Basic Training

```python
# Train AAPL with defaults
python scripts/train_mamba_models.py --symbols AAPL
```

Output:
```
[AAPL] Starting Mamba training...
[AAPL] Loaded 987 days of price data
[AAPL] Created 867 training samples
[AAPL] Model has 124,928 parameters
[AAPL] Training for 50 epochs...
[AAPL] ✓ Training complete! Best Val Loss: 0.0234, Time: 42.3s
```

### Example 2: Custom Configuration

```python
import asyncio
from src.ml.state_space.mamba_model import MambaConfig
from scripts.train_mamba_models import train_single_symbol

config = MambaConfig(
    d_model=128,
    d_state=32,
    num_layers=6,
    prediction_horizons=[1, 3, 7, 14, 30]
)

result = asyncio.run(
    train_single_symbol(
        symbol='AAPL',
        epochs=100,
        batch_size=64,
        config=config
    )
)

print(f"Best validation loss: {result[1]['best_val_loss']:.4f}")
```

### Example 3: Production Pipeline

```python
# 1. Fetch data
from src.api.ml_integration_helpers import fetch_historical_prices

prices = await fetch_historical_prices(['AAPL'], days=2000)

# 2. Validate
from src.ml.state_space.data_preprocessing import validate_data_quality

is_valid, msg = validate_data_quality(prices['AAPL'])
assert is_valid, msg

# 3. Train
from scripts.train_mamba_models import train_single_symbol

symbol, result = await train_single_symbol(
    'AAPL',
    epochs=50,
    historical_days=2000
)

# 4. Load and predict
from src.ml.state_space.mamba_model import MambaPredictor

predictor = MambaPredictor(['AAPL'])
predictions = await predictor.predict(
    'AAPL',
    prices['AAPL'],
    current_price=175.50
)

print(f"Predictions: {predictions}")
# {'1d': 176.2, '5d': 178.5, '10d': 180.1, '30d': 185.0}
```

---

## Further Reading

- [Mamba Paper](https://arxiv.org/abs/2312.00752) - Original research
- [State Space Models](https://huggingface.co/blog/lbourdois/get-on-the-ssm-train) - Background
- [Time Series Best Practices](https://otexts.com/fpp3/) - Forecasting principles
- [TensorFlow Performance Guide](https://www.tensorflow.org/guide/gpu) - GPU optimization

---

**Version**: 1.0.0
**Last Updated**: 2025-01-09
**Maintainer**: Options Probability Team

For questions or issues, please refer to the project README or open an issue on GitHub.
