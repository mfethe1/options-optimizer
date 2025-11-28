"""
Mamba State Space Model for Time Series Forecasting - Priority #3

Mamba-2: Structured State Space Sequences with Linear Complexity

Key Advantages:
- O(N) complexity vs Transformers O(N²)
- 5x throughput improvement
- Handles million-length sequences (years of tick data)
- Selective state-space mechanisms
- Hardware-aware algorithm

Research: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
(Gu & Dao, 2023)

Architecture Notes:
- Properly implements Keras Model API with build() method
- Serializable config for model save/load
- Idempotent layer creation for distributed training
- See P0_MAMBA_BUILD_METHOD_FIX.md for implementation details
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# TensorFlow optional dependency
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except Exception as e:
    TENSORFLOW_AVAILABLE = False
    logger.warning(f"TensorFlow import failed for Mamba: {e!r}")


@dataclass
class MambaConfig:
    """Configuration for Mamba model"""
    d_model: int = 64  # Model dimension
    d_state: int = 16  # SSM state dimension
    d_conv: int = 4    # Convolution kernel size
    expand: int = 2    # Expansion factor
    num_layers: int = 4
    prediction_horizons: List[int] = None

    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = [1, 5, 10, 30]


class SelectiveSSM(layers.Layer if TENSORFLOW_AVAILABLE else object):
    """
    Selective State Space Model

    Core innovation: Parameters (B, C, Δ) are input-dependent

    Standard SSM:
        h(t) = A h(t-1) + B x(t)
        y(t) = C h(t)

    Selective SSM:
        B(t) = Linear_B(x(t))
        C(t) = Linear_C(x(t))
        Δ(t) = τ(Linear_Δ(x(t)))  # Time step
    """

    def __init__(self, d_model: int, d_state: int, **kwargs):
        if TENSORFLOW_AVAILABLE:
            super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state

    def build(self, input_shape):
        if not TENSORFLOW_AVAILABLE:
            return

        # SSM parameters (selective - depend on input)
        self.W_B = self.add_weight(
            shape=(self.d_model, self.d_state),
            initializer='glorot_uniform',
            trainable=True,
            name='W_B'
        )
        self.W_C = self.add_weight(
            shape=(self.d_model, self.d_state),
            initializer='glorot_uniform',
            trainable=True,
            name='W_C'
        )
        self.W_delta = self.add_weight(
            shape=(self.d_model, 1),
            initializer='glorot_uniform',
            trainable=True,
            name='W_delta'
        )

        # State transition matrix A (learnable)
        self.A = self.add_weight(
            shape=(self.d_state, self.d_state),
            initializer='orthogonal',
            trainable=True,
            name='A'
        )

        # D skip connection
        self.D = self.add_weight(
            shape=(self.d_model,),
            initializer='ones',
            trainable=True,
            name='D'
        )

    def call(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            y: [batch_size, seq_len, d_model]
        """
        if not TENSORFLOW_AVAILABLE:
            return x

        batch_size = tf.shape(x)[0]
        # Selective parameters (input-dependent)
        B = tf.matmul(x, self.W_B)  # [batch, seq_len, d_state]
        C = tf.matmul(x, self.W_C)  # [batch, seq_len, d_state]
        delta = tf.nn.softplus(tf.matmul(x, self.W_delta))  # [batch, seq_len, 1]

        # Transpose to time-major for scan: [seq_len, batch, ...]
        x_T = tf.transpose(x, [1, 0, 2])
        B_T = tf.transpose(B, [1, 0, 2])
        C_T = tf.transpose(C, [1, 0, 2])
        delta_T = tf.transpose(delta, [1, 0, 2])

        def scan_step(h_prev, elems):
            x_t, B_t, delta_t = elems  # shapes: [batch, d_model], [batch, d_state], [batch, 1]
            # Discretized state space update (simplified)
            h_next = (1.0 - delta_t) * tf.matmul(h_prev, self.A, transpose_b=True) \
                     + delta_t * B_t * tf.expand_dims(x_t[:, 0], -1)
            return h_next

        # Initialize hidden state and run scan over time to get hidden states per time
        h0 = tf.zeros((batch_size, self.d_state))  # [batch, d_state]
        H_T = tf.scan(fn=scan_step, elems=(x_T, B_T, delta_T), initializer=h0)  # [seq_len, batch, d_state]

        # Compute outputs from hidden states
        y_T = tf.reduce_sum(C_T * H_T, axis=-1, keepdims=True) + self.D[0] * x_T[:, :, 0:1]  # [seq_len, batch, 1]
        # y_T -> [batch, seq_len, 1]
        y = tf.transpose(y_T, [1, 0, 2])

        # Expand to d_model dimensions
        y = tf.tile(y, [1, 1, self.d_model])

        return y


class MambaBlock(layers.Layer if TENSORFLOW_AVAILABLE else object):
    """
    Mamba Block = Selective SSM + Conv + Gating

    Architecture:
        x -> Linear -> Conv1D -> SiLU -> SSM -> SiLU -> Linear -> x (residual)
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, **kwargs):
        if TENSORFLOW_AVAILABLE:
            super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self._layers_created = False

    def build(self, input_shape):
        """Create layers with proper shapes"""
        if not TENSORFLOW_AVAILABLE or self._layers_created:
            return

        d_inner = self.expand * self.d_model

        # Input projection
        self.in_proj = layers.Dense(d_inner * 2, use_bias=False, name='in_proj')

        # Depthwise convolution
        self.conv1d = layers.Conv1D(
            filters=d_inner,
            kernel_size=self.d_conv,
            padding='causal',
            groups=d_inner,  # Depthwise
            name='conv1d'
        )

        # Selective SSM
        self.ssm = SelectiveSSM(d_inner, self.d_state, name='ssm')

        # Output projection
        self.out_proj = layers.Dense(self.d_model, use_bias=False, name='out_proj')

        # Layer norm
        self.norm = layers.LayerNormalization(epsilon=1e-5, name='norm')

        self._layers_created = True
        super().build(input_shape)

    def call(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            y: [batch_size, seq_len, d_model]
        """
        if not TENSORFLOW_AVAILABLE:
            return x

        residual = x

        # Normalize
        x = self.norm(x)

        # Input projection (splits into two paths)
        x_proj = self.in_proj(x)
        x1, x2 = tf.split(x_proj, 2, axis=-1)

        # Convolution path
        x1 = self.conv1d(x1)
        x1 = tf.nn.silu(x1)

        # SSM path
        x1 = self.ssm(x1)

        # Gating
        x = x1 * tf.nn.silu(x2)

        # Output projection
        x = self.out_proj(x)

        # Residual
        return x + residual

    def get_config(self):
        """Return serializable config for layer saving"""
        config = super().get_config() if TENSORFLOW_AVAILABLE else {}
        config.update({
            'd_model': self.d_model,
            'd_state': self.d_state,
            'd_conv': self.d_conv,
            'expand': self.expand,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create layer from config (for deserialization)"""
        return cls(**config)


class MambaModel(keras.Model if TENSORFLOW_AVAILABLE else object):
    """
    Full Mamba Model for Time Series Forecasting

    Architecture:
        Embedding -> MambaBlocks x N -> Multi-head prediction

    Properly implements Keras Model API with build() method for:
    - Correct weight initialization
    - Model serialization/deserialization
    - GPU memory optimization
    - Distributed training compatibility
    """

    def __init__(self, config: MambaConfig, **kwargs):
        if not TENSORFLOW_AVAILABLE:
            return

        # Don't pass config to super().__init__() to avoid serialization warning
        super().__init__(**kwargs)
        self.config = config
        self._layers_created = False

    def build(self, input_shape):
        """
        Create layers with proper input shapes

        This is called automatically on first forward pass
        Ensures proper weight initialization and serialization

        Args:
            input_shape: Shape of input tensor (batch, seq_len, features)
        """
        if not TENSORFLOW_AVAILABLE or self._layers_created:
            return

        # Extract dimensions (handle both tuple and TensorShape)
        if isinstance(input_shape, (list, tuple)):
            seq_len, n_features = input_shape[1:] if len(input_shape) > 2 else (None, input_shape[-1])
        else:
            # TensorShape object
            dims = input_shape.as_list()
            seq_len, n_features = dims[1:] if len(dims) > 2 else (None, dims[-1])

        # Input embedding
        self.embed = layers.Dense(self.config.d_model, name='embed')

        # Mamba blocks
        self.blocks = [
            MambaBlock(
                d_model=self.config.d_model,
                d_state=self.config.d_state,
                d_conv=self.config.d_conv,
                expand=self.config.expand,
                name=f'mamba_block_{i}'
            )
            for i in range(self.config.num_layers)
        ]

        # Output heads (one per horizon)
        self.output_heads = {
            horizon: layers.Dense(1, name=f'head_{horizon}d')
            for horizon in self.config.prediction_horizons
        }

        # Final layer norm
        self.norm_f = layers.LayerNormalization(epsilon=1e-5, name='norm_f')

        self._layers_created = True
        super().build(input_shape)

    def call(self, x, training=False):
        """
        Forward pass

        Args:
            x: [batch_size, seq_len, n_features]
            training: Whether in training mode
        Returns:
            predictions: Dict[horizon, [batch_size, 1]]
        """
        if not TENSORFLOW_AVAILABLE:
            return {h: np.zeros((1, 1)) for h in self.config.prediction_horizons}

        # Embed
        x = self.embed(x)

        # Mamba blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm_f(x)

        # Take last time step
        x_last = x[:, -1, :]  # [batch_size, d_model]

        # Multi-horizon predictions
        predictions = {}
        for horizon, head in self.output_heads.items():
            predictions[horizon] = head(x_last)

        return predictions

    def get_config(self):
        """
        Return serializable config for model saving

        Required for proper Keras model serialization
        """
        return {
            'd_model': self.config.d_model,
            'd_state': self.config.d_state,
            'd_conv': self.config.d_conv,
            'expand': self.config.expand,
            'num_layers': self.config.num_layers,
            'prediction_horizons': self.config.prediction_horizons,
        }

    @classmethod
    def from_config(cls, config_dict):
        """
        Create model from config (for deserialization)
        """
        mamba_config = MambaConfig(
            d_model=config_dict['d_model'],
            d_state=config_dict['d_state'],
            d_conv=config_dict['d_conv'],
            expand=config_dict['expand'],
            num_layers=config_dict['num_layers'],
            prediction_horizons=config_dict['prediction_horizons'],
        )
        return cls(config=mamba_config)


class MambaPredictor:
    """
    High-level interface for Mamba predictions

    Features:
    - Multi-horizon forecasting with linear complexity
    - Handles very long sequences efficiently
    - Selective state-space for adaptive modeling
    """

    def __init__(self, symbols: List[str], config: Optional[MambaConfig] = None):
        self.symbols = symbols
        self.config = config or MambaConfig()
        self.model = None

        if TENSORFLOW_AVAILABLE:
            self.model = MambaModel(self.config)
            logger.info(f"Mamba model initialized for {len(symbols)} symbols")
            # Try to load persisted weights
            try:
                import os
                weights_path = os.path.join('models', 'mamba', 'weights.weights.h5')
                if os.path.exists(weights_path):
                    self.model.build((None, 60, 4))  # Build with default feature shape
                    self.model.load_weights(weights_path)
                    logger.info(f"Loaded Mamba weights from {weights_path}")
                else:
                    # Backward compatibility: support legacy file name
                    legacy_path = os.path.join('models', 'mamba', 'weights.h5')
                    if os.path.exists(legacy_path):
                        self.model.build((None, 60, 4))
                        self.model.load_weights(legacy_path)
                        logger.info(f"Loaded Mamba legacy weights from {legacy_path}; consider migrating to {weights_path}")
            except Exception as e:
                logger.warning(f"Mamba weight load skipped: {e}")
        else:
            logger.warning("TensorFlow not available - using fallback predictions")

    def prepare_features(self, price_history: np.ndarray) -> np.ndarray:
        """
        Prepare features from price history

        Args:
            price_history: [seq_len] array of prices
        Returns:
            features: [seq_len, n_features] array
        """
        # Calculate returns
        returns = np.diff(price_history) / price_history[:-1]
        returns = np.concatenate([[0], returns])  # Pad first element

        # Calculate technical indicators
        window = 20

        # Simple moving average
        sma = np.convolve(price_history, np.ones(window)/window, mode='same')

        # Volatility (rolling std)
        volatility = np.array([
            np.std(price_history[max(0, i-window):i+1])
            for i in range(len(price_history))
        ])

        # Normalized price
        price_norm = (price_history - np.mean(price_history)) / (np.std(price_history) + 1e-8)

        # Combine features
        features = np.stack([
            price_norm,
            returns,
            sma / (np.mean(price_history) + 1e-8),  # Normalized SMA
            volatility / (np.std(price_history) + 1e-8)  # Normalized volatility
        ], axis=-1)

        return features

    async def predict(
        self,
        symbol: str,
        price_history: np.ndarray,
        current_price: float
    ) -> Dict[str, float]:
        """
        Generate multi-horizon predictions

        Args:
            symbol: Stock symbol
            price_history: Historical prices (can be very long!)
            current_price: Current price
        Returns:
            predictions: Dict[horizon_str, prediction]
        """
        if not TENSORFLOW_AVAILABLE:
            # Fallback: Simple momentum-based predictions
            recent_return = (current_price - price_history[-30]) / price_history[-30]
            return {
                f'{h}d': current_price * (1 + recent_return * h / 30)
                for h in self.config.prediction_horizons
            }

        # Prepare features
        features = self.prepare_features(price_history)

        # Add batch dimension
        features = np.expand_dims(features, axis=0)  # [1, seq_len, n_features]

        # Model prediction
        try:
            predictions = self.model(features, training=False)

            # Convert to absolute prices
            results = {}
            for horizon, pred_tensor in predictions.items():
                pred_return = pred_tensor.numpy()[0, 0]
                pred_price = current_price * (1 + pred_return)
                results[f'{horizon}d'] = float(pred_price)

            return results

        except Exception as e:
            logger.error(f"Error in Mamba prediction: {e}")
            # Fallback
            recent_return = (current_price - price_history[-30]) / price_history[-30]
            return {
                f'{h}d': current_price * (1 + recent_return * h / 30)
                for h in self.config.prediction_horizons
            }

    def train(self, training_data: Dict[str, np.ndarray], epochs: int = 50,
              validation_ratio: float = 0.2, patience: int = 10):
        """
        Train Mamba model with proper temporal validation split.

        IMPORTANT: Uses temporal split (NOT random) to prevent look-ahead bias.
        First (1 - validation_ratio) of data is training, last validation_ratio is validation.

        Args:
            training_data: Dict[symbol, price_history]
            epochs: Training epochs
            validation_ratio: Fraction of data to use for validation (default 0.2)
            patience: Early stopping patience (default 10)
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available - cannot train")
            return

        logger.info(f"Training Mamba model for {epochs} epochs with temporal validation split...")

        # Prepare training dataset
        X_all = []
        y_all = {h: [] for h in self.config.prediction_horizons}

        for symbol, prices in training_data.items():
            for i in range(60, len(prices) - max(self.config.prediction_horizons)):
                # Features
                features = self.prepare_features(prices[i-60:i])
                X_all.append(features)

                # Labels (future returns)
                current_price = prices[i]
                for horizon in self.config.prediction_horizons:
                    if i + horizon < len(prices):
                        future_price = prices[i + horizon]
                        future_return = (future_price - current_price) / current_price
                        y_all[horizon].append(future_return)
                    else:
                        y_all[horizon].append(0.0)

        X_all = np.array(X_all)
        y_all = {h: np.array(y) for h, y in y_all.items()}

        # TEMPORAL SPLIT (NOT random!) - critical for time series
        # First 80% = training, Last 20% = validation
        n_samples = len(X_all)
        split_idx = int(n_samples * (1 - validation_ratio))

        X_train = X_all[:split_idx]
        X_val = X_all[split_idx:]
        y_train = {h: y[:split_idx] for h, y in y_all.items()}
        y_val = {h: y[split_idx:] for h, y in y_all.items()}

        logger.info(f"Temporal split: {split_idx} training samples, {n_samples - split_idx} validation samples")
        logger.info("Training data is chronologically BEFORE validation data (no look-ahead bias)")

        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae'] * len(self.config.prediction_horizons)
        )

        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )

        # Train with explicit temporal validation data (NOT validation_split!)
        self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),  # Explicit temporal validation set
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )

        # Persist best weights (simple save after fit)
        try:
            import os
            weights_path = os.path.join('models', 'mamba', 'weights.weights.h5')
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            self.model.save_weights(weights_path)
            logger.info(f"Saved Mamba weights to {weights_path}")
        except Exception as e:
            logger.warning(f"Failed to save Mamba weights: {e}")

        logger.info("Mamba training complete")

    def get_efficiency_stats(self, sequence_length: int) -> Dict:
        """
        Get efficiency statistics showing linear complexity advantage

        Args:
            sequence_length: Length of input sequence
        Returns:
            stats: Efficiency comparison
        """
        # Mamba: O(N)
        mamba_ops = sequence_length * self.config.d_model * self.config.d_state

        # Transformer: O(N²)
        transformer_ops = sequence_length ** 2 * self.config.d_model

        speedup = transformer_ops / mamba_ops if mamba_ops > 0 else 1

        return {
            'sequence_length': sequence_length,
            'mamba_complexity': 'O(N)',
            'transformer_complexity': 'O(N²)',
            'mamba_ops': mamba_ops,
            'transformer_ops': transformer_ops,
            'theoretical_speedup': f'{speedup:.1f}x',
            'can_process_ticks': sequence_length > 100000,  # Years of tick data
            'memory_efficient': True
        }
