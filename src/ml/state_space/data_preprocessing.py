"""
Data Preprocessing Utilities for Mamba State Space Models

Advanced feature engineering and data preparation for time series forecasting.

Features:
- Multi-scale technical indicators
- Data augmentation strategies
- Sequence windowing
- Normalization techniques
- Feature selection and validation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TimeSeriesFeatureEngineer:
    """
    Advanced feature engineering for financial time series

    Generates multi-scale technical indicators optimized for
    state space models like Mamba.
    """

    def __init__(
        self,
        windows: List[int] = None,
        include_volume: bool = False
    ):
        """
        Args:
            windows: List of window sizes for multi-scale features
            include_volume: Whether to include volume-based features
        """
        self.windows = windows or [5, 10, 20, 60]
        self.include_volume = include_volume

    def extract_features(
        self,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract comprehensive feature set from price data

        Args:
            prices: [T] array of prices
            volumes: [T] optional array of volumes

        Returns:
            features: [T, n_features] array
        """
        features_list = []

        # 1. Price-based features
        features_list.append(self._normalize(prices).reshape(-1, 1))

        # 2. Returns
        returns = self._returns(prices)
        features_list.append(returns.reshape(-1, 1))

        # 3. Log returns (more stable for large price movements)
        log_returns = self._log_returns(prices)
        features_list.append(log_returns.reshape(-1, 1))

        # 4. Multi-scale moving averages
        # CRITICAL: Normalize using expanding mean to avoid look-ahead bias
        for window in self.windows:
            sma = self._sma(prices, window)
            # Use expanding mean instead of global mean
            expanding_mean = self._expanding_mean(prices)
            features_list.append((sma / (expanding_mean + 1e-8)).reshape(-1, 1))

        # 5. Exponential moving averages (more responsive)
        for window in self.windows:
            ema = self._ema(prices, window)
            # Use expanding mean instead of global mean
            expanding_mean = self._expanding_mean(prices)
            features_list.append((ema / (expanding_mean + 1e-8)).reshape(-1, 1))

        # 6. Volatility features
        for window in self.windows:
            vol = self._rolling_std(returns, window)
            features_list.append(vol.reshape(-1, 1))

        # 7. Momentum indicators
        for window in self.windows:
            momentum = self._momentum(prices, window)
            features_list.append(momentum.reshape(-1, 1))

        # 8. Rate of change
        for window in self.windows:
            roc = self._rate_of_change(prices, window)
            features_list.append(roc.reshape(-1, 1))

        # 9. Relative Strength Index (RSI)
        rsi_14 = self._rsi(prices, 14)
        features_list.append(rsi_14.reshape(-1, 1))

        # 10. Bollinger Band position
        bb_position = self._bollinger_position(prices, 20, 2.0)
        features_list.append(bb_position.reshape(-1, 1))

        # 11. Price range position
        for window in [10, 20]:
            price_pos = self._price_position(prices, window)
            features_list.append(price_pos.reshape(-1, 1))

        # 12. MACD features
        macd, signal, histogram = self._macd(prices)
        features_list.append(macd.reshape(-1, 1))
        features_list.append(signal.reshape(-1, 1))
        features_list.append(histogram.reshape(-1, 1))

        # 13. Volume features (if available)
        if self.include_volume and volumes is not None:
            vol_norm = self._normalize(volumes)
            features_list.append(vol_norm.reshape(-1, 1))

            # Volume-price correlation
            for window in [5, 20]:
                vp_corr = self._volume_price_correlation(prices, volumes, window)
                features_list.append(vp_corr.reshape(-1, 1))

        # Concatenate all features
        features = np.concatenate(features_list, axis=1)

        # Handle NaN and inf
        features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)

        return features

    @staticmethod
    def _normalize(data: np.ndarray) -> np.ndarray:
        """
        Z-score normalization using expanding window

        CRITICAL: Uses only past data to avoid look-ahead bias.
        At time t, uses statistics from data[0:t+1] only.
        """
        result = np.zeros_like(data)
        for i in range(len(data)):
            # Use only data up to current point (expanding window)
            window_data = data[:i+1]
            mean = np.mean(window_data)
            std = np.std(window_data)
            result[i] = (data[i] - mean) / (std + 1e-8)
        return result

    @staticmethod
    def _expanding_mean(data: np.ndarray) -> np.ndarray:
        """
        Expanding window mean (only uses past data)

        CRITICAL: At time t, uses mean of data[0:t+1] only.
        Avoids look-ahead bias.
        """
        result = np.zeros_like(data)
        for i in range(len(data)):
            result[i] = np.mean(data[:i+1])
        return result

    @staticmethod
    def _returns(prices: np.ndarray) -> np.ndarray:
        """Calculate simple returns"""
        returns = np.diff(prices) / (prices[:-1] + 1e-8)
        return np.concatenate([[0], returns])

    @staticmethod
    def _log_returns(prices: np.ndarray) -> np.ndarray:
        """Calculate log returns"""
        log_returns = np.diff(np.log(prices + 1e-8))
        return np.concatenate([[0], log_returns])

    @staticmethod
    def _sma(data: np.ndarray, window: int) -> np.ndarray:
        """
        Simple moving average using expanding window

        CRITICAL: Uses only past data to avoid look-ahead bias.
        For window size W, at time t:
        - If t < W: use mean of data[0:t+1] (expanding)
        - If t >= W: use mean of data[t-W+1:t+1] (rolling)
        """
        result = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - window + 1)
            result[i] = np.mean(data[start:i+1])
        return result

    @staticmethod
    def _ema(data: np.ndarray, window: int) -> np.ndarray:
        """Exponential moving average"""
        alpha = 2 / (window + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema

    @staticmethod
    def _rolling_std(data: np.ndarray, window: int) -> np.ndarray:
        """
        Rolling standard deviation using expanding window

        CRITICAL: Uses only past data to avoid look-ahead bias.
        For window size W, at time t:
        - If t < W: use std of data[0:t+1] (expanding)
        - If t >= W: use std of data[t-W+1:t+1] (rolling, exactly W values)
        """
        result = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - window + 1)
            result[i] = np.std(data[start:i+1])
        return result

    @staticmethod
    def _momentum(data: np.ndarray, window: int) -> np.ndarray:
        """Momentum (rate of change)"""
        result = np.zeros_like(data)
        for i in range(len(data)):
            if i >= window:
                result[i] = (data[i] - data[i-window]) / (data[i-window] + 1e-8)
        return result

    @staticmethod
    def _rate_of_change(data: np.ndarray, window: int) -> np.ndarray:
        """Rate of change percentage"""
        result = np.zeros_like(data)
        for i in range(len(data)):
            if i >= window:
                result[i] = 100 * (data[i] - data[i-window]) / (data[i-window] + 1e-8)
        return result

    @staticmethod
    def _rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
        """
        Relative Strength Index using expanding window

        CRITICAL: Uses only past data to avoid look-ahead bias.
        """
        deltas = np.diff(prices)
        deltas = np.concatenate([[0], deltas])

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)

        # Handle sequences shorter than window
        if len(prices) > window:
            # Initial averages using expanding window up to window size
            avg_gain[window] = np.mean(gains[:window+1])
            avg_loss[window] = np.mean(losses[:window+1])

            # Exponential averages for points beyond window
            for i in range(window + 1, len(prices)):
                avg_gain[i] = (avg_gain[i-1] * (window - 1) + gains[i]) / window
                avg_loss[i] = (avg_loss[i-1] * (window - 1) + losses[i]) / window
        else:
            # For short sequences, use expanding window
            for i in range(len(prices)):
                avg_gain[i] = np.mean(gains[:i+1])
                avg_loss[i] = np.mean(losses[:i+1])

        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def _bollinger_position(
        prices: np.ndarray,
        window: int = 20,
        num_std: float = 2.0
    ) -> np.ndarray:
        """
        Position within Bollinger Bands (0-1)
        0 = lower band, 0.5 = middle band, 1 = upper band
        """
        sma = TimeSeriesFeatureEngineer._sma(prices, window)

        result = np.zeros_like(prices)
        for i in range(len(prices)):
            start = max(0, i - window)
            std = np.std(prices[start:i+1])

            upper = sma[i] + num_std * std
            lower = sma[i] - num_std * std

            if upper > lower:
                result[i] = (prices[i] - lower) / (upper - lower)
            else:
                result[i] = 0.5

        return np.clip(result, 0, 1)

    @staticmethod
    def _price_position(data: np.ndarray, window: int) -> np.ndarray:
        """Position in recent price range (0-1)"""
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
    def _macd(
        prices: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MACD indicator

        Returns:
            (macd_line, signal_line, histogram)
        """
        ema_fast = TimeSeriesFeatureEngineer._ema(prices, fast)
        ema_slow = TimeSeriesFeatureEngineer._ema(prices, slow)

        macd_line = ema_fast - ema_slow
        signal_line = TimeSeriesFeatureEngineer._ema(macd_line, signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def _volume_price_correlation(
        prices: np.ndarray,
        volumes: np.ndarray,
        window: int
    ) -> np.ndarray:
        """Rolling correlation between price changes and volume"""
        price_changes = np.diff(prices)
        price_changes = np.concatenate([[0], price_changes])

        result = np.zeros_like(prices)
        for i in range(len(prices)):
            start = max(0, i - window)
            if i - start > 2:
                corr = np.corrcoef(
                    price_changes[start:i+1],
                    volumes[start:i+1]
                )[0, 1]
                result[i] = corr if not np.isnan(corr) else 0.0

        return result


class DataAugmentor:
    """
    Data augmentation for time series

    Techniques:
    - Gaussian noise injection
    - Window slicing
    - Magnitude warping
    - Time warping (stretch/compress)
    """

    def __init__(self, augmentation_rate: float = 0.5):
        """
        Args:
            augmentation_rate: Probability of applying augmentation
        """
        self.augmentation_rate = augmentation_rate

    def augment(
        self,
        X: np.ndarray,
        y: Dict[int, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Apply random augmentations

        Args:
            X: [n_samples, seq_len, n_features]
            y: {horizon: [n_samples, 1]}

        Returns:
            Augmented (X, y)
        """
        if np.random.rand() > self.augmentation_rate:
            return X, y

        # Choose augmentation
        aug_type = np.random.choice(['noise', 'magnitude', 'dropout'])

        if aug_type == 'noise':
            return self._add_noise(X, y)
        elif aug_type == 'magnitude':
            return self._magnitude_warp(X, y)
        else:  # dropout
            return self._feature_dropout(X, y)

    def _add_noise(
        self,
        X: np.ndarray,
        y: Dict[int, np.ndarray],
        noise_level: float = 0.05
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_level, X.shape)
        X_aug = X + noise * np.std(X, axis=(0, 1), keepdims=True)
        return np.clip(X_aug, -10, 10), y

    def _magnitude_warp(
        self,
        X: np.ndarray,
        y: Dict[int, np.ndarray],
        sigma: float = 0.2
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """Warp magnitude by smooth curve"""
        warp = np.random.normal(1.0, sigma, size=(X.shape[0], 1, X.shape[2]))
        X_aug = X * warp
        return X_aug, y

    def _feature_dropout(
        self,
        X: np.ndarray,
        y: Dict[int, np.ndarray],
        dropout_rate: float = 0.1
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """Randomly zero out features"""
        mask = np.random.binomial(1, 1 - dropout_rate, X.shape)
        X_aug = X * mask
        return X_aug, y


class SequenceGenerator:
    """
    Generate training sequences from time series data

    Handles:
    - Sliding window extraction
    - Multi-horizon target generation
    - Train/validation splitting
    """

    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizons: List[int] = None,
        stride: int = 1
    ):
        """
        Args:
            sequence_length: Length of input sequences
            prediction_horizons: Forecast horizons
            stride: Step size for sliding window
        """
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons or [1, 5, 10, 30]
        self.stride = stride

    def generate_sequences(
        self,
        features: np.ndarray,
        prices: np.ndarray
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Generate sequences from feature array

        Args:
            features: [T, n_features] feature array
            prices: [T] price array for targets

        Returns:
            (X, y_dict) where:
                X: [n_samples, seq_len, n_features]
                y_dict: {horizon: [n_samples, 1]}
        """
        X = []
        y = {h: [] for h in self.prediction_horizons}

        max_horizon = max(self.prediction_horizons)
        T = len(features)

        for i in range(self.sequence_length, T - max_horizon, self.stride):
            # Input sequence
            X.append(features[i-self.sequence_length:i])

            # Target returns
            current_price = prices[i]
            for horizon in self.prediction_horizons:
                if i + horizon < len(prices):
                    future_price = prices[i + horizon]
                    future_return = (future_price - current_price) / current_price
                    y[horizon].append(future_return)
                else:
                    y[horizon].append(0.0)

        X = np.array(X, dtype=np.float32)
        y = {h: np.array(vals, dtype=np.float32).reshape(-1, 1) for h, vals in y.items()}

        logger.info(f"Generated {len(X)} sequences from {T} timesteps")

        return X, y

    def split_sequences(
        self,
        X: np.ndarray,
        y: Dict[int, np.ndarray],
        train_ratio: float = 0.8
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray], np.ndarray, Dict[int, np.ndarray]]:
        """
        Split sequences into train/validation

        Args:
            X: Input sequences
            y: Target dict
            train_ratio: Fraction for training

        Returns:
            (X_train, y_train, X_val, y_val)
        """
        n_samples = len(X)
        split_idx = int(n_samples * train_ratio)

        X_train = X[:split_idx]
        X_val = X[split_idx:]

        y_train = {h: vals[:split_idx] for h, vals in y.items()}
        y_val = {h: vals[split_idx:] for h, vals in y.items()}

        logger.info(f"Split: {len(X_train)} train, {len(X_val)} validation samples")

        return X_train, y_train, X_val, y_val


def validate_data_quality(
    prices: np.ndarray,
    min_samples: int = 100,
    max_missing_ratio: float = 0.05
) -> Tuple[bool, str]:
    """
    Validate data quality before training

    Args:
        prices: Price array
        min_samples: Minimum required samples
        max_missing_ratio: Maximum allowed missing data ratio

    Returns:
        (is_valid, message)
    """
    # Check length
    if len(prices) < min_samples:
        return False, f"Insufficient data: {len(prices)} < {min_samples}"

    # Check for NaN/inf
    if np.isnan(prices).any():
        missing_ratio = np.isnan(prices).sum() / len(prices)
        if missing_ratio > max_missing_ratio:
            return False, f"Too many missing values: {missing_ratio:.2%}"

    if np.isinf(prices).any():
        return False, "Infinite values detected"

    # Check for zeros
    if (prices <= 0).any():
        return False, "Non-positive prices detected"

    # Check variance
    if np.std(prices) < 1e-6:
        return False, "Insufficient price variance"

    return True, "Data quality OK"
