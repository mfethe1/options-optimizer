"""
Historical Data Collection Service

Collects and prepares historical OHLCV data for ML model training.
Integrates with existing data providers (Polygon, Alpaca, etc.)
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import logging
import asyncio

logger = logging.getLogger(__name__)


# ============================================================================
# Data Collection Service
# ============================================================================

class DataCollectionService:
    """
    Collect historical data for ML training.

    Features:
    - Multi-provider support (Polygon, Alpaca, Yahoo Finance)
    - Automatic data quality checks
    - Missing data handling
    - Data normalization
    - Train/val/test splitting
    """

    def __init__(self, data_provider=None):
        """
        Initialize data collection service.

        Args:
            data_provider: Optional data provider (Polygon, Alpaca, etc.)
                          If None, will use yfinance as fallback
        """
        self.data_provider = data_provider
        self.cache: Dict[str, pd.DataFrame] = {}

    async def collect_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = '1d'
    ) -> Optional[pd.DataFrame]:
        """
        Collect historical OHLCV data.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Data interval ('1d', '1h', '5m', etc.)

        Returns:
            DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        cache_key = f"{symbol}_{interval}_{start_date.date()}_{end_date.date()}"

        # Check cache first
        if cache_key in self.cache:
            logger.info(f"Using cached data for {cache_key}")
            return self.cache[cache_key]

        # Try data provider first
        if self.data_provider:
            try:
                df = await self._fetch_from_provider(symbol, start_date, end_date, interval)
                if df is not None and len(df) > 0:
                    df = self._clean_and_validate(df)
                    self.cache[cache_key] = df
                    return df
            except Exception as e:
                logger.error(f"Error fetching from data provider: {e}")

        # Fallback to yfinance
        try:
            df = await self._fetch_from_yfinance(symbol, start_date, end_date, interval)
            if df is not None and len(df) > 0:
                df = self._clean_and_validate(df)
                self.cache[cache_key] = df
                return df
        except Exception as e:
            logger.error(f"Error fetching from yfinance: {e}")

        logger.error(f"Failed to collect data for {symbol}")
        return None

    async def _fetch_from_provider(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch from configured data provider"""
        # This would integrate with Polygon, Alpaca, etc.
        # For now, return None to fall back to yfinance
        return None

    async def _fetch_from_yfinance(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch from Yahoo Finance (fallback).

        This runs in executor to avoid blocking async event loop.
        """
        import yfinance as yf

        def fetch():
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval
                )

                if df.empty:
                    return None

                # Rename columns to match our schema
                df = df.reset_index()
                df = df.rename(columns={
                    'Date': 'timestamp' if 'Date' in df.columns else 'Datetime',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })

                # Select only needed columns
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

                return df

            except Exception as e:
                logger.error(f"yfinance error: {e}")
                return None

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(None, fetch)
        return df

    def _clean_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate data.

        - Remove duplicates
        - Handle missing values
        - Validate price ranges
        - Sort by timestamp
        """
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'])

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Forward fill missing values (conservative)
        df = df.fillna(method='ffill')

        # Drop any remaining NaNs
        df = df.dropna()

        # Validate prices are positive
        df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]

        # Validate high >= low
        df = df[df['high'] >= df['low']]

        # Validate volume is non-negative
        df = df[df['volume'] >= 0]

        logger.info(f"Cleaned data: {len(df)} rows")
        return df

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
        prediction_horizon: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for time series ML models.

        Args:
            df: DataFrame with features
            sequence_length: Number of time steps to look back
            prediction_horizon: Number of steps to predict forward

        Returns:
            (X, y) where:
            - X shape: (num_sequences, sequence_length, num_features)
            - y shape: (num_sequences, prediction_horizon)
        """
        if len(df) < sequence_length + prediction_horizon:
            raise ValueError(f"Need at least {sequence_length + prediction_horizon} rows")

        # Extract feature columns (exclude timestamp, symbol if present)
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'symbol']]
        data = df[feature_cols].values

        X, y = [], []

        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            # Input sequence
            X.append(data[i:i + sequence_length])

            # Target: future returns
            future_prices = df['close'].iloc[i + sequence_length:i + sequence_length + prediction_horizon]
            current_price = df['close'].iloc[i + sequence_length - 1]
            future_returns = (future_prices.values - current_price) / current_price

            y.append(future_returns)

        return np.array(X), np.array(y)

    def train_val_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> tuple:
        """
        Split data into train/validation/test sets.

        Time series aware - maintains temporal order.

        Args:
            X: Features
            y: Targets
            train_ratio: Fraction for training
            val_ratio: Fraction for validation (rest is test)

        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X_train = X[:train_end]
        y_train = y[:train_end]

        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]

        X_test = X[val_end:]
        y_test = y[val_end:]

        logger.info(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def normalize_features(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray
    ) -> tuple:
        """
        Normalize features using training set statistics.

        Args:
            X_train, X_val, X_test: Feature arrays

        Returns:
            (X_train_norm, X_val_norm, X_test_norm, scaler_params)
        """
        # Calculate mean and std from training set
        # Shape: (sequence_length, num_features)
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8  # Avoid division by zero

        # Normalize
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std
        X_test_norm = (X_test - mean) / std

        scaler_params = {'mean': mean, 'std': std}

        return X_train_norm, X_val_norm, X_test_norm, scaler_params

    async def collect_training_dataset(
        self,
        symbol: str,
        years: int = 5,
        interval: str = '1d'
    ) -> Optional[pd.DataFrame]:
        """
        Convenience method to collect training dataset.

        Args:
            symbol: Stock symbol
            years: Number of years of history
            interval: Data interval

        Returns:
            DataFrame ready for feature engineering
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)

        df = await self.collect_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )

        if df is None or len(df) < 200:
            logger.error(f"Insufficient data for {symbol}: {len(df) if df is not None else 0} rows")
            return None

        logger.info(f"Collected {len(df)} rows for {symbol} ({years} years)")
        return df

    def calculate_labels(
        self,
        df: pd.DataFrame,
        prediction_horizon: int = 5,
        threshold: float = 0.02
    ) -> pd.DataFrame:
        """
        Calculate classification labels for pattern recognition.

        Args:
            df: DataFrame with 'close' column
            prediction_horizon: Days ahead to predict
            threshold: Return threshold for classification (2% = 0.02)

        Returns:
            DataFrame with 'label' column:
            - 0: SELL (future return < -threshold)
            - 1: HOLD (future return between -threshold and threshold)
            - 2: BUY (future return > threshold)
        """
        df = df.copy()

        # Calculate future returns
        future_price = df['close'].shift(-prediction_horizon)
        current_price = df['close']
        future_return = (future_price - current_price) / current_price

        # Classify
        df['label'] = 1  # Default: HOLD
        df.loc[future_return < -threshold, 'label'] = 0  # SELL
        df.loc[future_return > threshold, 'label'] = 2  # BUY

        # Drop rows without labels (last prediction_horizon rows)
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)

        logger.info(f"Label distribution: "
                   f"SELL={sum(df['label']==0)}, "
                   f"HOLD={sum(df['label']==1)}, "
                   f"BUY={sum(df['label']==2)}")

        return df
