"""
ML Prediction Service

Unified service for ML-based price predictions.
Orchestrates data collection, feature engineering, and model predictions.

Supports:
- LSTM price prediction
- Model training and retraining
- Prediction caching
- Confidence scoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import asyncio
import os

from .data_collection import DataCollectionService
from .feature_engineering import FeatureEngineeringService, FeatureSet

logger = logging.getLogger(__name__)

# Check TensorFlow availability
try:
    from .lstm_model import LSTMPricePredictor, LSTMPrediction, ModelMetrics
    LSTM_AVAILABLE = True
except ImportError:
    logger.warning("LSTM model not available (TensorFlow not installed)")
    LSTM_AVAILABLE = False
    # Create dummy classes
    @dataclass
    class LSTMPrediction:
        symbol: str
        timestamp: datetime
        prediction_horizon: int
        predicted_returns: List[float]
        predicted_prices: List[float]
        confidence: float
        direction: str
        recommendation: str

    @dataclass
    class ModelMetrics:
        mae: float
        rmse: float
        directional_accuracy: float
        sharpe_ratio: float
        train_loss: float
        val_loss: float


# ============================================================================
# Ensemble Prediction
# ============================================================================

@dataclass
class EnsemblePrediction:
    """Ensemble prediction from multiple models"""
    symbol: str
    timestamp: datetime

    # LSTM predictions
    lstm_prediction: Optional[LSTMPrediction]

    # Ensemble results
    predicted_direction: str  # "UP", "DOWN", "NEUTRAL"
    recommendation: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0-1

    # Price targets
    target_price_1d: float
    target_price_5d: float

    # Risk metrics
    expected_return_5d: float
    downside_risk: float  # Estimated max drawdown

    # Supporting data
    current_price: float
    features_used: int
    models_used: List[str]


# ============================================================================
# ML Prediction Service
# ============================================================================

class MLPredictionService:
    """
    Unified ML prediction service.

    Features:
    - Automated data collection and feature engineering
    - LSTM-based price predictions
    - Model caching and management
    - Prediction caching
    - Confidence scoring
    """

    def __init__(self, model_dir: str = "models/ml"):
        """
        Initialize prediction service.

        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.data_collector = DataCollectionService()
        self.feature_engineer = FeatureEngineeringService()

        # Model cache
        self.lstm_models: Dict[str, LSTMPricePredictor] = {}

        # Prediction cache (valid for 1 hour)
        self.prediction_cache: Dict[str, tuple[datetime, EnsemblePrediction]] = {}
        self.cache_ttl = timedelta(hours=1)

    async def get_prediction(
        self,
        symbol: str,
        force_refresh: bool = False
    ) -> Optional[EnsemblePrediction]:
        """
        Get ML prediction for a symbol.

        Args:
            symbol: Stock symbol
            force_refresh: Force new prediction (ignore cache)

        Returns:
            EnsemblePrediction or None if prediction fails
        """
        symbol = symbol.upper()

        # Check cache
        if not force_refresh and symbol in self.prediction_cache:
            cache_time, prediction = self.prediction_cache[symbol]
            if datetime.now() - cache_time < self.cache_ttl:
                logger.info(f"Using cached prediction for {symbol}")
                return prediction

        try:
            # Get current data and features
            df = await self._get_recent_data(symbol, days=300)
            if df is None or len(df) < 200:
                logger.error(f"Insufficient data for {symbol}")
                return None

            # Generate features
            features = self.feature_engineer.generate_features(df, symbol)
            if len(features) < 60:
                logger.error(f"Insufficient features for {symbol}")
                return None

            # Get current price
            current_price = features[-1].close

            # Load or train LSTM model
            lstm_model = await self._get_or_train_lstm(symbol)

            # Make LSTM prediction
            lstm_pred = None
            if lstm_model and LSTM_AVAILABLE:
                lstm_pred = await self._make_lstm_prediction(
                    symbol, features, current_price, lstm_model
                )

            # Create ensemble prediction
            prediction = self._create_ensemble_prediction(
                symbol, current_price, lstm_pred, len(features)
            )

            # Cache prediction
            self.prediction_cache[symbol] = (datetime.now(), prediction)

            logger.info(f"Generated prediction for {symbol}: {prediction.recommendation} "
                       f"(confidence: {prediction.confidence:.2%})")

            return prediction

        except Exception as e:
            logger.error(f"Error generating prediction for {symbol}: {e}")
            return None

    async def train_model(
        self,
        symbol: str,
        years: int = 5,
        epochs: int = 100,
        force_retrain: bool = False
    ) -> Optional[ModelMetrics]:
        """
        Train LSTM model for a symbol.

        Args:
            symbol: Stock symbol
            years: Years of training data
            epochs: Training epochs
            force_retrain: Force retraining even if model exists

        Returns:
            ModelMetrics if successful
        """
        if not LSTM_AVAILABLE:
            logger.error("LSTM model not available (TensorFlow not installed)")
            return None

        symbol = symbol.upper()
        model_path = os.path.join(self.model_dir, f"{symbol}_lstm.h5")

        # Check if model exists
        if not force_retrain and os.path.exists(model_path):
            logger.info(f"Model already exists for {symbol}")
            return None

        try:
            logger.info(f"Training LSTM model for {symbol} ({years} years, {epochs} epochs)")

            # Use the training pipeline
            from .lstm_model import train_lstm_model
            model, metrics = await train_lstm_model(
                symbol=symbol,
                years=years,
                epochs=epochs
            )

            # Save model
            model.save_model(model_path)

            # Cache model
            self.lstm_models[symbol] = model

            logger.info(f"Model trained and saved: {model_path}")
            logger.info(f"Metrics: MAE={metrics.mae:.4f}, "
                       f"Directional Accuracy={metrics.directional_accuracy:.2%}, "
                       f"Sharpe={metrics.sharpe_ratio:.2f}")

            return metrics

        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return None

    async def _get_or_train_lstm(self, symbol: str) -> Optional['LSTMPricePredictor']:
        """Get LSTM model from cache or load/train it"""
        if not LSTM_AVAILABLE:
            return None

        # Check cache
        if symbol in self.lstm_models:
            return self.lstm_models[symbol]

        # Try to load from disk
        model_path = os.path.join(self.model_dir, f"{symbol}_lstm.h5")
        if os.path.exists(model_path):
            try:
                model = LSTMPricePredictor()
                model.load_model(model_path)
                self.lstm_models[symbol] = model
                logger.info(f"Loaded LSTM model for {symbol}")
                return model
            except Exception as e:
                logger.error(f"Error loading model for {symbol}: {e}")

        # Model doesn't exist - train it
        logger.info(f"No model found for {symbol}, training new model...")
        metrics = await self.train_model(symbol, years=3, epochs=50)  # Quick training

        if metrics:
            return self.lstm_models.get(symbol)

        return None

    async def _get_recent_data(self, symbol: str, days: int = 300) -> Optional[pd.DataFrame]:
        """Get recent historical data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        df = await self.data_collector.collect_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval='1d'
        )

        return df

    async def _make_lstm_prediction(
        self,
        symbol: str,
        features: List[FeatureSet],
        current_price: float,
        model: 'LSTMPricePredictor'
    ) -> Optional[LSTMPrediction]:
        """Make prediction using LSTM model"""
        try:
            # Get last 60 feature sets (sequence length)
            if len(features) < 60:
                logger.error(f"Need at least 60 features, got {len(features)}")
                return None

            recent_features = features[-60:]

            # Convert to array
            X = np.array([f.to_array() for f in recent_features])

            # Normalize using model's scaler params (if available)
            if model.scaler_params:
                mean = model.scaler_params['mean']
                std = model.scaler_params['std']
                X = (X - mean) / std

            # Predict
            prediction = model.predict(X, current_price, symbol)

            return prediction

        except Exception as e:
            logger.error(f"Error making LSTM prediction: {e}")
            return None

    def _create_ensemble_prediction(
        self,
        symbol: str,
        current_price: float,
        lstm_pred: Optional[LSTMPrediction],
        features_count: int
    ) -> EnsemblePrediction:
        """Create ensemble prediction from available models"""
        models_used = []

        # Default values
        direction = "NEUTRAL"
        recommendation = "HOLD"
        confidence = 0.5
        target_1d = current_price
        target_5d = current_price
        expected_return = 0.0
        downside_risk = 0.02  # 2% default

        # Use LSTM if available
        if lstm_pred:
            models_used.append("LSTM")
            direction = lstm_pred.direction
            recommendation = lstm_pred.recommendation
            confidence = lstm_pred.confidence

            if len(lstm_pred.predicted_prices) >= 1:
                target_1d = lstm_pred.predicted_prices[0]
            if len(lstm_pred.predicted_prices) >= 5:
                target_5d = lstm_pred.predicted_prices[4]

            expected_return = (target_5d - current_price) / current_price

            # Estimate downside risk from predicted returns
            if lstm_pred.predicted_returns:
                downside_risk = abs(min(lstm_pred.predicted_returns))

        # If no models available, use neutral prediction
        if not models_used:
            models_used.append("BASELINE")
            confidence = 0.3

        return EnsemblePrediction(
            symbol=symbol,
            timestamp=datetime.now(),
            lstm_prediction=lstm_pred,
            predicted_direction=direction,
            recommendation=recommendation,
            confidence=confidence,
            target_price_1d=target_1d,
            target_price_5d=target_5d,
            expected_return_5d=expected_return,
            downside_risk=downside_risk,
            current_price=current_price,
            features_used=features_count,
            models_used=models_used
        )

    def get_model_info(self, symbol: str) -> Optional[Dict]:
        """Get information about trained model"""
        symbol = symbol.upper()
        model_path = os.path.join(self.model_dir, f"{symbol}_lstm.h5")

        if not os.path.exists(model_path):
            return None

        info = {
            "symbol": symbol,
            "model_type": "LSTM",
            "model_path": model_path,
            "model_exists": True,
            "last_modified": datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
        }

        # Load model to get details
        if symbol in self.lstm_models:
            model = self.lstm_models[symbol]
            info.update({
                "sequence_length": model.sequence_length,
                "prediction_horizon": model.prediction_horizon,
                "num_parameters": model.model.count_params() if model.model else None
            })

        return info

    async def batch_predict(self, symbols: List[str]) -> Dict[str, EnsemblePrediction]:
        """Get predictions for multiple symbols"""
        predictions = {}

        tasks = [self.get_prediction(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for symbol, result in zip(symbols, results):
            if isinstance(result, EnsemblePrediction):
                predictions[symbol] = result
            else:
                logger.error(f"Failed to get prediction for {symbol}: {result}")

        return predictions
