"""
Advanced Forecasting Service - Priority #1 Integration

Combines:
1. Temporal Fusion Transformer (TFT) for multi-horizon forecasting
2. Conformal Prediction for guaranteed uncertainty quantification
3. TimesFM foundation model integration (placeholder for future)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

from .tft_model import TemporalFusionTransformer, TFTPredictor, MultiHorizonForecast
from .conformal_prediction import (
    MultiHorizonConformalPredictor,
    PredictionInterval
)
from ..feature_engineering import FeatureEngineeringService, FeatureSet

logger = logging.getLogger(__name__)


class AdvancedForecastService:
    """
    High-level service combining TFT + Conformal Prediction

    Provides production-ready multi-horizon forecasts with guaranteed
    uncertainty quantification.
    """

    def __init__(self,
                 horizons: List[int] = [1, 5, 10, 30],
                 coverage_level: float = 0.95):
        """
        Args:
            horizons: Forecast horizons in days
            coverage_level: Target coverage for prediction intervals (e.g., 0.95)
        """
        self.horizons = horizons
        self.coverage_level = coverage_level

        # Core models
        self.tft_predictor = TFTPredictor(horizons=horizons)
        self.conformal_predictor = MultiHorizonConformalPredictor(
            horizons=horizons,
            alpha=1 - coverage_level
        )

        self.feature_engineer = FeatureEngineeringService()

        # Caches
        self.model_cache = {}
        self.forecast_cache = {}

        logger.info(f"Initialized AdvancedForecastService with horizons {horizons}")

    async def get_forecast(self,
                          symbol: str,
                          current_price: float,
                          features: Optional[np.ndarray] = None,
                          use_cache: bool = True) -> Dict:
        """
        Get multi-horizon forecast with uncertainty quantification

        Args:
            symbol: Stock symbol
            current_price: Current stock price
            features: Pre-computed features [lookback, num_features]
            use_cache: Whether to use cached forecasts

        Returns:
            Dict with forecast data
        """
        # Check cache
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H')}"
        if use_cache and cache_key in self.forecast_cache:
            return self.forecast_cache[cache_key]

        # Get features if not provided
        if features is None:
            features = await self._prepare_features(symbol)

        # Get TFT forecast
        tft_forecast = await self.tft_predictor.forecast(
            symbol=symbol,
            features=features,
            current_price=current_price
        )

        # Get conformal prediction intervals
        predictions_by_horizon = {
            h: pred for h, pred in zip(self.horizons, tft_forecast.predictions)
        }

        conformal_intervals = self.conformal_predictor.predict_intervals(
            predictions_by_horizon
        )

        # Combine into result
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'current_price': current_price,
            'horizons': self.horizons,
            'coverage_level': self.coverage_level,

            # Point forecasts
            'predictions': tft_forecast.predictions,

            # TFT quantiles
            'tft_q10': tft_forecast.q10,
            'tft_q50': tft_forecast.q50,
            'tft_q90': tft_forecast.q90,

            # Conformal intervals (guaranteed coverage)
            'conformal_lower': [conformal_intervals[h].lower_bound for h in self.horizons],
            'conformal_upper': [conformal_intervals[h].upper_bound for h in self.horizons],
            'conformal_width': [conformal_intervals[h].width for h in self.horizons],

            # Expected returns
            'expected_returns': [
                (pred - current_price) / current_price
                for pred in tft_forecast.predictions
            ],

            # Feature importance (from TFT variable selection)
            'feature_importance': tft_forecast.feature_importance,

            # Metadata
            'model': 'TFT + Conformal',
            'is_calibrated': all(p.is_calibrated for p in self.conformal_predictor.predictors.values())
        }

        # Cache result
        if use_cache:
            self.forecast_cache[cache_key] = result

        return result

    async def _prepare_features(self, symbol: str) -> np.ndarray:
        """
        Prepare features for forecasting

        Args:
            symbol: Stock symbol

        Returns:
            Feature array [lookback, num_features]
        """
        try:
            # Get historical data (placeholder - would use real data service)
            import yfinance as yf

            # Get 90 days of data for 60-day lookback
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="3mo")

            if len(df) < 60:
                raise ValueError(f"Insufficient data for {symbol}: {len(df)} days")

            # Generate features
            feature_sets = self.feature_engineer.generate_features(df, symbol)

            # Convert to array (last 60 observations)
            features = np.array([fs.to_array() for fs in feature_sets[-60:]])

            return features

        except Exception as e:
            logger.error(f"Error preparing features for {symbol}: {e}")
            raise

    async def train_model(self,
                         symbols: List[str],
                         epochs: int = 50,
                         batch_size: int = 32) -> Dict:
        """
        Train TFT model on multiple symbols

        Args:
            symbols: List of stock symbols
            epochs: Training epochs
            batch_size: Batch size

        Returns:
            Training results
        """
        logger.info(f"Training TFT on {len(symbols)} symbols...")

        # Collect training data
        X_train_list = []
        y_train_list = []

        for symbol in symbols:
            try:
                features = await self._prepare_features(symbol)

                # Create training samples
                lookback = 60
                for i in range(len(features) - lookback - max(self.horizons)):
                    X = features[i:i+lookback]
                    y = []
                    for h in self.horizons:
                        y.append(features[i+lookback+h, 0])  # Close price

                    X_train_list.append(X)
                    y_train_list.append(y)

            except Exception as e:
                logger.warning(f"Skipping {symbol}: {e}")
                continue

        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)

        logger.info(f"Training data: {X_train.shape}, Targets: {y_train.shape}")

        # Split into train/val
        split_idx = int(len(X_train) * 0.8)
        X_val = X_train[split_idx:]
        y_val = y_train[split_idx:]
        X_train = X_train[:split_idx]
        y_train = y_train[:split_idx]

        # Train TFT
        history = self.tft_predictor.tft.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size
        )

        # Calibrate conformal predictor
        val_preds = self.tft_predictor.tft.predict(X_val)
        for i, h in enumerate(self.horizons):
            self.conformal_predictor.predictors[h].calibrate(
                val_preds['q50'][:, i],
                y_val[:, i]
            )

        results = {
            'status': 'success',
            'num_symbols': len(symbols),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'epochs': len(history.history['loss']),
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1] if 'val_loss' in history.history else None
        }

        logger.info(f"Training complete: {results}")
        return results

    async def evaluate_model(self, symbols: List[str]) -> Dict:
        """
        Evaluate model on test symbols

        Args:
            symbols: Test symbols

        Returns:
            Evaluation metrics
        """
        all_metrics = []

        for symbol in symbols:
            try:
                features = await self._prepare_features(symbol)

                # Get forecast
                current_price = features[-1, 0]
                forecast = await self.get_forecast(symbol, current_price, features)

                # Simple evaluation (would be more sophisticated in production)
                metrics = {
                    'symbol': symbol,
                    'current_price': current_price,
                    'forecast_1d': forecast['predictions'][0],
                    'expected_return_1d': forecast['expected_returns'][0]
                }

                all_metrics.append(metrics)

            except Exception as e:
                logger.warning(f"Error evaluating {symbol}: {e}")
                continue

        results = {
            'num_symbols': len(symbols),
            'evaluated': len(all_metrics),
            'metrics': all_metrics
        }

        return results

    def get_trading_signal(self, forecast: Dict) -> Dict:
        """
        Generate trading signal from forecast

        Args:
            forecast: Forecast dict from get_forecast()

        Returns:
            Trading signal dict
        """
        # 1-day forecast
        pred_1d = forecast['predictions'][0]
        current_price = forecast['current_price']
        expected_return = forecast['expected_returns'][0]

        # Conformal interval
        lower_1d = forecast['conformal_lower'][0]
        upper_1d = forecast['conformal_upper'][0]
        interval_width = forecast['conformal_width'][0]

        # Decision logic
        confidence = 1.0 - (interval_width / current_price)  # Narrower interval = higher confidence

        if expected_return > 0.02 and lower_1d > current_price:
            # Strong buy signal: +2%+ expected return, lower bound above current
            action = 'BUY'
            strength = min(confidence * abs(expected_return) * 10, 1.0)
        elif expected_return < -0.02 and upper_1d < current_price:
            # Strong sell signal
            action = 'SELL'
            strength = min(confidence * abs(expected_return) * 10, 1.0)
        elif expected_return > 0.01:
            # Weak buy
            action = 'HOLD_BULLISH'
            strength = confidence * 0.5
        elif expected_return < -0.01:
            # Weak sell
            action = 'HOLD_BEARISH'
            strength = confidence * 0.5
        else:
            # Neutral
            action = 'HOLD'
            strength = 0.5

        return {
            'action': action,
            'confidence': confidence,
            'strength': strength,
            'expected_return_1d': expected_return,
            'interval_width_pct': interval_width / current_price,
            'reasoning': self._generate_signal_reasoning(
                action, expected_return, confidence, interval_width, current_price
            )
        }

    def _generate_signal_reasoning(self,
                                   action: str,
                                   expected_return: float,
                                   confidence: float,
                                   interval_width: float,
                                   current_price: float) -> str:
        """Generate human-readable reasoning for trading signal"""
        reasoning = f"Expected return: {expected_return:+.2%}. "

        if action == 'BUY':
            reasoning += f"Strong buy signal - model predicts upside with {confidence:.1%} confidence. "
        elif action == 'SELL':
            reasoning += f"Strong sell signal - model predicts downside with {confidence:.1%} confidence. "
        elif action.startswith('HOLD'):
            reasoning += f"Weak signal - hold current position. "
        else:
            reasoning += "Neutral - no strong signal. "

        width_pct = interval_width / current_price
        reasoning += f"Prediction interval: ±${interval_width:.2f} ({width_pct:.1%}). "

        if width_pct < 0.03:
            reasoning += "High certainty (narrow interval)."
        elif width_pct < 0.05:
            reasoning += "Moderate certainty."
        else:
            reasoning += "Low certainty (wide interval) - consider waiting."

        return reasoning
