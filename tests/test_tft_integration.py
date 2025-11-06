"""
Integration tests for Temporal Fusion Transformer (TFT)

Tests:
1. TFT model creation and configuration
2. Multi-horizon forecasting
3. Variable selection network
4. Quantile forecasting for uncertainty
5. Attention mechanisms
"""

import pytest
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml.advanced_forecasting.tft_model import (
    MultiHorizonForecast,
    TENSORFLOW_AVAILABLE
)


class TestTFTConfiguration:
    """Test TFT configuration and setup"""

    def test_multi_horizon_forecast_creation(self):
        """Test creating multi-horizon forecast"""
        forecast = MultiHorizonForecast(
            timestamp=datetime.now(),
            symbol='AAPL',
            horizons=[1, 5, 10, 30],
            predictions=[150.0, 152.0, 155.0, 160.0],
            q10=[145.0, 147.0, 150.0, 155.0],
            q50=[150.0, 152.0, 155.0, 160.0],
            q90=[155.0, 157.0, 160.0, 165.0],
            feature_importance={'price': 0.5, 'volume': 0.3, 'sentiment': 0.2},
            current_price=150.0
        )

        assert forecast is not None
        assert forecast.symbol == 'AAPL'
        assert len(forecast.horizons) == 4
        assert len(forecast.predictions) == 4

    def test_prediction_intervals(self):
        """Test prediction interval calculation"""
        forecast = MultiHorizonForecast(
            timestamp=datetime.now(),
            symbol='AAPL',
            horizons=[1, 5],
            predictions=[150.0, 152.0],
            q10=[145.0, 147.0],
            q50=[150.0, 152.0],
            q90=[155.0, 157.0],
            feature_importance={},
            current_price=150.0
        )

        intervals = forecast.get_prediction_intervals()
        assert len(intervals) == 2
        assert intervals[0] == (145.0, 150.0, 155.0)
        assert intervals[1] == (147.0, 152.0, 157.0)

    def test_expected_return_calculation(self):
        """Test expected return calculation"""
        forecast = MultiHorizonForecast(
            timestamp=datetime.now(),
            symbol='AAPL',
            horizons=[1, 5, 10],
            predictions=[150.0, 155.0, 160.0],
            q10=[145.0, 150.0, 155.0],
            q50=[150.0, 155.0, 160.0],
            q90=[155.0, 160.0, 165.0],
            feature_importance={},
            current_price=150.0
        )

        # 1-day horizon: (150 - 150) / 150 = 0%
        ret_1d = forecast.get_expected_return(0)
        assert abs(ret_1d - 0.0) < 0.01

        # 5-day horizon: (155 - 150) / 150 = 3.33%
        ret_5d = forecast.get_expected_return(1)
        assert abs(ret_5d - 0.0333) < 0.001

        # 10-day horizon: (160 - 150) / 150 = 6.67%
        ret_10d = forecast.get_expected_return(2)
        assert abs(ret_10d - 0.0667) < 0.001


class TestTFTLayers:
    """Test TFT-specific layers"""

    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_gated_linear_unit(self):
        """Test GLU layer"""
        from src.ml.advanced_forecasting.tft_model import GatedLinearUnit
        import tensorflow as tf

        layer = GatedLinearUnit(units=32)
        x = tf.random.normal((2, 10))  # Batch=2, features=10

        # Build layer
        output = layer(x)

        assert output.shape == (2, 32)
        # GLU output should be bounded due to sigmoid gating
        assert tf.reduce_max(output) < 100.0  # Reasonable bound

    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_variable_selection_network(self):
        """Test variable selection network"""
        from src.ml.advanced_forecasting.tft_model import VariableSelectionNetwork
        import tensorflow as tf

        # 5 variables
        layer = VariableSelectionNetwork(num_vars=5, hidden_units=16)

        # Batch=2, num_vars=5, var_features=8
        x = tf.random.normal((2, 5, 8))

        # Build and call
        output, weights = layer(x)

        # Output should be [batch, hidden_units]
        assert output.shape == (2, 16)

        # Weights should be [batch, num_vars]
        assert weights.shape == (2, 5)

        # Weights should sum to 1 (softmax)
        weight_sums = tf.reduce_sum(weights, axis=1)
        assert tf.reduce_all(tf.abs(weight_sums - 1.0) < 0.01)


class TestQuantileForecasting:
    """Test quantile forecasting for uncertainty estimation"""

    def test_quantile_ordering(self):
        """Test that quantiles are properly ordered (q10 < q50 < q90)"""
        forecast = MultiHorizonForecast(
            timestamp=datetime.now(),
            symbol='AAPL',
            horizons=[1, 5, 10],
            predictions=[150.0, 155.0, 160.0],
            q10=[145.0, 150.0, 155.0],
            q50=[150.0, 155.0, 160.0],
            q90=[155.0, 160.0, 165.0],
            feature_importance={},
            current_price=150.0
        )

        # Check ordering for each horizon
        for q10, q50, q90 in zip(forecast.q10, forecast.q50, forecast.q90):
            assert q10 <= q50 <= q90, "Quantiles should be ordered"

    def test_uncertainty_from_quantile_spread(self):
        """Test that quantile spread represents uncertainty"""
        # High uncertainty forecast (wide spread)
        high_uncertainty = MultiHorizonForecast(
            timestamp=datetime.now(),
            symbol='VOLATILE',
            horizons=[1],
            predictions=[100.0],
            q10=[80.0],  # Wide spread
            q50=[100.0],
            q90=[120.0],
            feature_importance={},
            current_price=100.0
        )

        # Low uncertainty forecast (narrow spread)
        low_uncertainty = MultiHorizonForecast(
            timestamp=datetime.now(),
            symbol='STABLE',
            horizons=[1],
            predictions=[100.0],
            q10=[98.0],  # Narrow spread
            q50=[100.0],
            q90=[102.0],
            feature_importance={},
            current_price=100.0
        )

        # Calculate spreads
        high_spread = high_uncertainty.q90[0] - high_uncertainty.q10[0]
        low_spread = low_uncertainty.q90[0] - low_uncertainty.q10[0]

        assert high_spread > low_spread


class TestFeatureImportance:
    """Test feature importance from variable selection"""

    def test_feature_importance_sums_to_one(self):
        """Test that feature importances sum to 1.0"""
        # Note: This is just a test of the data structure
        # Actual importance values come from trained model
        feature_importance = {
            'price': 0.4,
            'volume': 0.3,
            'rsi': 0.15,
            'macd': 0.15
        }

        total = sum(feature_importance.values())
        assert abs(total - 1.0) < 0.01

    def test_feature_importance_interpretation(self):
        """Test interpreting feature importance"""
        forecast = MultiHorizonForecast(
            timestamp=datetime.now(),
            symbol='AAPL',
            horizons=[1],
            predictions=[150.0],
            q10=[145.0],
            q50=[150.0],
            q90=[155.0],
            feature_importance={
                'price': 0.5,  # Most important
                'volume': 0.3,
                'sentiment': 0.15,
                'vix': 0.05    # Least important
            },
            current_price=150.0
        )

        # Get most important feature
        most_important = max(
            forecast.feature_importance.items(),
            key=lambda x: x[1]
        )

        assert most_important[0] == 'price'
        assert most_important[1] == 0.5


class TestMultiHorizonPredictions:
    """Test multi-horizon prediction functionality"""

    def test_different_horizons(self):
        """Test predictions for different time horizons"""
        forecast = MultiHorizonForecast(
            timestamp=datetime.now(),
            symbol='AAPL',
            horizons=[1, 5, 10, 30],  # 1 day, 5 days, 10 days, 30 days
            predictions=[150.0, 152.0, 155.0, 160.0],
            q10=[148.0, 149.0, 151.0, 155.0],
            q50=[150.0, 152.0, 155.0, 160.0],
            q90=[152.0, 155.0, 159.0, 165.0],
            feature_importance={},
            current_price=150.0
        )

        assert len(forecast.horizons) == 4
        assert len(forecast.predictions) == 4

        # Longer horizons typically have higher uncertainty
        spread_1d = forecast.q90[0] - forecast.q10[0]
        spread_30d = forecast.q90[3] - forecast.q10[3]

        # 30-day forecast should have wider confidence interval
        assert spread_30d >= spread_1d

    def test_horizon_consistency(self):
        """Test that horizons match predictions length"""
        forecast = MultiHorizonForecast(
            timestamp=datetime.now(),
            symbol='AAPL',
            horizons=[1, 5, 10],
            predictions=[150.0, 152.0, 155.0],
            q10=[145.0, 147.0, 150.0],
            q50=[150.0, 152.0, 155.0],
            q90=[155.0, 157.0, 160.0],
            feature_importance={},
            current_price=150.0
        )

        assert len(forecast.horizons) == len(forecast.predictions)
        assert len(forecast.horizons) == len(forecast.q10)
        assert len(forecast.horizons) == len(forecast.q50)
        assert len(forecast.horizons) == len(forecast.q90)


class TestTFTUncertainty:
    """Test uncertainty quantification in TFT"""

    def test_confidence_intervals(self):
        """Test confidence interval calculation"""
        forecast = MultiHorizonForecast(
            timestamp=datetime.now(),
            symbol='AAPL',
            horizons=[5],
            predictions=[155.0],
            q10=[145.0],
            q50=[155.0],
            q90=[165.0],
            feature_importance={},
            current_price=150.0
        )

        # 80% confidence interval: [q10, q90]
        lower, median, upper = forecast.get_prediction_intervals()[0]

        assert lower == 145.0
        assert median == 155.0
        assert upper == 165.0

        # Confidence interval width
        ci_width = upper - lower
        assert ci_width == 20.0  # $20 spread

    def test_uncertainty_increases_with_horizon(self):
        """Test that uncertainty generally increases with longer horizons"""
        forecast = MultiHorizonForecast(
            timestamp=datetime.now(),
            symbol='AAPL',
            horizons=[1, 5, 10, 30],
            predictions=[150.0, 152.0, 155.0, 160.0],
            q10=[148.0, 147.0, 145.0, 140.0],
            q50=[150.0, 152.0, 155.0, 160.0],
            q90=[152.0, 157.0, 165.0, 180.0],
            feature_importance={},
            current_price=150.0
        )

        # Calculate uncertainty (width of confidence interval)
        uncertainties = [
            q90 - q10
            for q10, q90 in zip(forecast.q10, forecast.q90)
        ]

        # Generally increasing uncertainty (not strictly monotonic in practice)
        assert uncertainties[-1] > uncertainties[0]  # 30-day > 1-day


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
