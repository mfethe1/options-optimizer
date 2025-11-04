"""
Advanced Forecasting Module - Priority #1

Implements state-of-the-art forecasting architectures:
1. Temporal Fusion Transformer (TFT) - Proven 11% improvement
2. TimesFM Foundation Model - Google's 200M parameter model
3. Conformal Prediction - Guaranteed uncertainty quantification
"""

from .tft_model import (
    TemporalFusionTransformer,
    TFTPredictor,
    MultiHorizonForecast
)

from .conformal_prediction import (
    ConformalPredictor,
    PredictionInterval,
    AdaptiveConformalPredictor
)

__all__ = [
    'TemporalFusionTransformer',
    'TFTPredictor',
    'MultiHorizonForecast',
    'ConformalPredictor',
    'PredictionInterval',
    'AdaptiveConformalPredictor'
]
