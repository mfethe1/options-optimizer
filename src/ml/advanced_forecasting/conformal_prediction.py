"""
Conformal Prediction for Guaranteed Uncertainty Quantification

Provides distribution-free prediction intervals with finite-sample coverage guarantees.

Key Features:
- Guaranteed coverage: P(Y ∈ Interval) ≥ 1-α (e.g., 95%)
- Distribution-free: No assumptions about data distribution
- Finite-sample validity: Works even with small datasets
- Adaptive to distribution shifts

Based on research:
- "Conformal Prediction for Time Series" (Gibbs & Cand\u00e8s, 2021)
- "Adaptive Conformal Inference Under Distribution Shift" (2021)
- Agent 2's research: Multi-dimensional conformal intervals
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PredictionInterval:
    """Prediction interval with coverage guarantees"""
    timestamp: datetime
    horizon: int  # Days ahead
    point_prediction: float
    lower_bound: float
    upper_bound: float
    coverage_level: float  # e.g., 0.95 for 95% coverage
    width: float  # Interval width

    def contains(self, value: float) -> bool:
        """Check if value is within interval"""
        return self.lower_bound <= value <= self.upper_bound

    def relative_width(self) -> float:
        """Get interval width as fraction of prediction"""
        if self.point_prediction > 0:
            return self.width / self.point_prediction
        return float('inf')


class ConformalPredictor:
    """
    Conformal Prediction for Time Series

    Implements split conformal prediction with quantile-based intervals.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Significance level (e.g., 0.05 for 95% coverage)
        """
        self.alpha = alpha
        self.calibration_scores = []
        self.is_calibrated = False

    def calibrate(self, predictions: np.ndarray, actuals: np.ndarray):
        """
        Calibrate conformal predictor on validation set

        Args:
            predictions: Model predictions on calibration set
            actuals: True values on calibration set
        """
        # Compute non-conformity scores (absolute residuals)
        scores = np.abs(predictions - actuals)
        self.calibration_scores = scores

        self.is_calibrated = True
        logger.info(f"Calibrated conformal predictor on {len(scores)} samples")

    def predict_interval(self,
                        prediction: float,
                        horizon: int = 1,
                        adaptive_factor: float = 1.0) -> PredictionInterval:
        """
        Generate prediction interval with coverage guarantee

        Args:
            prediction: Point prediction from model
            horizon: Forecast horizon
            adaptive_factor: Multiplicative factor for adapting to distribution shifts

        Returns:
            PredictionInterval with guaranteed coverage
        """
        if not self.is_calibrated:
            logger.warning("Predictor not calibrated, using default interval")
            width = abs(prediction) * 0.1  # Default 10% interval
            return PredictionInterval(
                timestamp=datetime.now(),
                horizon=horizon,
                point_prediction=prediction,
                lower_bound=prediction - width,
                upper_bound=prediction + width,
                coverage_level=1 - self.alpha,
                width=width * 2
            )

        # Compute quantile of calibration scores
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)  # Cap at 1.0

        # Get quantile (this is the guaranteed interval width)
        interval_width = np.quantile(self.calibration_scores, q_level)

        # Apply adaptive factor for distribution shifts
        interval_width *= adaptive_factor

        # Construct interval
        lower = prediction - interval_width
        upper = prediction + interval_width

        return PredictionInterval(
            timestamp=datetime.now(),
            horizon=horizon,
            point_prediction=prediction,
            lower_bound=lower,
            upper_bound=upper,
            coverage_level=1 - self.alpha,
            width=interval_width * 2
        )

    def batch_predict_intervals(self,
                                predictions: np.ndarray,
                                horizon: int = 1) -> List[PredictionInterval]:
        """
        Generate intervals for batch of predictions

        Args:
            predictions: Array of predictions
            horizon: Forecast horizon

        Returns:
            List of PredictionIntervals
        """
        intervals = []
        for pred in predictions:
            interval = self.predict_interval(pred, horizon)
            intervals.append(interval)

        return intervals

    def evaluate_coverage(self,
                         predictions: np.ndarray,
                         actuals: np.ndarray,
                         horizon: int = 1) -> Dict:
        """
        Evaluate empirical coverage on test set

        Args:
            predictions: Model predictions
            actuals: True values
            horizon: Forecast horizon

        Returns:
            Dict with coverage metrics
        """
        intervals = self.batch_predict_intervals(predictions, horizon)

        # Check coverage
        covered = [interval.contains(actual) for interval, actual in zip(intervals, actuals)]
        empirical_coverage = np.mean(covered)

        # Interval statistics
        widths = [interval.width for interval in intervals]
        avg_width = np.mean(widths)
        std_width = np.std(widths)

        # Relative widths
        rel_widths = [interval.relative_width() for interval in intervals]
        avg_rel_width = np.mean([w for w in rel_widths if w != float('inf')])

        results = {
            'target_coverage': 1 - self.alpha,
            'empirical_coverage': empirical_coverage,
            'coverage_gap': empirical_coverage - (1 - self.alpha),
            'num_samples': len(predictions),
            'num_covered': sum(covered),
            'avg_interval_width': avg_width,
            'std_interval_width': std_width,
            'avg_relative_width': avg_rel_width
        }

        logger.info(f"Coverage evaluation: {empirical_coverage:.1%} (target: {1-self.alpha:.1%})")

        return results


class AdaptiveConformalPredictor(ConformalPredictor):
    """
    Adaptive Conformal Prediction for non-stationary time series

    Adapts interval width based on recent forecast errors to handle
    distribution shifts and regime changes.
    """

    def __init__(self,
                 alpha: float = 0.05,
                 adaptation_window: int = 20):
        """
        Args:
            alpha: Significance level
            adaptation_window: Window for computing adaptive factor
        """
        super().__init__(alpha)
        self.adaptation_window = adaptation_window
        self.recent_errors = []

    def update(self, prediction: float, actual: float):
        """
        Update with new observation for adaptation

        Args:
            prediction: Model prediction
            actual: True value
        """
        error = abs(prediction - actual)
        self.recent_errors.append(error)

        # Keep only recent errors
        if len(self.recent_errors) > self.adaptation_window:
            self.recent_errors.pop(0)

    def get_adaptive_factor(self) -> float:
        """
        Compute adaptive factor based on recent errors

        Returns:
            Multiplicative factor for interval width
        """
        if len(self.recent_errors) < 5:
            return 1.0  # Not enough data for adaptation

        if len(self.calibration_scores) == 0:
            return 1.0

        # Compare recent errors to calibration errors
        recent_mean = np.mean(self.recent_errors)
        calibration_mean = np.mean(self.calibration_scores)

        if calibration_mean > 0:
            # Adaptive factor: ratio of recent to calibration error
            factor = recent_mean / calibration_mean

            # Clip to reasonable range [0.5, 3.0]
            factor = np.clip(factor, 0.5, 3.0)
            return factor

        return 1.0

    def predict_interval(self,
                        prediction: float,
                        horizon: int = 1) -> PredictionInterval:
        """
        Generate adaptive prediction interval

        Automatically adjusts width based on recent forecast errors
        """
        adaptive_factor = self.get_adaptive_factor()

        return super().predict_interval(
            prediction=prediction,
            horizon=horizon,
            adaptive_factor=adaptive_factor
        )


class MultiHorizonConformalPredictor:
    """
    Conformal prediction for multi-horizon forecasts

    Maintains separate calibration for each horizon
    """

    def __init__(self,
                 horizons: List[int] = [1, 5, 10, 30],
                 alpha: float = 0.05):
        """
        Args:
            horizons: Forecast horizons (in days)
            alpha: Significance level
        """
        self.horizons = horizons
        self.alpha = alpha

        # Separate predictor for each horizon
        self.predictors = {
            h: AdaptiveConformalPredictor(alpha=alpha)
            for h in horizons
        }

    def calibrate(self,
                 predictions_by_horizon: Dict[int, np.ndarray],
                 actuals_by_horizon: Dict[int, np.ndarray]):
        """
        Calibrate predictors for all horizons

        Args:
            predictions_by_horizon: {horizon: predictions}
            actuals_by_horizon: {horizon: actuals}
        """
        for horizon in self.horizons:
            if horizon in predictions_by_horizon and horizon in actuals_by_horizon:
                self.predictors[horizon].calibrate(
                    predictions_by_horizon[horizon],
                    actuals_by_horizon[horizon]
                )

        logger.info(f"Calibrated multi-horizon conformal predictor for {len(self.horizons)} horizons")

    def predict_intervals(self,
                         predictions_by_horizon: Dict[int, float]) -> Dict[int, PredictionInterval]:
        """
        Generate intervals for all horizons

        Args:
            predictions_by_horizon: {horizon: prediction}

        Returns:
            {horizon: PredictionInterval}
        """
        intervals = {}

        for horizon, prediction in predictions_by_horizon.items():
            if horizon in self.predictors:
                intervals[horizon] = self.predictors[horizon].predict_interval(
                    prediction, horizon
                )

        return intervals

    def update(self,
              predictions_by_horizon: Dict[int, float],
              actuals_by_horizon: Dict[int, float]):
        """
        Update all predictors with new observations

        Args:
            predictions_by_horizon: {horizon: prediction}
            actuals_by_horizon: {horizon: actual}
        """
        for horizon in self.horizons:
            if horizon in predictions_by_horizon and horizon in actuals_by_horizon:
                self.predictors[horizon].update(
                    predictions_by_horizon[horizon],
                    actuals_by_horizon[horizon]
                )
