"""
Ensemble Neural Network Predictor

Combines all 5 neural network models for superior predictions:
1. Epidemic Volatility (SIR/SEIR + PINN)
2. Temporal Fusion Transformer (TFT) + Conformal Prediction
3. Graph Neural Networks (GNN)
4. Mamba State Space Model
5. Physics-Informed Neural Networks (PINN)

Ensemble Methods:
- Weighted averaging for price predictions
- Voting for trading signals (BUY/SELL/HOLD)
- Adaptive weighting based on recent performance
- Regime-aware model selection
- Uncertainty quantification via model agreement

Research: "Ensemble Methods in Machine Learning" (Dietterich, 2000)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TradingSignal(Enum):
    """Trading signal enum"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class TimeHorizon(Enum):
    """Trading time horizon"""
    INTRADAY = "intraday"      # Minutes to hours
    SHORT_TERM = "short_term"  # 1-5 days
    MEDIUM_TERM = "medium_term" # 5-30 days
    LONG_TERM = "long_term"    # 30+ days


@dataclass
class ModelPrediction:
    """Prediction from a single model"""
    model_name: str
    price_prediction: float
    confidence: float
    signal: TradingSignal
    timestamp: datetime
    horizon_days: int = 1

    # Optional: confidence interval
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

    # Optional: additional metadata
    metadata: Optional[Dict] = None


@dataclass
class EnsemblePrediction:
    """Ensemble prediction combining all models"""
    symbol: str
    timestamp: datetime
    current_price: float
    time_horizon: TimeHorizon

    # Ensemble results
    ensemble_prediction: float
    ensemble_signal: TradingSignal
    ensemble_confidence: float

    # Individual model predictions
    model_predictions: List[ModelPrediction]

    # Model weights used
    model_weights: Dict[str, float]

    # Uncertainty metrics
    prediction_std: float  # Standard deviation across models
    model_agreement: float  # 0-1, how much models agree

    # Recommendation
    position_size: float  # 0-1, suggested position size
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class PerformanceTracker:
    """
    Track historical performance of each model for adaptive weighting
    """

    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Number of recent predictions to track
        """
        self.window_size = window_size
        self.history: Dict[str, List[Dict]] = {}

    def record_prediction(
        self,
        model_name: str,
        predicted: float,
        actual: float,
        timestamp: datetime
    ):
        """Record a prediction for performance tracking"""
        if model_name not in self.history:
            self.history[model_name] = []

        error = abs(predicted - actual) / actual  # Percentage error

        self.history[model_name].append({
            'timestamp': timestamp,
            'predicted': predicted,
            'actual': actual,
            'error': error,
            'squared_error': error ** 2
        })

        # Keep only recent window
        if len(self.history[model_name]) > self.window_size:
            self.history[model_name] = self.history[model_name][-self.window_size:]

    def get_model_accuracy(self, model_name: str) -> float:
        """
        Get recent accuracy for a model (1 - MAPE)

        Returns:
            accuracy: 0-1, where 1 is perfect
        """
        if model_name not in self.history or len(self.history[model_name]) == 0:
            return 0.5  # Default: neutral

        recent = self.history[model_name][-self.window_size:]
        mape = np.mean([h['error'] for h in recent])

        # Convert MAPE to accuracy (clamp to reasonable range)
        accuracy = max(0.0, min(1.0, 1 - mape))

        return accuracy

    def get_model_sharpe(self, model_name: str) -> float:
        """
        Calculate Sharpe-like ratio for model predictions

        Returns:
            sharpe: Higher is better
        """
        if model_name not in self.history or len(self.history[model_name]) < 2:
            return 0.0

        recent = self.history[model_name][-self.window_size:]

        # Calculate returns if we followed model predictions
        returns = []
        for i in range(1, len(recent)):
            pred_return = (recent[i]['predicted'] - recent[i-1]['actual']) / recent[i-1]['actual']
            actual_return = (recent[i]['actual'] - recent[i-1]['actual']) / recent[i-1]['actual']

            # Return if we traded based on prediction
            trade_return = pred_return * actual_return  # Simplified
            returns.append(trade_return)

        if len(returns) == 0:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns) + 1e-8

        sharpe = mean_return / std_return

        return sharpe

    def get_all_accuracies(self) -> Dict[str, float]:
        """Get accuracies for all tracked models"""
        return {
            model: self.get_model_accuracy(model)
            for model in self.history.keys()
        }


class EnsemblePredictor:
    """
    Ensemble predictor combining all 5 neural network models

    Features:
    - Weighted averaging for price predictions
    - Voting for trading signals
    - Adaptive weights based on performance
    - Regime-aware model selection
    - Intraday and long-term modes
    """

    def __init__(
        self,
        weighting_method: str = 'adaptive',  # 'equal', 'performance', 'adaptive'
        voting_threshold: float = 0.6,  # 60% of models must agree
        track_performance: bool = True
    ):
        """
        Args:
            weighting_method: How to weight model predictions
            voting_threshold: Fraction of models that must agree for signal
            track_performance: Whether to track and adapt to performance
        """
        self.weighting_method = weighting_method
        self.voting_threshold = voting_threshold
        self.track_performance = track_performance

        # Model names
        self.models = [
            'epidemic_volatility',
            'tft_conformal',
            'gnn',
            'mamba',
            'pinn'
        ]

        # Initialize weights
        if weighting_method == 'equal':
            self.weights = {model: 1.0 / len(self.models) for model in self.models}
        else:
            # Start with equal, will adapt
            self.weights = {model: 1.0 / len(self.models) for model in self.models}

        # Performance tracker
        self.performance_tracker = PerformanceTracker() if track_performance else None

        logger.info(f"Ensemble predictor initialized with {weighting_method} weighting")

    def update_weights_from_performance(self):
        """Update model weights based on recent performance"""
        if not self.performance_tracker:
            return

        accuracies = self.performance_tracker.get_all_accuracies()

        if len(accuracies) == 0:
            return

        # Softmax weighting based on accuracy
        exp_accuracies = {k: np.exp(v * 5) for k, v in accuracies.items()}  # Scale factor
        total = sum(exp_accuracies.values()) + 1e-8

        new_weights = {k: v / total for k, v in exp_accuracies.items()}

        # Update weights (with momentum to avoid drastic changes)
        momentum = 0.7
        for model in self.models:
            if model in new_weights:
                old_weight = self.weights.get(model, 1.0 / len(self.models))
                self.weights[model] = momentum * old_weight + (1 - momentum) * new_weights[model]

        # Normalize
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

        logger.info(f"Updated ensemble weights: {self.weights}")

    def adjust_weights_for_regime(
        self,
        market_regime: str
    ) -> Dict[str, float]:
        """
        Adjust model weights based on market regime

        Args:
            market_regime: 'volatile', 'calm', 'trending', 'ranging'
        Returns:
            adjusted_weights: Regime-specific weights
        """
        adjusted = self.weights.copy()

        if market_regime == 'volatile':
            # Boost epidemic volatility model
            adjusted['epidemic_volatility'] *= 2.0
            adjusted['pinn'] *= 1.5  # PINN helps with risk modeling

        elif market_regime == 'calm':
            # Boost TFT and Mamba for pattern recognition
            adjusted['tft_conformal'] *= 1.5
            adjusted['mamba'] *= 1.5

        elif market_regime == 'trending':
            # Boost Mamba (long sequences) and GNN (correlations)
            adjusted['mamba'] *= 1.8
            adjusted['gnn'] *= 1.3

        elif market_regime == 'ranging':
            # Balanced, slight boost to TFT
            adjusted['tft_conformal'] *= 1.2

        # Normalize
        total = sum(adjusted.values())
        adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def adjust_weights_for_horizon(
        self,
        time_horizon: TimeHorizon
    ) -> Dict[str, float]:
        """
        Adjust model weights based on time horizon

        Args:
            time_horizon: Trading time horizon
        Returns:
            adjusted_weights: Horizon-specific weights
        """
        adjusted = self.weights.copy()

        if time_horizon == TimeHorizon.INTRADAY:
            # Mamba excels at long sequences (intraday ticks)
            # Epidemic for volatility spikes
            adjusted['mamba'] *= 2.5
            adjusted['epidemic_volatility'] *= 1.5
            adjusted['tft_conformal'] *= 0.7  # Less relevant

        elif time_horizon == TimeHorizon.SHORT_TERM:
            # TFT and GNN excel here
            adjusted['tft_conformal'] *= 2.0
            adjusted['gnn'] *= 1.8
            adjusted['mamba'] *= 1.2

        elif time_horizon == TimeHorizon.MEDIUM_TERM:
            # Balanced, slight boost to TFT and GNN
            adjusted['tft_conformal'] *= 1.5
            adjusted['gnn'] *= 1.3
            adjusted['pinn'] *= 1.2

        elif time_horizon == TimeHorizon.LONG_TERM:
            # PINN for fundamentals, GNN for correlations
            adjusted['pinn'] *= 1.8
            adjusted['gnn'] *= 1.5
            adjusted['epidemic_volatility'] *= 0.8  # Less relevant

        # Normalize
        total = sum(adjusted.values())
        adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def aggregate_price_predictions(
        self,
        predictions: List[ModelPrediction],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[float, float]:
        """
        Aggregate price predictions using weighted average

        Args:
            predictions: List of model predictions
            weights: Optional custom weights (defaults to self.weights)
        Returns:
            (ensemble_price, prediction_std)
        """
        if weights is None:
            weights = self.weights

        # Weighted average
        total_weight = 0.0
        weighted_sum = 0.0

        prices = []

        for pred in predictions:
            weight = weights.get(pred.model_name, 0.0)
            weighted_sum += pred.price_prediction * weight
            total_weight += weight
            prices.append(pred.price_prediction)

        ensemble_price = weighted_sum / (total_weight + 1e-8)

        # Standard deviation across predictions
        prediction_std = np.std(prices) if len(prices) > 1 else 0.0

        return ensemble_price, prediction_std

    def aggregate_signals(
        self,
        predictions: List[ModelPrediction],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[TradingSignal, float]:
        """
        Aggregate trading signals using weighted voting

        Args:
            predictions: List of model predictions
            weights: Optional custom weights
        Returns:
            (ensemble_signal, agreement_score)
        """
        if weights is None:
            weights = self.weights

        # Map signals to numeric scores
        signal_scores = {
            TradingSignal.STRONG_SELL: -2,
            TradingSignal.SELL: -1,
            TradingSignal.HOLD: 0,
            TradingSignal.BUY: 1,
            TradingSignal.STRONG_BUY: 2
        }

        # Weighted vote
        total_weight = 0.0
        weighted_score = 0.0

        for pred in predictions:
            weight = weights.get(pred.model_name, 0.0) * pred.confidence
            score = signal_scores[pred.signal]
            weighted_score += score * weight
            total_weight += weight

        avg_score = weighted_score / (total_weight + 1e-8)

        # Convert back to signal
        if avg_score >= 1.5:
            ensemble_signal = TradingSignal.STRONG_BUY
        elif avg_score >= 0.5:
            ensemble_signal = TradingSignal.BUY
        elif avg_score <= -1.5:
            ensemble_signal = TradingSignal.STRONG_SELL
        elif avg_score <= -0.5:
            ensemble_signal = TradingSignal.SELL
        else:
            ensemble_signal = TradingSignal.HOLD

        # Calculate agreement (how concentrated votes are)
        vote_counts = {}
        for pred in predictions:
            sig = pred.signal.value
            vote_counts[sig] = vote_counts.get(sig, 0) + weights.get(pred.model_name, 0.0)

        max_vote = max(vote_counts.values()) if vote_counts else 0
        agreement = max_vote / (total_weight + 1e-8)

        return ensemble_signal, agreement

    def calculate_model_agreement(
        self,
        predictions: List[ModelPrediction]
    ) -> float:
        """
        Calculate how much models agree (0-1)

        High agreement = low uncertainty
        Low agreement = high uncertainty

        Returns:
            agreement: 0-1, where 1 is perfect agreement
        """
        if len(predictions) < 2:
            return 1.0

        prices = [p.price_prediction for p in predictions]

        # Coefficient of variation (CV)
        mean_price = np.mean(prices)
        std_price = np.std(prices)

        cv = std_price / (mean_price + 1e-8)

        # Convert to agreement (lower CV = higher agreement)
        agreement = max(0.0, min(1.0, 1.0 - cv * 10))  # Scale factor

        return agreement

    def calculate_position_size(
        self,
        ensemble_confidence: float,
        model_agreement: float,
        signal_strength: float
    ) -> float:
        """
        Calculate suggested position size based on confidence metrics

        Args:
            ensemble_confidence: Overall confidence (0-1)
            model_agreement: How much models agree (0-1)
            signal_strength: Signal strength (-2 to 2)
        Returns:
            position_size: 0-1, where 1 is full position
        """
        # Kelly criterion inspired, but conservative

        # Base size from confidence and agreement
        base_size = (ensemble_confidence + model_agreement) / 2

        # Scale by signal strength
        size = base_size * min(abs(signal_strength) / 2, 1.0)

        # Cap at reasonable levels
        max_size = 0.5  # Never more than 50% of portfolio
        size = min(size, max_size)

        # Minimum threshold
        if size < 0.1:
            size = 0.0  # Don't trade if too uncertain

        return size

    async def predict(
        self,
        symbol: str,
        current_price: float,
        model_predictions: List[ModelPrediction],
        time_horizon: TimeHorizon = TimeHorizon.SHORT_TERM,
        market_regime: Optional[str] = None
    ) -> EnsemblePrediction:
        """
        Generate ensemble prediction

        Args:
            symbol: Stock symbol
            current_price: Current stock price
            model_predictions: Predictions from individual models
            time_horizon: Trading time horizon
            market_regime: Current market regime (optional)
        Returns:
            ensemble_prediction: Combined prediction
        """
        # Update weights if adaptive
        if self.weighting_method == 'adaptive' and self.performance_tracker:
            self.update_weights_from_performance()

        # Adjust weights for time horizon
        horizon_weights = self.adjust_weights_for_horizon(time_horizon)

        # Adjust weights for market regime
        if market_regime:
            horizon_weights = self.adjust_weights_for_regime(market_regime)

        # Aggregate price predictions
        ensemble_price, prediction_std = self.aggregate_price_predictions(
            model_predictions,
            weights=horizon_weights
        )

        # Aggregate trading signals
        ensemble_signal, signal_agreement = self.aggregate_signals(
            model_predictions,
            weights=horizon_weights
        )

        # Calculate model agreement
        model_agreement = self.calculate_model_agreement(model_predictions)

        # Calculate ensemble confidence
        confidences = [p.confidence for p in model_predictions]
        ensemble_confidence = np.mean(confidences) * model_agreement

        # Calculate position size
        signal_scores = {
            TradingSignal.STRONG_SELL: -2,
            TradingSignal.SELL: -1,
            TradingSignal.HOLD: 0,
            TradingSignal.BUY: 1,
            TradingSignal.STRONG_BUY: 2
        }
        signal_strength = signal_scores[ensemble_signal]

        position_size = self.calculate_position_size(
            ensemble_confidence,
            model_agreement,
            signal_strength
        )

        # Calculate stop loss and take profit
        risk_multiple = 2.0  # Risk/reward ratio
        stop_loss = None
        take_profit = None

        if ensemble_signal in [TradingSignal.BUY, TradingSignal.STRONG_BUY]:
            # Use prediction std for stop/target
            stop_distance = prediction_std * 1.5
            stop_loss = current_price - stop_distance
            take_profit = current_price + (stop_distance * risk_multiple)

        elif ensemble_signal in [TradingSignal.SELL, TradingSignal.STRONG_SELL]:
            stop_distance = prediction_std * 1.5
            stop_loss = current_price + stop_distance
            take_profit = current_price - (stop_distance * risk_multiple)

        return EnsemblePrediction(
            symbol=symbol,
            timestamp=datetime.now(),
            current_price=current_price,
            time_horizon=time_horizon,
            ensemble_prediction=ensemble_price,
            ensemble_signal=ensemble_signal,
            ensemble_confidence=ensemble_confidence,
            model_predictions=model_predictions,
            model_weights=horizon_weights,
            prediction_std=prediction_std,
            model_agreement=model_agreement,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

    def record_actual_price(
        self,
        predictions: List[ModelPrediction],
        actual_price: float,
        timestamp: datetime
    ):
        """
        Record actual price for performance tracking

        Args:
            predictions: Previous predictions to evaluate
            actual_price: Actual price observed
            timestamp: When price was observed
        """
        if not self.performance_tracker:
            return

        for pred in predictions:
            self.performance_tracker.record_prediction(
                model_name=pred.model_name,
                predicted=pred.price_prediction,
                actual=actual_price,
                timestamp=timestamp
            )
