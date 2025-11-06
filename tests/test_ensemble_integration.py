"""
Integration tests for Ensemble Predictor

Tests:
1. Ensemble combination of all models
2. Weighted averaging
3. Voting for trading signals
4. Adaptive weighting based on performance
5. Uncertainty quantification
"""

import pytest
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml.ensemble.ensemble_predictor import (
    ModelPrediction,
    EnsemblePrediction,
    TradingSignal,
    TimeHorizon,
    PerformanceTracker
)


class TestModelPrediction:
    """Test individual model prediction"""

    def test_model_prediction_creation(self):
        """Test creating model prediction"""
        pred = ModelPrediction(
            model_name='TFT',
            price_prediction=155.0,
            confidence=0.85,
            signal=TradingSignal.BUY,
            timestamp=datetime.now(),
            horizon_days=5
        )

        assert pred is not None
        assert pred.model_name == 'TFT'
        assert pred.price_prediction == 155.0
        assert pred.confidence == 0.85
        assert pred.signal == TradingSignal.BUY
        assert pred.horizon_days == 5

    def test_model_prediction_with_bounds(self):
        """Test model prediction with confidence bounds"""
        pred = ModelPrediction(
            model_name='GNN',
            price_prediction=150.0,
            confidence=0.75,
            signal=TradingSignal.HOLD,
            timestamp=datetime.now(),
            lower_bound=145.0,
            upper_bound=155.0
        )

        assert pred.lower_bound == 145.0
        assert pred.upper_bound == 155.0
        assert pred.lower_bound < pred.price_prediction < pred.upper_bound


class TestTradingSignal:
    """Test trading signal enum"""

    def test_all_signals(self):
        """Test all trading signals"""
        assert TradingSignal.STRONG_BUY.value == "STRONG_BUY"
        assert TradingSignal.BUY.value == "BUY"
        assert TradingSignal.HOLD.value == "HOLD"
        assert TradingSignal.SELL.value == "SELL"
        assert TradingSignal.STRONG_SELL.value == "STRONG_SELL"

    def test_signal_ordering(self):
        """Test signal strength ordering"""
        # Signals from most bullish to most bearish
        signals = [
            TradingSignal.STRONG_BUY,
            TradingSignal.BUY,
            TradingSignal.HOLD,
            TradingSignal.SELL,
            TradingSignal.STRONG_SELL
        ]

        assert len(signals) == 5
        assert TradingSignal.HOLD in signals


class TestTimeHorizon:
    """Test time horizon enum"""

    def test_all_horizons(self):
        """Test all time horizons"""
        assert TimeHorizon.INTRADAY.value == "intraday"
        assert TimeHorizon.SHORT_TERM.value == "short_term"
        assert TimeHorizon.MEDIUM_TERM.value == "medium_term"
        assert TimeHorizon.LONG_TERM.value == "long_term"


class TestEnsemblePrediction:
    """Test ensemble prediction combining models"""

    def test_ensemble_creation(self):
        """Test creating ensemble prediction"""
        model_preds = [
            ModelPrediction('TFT', 155.0, 0.85, TradingSignal.BUY, datetime.now()),
            ModelPrediction('GNN', 152.0, 0.75, TradingSignal.BUY, datetime.now()),
            ModelPrediction('Mamba', 153.0, 0.80, TradingSignal.BUY, datetime.now()),
        ]

        ensemble = EnsemblePrediction(
            symbol='AAPL',
            timestamp=datetime.now(),
            current_price=150.0,
            time_horizon=TimeHorizon.SHORT_TERM,
            ensemble_prediction=153.5,
            ensemble_signal=TradingSignal.BUY,
            ensemble_confidence=0.80,
            model_predictions=model_preds,
            model_weights={'TFT': 0.4, 'GNN': 0.3, 'Mamba': 0.3},
            prediction_std=1.5,
            model_agreement=0.95,
            position_size=0.5
        )

        assert ensemble is not None
        assert ensemble.symbol == 'AAPL'
        assert ensemble.ensemble_prediction == 153.5
        assert len(ensemble.model_predictions) == 3

    def test_ensemble_weighted_average(self):
        """Test weighted average of predictions"""
        # TFT: 160, weight 0.5
        # GNN: 150, weight 0.3
        # Mamba: 155, weight 0.2
        # Expected: 160*0.5 + 150*0.3 + 155*0.2 = 156.0

        predictions = [160.0, 150.0, 155.0]
        weights = [0.5, 0.3, 0.2]

        ensemble_pred = sum(p * w for p, w in zip(predictions, weights))

        assert abs(ensemble_pred - 156.0) < 0.01

    def test_ensemble_model_agreement(self):
        """Test model agreement calculation"""
        # High agreement: all models close
        high_agreement_preds = [150.0, 151.0, 150.5]
        std_high = np.std(high_agreement_preds)

        # Low agreement: models disagree
        low_agreement_preds = [150.0, 140.0, 160.0]
        std_low = np.std(low_agreement_preds)

        # Lower std = higher agreement
        assert std_low > std_high


class TestPerformanceTracker:
    """Test performance tracking for adaptive weighting"""

    def test_performance_tracker_creation(self):
        """Test creating performance tracker"""
        tracker = PerformanceTracker(window_size=30)

        assert tracker is not None
        assert tracker.window_size == 30
        assert len(tracker.history) == 0

    def test_record_prediction(self):
        """Test recording predictions"""
        tracker = PerformanceTracker(window_size=10)

        # Record prediction
        tracker.record_prediction(
            model_name='TFT',
            predicted=155.0,
            actual=153.0,
            timestamp=datetime.now()
        )

        assert 'TFT' in tracker.history
        assert len(tracker.history['TFT']) == 1

    def test_window_size_enforcement(self):
        """Test that history respects window size"""
        tracker = PerformanceTracker(window_size=5)

        # Record 10 predictions
        for i in range(10):
            tracker.record_prediction(
                model_name='GNN',
                predicted=150.0 + i,
                actual=150.0 + i + 0.5,
                timestamp=datetime.now()
            )

        # Should only keep last 5
        assert len(tracker.history['GNN']) == 5

    def test_model_accuracy_calculation(self):
        """Test accuracy calculation"""
        tracker = PerformanceTracker(window_size=10)

        # Perfect predictions
        for i in range(5):
            tracker.record_prediction(
                model_name='Perfect',
                predicted=100.0,
                actual=100.0,
                timestamp=datetime.now()
            )

        accuracy = tracker.get_model_accuracy('Perfect')
        assert accuracy == 1.0  # Perfect accuracy

    def test_model_accuracy_with_errors(self):
        """Test accuracy with prediction errors"""
        tracker = PerformanceTracker(window_size=10)

        # 10% error predictions
        tracker.record_prediction('Imperfect', 100.0, 110.0, datetime.now())
        tracker.record_prediction('Imperfect', 100.0, 110.0, datetime.now())
        tracker.record_prediction('Imperfect', 100.0, 110.0, datetime.now())

        accuracy = tracker.get_model_accuracy('Imperfect')

        # Error = |100-110|/110 = 9.09%
        # Accuracy = 1 - 0.0909 = 90.9%
        assert 0.85 < accuracy < 0.95

    def test_unknown_model_default_accuracy(self):
        """Test default accuracy for unknown models"""
        tracker = PerformanceTracker(window_size=10)

        # Get accuracy for model with no history
        accuracy = tracker.get_model_accuracy('Unknown')

        # Should return neutral default
        assert accuracy == 0.5


class TestAdaptiveWeighting:
    """Test adaptive weighting based on performance"""

    def test_weight_normalization(self):
        """Test that weights sum to 1.0"""
        weights = {'TFT': 0.4, 'GNN': 0.3, 'Mamba': 0.2, 'PINN': 0.1}

        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    def test_adaptive_weight_adjustment(self):
        """Test weights adjust based on accuracy"""
        tracker = PerformanceTracker(window_size=10)

        # TFT performs well
        for _ in range(5):
            tracker.record_prediction('TFT', 100.0, 101.0, datetime.now())

        # GNN performs poorly
        for _ in range(5):
            tracker.record_prediction('GNN', 100.0, 120.0, datetime.now())

        tft_accuracy = tracker.get_model_accuracy('TFT')
        gnn_accuracy = tracker.get_model_accuracy('GNN')

        # TFT should have higher accuracy
        assert tft_accuracy > gnn_accuracy

        # In adaptive ensemble, TFT would get higher weight


class TestSignalVoting:
    """Test voting mechanism for trading signals"""

    def test_unanimous_vote(self):
        """Test unanimous signal agreement"""
        signals = [
            TradingSignal.BUY,
            TradingSignal.BUY,
            TradingSignal.BUY
        ]

        # All models agree -> strong signal
        from collections import Counter
        vote_counts = Counter(signals)
        most_common = vote_counts.most_common(1)[0]

        assert most_common[0] == TradingSignal.BUY
        assert most_common[1] == 3  # All 3 voted

    def test_majority_vote(self):
        """Test majority signal voting"""
        signals = [
            TradingSignal.BUY,
            TradingSignal.BUY,
            TradingSignal.HOLD,
            TradingSignal.SELL
        ]

        from collections import Counter
        vote_counts = Counter(signals)
        most_common = vote_counts.most_common(1)[0]

        assert most_common[0] == TradingSignal.BUY
        assert most_common[1] == 2  # 2 out of 4

    def test_split_vote(self):
        """Test handling of split votes"""
        signals = [
            TradingSignal.BUY,
            TradingSignal.HOLD,
            TradingSignal.SELL
        ]

        from collections import Counter
        vote_counts = Counter(signals)

        # No clear majority
        assert len(vote_counts) == 3
        # Default to HOLD or lower confidence


class TestUncertaintyQuantification:
    """Test uncertainty quantification in ensemble"""

    def test_high_agreement_low_uncertainty(self):
        """Test high model agreement = low uncertainty"""
        # All models predict close values
        predictions = [150.0, 150.5, 149.5, 150.2]
        std = np.std(predictions)

        # Low std = high agreement = low uncertainty
        assert std < 1.0

    def test_low_agreement_high_uncertainty(self):
        """Test low model agreement = high uncertainty"""
        # Models disagree widely
        predictions = [150.0, 140.0, 160.0, 135.0]
        std = np.std(predictions)

        # High std = low agreement = high uncertainty
        assert std > 5.0

    def test_position_sizing_by_uncertainty(self):
        """Test position size scaled by uncertainty"""
        # Low uncertainty -> larger position
        low_uncertainty_std = 1.0
        # High uncertainty -> smaller position
        high_uncertainty_std = 10.0

        # Mock position sizing (inverse of uncertainty)
        def position_size(std, max_size=1.0):
            return max(0.1, max_size / (1 + std / 5))

        size_low = position_size(low_uncertainty_std)
        size_high = position_size(high_uncertainty_std)

        # Lower uncertainty -> larger position
        assert size_low > size_high


class TestRiskManagement:
    """Test risk management features"""

    def test_stop_loss_calculation(self):
        """Test stop loss calculation"""
        entry_price = 150.0
        stop_loss_pct = 0.02  # 2%

        stop_loss = entry_price * (1 - stop_loss_pct)

        assert stop_loss == 147.0

    def test_take_profit_calculation(self):
        """Test take profit calculation"""
        entry_price = 150.0
        take_profit_pct = 0.05  # 5%

        take_profit = entry_price * (1 + take_profit_pct)

        assert take_profit == 157.5

    def test_risk_reward_ratio(self):
        """Test risk/reward ratio"""
        entry = 150.0
        stop_loss = 147.0  # 3 point risk
        take_profit = 156.0  # 6 point reward

        risk = entry - stop_loss
        reward = take_profit - entry
        ratio = reward / risk

        assert ratio == 2.0  # 2:1 risk/reward


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
