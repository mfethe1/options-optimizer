"""
ML-Based Alpha Scoring Engine - Renaissance Technologies Level

This module implements machine learning-driven alpha signals:
- Ensemble ML Alpha Score (combining multiple models)
- Market Regime Detection (HMM, clustering)
- Anomaly Detection (autoencoders, one-class SVM)
- Feature Importance Tracking (SHAP values)
- Model Confidence Monitoring (out-of-sample validation)

Inspired by:
- Renaissance Technologies' approach to non-intuitive signals
- ExtractAlpha's ML-driven strategies
- Academic research on ML in finance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    MEAN_REVERTING = "mean_reverting"
    MOMENTUM = "momentum"
    CRISIS = "crisis"
    RECOVERY = "recovery"


@dataclass
class AlphaScore:
    """ML-generated alpha score for a stock"""
    symbol: str
    score: float  # -100 to +100 (negative=sell, positive=buy)
    confidence: float  # 0-1
    predicted_return: float  # Expected excess return
    regime: MarketRegime
    feature_contributions: Dict[str, float]  # Which features drove the score
    timestamp: datetime


@dataclass
class RegimeIndicator:
    """Market regime detection result"""
    current_regime: MarketRegime
    regime_probability: float  # Confidence in regime classification
    regime_duration_days: int  # How long in current regime
    transition_probability: Dict[MarketRegime, float]  # Prob of transitioning to other regimes
    recommended_strategy: str  # Strategy for this regime


@dataclass
class AnomalyAlert:
    """Anomaly detection alert"""
    symbol: str
    anomaly_score: float  # 0-1 (higher = more anomalous)
    anomaly_type: str  # "price", "volume", "correlation", "pattern"
    description: str
    severity: str  # "low", "medium", "high", "critical"
    timestamp: datetime
    context: Dict[str, Any]


class MLAlphaEngine:
    """
    Machine Learning Alpha Generation Engine
    
    Combines multiple ML models to generate alpha scores:
    1. Gradient Boosting (XGBoost/LightGBM) for feature-based predictions
    2. Deep Neural Networks for complex pattern recognition
    3. LSTM/Transformer for time-series forecasting
    4. Ensemble voting for final alpha score
    
    Features used:
    - Technical indicators (momentum, volatility, volume)
    - Fundamental metrics (P/E, growth, quality)
    - Sentiment scores (news, social, analyst)
    - Alternative data (web traffic, options flow)
    - Cross-asset correlations
    - Macro factors
    """
    
    def __init__(
        self,
        lookback_days: int = 252,
        retrain_frequency_days: int = 30,
        min_confidence_threshold: float = 0.6
    ):
        """
        Initialize ML Alpha Engine
        
        Args:
            lookback_days: Historical data window
            retrain_frequency_days: How often to retrain models
            min_confidence_threshold: Minimum confidence to act on signals
        """
        self.lookback_days = lookback_days
        self.retrain_frequency_days = retrain_frequency_days
        self.min_confidence_threshold = min_confidence_threshold
        
        # Model components (placeholders - will be implemented with actual ML libraries)
        self.gradient_boosting_model = None
        self.neural_network_model = None
        self.lstm_model = None
        self.ensemble_weights = {'gb': 0.4, 'nn': 0.3, 'lstm': 0.3}
        
        # Regime detection
        self.regime_detector = None  # HMM or clustering model
        self.current_regime = MarketRegime.BULL_TRENDING
        
        # Anomaly detection
        self.anomaly_detector = None  # Autoencoder or One-Class SVM
        
        # Feature importance tracker
        self.feature_importance = {}
        self.feature_novelty_scores = {}
        
        # Model performance tracking
        self.out_of_sample_accuracy = 0.0
        self.rolling_sharpe = 0.0
        self.last_retrain_date = None
        
        logger.info("🤖 ML Alpha Engine initialized")
    
    def calculate_alpha_score(
        self,
        symbol: str,
        features: Dict[str, float],
        market_data: pd.DataFrame,
        regime: Optional[MarketRegime] = None
    ) -> AlphaScore:
        """
        Calculate ML-based alpha score for a stock
        
        Args:
            symbol: Stock symbol
            features: Feature dictionary (technical, fundamental, sentiment, etc.)
            market_data: Historical market data
            regime: Current market regime (auto-detected if None)
            
        Returns:
            AlphaScore with prediction and confidence
        """
        logger.info(f"🎯 Calculating alpha score for {symbol}")
        
        # Detect regime if not provided
        if regime is None:
            regime = self._detect_regime(market_data)
        
        # Extract feature vector
        feature_vector = self._prepare_features(features, market_data)
        
        # Get predictions from each model
        gb_pred = self._predict_gradient_boosting(feature_vector)
        nn_pred = self._predict_neural_network(feature_vector)
        lstm_pred = self._predict_lstm(market_data)
        
        # Ensemble prediction
        ensemble_pred = (
            self.ensemble_weights['gb'] * gb_pred +
            self.ensemble_weights['nn'] * nn_pred +
            self.ensemble_weights['lstm'] * lstm_pred
        )
        
        # Calculate confidence based on model agreement
        predictions = [gb_pred, nn_pred, lstm_pred]
        confidence = self._calculate_prediction_confidence(predictions)
        
        # Adjust for regime
        adjusted_score, adjusted_return = self._adjust_for_regime(
            ensemble_pred, regime
        )
        
        # Get feature contributions (SHAP-like)
        feature_contributions = self._calculate_feature_importance(
            feature_vector, ensemble_pred
        )
        
        alpha_score = AlphaScore(
            symbol=symbol,
            score=adjusted_score,
            confidence=confidence,
            predicted_return=adjusted_return,
            regime=regime,
            feature_contributions=feature_contributions,
            timestamp=datetime.utcnow()
        )
        
        logger.info(f"✅ Alpha score for {symbol}: {adjusted_score:.2f} (confidence: {confidence:.2%})")
        return alpha_score
    
    def detect_market_regime(
        self,
        market_data: pd.DataFrame,
        macro_indicators: Optional[Dict[str, float]] = None
    ) -> RegimeIndicator:
        """
        Detect current market regime using unsupervised learning
        
        Uses Hidden Markov Models or clustering to classify market state
        
        Args:
            market_data: Market returns and volatility data
            macro_indicators: Optional macro factors (VIX, rates, etc.)
            
        Returns:
            RegimeIndicator with current regime and transition probabilities
        """
        logger.info("🔍 Detecting market regime...")
        
        # Calculate regime features
        returns = market_data['returns'] if 'returns' in market_data else market_data.pct_change()
        volatility = returns.rolling(20).std()
        trend = returns.rolling(50).mean()
        
        # Placeholder: In production, use HMM or clustering
        # For now, use simple heuristics
        current_vol = volatility.iloc[-1]
        current_trend = trend.iloc[-1]
        
        if current_vol > volatility.quantile(0.75):
            if current_trend < 0:
                regime = MarketRegime.CRISIS
            else:
                regime = MarketRegime.HIGH_VOLATILITY
        elif current_vol < volatility.quantile(0.25):
            if current_trend > 0:
                regime = MarketRegime.BULL_TRENDING
            else:
                regime = MarketRegime.LOW_VOLATILITY
        elif abs(current_trend) < 0.001:
            regime = MarketRegime.MEAN_REVERTING
        else:
            regime = MarketRegime.MOMENTUM if current_trend > 0 else MarketRegime.BEAR_TRENDING
        
        # Calculate transition probabilities (placeholder)
        transition_probs = {
            MarketRegime.BULL_TRENDING: 0.3,
            MarketRegime.BEAR_TRENDING: 0.1,
            MarketRegime.HIGH_VOLATILITY: 0.2,
            MarketRegime.LOW_VOLATILITY: 0.15,
            MarketRegime.MEAN_REVERTING: 0.15,
            MarketRegime.MOMENTUM: 0.1
        }
        
        # Recommend strategy based on regime
        strategy_map = {
            MarketRegime.BULL_TRENDING: "momentum_long",
            MarketRegime.BEAR_TRENDING: "defensive_short",
            MarketRegime.HIGH_VOLATILITY: "volatility_arbitrage",
            MarketRegime.LOW_VOLATILITY: "carry_trade",
            MarketRegime.MEAN_REVERTING: "pairs_trading",
            MarketRegime.MOMENTUM: "trend_following",
            MarketRegime.CRISIS: "risk_off",
            MarketRegime.RECOVERY: "value_rotation"
        }
        
        indicator = RegimeIndicator(
            current_regime=regime,
            regime_probability=0.75,  # Placeholder
            regime_duration_days=30,  # Placeholder
            transition_probability=transition_probs,
            recommended_strategy=strategy_map.get(regime, "balanced")
        )
        
        self.current_regime = regime
        logger.info(f"📊 Current regime: {regime.value} (strategy: {indicator.recommended_strategy})")
        
        return indicator
    
    # ==================== Private Helper Methods ====================
    
    def _detect_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """Quick regime detection"""
        indicator = self.detect_market_regime(market_data)
        return indicator.current_regime
    
    def _prepare_features(self, features: Dict[str, float], market_data: pd.DataFrame) -> np.ndarray:
        """Prepare feature vector for ML models"""
        # Placeholder: Convert dict to numpy array
        feature_list = list(features.values())
        return np.array(feature_list)
    
    def _predict_gradient_boosting(self, features: np.ndarray) -> float:
        """Gradient boosting prediction (placeholder)"""
        # In production: return self.gradient_boosting_model.predict(features)
        return np.random.uniform(-50, 50)  # Placeholder
    
    def _predict_neural_network(self, features: np.ndarray) -> float:
        """Neural network prediction (placeholder)"""
        # In production: return self.neural_network_model.predict(features)
        return np.random.uniform(-50, 50)  # Placeholder
    
    def _predict_lstm(self, market_data: pd.DataFrame) -> float:
        """LSTM time-series prediction (placeholder)"""
        # In production: return self.lstm_model.predict(market_data)
        return np.random.uniform(-50, 50)  # Placeholder
    
    def _calculate_prediction_confidence(self, predictions: List[float]) -> float:
        """Calculate confidence based on model agreement"""
        # High agreement = high confidence
        std = np.std(predictions)
        mean = np.mean(predictions)
        
        if mean == 0:
            return 0.5
        
        # Coefficient of variation (lower = more agreement)
        cv = abs(std / mean)
        confidence = max(0.0, min(1.0, 1.0 - cv))
        
        return confidence
    
    def _adjust_for_regime(self, score: float, regime: MarketRegime) -> Tuple[float, float]:
        """Adjust alpha score based on market regime"""
        # Different regimes favor different strategies
        regime_multipliers = {
            MarketRegime.BULL_TRENDING: 1.2,
            MarketRegime.BEAR_TRENDING: 0.8,
            MarketRegime.HIGH_VOLATILITY: 0.7,
            MarketRegime.LOW_VOLATILITY: 1.1,
            MarketRegime.MEAN_REVERTING: 1.0,
            MarketRegime.MOMENTUM: 1.3,
            MarketRegime.CRISIS: 0.5,
            MarketRegime.RECOVERY: 1.4
        }
        
        multiplier = regime_multipliers.get(regime, 1.0)
        adjusted_score = score * multiplier
        predicted_return = adjusted_score / 100.0  # Convert to decimal return
        
        return adjusted_score, predicted_return
    
    def _calculate_feature_importance(
        self,
        features: np.ndarray,
        prediction: float
    ) -> Dict[str, float]:
        """Calculate feature contributions (SHAP-like)"""
        # Placeholder: In production, use SHAP values
        feature_names = ['momentum', 'value', 'quality', 'sentiment', 'volatility']
        contributions = {
            name: np.random.uniform(-10, 10)
            for name in feature_names
        }
        return contributions

    # ==================== Anomaly Detection ====================

    def detect_anomalies(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> List[AnomalyAlert]:
        """
        Detect anomalous behavior using advanced ML techniques

        Uses:
        - Autoencoders for pattern anomalies
        - One-Class SVM for outlier detection
        - Graph Neural Networks for correlation anomalies (STAGE framework)

        Args:
            symbol: Stock symbol
            price_data: Historical price data
            volume_data: Historical volume data
            correlation_matrix: Optional correlation with other stocks

        Returns:
            List of anomaly alerts
        """
        logger.info(f"🔍 Detecting anomalies for {symbol}")

        alerts = []

        # Price anomaly detection
        price_anomaly = self._detect_price_anomaly(symbol, price_data)
        if price_anomaly:
            alerts.append(price_anomaly)

        # Volume anomaly detection
        volume_anomaly = self._detect_volume_anomaly(symbol, volume_data)
        if volume_anomaly:
            alerts.append(volume_anomaly)

        # Correlation anomaly detection
        if correlation_matrix is not None:
            corr_anomaly = self._detect_correlation_anomaly(symbol, correlation_matrix)
            if corr_anomaly:
                alerts.append(corr_anomaly)

        # Pattern anomaly detection (using autoencoder)
        pattern_anomaly = self._detect_pattern_anomaly(symbol, price_data, volume_data)
        if pattern_anomaly:
            alerts.append(pattern_anomaly)

        logger.info(f"🚨 Found {len(alerts)} anomalies for {symbol}")
        return alerts

    def _detect_price_anomaly(self, symbol: str, price_data: pd.DataFrame) -> Optional[AnomalyAlert]:
        """Detect unusual price movements"""
        returns = price_data.pct_change()
        recent_return = returns.iloc[-1]
        historical_std = returns.std()

        # Z-score anomaly detection
        z_score = abs(recent_return / historical_std) if historical_std > 0 else 0

        if z_score > 3.0:  # 3-sigma event
            severity = "critical" if z_score > 5.0 else "high"
            return AnomalyAlert(
                symbol=symbol,
                anomaly_score=min(1.0, z_score / 5.0),
                anomaly_type="price",
                description=f"Unusual price movement: {recent_return:.2%} ({z_score:.1f} sigma event)",
                severity=severity,
                timestamp=datetime.utcnow(),
                context={'z_score': z_score, 'return': recent_return}
            )

        return None

    def _detect_volume_anomaly(self, symbol: str, volume_data: pd.DataFrame) -> Optional[AnomalyAlert]:
        """Detect unusual volume spikes"""
        recent_volume = volume_data.iloc[-1]
        avg_volume = volume_data.rolling(20).mean().iloc[-1]

        if avg_volume == 0:
            return None

        volume_ratio = recent_volume / avg_volume

        if volume_ratio > 3.0:  # 3x average volume
            severity = "high" if volume_ratio > 5.0 else "medium"
            return AnomalyAlert(
                symbol=symbol,
                anomaly_score=min(1.0, volume_ratio / 5.0),
                anomaly_type="volume",
                description=f"Unusual volume spike: {volume_ratio:.1f}x average",
                severity=severity,
                timestamp=datetime.utcnow(),
                context={'volume_ratio': volume_ratio, 'recent_volume': recent_volume}
            )

        return None

    def _detect_correlation_anomaly(
        self,
        symbol: str,
        correlation_matrix: pd.DataFrame
    ) -> Optional[AnomalyAlert]:
        """Detect unusual correlation changes"""
        # Placeholder: In production, use Graph Neural Networks (STAGE framework)
        # to detect anomalous correlation patterns

        if symbol not in correlation_matrix.columns:
            return None

        # Check if correlation with market suddenly changed
        correlations = correlation_matrix[symbol]
        avg_corr = correlations.mean()

        # If stock suddenly decorrelates from market (or becomes highly correlated)
        if abs(avg_corr) < 0.1 or abs(avg_corr) > 0.95:
            severity = "medium"
            return AnomalyAlert(
                symbol=symbol,
                anomaly_score=0.7,
                anomaly_type="correlation",
                description=f"Unusual correlation pattern: avg correlation = {avg_corr:.2f}",
                severity=severity,
                timestamp=datetime.utcnow(),
                context={'avg_correlation': avg_corr}
            )

        return None

    def _detect_pattern_anomaly(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame
    ) -> Optional[AnomalyAlert]:
        """Detect anomalous patterns using autoencoder"""
        # Placeholder: In production, use trained autoencoder
        # High reconstruction error = anomalous pattern

        # For now, use simple heuristic: simultaneous price drop + volume spike
        price_change = price_data.pct_change().iloc[-1]
        volume_ratio = volume_data.iloc[-1] / volume_data.rolling(20).mean().iloc[-1]

        if price_change < -0.05 and volume_ratio > 2.0:
            return AnomalyAlert(
                symbol=symbol,
                anomaly_score=0.8,
                anomaly_type="pattern",
                description="Unusual pattern: Large price drop with volume spike (potential capitulation or news event)",
                severity="high",
                timestamp=datetime.utcnow(),
                context={'price_change': price_change, 'volume_ratio': volume_ratio}
            )

        return None

    # ==================== Model Monitoring ====================

    def get_model_health(self) -> Dict[str, Any]:
        """
        Get model health metrics

        Returns:
            Dictionary with model performance and confidence metrics
        """
        return {
            'out_of_sample_accuracy': self.out_of_sample_accuracy,
            'rolling_sharpe': self.rolling_sharpe,
            'last_retrain_date': self.last_retrain_date.isoformat() if self.last_retrain_date else None,
            'days_since_retrain': (datetime.utcnow() - self.last_retrain_date).days if self.last_retrain_date else None,
            'needs_retraining': self._needs_retraining(),
            'feature_importance': self.feature_importance,
            'current_regime': self.current_regime.value if self.current_regime else None
        }

    def _needs_retraining(self) -> bool:
        """Check if models need retraining"""
        if self.last_retrain_date is None:
            return True

        days_since = (datetime.utcnow() - self.last_retrain_date).days

        # Retrain if:
        # 1. Past retrain frequency
        # 2. Performance degraded significantly
        # 3. Regime changed

        return (
            days_since >= self.retrain_frequency_days or
            self.out_of_sample_accuracy < 0.55 or
            self.rolling_sharpe < 0.5
        )

    def update_performance_metrics(
        self,
        predictions: List[float],
        actuals: List[float],
        returns: List[float]
    ) -> None:
        """
        Update model performance metrics

        Args:
            predictions: Model predictions
            actuals: Actual outcomes
            returns: Strategy returns
        """
        # Calculate accuracy
        correct = sum(1 for p, a in zip(predictions, actuals) if (p > 0) == (a > 0))
        self.out_of_sample_accuracy = correct / len(predictions) if predictions else 0.0

        # Calculate Sharpe ratio
        if returns:
            returns_array = np.array(returns)
            self.rolling_sharpe = (
                returns_array.mean() / returns_array.std()
                if returns_array.std() > 0 else 0.0
            )

        logger.info(f"📊 Model performance updated: Accuracy={self.out_of_sample_accuracy:.2%}, Sharpe={self.rolling_sharpe:.2f}")


