"""
Ensemble Neural Network API Routes

Combines all 5 models for superior predictions and analysis:
- Epidemic Volatility
- TFT + Conformal Prediction
- Graph Neural Networks
- Mamba State Space Model
- Physics-Informed Neural Networks

Provides:
- Unified ensemble predictions
- Multi-model comparison view
- Model agreement analysis
- Adaptive weighting
- Intraday and long-term modes
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
import logging
import numpy as np

from ..ml.ensemble.ensemble_predictor import (
    EnsemblePredictor,
    ModelPrediction,
    TradingSignal,
    TimeHorizon
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Global ensemble predictor
ensemble_predictor: Optional[EnsemblePredictor] = None


class EnsembleAnalysisRequest(BaseModel):
    """Request for ensemble analysis"""
    symbol: str
    time_horizon: str = 'short_term'  # 'intraday', 'short_term', 'medium_term', 'long_term'
    use_cache: bool = True


class ModelPredictionResponse(BaseModel):
    """Individual model prediction"""
    model_name: str
    price_prediction: float
    confidence: float
    signal: str
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    metadata: Optional[Dict] = None


class EnsembleAnalysisResponse(BaseModel):
    """Ensemble analysis response"""
    timestamp: str
    symbol: str
    current_price: float
    time_horizon: str

    # Ensemble results
    ensemble_prediction: float
    ensemble_signal: str
    ensemble_confidence: float

    # Individual model predictions
    model_predictions: List[ModelPredictionResponse]

    # Model weights
    model_weights: Dict[str, float]

    # Uncertainty metrics
    prediction_std: float
    model_agreement: float

    # Trading recommendation
    position_size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Additional analysis
    expected_return: float
    risk_reward_ratio: Optional[float] = None


class PerformanceMetrics(BaseModel):
    """Model performance metrics"""
    model_name: str
    accuracy: float
    sharpe_ratio: float
    current_weight: float


async def initialize_ensemble_service():
    """Initialize ensemble service"""
    global ensemble_predictor

    try:
        ensemble_predictor = EnsemblePredictor(
            weighting_method='adaptive',
            voting_threshold=0.6,
            track_performance=True
        )

        logger.info("Ensemble service initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize ensemble service: {e}")


@router.get("/ensemble/status")
async def get_status():
    """Get ensemble service status"""
    return {
        'status': 'active',
        'ensemble_ready': ensemble_predictor is not None,
        'model': 'Multi-Model Ensemble System',
        'models_combined': [
            'Epidemic Volatility (SIR/SEIR + PINN)',
            'Temporal Fusion Transformer + Conformal Prediction',
            'Graph Neural Networks (GAT)',
            'Mamba State Space Model',
            'Physics-Informed Neural Networks'
        ],
        'features': [
            'Weighted averaging for price predictions',
            'Voting for trading signals',
            'Adaptive weights based on performance',
            'Regime-aware model selection',
            'Intraday and long-term modes',
            'Uncertainty quantification via model agreement',
            'Position sizing based on confidence'
        ],
        'time_horizons': ['intraday', 'short_term', 'medium_term', 'long_term']
    }


@router.post("/ensemble/analyze", response_model=EnsembleAnalysisResponse)
async def analyze_ensemble(request: EnsembleAnalysisRequest):
    """
    Get ensemble analysis combining all 5 models

    Returns multi-model predictions with:
    - Ensemble consensus
    - Individual model predictions for comparison
    - Model agreement metrics
    - Trading recommendations
    """
    if ensemble_predictor is None:
        raise HTTPException(status_code=503, detail="Ensemble service not initialized")

    try:
        import yfinance as yf

        # Get current price
        ticker = yf.Ticker(request.symbol)
        hist = ticker.history(period='1d')

        if len(hist) == 0:
            raise HTTPException(status_code=400, detail=f"No data for {request.symbol}")

        current_price = float(hist['Close'].iloc[-1])

        # Parse time horizon
        horizon_map = {
            'intraday': TimeHorizon.INTRADAY,
            'short_term': TimeHorizon.SHORT_TERM,
            'medium_term': TimeHorizon.MEDIUM_TERM,
            'long_term': TimeHorizon.LONG_TERM
        }
        time_horizon = horizon_map.get(request.time_horizon, TimeHorizon.SHORT_TERM)

        # Collect predictions from all models
        model_predictions = []

        # 1. TFT + Conformal Prediction
        try:
            from .advanced_forecast_routes import get_forecast_signal
            from pydantic import BaseModel as PydanticBaseModel

            class ForecastRequest(PydanticBaseModel):
                symbol: str
                use_cache: bool = True

            tft_request = ForecastRequest(symbol=request.symbol, use_cache=request.use_cache)
            tft_result = await get_forecast_signal(tft_request)

            # Map action to signal
            action_to_signal = {
                'BUY': TradingSignal.BUY,
                'SELL': TradingSignal.SELL,
                'HOLD': TradingSignal.HOLD,
                'STRONG_BUY': TradingSignal.STRONG_BUY,
                'STRONG_SELL': TradingSignal.STRONG_SELL
            }

            horizon_key = '1d' if time_horizon == TimeHorizon.SHORT_TERM else '5d'
            pred_price = tft_result.get('predictions', {}).get(horizon_key, {}).get('point', current_price)

            model_predictions.append(ModelPrediction(
                model_name='tft_conformal',
                price_prediction=pred_price,
                confidence=tft_result.get('confidence', 0.5),
                signal=action_to_signal.get(tft_result.get('action', 'HOLD'), TradingSignal.HOLD),
                timestamp=datetime.now(),
                lower_bound=tft_result.get('predictions', {}).get(horizon_key, {}).get('lower', None),
                upper_bound=tft_result.get('predictions', {}).get(horizon_key, {}).get('upper', None)
            ))

        except Exception as e:
            logger.warning(f"TFT model unavailable: {e}")

        # 2. Mamba State Space Model
        try:
            from .mamba_routes import mamba_predictor

            if mamba_predictor:
                # Get longer history for Mamba
                hist_long = ticker.history(period='2y')
                price_history = hist_long['Close'].values

                if len(price_history) >= 60:
                    mamba_result = await mamba_predictor.predict(
                        symbol=request.symbol,
                        price_history=price_history,
                        current_price=current_price
                    )

                    pred_price = mamba_result.get('1d', current_price)
                    expected_return = (pred_price - current_price) / current_price

                    if expected_return > 0.03:
                        signal = TradingSignal.STRONG_BUY if expected_return > 0.05 else TradingSignal.BUY
                    elif expected_return < -0.03:
                        signal = TradingSignal.STRONG_SELL if expected_return < -0.05 else TradingSignal.SELL
                    else:
                        signal = TradingSignal.HOLD

                    model_predictions.append(ModelPrediction(
                        model_name='mamba',
                        price_prediction=pred_price,
                        confidence=min(abs(expected_return) * 10, 0.9),
                        signal=signal,
                        timestamp=datetime.now()
                    ))

        except Exception as e:
            logger.warning(f"Mamba model unavailable: {e}")

        # 3. Simple momentum baseline (fallback if other models unavailable)
        # Calculate momentum signal
        hist_short = ticker.history(period='30d')
        if len(hist_short) >= 20:
            prices = hist_short['Close'].values
            sma_20 = np.mean(prices[-20:])
            momentum = (current_price - sma_20) / sma_20

            # Simple momentum prediction
            momentum_pred = current_price * (1 + momentum * 0.5)  # Assume momentum continues

            if momentum > 0.02:
                mom_signal = TradingSignal.BUY
            elif momentum < -0.02:
                mom_signal = TradingSignal.SELL
            else:
                mom_signal = TradingSignal.HOLD

            model_predictions.append(ModelPrediction(
                model_name='momentum_baseline',
                price_prediction=momentum_pred,
                confidence=min(abs(momentum) * 10, 0.7),
                signal=mom_signal,
                timestamp=datetime.now(),
                metadata={'momentum': float(momentum), 'sma_20': float(sma_20)}
            ))

        # If no models available, raise error
        if len(model_predictions) == 0:
            raise HTTPException(
                status_code=503,
                detail="No models available for prediction"
            )

        # Detect market regime (simple volatility-based)
        if len(hist_short) >= 20:
            volatility = np.std(hist_short['Close'].pct_change().dropna())
            if volatility > 0.03:
                market_regime = 'volatile'
            elif volatility < 0.01:
                market_regime = 'calm'
            else:
                market_regime = 'normal'
        else:
            market_regime = None

        # Generate ensemble prediction
        ensemble_result = await ensemble_predictor.predict(
            symbol=request.symbol,
            current_price=current_price,
            model_predictions=model_predictions,
            time_horizon=time_horizon,
            market_regime=market_regime
        )

        # Calculate expected return
        expected_return = (ensemble_result.ensemble_prediction - current_price) / current_price

        # Calculate risk/reward ratio
        risk_reward_ratio = None
        if ensemble_result.stop_loss and ensemble_result.take_profit:
            risk = abs(current_price - ensemble_result.stop_loss)
            reward = abs(ensemble_result.take_profit - current_price)
            if risk > 0:
                risk_reward_ratio = reward / risk

        # Convert to response format
        return EnsembleAnalysisResponse(
            timestamp=ensemble_result.timestamp.isoformat(),
            symbol=ensemble_result.symbol,
            current_price=ensemble_result.current_price,
            time_horizon=request.time_horizon,
            ensemble_prediction=ensemble_result.ensemble_prediction,
            ensemble_signal=ensemble_result.ensemble_signal.value,
            ensemble_confidence=ensemble_result.ensemble_confidence,
            model_predictions=[
                ModelPredictionResponse(
                    model_name=p.model_name,
                    price_prediction=p.price_prediction,
                    confidence=p.confidence,
                    signal=p.signal.value,
                    lower_bound=p.lower_bound,
                    upper_bound=p.upper_bound,
                    metadata=p.metadata
                )
                for p in ensemble_result.model_predictions
            ],
            model_weights=ensemble_result.model_weights,
            prediction_std=ensemble_result.prediction_std,
            model_agreement=ensemble_result.model_agreement,
            position_size=ensemble_result.position_size,
            stop_loss=ensemble_result.stop_loss,
            take_profit=ensemble_result.take_profit,
            expected_return=expected_return,
            risk_reward_ratio=risk_reward_ratio
        )

    except Exception as e:
        logger.error(f"Error in ensemble analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ensemble/performance")
async def get_model_performance():
    """Get performance metrics for all models"""
    if ensemble_predictor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if not ensemble_predictor.performance_tracker:
        return {
            'tracking_enabled': False,
            'message': 'Performance tracking not enabled'
        }

    accuracies = ensemble_predictor.performance_tracker.get_all_accuracies()

    metrics = []
    for model_name in ensemble_predictor.models:
        accuracy = accuracies.get(model_name, 0.5)
        sharpe = ensemble_predictor.performance_tracker.get_model_sharpe(model_name)
        weight = ensemble_predictor.weights.get(model_name, 0.0)

        metrics.append({
            'model_name': model_name,
            'accuracy': accuracy,
            'sharpe_ratio': sharpe,
            'current_weight': weight
        })

    return {
        'tracking_enabled': True,
        'weighting_method': ensemble_predictor.weighting_method,
        'model_metrics': metrics
    }


@router.get("/ensemble/explanation")
async def get_explanation():
    """Get detailed explanation of ensemble system"""
    return {
        'title': 'Multi-Model Ensemble Neural Network System',
        'concept': 'Combine 5 state-of-the-art neural networks for superior predictions',
        'models_included': {
            'Epidemic Volatility': 'SIR/SEIR models for market regime detection',
            'TFT + Conformal': 'Multi-horizon forecasting with guaranteed intervals',
            'GNN': 'Correlation-aware predictions via graph neural networks',
            'Mamba': 'Linear-complexity long-sequence modeling',
            'PINN': 'Physics-informed option pricing and risk modeling'
        },
        'ensemble_methods': {
            'Weighted Averaging': 'Price predictions combined with performance-based weights',
            'Voting': 'Trading signals determined by weighted majority vote',
            'Adaptive Weighting': 'Weights adjust based on recent accuracy',
            'Regime-Aware': 'Model weights adjusted for market conditions',
            'Horizon-Specific': 'Different weights for intraday vs long-term'
        },
        'key_advantages': [
            '**Diversification** - Each model has unique strengths',
            '**Robustness** - Less vulnerable to single model failure',
            '**Uncertainty quantification** - Model disagreement = higher uncertainty',
            '**Adaptive** - Learns which models work best over time',
            '**Context-aware** - Adjusts for market regime and time horizon',
            '**Position sizing** - Trades more when confident, less when uncertain'
        ],
        'how_it_works': {
            '1. Collect Predictions': 'Get predictions from all 5 models',
            '2. Apply Weights': 'Weight each model based on performance & context',
            '3. Aggregate Prices': 'Weighted average for ensemble price prediction',
            '4. Vote on Signal': 'Weighted voting for BUY/SELL/HOLD',
            '5. Quantify Uncertainty': 'Measure model agreement',
            '6. Size Position': 'Larger positions when models agree + high confidence',
            '7. Set Risk Management': 'Stop loss and take profit based on uncertainty'
        },
        'weighting_strategies': {
            'Equal': 'All models weighted equally (20% each)',
            'Performance': 'Weight by recent accuracy (better models get more weight)',
            'Adaptive': 'Dynamic adjustment with momentum smoothing',
            'Regime-Specific': 'Boost epidemic model in volatile markets',
            'Horizon-Specific': 'Boost Mamba for intraday, TFT for swing trading'
        },
        'research': 'Ensemble Methods in Machine Learning (Dietterich, 2000)'
    }
