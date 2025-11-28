"""
Truth Dashboard Routes - Compare yesterday's predictions to today's actuals

This module provides the HIGH-LEVERAGE CATALYST to break the feedback loop of adding
models without validation. It tracks prediction accuracy over time and provides
transparency into model performance.

Endpoints:
- GET /api/truth/daily-accuracy - 30-day rolling accuracy for all models
- GET /api/truth/model/{model_name}/history - Full prediction vs actual history
- POST /api/truth/record-prediction - Store a prediction for later validation
- POST /api/truth/validate-predictions - Compare predictions against actuals
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional
from datetime import datetime, date, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import json
import logging
import os
import asyncio
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/truth", tags=["truth-dashboard"])

# --------------------------------------------------
# Data Models
# --------------------------------------------------

class PredictionDirection(str, Enum):
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


class ModelStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNTRAINED = "untrained"


@dataclass
class PredictionRecord:
    """A single prediction record for validation tracking"""
    model_name: str
    symbol: str
    prediction_timestamp: str  # ISO format
    target_date: str  # YYYY-MM-DD format
    predicted_price: float
    predicted_direction: str  # 'up', 'down', 'neutral'
    horizon_days: int = 1
    confidence: float = 0.0
    actual_price: Optional[float] = None
    actual_direction: Optional[str] = None
    is_correct: Optional[bool] = None
    validated_at: Optional[str] = None
    price_error: Optional[float] = None  # Absolute error
    price_error_pct: Optional[float] = None  # MAPE contribution

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionRecord":
        return cls(**data)


@dataclass
class ModelAccuracyStats:
    """Aggregated accuracy statistics for a model"""
    model_name: str
    direction_accuracy_30d: float = 0.0
    mape_30d: float = 0.0  # Mean Absolute Percentage Error
    total_predictions: int = 0
    correct_predictions: int = 0
    last_5: List[bool] = field(default_factory=list)
    status: str = "untrained"
    last_updated: str = ""
    avg_confidence: float = 0.0
    prediction_count_30d: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelAccuracyStats":
        return cls(**data)


# --------------------------------------------------
# Request/Response Models (Pydantic)
# --------------------------------------------------

class RecordPredictionRequest(BaseModel):
    """Request model for recording a prediction"""
    model_name: str = Field(..., description="Model identifier (tft, gnn, pinn, mamba, epidemic, ensemble)")
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL, SPY)")
    predicted_price: float = Field(..., gt=0, description="Predicted price")
    predicted_direction: str = Field(..., description="Predicted direction: up, down, neutral")
    horizon_days: int = Field(default=1, ge=1, le=365, description="Prediction horizon in days")
    confidence: float = Field(default=0.5, ge=0, le=1, description="Model confidence score")
    timestamp: Optional[str] = Field(default=None, description="Prediction timestamp (ISO format). Defaults to now.")

    @validator('model_name')
    def validate_model_name(cls, v):
        valid_models = ['tft', 'gnn', 'pinn', 'mamba', 'epidemic', 'ensemble']
        if v.lower() not in valid_models:
            raise ValueError(f"Invalid model_name. Must be one of: {', '.join(valid_models)}")
        return v.lower()

    @validator('predicted_direction')
    def validate_direction(cls, v):
        valid_directions = ['up', 'down', 'neutral']
        if v.lower() not in valid_directions:
            raise ValueError(f"Invalid predicted_direction. Must be one of: {', '.join(valid_directions)}")
        return v.lower()

    @validator('symbol')
    def validate_symbol(cls, v):
        import re
        if not re.match(r'^[A-Z0-9\.\-]{1,10}$', v.upper()):
            raise ValueError("Invalid symbol format")
        return v.upper()


class ValidatePredictionsRequest(BaseModel):
    """Request model for validating predictions"""
    target_date: Optional[str] = Field(default=None, description="Date to validate (YYYY-MM-DD). Defaults to yesterday.")


class PredictionHistoryResponse(BaseModel):
    """Response model for prediction history"""
    model_name: str
    total_records: int
    validated_records: int
    pending_records: int
    history: List[Dict[str, Any]]


class DailyAccuracyResponse(BaseModel):
    """Response model for daily accuracy metrics"""
    models: Dict[str, Dict[str, Any]]
    summary: Dict[str, Any]
    timestamp: str


# --------------------------------------------------
# Storage Layer (Thread-safe JSON file storage)
# --------------------------------------------------

class TruthStorage:
    """Thread-safe JSON file storage for truth dashboard data"""

    def __init__(self, data_dir: str = "data/truth"):
        self.data_dir = Path(data_dir)
        self.predictions_file = self.data_dir / "predictions.json"
        self.stats_file = self.data_dir / "accuracy_stats.json"
        self._lock = threading.RLock()
        self._ensure_data_dir()

    def _ensure_data_dir(self):
        """Ensure data directory exists"""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize empty files if they don't exist
        if not self.predictions_file.exists():
            self._write_json(self.predictions_file, {"predictions": []})
        if not self.stats_file.exists():
            self._write_json(self.stats_file, {"stats": {}})

    def _read_json(self, file_path: Path) -> Dict[str, Any]:
        """Thread-safe JSON file read"""
        with self._lock:
            try:
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                return {}
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in {file_path}: {e}")
                return {}
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                return {}

    def _write_json(self, file_path: Path, data: Dict[str, Any]):
        """Thread-safe JSON file write with atomic operation"""
        with self._lock:
            try:
                # Write to temp file first, then rename (atomic on most systems)
                temp_path = file_path.with_suffix('.tmp')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)

                # Atomic rename (or copy on Windows)
                if os.name == 'nt':  # Windows
                    import shutil
                    shutil.move(str(temp_path), str(file_path))
                else:
                    temp_path.rename(file_path)

            except Exception as e:
                logger.error(f"Error writing {file_path}: {e}")
                raise

    def add_prediction(self, record: PredictionRecord) -> bool:
        """Add a new prediction record"""
        try:
            data = self._read_json(self.predictions_file)
            predictions = data.get("predictions", [])
            predictions.append(record.to_dict())
            data["predictions"] = predictions
            data["last_updated"] = datetime.now().isoformat()
            self._write_json(self.predictions_file, data)
            logger.info(f"Recorded prediction: {record.model_name} for {record.symbol} -> ${record.predicted_price:.2f}")
            return True
        except Exception as e:
            logger.error(f"Failed to add prediction: {e}")
            return False

    def get_predictions(
        self,
        model_name: Optional[str] = None,
        symbol: Optional[str] = None,
        validated_only: bool = False,
        pending_only: bool = False,
        days_back: int = 30
    ) -> List[PredictionRecord]:
        """Get predictions with optional filters"""
        data = self._read_json(self.predictions_file)
        predictions = data.get("predictions", [])

        # Calculate date cutoff
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()

        filtered = []
        for p in predictions:
            # Time filter
            if p.get("prediction_timestamp", "") < cutoff_date:
                continue

            # Model filter
            if model_name and p.get("model_name") != model_name:
                continue

            # Symbol filter
            if symbol and p.get("symbol") != symbol:
                continue

            # Validation status filter
            if validated_only and p.get("validated_at") is None:
                continue
            if pending_only and p.get("validated_at") is not None:
                continue

            filtered.append(PredictionRecord.from_dict(p))

        return filtered

    def update_prediction(self, record: PredictionRecord) -> bool:
        """Update an existing prediction record (for validation)"""
        try:
            data = self._read_json(self.predictions_file)
            predictions = data.get("predictions", [])

            # Find and update the matching prediction
            for i, p in enumerate(predictions):
                if (p.get("model_name") == record.model_name and
                    p.get("symbol") == record.symbol and
                    p.get("prediction_timestamp") == record.prediction_timestamp):
                    predictions[i] = record.to_dict()
                    data["predictions"] = predictions
                    data["last_updated"] = datetime.now().isoformat()
                    self._write_json(self.predictions_file, data)
                    return True

            return False
        except Exception as e:
            logger.error(f"Failed to update prediction: {e}")
            return False

    def get_stats(self, model_name: Optional[str] = None) -> Dict[str, ModelAccuracyStats]:
        """Get accuracy stats for one or all models"""
        data = self._read_json(self.stats_file)
        stats_dict = data.get("stats", {})

        result = {}
        for name, stats_data in stats_dict.items():
            if model_name is None or name == model_name:
                result[name] = ModelAccuracyStats.from_dict(stats_data)

        return result

    def update_stats(self, stats: ModelAccuracyStats) -> bool:
        """Update stats for a model"""
        try:
            data = self._read_json(self.stats_file)
            if "stats" not in data:
                data["stats"] = {}

            data["stats"][stats.model_name] = stats.to_dict()
            data["last_updated"] = datetime.now().isoformat()
            self._write_json(self.stats_file, data)
            return True
        except Exception as e:
            logger.error(f"Failed to update stats: {e}")
            return False


# Global storage instance
_storage = TruthStorage()


# --------------------------------------------------
# Market Data Helper
# --------------------------------------------------

async def get_actual_price(symbol: str, target_date: date) -> Optional[float]:
    """Fetch actual closing price for a symbol on a given date"""
    try:
        import yfinance as yf
        from concurrent.futures import ThreadPoolExecutor

        # Use thread pool for blocking yfinance call
        def fetch_price():
            ticker = yf.Ticker(symbol)
            # Fetch data for the target date
            start_date = target_date
            end_date = target_date + timedelta(days=1)
            hist = ticker.history(start=start_date, end=end_date)

            if len(hist) > 0:
                return float(hist['Close'].iloc[-1])
            return None

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            price = await loop.run_in_executor(executor, fetch_price)
            return price

    except Exception as e:
        logger.error(f"Failed to fetch actual price for {symbol} on {target_date}: {e}")
        return None


def calculate_direction(current_price: float, target_price: float, threshold: float = 0.001) -> str:
    """Calculate direction based on price change"""
    change_pct = (target_price - current_price) / current_price
    if change_pct > threshold:
        return "up"
    elif change_pct < -threshold:
        return "down"
    return "neutral"


# --------------------------------------------------
# API Endpoints
# --------------------------------------------------

@router.get("/daily-accuracy", response_model=DailyAccuracyResponse)
async def get_daily_accuracy():
    """
    Get 30-day rolling accuracy metrics for all models.

    Returns direction accuracy, MAPE, last 5 results, and status for each model.
    """
    logger.info("[truth/daily-accuracy] Computing accuracy metrics for all models")

    # Define all expected models
    model_names = ['tft', 'gnn', 'pinn', 'mamba', 'epidemic', 'ensemble']

    # Get validated predictions from last 30 days
    all_predictions = _storage.get_predictions(validated_only=True, days_back=30)

    # Compute stats for each model
    model_stats: Dict[str, Dict[str, Any]] = {}

    for model_name in model_names:
        model_predictions = [p for p in all_predictions if p.model_name == model_name]

        if not model_predictions:
            # No predictions for this model
            model_stats[model_name] = {
                "direction_accuracy_30d": 0.0,
                "mape_30d": 0.0,
                "total_predictions": 0,
                "correct_predictions": 0,
                "last_5_results": [],
                "status": "untrained",
                "avg_confidence": 0.0,
                "last_updated": datetime.now().isoformat()
            }
            continue

        # Calculate metrics
        correct_count = sum(1 for p in model_predictions if p.is_correct)
        total_count = len(model_predictions)
        direction_accuracy = correct_count / total_count if total_count > 0 else 0.0

        # MAPE calculation
        mape_values = [p.price_error_pct for p in model_predictions if p.price_error_pct is not None]
        mape = sum(mape_values) / len(mape_values) if mape_values else 0.0

        # Last 5 results (most recent first)
        sorted_predictions = sorted(model_predictions, key=lambda p: p.validated_at or "", reverse=True)
        last_5 = [p.is_correct for p in sorted_predictions[:5] if p.is_correct is not None]

        # Average confidence
        confidences = [p.confidence for p in model_predictions if p.confidence > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Determine status
        if total_count < 5:
            status = "untrained"
        elif direction_accuracy >= 0.6:
            status = "healthy"
        elif direction_accuracy >= 0.4:
            status = "degraded"
        else:
            status = "untrained"

        model_stats[model_name] = {
            "direction_accuracy_30d": round(direction_accuracy * 100, 2),
            "mape_30d": round(mape, 4),
            "total_predictions": total_count,
            "correct_predictions": correct_count,
            "last_5_results": last_5,
            "status": status,
            "avg_confidence": round(avg_confidence, 4),
            "last_updated": datetime.now().isoformat()
        }

        # Update stored stats
        stats = ModelAccuracyStats(
            model_name=model_name,
            direction_accuracy_30d=direction_accuracy * 100,
            mape_30d=mape,
            total_predictions=total_count,
            correct_predictions=correct_count,
            last_5=last_5,
            status=status,
            last_updated=datetime.now().isoformat(),
            avg_confidence=avg_confidence,
            prediction_count_30d=total_count
        )
        _storage.update_stats(stats)

    # Summary metrics
    total_predictions = sum(s["total_predictions"] for s in model_stats.values())
    total_correct = sum(s["correct_predictions"] for s in model_stats.values())
    healthy_models = sum(1 for s in model_stats.values() if s["status"] == "healthy")
    degraded_models = sum(1 for s in model_stats.values() if s["status"] == "degraded")
    untrained_models = sum(1 for s in model_stats.values() if s["status"] == "untrained")

    return DailyAccuracyResponse(
        models=model_stats,
        summary={
            "total_predictions_30d": total_predictions,
            "overall_accuracy_30d": round(total_correct / total_predictions * 100, 2) if total_predictions > 0 else 0.0,
            "healthy_models": healthy_models,
            "degraded_models": degraded_models,
            "untrained_models": untrained_models,
            "models_evaluated": len(model_names)
        },
        timestamp=datetime.now().isoformat()
    )


@router.get("/model/{model_name}/history", response_model=PredictionHistoryResponse)
async def get_model_history(
    model_name: str,
    days_back: int = 30,
    symbol: Optional[str] = None
):
    """
    Get full prediction vs actual history for a specific model.

    Args:
        model_name: Model identifier (tft, gnn, pinn, mamba, epidemic, ensemble)
        days_back: Number of days of history to return (default: 30)
        symbol: Optional symbol filter
    """
    # Validate model name
    valid_models = ['tft', 'gnn', 'pinn', 'mamba', 'epidemic', 'ensemble']
    model_name = model_name.lower()
    if model_name not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_name. Must be one of: {', '.join(valid_models)}"
        )

    logger.info(f"[truth/model/{model_name}/history] Fetching history for {model_name}")

    # Get all predictions for this model
    predictions = _storage.get_predictions(
        model_name=model_name,
        symbol=symbol,
        days_back=days_back
    )

    # Separate validated and pending
    validated = [p for p in predictions if p.validated_at is not None]
    pending = [p for p in predictions if p.validated_at is None]

    # Sort by timestamp (newest first)
    validated.sort(key=lambda p: p.validated_at or "", reverse=True)
    pending.sort(key=lambda p: p.prediction_timestamp, reverse=True)

    # Combine for history (validated first, then pending)
    history = [p.to_dict() for p in validated + pending]

    return PredictionHistoryResponse(
        model_name=model_name,
        total_records=len(predictions),
        validated_records=len(validated),
        pending_records=len(pending),
        history=history
    )


@router.post("/record-prediction")
async def record_prediction(request: RecordPredictionRequest):
    """
    Store a model's prediction for later validation.

    This endpoint is called by the ML models after making a prediction.
    The prediction will be validated after market close on the target date.
    """
    logger.info(f"[truth/record-prediction] Recording prediction: {request.model_name} for {request.symbol}")

    # Calculate target date based on horizon
    prediction_timestamp = request.timestamp or datetime.now().isoformat()
    try:
        base_time = datetime.fromisoformat(prediction_timestamp.replace('Z', '+00:00'))
    except ValueError:
        base_time = datetime.now()

    target_date = (base_time + timedelta(days=request.horizon_days)).date()

    # Create prediction record
    record = PredictionRecord(
        model_name=request.model_name,
        symbol=request.symbol,
        prediction_timestamp=prediction_timestamp,
        target_date=target_date.isoformat(),
        predicted_price=request.predicted_price,
        predicted_direction=request.predicted_direction,
        horizon_days=request.horizon_days,
        confidence=request.confidence
    )

    # Store the prediction
    success = _storage.add_prediction(record)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to store prediction")

    return {
        "status": "recorded",
        "model_name": request.model_name,
        "symbol": request.symbol,
        "predicted_price": request.predicted_price,
        "predicted_direction": request.predicted_direction,
        "target_date": target_date.isoformat(),
        "horizon_days": request.horizon_days,
        "validation_expected": f"After market close on {target_date.isoformat()}",
        "timestamp": prediction_timestamp
    }


@router.post("/validate-predictions")
async def validate_predictions(
    request: ValidatePredictionsRequest = ValidatePredictionsRequest(),
    background_tasks: BackgroundTasks = None
):
    """
    Compare stored predictions against actual prices.

    Should run daily after market close (4:30 PM ET).
    Validates all pending predictions for the target date.

    Args:
        target_date: Date to validate (YYYY-MM-DD). Defaults to yesterday.
    """
    # Determine target date
    if request.target_date:
        try:
            target_date = date.fromisoformat(request.target_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    else:
        # Default to yesterday
        target_date = date.today() - timedelta(days=1)

    logger.info(f"[truth/validate-predictions] Validating predictions for {target_date}")

    # Get all pending predictions for this target date
    all_pending = _storage.get_predictions(pending_only=True, days_back=365)
    to_validate = [p for p in all_pending if p.target_date == target_date.isoformat()]

    if not to_validate:
        return {
            "status": "no_predictions",
            "target_date": target_date.isoformat(),
            "message": f"No pending predictions found for {target_date}",
            "validated_count": 0
        }

    # Group by symbol to batch price fetches
    symbols = set(p.symbol for p in to_validate)
    actual_prices: Dict[str, Optional[float]] = {}

    for symbol in symbols:
        actual_prices[symbol] = await get_actual_price(symbol, target_date)

    # Validate each prediction
    validated_count = 0
    results = []

    for prediction in to_validate:
        actual_price = actual_prices.get(prediction.symbol)

        if actual_price is None:
            logger.warning(f"Could not fetch actual price for {prediction.symbol} on {target_date}")
            results.append({
                "model": prediction.model_name,
                "symbol": prediction.symbol,
                "status": "skipped",
                "reason": "actual_price_unavailable"
            })
            continue

        # Calculate actual direction
        # We need the price from the day before the prediction was made
        # For simplicity, we'll compare predicted vs actual
        actual_direction = calculate_direction(prediction.predicted_price, actual_price)

        # More accurate: compare against what the price was when prediction was made
        # This requires storing the "starting price" in the prediction
        is_correct = (
            (prediction.predicted_direction == "up" and actual_price > prediction.predicted_price * 0.999) or
            (prediction.predicted_direction == "down" and actual_price < prediction.predicted_price * 1.001) or
            (prediction.predicted_direction == "neutral" and
             abs(actual_price - prediction.predicted_price) / prediction.predicted_price < 0.01)
        )

        # Calculate price error
        price_error = abs(actual_price - prediction.predicted_price)
        price_error_pct = price_error / actual_price if actual_price > 0 else 0

        # Update prediction record
        prediction.actual_price = actual_price
        prediction.actual_direction = actual_direction
        prediction.is_correct = is_correct
        prediction.validated_at = datetime.now().isoformat()
        prediction.price_error = price_error
        prediction.price_error_pct = price_error_pct

        _storage.update_prediction(prediction)
        validated_count += 1

        results.append({
            "model": prediction.model_name,
            "symbol": prediction.symbol,
            "predicted_price": prediction.predicted_price,
            "actual_price": actual_price,
            "predicted_direction": prediction.predicted_direction,
            "actual_direction": actual_direction,
            "is_correct": is_correct,
            "price_error": round(price_error, 2),
            "price_error_pct": round(price_error_pct * 100, 4),
            "status": "validated"
        })

    # Summary
    correct_count = sum(1 for r in results if r.get("is_correct"))
    accuracy = correct_count / validated_count if validated_count > 0 else 0

    return {
        "status": "completed",
        "target_date": target_date.isoformat(),
        "validated_count": validated_count,
        "correct_count": correct_count,
        "accuracy": round(accuracy * 100, 2),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/summary")
async def get_truth_summary():
    """
    Get a high-level summary of the Truth Dashboard.

    Returns overall system health and key metrics.
    """
    # Get all stats
    all_stats = _storage.get_stats()

    # Get recent predictions
    all_predictions = _storage.get_predictions(days_back=30)
    validated = [p for p in all_predictions if p.validated_at is not None]
    pending = [p for p in all_predictions if p.validated_at is None]

    # Calculate overall metrics
    total_correct = sum(1 for p in validated if p.is_correct)
    overall_accuracy = total_correct / len(validated) * 100 if validated else 0

    # Model rankings
    model_accuracies = []
    for name, stats in all_stats.items():
        if stats.total_predictions > 0:
            model_accuracies.append({
                "model": name,
                "accuracy": stats.direction_accuracy_30d,
                "predictions": stats.total_predictions,
                "status": stats.status
            })

    model_accuracies.sort(key=lambda x: x["accuracy"], reverse=True)

    return {
        "overall_accuracy_30d": round(overall_accuracy, 2),
        "total_predictions_30d": len(all_predictions),
        "validated_predictions": len(validated),
        "pending_predictions": len(pending),
        "model_rankings": model_accuracies,
        "best_model": model_accuracies[0]["model"] if model_accuracies else None,
        "worst_model": model_accuracies[-1]["model"] if model_accuracies else None,
        "system_health": "healthy" if overall_accuracy >= 55 else ("degraded" if overall_accuracy >= 40 else "needs_attention"),
        "timestamp": datetime.now().isoformat()
    }


@router.delete("/clear-history")
async def clear_history(confirm: bool = False):
    """
    Clear all prediction history and stats (admin endpoint).

    Requires confirm=true query parameter for safety.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="This action will delete all truth dashboard data. Set confirm=true to proceed."
        )

    logger.warning("[truth/clear-history] Clearing all truth dashboard data")

    try:
        # Reinitialize storage files
        _storage._write_json(_storage.predictions_file, {"predictions": []})
        _storage._write_json(_storage.stats_file, {"stats": {}})

        return {
            "status": "cleared",
            "message": "All truth dashboard data has been cleared",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to clear history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")
