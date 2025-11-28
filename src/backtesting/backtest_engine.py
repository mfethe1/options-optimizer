"""
Backtesting Engine for ML Stock Price Predictions

Implements walk-forward analysis to validate model performance on historical data.
Prevents lookahead bias by ensuring models only use data available at prediction time.
"""

import asyncio
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

from .metrics import (
    PerformanceMetrics,
    calculate_rmse,
    calculate_mae,
    calculate_mape,
    calculate_directional_accuracy,
    calculate_sharpe_ratio,
    calculate_max_drawdown
)

logger = logging.getLogger(__name__)


def _validate_prediction(pred_value: Any, symbol: str, date: Any, current_price: float) -> Optional[float]:
    """
    Validate prediction value is a finite, positive number.

    Args:
        pred_value: Predicted price from model
        symbol: Stock symbol (for logging)
        date: Prediction date (for logging)
        current_price: Current stock price (for sanity check)

    Returns:
        Validated float value or None if invalid
    """
    if pred_value is None:
        logger.warning(f"[{symbol}] Prediction is None at {date}")
        return None

    # Type check
    if not isinstance(pred_value, (int, float, np.number)):
        logger.warning(
            f"[{symbol}] Invalid prediction type at {date}: "
            f"expected number, got {type(pred_value).__name__}"
        )
        return None

    # Convert to float
    try:
        pred_float = float(pred_value)
    except (ValueError, TypeError) as e:
        logger.warning(f"[{symbol}] Cannot convert prediction to float at {date}: {e}")
        return None

    # Check for NaN/Inf
    if not np.isfinite(pred_float):
        logger.warning(
            f"[{symbol}] Invalid prediction value at {date}: {pred_float}"
        )
        return None

    # Check for negative prices (invalid for stocks)
    if pred_float <= 0:
        logger.warning(
            f"[{symbol}] Non-positive prediction at {date}: {pred_float}"
        )
        return None

    # Sanity check: prediction shouldn't be more than 10x or less than 0.1x current price
    # This catches obvious model bugs (e.g., outputting raw probabilities instead of prices)
    if current_price > 0:
        ratio = pred_float / current_price
        if ratio > 10.0 or ratio < 0.1:
            logger.warning(
                f"[{symbol}] Unrealistic prediction at {date}: ${pred_float:.2f} "
                f"(current: ${current_price:.2f}, ratio: {ratio:.1f}x)"
            )
            return None

    return pred_float


@dataclass
class BacktestResult:
    """Results from backtesting a single model"""
    symbol: str
    model_name: str
    start_date: str
    end_date: str
    metrics: PerformanceMetrics
    predictions: List[float]
    actuals: List[float]
    dates: List[str]
    grade: str


class BacktestEngine:
    """
    Production-grade backtesting engine for ML models

    Features:
    - Walk-forward analysis with expanding window
    - No lookahead bias (uses only historical data)
    - Parallel execution for multiple stocks
    - Comprehensive performance metrics
    """

    def __init__(self):
        self.results_cache: Dict[str, BacktestResult] = {}

        # Cache directory for historical data
        self.cache_dir = Path("data/backtest_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_hours = 24

    async def backtest_model(
        self,
        model_name: str,
        symbol: str,
        start_date: str,
        end_date: str,
        prediction_function: callable
    ) -> BacktestResult:
        """
        Backtest a single model on one symbol

        Args:
            model_name: 'gnn', 'mamba', 'pinn', 'epidemic', or 'ensemble'
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            prediction_function: Async function that takes (symbol, price) -> prediction

        Returns:
            BacktestResult with performance metrics
        """
        logger.info(f"[Backtest] Starting {model_name} for {symbol} ({start_date} to {end_date})")

        try:
            # Load historical data
            historical_data = await self._load_historical_data(symbol, start_date, end_date)

            if historical_data is None or len(historical_data) < 30:
                logger.error(f"Insufficient data for {symbol}")
                return None

            # Walk-forward predictions
            predictions = []
            actuals = []
            current_prices = []
            dates = []

            # Start predictions after 30 days (need history for features)
            for i in range(30, len(historical_data)):
                current_price = historical_data['Close'].iloc[i-1]
                actual_next_price = historical_data['Close'].iloc[i]
                current_date = historical_data.index[i]

                try:
                    # CRITICAL: No lookahead bias - prediction_function only receives:
                    # - symbol: Stock identifier
                    # - current_price: Price at time i-1 (before actual_next_price at time i)
                    # Model does NOT have access to:
                    # - actual_next_price (future data at time i)
                    # - historical_data after index i-1
                    # This ensures walk-forward validation integrity
                    pred_result = await prediction_function(symbol, current_price)

                    # Validate prediction result exists
                    if not pred_result or 'prediction' not in pred_result:
                        logger.warning(f"No prediction returned for {symbol} at {current_date}")
                        continue  # Skip entire row - don't append to ANY array

                    # Validate prediction value (type, range, sanity checks)
                    predicted_price = _validate_prediction(
                        pred_result['prediction'],
                        symbol,
                        current_date,
                        current_price
                    )

                    if predicted_price is None:
                        # Validation failed - skip this prediction
                        continue

                    # CRITICAL: Append to ALL arrays atomically (all or nothing)
                    # This prevents array misalignment when predictions fail
                    predictions.append(predicted_price)
                    actuals.append(actual_next_price)
                    current_prices.append(current_price)
                    dates.append(str(current_date.date()))

                except Exception as e:
                    logger.warning(f"Prediction failed at {current_date}: {e}")
                    continue  # Skip entire row - arrays remain aligned

            if len(predictions) < 10:
                logger.error(f"Too few predictions for {symbol}: {len(predictions)}")
                return None

            # Validate array alignment (critical for correct metrics)
            assert len(predictions) == len(actuals) == len(current_prices) == len(dates), \
                f"Array misalignment detected: predictions={len(predictions)}, actuals={len(actuals)}, " \
                f"current_prices={len(current_prices)}, dates={len(dates)}"

            # Calculate metrics
            predictions_arr = np.array(predictions)
            actuals_arr = np.array(actuals)
            current_prices_arr = np.array(current_prices)

            # Basic metrics
            rmse = calculate_rmse(predictions_arr, actuals_arr)
            mae = calculate_mae(predictions_arr, actuals_arr)
            mape = calculate_mape(predictions_arr, actuals_arr)
            dir_accuracy = calculate_directional_accuracy(predictions_arr, actuals_arr, current_prices_arr)

            # Risk-adjusted returns
            sharpe = calculate_sharpe_ratio(predictions_arr, actuals_arr, current_prices_arr)

            # Drawdown (calculate equity curve first)
            from .metrics import calculate_equity_curve, calculate_sortino_ratio, calculate_calmar_ratio
            equity_curve = calculate_equity_curve(predictions_arr, actuals_arr, current_prices_arr)
            max_dd, _, _ = calculate_max_drawdown(equity_curve)

            # Annual return from equity curve
            if len(equity_curve) > 1:
                total_return = (equity_curve[-1] / equity_curve[0]) - 1
                annual_return = (1 + total_return) ** (252 / len(predictions)) - 1
            else:
                annual_return = 0.0

            # Sortino ratio and Calmar ratio
            sortino = calculate_sortino_ratio(predictions_arr, actuals_arr, current_prices_arr)
            calmar = calculate_calmar_ratio(annual_return, max_dd)

            # Create metrics object
            metrics = PerformanceMetrics(
                rmse=rmse,
                mae=mae,
                mape=mape,
                directional_accuracy=dir_accuracy,
                precision_buy=0.0,  # Simplified
                recall_buy=0.0,
                f1_buy=0.0,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                max_drawdown=max_dd,
                recovery_time_days=0,  # Simplified
                calmar_ratio=calmar,
                annual_return=annual_return,
                total_trades=len(predictions),
                win_rate=dir_accuracy
            )

            # Grade performance
            grade = metrics.grade()

            result = BacktestResult(
                symbol=symbol,
                model_name=model_name,
                start_date=start_date,
                end_date=end_date,
                metrics=metrics,
                predictions=predictions,
                actuals=actuals,
                dates=dates,
                grade=grade
            )

            logger.info(f"[Backtest] {model_name}/{symbol}: RMSE={rmse:.3f}, Accuracy={dir_accuracy:.1%}, Grade={grade}")

            return result

        except Exception as e:
            logger.error(f"[Backtest] Failed for {model_name}/{symbol}: {e}", exc_info=True)
            return None

    async def backtest_all_models(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, BacktestResult]:
        """
        Backtest all 5 models on one symbol

        Returns dict: {'gnn': BacktestResult, 'mamba': ..., ...}
        """
        from ..api.ml_integration_helpers import (
            get_gnn_prediction,
            get_mamba_prediction,
            get_pinn_prediction
        )

        models = {
            'gnn': get_gnn_prediction,
            'mamba': get_mamba_prediction,
            'pinn': get_pinn_prediction
        }

        results = {}

        for model_name, pred_func in models.items():
            result = await self.backtest_model(
                model_name=model_name,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                prediction_function=pred_func
            )

            if result:
                results[model_name] = result

        return results

    async def _load_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Load historical price data with file-based caching

        Cache strategy:
        - First check disk cache (24-hour expiry)
        - If cache miss or expired, fetch from yfinance
        - Save fetched data to cache for future use
        """
        # Generate cache key and file path
        cache_key = f"{symbol}_{start_date}_{end_date}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        # Check if cache exists and is fresh (< 24 hours old)
        if cache_file.exists():
            try:
                # Check file age
                file_modified_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                cache_age_hours = (datetime.now() - file_modified_time).total_seconds() / 3600

                if cache_age_hours < self.cache_expiry_hours:
                    # Load from cache
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)

                    logger.info(
                        f"[Cache HIT] Loaded {symbol} from cache "
                        f"(age: {cache_age_hours:.1f}h, {len(data)} rows)"
                    )
                    return data
                else:
                    logger.info(
                        f"[Cache EXPIRED] Cache for {symbol} is {cache_age_hours:.1f}h old "
                        f"(max: {self.cache_expiry_hours}h)"
                    )
            except Exception as e:
                logger.warning(f"[Cache ERROR] Failed to load cache for {symbol}: {e}")
                # Continue to fetch from yfinance

        # Cache miss or expired - fetch from yfinance
        logger.info(f"[Cache MISS] Fetching {symbol} from yfinance ({start_date} to {end_date})")

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)

            if data.empty:
                logger.error(f"No data returned for {symbol}")
                return None

            # Save to cache for future use
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

                logger.info(
                    f"[Cache SAVE] Cached {symbol} to disk "
                    f"({len(data)} rows, {cache_file.stat().st_size / 1024:.1f} KB)"
                )
            except Exception as e:
                logger.warning(f"[Cache SAVE ERROR] Failed to cache {symbol}: {e}")
                # Non-fatal - we still have the data

            return data

        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return None

    def generate_summary_report(self, results: Dict[str, BacktestResult]) -> str:
        """Generate markdown summary report"""
        report = "# Backtesting Results\n\n"
        report += f"**Symbol:** {list(results.values())[0].symbol if results else 'N/A'}\n"
        report += f"**Period:** {list(results.values())[0].start_date if results else 'N/A'} to {list(results.values())[0].end_date if results else 'N/A'}\n\n"

        report += "## Model Performance Comparison\n\n"
        report += "| Model | RMSE | MAE | Dir. Accuracy | Sharpe | Grade |\n"
        report += "|-------|------|-----|---------------|--------|-------|\n"

        for model_name, result in results.items():
            m = result.metrics
            report += f"| {model_name.upper()} | {m.rmse:.3f} | {m.mae:.2f} | {m.directional_accuracy:.1%} | {m.sharpe_ratio:.2f} | {result.grade} |\n"

        report += "\n## Key Findings\n\n"

        # Find best model
        best_model = min(results.items(), key=lambda x: x[1].metrics.rmse)
        report += f"- **Best Model:** {best_model[0].upper()} (RMSE: {best_model[1].metrics.rmse:.3f})\n"

        # Check if beats random
        avg_dir_accuracy = np.mean([r.metrics.directional_accuracy for r in results.values()])
        beats_random = avg_dir_accuracy > 0.50
        report += f"- **Beats Random:** {'✅ Yes' if beats_random else '❌ No'} (Avg: {avg_dir_accuracy:.1%})\n"

        return report
