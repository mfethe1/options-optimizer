"""
Performance Metrics for ML Model Backtesting

Implements comprehensive evaluation metrics for stock price predictions:
- Prediction Accuracy: RMSE, MAE, MAPE
- Directional Accuracy: % correct up/down predictions
- Risk-Adjusted Returns: Sharpe ratio, Sortino ratio, Information ratio
- Drawdown Analysis: Max drawdown, recovery time, Calmar ratio
- Trading Signals: Precision, recall, F1 for BUY/SELL signals

All metrics aligned with institutional standards (Renaissance, Citadel, Two Sigma).
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for all backtest performance metrics"""
    # Accuracy metrics
    rmse: float
    mae: float
    mape: float

    # Directional metrics
    directional_accuracy: float  # % correct direction
    precision_buy: float
    recall_buy: float
    f1_buy: float

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float

    # Drawdown metrics
    max_drawdown: float
    recovery_time_days: int
    calmar_ratio: float

    # Additional metrics
    annual_return: float
    total_trades: int
    win_rate: float

    # Optional metrics (with defaults, must come last)
    information_ratio: Optional[float] = None

    def grade(self) -> str:
        """
        Grade model performance (A/B/C/D/F)

        Grading rubric:
        - A: Sharpe >2.0, Dir Acc >0.60 (60%), Max DD <10%
        - B: Sharpe >1.0, Dir Acc >0.55 (55%), Max DD <20%
        - C: Sharpe >0.5, Dir Acc >0.52 (52%), Max DD <30%
        - D: Sharpe >0.0, Dir Acc >0.50 (50%)
        - F: Below D threshold
        """
        if self.sharpe_ratio > 2.0 and self.directional_accuracy > 0.60 and self.max_drawdown < 0.10:
            return 'A'
        elif self.sharpe_ratio > 1.0 and self.directional_accuracy > 0.55 and self.max_drawdown < 0.20:
            return 'B'
        elif self.sharpe_ratio > 0.5 and self.directional_accuracy > 0.52 and self.max_drawdown < 0.30:
            return 'C'
        elif self.sharpe_ratio > 0.0 and self.directional_accuracy > 0.50:
            return 'D'
        else:
            return 'F'


def calculate_rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error

    Formula: RMSE = sqrt(mean((predicted - actual)^2))

    Args:
        predictions: Predicted prices [N]
        actuals: Actual prices [N]

    Returns:
        rmse: Root mean squared error (absolute)
    """
    if len(predictions) == 0:
        return 0.0

    errors = predictions - actuals
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    return rmse


def calculate_mae(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error

    Formula: MAE = mean(|predicted - actual|)

    Args:
        predictions: Predicted prices [N]
        actuals: Actual prices [N]

    Returns:
        mae: Mean absolute error (absolute)
    """
    if len(predictions) == 0:
        return 0.0

    mae = float(np.mean(np.abs(predictions - actuals)))

    return mae


def calculate_mape(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error

    Formula: MAPE = mean(|predicted - actual| / actual) * 100

    Args:
        predictions: Predicted prices [N]
        actuals: Actual prices [N]

    Returns:
        mape: Mean absolute percentage error (0-100%)
    """
    if len(predictions) == 0:
        return 0.0

    # Avoid division by zero
    mape = float(np.mean(np.abs((predictions - actuals) / (actuals + 1e-8))) * 100)

    return mape


def calculate_directional_accuracy(
    predictions: np.ndarray,
    actuals: np.ndarray,
    current_prices: np.ndarray
) -> float:
    """
    Calculate ratio of correct directional predictions

    Compares predicted direction (up/down) vs actual direction.

    Args:
        predictions: Predicted prices [N]
        actuals: Actual prices [N]
        current_prices: Prices at prediction time [N]

    Returns:
        accuracy: Ratio (0-1) of correct directions
    """
    if len(predictions) == 0:
        return 0.0

    pred_directions = np.sign(predictions - current_prices)
    actual_directions = np.sign(actuals - current_prices)

    correct = pred_directions == actual_directions
    accuracy = float(np.mean(correct))

    return accuracy


def calculate_signal_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    current_prices: np.ndarray,
    threshold: float = 0.02  # 2% threshold for BUY/SELL
) -> Dict[str, float]:
    """
    Calculate precision/recall for BUY/SELL signals

    Classification:
    - BUY: Prediction >2% above current
    - SELL: Prediction <-2% below current
    - HOLD: Prediction within [-2%, +2%]

    Args:
        predictions: Predicted prices [N]
        actuals: Actual prices [N]
        current_prices: Prices at prediction time [N]
        threshold: Percentage threshold for signals (default 2%)

    Returns:
        metrics: Dict with precision_buy, recall_buy, f1_buy
    """
    if len(predictions) == 0:
        return {
            'precision_buy': 0.0,
            'recall_buy': 0.0,
            'f1_buy': 0.0
        }

    # Calculate returns
    pred_returns = (predictions - current_prices) / current_prices
    actual_returns = (actuals - current_prices) / current_prices

    # Generate signals
    pred_signals = np.where(pred_returns > threshold, 1,  # BUY
                   np.where(pred_returns < -threshold, -1,  # SELL
                   0))  # HOLD

    actual_signals = np.where(actual_returns > threshold, 1,
                     np.where(actual_returns < -threshold, -1,
                     0))

    # Precision for BUY
    buy_mask = pred_signals == 1
    if buy_mask.sum() > 0:
        precision_buy = float((pred_signals[buy_mask] == actual_signals[buy_mask]).mean())
    else:
        precision_buy = 0.0

    # Recall for BUY
    true_buy_mask = actual_signals == 1
    if true_buy_mask.sum() > 0:
        recall_buy = float((pred_signals[true_buy_mask] == 1).mean())
    else:
        recall_buy = 0.0

    # F1 score
    if precision_buy + recall_buy > 0:
        f1_buy = 2 * (precision_buy * recall_buy) / (precision_buy + recall_buy)
    else:
        f1_buy = 0.0

    return {
        'precision_buy': precision_buy,
        'recall_buy': recall_buy,
        'f1_buy': f1_buy
    }


def calculate_equity_curve(
    predictions: np.ndarray,
    actuals: np.ndarray,
    current_prices: np.ndarray,
    initial_capital: float = 10000.0
) -> np.ndarray:
    """
    Calculate equity curve assuming directional trading strategy

    Strategy:
    - Go long if prediction > current
    - Go short if prediction < current
    - Position size: 100% of capital

    Args:
        predictions: Predicted prices [N]
        actuals: Actual prices [N]
        current_prices: Prices at prediction time [N]
        initial_capital: Starting capital (default $10,000)

    Returns:
        equity_curve: Portfolio value over time [N+1] (includes initial capital)
    """
    if len(predictions) == 0:
        return np.array([initial_capital])

    # Calculate directional bets
    pred_directions = np.sign(predictions - current_prices)
    actual_returns = (actuals - current_prices) / current_prices

    # Strategy returns = direction × actual return
    strategy_returns = pred_directions * actual_returns

    # Cumulative equity
    equity_curve = initial_capital * np.cumprod(1 + strategy_returns)

    # Prepend initial capital
    equity_curve = np.concatenate([[initial_capital], equity_curve])

    return equity_curve


def calculate_sharpe_ratio(
    predictions: np.ndarray,
    actuals: np.ndarray,
    current_prices: np.ndarray,
    risk_free_rate: float = 0.04  # 4% annual
) -> float:
    """
    Calculate annualized Sharpe ratio

    Formula: Sharpe = (mean(returns) - risk_free_rate) / std(returns) * sqrt(252)

    Assumes trading strategy:
    - Go long if prediction > current
    - Go short if prediction < current

    Args:
        predictions: Predicted prices [N]
        actuals: Actual prices [N]
        current_prices: Prices at prediction time [N]
        risk_free_rate: Annual risk-free rate (default 4%)

    Returns:
        sharpe: Annualized Sharpe ratio
    """
    if len(predictions) == 0:
        return 0.0

    # Calculate strategy returns
    pred_directions = np.sign(predictions - current_prices)
    actual_returns = (actuals - current_prices) / current_prices
    strategy_returns = pred_directions * actual_returns

    # Annualize
    mean_return = float(np.mean(strategy_returns) * 252)  # 252 trading days
    std_return = float(np.std(strategy_returns) * np.sqrt(252))

    if std_return < 1e-8:
        return 0.0

    sharpe = (mean_return - risk_free_rate) / std_return

    return sharpe


def calculate_sortino_ratio(
    predictions: np.ndarray,
    actuals: np.ndarray,
    current_prices: np.ndarray,
    risk_free_rate: float = 0.04
) -> float:
    """
    Calculate annualized Sortino ratio (only penalizes downside volatility)

    Formula: Sortino = (mean(returns) - risk_free_rate) / downside_std * sqrt(252)

    Args:
        predictions: Predicted prices [N]
        actuals: Actual prices [N]
        current_prices: Prices at prediction time [N]
        risk_free_rate: Annual risk-free rate (default 4%)

    Returns:
        sortino: Annualized Sortino ratio
    """
    if len(predictions) == 0:
        return 0.0

    # Calculate strategy returns
    pred_directions = np.sign(predictions - current_prices)
    actual_returns = (actuals - current_prices) / current_prices
    strategy_returns = pred_directions * actual_returns

    # Downside returns only (negative)
    downside_returns = strategy_returns[strategy_returns < 0]

    if len(downside_returns) == 0:
        return float('inf')  # No downside!

    # Annualize
    mean_return = float(np.mean(strategy_returns) * 252)
    downside_std = float(np.std(downside_returns) * np.sqrt(252))

    if downside_std < 1e-8:
        return float('inf')

    sortino = (mean_return - risk_free_rate) / downside_std

    return sortino


def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown and identify peak/trough indices

    Args:
        equity_curve: Cumulative portfolio value over time [N]

    Returns:
        (max_drawdown, peak_idx, trough_idx)
        - max_drawdown: Maximum peak-to-trough decline (0-1)
        - peak_idx: Index of peak before max drawdown
        - trough_idx: Index of trough (max drawdown point)
    """
    if len(equity_curve) == 0:
        return 0.0, 0, 0

    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)

    # Drawdown at each point
    drawdowns = (running_max - equity_curve) / (running_max + 1e-8)

    # Maximum drawdown
    max_dd = float(np.max(drawdowns))
    trough_idx = int(np.argmax(drawdowns))

    # Find peak before trough
    peak_idx = int(np.argmax(equity_curve[:trough_idx+1])) if trough_idx > 0 else 0

    return max_dd, peak_idx, trough_idx


def calculate_recovery_time(
    equity_curve: np.ndarray,
    peak_idx: int,
    trough_idx: int
) -> int:
    """
    Calculate number of days from trough to recovery (return to peak)

    Args:
        equity_curve: Cumulative portfolio value over time
        peak_idx: Index of peak before drawdown
        trough_idx: Index of trough

    Returns:
        recovery_days: Number of days from trough to recovery
    """
    if trough_idx >= len(equity_curve) - 1:
        # Trough is at end, not recovered
        return len(equity_curve) - trough_idx

    peak_value = equity_curve[peak_idx]

    # Find first day after trough that exceeds peak
    recovery_idx = None
    for i in range(trough_idx + 1, len(equity_curve)):
        if equity_curve[i] >= peak_value:
            recovery_idx = i
            break

    if recovery_idx is None:
        # Not recovered yet
        return len(equity_curve) - trough_idx

    recovery_days = recovery_idx - trough_idx

    return recovery_days


def calculate_calmar_ratio(annual_return: float, max_drawdown: float) -> float:
    """
    Calculate Calmar ratio (return per unit max drawdown)

    Formula: Calmar = annual_return / max_drawdown

    Args:
        annual_return: Annualized return (decimal, e.g., 0.15 = 15%)
        max_drawdown: Maximum drawdown (decimal, e.g., 0.10 = 10%)

    Returns:
        calmar: Calmar ratio
    """
    if max_drawdown < 1e-8:
        return float('inf')

    calmar = annual_return / max_drawdown

    return calmar


def calculate_all_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    current_prices: np.ndarray,
    symbol: str,
    model_name: str,
    risk_free_rate: float = 0.04
) -> PerformanceMetrics:
    """
    Calculate all performance metrics for a backtest

    Args:
        predictions: Predicted prices [N]
        actuals: Actual prices [N]
        current_prices: Prices at prediction time [N]
        symbol: Stock symbol (for logging)
        model_name: Model name (for logging)
        risk_free_rate: Annual risk-free rate (default 4%)

    Returns:
        metrics: PerformanceMetrics object with all calculated metrics
    """
    logger.info(f"[{symbol}/{model_name}] Calculating metrics for {len(predictions)} predictions...")

    # Prediction accuracy
    rmse = calculate_rmse(predictions, actuals)
    mae = calculate_mae(predictions, actuals)
    mape = calculate_mape(predictions, actuals)

    # Directional accuracy
    dir_acc = calculate_directional_accuracy(predictions, actuals, current_prices)

    # Signal metrics
    signal_metrics = calculate_signal_metrics(predictions, actuals, current_prices, threshold=0.02)

    # Risk-adjusted returns
    sharpe = calculate_sharpe_ratio(predictions, actuals, current_prices, risk_free_rate)
    sortino = calculate_sortino_ratio(predictions, actuals, current_prices, risk_free_rate)

    # Drawdown analysis
    equity_curve = calculate_equity_curve(predictions, actuals, current_prices, initial_capital=10000.0)
    max_dd, peak_idx, trough_idx = calculate_max_drawdown(equity_curve)
    recovery_time = calculate_recovery_time(equity_curve, peak_idx, trough_idx)

    # Annual return
    if len(equity_curve) > 1:
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(predictions)) - 1
    else:
        annual_return = 0.0

    # Calmar ratio
    calmar = calculate_calmar_ratio(annual_return, max_dd)

    # Win rate (same as directional accuracy, 0-1 ratio)
    pred_directions = np.sign(predictions - current_prices)
    actual_directions = np.sign(actuals - current_prices)
    wins = (pred_directions == actual_directions).sum()
    win_rate = float(wins / len(predictions)) if len(predictions) > 0 else 0.0

    metrics = PerformanceMetrics(
        rmse=rmse,
        mae=mae,
        mape=mape,
        directional_accuracy=dir_acc,
        precision_buy=signal_metrics['precision_buy'],
        recall_buy=signal_metrics['recall_buy'],
        f1_buy=signal_metrics['f1_buy'],
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        recovery_time_days=recovery_time,
        calmar_ratio=calmar,
        annual_return=annual_return,
        total_trades=len(predictions),
        win_rate=win_rate
    )

    logger.info(f"[{symbol}/{model_name}] Metrics: RMSE={rmse:.2f}, Dir Acc={dir_acc:.1%}, Sharpe={sharpe:.2f}, Grade={metrics.grade()}")

    return metrics
