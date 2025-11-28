"""
Backtesting Module for ML Model Validation

Institutional-grade walk-forward backtesting framework for validating
GNN, Mamba, PINN, Epidemic, and Ensemble models on historical stock data.

Key Features:
- Walk-forward analysis (expanding window, no lookahead bias)
- Comprehensive metrics (RMSE, MAE, directional accuracy, Sharpe, max drawdown)
- 3-tier caching (filesystem, memory, API with circuit breaker)
- Feature alignment with live prediction pipeline
- Parallel execution support

Architecture specified in: BACKTESTING_ARCHITECTURE.md
"""

from .metrics import (
    calculate_rmse,
    calculate_mae,
    calculate_mape,
    calculate_directional_accuracy,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_all_metrics
)

__version__ = "1.0.0"
__author__ = "ML Team"
__status__ = "Production Ready"
