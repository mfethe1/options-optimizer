# ML Model Backtesting Architecture

**Project:** Options Probability Analysis System
**Date:** 2025-11-09
**Architect:** ML Neural Network Architect
**Status:** Architecture Design Complete
**Objective:** Validate 5 neural network models through rigorous backtesting to guide production deployment

---

## 1. Executive Summary

### Mission

Validate the performance of 5 neural network models (GNN, Mamba, PINN, Epidemic, Ensemble) on 1-3 years of historical stock data to answer:
- Which models deliver superior risk-adjusted returns?
- What are optimal ensemble weights by sector/regime?
- Where do models excel vs fail?
- Are we ready for production deployment?

### Methodology Overview

**Backtesting Approach:** Walk-forward analysis with expanding window
- Prevents lookahead bias (no future information)
- Mimics production usage (retrain periodically, predict forward)
- Industry standard (Renaissance Technologies, Two Sigma methodology)

**Time Period:** 3 years (2022-01-01 to 2024-12-31)
- Captures multiple market regimes:
  - 2022: Bear market (-19% SPY)
  - 2023: Bull recovery (+24% SPY)
  - 2024: Continued growth
- Sufficient for statistical significance (750+ trading days)
- Balances regime diversity vs iteration speed

**Stock Universe:** Initial validation on 10 stocks → Full 46-stock rollout
- Phase 1: 10 stocks (2 tech, 2 finance, 2 healthcare, 2 consumer, 2 indices)
- Phase 2: Expand to all 46 pre-trained GNN models
- Rationale: Fast iteration, early failure detection, sector diversity

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Methodology** | Walk-forward expanding window | No lookahead bias, production-realistic |
| **Time Period** | 3 years (2022-2024) | Multiple regimes, statistical significance |
| **Initial Scope** | 10 stocks → 46 stocks | Fast validation → comprehensive coverage |
| **Parallelization** | 5 workers (stocks) × 1 worker (models) | Balance speed vs memory |
| **Data Caching** | Persistent filesystem cache | Avoid repeated API calls, respect rate limits |
| **Prediction Horizons** | 1d (primary), 30d (secondary) | Match production usage, multi-horizon validation |

### Expected Outcomes

**Performance Benchmarks:**
- RMSE: <5% for 1-day, <10% for 30-day predictions
- Directional Accuracy: >55% (vs 50% random baseline)
- Sharpe Ratio: >1.0 (good), >2.0 (excellent)
- Ensemble outperforms best individual model by >5%

**Deliverables:**
1. Comprehensive backtesting framework (6 modules, ~1,550 lines)
2. Initial validation results (10 stocks × 3 years)
3. Performance report with model rankings
4. Recommendations for production deployment

**Timeline:**
- Phase 1 (Design): 90 minutes ✅ (this document)
- Phase 2 (Implementation): 3-4 hours (Expert Code Writer)
- Phase 3 (Initial Backtest): 30 minutes (10 stocks)
- Phase 4 (Full Backtest): 2-4 hours (46 stocks)

---

## 2. Backtesting Methodology

### 2.1 Walk-Forward Analysis Specification

**Architecture:** Expanding window (anchored)

```
Training Window = [t0, t_train]  (expands over time)
Prediction Point = t_train + 1
Validation Point = t_train + horizon (1d or 30d)

Example (1-day horizon):
  Day 1: Train on [2022-01-01 to 2022-12-31], Predict 2023-01-02, Validate 2023-01-02
  Day 2: Train on [2022-01-01 to 2023-01-02], Predict 2023-01-03, Validate 2023-01-03
  Day 3: Train on [2022-01-01 to 2023-01-03], Predict 2023-01-04, Validate 2023-01-04
  ...

Visualization:
|------------- Train Window (expanding) -------------|→ Predict → Validate
2022-01-01                                    2023-01-01    2023-01-02
|----------------------- Train Window -------------------|→ Predict → Validate
2022-01-01                                        2023-01-02    2023-01-03
```

**Why Expanding Window?**
- Mimics production: Models retrained with all available history
- No data waste: Uses all historical information
- Realistic: Matches how models are actually deployed
- Industry standard: Used by Renaissance, Citadel, Two Sigma

**Alternative Considered:** Rolling window (fixed size)
- Pros: Adapts to regime changes faster
- Cons: Wastes historical data, unstable in long-term trends
- Decision: Expanding window for stability and max information usage

**No Lookahead Guarantee:**
- All features computed using only data up to `t_train`
- Predictions made for `t_train + horizon`
- Validation uses actual prices at `t_train + horizon`
- No peeking at future volatility, sentiment, or prices

### 2.2 Time Period Selection

**Selected Period:** 3 years (2022-01-01 to 2024-12-31)

**Rationale:**

1. **Multiple Regime Coverage:**
   - 2022: Bear market (Fed tightening, -19% SPY)
   - 2023: Bull recovery (AI boom, +24% SPY)
   - 2024: Consolidation (geopolitical tensions)
   - Tests models in diverse conditions

2. **Statistical Significance:**
   - ~750 trading days
   - >100 prediction samples per stock per model
   - Sufficient for Sharpe ratio confidence intervals

3. **Data Availability:**
   - yfinance free tier: 5+ years available
   - 3 years balances depth vs API calls
   - Recent enough for regime relevance

4. **Iteration Speed:**
   - 3 years: ~4 hours for 46 stocks (parallel)
   - 5 years: ~7 hours (too slow for iteration)
   - 1 year: ~1.5 hours (too little regime diversity)

**Alternative Periods Considered:**

| Period | Pros | Cons | Decision |
|--------|------|------|----------|
| 1 year | Fast iteration (1.5h) | Limited regimes, low statistical power | Rejected |
| 3 years | Multiple regimes, good statistics | Moderate iteration time | **Selected** |
| 5 years | Maximum history | Slow iteration, stale data (pre-COVID) | Rejected |

**Implementation:** Configurable via CLI argument `--years {1,3,5}`

### 2.3 Prediction Horizon Strategy

**Primary Horizon:** 1-day ahead (most common production use case)

**Secondary Horizon:** 30-day ahead (strategic allocation, Mamba specialization)

**Evaluation Approach:**

1. **1-Day Predictions:**
   - Evaluate all models daily
   - Use for Sharpe ratio, directional accuracy
   - Primary metrics: RMSE, MAE, % correct direction

2. **30-Day Predictions:**
   - Evaluate Mamba and Ensemble specifically (Mamba's strength)
   - Use for long-term strategy validation
   - Metrics: RMSE, correlation with actual 30-day return

3. **Multi-Horizon Comparison:**
   - Mamba predicts 1d, 5d, 10d, 30d simultaneously
   - Validate each horizon independently
   - Identify optimal horizon per model

**Horizon-Specific Benchmarks:**

| Horizon | RMSE Target | Directional Accuracy | Use Case |
|---------|-------------|---------------------|----------|
| 1-day | <5% | >55% | Day trading, market making |
| 5-day | <7% | >53% | Swing trading |
| 10-day | <9% | >52% | Position trading |
| 30-day | <12% | >50% | Strategic allocation |

### 2.4 Feature Alignment Validation

**Critical Requirement:** Backtest features MUST match live prediction features exactly.

**Live Prediction Features (from `ml_integration_helpers.py`):**

1. **GNN:**
   - 20-day price history (correlations)
   - Node features: volatility (std of 20-day returns), momentum (mean return), volume_norm (1.0)
   - 60-day correlation matrix (for graph structure)

2. **Mamba:**
   - 1000-day price history (long sequences)
   - Normalized prices (z-score normalization)

3. **PINN:**
   - Implied volatility (from options chain or 30-day historical vol)
   - Risk-free rate (10-year Treasury)
   - Current price, strike (ATM), time to expiration (3 months)

4. **Epidemic:**
   - VIX index (volatility proxy)
   - Realized volatility (30-day)
   - Sentiment (if available, else neutral)
   - Volume (normalized)

**Alignment Strategy:**

1. **Feature Extraction Function:**
   ```python
   async def extract_features_at_time(symbol: str, date: datetime, lookback: int) -> Dict:
       """
       Extract features using ONLY data up to `date` (no lookahead)

       Args:
           symbol: Stock symbol
           date: Current date (t_train)
           lookback: Historical days needed (20, 60, 1000)

       Returns:
           features: Dict with all required features
       """
       # Fetch prices from (date - lookback) to date
       prices = fetch_historical_prices_until(symbol, end_date=date, days=lookback)

       # Compute features (volatility, momentum, etc.)
       features = {
           'volatility': np.std(prices[-20:]),
           'momentum': np.mean(prices[-20:] - prices[-40:-20]) / prices[-40:-20],
           'prices': prices,
           # ...
       }

       return features
   ```

2. **No Lookahead Checks:**
   - Assert all features computed before `t_train`
   - Log warning if any feature uses future data
   - Unit test: Generate features for 2023-01-01, assert no data from 2023-01-02+

3. **Cold Start Handling:**
   - First 1000 days: Skip (Mamba needs full history)
   - First 60 days: Skip (GNN needs correlation matrix)
   - First 20 days: Skip (volatility calculation)
   - **Effective start date:** `start_date + 1000 days`

**Validation Tests:**

1. **Feature Reproducibility Test:**
   ```python
   # Extract features for 2023-01-15
   features_v1 = extract_features_at_time('AAPL', date='2023-01-15', lookback=1000)
   features_v2 = extract_features_at_time('AAPL', date='2023-01-15', lookback=1000)
   assert features_v1 == features_v2  # Must be deterministic
   ```

2. **No Lookahead Test:**
   ```python
   # Predict for 2023-01-15 using 2023-01-14 data
   prediction = predict('AAPL', date='2023-01-14')
   # Validate: prediction should use ONLY data <= 2023-01-14
   assert max(prediction.metadata['data_dates']) <= datetime(2023, 1, 14)
   ```

---

## 3. Performance Metrics

### 3.1 Prediction Accuracy Metrics

**3.1.1 RMSE (Root Mean Squared Error)**

**Formula:**
```
RMSE = sqrt(mean((y_pred - y_actual)^2))
```

**Interpretation:**
- Measures average prediction error magnitude
- Penalizes large errors more (squared term)
- Units: Same as target (dollars or % for normalized)

**Benchmark:**
- 1-day: <5% of current price (excellent), <8% (acceptable)
- 30-day: <10% of current price (excellent), <15% (acceptable)

**Example:**
```
Stock: AAPL @ $150
Prediction: $153 (1-day ahead)
Actual: $151
Error: $2
RMSE (single): $2 = 1.3% ✅ Excellent
```

**Implementation:**
```python
def calculate_rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate RMSE for predictions

    Args:
        predictions: Predicted prices [N]
        actuals: Actual prices [N]

    Returns:
        rmse: Root mean squared error
    """
    return np.sqrt(np.mean((predictions - actuals) ** 2))
```

---

**3.1.2 MAE (Mean Absolute Error)**

**Formula:**
```
MAE = mean(abs(y_pred - y_actual))
```

**Interpretation:**
- Average absolute error (more robust than RMSE to outliers)
- Easier to interpret (units match target)

**Benchmark:**
- 1-day: <3% (excellent), <5% (acceptable)
- 30-day: <7% (excellent), <10% (acceptable)

**Implementation:**
```python
def calculate_mae(predictions: np.ndarray, actuals: np.ndarray) -> float:
    return np.mean(np.abs(predictions - actuals))
```

---

**3.1.3 MAPE (Mean Absolute Percentage Error)**

**Formula:**
```
MAPE = mean(abs((y_pred - y_actual) / y_actual)) * 100
```

**Interpretation:**
- Percentage error (normalized across price levels)
- Good for comparing stocks at different price ranges

**Benchmark:**
- 1-day: <5% (excellent), <8% (acceptable)
- 30-day: <10% (excellent), <15% (acceptable)

**Caveat:** Undefined if `y_actual = 0` (not an issue for stock prices)

**Implementation:**
```python
def calculate_mape(predictions: np.ndarray, actuals: np.ndarray) -> float:
    return np.mean(np.abs((predictions - actuals) / (actuals + 1e-8))) * 100
```

---

### 3.2 Directional Accuracy Metrics

**3.2.1 Directional Accuracy (%)**

**Formula:**
```
Directional_Accuracy = sum(sign(pred - current) == sign(actual - current)) / N * 100
```

**Interpretation:**
- Did we correctly predict UP vs DOWN?
- More important than absolute price for trading
- Baseline: 50% (random coin flip)

**Benchmark:**
- >55%: Profitable (with transaction costs)
- >60%: Excellent
- >65%: Exceptional (institutional target)

**Example:**
```
Current: $150
Prediction: $152 (UP)
Actual: $151 (UP)
Correct ✅

Current: $150
Prediction: $148 (DOWN)
Actual: $151 (UP)
Wrong ❌
```

**Implementation:**
```python
def calculate_directional_accuracy(
    predictions: np.ndarray,
    actuals: np.ndarray,
    current_prices: np.ndarray
) -> float:
    """
    Calculate % of correct directional predictions

    Args:
        predictions: Predicted prices [N]
        actuals: Actual prices [N]
        current_prices: Prices at prediction time [N]

    Returns:
        accuracy: Percentage (0-100)
    """
    pred_directions = np.sign(predictions - current_prices)
    actual_directions = np.sign(actuals - current_prices)
    correct = pred_directions == actual_directions
    return np.mean(correct) * 100
```

---

**3.2.2 Precision & Recall for BUY/SELL Signals**

**Classification Setup:**
- BUY signal: Prediction is >2% above current (significant upside)
- SELL signal: Prediction is <-2% below current (significant downside)
- HOLD signal: Prediction within [-2%, +2%]

**Formulas:**
```
Precision_BUY = True_Positives_BUY / (True_Positives_BUY + False_Positives_BUY)
Recall_BUY = True_Positives_BUY / (True_Positives_BUY + False_Negatives_BUY)
F1_BUY = 2 * (Precision * Recall) / (Precision + Recall)
```

**Interpretation:**
- Precision: When we say BUY, how often is it correct?
- Recall: Of all true BUY opportunities, how many did we catch?
- F1: Harmonic mean (balanced metric)

**Benchmark:**
- Precision >60%: Good (avoid false signals)
- Recall >50%: Acceptable (don't miss opportunities)
- F1 >0.55: Excellent balance

**Implementation:**
```python
def calculate_signal_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    current_prices: np.ndarray,
    threshold: float = 0.02  # 2%
) -> Dict[str, float]:
    """Calculate precision/recall for BUY/SELL signals"""

    # Generate signals
    pred_returns = (predictions - current_prices) / current_prices
    actual_returns = (actuals - current_prices) / current_prices

    pred_signals = np.where(pred_returns > threshold, 1,  # BUY
                   np.where(pred_returns < -threshold, -1,  # SELL
                   0))  # HOLD

    actual_signals = np.where(actual_returns > threshold, 1,
                     np.where(actual_returns < -threshold, -1,
                     0))

    # Precision/Recall for BUY
    buy_mask = pred_signals == 1
    if buy_mask.sum() > 0:
        precision_buy = (pred_signals[buy_mask] == actual_signals[buy_mask]).mean()
    else:
        precision_buy = 0.0

    true_buy_mask = actual_signals == 1
    if true_buy_mask.sum() > 0:
        recall_buy = (pred_signals[true_buy_mask] == 1).mean()
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
        'f1_buy': f1_buy,
        # Similar for SELL...
    }
```

---

### 3.3 Risk-Adjusted Return Metrics

**3.3.1 Sharpe Ratio**

**Formula:**
```
Sharpe = (mean(returns) - risk_free_rate) / std(returns) * sqrt(252)
```

**Interpretation:**
- Risk-adjusted return (return per unit volatility)
- Higher is better
- Annualized (252 trading days)

**Benchmark:**
- <0.5: Poor
- 0.5-1.0: Acceptable
- 1.0-2.0: Good
- >2.0: Excellent (institutional target)

**Implementation:**
```python
def calculate_sharpe_ratio(
    predictions: np.ndarray,
    actuals: np.ndarray,
    current_prices: np.ndarray,
    risk_free_rate: float = 0.04  # 4% annual
) -> float:
    """
    Calculate annualized Sharpe ratio

    Assumes trading strategy:
      - Go long if prediction > current
      - Go short if prediction < current

    Returns:
        sharpe: Annualized Sharpe ratio
    """
    # Calculate returns if we followed predictions
    pred_directions = np.sign(predictions - current_prices)
    actual_returns = (actuals - current_prices) / current_prices

    # Strategy returns: directional bet × actual return
    strategy_returns = pred_directions * actual_returns

    # Annualized Sharpe
    mean_return = np.mean(strategy_returns) * 252  # Annualize
    std_return = np.std(strategy_returns) * np.sqrt(252)  # Annualize

    sharpe = (mean_return - risk_free_rate) / (std_return + 1e-8)

    return sharpe
```

---

**3.3.2 Sortino Ratio**

**Formula:**
```
Sortino = (mean(returns) - risk_free_rate) / std(downside_returns) * sqrt(252)
```

**Difference from Sharpe:**
- Only penalizes downside volatility (below 0%)
- Upside volatility is good (ignored)
- Better for asymmetric strategies

**Benchmark:**
- >1.5: Good
- >3.0: Excellent

**Implementation:**
```python
def calculate_sortino_ratio(
    predictions: np.ndarray,
    actuals: np.ndarray,
    current_prices: np.ndarray,
    risk_free_rate: float = 0.04
) -> float:
    """Calculate annualized Sortino ratio (downside risk only)"""

    # Strategy returns
    pred_directions = np.sign(predictions - current_prices)
    actual_returns = (actuals - current_prices) / current_prices
    strategy_returns = pred_directions * actual_returns

    # Downside returns (negative only)
    downside_returns = strategy_returns[strategy_returns < 0]

    if len(downside_returns) == 0:
        return float('inf')  # No downside!

    mean_return = np.mean(strategy_returns) * 252
    downside_std = np.std(downside_returns) * np.sqrt(252)

    sortino = (mean_return - risk_free_rate) / (downside_std + 1e-8)

    return sortino
```

---

**3.3.3 Information Ratio (vs SPY Benchmark)**

**Formula:**
```
IR = (mean(strategy_returns - benchmark_returns)) / std(strategy_returns - benchmark_returns)
```

**Interpretation:**
- Excess return vs benchmark per unit tracking error
- Measures skill vs just market beta

**Benchmark:**
- >0.5: Good
- >1.0: Excellent (outperforming market with consistency)

**Implementation:**
```python
def calculate_information_ratio(
    predictions: np.ndarray,
    actuals: np.ndarray,
    current_prices: np.ndarray,
    benchmark_returns: np.ndarray  # SPY returns
) -> float:
    """Calculate Information Ratio vs benchmark (e.g., SPY)"""

    # Strategy returns
    pred_directions = np.sign(predictions - current_prices)
    actual_returns = (actuals - current_prices) / current_prices
    strategy_returns = pred_directions * actual_returns

    # Excess returns
    excess_returns = strategy_returns - benchmark_returns

    # IR
    ir = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)

    return ir * np.sqrt(252)  # Annualize
```

---

### 3.4 Drawdown Analysis

**3.4.1 Maximum Drawdown**

**Formula:**
```
Drawdown_i = (Peak_price - Current_price) / Peak_price
MaxDrawdown = max(Drawdown_i)
```

**Interpretation:**
- Largest peak-to-trough decline
- Measures worst-case loss
- Critical for risk management

**Benchmark:**
- <10%: Excellent
- <20%: Acceptable
- >30%: High risk

**Implementation:**
```python
def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown and recovery time

    Args:
        equity_curve: Cumulative portfolio value over time

    Returns:
        (max_drawdown, peak_idx, trough_idx)
    """
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)

    # Drawdown at each point
    drawdowns = (running_max - equity_curve) / running_max

    # Maximum drawdown
    max_dd = np.max(drawdowns)
    trough_idx = np.argmax(drawdowns)

    # Find peak before trough
    peak_idx = np.argmax(equity_curve[:trough_idx+1])

    return max_dd, peak_idx, trough_idx
```

---

**3.4.2 Recovery Time**

**Definition:** Number of days from trough to recovery (return to peak)

**Benchmark:**
- <30 days: Fast recovery
- <90 days: Acceptable
- >180 days: Slow recovery (concern)

**Implementation:**
```python
def calculate_recovery_time(
    equity_curve: np.ndarray,
    peak_idx: int,
    trough_idx: int
) -> int:
    """Days from trough to recovery"""

    peak_value = equity_curve[peak_idx]

    # Find first day after trough that exceeds peak
    recovery_idx = None
    for i in range(trough_idx + 1, len(equity_curve)):
        if equity_curve[i] >= peak_value:
            recovery_idx = i
            break

    if recovery_idx is None:
        return len(equity_curve) - trough_idx  # Not recovered yet

    return recovery_idx - trough_idx
```

---

**3.4.3 Calmar Ratio**

**Formula:**
```
Calmar = Annual_Return / Max_Drawdown
```

**Interpretation:**
- Return per unit maximum drawdown
- Higher is better
- Balances return and worst-case risk

**Benchmark:**
- >3.0: Excellent
- >1.5: Good
- <1.0: Poor (not worth the risk)

**Implementation:**
```python
def calculate_calmar_ratio(
    annual_return: float,
    max_drawdown: float
) -> float:
    """Calmar ratio = return / max drawdown"""
    return annual_return / (max_drawdown + 1e-8)
```

---

### 3.5 Model-Specific Metrics

**3.5.1 GNN: Correlation Strength & Network Centrality**

**Metrics:**
1. **Average Correlation:**
   - Mean absolute correlation between target stock and neighbors
   - Higher = stronger network effects = better GNN performance

2. **Network Degree Centrality:**
   - Number of edges (correlations >0.3 threshold)
   - Higher = more connected = GNN has more information

**Benchmark:**
- Avg correlation >0.5: Strong network (GNN should excel)
- Avg correlation <0.3: Weak network (GNN may struggle)

**Implementation:**
```python
def calculate_gnn_specific_metrics(
    symbol: str,
    correlation_matrix: np.ndarray,
    symbols: List[str]
) -> Dict[str, float]:
    """Calculate GNN-specific performance indicators"""

    symbol_idx = symbols.index(symbol)
    correlations = np.abs(correlation_matrix[symbol_idx])

    # Remove self-correlation
    correlations = correlations[correlations < 1.0]

    avg_correlation = np.mean(correlations)

    # Degree centrality (# of strong correlations)
    degree = (correlations > 0.3).sum()

    return {
        'avg_correlation': avg_correlation,
        'network_degree': degree
    }
```

---

**3.5.2 PINN: Greeks Accuracy (if options data available)**

**Metrics:**
1. **Delta Accuracy:** `|predicted_delta - actual_delta|`
2. **Gamma Accuracy:** `|predicted_gamma - actual_gamma|`
3. **Implied Vol Error:** `|predicted_IV - actual_IV|`

**Benchmark:**
- Delta error <0.05: Excellent
- IV error <0.03 (3%): Excellent

**Note:** Requires options data (not available in yfinance free tier)
- Backtest without Greeks accuracy if data unavailable
- Future: Integrate Polygon/Intrinio for options data

---

**3.5.3 Mamba: Horizon-Specific Performance**

**Metrics:**
- RMSE for 1d, 5d, 10d, 30d predictions separately
- Identify Mamba's optimal horizon

**Expected:**
- Mamba excels at 30-day predictions (long sequences)
- May underperform GNN at 1-day (GNN uses real-time correlations)

**Implementation:**
```python
def calculate_mamba_horizon_metrics(
    mamba_predictions: Dict[str, np.ndarray],  # {'1d': [...], '5d': [...], ...}
    actuals: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """Calculate Mamba performance by horizon"""

    metrics = {}
    for horizon in ['1d', '5d', '10d', '30d']:
        preds = mamba_predictions[horizon]
        acts = actuals[horizon]

        rmse = calculate_rmse(preds, acts)
        dir_acc = calculate_directional_accuracy(preds, acts, current_prices)

        metrics[f'rmse_{horizon}'] = rmse
        metrics[f'dir_acc_{horizon}'] = dir_acc

    return metrics
```

---

**3.5.4 Epidemic: Regime Classification Accuracy**

**Metrics:**
1. **Regime Prediction Accuracy:**
   - Predict market regime (calm, pre-volatile, volatile, stabilized)
   - Validate against actual VIX levels

2. **VIX Forecast Error:**
   - RMSE for VIX predictions (if available)

**Benchmark:**
- Regime accuracy >70%: Good (better than naive persistence)
- VIX RMSE <3 points: Excellent

**VIX Regime Thresholds:**
- VIX <15: Calm
- VIX 15-20: Pre-volatile
- VIX 20-30: Volatile
- VIX >30: Extreme volatility

---

**3.5.5 Ensemble: Improvement Over Best Individual**

**Metric:**
```
Ensemble_Improvement = (Ensemble_Sharpe - Best_Individual_Sharpe) / Best_Individual_Sharpe * 100
```

**Benchmark:**
- >5%: Ensemble adds value
- >10%: Strong ensemble benefit
- <0%: Ensemble underperforming (investigate weighting)

**Implementation:**
```python
def calculate_ensemble_improvement(
    ensemble_sharpe: float,
    individual_sharpes: Dict[str, float]
) -> float:
    """Calculate % improvement of ensemble over best individual"""

    best_individual = max(individual_sharpes.values())

    improvement = (ensemble_sharpe - best_individual) / best_individual * 100

    return improvement
```

---

### 3.6 Comprehensive Metrics Summary Table

| Category | Metric | Formula | Benchmark | Priority |
|----------|--------|---------|-----------|----------|
| **Accuracy** | RMSE (1d) | `sqrt(mean((pred-actual)^2))` | <5% | High |
| | RMSE (30d) | | <10% | Medium |
| | MAE | `mean(abs(pred-actual))` | <3% (1d) | High |
| | MAPE | `mean(abs((pred-actual)/actual))` | <5% (1d) | Medium |
| **Directional** | Dir. Accuracy | `mean(sign(pred-cur) == sign(act-cur))` | >55% | **Critical** |
| | Precision (BUY) | `TP / (TP + FP)` | >60% | High |
| | Recall (BUY) | `TP / (TP + FN)` | >50% | Medium |
| | F1 (BUY) | `2 * P * R / (P + R)` | >0.55 | High |
| **Risk-Adj** | Sharpe Ratio | `(ret - rf) / std * √252` | >1.0 | **Critical** |
| | Sortino Ratio | `(ret - rf) / downside_std * √252` | >1.5 | High |
| | Info Ratio | `excess_ret / tracking_error` | >0.5 | Medium |
| **Drawdown** | Max Drawdown | `max((peak - trough) / peak)` | <20% | **Critical** |
| | Recovery Time | Days from trough to recovery | <90d | Medium |
| | Calmar Ratio | `annual_ret / max_dd` | >1.5 | High |
| **GNN** | Avg Correlation | `mean(abs(corr_matrix))` | >0.5 | Low |
| | Network Degree | `sum(corr > 0.3)` | >5 | Low |
| **Mamba** | RMSE by horizon | Per 1d/5d/10d/30d | Varies | Medium |
| **Ensemble** | Improvement | `(ens - best) / best * 100` | >5% | High |

**Priority Legend:**
- **Critical:** Must meet benchmark for production deployment
- **High:** Important for model selection and weighting
- **Medium:** Useful for diagnostics and optimization
- **Low:** Informational only

---

## 4. Data Pipeline Architecture

### 4.1 Historical Data Fetching Strategy

**Objective:** Minimize yfinance API calls while ensuring data freshness and completeness.

**Approach:** 3-Tier Caching Strategy

**Tier 1: Persistent Filesystem Cache**
- Location: `data/backtest_cache/{symbol}_{start_date}_{end_date}.parquet`
- Format: Parquet (fast, compressed)
- TTL: 7 days (refresh weekly)
- Size estimate: ~50 KB per symbol-period

**Tier 2: In-Memory Cache (LRU)**
- Cache recent price fetches during backtest run
- Max size: 100 MB (avoids repeated disk reads)
- Eviction: LRU when cache full

**Tier 3: Circuit Breaker + Rate Limiting**
- Reuse existing `CircuitBreaker` from `ml_integration_helpers.py`
- Rate limit: 100 requests/hour (conservative)
- Exponential backoff: 2s, 4s, 8s on retries

**Data Fetching Workflow:**

```
1. Check Tier 1 (Filesystem):
   - If cached & fresh (<7 days old): Load from disk
   - Else: Proceed to Tier 3

2. Check Tier 2 (Memory):
   - If recently fetched this session: Return from memory
   - Else: Proceed to Tier 3

3. Fetch from yfinance (Tier 3):
   - Respect circuit breaker (open = skip, use fallback)
   - Batch fetch (multiple symbols in single request if possible)
   - Rate limit: Sleep if approaching 100/hour
   - On success: Save to Tier 1 + Tier 2
   - On failure: Circuit breaker records failure, retry with backoff

4. Fallback (if circuit breaker open):
   - Load most recent cached data (even if stale >7 days)
   - Log warning for monitoring
```

**Implementation:**

```python
# In src/backtesting/historical_data_loader.py

import os
import time
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, Optional

CACHE_DIR = Path("data/backtest_cache")
CACHE_TTL_DAYS = 7

class HistoricalDataLoader:
    """
    3-Tier historical data loader for backtesting

    Tier 1: Persistent filesystem cache (Parquet)
    Tier 2: In-memory LRU cache
    Tier 3: yfinance API with circuit breaker
    """

    def __init__(self, cache_ttl_days: int = 7):
        self.cache_ttl_days = cache_ttl_days
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting
        self.request_times = []  # Track API calls
        self.max_requests_per_hour = 100

    def _get_cache_path(self, symbol: str, start_date: str, end_date: str) -> Path:
        """Generate cache file path"""
        filename = f"{symbol}_{start_date}_{end_date}.parquet"
        return self.cache_dir / filename

    def _is_cache_fresh(self, cache_path: Path) -> bool:
        """Check if cache is within TTL"""
        if not cache_path.exists():
            return False

        mtime = cache_path.stat().st_mtime
        age_days = (time.time() - mtime) / 86400

        return age_days < self.cache_ttl_days

    def _load_from_cache(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load from Tier 1 (filesystem cache)"""
        cache_path = self._get_cache_path(symbol, start_date, end_date)

        if self._is_cache_fresh(cache_path):
            logger.info(f"[Cache HIT] Loading {symbol} from cache")
            return pd.read_parquet(cache_path)

        logger.info(f"[Cache MISS] {symbol} cache stale or missing")
        return None

    def _save_to_cache(self, symbol: str, start_date: str, end_date: str, data: pd.DataFrame):
        """Save to Tier 1 (filesystem cache)"""
        cache_path = self._get_cache_path(symbol, start_date, end_date)
        data.to_parquet(cache_path, compression='snappy')
        logger.info(f"[Cache SAVE] Saved {symbol} to cache")

    def _check_rate_limit(self):
        """Enforce rate limiting (100 requests/hour)"""
        now = time.time()

        # Remove requests older than 1 hour
        self.request_times = [t for t in self.request_times if now - t < 3600]

        if len(self.request_times) >= self.max_requests_per_hour:
            # Calculate sleep time until oldest request expires
            oldest = min(self.request_times)
            sleep_time = 3600 - (now - oldest) + 1
            logger.warning(f"[Rate Limit] Sleeping {sleep_time:.0f}s to respect rate limit")
            time.sleep(sleep_time)

        # Record this request
        self.request_times.append(now)

    async def fetch_historical_data(
        self,
        symbol: str,
        start_date: str,  # 'YYYY-MM-DD'
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch historical data with 3-tier caching

        Returns:
            DataFrame with columns: [Date, Open, High, Low, Close, Volume]
        """
        # Tier 1: Check filesystem cache
        cached_data = self._load_from_cache(symbol, start_date, end_date)
        if cached_data is not None:
            return cached_data

        # Tier 3: Fetch from yfinance
        from src.api.ml_integration_helpers import _yfinance_circuit_breaker

        if not _yfinance_circuit_breaker.can_execute():
            logger.warning(f"[Circuit Breaker OPEN] Cannot fetch {symbol}, loading stale cache if available")
            # Try to load stale cache as fallback
            cache_path = self._get_cache_path(symbol, start_date, end_date)
            if cache_path.exists():
                logger.warning(f"[Fallback] Using stale cache for {symbol}")
                return pd.read_parquet(cache_path)
            else:
                raise ValueError(f"Circuit breaker open and no cache for {symbol}")

        # Rate limiting
        self._check_rate_limit()

        # Fetch from yfinance
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)

            if data.empty:
                raise ValueError(f"No data returned for {symbol}")

            # Success: Record in circuit breaker
            _yfinance_circuit_breaker.record_success()

            # Save to cache
            self._save_to_cache(symbol, start_date, end_date, data)

            return data

        except Exception as e:
            logger.error(f"[Fetch FAILED] {symbol}: {e}")
            _yfinance_circuit_breaker.record_failure()
            raise

    @lru_cache(maxsize=100)  # Tier 2: In-memory cache
    def fetch_historical_data_cached(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Wrapper with in-memory LRU cache (Tier 2)

        Note: LRU cache is synchronous, so we wrap async function
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.fetch_historical_data(symbol, start_date, end_date)
        )
```

---

### 4.2 Batch Fetching for Parallel Efficiency

**Strategy:** Fetch data for all stocks at backtest start (upfront caching)

**Rationale:**
- Minimizes API calls during backtest loop
- Parallelizable (fetch 5-10 stocks concurrently)
- Circuit breaker protects against rate limits

**Implementation:**

```python
async def prefetch_all_historical_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    parallel: int = 5
) -> Dict[str, pd.DataFrame]:
    """
    Prefetch historical data for all symbols in parallel

    Args:
        symbols: List of stock symbols
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        parallel: Number of concurrent fetches

    Returns:
        Dict mapping symbol to DataFrame
    """
    loader = HistoricalDataLoader()

    results = {}

    # Batch into groups of `parallel`
    for i in range(0, len(symbols), parallel):
        batch = symbols[i:i+parallel]

        # Fetch batch in parallel
        batch_results = await asyncio.gather(
            *[loader.fetch_historical_data(sym, start_date, end_date) for sym in batch],
            return_exceptions=True
        )

        # Handle results/errors
        for sym, data in zip(batch, batch_results):
            if isinstance(data, Exception):
                logger.error(f"[Prefetch] Failed to fetch {sym}: {data}")
            else:
                results[sym] = data

    logger.info(f"[Prefetch] Fetched {len(results)}/{len(symbols)} symbols successfully")

    return results
```

---

### 4.3 Feature Engineering Alignment

**Critical Requirement:** Backtest features MUST match live prediction features exactly.

**Feature Extraction Module:**

```python
# In src/backtesting/feature_extractor.py

async def extract_features_for_backtest(
    symbol: str,
    date: datetime,
    historical_data: pd.DataFrame,
    model_name: str
) -> Dict[str, Any]:
    """
    Extract features for a specific model at a specific date

    GUARANTEES: No lookahead bias (only uses data <= date)

    Args:
        symbol: Stock symbol
        date: Current date (t_train)
        historical_data: Full historical DataFrame
        model_name: 'gnn', 'mamba', 'pinn', 'epidemic', 'ensemble'

    Returns:
        features: Dict with model-specific features
    """
    # Filter data up to date (no lookahead)
    data_until_date = historical_data[historical_data.index <= date]

    if len(data_until_date) == 0:
        raise ValueError(f"No data available for {symbol} up to {date}")

    # Extract features based on model
    if model_name == 'gnn':
        # GNN needs 20-day volatility, momentum, 60-day correlation
        lookback_20d = data_until_date['Close'].iloc[-20:]
        lookback_60d = data_until_date['Close'].iloc[-60:]

        if len(lookback_20d) < 20:
            raise ValueError(f"Insufficient data for GNN (need 20 days, have {len(lookback_20d)})")

        returns_20d = lookback_20d.pct_change().dropna()

        features = {
            'prices_20d': lookback_20d.values,
            'prices_60d': lookback_60d.values,
            'volatility': float(returns_20d.std()),
            'momentum': float(returns_20d.mean()),
            'volume_norm': 1.0,  # Placeholder
        }

    elif model_name == 'mamba':
        # Mamba needs 1000-day price history
        lookback_1000d = data_until_date['Close'].iloc[-1000:]

        if len(lookback_1000d) < 60:  # Minimum for Mamba
            raise ValueError(f"Insufficient data for Mamba (need 60 days minimum, have {len(lookback_1000d)})")

        features = {
            'prices': lookback_1000d.values,
            'sequence_length': len(lookback_1000d),
        }

    elif model_name == 'pinn':
        # PINN needs IV, risk-free rate, current price
        lookback_30d = data_until_date['Close'].iloc[-30:]

        if len(lookback_30d) < 30:
            raise ValueError(f"Insufficient data for PINN")

        returns_30d = lookback_30d.pct_change().dropna()
        historical_vol = float(returns_30d.std() * np.sqrt(252))  # Annualize

        features = {
            'current_price': float(data_until_date['Close'].iloc[-1]),
            'implied_volatility': historical_vol,  # Proxy (no options data)
            'risk_free_rate': 0.04,  # Fixed (or fetch 10-year treasury)
        }

    elif model_name == 'epidemic':
        # Epidemic needs VIX, realized vol, sentiment (optional)
        lookback_30d = data_until_date['Close'].iloc[-30:]

        returns_30d = lookback_30d.pct_change().dropna()
        realized_vol = float(returns_30d.std() * np.sqrt(252))

        features = {
            'realized_vol': realized_vol,
            'vix': realized_vol * 100,  # Proxy (no VIX data in yfinance)
            'sentiment': 0.0,  # Neutral (no sentiment data)
            'volume': float(data_until_date['Volume'].iloc[-1]) if 'Volume' in data_until_date else 0.0,
        }

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return features
```

---

### 4.4 Data Validation & Quality Checks

**Validation Checks:**

1. **Missing Data Detection:**
   ```python
   def detect_missing_days(data: pd.DataFrame) -> List[str]:
       """Detect gaps in trading days"""
       date_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='B')  # Business days
       missing = date_index.difference(data.index)
       return [str(d.date()) for d in missing]
   ```

2. **Outlier Detection (Flash Crashes):**
   ```python
   def detect_outliers(data: pd.DataFrame, threshold: float = 5.0) -> List[str]:
       """Detect days with >5 std dev returns (outliers)"""
       returns = data['Close'].pct_change().dropna()
       z_scores = (returns - returns.mean()) / returns.std()
       outliers = data.index[np.abs(z_scores) > threshold]
       return [str(d.date()) for d in outliers]
   ```

3. **Duplicate Detection:**
   ```python
   def detect_duplicates(data: pd.DataFrame) -> List[str]:
       """Detect duplicate dates"""
       duplicates = data.index[data.index.duplicated()]
       return [str(d.date()) for d in duplicates]
   ```

**Handling Strategy:**

| Issue | Strategy |
|-------|----------|
| Missing days (gaps) | Forward-fill (use previous day's close) |
| Outliers (flash crashes) | Log warning, keep data (real events) |
| Duplicates | Keep first occurrence, drop rest |
| Insufficient data | Skip symbol or shorten backtest period |

---

### 4.5 Data Pipeline Flowchart

```
┌─────────────────────────────────────────────────────────────┐
│ BACKTEST START: User specifies symbols, start_date, end_date │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────┐
         │ Prefetch All Historical Data       │
         │ (Parallel: 5 workers)              │
         │ - Check Tier 1 cache (filesystem)  │
         │ - Fetch from yfinance if needed    │
         │ - Save to cache                    │
         └────────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────┐
         │ Validate Data Quality              │
         │ - Detect missing days              │
         │ - Detect outliers                  │
         │ - Handle gaps (forward-fill)       │
         └────────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────┐
         │ Walk-Forward Loop (Each Symbol)    │
         │ FOR date in [start_date + 1000d    │
         │              to end_date]:         │
         │   1. Extract features up to date   │
         │   2. Predict for date + horizon    │
         │   3. Record prediction             │
         │   4. Validate against actual       │
         └────────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────┐
         │ Calculate Metrics                  │
         │ - RMSE, MAE, MAPE                  │
         │ - Directional accuracy             │
         │ - Sharpe, Sortino, Info ratio      │
         │ - Max drawdown, recovery time      │
         └────────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────┐
         │ Generate Report                    │
         │ - Performance summary table        │
         │ - Model rankings                   │
         │ - Visualizations (optional)        │
         └────────────────────────────────────┘
```

---

## 5. Implementation Plan

### 5.1 Module Structure

**Files to Create:**

1. **`src/backtesting/backtest_engine.py`** (~400 lines)
   - Core walk-forward backtesting logic
   - Model prediction orchestration
   - Results aggregation

2. **`src/backtesting/metrics.py`** (~200 lines)
   - All performance metric calculations
   - Grading/ranking logic

3. **`src/backtesting/historical_data_loader.py`** (~250 lines)
   - 3-tier caching system
   - Data fetching + validation

4. **`src/backtesting/feature_extractor.py`** (~150 lines)
   - Model-specific feature extraction
   - No-lookahead validation

5. **`src/backtesting/visualizer.py`** (~300 lines)
   - Matplotlib charts (cumulative returns, drawdown, etc.)
   - Markdown report generation

6. **`scripts/run_backtest.py`** (~150 lines)
   - CLI interface
   - Argument parsing

7. **`tests/test_backtesting.py`** (~250 lines)
   - Unit tests for metrics
   - Integration tests for backtest engine
   - End-to-end test (1 stock, 1 year)

**Total: ~1,700 lines**

---

### 5.2 Detailed Pseudocode

**5.2.1 backtest_engine.py**

```python
# src/backtesting/backtest_engine.py

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class BacktestResult:
    """Results for a single model on a single symbol"""
    symbol: str
    model_name: str
    predictions: np.ndarray  # [N]
    actuals: np.ndarray  # [N]
    dates: List[datetime]
    current_prices: np.ndarray  # [N] (price at prediction time)
    metrics: Dict[str, float]  # Calculated metrics

class BacktestEngine:
    """
    Walk-forward backtesting engine for ML models

    Features:
    - Expanding window (no lookahead bias)
    - Parallel stock processing
    - Model-specific feature extraction
    """

    def __init__(
        self,
        start_date: str,  # 'YYYY-MM-DD'
        end_date: str,
        horizon_days: int = 1,  # 1-day or 30-day predictions
        min_train_days: int = 1000  # Minimum history before predictions start
    ):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.horizon_days = horizon_days
        self.min_train_days = min_train_days

    async def backtest_model(
        self,
        symbol: str,
        model_name: str,
        historical_data: pd.DataFrame
    ) -> BacktestResult:
        """
        Backtest a single model on a single symbol

        Args:
            symbol: Stock symbol
            model_name: 'gnn', 'mamba', 'pinn', 'epidemic'
            historical_data: Full historical DataFrame

        Returns:
            BacktestResult with predictions and metrics
        """
        from .feature_extractor import extract_features_for_backtest
        from src.api.ml_integration_helpers import (
            get_gnn_prediction,
            get_mamba_prediction,
            get_pinn_prediction
        )

        predictions = []
        actuals = []
        dates = []
        current_prices = []

        # Walk-forward loop
        prediction_start = self.start_date + timedelta(days=self.min_train_days)

        current_date = prediction_start
        while current_date <= self.end_date:
            # Skip weekends
            if current_date.weekday() >= 5:  # Sat/Sun
                current_date += timedelta(days=1)
                continue

            # Extract features up to current_date (no lookahead)
            try:
                features = await extract_features_for_backtest(
                    symbol, current_date, historical_data, model_name
                )
            except ValueError as e:
                # Insufficient data, skip this date
                logger.warning(f"[{symbol}] Skipping {current_date}: {e}")
                current_date += timedelta(days=1)
                continue

            # Get current price
            current_price = features.get('current_price') or \
                           historical_data.loc[current_date, 'Close']

            # Make prediction
            if model_name == 'gnn':
                result = await get_gnn_prediction(symbol, current_price)
                predicted_price = result['prediction']

            elif model_name == 'mamba':
                result = await get_mamba_prediction(symbol, current_price)
                # Use horizon-specific prediction
                if self.horizon_days == 1:
                    predicted_price = result['multi_horizon']['1d']
                elif self.horizon_days == 30:
                    predicted_price = result['multi_horizon']['30d']
                else:
                    predicted_price = result['prediction']  # Default

            elif model_name == 'pinn':
                result = await get_pinn_prediction(symbol, current_price)
                predicted_price = result['prediction']

            # Similar for epidemic, ensemble...

            # Get actual price at horizon
            future_date = current_date + timedelta(days=self.horizon_days)

            # Find actual price (skip weekends)
            actual_price = None
            for offset in range(self.horizon_days + 5):  # Lookahead up to 5 extra days
                check_date = current_date + timedelta(days=offset)
                if check_date in historical_data.index:
                    actual_price = historical_data.loc[check_date, 'Close']
                    break

            if actual_price is None:
                # No data for validation (end of dataset)
                break

            # Record prediction
            predictions.append(predicted_price)
            actuals.append(actual_price)
            dates.append(current_date)
            current_prices.append(current_price)

            # Move to next day
            current_date += timedelta(days=1)

        # Convert to numpy
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        current_prices = np.array(current_prices)

        # Calculate metrics
        from .metrics import calculate_all_metrics
        metrics = calculate_all_metrics(
            predictions, actuals, current_prices, symbol, model_name
        )

        return BacktestResult(
            symbol=symbol,
            model_name=model_name,
            predictions=predictions,
            actuals=actuals,
            dates=dates,
            current_prices=current_prices,
            metrics=metrics
        )

    async def backtest_all_models(
        self,
        symbol: str,
        historical_data: pd.DataFrame,
        models: List[str] = ['gnn', 'mamba', 'pinn', 'epidemic']
    ) -> List[BacktestResult]:
        """
        Backtest all models on a single symbol

        Returns:
            List of BacktestResult (one per model)
        """
        results = []

        for model_name in models:
            logger.info(f"[{symbol}] Backtesting {model_name}...")
            result = await self.backtest_model(symbol, model_name, historical_data)
            results.append(result)

        return results

    async def backtest_all_symbols(
        self,
        symbols: List[str],
        models: List[str] = ['gnn', 'mamba', 'pinn', 'epidemic'],
        parallel: int = 5
    ) -> Dict[str, List[BacktestResult]]:
        """
        Backtest all models on all symbols (parallel)

        Args:
            symbols: List of stock symbols
            models: List of model names
            parallel: Number of symbols to process in parallel

        Returns:
            Dict mapping symbol to list of BacktestResult
        """
        from .historical_data_loader import HistoricalDataLoader

        # Prefetch all data
        loader = HistoricalDataLoader()
        historical_data = {}

        logger.info(f"Prefetching data for {len(symbols)} symbols...")
        for i in range(0, len(symbols), parallel):
            batch = symbols[i:i+parallel]
            batch_data = await asyncio.gather(
                *[loader.fetch_historical_data(
                    sym, self.start_date.strftime('%Y-%m-%d'), self.end_date.strftime('%Y-%m-%d')
                ) for sym in batch],
                return_exceptions=True
            )

            for sym, data in zip(batch, batch_data):
                if isinstance(data, Exception):
                    logger.error(f"Failed to fetch {sym}: {data}")
                else:
                    historical_data[sym] = data

        # Backtest each symbol
        all_results = {}

        for symbol in symbols:
            if symbol not in historical_data:
                logger.warning(f"Skipping {symbol} (no data)")
                continue

            logger.info(f"Backtesting {symbol} ({len(models)} models)...")
            results = await self.backtest_all_models(symbol, historical_data[symbol], models)
            all_results[symbol] = results

        return all_results
```

---

**5.2.2 metrics.py**

```python
# src/backtesting/metrics.py

import numpy as np
from typing import Dict, Tuple

def calculate_all_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    current_prices: np.ndarray,
    symbol: str,
    model_name: str
) -> Dict[str, float]:
    """
    Calculate all performance metrics for a backtest

    Returns:
        metrics: Dict with all calculated metrics
    """
    metrics = {}

    # Prediction accuracy
    metrics['rmse'] = calculate_rmse(predictions, actuals)
    metrics['mae'] = calculate_mae(predictions, actuals)
    metrics['mape'] = calculate_mape(predictions, actuals)

    # Directional accuracy
    metrics['directional_accuracy'] = calculate_directional_accuracy(
        predictions, actuals, current_prices
    )

    # Signal metrics
    signal_metrics = calculate_signal_metrics(
        predictions, actuals, current_prices, threshold=0.02
    )
    metrics.update(signal_metrics)

    # Risk-adjusted returns
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(
        predictions, actuals, current_prices, risk_free_rate=0.04
    )
    metrics['sortino_ratio'] = calculate_sortino_ratio(
        predictions, actuals, current_prices, risk_free_rate=0.04
    )

    # Drawdown analysis
    equity_curve = calculate_equity_curve(predictions, actuals, current_prices)
    max_dd, peak_idx, trough_idx = calculate_max_drawdown(equity_curve)
    recovery_time = calculate_recovery_time(equity_curve, peak_idx, trough_idx)

    metrics['max_drawdown'] = max_dd
    metrics['recovery_time_days'] = recovery_time

    # Calmar ratio
    annual_return = (equity_curve[-1] / equity_curve[0]) ** (252 / len(equity_curve)) - 1
    metrics['annual_return'] = annual_return
    metrics['calmar_ratio'] = calculate_calmar_ratio(annual_return, max_dd)

    return metrics

# (Individual metric functions as defined in Section 3)
# calculate_rmse, calculate_mae, calculate_mape, etc.
```

---

**5.2.3 historical_data_loader.py**

(Already detailed in Section 4.1)

---

**5.2.4 feature_extractor.py**

(Already detailed in Section 4.3)

---

**5.2.5 visualizer.py**

```python
# src/backtesting/visualizer.py

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from .backtest_engine import BacktestResult

def plot_cumulative_returns(results: List[BacktestResult], output_path: str = 'cumulative_returns.png'):
    """
    Plot cumulative returns for all models

    Args:
        results: List of BacktestResult (one per model)
        output_path: Where to save plot
    """
    plt.figure(figsize=(12, 6))

    for result in results:
        # Calculate equity curve
        equity_curve = calculate_equity_curve(
            result.predictions, result.actuals, result.current_prices
        )

        # Normalize to start at 1.0
        equity_curve = equity_curve / equity_curve[0]

        plt.plot(result.dates, equity_curve, label=result.model_name)

    plt.title(f'Cumulative Returns: {results[0].symbol}')
    plt.xlabel('Date')
    plt.ylabel('Equity (Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_drawdown_chart(result: BacktestResult, output_path: str):
    """Plot drawdown over time"""
    equity_curve = calculate_equity_curve(
        result.predictions, result.actuals, result.current_prices
    )

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / running_max * 100

    plt.figure(figsize=(12, 6))
    plt.fill_between(result.dates, drawdowns, 0, alpha=0.3, color='red')
    plt.plot(result.dates, drawdowns, color='red')
    plt.title(f'Drawdown: {result.symbol} ({result.model_name})')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_markdown_report(
    all_results: Dict[str, List[BacktestResult]],
    output_path: str = 'backtest_report.md'
) -> str:
    """
    Generate comprehensive Markdown report

    Args:
        all_results: Dict mapping symbol to list of BacktestResult
        output_path: Where to save report

    Returns:
        report_text: Markdown string
    """
    lines = []

    lines.append("# Backtesting Results Report\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Symbols:** {len(all_results)}\n\n")

    # Summary table
    lines.append("## Performance Summary\n")
    lines.append("| Symbol | Model | RMSE | Dir. Acc | Sharpe | Max DD | Grade |\n")
    lines.append("|--------|-------|------|----------|--------|--------|-------|\n")

    for symbol, results in all_results.items():
        for result in results:
            m = result.metrics
            grade = grade_model(m)

            lines.append(
                f"| {symbol} | {result.model_name} | {m['rmse']:.2f}% | "
                f"{m['directional_accuracy']:.1f}% | {m['sharpe_ratio']:.2f} | "
                f"{m['max_drawdown']*100:.1f}% | {grade} |\n"
            )

    # Model rankings
    lines.append("\n## Model Rankings (by Sharpe Ratio)\n")

    # Aggregate metrics by model
    model_metrics = {}
    for symbol, results in all_results.items():
        for result in results:
            if result.model_name not in model_metrics:
                model_metrics[result.model_name] = []
            model_metrics[result.model_name].append(result.metrics['sharpe_ratio'])

    # Average Sharpe per model
    avg_sharpes = {
        model: np.mean(sharpes)
        for model, sharpes in model_metrics.items()
    }

    ranked_models = sorted(avg_sharpes.items(), key=lambda x: x[1], reverse=True)

    for rank, (model, sharpe) in enumerate(ranked_models, start=1):
        lines.append(f"{rank}. **{model}**: Sharpe = {sharpe:.2f}\n")

    # Write to file
    report_text = ''.join(lines)
    with open(output_path, 'w') as f:
        f.write(report_text)

    return report_text

def grade_model(metrics: Dict[str, float]) -> str:
    """
    Grade model performance (A/B/C/D/F)

    Grading rubric:
    - A: Sharpe >2.0, Dir Acc >60%, Max DD <10%
    - B: Sharpe >1.0, Dir Acc >55%, Max DD <20%
    - C: Sharpe >0.5, Dir Acc >52%, Max DD <30%
    - D: Sharpe >0.0, Dir Acc >50%
    - F: Below D threshold
    """
    sharpe = metrics.get('sharpe_ratio', 0.0)
    dir_acc = metrics.get('directional_accuracy', 50.0)
    max_dd = metrics.get('max_drawdown', 1.0)

    if sharpe > 2.0 and dir_acc > 60 and max_dd < 0.10:
        return 'A'
    elif sharpe > 1.0 and dir_acc > 55 and max_dd < 0.20:
        return 'B'
    elif sharpe > 0.5 and dir_acc > 52 and max_dd < 0.30:
        return 'C'
    elif sharpe > 0.0 and dir_acc > 50:
        return 'D'
    else:
        return 'F'
```

---

**5.2.6 scripts/run_backtest.py**

```python
#!/usr/bin/env python3
"""
ML Model Backtesting CLI

Usage:
    python scripts/run_backtest.py --symbols AAPL,MSFT,GOOGL --years 3 --models gnn,mamba,pinn
    python scripts/run_backtest.py --symbols TIER_1 --years 1 --models all
"""

import argparse
import asyncio
import logging
from datetime import datetime, timedelta

from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.visualizer import generate_markdown_report, plot_cumulative_returns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TIER_1_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    'JPM', 'BAC', 'GS'
]  # 10 stocks for initial validation

ALL_MODELS = ['gnn', 'mamba', 'pinn', 'epidemic', 'ensemble']

async def main(args):
    # Parse symbols
    if args.symbols == 'TIER_1':
        symbols = TIER_1_STOCKS
    else:
        symbols = args.symbols.split(',')

    # Parse models
    if args.models == 'all':
        models = ALL_MODELS
    else:
        models = args.models.split(',')

    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * args.years)

    logger.info(f"Backtesting {len(symbols)} symbols, {len(models)} models")
    logger.info(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Initialize engine
    engine = BacktestEngine(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        horizon_days=args.horizon
    )

    # Run backtest
    all_results = await engine.backtest_all_symbols(symbols, models, parallel=args.parallel)

    # Generate report
    logger.info("Generating report...")
    report = generate_markdown_report(all_results, output_path='backtest_report.md')

    logger.info("✓ Backtest complete! Report: backtest_report.md")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Model Backtesting')
    parser.add_argument('--symbols', default='TIER_1', help='Comma-separated symbols or TIER_1')
    parser.add_argument('--years', type=int, default=3, help='Years of history (1, 3, or 5)')
    parser.add_argument('--models', default='all', help='Comma-separated models or "all"')
    parser.add_argument('--horizon', type=int, default=1, help='Prediction horizon (1 or 30 days)')
    parser.add_argument('--parallel', type=int, default=5, help='Parallel workers')

    args = parser.parse_args()

    asyncio.run(main(args))
```

---

### 5.3 Expected Runtime Analysis

**Per-Symbol Backtest (3 years, 1 model):**

| Step | Time | Notes |
|------|------|-------|
| Data fetch (cached) | 50ms | From filesystem cache |
| Walk-forward loop (750 days) | ~120s | ~160ms per prediction × 750 |
| Metrics calculation | 10ms | Vectorized numpy |
| **Total per symbol-model** | **~2 min** | |

**Full Backtest (10 stocks, 4 models, 3 years):**

- Sequential: 10 stocks × 4 models × 2 min = **80 minutes**
- Parallel (5 workers): 10 stocks / 5 × 4 models × 2 min = **16 minutes** ✅

**Full Backtest (46 stocks, 4 models, 3 years):**

- Sequential: 46 stocks × 4 models × 2 min = **368 minutes** (~6 hours)
- Parallel (10 workers): 46 stocks / 10 × 4 models × 2 min = **~40 minutes** ✅

**Optimization Opportunities:**
- Use pre-trained GNN models (already cached) → Save 5-8s per prediction
- Batch feature extraction (compute once, reuse for all models)
- Parallel model prediction (predict all 4 models concurrently per day)

**Optimized Runtime (with batching):**

- 10 stocks: **~8 minutes** (50% reduction)
- 46 stocks: **~20 minutes** (50% reduction)

---

## 6. Testing Strategy

### 6.1 Test Coverage Plan

**Unit Tests (tests/test_backtesting.py):**

1. **Metrics Calculations:**
   - `test_rmse_calculation()` - RMSE formula correctness
   - `test_mae_calculation()` - MAE formula correctness
   - `test_directional_accuracy()` - Direction prediction correctness
   - `test_sharpe_ratio()` - Sharpe calculation with known inputs
   - `test_sortino_ratio()` - Sortino calculation (downside only)
   - `test_max_drawdown()` - Drawdown on synthetic equity curve
   - `test_recovery_time()` - Recovery time calculation

2. **Data Validation:**
   - `test_detect_missing_days()` - Gap detection
   - `test_detect_outliers()` - Outlier detection (flash crashes)
   - `test_forward_fill_gaps()` - Missing data handling

3. **Feature Extraction:**
   - `test_gnn_features_no_lookahead()` - Assert no future data used
   - `test_mamba_features_sufficient_length()` - Mamba needs 60+ days
   - `test_feature_reproducibility()` - Same date = same features

**Integration Tests:**

4. **Backtest Engine:**
   - `test_backtest_single_model()` - Backtest GNN on AAPL (1 year)
   - `test_backtest_all_models()` - Backtest 4 models on AAPL (6 months)
   - `test_walk_forward_no_lookahead()` - Verify walk-forward correctness

**End-to-End Tests:**

5. **Full Workflow:**
   - `test_backtest_e2e()` - 3 stocks × 1 year × 4 models
   - `test_report_generation()` - Generate markdown report
   - `test_visualization_creation()` - Create charts without errors

**Expected Test Count:** 15 tests

**Test Execution Time:**
- Unit tests: <5s
- Integration tests: ~30s (1 stock × 1 year)
- E2E tests: ~2 min (3 stocks × 1 year)
- **Total: <3 minutes**

---

### 6.2 Example Test Cases

```python
# tests/test_backtesting.py

import numpy as np
from src.backtesting.metrics import calculate_rmse, calculate_directional_accuracy

def test_rmse_calculation():
    """Test RMSE formula correctness"""
    predictions = np.array([100, 102, 98, 105])
    actuals = np.array([101, 101, 99, 104])

    # Manual calculation
    errors = predictions - actuals
    expected_rmse = np.sqrt(np.mean(errors ** 2))

    actual_rmse = calculate_rmse(predictions, actuals)

    assert np.isclose(actual_rmse, expected_rmse), f"Expected {expected_rmse}, got {actual_rmse}"

def test_directional_accuracy():
    """Test directional prediction accuracy"""
    predictions = np.array([102, 98, 101, 105])  # UP, DOWN, UP, UP
    actuals = np.array([101, 99, 100, 106])      # UP, DOWN, DOWN, UP
    current_prices = np.array([100, 100, 100, 100])

    # Expected: 3/4 correct (75%)
    # predictions[0]: UP, actuals[0]: UP ✅
    # predictions[1]: DOWN, actuals[1]: DOWN ✅
    # predictions[2]: UP, actuals[2]: DOWN ❌
    # predictions[3]: UP, actuals[3]: UP ✅

    accuracy = calculate_directional_accuracy(predictions, actuals, current_prices)

    assert accuracy == 75.0, f"Expected 75%, got {accuracy}%"

def test_no_lookahead_bias(historical_data_aapl):
    """Test that features don't use future data"""
    from src.backtesting.feature_extractor import extract_features_for_backtest
    from datetime import datetime

    # Extract features for 2023-01-15
    date = datetime(2023, 1, 15)
    features = await extract_features_for_backtest('AAPL', date, historical_data_aapl, 'gnn')

    # Verify all prices are <= date
    assert all(features['prices_20d'].index <= date), "Lookahead bias detected!"

# ... more tests
```

---

## 7. Trade-Off Analysis

### 7.1 Accuracy vs Speed

| Approach | Accuracy | Speed | Decision |
|----------|----------|-------|----------|
| **3 years (750 days)** | High (multiple regimes) | Medium (~40 min for 46 stocks) | **Selected** |
| **1 year (250 days)** | Medium (limited regimes) | Fast (~15 min) | Initial validation only |
| **5 years (1250 days)** | High (max history) | Slow (~70 min) | Future work |

**Rationale:** 3 years balances statistical significance with iteration speed.

---

### 7.2 Backtesting Depth vs Iteration Speed

| Depth | Stocks | Models | Time | Use Case |
|-------|--------|--------|------|----------|
| **Shallow** | 3 | 2 | ~5 min | Rapid prototyping |
| **Medium** | 10 | 4 | ~16 min | Initial validation |
| **Deep** | 46 | 4 | ~40 min | Production readiness |

**Recommendation:** Start medium (10 stocks) → Expand to deep (46 stocks) after validation.

---

### 7.3 Memory vs Caching Strategy

| Strategy | Memory Usage | Speed | Complexity |
|----------|--------------|-------|------------|
| **No cache** | Low (50 MB) | Slow (repeated API calls) | Simple |
| **In-memory only** | High (2-5 GB) | Fast (no disk I/O) | Medium |
| **Filesystem cache** | Medium (500 MB) | Fast (cached) | Medium |
| **Redis cache** | Medium | Fast | High (requires Redis) |

**Decision:** Filesystem cache (persistent, simple, fast enough)

---

## 8. Production Considerations

### 8.1 Retraining Triggers

**When to retrain models based on backtest results:**

1. **Performance Degradation:**
   - If Sharpe ratio drops below 0.5 for 2 consecutive weeks → Retrain
   - If directional accuracy drops below 52% → Investigate and retrain

2. **Regime Change Detection:**
   - If VIX spikes >30 (volatile regime) → Boost Epidemic model weight
   - If VIX <15 (calm regime) → Boost TFT model weight

3. **Correlation Drift (GNN):**
   - If average correlation drops <0.3 → Retrain GNN with new correlations
   - Schedule: Daily correlation refresh (already implemented)

4. **New Data Availability:**
   - Weekly: Retrain with latest week's data
   - Monthly: Full retraining for all models

---

### 8.2 Model Selection Strategy

**By Sector:**

| Sector | Best Model | Rationale |
|--------|-----------|-----------|
| **Technology** | GNN | High correlations (AAPL, MSFT, GOOGL move together) |
| **Finance** | GNN | Bank stocks correlated with interest rates |
| **Healthcare** | Mamba | Long-term drug pipelines (30-day predictions) |
| **Energy** | Epidemic | Volatility clustering (oil shocks) |
| **Indices (SPY, QQQ)** | Ensemble | Diversified exposure |

**By Time Horizon:**

| Horizon | Best Model | Rationale |
|---------|-----------|-----------|
| **1-day** | GNN | Real-time correlations |
| **5-day** | Ensemble | Balanced |
| **30-day** | Mamba | Long sequences |

**By Market Regime:**

| Regime | Best Model | Rationale |
|--------|-----------|-----------|
| **Volatile (VIX >20)** | Epidemic | Contagion modeling |
| **Calm (VIX <15)** | TFT | Pattern recognition |
| **Trending** | Mamba | Momentum capture |

---

### 8.3 Ensemble Weight Optimization

**Initial Weights (Equal):**
- GNN: 25%
- Mamba: 25%
- PINN: 25%
- Epidemic: 25%

**Optimized Weights (After Backtest):**

Based on expected Sharpe ratios from backtest:

```python
# Pseudocode for weight optimization
sharpe_ratios = {
    'gnn': 1.45,
    'mamba': 1.23,
    'pinn': 1.34,
    'epidemic': 0.98
}

# Softmax weighting (higher Sharpe = higher weight)
exp_sharpes = {k: np.exp(v * 2) for k, v in sharpe_ratios.items()}
total = sum(exp_sharpes.values())
optimized_weights = {k: v / total for k, v in exp_sharpes.items()}

# Result:
# gnn: 35%
# mamba: 25%
# pinn: 28%
# epidemic: 12%
```

**Validation:**
- Backtest ensemble with optimized weights
- Verify >5% improvement over equal weights
- Monitor for overfitting (validate on out-of-sample period)

---

## 9. Success Criteria

### 9.1 Quantitative Targets

**Minimum Viable Performance (Go/No-Go Thresholds):**

| Metric | Minimum | Target | Stretch Goal |
|--------|---------|--------|--------------|
| **Directional Accuracy** | >55% | >58% | >60% |
| **Sharpe Ratio** | >0.5 | >1.0 | >1.5 |
| **Max Drawdown** | <30% | <20% | <15% |
| **RMSE (1-day)** | <8% | <5% | <3% |
| **Ensemble Improvement** | >0% | >5% | >10% |

**Decision Framework:**

1. **All models meet minimum:** **DEPLOY TO PRODUCTION** ✅
2. **Some models below minimum:** Deploy only models above threshold
3. **All models below minimum:** **DO NOT DEPLOY** ❌ (investigate issues)

---

### 9.2 Qualitative Validation

**Additional Checks:**

1. **Plausibility:**
   - Do predictions make economic sense?
   - Are extreme predictions (>20% moves) justified by news/events?

2. **Consistency:**
   - Do models agree on direction in normal conditions?
   - High disagreement (model_agreement <0.5) = uncertainty → reduce position size

3. **Robustness:**
   - Do models perform well across sectors? (tech, finance, healthcare)
   - Do models handle different regimes? (2022 bear, 2023 bull)

---

## 10. Recommendations

### 10.1 Prioritized Model Ranking (Expected)

**Based on anticipated backtest results:**

1. **GNN (Graph Neural Networks)** - Grade: **A-**
   - Expected Sharpe: 1.45
   - Strengths: Tech sector, correlated stocks
   - Weaknesses: Weak on uncorrelated assets

2. **PINN (Physics-Informed NN)** - Grade: **B+**
   - Expected Sharpe: 1.34
   - Strengths: Risk bounds (Greeks), volatility modeling
   - Weaknesses: Requires options data (limited in free tier)

3. **Mamba (State-Space Model)** - Grade: **B**
   - Expected Sharpe: 1.23
   - Strengths: 30-day predictions, long sequences
   - Weaknesses: Underperforms on 1-day predictions

4. **Epidemic Volatility** - Grade: **C+**
   - Expected Sharpe: 0.98
   - Strengths: Volatile regimes (VIX spikes)
   - Weaknesses: Underperforms in calm markets

5. **Ensemble** - Grade: **A** (if optimized)
   - Expected Sharpe: 1.67 (5-15% improvement over GNN)
   - Strengths: Combines strengths, reduces weaknesses
   - Weaknesses: Complexity (requires all models operational)

---

### 10.2 Suggested Ensemble Weights

**Conservative (Equal):**
- GNN: 25%, Mamba: 25%, PINN: 25%, Epidemic: 25%

**Optimized (Sharpe-Weighted):**
- GNN: 35%, Mamba: 25%, PINN: 28%, Epidemic: 12%

**Regime-Aware (Dynamic):**
- Volatile (VIX >20): Epidemic 40%, PINN 30%, GNN 20%, Mamba 10%
- Calm (VIX <15): Mamba 35%, GNN 30%, PINN 25%, Epidemic 10%

---

### 10.3 Next Steps for Production Deployment

**Phase 1: Validation (Week 1)**
- Run backtest on 10 stocks × 3 years
- Verify metrics meet minimum thresholds
- Generate initial performance report

**Phase 2: Optimization (Week 2)**
- Optimize ensemble weights based on backtest
- Implement regime-aware weighting
- Backtest optimized ensemble

**Phase 3: Expansion (Week 3)**
- Expand backtest to all 46 stocks
- Sector-specific analysis (tech vs finance vs healthcare)
- Identify best models per sector

**Phase 4: Production Deployment (Week 4)**
- Deploy top-performing models to production API
- Implement live performance monitoring
- Set up retraining schedule (weekly)
- A/B test: Ensemble vs individual models

**Phase 5: Continuous Improvement (Ongoing)**
- Monthly backtest refresh (rolling 3 years)
- Adaptive weight adjustment based on recent performance
- Expand to options trading strategies (if Sharpe >1.5)

---

## Conclusion

This architecture provides a **rigorous, production-ready backtesting framework** for validating 5 neural network models on historical stock data.

**Key Features:**
- Walk-forward analysis (no lookahead bias)
- Comprehensive metrics (12+ metrics covering accuracy, risk, drawdown)
- 3-tier caching (filesystem, memory, API with circuit breaker)
- Feature alignment with live prediction pipeline
- Actionable insights (model rankings, ensemble weights, go/no-go decisions)

**Expected Deliverables:**
- 6 modules (~1,700 lines of code)
- Initial validation (10 stocks × 3 years) in ~16 minutes
- Full backtest (46 stocks × 3 years) in ~40 minutes
- Performance report with model rankings and recommendations

**Success Criteria:**
- Directional accuracy >55%
- Sharpe ratio >1.0
- Max drawdown <20%
- Ensemble improvement >5%

**Risk Mitigation:**
- Incremental validation (10 stocks → 46 stocks)
- Conservative benchmarks (industry-tested)
- Graceful degradation (circuit breaker, fallback data)
- Comprehensive testing (15 tests covering all components)

**Production Readiness:** HIGH
- Institutional-grade methodology (Renaissance, Citadel standards)
- Measurable success criteria
- Clear go/no-go framework
- Actionable deployment plan

---

**RECOMMENDATION: APPROVE FOR IMPLEMENTATION**

Hand off to **Expert Code Writer** for Phase 2 implementation (3-4 hours).

---

**Appendix: Additional Resources**

**Papers:**
- Bailey et al. (2015): "The Probability of Backtest Overfitting"
- López de Prado (2018): "Advances in Financial Machine Learning"
- Sharpe (1994): "The Sharpe Ratio"

**Industry Benchmarks:**
- Renaissance Medallion Fund: Sharpe >2.0 (after fees)
- Two Sigma: Sharpe ~1.5-2.0
- Citadel Kensington: Sharpe ~1.2-1.8

**Risk Management Standards:**
- BlackRock Aladdin: Max drawdown <15% for institutional funds
- Basel III: VaR 99% confidence (1-day) <3%

---

**End of Architecture Document**
