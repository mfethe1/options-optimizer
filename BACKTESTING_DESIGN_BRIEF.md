# ML Backtesting Framework - Design Brief

**Project:** Options Probability Analysis System
**Date:** 2025-11-09
**Phase:** 1 - Architecture Design
**Assigned To:** ML Neural Network Architect
**Estimated Duration:** 90 minutes
**Deliverable:** BACKTESTING_ARCHITECTURE.md

---

## Mission

Design an institutional-grade backtesting framework to validate the performance of all 5 neural network models (GNN, Mamba, PINN, Epidemic, Ensemble) on historical data. The framework must provide rigorous performance metrics to guide production deployment decisions.

---

## Current System State

### Available Models

1. **GNN (Graph Neural Networks)** - 46 pre-trained models in `models/gnn/weights/`
   - Symbols: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, GS, BAC, BLK, MS, C, WFC, PFE, JNJ, MRK, UNH, LLY, ABBV, DIS, HD, WMT, COST, MCD, NKE, COP, XOM, CVX, SPY, IWM, DIA, QQQ, NFLX, INTC, CSCO, ADBE, AMD, CRM, ORCL, PYPL, MA, SQ, V, UBER, ABNB
   - Prediction: Daily returns via stock correlation networks
   - Latency: 315ms (cached) to 815ms (uncached)

2. **Mamba (State-Space Model)** - `src/ml/state_space/mamba_model.py`
   - Multi-horizon forecasting: 1d, 5d, 10d, 30d
   - Long-sequence modeling (1000+ days)
   - Latency: ~500ms

3. **PINN (Physics-Informed Neural Network)** - `src/ml/physics_informed/general_pinn.py`
   - Black-Scholes PDE constraints
   - Options pricing + Greeks (delta, gamma, theta)
   - Latency: ~600ms

4. **Epidemic Volatility** - `src/ml/bio_financial/epidemic_volatility.py`
   - VIX prediction via SIR/SEIR epidemic models
   - Market regime classification (calm, pre-volatile, volatile, stabilized)
   - Latency: ~400ms

5. **Ensemble** - `src/ml/ensemble/ensemble_predictor.py`
   - Weighted combination of all 4 models
   - Adaptive weighting based on performance tracking
   - Signal aggregation (BUY/SELL/HOLD)

### Integration Points

**Live Prediction API:**
- `src/api/ml_integration_helpers.py`:
  - `get_gnn_prediction(symbol, current_price)` → Returns prediction dict
  - `get_mamba_prediction(symbol, current_price)` → Multi-horizon forecast
  - `get_pinn_prediction(symbol, current_price)` → Options-based prediction

**Ensemble Service:**
- `src/ml/ensemble/ensemble_predictor.py`:
  - `EnsemblePredictor.predict()` → Combines all models
  - `PerformanceTracker.record_prediction()` → Tracks accuracy for adaptive weighting

**Data Pipeline:**
- `fetch_historical_prices(symbols, days)` - yfinance with circuit breaker
- Circuit breaker pattern: Exponential backoff, max 3 retries, 120s recovery timeout
- Handles missing data gracefully with fallback

### Existing Test Patterns

Reference files in `tests/`:
- `test_gnn_integration.py` - GNN testing patterns
- `test_ensemble_integration.py` - Ensemble testing
- `test_epidemic_integration.py` - Epidemic model testing
- `test_ml_integration_p0_fix.py` - Integration testing

---

## Requirements from User

### Primary Objectives

1. **Backtest all 5 models** on historical data (1-3 years recommended)
2. **Calculate comprehensive performance metrics**:
   - Prediction accuracy (RMSE, MAE, MAPE)
   - Directional accuracy (% correct up/down predictions)
   - Risk-adjusted returns (Sharpe ratio, Sortino ratio)
   - Drawdown analysis (maximum drawdown, recovery time)
   - Model-specific metrics (GNN correlation strength, PINN Greeks accuracy, Mamba horizon performance)
   - Ensemble improvement over individual models

3. **Compare models** to identify best performers by:
   - Time horizon (1d vs 30d)
   - Market regime (volatile vs calm)
   - Sector (tech vs finance vs healthcare)

4. **Generate actionable insights** for production deployment:
   - Which models to prioritize?
   - Optimal ensemble weights?
   - Sector-specific model selection?
   - Risk thresholds for position sizing?

### Constraints

**Data Availability:**
- yfinance API: Free tier, rate-limited (2000 requests/hour)
- Historical depth: Up to 5 years available
- Granularity: Daily close prices (no intraday for free tier)
- Quality: Adjusted for splits/dividends

**Compute Resources:**
- Single machine (Windows, 16+ GB RAM)
- No GPU required (pre-trained models used)
- Runtime budget: <6 hours for full 46-stock backtest

**Technical Debt:**
- Must use existing data pipeline (yfinance + circuit breaker)
- Must respect rate limits (batch requests, caching)
- Must handle missing data gracefully

---

## Architecture Design Tasks

You are the **ML Neural Network Architect**. Design a comprehensive backtesting framework addressing:

### 1. Backtesting Methodology

**Design Decisions Required:**
- **Walk-forward analysis** vs fixed train/test split?
  - Walk-forward: Expanding window (prevents lookahead bias)
  - Fixed split: Simple but doesn't reflect production usage
  - Recommendation: Which and why?

- **Time period selection:**
  - 1 year: Fast iteration, but limited regimes
  - 3 years: Captures COVID crash + recovery (2020-2023)
  - 5 years: Maximum history, slow iteration
  - Recommendation: Which and why?

- **Prediction horizon:**
  - Models predict 1d, 5d, 10d, 30d
  - Should we backtest all horizons or focus on 1d + 30d?
  - How to handle multi-horizon evaluation?

- **Feature alignment:**
  - Live predictions use 20-day rolling windows
  - Backtest must match this exactly (no lookahead)
  - How to ensure feature parity?

**Deliverable:** Detailed methodology with rationale for each choice.

### 2. Performance Metrics

**Required Metrics:**

**Prediction Accuracy:**
- RMSE (Root Mean Squared Error): `sqrt(mean((y_pred - y_actual)^2))`
- MAE (Mean Absolute Error): `mean(abs(y_pred - y_actual))`
- MAPE (Mean Absolute Percentage Error): `mean(abs((y_pred - y_actual) / y_actual))`

**Directional Accuracy:**
- % Correct: `sum(sign(y_pred - y_current) == sign(y_actual - y_current)) / n`
- Precision/Recall for BUY/SELL signals

**Risk-Adjusted Returns:**
- Sharpe Ratio: `mean(returns) / std(returns) * sqrt(252)`
- Sortino Ratio: Like Sharpe but only downside volatility
- Information Ratio: Excess return vs benchmark (SPY)

**Drawdown Analysis:**
- Maximum Drawdown: Largest peak-to-trough decline
- Recovery Time: Days to recover from max drawdown
- Calmar Ratio: Annual return / max drawdown

**Model-Specific:**
- GNN: Average correlation strength, network centrality
- PINN: Greeks accuracy (if options data available)
- Mamba: Performance by horizon (1d vs 30d)
- Epidemic: Regime classification accuracy (if VIX available)
- Ensemble: Improvement over best individual model

**Deliverable:**
- Formulas for each metric
- Expected benchmark values (what's "good"?)
- Implementation pseudocode

### 3. Data Pipeline Architecture

**Historical Data Fetching:**
- How to minimize yfinance API calls? (caching strategy)
- Batch size for parallel fetching? (avoid rate limits)
- Cache duration? (daily refresh vs persistent)
- Fallback for missing data? (forward-fill, interpolation, skip?)

**Feature Engineering:**
- Must match live prediction pipeline exactly
- 20-day rolling windows for volatility, momentum
- Correlation matrix calculation (60-day lookback for GNN)
- How to handle cold start (first 20 days)?

**Data Validation:**
- Detect gaps (missing trading days)
- Detect outliers (flash crashes, data errors)
- Handle corporate actions (splits, dividends - yfinance auto-adjusts)

**Deliverable:**
- Data pipeline flowchart
- Caching strategy with invalidation rules
- Error handling strategy

### 4. Implementation Plan

**Module Structure:**
- `src/backtesting/backtest_engine.py` - Core backtesting logic
- `src/backtesting/metrics.py` - Performance metric calculations
- `src/backtesting/historical_data_loader.py` - Data fetching + caching
- `src/backtesting/visualizer.py` - Results visualization
- `scripts/run_backtest.py` - CLI interface
- `tests/test_backtesting.py` - Comprehensive tests

**Expected File Sizes:**
- backtest_engine.py: ~400 lines
- metrics.py: ~200 lines
- historical_data_loader.py: ~250 lines
- visualizer.py: ~300 lines
- run_backtest.py: ~150 lines
- test_backtesting.py: ~250 lines
- **Total: ~1,550 lines**

**Performance Expectations:**
- Per-stock backtest: <5 minutes (3 years)
- Full 46-stock backtest: <4 hours (parallel: 10 workers)
- Memory usage: <4 GB

**Deliverable:**
- Detailed pseudocode for each module
- Class/function signatures
- Expected runtime analysis

### 5. Expected Benchmarks

**Define "good" performance:**

**Prediction Accuracy Targets:**
- RMSE: <5% for 1-day, <10% for 30-day (% of current price)
- Directional Accuracy: >55% (baseline 50% random)
- MAPE: <8% for 1-day, <15% for 30-day

**Risk-Adjusted Returns:**
- Sharpe Ratio: >1.0 (good), >2.0 (excellent)
- Sortino Ratio: >1.5 (good), >3.0 (excellent)
- Max Drawdown: <20% (acceptable), <10% (excellent)

**Model Comparison:**
- Ensemble should outperform best individual by >5%
- GNN should excel on correlated stocks (tech sector)
- Mamba should excel on long-term (30d) predictions
- PINN should provide tighter confidence bounds

**Deliverable:**
- Table of expected benchmarks with rationale
- Tier system (A/B/C/D/F grades based on metrics)

---

## Deliverable Format

Create `BACKTESTING_ARCHITECTURE.md` with the following structure:

```markdown
# Backtesting Architecture Design

## 1. Executive Summary
- Methodology overview
- Key design decisions
- Expected outcomes

## 2. Backtesting Methodology
- Walk-forward analysis specification
- Time period selection (with rationale)
- Prediction horizon strategy
- Feature alignment validation

## 3. Performance Metrics
- Complete metric definitions (with formulas)
- Expected benchmarks table
- Grading rubric (A/B/C/D/F)

## 4. Data Pipeline Architecture
- Historical data fetching strategy
- Caching implementation
- Feature engineering alignment
- Data validation rules

## 5. Implementation Plan
- Module structure (6 files)
- Detailed pseudocode for each module
- Function signatures
- Expected runtime analysis

## 6. Testing Strategy
- Test coverage plan (unit, integration, end-to-end)
- Expected test cases (~15 tests)
- Validation approach

## 7. Trade-Off Analysis
- Accuracy vs Speed
- Backtesting depth vs iteration speed
- Memory vs caching strategy

## 8. Production Considerations
- Retraining triggers (when to retrain?)
- Model selection strategy (by sector/regime)
- Ensemble weight optimization

## 9. Success Criteria
- Quantitative targets
- Go/No-Go decision framework

## 10. Recommendations
- Prioritized model ranking
- Suggested ensemble weights
- Next steps for production deployment
```

---

## Reference Materials

**Key Files to Review:**
1. `src/api/ml_integration_helpers.py` - Live prediction functions (lines 431-830)
2. `src/ml/ensemble/ensemble_predictor.py` - Ensemble logic (lines 191-636)
3. `src/ml/graph_neural_network/stock_gnn.py` - GNN architecture
4. `GNN_PRETRAINING_ARCHITECTURE.md` - Recent GNN work (reference for design quality)
5. `tests/test_gnn_pretraining.py` - Testing patterns

**Research References:**
- Walk-forward analysis: "Advances in Financial Machine Learning" (López de Prado, 2018)
- Sharpe ratio: "The Sharpe Ratio" (Sharpe, 1994)
- Backtesting pitfalls: "The Probability of Backtest Overfitting" (Bailey et al., 2015)

---

## Timeline

- **T+0 to T+30 min:** Review codebase and reference materials
- **T+30 to T+60 min:** Design methodology + metrics
- **T+60 to T+80 min:** Implementation plan + pseudocode
- **T+80 to T+90 min:** Document review + finalization

**Estimated Completion:** 90 minutes
**Next Phase:** Hand off to Expert Code Writer for implementation (3-4 hours)

---

## Questions to Address

1. Should we backtest all 46 stocks or start with a subset (e.g., 10 stocks across sectors)?
2. Should we implement incremental backtesting (1 year → 3 years → 5 years) or go straight to 3 years?
3. Should we prioritize speed (parallel execution) or memory (sequential)?
4. Should we cache all historical data upfront or fetch on-demand?
5. Should we generate visualizations (matplotlib) or focus on metrics tables?

**Your Recommendation:** Provide clear answers with rationale.

---

## Success Criteria for This Phase

- [ ] Comprehensive methodology designed (no lookahead bias)
- [ ] 10+ performance metrics defined with formulas
- [ ] Data pipeline leverages existing infrastructure (yfinance + circuit breaker)
- [ ] Implementation plan is clear and actionable (6 files, ~1,550 lines)
- [ ] Benchmarks are realistic and measurable (RMSE <5%, Sharpe >1.0)
- [ ] Document is ready for handoff to Expert Code Writer

---

**BEGIN ARCHITECTURE DESIGN NOW.**

Focus on institutional rigor, methodological soundness, and production readiness. This backtesting framework will validate $500K-$2M in production decisions.
