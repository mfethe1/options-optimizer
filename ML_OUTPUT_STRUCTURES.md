# ML Model Output Structures Analysis - Visualization Data Availability

## Executive Summary

This analysis covers 5 advanced ML models with distinct output architectures optimized for different financial forecasting tasks. Each model produces multi-horizon predictions with varying certainty measures and interpretability features.

---

## 1. TFT (Temporal Fusion Transformer) - Priority #1

### Location
- **Model**: `/home/user/options-optimizer/src/ml/advanced_forecasting/tft_model.py`
- **Routes**: `/home/user/options-optimizer/src/api/advanced_forecast_routes.py`
- **Output Dataclass**: `MultiHorizonForecast`

### Time Horizons/Scales
```
Horizons: [1, 5, 10, 30] days
Granularity: Multi-horizon simultaneous forecasting
```

### Output Metrics Available
```python
class MultiHorizonForecast:
    timestamp: datetime
    symbol: str
    horizons: List[int]           # [1, 5, 10, 30]
    
    # Point Predictions
    predictions: List[float]      # Mean predictions (one per horizon)
    
    # Quantile Forecasts (Uncertainty Estimation)
    q10: List[float]              # 10th percentile (lower bound)
    q50: List[float]              # 50th percentile (median)
    q90: List[float]              # 90th percentile (upper bound)
    
    # Interpretability
    feature_importance: Dict[str, float]  # Which features matter most
    attention_weights: Optional[np.ndarray]  # Attention mechanism weights
    
    # Context
    current_price: float          # Reference price for return calculations
```

### Data Format (JSON Response Example)
```json
{
  "symbol": "AAPL",
  "timestamp": "2025-11-04T10:30:00",
  "current_price": 230.50,
  "horizons": [1, 5, 10, 30],
  "coverage_level": 0.95,
  
  "predictions": [231.25, 235.10, 240.80, 255.40],
  
  "tft_q10": [229.50, 232.40, 235.60, 245.20],
  "tft_q50": [231.25, 235.10, 240.80, 255.40],
  "tft_q90": [233.00, 237.80, 246.00, 265.60],
  
  "conformal_lower": [229.10, 231.80, 234.90, 243.50],
  "conformal_upper": [233.40, 238.40, 246.70, 267.30],
  "conformal_width": [4.30, 6.60, 11.80, 23.80],
  
  "expected_returns": [0.0033, 0.0200, 0.0450, 0.1083],
  "feature_importance": {
    "momentum": 0.34,
    "volatility": 0.28,
    "volume": 0.18,
    "price_level": 0.20
  },
  
  "model": "Temporal Fusion Transformer + Conformal Prediction",
  "is_calibrated": true
}
```

### Visualization Suitability
✅ **EXCELLENT** for time series plotting:
- Multi-horizon point predictions as time series
- Cone chart: q10/q50/q90 forming expanding uncertainty zones
- Conformal prediction intervals as guaranteed coverage band
- Feature importance as bar chart
- Expected return trajectory

### Advanced Metrics
- **Prediction Intervals**: 3-level quantile structure (10%, 50%, 90%)
- **Conformal Guarantees**: Mathematically guaranteed 95% coverage
- **Multi-horizon learning**: All horizons trained jointly (shared feature learning)

---

## 2. GNN (Graph Neural Network) - Priority #2

### Location
- **Model**: `/home/user/options-optimizer/src/ml/graph_neural_network/stock_gnn.py`
- **Routes**: `/home/user/options-optimizer/src/api/gnn_routes.py`
- **Output Dataclass**: `StockGraph`

### Time Horizons/Scales
```
Correlation Window: 20 days (lookback for correlation calculation)
Prediction Type: Single-horizon return predictions
Dynamic Updates: Daily graph reconstruction
```

### Output Metrics Available
```python
@dataclass
class StockGraph:
    symbols: List[str]                    # [AAPL, MSFT, GOOGL, ...]
    correlation_matrix: np.ndarray        # [n_stocks, n_stocks]
    edge_index: np.ndarray               # [2, n_edges] - connectivity pairs
    edge_weights: np.ndarray             # [n_edges] - correlation strengths
    node_features: np.ndarray            # [n_stocks, n_features] per-stock features
    timestamp: datetime
```

### Data Format (JSON Response Example)
```json
{
  "timestamp": "2025-11-04T10:30:00",
  "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
  
  "predictions": {
    "AAPL": 0.0245,
    "MSFT": 0.0183,
    "GOOGL": -0.0052,
    "AMZN": 0.0341,
    "NVDA": 0.0512
  },
  
  "correlations": {
    "AAPL": {
      "MSFT": 0.72,
      "GOOGL": 0.65,
      "AMZN": 0.54,
      "NVDA": 0.68
    },
    "MSFT": {
      "AAPL": 0.72,
      "GOOGL": 0.71,
      ...
    }
  },
  
  "graph_stats": {
    "num_nodes": 5,
    "num_edges": 10,
    "avg_correlation": 0.642,
    "max_correlation": 0.82
  },
  
  "top_correlations": [
    {"symbol1": "MSFT", "symbol2": "GOOGL", "correlation": 0.82},
    {"symbol1": "AAPL", "symbol2": "NVDA", "correlation": 0.78},
    {"symbol1": "AAPL", "symbol2": "MSFT", "correlation": 0.72}
  ]
}
```

### Visualization Suitability
✅ **EXCELLENT** for network/graph visualization:
- **Correlation Matrix Heatmap**: Visual representation of all pairwise correlations
- **Network Graph**: Nodes = stocks, edges = strong correlations (thickness = strength)
- **Edge Weight Distribution**: Histogram of correlation strengths
- **Adjacency Matrix**: Reorderable correlation matrix with dendrogram
- **Sector/Cluster Visualization**: Related stocks grouped by correlation

### Graph Structure Details
- **Nodes**: Individual stocks with feature vectors (60 dimensions)
- **Edges**: Created when |correlation| > 0.3 threshold
- **Edge Weights**: Absolute correlation values (0 to 1)
- **Graph Type**: Undirected, dynamic (updates daily)

### Advanced Metrics
- **Message Passing**: GCN aggregates neighbor features through learned weights
- **Attention Mechanisms**: GAT learns importance of each neighbor
- **Temporal Evolution**: Graph structure changes as correlations evolve

---

## 3. PINN (Physics-Informed Neural Networks) - Priority #4

### Location
- **Model**: `/home/user/options-optimizer/src/ml/physics_informed/general_pinn.py`
- **Routes**: `/home/user/options-optimizer/src/api/pinn_routes.py`

### Time Horizons/Scales
```
Application 1 - Option Pricing:
  Maturity Range: 0.1 to 2.0 years
  Price Range: $50 to $150 (parameterized)
  
Application 2 - Portfolio Optimization:
  Historical Lookback: 252 days (1 year)
```

### Output Metrics - Option Pricing
```python
# OptionPricingPINN.predict() output
{
    'price': float,              # Option fair value
    'method': str,               # 'PINN' or 'Black-Scholes (fallback)'
    'delta': float,              # ∂V/∂S - price sensitivity
    'gamma': float,              # ∂²V/∂S² - delta sensitivity
    'theta': float,              # -∂V/∂τ - time decay
}
```

### Data Format - Option Pricing (JSON Response)
```json
{
  "timestamp": "2025-11-04T10:30:00",
  "option_type": "call",
  "stock_price": 100.0,
  "strike_price": 100.0,
  "time_to_maturity": 1.0,
  "price": 10.45,
  "method": "PINN",
  
  "greeks": {
    "delta": 0.6254,      # 62.54% price elasticity
    "gamma": 0.0185,      # Curvature of delta
    "theta": -0.0247      # Daily value decay
  }
}
```

### Output Metrics - Portfolio Optimization
```python
{
    'weights': List[float],          # Optimal allocation per asset
    'expected_return': float,        # Portfolio annual return
    'risk': float,                   # Portfolio standard deviation
    'sharpe_ratio': float,           # Return per unit risk
    'method': str,                   # 'Markowitz (PINN-inspired)'
}
```

### Data Format - Portfolio (JSON Response)
```json
{
  "timestamp": "2025-11-04T10:30:00",
  "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
  "weights": [0.20, 0.18, 0.22, 0.18, 0.22],
  "expected_return": 0.1234,
  "risk": 0.1856,
  "sharpe_ratio": 0.665,
  "method": "Markowitz (PINN-inspired)"
}
```

### Visualization Suitability
✅ **GOOD** for specialized financial charts:

**Option Pricing:**
- Greeks surface plots: price sensitivity across S/τ space
- Greeks values at current parameters
- Price vs strike (payoff diagram with option value overlay)

**Portfolio:**
- Efficient frontier: risk vs return curve
- Allocation pie chart
- Weight comparison bar chart

### Special Features
- **Automatic Differentiation**: Greeks computed via TensorFlow autodiff
- **Physics Constraints**: Black-Scholes PDE enforced during training
- **Data Efficiency**: 15-100x less training data needed vs standard NN
- **No-Arbitrage**: Monotonicity and convexity constraints automatically enforced

---

## 4. Mamba (State Space Model) - Priority #3

### Location
- **Model**: `/home/user/options-optimizer/src/ml/state_space/mamba_model.py`
- **Routes**: `/home/user/options-optimizer/src/api/mamba_routes.py`

### Time Horizons/Scales
```
Input Sequence Length: Up to 10M+ time steps
  - 1000-5000 daily bars (4-20 years of daily data)
  - 100,000+ 1-minute bars (weeks of intraday data)
  - 1M+ tick data (hours of real-time data)

Output Horizons: [1, 5, 10, 30] days
Processing Complexity: O(N) - LINEAR vs Transformer O(N²)
```

### Output Metrics Available
```python
class MambaConfig:
    d_model: int = 64              # Model dimension
    d_state: int = 16              # SSM state dimension
    num_layers: int = 4            # Network depth
    prediction_horizons: List[int] = [1, 5, 10, 30]

# MambaPredictor.predict() output
{
    '1d': float,   # 1-day-ahead price prediction
    '5d': float,   # 5-day-ahead price prediction
    '10d': float,  # 10-day-ahead price prediction
    '30d': float   # 30-day-ahead price prediction
}
```

### Data Format (JSON Response Example)
```json
{
  "timestamp": "2025-11-04T10:30:00",
  "symbol": "AAPL",
  "current_price": 230.50,
  
  "predictions": {
    "1d": 231.85,
    "5d": 235.42,
    "10d": 240.16,
    "30d": 255.73
  },
  
  "efficiency_stats": {
    "sequence_length": 1250,
    "mamba_complexity": "O(N)",
    "transformer_complexity": "O(N²)",
    "mamba_ops": 1280000,
    "transformer_ops": 156250000,
    "theoretical_speedup": "121.9x",
    "can_process_ticks": true,
    "memory_efficient": true
  },
  
  "signal": "BUY",
  "confidence": 0.68
}
```

### Visualization Suitability
✅ **EXCELLENT** for long-horizon time series:
- Multi-horizon predictions as extended forecast curve
- Input sequence visualization: 5+ years of daily data in single plot
- Efficiency comparison charts: Mamba vs Transformer complexity
- Real-time streaming data plots (unique capability due to O(N) complexity)

### Special Architectural Features
```
Selective State Space Model (Core Innovation):
  - Input-dependent parameters: B(t), C(t), Δ(t)
  - Learns what to remember vs forget (selectivity)
  - Depthwise convolution for local context
  - Gating mechanism for information flow control
  
Efficiency Advantage:
  - Can process 10M+ time steps where Transformer would be impossible
  - Constant memory per timestep (not quadratic)
  - 5x throughput improvement vs Transformers
  - Hardware-aware algorithm (GPU/TPU optimized)
```

### Use Cases (Enabled by Linear Complexity)
- High-frequency trading: Process every millisecond tick
- Multi-year analysis: 20 years of daily data in one batch
- Intraday patterns: 5 years of 1-minute bars (975,000 points)
- Real-time streaming: Constant latency per new tick

---

## 5. Epidemic Volatility Model - Bio-Financial Innovation

### Location
- **Model**: `/home/user/options-optimizer/src/ml/bio_financial/epidemic_volatility.py`
- **Routes**: `/home/user/options-optimizer/src/api/epidemic_volatility_routes.py`

### Time Horizons/Scales
```
Forecast Horizon: 30 days (configurable)
Simulation Timestep: 0.1 days
Time Steps: 300 steps per forecast (daily resolution in output)
Market Regimes: 4 distinct states
```

### Output Metrics Available
```python
@dataclass
class EpidemicForecast:
    timestamp: datetime
    horizon_days: int = 30
    
    # VIX Prediction
    predicted_vix: float
    
    # Market Regime
    predicted_regime: MarketRegime  # SUSCEPTIBLE | EXPOSED | INFECTED | RECOVERED
    confidence: float
    
    # Epidemic Trajectories (time series, daily resolution)
    S_forecast: List[float]        # Susceptible proportion trajectory [0,1]
    I_forecast: List[float]        # Infected proportion trajectory [0,1]
    R_forecast: List[float]        # Recovered proportion trajectory [0,1]
    E_forecast: Optional[List[float]]  # Exposed (SEIR only) [0,1]
    
    # Parameter Trajectories
    beta_trajectory: List[float]    # Infection rate over time
    gamma_trajectory: List[float]   # Recovery rate over time
    
    # Key Events
    herd_immunity_days: Optional[int]    # When market stabilizes
    peak_volatility_days: Optional[int]  # When VIX peaks
    peak_vix: Optional[float]
```

### Data Format (JSON Response Example)
```json
{
  "timestamp": "2025-11-04T10:30:00",
  "horizon_days": 30,
  "predicted_vix": 18.5,
  "predicted_regime": "infected",
  "confidence": 0.75,
  "current_vix": 16.2,
  "current_sentiment": 0.34,
  
  "trading_signal": {
    "action": "buy_protection",
    "confidence": 0.75,
    "reasoning": "Market entering pre-volatile state. Infection spreading. Expected VIX: 18.5"
  },
  
  "interpretation": "Market regime: INFECTED. Volatility contagion active. High fear spreading. VIX at 16.2. Contagion ongoing. Hold protection. Trading signal: BUY_PROTECTION. Confidence: 75%."
}
```

### Current State Endpoint
```json
{
  "timestamp": "2025-11-04T10:30:00",
  "regime": "infected",
  
  "susceptible": 0.45,      # Calm market proportion
  "exposed": 0.12,          # Pre-volatile proportion
  "infected": 0.28,         # Volatile proportion
  "recovered": 0.15,        # Stabilized proportion
  
  "beta": 0.3542,           # Fear transmission rate
  "gamma": 0.2156,          # Stabilization rate
  
  "current_vix": 16.2,
  "current_sentiment": 0.34
}
```

### Visualization Suitability
✅ **EXCELLENT** for stacked area charts and regime analysis:

- **Stacked Area Chart**: S/E/I/R proportions over 30-day forecast
  - Bottom to top: Susceptible | Exposed | Infected | Recovered
  - Colors indicate market health
  - Shows contagion dynamics visually

- **Dual-Axis Chart**: 
  - Left axis: Infected proportion (I)
  - Right axis: Predicted VIX
  - Visual correlation between epidemic state and volatility

- **Phase Diagram**:
  - X-axis: β (infection rate/fear transmission)
  - Y-axis: γ (recovery rate/stabilization)
  - Shows market dynamics in 2D parameter space

- **Historical Episodes**:
  - Timeline of past "epidemic" events
  - Each event: start date, peak VIX, duration, severity classification

### Epidemic Model Details
```
SIR (Simpler):
  dS/dt = -β(t) * S * I
  dI/dt = β(t) * S * I - γ(t) * I
  dR/dt = γ(t) * I

SEIR (More detailed):
  dS/dt = -β(t) * S * I
  dE/dt = β(t) * S * I - σ(t) * E
  dI/dt = σ(t) * E - γ(t) * I
  dR/dt = γ(t) * I

Where:
  S = Susceptible (calm market)
  E = Exposed (pre-volatile state)
  I = Infected (volatile market)
  R = Recovered (stabilized market)
  β = Infection rate (learned from sentiment, volume)
  γ = Recovery rate (learned from capital inflows)
  σ = Incubation rate (E→I transition speed)
```

### Input Features for Prediction
```
market_features = [
    current_vix / 100.0,        # Normalized VIX level
    realized_vol / 100.0,       # Rolling volatility
    (sentiment + 1) / 2,        # Market sentiment [-1,1] → [0,1]
    volume                      # Trading volume indicator
]
```

### Special Insights
- **Herd Immunity Signal**: When S crosses below threshold, market likely to stabilize soon
- **Contagion Pathways**: Predicted spread of fear through market
- **Trading Signals by Regime**:
  - SUSCEPTIBLE: Monitor for early warnings
  - EXPOSED: Buy protection before spike
  - INFECTED: Hold protection during contagion
  - RECOVERED: Sell volatility as stabilization continues

---

## Comparison Matrix

| Feature | TFT | GNN | PINN | Mamba | Epidemic |
|---------|-----|-----|------|-------|----------|
| **Horizons** | 1,5,10,30 days | Single return | 0.1-2 years (options) | 1,5,10,30 days | 30 days (stochastic) |
| **Time Scale** | Days | Days | Continuous | Flexible (daily to tick) | Days |
| **Uncertainty** | Quantiles (3-level) + Conformal | Correlation matrix | Greeks | None | Regime + confidence |
| **Outputs** | Point + intervals | Predictions + correlations | Price + Greeks | Prices | VIX + regimes |
| **Complexity** | O(N²) | O(N) graph ops | O(1) for fixed topology | **O(N)** | O(M·N) where M=30 |
| **Visualization** | Time series cone | Network heatmap | Greeks surface | Long sequences | Area charts |
| **Data Efficiency** | Standard | Standard | **15-100x** (physics) | Standard | Moderate (PINN-informed) |
| **Interpretability** | Attention weights | Edge weights | Physics-based | Selective mechanism | Epidemic dynamics |
| **Best For** | Multi-day forecasts | Portfolio correlation | Option pricing | Long histories | Volatility regimes |

---

## Visualization Architecture Recommendations

### 1. Dashboard Layout Strategy

**Primary Visualization Grid:**
```
┌─────────────────────────────────────────────┐
│        TFT Multi-Horizon Forecast            │
│  (Cone chart: q10/q50/q90 uncertainty)      │
├─────────────────────────────────────────────┤
│    GNN Correlation Matrix Heatmap            │
│  (Top N correlations with network overlay)  │
├─────────────────────────────────────────────┤
│   Epidemic State Stacked Area Chart          │
│  (S/E/I/R proportions + predicted VIX)      │
├─────────────────────────────────────────────┤
│     Mamba Long Sequence + Efficiency        │
│  (Multi-year daily data + speedup metrics)  │
└─────────────────────────────────────────────┘
```

### 2. Data Flow for Visualization

```
API Layer (Routes)
    ↓
Response Models (BaseModel)
    ↓
JSON Serialization
    ↓
Frontend (React/D3.js/Plotly)
    ↓
Interactive Charts
```

### 3. Key Data Structures for Frontend

All models return JSON with:
- **timestamp**: ISO format datetime for synchronization
- **symbol**: For multi-symbol comparison
- **Predictions**: Arrays suitable for line charts
- **Confidence/Intervals**: For error bands
- **Metadata**: Model type, horizon, coverage guarantees

---

## Conclusion

Each model produces visualization-ready outputs:
- **TFT**: Probability cones + feature importance
- **GNN**: Correlation networks + edge weights
- **PINN**: Greeks surfaces + constraint satisfaction
- **Mamba**: Extended time series + efficiency metrics
- **Epidemic**: Regime trajectories + VIX mapping

All outputs are structured as JSON with array/object nesting suitable for modern visualization libraries (D3.js, Plotly, ECharts, etc.).
