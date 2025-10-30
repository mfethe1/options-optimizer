# Phase 2: Earnings Intelligence System

## 🎯 **Objective**

Build a comprehensive earnings intelligence system that:
1. Collects and stores historical earnings data
2. Calculates implied earnings moves from options
3. Recommends earnings-specific strategies
4. Enhances correlation analysis with headline search
5. Detects emerging trends from correlated stocks

---

## 📊 **Component 1: Historical Earnings Data Collection**

### **Goal**: Build a database of historical earnings for backtesting and analysis

### **Data to Collect**:
- **Earnings Date**: When earnings were reported
- **EPS Actual**: Actual earnings per share
- **EPS Estimate**: Analyst consensus estimate
- **EPS Surprise**: Actual - Estimate
- **Revenue Actual**: Actual revenue
- **Revenue Estimate**: Analyst consensus
- **Revenue Surprise**: Actual - Estimate
- **Guidance**: Forward guidance (if available)
- **Price Move**: Stock price change on earnings day
- **Pre-earnings Price**: Price before earnings
- **Post-earnings Price**: Price after earnings
- **Implied Move**: Options-implied move (if available)

### **Data Sources** (in priority order):
1. **Finnhub** (primary): `/calendar/earnings`, `/stock/earnings`
2. **Alpha Vantage**: `EARNINGS` endpoint
3. **Polygon**: `/v2/reference/financials`
4. **yfinance**: `earnings_dates`, `earnings_history`

### **Storage Format**:
```
data/earnings/historical/{SYMBOL}_earnings_history.parquet

Columns:
- symbol: str
- earnings_date: datetime
- fiscal_quarter: str (e.g., "Q1 2024")
- eps_actual: float
- eps_estimate: float
- eps_surprise: float
- eps_surprise_pct: float
- revenue_actual: float
- revenue_estimate: float
- revenue_surprise: float
- revenue_surprise_pct: float
- guidance: str (optional)
- pre_price: float
- post_price: float
- price_move: float
- price_move_pct: float
- implied_move_pct: float (optional)
- timestamp: datetime
```

### **Implementation**:
```python
# src/services/earnings_data_collector.py

class EarningsDataCollector:
    def __init__(self):
        self.cache_dir = Path("data/earnings/historical")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_historical_earnings(
        self,
        symbol: str,
        quarters: int = 8,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Collect last N quarters of earnings data
        
        Returns DataFrame with earnings history
        """
        pass
    
    def _fetch_from_finnhub(self, symbol: str) -> List[Dict]:
        """Fetch from Finnhub API"""
        pass
    
    def _fetch_from_alpha_vantage(self, symbol: str) -> List[Dict]:
        """Fetch from Alpha Vantage API"""
        pass
    
    def _calculate_price_moves(
        self,
        symbol: str,
        earnings_dates: List[datetime]
    ) -> Dict[datetime, float]:
        """Calculate price moves around earnings dates"""
        pass
    
    def _save_to_parquet(self, symbol: str, data: pd.DataFrame):
        """Save to parquet file"""
        pass
```

### **Batch Collection Script**:
```python
# scripts/collect_earnings_data.py

# Top 100 stocks by market cap
SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
    'UNH', 'JNJ', 'V', 'XOM', 'WMT', 'JPM', 'MA', 'PG', 'CVX', 'HD',
    'LLY', 'ABBV', 'MRK', 'KO', 'PEP', 'COST', 'AVGO', 'TMO', 'MCD',
    'CSCO', 'ACN', 'ABT', 'DHR', 'NKE', 'VZ', 'ADBE', 'CRM', 'NFLX',
    'CMCSA', 'TXN', 'INTC', 'AMD', 'QCOM', 'PM', 'UNP', 'NEE', 'HON',
    # ... add more
]

def collect_all():
    collector = EarningsDataCollector()
    
    for symbol in SYMBOLS:
        print(f"Collecting {symbol}...")
        try:
            data = collector.collect_historical_earnings(symbol, quarters=8)
            print(f"✓ {symbol}: {len(data)} quarters collected")
        except Exception as e:
            print(f"✗ {symbol}: {e}")
        
        time.sleep(1)  # Rate limiting
```

---

## 📊 **Component 2: Implied Move Calculator**

### **Goal**: Calculate options-implied earnings move and compare to historical

### **Formula**:
```
Implied Move % = (ATM Straddle Price / Stock Price) * 100
```

### **Data Needed**:
- Current stock price
- Nearest expiration after earnings
- ATM call price
- ATM put price

### **Implementation**:
```python
# src/analytics/implied_move_calculator.py

class ImpliedMoveCalculator:
    def calculate_implied_move(
        self,
        symbol: str,
        earnings_date: datetime,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Calculate implied earnings move from options
        
        Returns:
        {
            'implied_move_pct': 8.5,
            'implied_move_dollars': 15.60,
            'historical_avg_move': 7.2,
            'iv_percentile': 75,  # Current IV vs historical
            'recommendation': 'elevated' | 'normal' | 'subdued'
        }
        """
        # 1. Find nearest expiration after earnings
        # 2. Get ATM strike
        # 3. Get call + put prices
        # 4. Calculate straddle price
        # 5. Calculate implied move %
        # 6. Compare to historical average
        # 7. Determine if IV is elevated/subdued
        pass
    
    def _get_nearest_expiration(
        self,
        earnings_date: datetime,
        expirations: List[str]
    ) -> str:
        """Find nearest expiration after earnings"""
        pass
    
    def _get_atm_strike(
        self,
        current_price: float,
        strikes: List[float]
    ) -> float:
        """Find at-the-money strike"""
        pass
    
    def _get_historical_avg_move(
        self,
        symbol: str
    ) -> float:
        """Get average historical earnings move"""
        # Read from parquet file
        pass
```

---

## 📊 **Component 3: Earnings Strategy Selector**

### **Goal**: Recommend specific strategies based on earnings risk

### **Decision Tree**:
```
IF days_to_earnings < 3:
    IF position_size > 20% of portfolio:
        → CLOSE or REDUCE
    ELIF implied_move > historical_avg * 1.5:
        → HEDGE with protective puts
    ELSE:
        → HOLD with tight stop
        
ELIF days_to_earnings < 7:
    IF implied_move > historical_avg * 1.3:
        → REDUCE 25-50%
    ELIF position_size > 30% of portfolio:
        → REDUCE to 20%
    ELSE:
        → HOLD or HEDGE
        
ELIF days_to_earnings < 14:
    IF implied_move > historical_avg * 1.2:
        → CONSIDER HEDGING
    ELSE:
        → MONITOR
        
ELSE:
    → NO EARNINGS RISK
```

### **Strategies**:
1. **CLOSE**: Exit entire position before earnings
2. **REDUCE**: Trim 25-50% of position
3. **HEDGE**: Buy protective puts
4. **HOLD_TIGHT**: Hold with tight stop loss
5. **PLAY_VOLATILITY**: Sell premium (iron condor, straddle)
6. **MONITOR**: Watch closely, no action yet

### **Implementation**:
```python
# src/analytics/earnings_strategy_selector.py

class EarningsStrategySelector:
    def select_strategy(
        self,
        symbol: str,
        position: Dict,
        earnings_date: datetime,
        implied_move: float,
        historical_avg_move: float
    ) -> Dict[str, Any]:
        """
        Select optimal earnings strategy
        
        Returns:
        {
            'strategy': 'REDUCE',
            'actions': [
                {'action': 'SELL', 'quantity': 50, 'reason': '...'},
                {'action': 'SET_STOP', 'price': 175, 'reason': '...'}
            ],
            'reasoning': '...',
            'risk_level': 'high' | 'moderate' | 'low'
        }
        """
        pass
```

---

## 📊 **Component 4: Enhanced Correlation with Headlines**

### **Goal**: Search headlines for correlated stocks and detect emerging trends

### **Workflow**:
```
1. Get correlated stocks (e.g., NVDA → AMD, INTC, TSM)
2. Search headlines for each correlated stock (last 7 days)
3. Extract themes with LLM (GPT-4 or Claude)
4. Detect emerging trends
5. Calculate impact on target symbol
```

### **Implementation**:
```python
# src/analytics/headline_analyzer.py

class HeadlineAnalyzer:
    def __init__(self):
        self.firecrawl_api_key = os.getenv('FIRECRAWL_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
    
    def analyze_correlated_headlines(
        self,
        symbol: str,
        correlated_stocks: List[str],
        lookback_days: int = 7
    ) -> Dict[str, Any]:
        """
        Analyze headlines for correlated stocks
        
        Returns:
        {
            'emerging_trends': [
                {
                    'theme': 'AI chip demand surge',
                    'stocks': ['AMD', 'NVDA'],
                    'sentiment': 'positive',
                    'impact_on_target': 'bullish',
                    'confidence': 0.85
                }
            ],
            'sector_themes': ['AI', 'data center', 'gaming'],
            'overall_sentiment': 'bullish',
            'impact_score': 75  # 0-100
        }
        """
        pass
    
    def _search_headlines(
        self,
        symbol: str,
        lookback_days: int
    ) -> List[Dict]:
        """Search headlines using Firecrawl"""
        pass
    
    def _extract_themes_with_llm(
        self,
        headlines: List[Dict]
    ) -> List[str]:
        """Extract themes using GPT-4/Claude"""
        prompt = f"""
        Analyze these headlines and extract key themes:
        
        {headlines}
        
        Return a list of themes (max 5) that are emerging across these headlines.
        Focus on themes that would impact stock prices.
        """
        pass
    
    def _detect_emerging_trends(
        self,
        themes_by_stock: Dict[str, List[str]]
    ) -> List[Dict]:
        """Detect trends emerging across multiple stocks"""
        # Find common themes
        # Identify leading indicators
        # Calculate impact
        pass
```

---

## 📊 **Component 5: Position Sizing (Kelly Criterion)**

### **Goal**: Calculate optimal position size based on win probability

### **Kelly Formula**:
```
f* = (p * b - q) / b

Where:
- f* = fraction of capital to bet
- p = probability of winning
- q = probability of losing (1 - p)
- b = win/loss ratio (avg_win / avg_loss)
```

### **Implementation**:
```python
# src/analytics/position_sizer.py

class PositionSizer:
    def calculate_optimal_size(
        self,
        symbol: str,
        win_probability: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        portfolio_value: float,
        current_price: float,
        max_position_pct: float = 0.20  # Max 20% of portfolio
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size using Kelly Criterion
        
        Returns:
        {
            'kelly_fraction': 0.15,
            'conservative_kelly': 0.075,  # Half Kelly
            'suggested_shares': 50,
            'suggested_value': 9158.00,
            'position_pct': 7.5,
            'reasoning': '...'
        }
        """
        # Calculate Kelly fraction
        b = avg_win_pct / avg_loss_pct
        q = 1 - win_probability
        f_kelly = (win_probability * b - q) / b
        
        # Apply conservative Kelly (half Kelly)
        f_conservative = f_kelly / 2
        
        # Respect maximum position limit
        f_final = min(f_conservative, max_position_pct)
        
        # Calculate shares
        position_value = portfolio_value * f_final
        shares = int(position_value / current_price)
        
        return {
            'kelly_fraction': f_kelly,
            'conservative_kelly': f_conservative,
            'suggested_shares': shares,
            'suggested_value': shares * current_price,
            'position_pct': f_final * 100,
            'reasoning': f"Kelly: {f_kelly:.2%}, Conservative: {f_conservative:.2%}"
        }
    
    def estimate_win_probability(
        self,
        symbol: str,
        recommendation_score: float
    ) -> float:
        """
        Estimate win probability from recommendation score
        
        Score 80-100 → 70-80% win probability
        Score 60-79 → 55-70% win probability
        Score 40-59 → 45-55% win probability
        Score 20-39 → 30-45% win probability
        Score 0-19 → 20-30% win probability
        """
        pass
```

---

## 📋 **Implementation Timeline**

### **Week 1: Data Collection**
- Day 1-2: Build EarningsDataCollector
- Day 3-4: Collect data for top 100 stocks
- Day 5: Validate and clean data

### **Week 2: Implied Move & Strategy**
- Day 1-2: Build ImpliedMoveCalculator
- Day 3-4: Build EarningsStrategySelector
- Day 5: Test and validate

### **Week 3: Headlines & Position Sizing**
- Day 1-2: Build HeadlineAnalyzer
- Day 3-4: Build PositionSizer
- Day 5: Integration and testing

---

## 🎯 **Success Criteria**

- ✅ Historical earnings data for 100+ stocks (8 quarters each)
- ✅ Implied move calculator working with real options data
- ✅ Earnings strategy recommendations for all scenarios
- ✅ Headline search for correlated stocks
- ✅ LLM-powered trend detection
- ✅ Kelly Criterion position sizing
- ✅ All components integrated into recommendation engine
- ✅ API endpoints for all new features
- ✅ Comprehensive testing

---

## 📁 **Files to Create**

1. `src/services/earnings_data_collector.py`
2. `src/analytics/implied_move_calculator.py`
3. `src/analytics/earnings_strategy_selector.py`
4. `src/analytics/headline_analyzer.py`
5. `src/analytics/position_sizer.py`
6. `scripts/collect_earnings_data.py`
7. `test_earnings_intelligence.py`

---

**Ready to start implementation!** 🚀

