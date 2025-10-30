# Detailed Implementation Plan - Recommendation System

## 🎯 **Vision: World-Class Recommendation Engine**

Build a system that ranks at the top on all metrics:
- ✅ **Accuracy**: >70% recommendation success rate
- ✅ **Speed**: <1 second response time
- ✅ **Depth**: Multi-factor analysis with correlation detection
- ✅ **Intelligence**: Emerging trend detection from correlated stocks
- ✅ **Robustness**: Handles edge cases, missing data, API failures
- ✅ **Explainability**: Clear reasoning for every recommendation

---

## 📋 **Phase 1: Core Recommendation Engine** (Starting Now)

### **1.1 Scoring Framework** ⭐ Priority 1

**Components**:
```python
class RecommendationEngine:
    """
    Multi-factor scoring system with correlation awareness
    """
    
    def __init__(self):
        self.scorers = {
            'technical': TechnicalScorer(),
            'fundamental': FundamentalScorer(),
            'sentiment': SentimentScorer(),
            'risk': RiskScorer(),
            'earnings': EarningsRiskScorer(),
            'correlation': CorrelationScorer()  # NEW: Detect sector trends
        }
        
        # Weights (sum to 1.0)
        self.weights = {
            'technical': 0.25,
            'fundamental': 0.20,
            'sentiment': 0.20,
            'risk': 0.15,
            'earnings': 0.10,
            'correlation': 0.10  # NEW: Correlation signals
        }
```

**Key Innovation**: **Correlation-Aware Scoring**
- Don't just analyze NVDA in isolation
- Check AMD, INTC, TSM, AVGO (chip sector)
- If AMD drops 5% on news, NVDA likely affected
- If sector sentiment shifts, adjust NVDA score

---

### **1.2 Technical Scorer** (Build First)

**Indicators to Implement**:
```python
class TechnicalScorer:
    """
    Calculate technical score (0-100) from multiple indicators
    """
    
    def calculate_score(self, symbol: str) -> Dict[str, Any]:
        """
        Combine multiple technical indicators
        
        Returns:
        {
            'score': 72,  # 0-100
            'trend': 'bullish',  # bullish/neutral/bearish
            'strength': 'moderate',  # weak/moderate/strong
            'signals': {
                'ma_cross': 'bullish',  # 50 MA > 200 MA
                'rsi': 'neutral',  # RSI 45-55
                'macd': 'bullish',  # MACD > signal
                'volume': 'strong'  # Above average
            },
            'support_resistance': {
                'support': [175.00, 170.00, 165.00],
                'resistance': [185.00, 190.00, 195.00],
                'current': 183.16
            }
        }
        """
        
        # Get price data
        data = self._fetch_price_data(symbol, period='6mo')
        
        # Calculate indicators
        ma_score = self._moving_average_score(data)
        momentum_score = self._momentum_score(data)
        volume_score = self._volume_score(data)
        sr_score = self._support_resistance_score(data)
        
        # Weighted combination
        total_score = (
            ma_score * 0.30 +
            momentum_score * 0.30 +
            volume_score * 0.20 +
            sr_score * 0.20
        )
        
        return {
            'score': total_score,
            'components': {
                'ma': ma_score,
                'momentum': momentum_score,
                'volume': volume_score,
                'support_resistance': sr_score
            }
        }
```

**Indicators**:
1. **Moving Averages** (30% weight)
   - 20/50/200 day SMA
   - Golden cross / Death cross
   - Price vs. MA position

2. **Momentum** (30% weight)
   - RSI (14-day)
   - MACD
   - Rate of change

3. **Volume** (20% weight)
   - Volume trend
   - Volume vs. average
   - On-balance volume (OBV)

4. **Support/Resistance** (20% weight)
   - Distance from support
   - Distance from resistance
   - Breakout detection

---

### **1.3 Sentiment Scorer** (Enhanced with Correlation)

**Key Innovation**: **Correlated Stock Sentiment**
```python
class SentimentScorer:
    """
    Aggregate sentiment from multiple sources + correlated stocks
    """
    
    def calculate_score(self, symbol: str) -> Dict[str, Any]:
        """
        Multi-source sentiment with correlation awareness
        
        Process:
        1. Get direct sentiment (news, social, analysts)
        2. Get correlated stocks (sector peers)
        3. Analyze correlated stock sentiment
        4. Detect emerging trends (sentiment shifts in peers)
        5. Combine with weights
        """
        
        # Direct sentiment
        direct_sentiment = self._get_direct_sentiment(symbol)
        
        # Correlated stocks sentiment
        correlated_stocks = self._get_correlated_stocks(symbol)
        correlated_sentiment = self._analyze_correlated_sentiment(
            correlated_stocks
        )
        
        # Emerging trends (NEW!)
        emerging_trends = self._detect_emerging_trends(
            symbol, 
            correlated_stocks
        )
        
        # Combine
        score = (
            direct_sentiment['score'] * 0.60 +
            correlated_sentiment['score'] * 0.25 +
            emerging_trends['score'] * 0.15
        )
        
        return {
            'score': score,
            'direct': direct_sentiment,
            'correlated': correlated_sentiment,
            'emerging_trends': emerging_trends,
            'reasoning': self._generate_reasoning(...)
        }
```

**Emerging Trend Detection**:
```python
def _detect_emerging_trends(self, symbol: str, correlated_stocks: List[str]):
    """
    Detect trends bubbling up from correlated stocks
    
    Examples:
    - AMD announces strong AI chip sales → NVDA likely benefits
    - INTC warns on margins → Sector-wide concern
    - TSM raises prices → Positive for fabless (NVDA, AMD)
    - AVGO beats on AI networking → Validates AI demand
    
    Process:
    1. Get recent news for all correlated stocks (last 7 days)
    2. Extract key themes (AI demand, pricing, margins, competition)
    3. Detect sentiment shifts (improving/deteriorating)
    4. Calculate impact on target symbol
    5. Assign trend score (0-100)
    """
    
    trends = []
    
    for peer in correlated_stocks:
        # Get recent news
        news = self._get_recent_news(peer, days=7)
        
        # Extract themes with LLM
        themes = self._extract_themes_with_llm(news)
        
        # Detect sentiment shift
        shift = self._detect_sentiment_shift(peer, days=7)
        
        # Calculate impact on target
        impact = self._calculate_impact(peer, symbol, themes, shift)
        
        if impact['score'] > 60:  # Significant trend
            trends.append({
                'peer': peer,
                'theme': themes['primary'],
                'sentiment': shift['direction'],
                'impact_score': impact['score'],
                'reasoning': impact['reasoning']
            })
    
    return {
        'score': self._aggregate_trend_scores(trends),
        'trends': trends,
        'summary': self._summarize_trends(trends)
    }
```

---

### **1.4 Correlation Analyzer** (NEW - Critical Component)

**Purpose**: Identify correlated stocks and analyze their signals

```python
class CorrelationAnalyzer:
    """
    Analyze stock correlations and detect sector trends
    """
    
    def get_correlated_stocks(
        self, 
        symbol: str, 
        min_correlation: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Find stocks correlated with target symbol
        
        Process:
        1. Get sector/industry peers
        2. Calculate price correlation (90-day)
        3. Calculate sentiment correlation
        4. Rank by combined correlation
        5. Return top N correlated stocks
        
        Returns:
        [
            {
                'symbol': 'AMD',
                'correlation': 0.85,
                'sector': 'Semiconductors',
                'relationship': 'direct_competitor',
                'impact_weight': 0.30
            },
            {
                'symbol': 'TSM',
                'correlation': 0.72,
                'sector': 'Semiconductors',
                'relationship': 'supplier',
                'impact_weight': 0.20
            },
            ...
        ]
        """
        
        # Get sector peers
        peers = self._get_sector_peers(symbol)
        
        # Calculate correlations
        correlations = []
        for peer in peers:
            price_corr = self._calculate_price_correlation(symbol, peer)
            sentiment_corr = self._calculate_sentiment_correlation(symbol, peer)
            
            # Combined correlation
            combined = (price_corr * 0.6 + sentiment_corr * 0.4)
            
            if combined >= min_correlation:
                correlations.append({
                    'symbol': peer,
                    'correlation': combined,
                    'price_correlation': price_corr,
                    'sentiment_correlation': sentiment_corr,
                    'relationship': self._determine_relationship(symbol, peer),
                    'impact_weight': self._calculate_impact_weight(combined)
                })
        
        # Sort by correlation
        correlations.sort(key=lambda x: x['correlation'], reverse=True)
        
        return correlations[:10]  # Top 10
    
    def analyze_correlated_signals(
        self, 
        symbol: str, 
        correlated_stocks: List[Dict]
    ) -> Dict[str, Any]:
        """
        Analyze signals from correlated stocks
        
        Returns:
        {
            'sector_trend': 'bullish',  # bullish/neutral/bearish
            'sector_strength': 0.75,  # 0-1
            'divergence': False,  # True if symbol diverges from sector
            'leading_indicators': [
                {
                    'symbol': 'AMD',
                    'signal': 'bullish',
                    'strength': 0.80,
                    'reason': 'Strong earnings beat, raised guidance'
                }
            ],
            'risk_signals': [
                {
                    'symbol': 'INTC',
                    'signal': 'bearish',
                    'strength': 0.60,
                    'reason': 'Margin pressure, competitive losses'
                }
            ]
        }
        """
        pass
```

**Correlation Categories**:
1. **Direct Competitors**: AMD, INTC (high impact)
2. **Suppliers**: TSM, ASML (medium impact)
3. **Customers**: MSFT, GOOGL, META (medium impact)
4. **Sector Peers**: AVGO, QCOM, MU (medium impact)
5. **Market Leaders**: SPY, QQQ (low impact)

---

### **1.5 Headline Search for Correlated Stocks**

**Purpose**: Search news headlines for correlated stocks to detect emerging trends

```python
class HeadlineAnalyzer:
    """
    Search and analyze headlines for correlated stocks
    """
    
    def search_correlated_headlines(
        self, 
        symbol: str, 
        correlated_stocks: List[str],
        lookback_days: int = 7
    ) -> Dict[str, Any]:
        """
        Search headlines for all correlated stocks
        
        Process:
        1. For each correlated stock:
           - Search news headlines (Firecrawl)
           - Extract key themes
           - Detect sentiment
           - Identify catalysts
        
        2. Aggregate findings:
           - Common themes across sector
           - Divergent signals
           - Emerging trends
           - Risk factors
        
        3. Calculate impact on target symbol
        
        Returns:
        {
            'sector_themes': [
                {
                    'theme': 'AI chip demand',
                    'sentiment': 'positive',
                    'mentions': 15,
                    'stocks': ['NVDA', 'AMD', 'AVGO'],
                    'impact': 'bullish'
                }
            ],
            'individual_catalysts': [
                {
                    'symbol': 'AMD',
                    'headline': 'AMD announces new AI chip',
                    'sentiment': 'positive',
                    'impact_on_nvda': 'neutral_to_positive'
                }
            ],
            'risk_factors': [
                {
                    'theme': 'China export restrictions',
                    'sentiment': 'negative',
                    'stocks': ['NVDA', 'AMD', 'INTC'],
                    'impact': 'bearish'
                }
            ]
        }
        """
        
        # Search headlines for each stock
        all_headlines = {}
        for stock in [symbol] + correlated_stocks:
            headlines = self._search_headlines(stock, lookback_days)
            all_headlines[stock] = headlines
        
        # Extract themes with LLM
        themes = self._extract_sector_themes(all_headlines)
        
        # Detect emerging trends
        emerging = self._detect_emerging_from_headlines(all_headlines)
        
        # Identify risk factors
        risks = self._identify_risk_factors(all_headlines)
        
        return {
            'sector_themes': themes,
            'emerging_trends': emerging,
            'risk_factors': risks,
            'impact_summary': self._summarize_impact(themes, emerging, risks)
        }
    
    def _search_headlines(self, symbol: str, days: int) -> List[Dict]:
        """
        Search headlines using Firecrawl
        
        Sources:
        - Google News
        - Yahoo Finance
        - MarketWatch
        - Seeking Alpha
        - Bloomberg
        - Reuters
        """
        
        # Use Firecrawl to search
        query = f"{symbol} stock news"
        results = self.firecrawl.search(
            query=query,
            time_range=f"last_{days}_days",
            sources=['news', 'finance']
        )
        
        return results
    
    def _extract_sector_themes(self, headlines: Dict) -> List[Dict]:
        """
        Use LLM to extract common themes across headlines
        
        Prompt:
        "Analyze these headlines for semiconductor stocks and identify:
        1. Common themes (AI demand, pricing, competition, regulation)
        2. Sentiment for each theme
        3. Which stocks are mentioned for each theme
        4. Overall sector trend
        
        Headlines:
        {headlines}
        
        Return structured JSON."
        """
        
        # Call LLM (GPT-4 or Claude)
        prompt = self._build_theme_extraction_prompt(headlines)
        response = self.llm.call(prompt)
        
        return response['themes']
```

---

### **1.6 Action Generator**

**Purpose**: Convert scores into specific, actionable recommendations

```python
class ActionGenerator:
    """
    Generate specific trade recommendations from scores
    """
    
    def generate_actions(
        self,
        symbol: str,
        position: Dict,
        scores: Dict,
        market_data: Dict
    ) -> Dict[str, Any]:
        """
        Generate specific actions based on scores
        
        Returns:
        {
            'recommendation': 'HOLD / TRIM',
            'confidence': 85,
            'actions': [
                {
                    'action': 'SELL',
                    'instrument': 'stock',
                    'quantity': 51,
                    'price_target': 'market',
                    'reason': 'Take partial profits',
                    'priority': 1,
                    'expected_proceeds': 9341
                }
            ],
            'reasoning': 'Detailed explanation...',
            'risk_factors': ['High valuation', 'Sector rotation risk'],
            'catalysts': ['AI demand growth', 'New product launches']
        }
        """
        
        # Determine recommendation
        recommendation = self._get_recommendation(
            scores['combined'],
            position['status']
        )
        
        # Generate specific actions
        actions = []
        
        if recommendation in ['REDUCE POSITION', 'HOLD / TRIM']:
            # Calculate how much to sell
            sell_pct = self._calculate_trim_percentage(scores)
            sell_qty = int(position['quantity'] * sell_pct)
            
            actions.append({
                'action': 'SELL',
                'instrument': 'stock',
                'quantity': sell_qty,
                'reason': self._explain_trim_reason(scores),
                'priority': 1
            })
        
        # Add stop loss if needed
        if not position.get('stop_loss'):
            stop_price = self._calculate_stop_loss(
                market_data['current_price'],
                scores['risk']
            )
            actions.append({
                'action': 'SET_STOP',
                'price': stop_price,
                'reason': 'Protect downside',
                'priority': 2
            })
        
        return {
            'recommendation': recommendation,
            'confidence': self._calculate_confidence(scores),
            'actions': actions,
            'reasoning': self._generate_reasoning(scores, position),
            'expected_outcome': self._project_outcome(actions, position)
        }
```

---

## 📋 **Phase 2: Earnings Intelligence** (Next)

### **2.1 Historical Earnings Data Collection**

**Data Structure**:
```python
{
    'symbol': 'NVDA',
    'earnings_history': [
        {
            'date': '2024-08-28',
            'quarter': 'Q2 2025',
            'fiscal_year': 2025,
            'eps_estimate': 0.64,
            'eps_actual': 0.68,
            'eps_surprise_pct': 6.25,
            'revenue_estimate': 28.7e9,
            'revenue_actual': 30.0e9,
            'revenue_surprise_pct': 4.53,
            'guidance': 'raised',
            'guidance_details': {
                'next_quarter_revenue': '32.5B',
                'next_quarter_margin': '75%'
            },
            'price_before': 120.00,
            'price_after_1d': 108.00,
            'price_after_1w': 115.00,
            'price_after_1m': 125.00,
            'move_pct_1d': -10.0,
            'move_pct_1w': -4.2,
            'move_pct_1m': 4.2,
            'time': 'AMC',
            'beat_or_miss': 'beat',
            'analyst_reactions': [
                {'firm': 'Goldman', 'action': 'upgrade', 'target': 135},
                {'firm': 'Morgan Stanley', 'action': 'maintain', 'target': 125}
            ]
        },
        # ... last 12 quarters
    ],
    'statistics': {
        'avg_move_pct': 8.5,
        'median_move_pct': 7.2,
        'std_dev': 3.1,
        'max_up': 15.2,
        'max_down': -12.8,
        'beat_avg_move': 9.8,
        'miss_avg_move': -11.2,
        'beat_rate': 0.83,  # 83% beat rate
        'guidance_raise_rate': 0.75
    }
}
```

**Data Sources**:
1. **Finnhub**: `/stock/earnings` - Historical earnings
2. **Alpha Vantage**: `EARNINGS` - EPS/Revenue data
3. **Firecrawl**: Scrape from:
   - Earnings Whispers
   - Yahoo Finance earnings calendar
   - MarketWatch earnings
4. **Manual CSV**: Backup historical data

**Collection Script**:
```python
class EarningsDataCollector:
    """
    Collect historical earnings data for top stocks
    """
    
    def collect_earnings_history(
        self,
        symbols: List[str],
        quarters: int = 12
    ) -> Dict[str, Any]:
        """
        Collect last N quarters of earnings data
        
        Process:
        1. Try Finnhub first
        2. Fallback to Alpha Vantage
        3. Supplement with Firecrawl (analyst reactions, guidance)
        4. Calculate statistics
        5. Cache to Parquet
        """
        pass
```

---

### **2.2 Implied Move Calculator**

```python
class ImpliedMoveCalculator:
    """
    Calculate implied move from options
    """
    
    def calculate_implied_move(
        self,
        symbol: str,
        earnings_date: str
    ) -> Dict[str, Any]:
        """
        Calculate implied move from ATM straddle
        
        Process:
        1. Get options chain for earnings week
        2. Find ATM strike (closest to current price)
        3. Get ATM call + put prices
        4. Calculate straddle price
        5. Implied move = (straddle / stock price) * 100
        6. Calculate upper/lower ranges
        7. Compare to historical moves
        
        Returns:
        {
            'stock_price': 183.16,
            'atm_strike': 185,
            'call_price': 8.50,
            'put_price': 7.00,
            'straddle_price': 15.50,
            'implied_move_pct': 8.5,
            'implied_move_dollars': 15.57,
            'upper_range': 198.73,
            'lower_range': 167.59,
            'vs_historical': 'in-line',  # or 'elevated' or 'subdued'
            'iv_rank': 75,
            'confidence': 'high'  # based on volume/OI
        }
        """
        pass
```

---

### **2.3 Earnings Strategy Selector**

```python
class EarningsStrategySelector:
    """
    Recommend earnings-specific strategies
    """
    
    def recommend_strategy(
        self,
        symbol: str,
        position: Dict,
        earnings_analysis: Dict
    ) -> Dict[str, Any]:
        """
        Recommend strategy based on earnings risk
        
        Decision Tree:
        
        IF days_to_earnings < 7:
            IF position_size > 20% of portfolio:
                → REDUCE POSITION (high risk)
            ELIF implied_move > historical_avg * 1.2:
                → HEDGE WITH OPTIONS (elevated IV)
            ELIF beat_rate > 0.80 AND guidance_raise_rate > 0.70:
                → HOLD THROUGH (high confidence)
            ELSE:
                → REDUCE 25-50% (moderate risk)
        
        ELIF days_to_earnings < 14:
            IF unrealized_gain > 50%:
                → TAKE PARTIAL PROFITS
            ELSE:
                → MONITOR CLOSELY
        
        Returns:
        {
            'primary_strategy': 'reduce_position',
            'alternatives': ['hedge_with_puts', 'sell_covered_calls'],
            'recommended_actions': [
                {
                    'action': 'SELL',
                    'quantity': 50,
                    'reason': 'Reduce exposure before earnings'
                }
            ],
            'risk_level': 7,
            'expected_move_range': [167.59, 198.73],
            'reasoning': 'Detailed explanation...'
        }
        """
        pass
```

---

## 📋 **Phase 3: Position Sizing** (Kelly Criterion)

### **3.1 Kelly Criterion Implementation**

```python
class PositionSizer:
    """
    Calculate optimal position size using Kelly Criterion
    """
    
    def calculate_optimal_size(
        self,
        symbol: str,
        win_probability: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        portfolio_value: float,
        current_price: float,
        max_position_pct: float = 0.25
    ) -> Dict[str, Any]:
        """
        Kelly Criterion: f* = (p*b - q) / b
        
        where:
        - p = probability of win
        - q = probability of loss (1-p)
        - b = win/loss ratio
        - f* = fraction of portfolio to risk
        
        Returns:
        {
            'kelly_fraction': 0.463,  # 46.3%
            'conservative_fraction': 0.232,  # Half Kelly
            'recommended_fraction': 0.232,  # After applying limits
            'position_value': 23200,
            'shares': 126,
            'reasoning': 'Using half-Kelly for safety...'
        }
        """
        
        # Calculate Kelly fraction
        b = avg_win_pct / avg_loss_pct
        q = 1 - win_probability
        f_kelly = (win_probability * b - q) / b
        
        # Conservative Kelly (half Kelly)
        f_conservative = f_kelly / 2
        
        # Apply maximum position limit
        f_recommended = min(f_conservative, max_position_pct)
        
        # Calculate position size
        position_value = portfolio_value * f_recommended
        shares = int(position_value / current_price)
        
        return {
            'kelly_fraction': f_kelly,
            'conservative_fraction': f_conservative,
            'recommended_fraction': f_recommended,
            'position_value': position_value,
            'shares': shares,
            'current_price': current_price,
            'reasoning': self._explain_sizing(f_kelly, f_conservative, f_recommended)
        }
```

---

## 🎯 **Implementation Timeline**

### **Week 1: Core Recommendation Engine**
- Day 1-2: Technical Scorer + Fundamental Scorer
- Day 3-4: Sentiment Scorer + Correlation Analyzer
- Day 5-6: Action Generator + API Integration
- Day 7: Testing + Refinement

### **Week 2: Earnings Intelligence**
- Day 1-3: Historical data collection (top 100 stocks)
- Day 4-5: Implied move calculator
- Day 6-7: Earnings strategy selector

### **Week 3: Position Sizing + Advanced Features**
- Day 1-2: Kelly Criterion implementation
- Day 3-4: Headline analyzer for correlated stocks
- Day 5-6: Emerging trend detection
- Day 7: Integration + testing

---

## ✅ **Quality Rubric (Top Tier)**

### **Accuracy** (Target: >70%)
- [ ] Backtested on historical data
- [ ] Validated against real positions
- [ ] Confidence calibration (85% confidence = 85% accuracy)

### **Speed** (Target: <1 second)
- [ ] Parallel API calls
- [ ] Efficient caching
- [ ] Optimized calculations

### **Depth** (Target: Multi-factor)
- [ ] 6+ scoring factors
- [ ] Correlation analysis
- [ ] Emerging trend detection
- [ ] Earnings intelligence

### **Robustness** (Target: 99.9% uptime)
- [ ] Handles missing data gracefully
- [ ] API fallbacks
- [ ] Error recovery
- [ ] Comprehensive logging

### **Explainability** (Target: Clear reasoning)
- [ ] Detailed score breakdown
- [ ] Specific action reasons
- [ ] Risk factor identification
- [ ] Catalyst highlighting

---

**Ready to start building! Beginning with Phase 1: Core Recommendation Engine...**

