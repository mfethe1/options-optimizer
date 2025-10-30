# Recommendation Engine - Quick Start Guide

## 🎯 **What Is This?**

A **world-class, multi-factor recommendation engine** that analyzes stocks and provides:
- ✅ **BUY/SELL/HOLD recommendations** with confidence levels
- ✅ **Specific actions** (sell 51 shares, set stop at $175, etc.)
- ✅ **Multi-factor scoring** (technical, fundamental, sentiment, risk, earnings, correlation)
- ✅ **Risk factors and catalysts** identification
- ✅ **Expected outcomes** projection

---

## 🚀 **Quick Start**

### **1. API Endpoint**

```bash
GET /api/recommendations/{symbol}
```

### **2. Example Request**

```bash
curl http://localhost:8000/api/recommendations/NVDA
```

### **3. Example Response**

```json
{
  "symbol": "NVDA",
  "recommendation": "HOLD_TRIM",
  "confidence": 85,
  "combined_score": 68.3,
  "actions": [
    {
      "action": "SELL",
      "quantity": 51,
      "reason": "Take partial profits (25% of position)",
      "expected_proceeds": 9341.16,
      "priority": 1
    },
    {
      "action": "SET_STOP",
      "stop_price": 175.00,
      "reason": "Protect downside",
      "priority": 3
    }
  ],
  "risk_factors": [
    "High volatility",
    "Expensive valuation"
  ],
  "catalysts": [
    "Strong revenue growth",
    "Positive sector trend"
  ]
}
```

---

## 📊 **How It Works**

### **Step 1: Multi-Factor Scoring**

The engine calculates 6 independent scores (0-100):

1. **Technical** (25%): Moving averages, RSI, MACD, volume, support/resistance
2. **Fundamental** (20%): Valuation, growth, profitability, financial health
3. **Sentiment** (20%): News, social media, analysts, sector sentiment
4. **Risk** (15%): Volatility, beta, liquidity (inverted - lower is better)
5. **Earnings** (10%): Time to earnings, historical volatility (inverted)
6. **Correlation** (10%): Sector performance, peer analysis

### **Step 2: Weighted Combination**

```
Combined Score = (
  Technical * 0.25 +
  Fundamental * 0.20 +
  Sentiment * 0.20 +
  (100 - Risk) * 0.15 +
  (100 - Earnings) * 0.10 +
  Correlation * 0.10
)
```

### **Step 3: Recommendation Mapping**

| Score | Recommendation | Action |
|-------|----------------|--------|
| 80-100 | STRONG_HOLD_ADD | Hold + add on dips |
| 65-79 | HOLD | Maintain position |
| 50-64 | HOLD_TRIM | Trim 15-25% |
| 35-49 | REDUCE | Trim 35-50% |
| 0-34 | CLOSE | Exit position |

### **Step 4: Action Generation**

Converts recommendation into specific trades:
- Sell X shares at market
- Set stop loss at $Y
- Set profit target at $Z
- Consider adding on dips

---

## 🔍 **Detailed Scoring**

### **Technical Score (0-100)**

**Indicators**:
- Moving Averages (30%): 20/50/200 SMA, golden/death cross
- Momentum (30%): RSI, MACD, rate of change
- Volume (20%): OBV, volume trends
- Support/Resistance (20%): Key levels, breakouts

**Interpretation**:
- 80-100: Strong bullish
- 60-79: Bullish
- 40-59: Neutral
- 20-39: Bearish
- 0-19: Strong bearish

---

### **Fundamental Score (0-100)**

**Metrics**:
- Valuation (25%): P/E, PEG, P/S, P/B
- Growth (30%): Revenue, earnings, analyst targets
- Profitability (25%): ROE, ROA, margins
- Financial Health (15%): Debt, liquidity, cash flow
- Competitive (5%): Market cap, analyst ratings

**Interpretation**:
- 80-100: Excellent fundamentals
- 60-79: Good fundamentals
- 40-59: Mixed fundamentals
- 20-39: Weak fundamentals
- 0-19: Poor fundamentals

---

### **Sentiment Score (0-100)**

**Sources**:
- Direct (60%): News, social, analysts
- Correlated (25%): Sector peer sentiment
- Emerging (15%): Trend detection

**Interpretation**:
- 80-100: Strong positive
- 60-79: Positive
- 40-59: Neutral
- 20-39: Negative
- 0-19: Strong negative

---

### **Risk Score (0-100, inverted)**

**Factors**:
- Volatility (30%): Historical volatility
- Beta (20%): Market correlation
- Liquidity (15%): Trading volume
- Concentration (20%): Position size
- Correlation (15%): Portfolio correlation

**Interpretation** (after inversion):
- 80-100: Low risk
- 60-79: Moderate risk
- 40-59: Elevated risk
- 20-39: High risk
- 0-19: Very high risk

---

### **Earnings Risk Score (0-100, inverted)**

**Components**:
- Time (25%): Days to earnings
- Volatility (30%): Historical moves
- Implied (25%): Options-implied move
- Estimates (15%): Analyst spread
- Guidance (5%): History

**Interpretation** (after inversion):
- 80-100: Low earnings risk
- 60-79: Moderate earnings risk
- 40-59: Elevated earnings risk
- 20-39: High earnings risk
- 0-19: Very high earnings risk

---

### **Correlation Score (0-100)**

**Analysis**:
- Sector Performance (50%): Peer returns
- Sector Sentiment (30%): Peer ratings
- Divergence (20%): Symbol vs. sector

**Interpretation**:
- 80-100: Strong sector tailwind
- 60-79: Positive sector trend
- 40-59: Neutral sector trend
- 20-39: Negative sector trend
- 0-19: Strong sector headwind

---

## 🎯 **Use Cases**

### **1. Daily Position Review**

```bash
# Check all your positions
curl http://localhost:8000/api/recommendations/NVDA
curl http://localhost:8000/api/recommendations/AAPL
curl http://localhost:8000/api/recommendations/TSLA
```

### **2. New Position Analysis**

```bash
# Evaluate before buying
curl http://localhost:8000/api/recommendations/AMD
```

### **3. Risk Management**

```bash
# Check risk factors
curl http://localhost:8000/api/recommendations/NVDA | jq '.risk_factors'
```

### **4. Sector Analysis**

```bash
# Compare sector peers
curl http://localhost:8000/api/recommendations/NVDA | jq '.scores.correlation'
```

---

## 📁 **File Structure**

```
src/analytics/
├── recommendation_engine.py      # Main orchestrator
├── technical_scorer.py           # Technical analysis
├── fundamental_scorer.py         # Fundamental analysis
├── sentiment_scorer.py           # Sentiment analysis
├── risk_scorer.py                # Risk analysis
├── earnings_risk_scorer.py       # Earnings risk
├── correlation_scorer.py         # Sector correlation
└── action_generator.py           # Action generation
```

---

## 🔧 **Configuration**

### **Adjust Weights**

Edit `src/analytics/recommendation_engine.py`:

```python
self.weights = {
    'technical': 0.25,      # Increase for more technical focus
    'fundamental': 0.20,    # Increase for more fundamental focus
    'sentiment': 0.20,      # Increase for more sentiment focus
    'risk': 0.15,           # Increase for more risk aversion
    'earnings': 0.10,       # Increase for more earnings focus
    'correlation': 0.10     # Increase for more sector focus
}
```

### **Customize Thresholds**

Edit recommendation thresholds in `_get_recommendation()` method.

---

## 🐛 **Troubleshooting**

### **Issue**: "unsupported format string passed to NoneType"

**Cause**: One of the scorers is returning None values

**Fix**: Check that all data sources are available (yfinance, research service, etc.)

### **Issue**: Slow response time

**Cause**: Multiple API calls to yfinance

**Solution**: Implement caching (coming in Phase 2)

### **Issue**: Inaccurate recommendations

**Cause**: Needs calibration with historical data

**Solution**: Backtesting (coming in Phase 3)

---

## 📈 **Roadmap**

### **Phase 1: Core Engine** ✅ (COMPLETE)
- Multi-factor scoring
- Action generation
- API integration

### **Phase 2: Earnings Intelligence** (Next Week)
- Historical earnings data
- Implied move calculator
- Earnings strategy selector

### **Phase 3: Advanced Features** (Week 3)
- Kelly Criterion position sizing
- Dynamic correlation calculation
- LLM-powered trend detection

### **Phase 4: Validation** (Week 4)
- Backtesting framework
- Accuracy measurement
- Confidence calibration

---

## 💡 **Tips**

1. **Use with existing positions**: Most valuable for managing current holdings
2. **Check confidence levels**: Higher confidence = more reliable
3. **Review risk factors**: Understand what could go wrong
4. **Consider catalysts**: Know what's driving the recommendation
5. **Don't blindly follow**: Use as input, not gospel

---

## 📞 **Support**

- **Documentation**: See `PHASE1_COMPLETE_SUMMARY.md`
- **Implementation Details**: See `IMPLEMENTATION_PLAN_DETAILED.md`
- **API Docs**: See `RECOMMENDATION_SYSTEM_ROADMAP.md`

---

## ⚖️ **Disclaimer**

This recommendation engine is for **educational and informational purposes only**. It is **NOT financial advice**. Always:
- Do your own research
- Understand the risks
- Consult a financial advisor
- Never invest more than you can afford to lose

---

**Built with ❤️ using Python, FastAPI, yfinance, and pandas**

**Version**: 1.0.0  
**Status**: Phase 1 Complete  
**Last Updated**: 2025-10-11

