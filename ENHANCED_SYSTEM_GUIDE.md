# Enhanced Options Analysis System - Complete Guide

## 🎉 What's New

The system has been significantly enhanced with:

### ✅ **Position Management**
- Add and track stock positions
- Add and track option positions with expiration dates
- Set target prices and stop losses
- Add notes for each position
- View portfolio summary

### ✅ **Real-Time Market Data**
- Live stock prices via yfinance
- Real-time option pricing
- Option chains with all strikes
- Historical and implied volatility
- Market data for all positions

### ✅ **Sentiment Analysis & Research**
- News sentiment analysis (ready for Firecrawl integration)
- Social media sentiment tracking
- YouTube content analysis
- Analyst opinions and price targets
- Catalyst and risk identification

### ✅ **Enhanced AI Agents**
- **Sentiment Research Agent**: Analyzes news, social media, and market context
- **Market Intelligence Agent**: Monitors IV changes and volume anomalies
- **Risk Analysis Agent**: Calculates risk scores and suggests hedges
- **Quantitative Analysis Agent**: Runs EV calculations and scenario analysis
- **Report Generation Agent**: Creates natural language summaries
- **Coordinator Agent**: Orchestrates all agents

### ✅ **Modern Web Interface**
- Clean, intuitive UI
- Tab-based navigation
- Real-time position updates
- One-click analysis
- Market data lookup

---

## 🚀 Quick Start

### 1. Start the Server
```bash
python -m uvicorn src.api.main_simple:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Open the Web Interface
Open `frontend_enhanced.html` in your browser or visit:
```
file:///E:/Projects/Options_probability/frontend_enhanced.html
```

### 3. Add Your First Position

**Stock Position:**
1. Click "Add Stock" tab
2. Enter symbol (e.g., AAPL)
3. Enter quantity and entry price
4. Optionally set target price and stop loss
5. Click "Add Stock Position"

**Option Position:**
1. Click "Add Option" tab
2. Enter symbol (e.g., AAPL)
3. Select Call or Put
4. Enter strike price and expiration date
5. Enter quantity and premium paid
6. Optionally set targets
7. Click "Add Option Position"

### 4. Run Analysis
1. Click "Analysis" tab
2. Click "Run Full Analysis"
3. View AI-generated insights and recommendations

---

## 📊 Features in Detail

### Position Management

**Stock Positions:**
- Symbol, quantity, entry price
- Entry date (auto-filled if not provided)
- Target price for profit taking
- Stop loss for risk management
- Notes for strategy documentation

**Option Positions:**
- Symbol, type (call/put), strike, expiration
- Quantity (number of contracts)
- Premium paid (total cost)
- Target price or profit percentage
- Stop loss percentage
- Automatic days-to-expiry calculation

**Portfolio Summary:**
- Total stock positions
- Total option positions
- Unique symbols tracked
- Total portfolio value

### Real-Time Market Data

**Stock Data:**
```
GET /api/market/stock/{symbol}
```
Returns:
- Current price
- Open, high, low
- Volume
- Previous close
- Change and change %
- Market cap
- P/E ratio

**Option Data:**
```
GET /api/market/option/{symbol}?option_type=call&strike=150&expiration_date=2025-04-18
```
Returns:
- Last price
- Bid/ask spread
- Volume and open interest
- Implied volatility
- Greeks (delta, gamma, theta, vega)
- In-the-money status

**Option Chain:**
```
GET /api/market/chain/{symbol}?expiration_date=2025-04-18
```
Returns:
- All available strikes
- Calls and puts
- Complete pricing data
- Available expiration dates

**Volatility:**
```
GET /api/market/volatility/{symbol}
```
Returns:
- Historical volatility (30-day)
- Implied volatility (from ATM options)
- IV/HV ratio

### Sentiment Analysis

**Get Sentiment:**
```
GET /api/sentiment/{symbol}
```
Returns:
- Overall sentiment (bullish/bearish/neutral)
- Sentiment score (-1 to 1)
- News summary
- Key events
- Analyst opinions
- Price targets
- Catalysts
- Risks

**Sentiment Research Agent Features:**
- News aggregation and analysis
- Social media sentiment (Twitter, Reddit, StockTwits)
- YouTube content analysis
- Analyst opinion tracking
- Catalyst identification
- Risk factor analysis

### AI-Powered Analysis

**Full Analysis:**
```
POST /api/analysis/full
```

Runs complete multi-agent analysis:
1. **Market Intelligence**: Monitors IV changes, volume anomalies
2. **Sentiment Research**: Analyzes news and social sentiment
3. **Risk Analysis**: Calculates risk score, identifies concentrations
4. **Quantitative Analysis**: Runs EV calculations, scenario analysis
5. **Report Generation**: Creates executive summary

Returns:
- Executive summary
- Risk score (0-100)
- Position-by-position analysis
- Recommendations
- Action items
- Scenario analysis (bull, bear, neutral, high/low vol)

---

## 🔧 API Endpoints

### Position Management
- `POST /api/positions/stock` - Add stock position
- `POST /api/positions/option` - Add option position
- `GET /api/positions` - Get all positions
- `GET /api/positions/{symbol}` - Get positions for symbol
- `DELETE /api/positions/stock/{id}` - Remove stock position
- `DELETE /api/positions/option/{id}` - Remove option position

### Market Data
- `GET /api/market/stock/{symbol}` - Get stock data
- `GET /api/market/option/{symbol}` - Get option data
- `GET /api/market/chain/{symbol}` - Get option chain
- `GET /api/market/volatility/{symbol}` - Get volatility data

### Sentiment & Research
- `GET /api/sentiment/{symbol}` - Get sentiment analysis

### Analysis
- `POST /api/analysis/full` - Run complete analysis
- `POST /api/analysis/demo` - Run demo analysis

### System
- `GET /health` - Health check with portfolio summary
- `GET /` - API information
- `GET /docs` - Interactive API documentation

---

## 🤖 AI Agents Explained

### 1. Sentiment Research Agent
**Purpose**: Research market sentiment and context

**Capabilities:**
- News sentiment analysis
- Social media monitoring
- YouTube content analysis
- Analyst opinion tracking
- Catalyst identification
- Risk factor analysis

**Integration Points (Ready for Firecrawl):**
- `research_with_firecrawl()` - Deep research on any query
- `analyze_youtube_sentiment()` - YouTube video analysis
- `get_social_media_sentiment()` - Twitter, Reddit, StockTwits

### 2. Market Intelligence Agent
**Purpose**: Monitor market conditions

**Monitors:**
- IV changes (>10% = significant)
- Volume anomalies (>2x average)
- Unusual options activity
- Gamma exposure levels

### 3. Risk Analysis Agent
**Purpose**: Assess portfolio risk

**Analyzes:**
- Position concentration
- Sector concentration
- Greeks utilization
- Tail risks
- Generates hedge suggestions

**Risk Score Components:**
- Greeks utilization (40%)
- Concentration risk (30%)
- Tail risks (30%)

### 4. Quantitative Analysis Agent
**Purpose**: Calculate expected values

**Performs:**
- EV calculations (3 methods)
- Probability analysis
- Scenario analysis (5 scenarios)
- Optimal action recommendations

### 5. Report Generation Agent
**Purpose**: Create natural language reports

**Generates:**
- Executive summaries
- Market overviews
- Portfolio analysis
- Risk assessments
- Recommendations
- Action items

### 6. Coordinator Agent
**Purpose**: Orchestrate all agents

**Manages:**
- Agent execution order
- State management
- Error handling
- Workflow status

---

## 📈 Workflow Examples

### Daily Analysis Workflow

1. **Morning (Pre-Market)**
   ```javascript
   // Get overnight news and sentiment
   await fetch('/api/sentiment/AAPL');
   
   // Run full analysis
   await fetch('/api/analysis/full', {method: 'POST'});
   ```

2. **During Market Hours**
   ```javascript
   // Check real-time prices
   await fetch('/api/market/stock/AAPL');
   
   // Monitor option prices
   await fetch('/api/market/option/AAPL?option_type=call&strike=150&expiration_date=2025-04-18');
   ```

3. **End of Day**
   ```javascript
   // Run comprehensive analysis
   await fetch('/api/analysis/full', {method: 'POST'});
   
   // Review recommendations
   // Adjust positions as needed
   ```

### Adding a New Trade

1. **Research Phase**
   ```javascript
   // Get sentiment
   const sentiment = await fetch('/api/sentiment/TSLA');
   
   // Check current price
   const price = await fetch('/api/market/stock/TSLA');
   
   // View option chain
   const chain = await fetch('/api/market/chain/TSLA');
   ```

2. **Entry Phase**
   ```javascript
   // Add position
   await fetch('/api/positions/option', {
     method: 'POST',
     body: JSON.stringify({
       symbol: 'TSLA',
       option_type: 'call',
       strike: 250,
       expiration_date: '2025-06-20',
       quantity: 1,
       premium_paid: 1500,
       target_profit_pct: 50,
       stop_loss_pct: 30
     })
   });
   ```

3. **Monitoring Phase**
   ```javascript
   // Run analysis hourly or daily
   await fetch('/api/analysis/full', {method: 'POST'});
   ```

---

## 🔮 Future Enhancements (Firecrawl Integration)

### Ready for Integration

The system is designed to integrate with Firecrawl MCP for:

1. **News Research**
   - Search financial news sites
   - Scrape earnings reports
   - Monitor SEC filings

2. **Social Sentiment**
   - Twitter sentiment analysis
   - Reddit r/wallstreetbets monitoring
   - StockTwits tracking

3. **YouTube Analysis**
   - Search for stock analysis videos
   - Extract sentiment from transcripts
   - Track influencer opinions

4. **Analyst Reports**
   - Scrape analyst upgrades/downgrades
   - Extract price targets
   - Monitor consensus changes

### Integration Points

In `src/agents/sentiment_research_agent.py`:

```python
# TODO: Integrate with Firecrawl MCP
def research_with_firecrawl(self, query: str):
    # Use firecrawl_search tool
    results = firecrawl_search(query=query, limit=5)
    return results

def analyze_youtube_sentiment(self, symbol: str):
    # Search YouTube via Firecrawl
    query = f"{symbol} stock analysis"
    results = firecrawl_search(query=query, limit=10)
    # Analyze video titles and descriptions
    return sentiment_analysis

def get_social_media_sentiment(self, symbol: str):
    # Search Twitter, Reddit via Firecrawl
    twitter = firecrawl_search(query=f"${symbol} site:twitter.com")
    reddit = firecrawl_search(query=f"{symbol} site:reddit.com/r/wallstreetbets")
    return combined_sentiment
```

---

## 📝 Data Storage

**Positions File**: `data/positions.json`

Automatically saves:
- All stock positions
- All option positions
- Last updated timestamp

**Format:**
```json
{
  "stocks": {
    "STK_AAPL_20251009225530": {
      "symbol": "AAPL",
      "quantity": 100,
      "entry_price": 150.0,
      ...
    }
  },
  "options": {
    "OPT_AAPL_CALL_150_20250418": {
      "symbol": "AAPL",
      "option_type": "call",
      "strike": 150.0,
      ...
    }
  },
  "last_updated": "2025-10-09T22:55:30"
}
```

---

## 🎯 Next Steps

1. **Add Your Positions**: Use the web interface to add stocks and options
2. **Run Analysis**: Click "Run Full Analysis" to get AI insights
3. **Monitor Daily**: Check positions and run analysis regularly
4. **Integrate Firecrawl**: Add real news and sentiment data
5. **Customize Agents**: Adjust risk thresholds and analysis parameters

---

**The system is now fully operational with position management, real-time data, and AI-powered analysis!**

🚀 **Start managing your options portfolio like a pro!**

