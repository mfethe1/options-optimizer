# Position Management System - Complete Guide

## 📋 Overview

A comprehensive manual position entry system for stocks and options with:
- ✅ **CSV Import/Export** - Bulk position management
- ✅ **Manual Entry UI** - Professional web interface
- ✅ **Real-time Enrichment** - Auto-calculate Greeks, P&L, metrics
- ✅ **AI Agent Integration** - Positions accessible to recommendation engine
- ✅ **Conversation Memory** - Multi-conversation context for agents

---

## 🏗️ Architecture

### Backend Components

#### 1. **Position Manager** (`src/data/position_manager.py`)
Core position storage and management.

**Features**:
- Stock and option position data structures
- CRUD operations for positions
- JSON-based persistence
- Portfolio summary calculations

**Key Classes**:
- `StockPosition` - Stock position with P&L tracking
- `OptionPosition` - Option position with Greeks and metrics
- `PositionManager` - Main management interface

#### 2. **CSV Position Service** (`src/data/csv_position_service.py`)
Handles CSV import/export with validation.

**Features**:
- Template generation with examples
- Row-by-row validation
- Bulk import with error reporting
- Export current positions

**CSV Format - Stocks**:
```csv
symbol,quantity,entry_price,entry_date,target_price,stop_loss,notes
AAPL,100,150.50,2025-01-15,175.00,140.00,Long-term hold
```

**CSV Format - Options**:
```csv
symbol,option_type,strike,expiration_date,quantity,premium_paid,entry_date,target_price,target_profit_pct,stop_loss_pct,notes
TSLA,call,250.00,2025-12-20,10,15.50,2025-10-01,25.00,50,30,Bullish on EV
```

#### 3. **Position Enrichment Service** (`src/data/position_enrichment_service.py`)
Calculates real-time metrics and Greeks.

**Features**:
- Fetch real-time prices from yfinance
- Calculate option Greeks (Delta, Gamma, Theta, Vega, Rho)
- Compute P&L and percentages
- Calculate IV, HV, probability of profit
- Determine position status and risk level

**Metrics Calculated**:

**For Stocks**:
- Current price, P&L, P&L %
- PE ratio, dividend yield, market cap
- Analyst ratings and targets
- Position status (PROFITABLE, LOSING, etc.)

**For Options**:
- Current option price, underlying price
- All Greeks (Delta, Gamma, Theta, Vega, Rho)
- Implied volatility, historical volatility
- Intrinsic/extrinsic value
- Break-even price, max profit/loss
- Probability of profit
- Days to expiry, risk level

#### 4. **Position Context Service** (`src/agents/position_context_service.py`)
Provides position data to AI agents with conversation memory.

**Features**:
- Conversation history tracking
- Position context formatting for agents
- Multi-conversation support
- Agent interaction logging

**Usage**:
```python
from src.agents.position_context_service import PositionContextService

service = PositionContextService()

# Create context for agent
context = service.create_agent_context(
    conversation_id="user_123_session_1",
    include_summary=True,
    include_positions=True,
    symbol="AAPL",
    include_expiring=True
)

# Log agent interaction
service.log_agent_interaction(
    conversation_id="user_123_session_1",
    user_query="What's my AAPL position looking like?",
    agent_response="Your AAPL position is up 15%...",
    positions_accessed=["STK_AAPL_20251015120000"]
)
```

#### 5. **API Routes** (`src/api/position_routes.py`)
RESTful API endpoints for all position operations.

**Endpoints**:

**Stock Positions**:
- `POST /api/positions/stocks` - Create stock position
- `GET /api/positions/stocks` - List all stock positions
- `GET /api/positions/stocks/{id}` - Get specific stock position
- `PATCH /api/positions/stocks/{id}` - Update stock position
- `DELETE /api/positions/stocks/{id}` - Delete stock position

**Option Positions**:
- `POST /api/positions/options` - Create option position
- `GET /api/positions/options` - List all option positions
- `GET /api/positions/options/{id}` - Get specific option position
- `PATCH /api/positions/options/{id}` - Update option position
- `DELETE /api/positions/options/{id}` - Delete option position

**CSV Operations**:
- `GET /api/positions/templates/stocks` - Download stock template
- `GET /api/positions/templates/options` - Download option template
- `GET /api/positions/export/stocks` - Export stock positions
- `GET /api/positions/export/options` - Export option positions
- `POST /api/positions/import/stocks` - Import stock positions
- `POST /api/positions/import/options` - Import option positions

**Portfolio**:
- `GET /api/positions/summary` - Get portfolio summary
- `POST /api/positions/enrich` - Enrich all positions

### Frontend Components

#### 1. **Position Service** (`frontend/src/services/positionService.ts`)
TypeScript service for API communication.

**Features**:
- Type-safe API calls
- File download handling
- Error handling
- CSV import/export

#### 2. **Positions Page** (`frontend/src/pages/PositionsPage.tsx`)
Main UI for position management.

**Features**:
- Portfolio summary dashboard
- Stock/option position tabs
- Add position dialog
- CSV import/export
- Real-time data refresh
- Position editing and deletion

---

## 🚀 Quick Start

### 1. Download CSV Template

**Via API**:
```bash
# Stock template
curl http://localhost:8000/api/positions/templates/stocks -o stock_template.csv

# Option template
curl http://localhost:8000/api/positions/templates/options -o option_template.csv
```

**Via UI**:
1. Navigate to Positions page
2. Click "Download Template"
3. Choose Stocks or Options

### 2. Fill Out Template

**Stock Example**:
```csv
symbol,quantity,entry_price,entry_date,target_price,stop_loss,notes
AAPL,100,150.50,2025-01-15,175.00,140.00,Long-term hold
NVDA,50,500.00,2025-02-01,600.00,450.00,AI growth play
MSFT,75,380.00,2025-01-20,420.00,360.00,Cloud leader
```

**Option Example**:
```csv
symbol,option_type,strike,expiration_date,quantity,premium_paid,entry_date,target_price,target_profit_pct,stop_loss_pct,notes
TSLA,call,250.00,2025-12-20,10,15.50,2025-10-01,25.00,50,30,Bullish on EV sector
SPY,put,450.00,2025-11-15,5,8.25,2025-10-10,12.00,40,25,Hedge position
AAPL,call,180.00,2025-11-21,20,5.75,2025-10-05,10.00,60,35,Earnings play
```

### 3. Import Positions

**Via API**:
```bash
curl -X POST \
  -F "file=@stock_positions.csv" \
  http://localhost:8000/api/positions/import/stocks
```

**Via UI**:
1. Click "Import CSV"
2. Select file
3. Choose "Replace existing" if needed
4. Click "Import"

### 4. View Enriched Positions

Positions are automatically enriched with:
- Real-time prices
- Greeks (for options)
- P&L calculations
- Risk metrics

---

## 📊 Data Flow

```
User Input (CSV/UI)
    ↓
Position Manager (Storage)
    ↓
Enrichment Service (yfinance + Greeks Calculator)
    ↓
Position Context Service (AI Agent Access)
    ↓
Recommendation Engine / AI Agents
```

---

## 🤖 AI Agent Integration

### How Agents Access Positions

```python
from src.agents.position_context_service import PositionContextService

# Initialize service
context_service = PositionContextService()

# Get context for agent
context = context_service.create_agent_context(
    conversation_id="session_123",
    include_summary=True,      # Portfolio summary
    include_positions=True,    # Detailed positions
    symbol="AAPL",            # Filter by symbol (optional)
    include_expiring=True     # Expiring options
)

# Pass context to agent
agent_response = coordinator.analyze_with_context(
    query="Should I hold or sell my AAPL calls?",
    context=context
)

# Log interaction
context_service.log_agent_interaction(
    conversation_id="session_123",
    user_query="Should I hold or sell my AAPL calls?",
    agent_response=agent_response,
    positions_accessed=["OPT_AAPL_CALL_180_20251121"]
)
```

### Conversation Memory

The system maintains conversation history across sessions:

```python
# Add message to conversation
context_service.conversation_memory.add_message(
    conversation_id="session_123",
    role="user",
    content="What's my portfolio looking like?",
    metadata={"timestamp": "2025-10-16T10:30:00"}
)

# Get conversation history
history = context_service.conversation_memory.get_conversation(
    conversation_id="session_123",
    limit=10  # Last 10 messages
)

# Get recent conversations
recent = context_service.conversation_memory.get_recent_conversations(limit=5)
```

---

## 📈 Position Metrics

### Stock Position Metrics

| Metric | Description | Source |
|--------|-------------|--------|
| Current Price | Real-time stock price | yfinance |
| Unrealized P&L | (Current - Entry) × Quantity | Calculated |
| P&L % | ((Current / Entry) - 1) × 100 | Calculated |
| PE Ratio | Price-to-earnings ratio | yfinance |
| Dividend Yield | Annual dividend yield | yfinance |
| Market Cap | Total market capitalization | yfinance |
| Analyst Target | Average analyst price target | yfinance |
| Status | PROFITABLE, LOSING, TARGET_REACHED, etc. | Calculated |

### Option Position Metrics

| Metric | Description | Source |
|--------|-------------|--------|
| Current Price | Real-time option price | yfinance / Black-Scholes |
| Underlying Price | Current stock price | yfinance |
| Delta | Price sensitivity to underlying | Black-Scholes |
| Gamma | Delta sensitivity | Black-Scholes |
| Theta | Time decay per day | Black-Scholes |
| Vega | IV sensitivity | Black-Scholes |
| Rho | Interest rate sensitivity | Black-Scholes |
| Implied Volatility | Market's volatility expectation | yfinance |
| Historical Volatility | 30-day realized volatility | Calculated |
| IV Rank | IV relative to historical range | Calculated |
| Probability of Profit | Estimated POP | Delta-based |
| Break-even Price | Price needed to break even | Calculated |
| Intrinsic Value | In-the-money value | Calculated |
| Extrinsic Value | Time value | Calculated |
| Days to Expiry | Days until expiration | Calculated |
| Risk Level | LOW, MEDIUM, HIGH, CRITICAL | Calculated |

---

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
API_BASE_URL=http://localhost:8000

# Data Storage
POSITIONS_FILE=data/positions.json
CONVERSATION_MEMORY_FILE=data/conversation_memory.json

# Market Data
YFINANCE_ENABLED=true
RISK_FREE_RATE=0.045  # 4.5% default
```

### File Locations

```
data/
├── positions.json              # Position storage
├── conversation_memory.json    # AI conversation history
└── opt/
    └── chains/                 # Options chain cache
        └── {SYMBOL}/
            └── {DATE}.parquet
```

---

## 📝 Best Practices

### 1. **Regular Updates**
- Update positions weekly or after major trades
- Use "Refresh Data" button to enrich with latest prices
- Export positions regularly as backup

### 2. **CSV Import**
- Always download template first
- Validate data before import
- Use `replace_existing=false` to add to existing positions
- Review import results for errors

### 3. **Position Notes**
- Document trade thesis in notes field
- Include entry/exit criteria
- Note any special circumstances

### 4. **Risk Management**
- Set target prices and stop losses
- Monitor expiring options (7-day alert)
- Review risk levels regularly

### 5. **AI Agent Usage**
- Provide conversation context for better recommendations
- Reference specific positions in queries
- Review agent reasoning before acting

---

## 🎯 Next Steps

1. ✅ **Test the system** - Import sample positions
2. ✅ **Enrich data** - Click "Refresh Data" to calculate metrics
3. ✅ **Review positions** - Check P&L and Greeks
4. ✅ **Ask AI agents** - Get recommendations on your positions
5. ✅ **Export backup** - Download CSV of current positions

---

## 📞 Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review API responses for error details
3. Validate CSV format against templates
4. Ensure yfinance is working (test with `yf.Ticker("AAPL").info`)

---

**Built with**: FastAPI, React, TypeScript, yfinance, Black-Scholes, Multi-Agent AI System

