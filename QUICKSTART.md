# Quick Start Guide - Options Analysis System

## 🚀 Get Started in 5 Minutes

This guide will get you up and running with the Options Analysis System quickly.

## Prerequisites

- Python 3.11 or higher
- Node.js 18 or higher
- PostgreSQL 15 or higher (optional for demo)
- Redis 7 or higher (optional for demo)

## Option 1: Run the Demo (No Database Required)

The fastest way to see the system in action:

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Run the demo
python demo/run_demo.py
```

This will demonstrate:
- ✅ Expected Value calculation
- ✅ Greeks calculation
- ✅ Multi-agent analysis workflow
- ✅ Report generation

**Expected Output:**
```
================================================================================
  OPTIONS ANALYSIS SYSTEM - DEMO
  AI-Powered Multi-Agent Options Analysis
================================================================================

================================================================================
  Expected Value Calculation Demo
================================================================================

Position: Long Call @ $150.0
Underlying Price: $155.0
Premium Paid: $500.0

Results:
  Expected Value: $XX.XX
  Expected Return: XX.XX%
  Probability of Profit: XX.X%
  95% Confidence Interval: $XX.XX to $XX.XX

... (more output)

================================================================================
  Demo Complete
================================================================================
✓ Expected Value calculation working
✓ Greeks calculation working
✓ Multi-agent system working
✓ All agents coordinated successfully

The system is ready for production use!
```

## Option 2: Full System Setup

### Step 1: Database Setup

```bash
# Create database
createdb options_analysis

# Run schema
psql -U postgres -d options_analysis -f src/database/schema.sql
```

### Step 2: Backend Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env

# Edit .env and set:
# DATABASE_URL=postgresql://postgres:postgres@localhost:5432/options_analysis
# REDIS_URL=redis://localhost:6379/0

# Run server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: http://localhost:8000

API Documentation: http://localhost:8000/docs

### Step 3: Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.example .env

# Edit .env and set:
# VITE_API_URL=http://localhost:8000/api
# VITE_WS_URL=ws://localhost:8000/ws

# Run development server
npm run dev
```

The frontend will be available at: http://localhost:5173

## Option 3: Run Tests

```bash
# Run all backend tests
pytest tests/ -v

# Run specific test file
pytest tests/test_ev_calculator.py -v
pytest tests/test_agents.py -v
```

**Expected Output:**
```
tests/test_ev_calculator.py::TestEVCalculator::test_calculate_ev_long_call PASSED
tests/test_ev_calculator.py::TestEVCalculator::test_probability_distributions_sum_to_one PASSED
tests/test_ev_calculator.py::TestEVCalculator::test_payoff_calculation PASSED
...
======================== XX passed in X.XXs ========================
```

## Quick API Examples

### 1. Health Check

```bash
curl http://localhost:8000/health
```

### 2. Create a Position

```bash
curl -X POST http://localhost:8000/api/positions \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo-user-1",
    "symbol": "AAPL",
    "strategy_type": "long_call",
    "entry_date": "2025-01-10",
    "expiration_date": "2025-04-18",
    "total_premium": 500.0,
    "legs": [
      {
        "option_type": "call",
        "strike": 150.0,
        "quantity": 1,
        "is_short": false,
        "entry_price": 5.0,
        "multiplier": 100
      }
    ]
  }'
```

### 3. Run Analysis

```bash
curl -X POST http://localhost:8000/api/analysis/run \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo-user-1",
    "report_type": "daily"
  }'
```

### 4. Get Reports

```bash
curl http://localhost:8000/api/reports?user_id=demo-user-1&limit=5
```

## Understanding the Output

### EV Calculation Result

```json
{
  "expected_value": 125.50,
  "expected_return_pct": 25.1,
  "probability_profit": 0.65,
  "confidence_interval": [-200.0, 800.0],
  "method_breakdown": {
    "black_scholes": 120.0,
    "risk_neutral_density": 130.0,
    "monte_carlo": 126.0
  }
}
```

- **expected_value**: Net expected profit/loss
- **expected_return_pct**: Return as percentage of premium
- **probability_profit**: Chance of making any profit (0-1)
- **confidence_interval**: 95% confidence range
- **method_breakdown**: EV from each probability method

### Greeks Result

```json
{
  "delta": 0.6234,
  "gamma": 0.0123,
  "theta": -0.0456,
  "vega": 0.1234,
  "rho": 0.0234
}
```

- **delta**: Change in option price per $1 stock move
- **gamma**: Change in delta per $1 stock move
- **theta**: Daily time decay (negative = losing value)
- **vega**: Change per 1% IV increase
- **rho**: Change per 1% interest rate increase

### Analysis Report

```json
{
  "status": "completed",
  "report": {
    "executive_summary": "Portfolio Overview...",
    "risk_score": 45.2,
    "recommendations": [
      {
        "type": "position_management",
        "priority": "high",
        "action": "close AAPL",
        "description": "Near expiration with 50%+ profit"
      }
    ]
  }
}
```

## Common Issues & Solutions

### Issue: Database Connection Error

**Error**: `could not connect to server: Connection refused`

**Solution**:
```bash
# Check if PostgreSQL is running
pg_isready

# Start PostgreSQL
# macOS: brew services start postgresql
# Linux: sudo systemctl start postgresql
# Windows: net start postgresql-x64-15
```

### Issue: Redis Connection Error

**Error**: `Error connecting to Redis`

**Solution**:
```bash
# Check if Redis is running
redis-cli ping

# Start Redis
# macOS: brew services start redis
# Linux: sudo systemctl start redis
# Windows: redis-server
```

### Issue: Module Not Found

**Error**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**:
```bash
# Ensure you're in the correct virtual environment
pip install -r requirements.txt
```

### Issue: Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Find process using port 8000
# macOS/Linux: lsof -i :8000
# Windows: netstat -ano | findstr :8000

# Kill the process or use a different port
uvicorn src.api.main:app --reload --port 8001
```

## Next Steps

1. **Explore the Demo**: Run `python demo/run_demo.py` to see all features
2. **Read the Docs**: Check `docs/COMPREHENSIVE_SYSTEM_ROADMAP.md` for architecture
3. **Run Tests**: Execute `pytest tests/ -v` to verify everything works
4. **Try the API**: Use the examples above to interact with the system
5. **Build the Frontend**: Follow the frontend setup to see the web interface

## Key Files to Review

- **README.md**: Complete system overview
- **SYSTEM_COMPLETE.md**: Implementation report
- **docs/COMPREHENSIVE_SYSTEM_ROADMAP.md**: Full architecture and roadmap
- **docs/SYSTEM_IMPLEMENTATION.md**: Technical implementation details

## Support

For detailed information:
- Architecture: `docs/COMPREHENSIVE_SYSTEM_ROADMAP.md`
- Implementation: `docs/SYSTEM_IMPLEMENTATION.md`
- API Reference: http://localhost:8000/docs (when server is running)

---

**Ready to analyze options like a pro!** 🚀

Start with the demo, then explore the full system capabilities.

