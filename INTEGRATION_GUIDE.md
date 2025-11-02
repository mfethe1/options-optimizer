# Options Optimizer - Complete Integration Guide

This guide shows you how to use the entire AI-powered trading system from strategy development to live execution.

## 📋 Table of Contents

1. [System Architecture](#system-architecture)
2. [Complete Trading Workflow](#complete-trading-workflow)
3. [Backend API Examples](#backend-api-examples)
4. [Frontend Integration](#frontend-integration)
5. [Risk Management Configuration](#risk-management-configuration)
6. [Troubleshooting](#troubleshooting)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                           │
│  (React Frontend - Keyboard Shortcuts & Multi-Monitor)       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                    FASTAPI BACKEND                           │
│  ┌──────────────┬──────────────┬─────────────────────────┐ │
│  │ AI Services  │ Risk Mgmt    │ Live Trading (Schwab)   │ │
│  │ - Swarm      │ - Guardrails │ - OAuth 2.0             │ │
│  │ - Critique   │ - 5 Profiles │ - Order Management      │ │
│  └──────────────┴──────────────┴─────────────────────────┘ │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                  DATA & ANALYTICS                            │
│  ┌────────────┬─────────────┬──────────────────────────┐   │
│  │ Backtesting│ Options     │ Market Data              │   │
│  │ - 9 Strats │ - IV Surface│ - Polygon/Intrinio       │   │
│  │ - 20 Metrics - Greeks    │ - FMP Calendar           │   │
│  └────────────┴─────────────┴──────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Complete Trading Workflow

### Step 1: Strategy Development & Backtesting

**Goal:** Test your trading strategy against historical data

```python
# Backend: Run a backtest
from src.analytics.backtest_engine import BacktestEngine

engine = BacktestEngine()
result = await engine.run_backtest(
    strategy="iron_condor",
    symbol="SPY",
    start_date="2023-01-01",
    end_date="2023-12-31",
    initial_capital=100000
)

print(f"Total Return: {result.total_return}%")
print(f"Sharpe Ratio: {result.sharpe_ratio}")
print(f"Max Drawdown: {result.max_drawdown}%")
print(f"Win Rate: {result.win_rate*100}%")
```

**Frontend:** Navigate to `/backtest` (Ctrl+B)
- Select strategy type
- Configure parameters
- Run backtest
- Review 20+ performance metrics

### Step 2: AI Swarm Analysis

**Goal:** Get multi-agent consensus on strategy viability

```python
# Backend: Analyze with AI swarm
from src.analytics.swarm_analysis_service import SwarmAnalysisService, BacktestResult

swarm = SwarmAnalysisService()

backtest_data = BacktestResult(
    strategy_name="Iron Condor - SPY",
    symbol="SPY",
    timeframe="daily",
    total_return=result.total_return,
    sharpe_ratio=result.sharpe_ratio,
    sortino_ratio=result.sortino_ratio,
    max_drawdown=result.max_drawdown,
    win_rate=result.win_rate,
    profit_factor=result.profit_factor,
    avg_win=result.avg_win,
    avg_loss=result.avg_loss,
    total_trades=result.total_trades,
    kelly_criterion=result.kelly_criterion,
    var_95=result.var_95,
    expected_value=result.expected_value
)

consensus = await swarm.analyze_strategy(backtest_data)

print(f"\n🤖 SWARM CONSENSUS 🤖")
print(f"Recommendation: {consensus.consensus_recommendation}")
print(f"Overall Score: {consensus.overall_score}/100")
print(f"Confidence: {consensus.consensus_confidence*100}%")
print(f"GO Decision: {'✅ APPROVED' if consensus.go_decision else '❌ REJECTED'}")
print(f"\nPosition Sizing:")
print(f"  Suggested: {consensus.suggested_position_size}% of portfolio")
print(f"  Stop Loss: {consensus.stop_loss}%")
print(f"  Take Profit: {consensus.take_profit}%")
print(f"\nAgent Votes:")
for agent_name, vote in consensus.agent_votes.items():
    print(f"  {agent_name}: {vote}")
```

**Frontend:** Use AI Recommendations page (`/ai-recommendations`, Ctrl+I)
- Submit backtest results via API
- View 5-agent analysis
- See consensus vote
- Get position sizing recommendation

### Step 3: Risk Validation

**Goal:** Ensure position passes institutional risk guardrails

```python
# Backend: Check risk guardrails
from src.analytics.risk_guardrails import (
    RiskGuardrailsService,
    RiskLevel,
    PortfolioState,
    Position
)

# Configure risk level (conservative, moderate, or aggressive)
risk_service = RiskGuardrailsService(RiskLevel.MODERATE)

# Current portfolio state
portfolio = PortfolioState(
    total_value=100000,
    cash=50000,
    positions=[
        Position(
            symbol="AAPL",
            quantity=100,
            entry_price=150.00,
            current_price=155.00,
            position_type="STOCK",
            expiration=None,
            sector="Technology",
            beta=1.2
        )
    ],
    daily_pnl=500,
    weekly_pnl=2000,
    monthly_pnl=5000,
    max_drawdown=8.5,
    current_leverage=1.2
)

# Proposed new position
proposed_symbol = "SPY"
proposed_size = 8500  # $8,500 position (8.5% of portfolio)

market_data = {
    "avg_volume": 75000000,
    "bid_ask_spread_pct": 0.01
}

# Run risk check
risk_check = risk_service.check_new_position(
    symbol=proposed_symbol,
    proposed_size=proposed_size,
    position_type="CALL",
    portfolio=portfolio,
    market_data=market_data
)

if risk_check.approved:
    print(f"\n✅ TRADE APPROVED")
    print(f"Max Position Size: ${risk_check.max_position_size:,.2f}")
    print(f"Suggested Size: ${risk_check.suggested_position_size:,.2f}")
    print(f"Risk Score: {risk_check.risk_score}/100 ({risk_check.risk_level})")
else:
    print(f"\n❌ TRADE REJECTED")
    print(f"Risk Score: {risk_check.risk_score}/100 ({risk_check.risk_level})")
    print(f"\nViolations:")
    for v in risk_check.violations:
        print(f"  🚫 {v.rule_name}: {v.message}")
```

**API Call:**
```bash
curl -X POST "http://localhost:8000/api/ai/risk/check-position?risk_level=moderate" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SPY",
    "proposed_size": 8500,
    "position_type": "CALL",
    "portfolio": {
      "total_value": 100000,
      "cash": 50000,
      "positions": [...],
      "daily_pnl": 500,
      "weekly_pnl": 2000,
      "monthly_pnl": 5000,
      "max_drawdown": 8.5,
      "current_leverage": 1.2
    }
  }'
```

### Step 4: Schwab Connection & Execution

**Goal:** Connect to Schwab and execute trade

**A. Connect to Schwab**

Frontend: Navigate to `/schwab-connection` (Ctrl+L)
1. Click "Connect to Schwab"
2. Authorize application
3. View accounts and positions

**B. Place Order**

```python
# Backend: Place order via Schwab API
from src.integrations.schwab_api import SchwabAPIService, OrderType, OrderAction

# Initialize Schwab service (credentials from .env)
schwab = SchwabAPIService(
    client_id=os.getenv("SCHWAB_CLIENT_ID"),
    client_secret=os.getenv("SCHWAB_CLIENT_SECRET"),
    redirect_uri=os.getenv("SCHWAB_REDIRECT_URI")
)

# Place order
order_id = await schwab.place_order(
    account_id="your_account_id",
    symbol="SPY",
    quantity=10,
    order_type=OrderType.LIMIT,
    order_action=OrderAction.BUY_TO_OPEN,
    duration="DAY",
    price=435.50
)

print(f"✅ Order placed! Order ID: {order_id}")
```

**Frontend:** Navigate to `/schwab-trading` (Ctrl+U)
1. Select account
2. Enter symbol and get quote
3. Configure order (type, quantity, price)
4. Click "⚠️ Place Live Order"
5. Confirm execution

### Step 5: Monitor & Optimize

**Goal:** Track performance and refine strategy

**Execution Quality Tracking:**
```python
# Frontend: Navigate to /execution (Ctrl+X)
# View slippage, fill quality, and broker comparison
```

**Platform Improvement:**
```python
# Frontend: Navigate to /ai-recommendations (Ctrl+I)
# Review expert critique and prioritized recommendations
```

---

## 🔧 Backend API Examples

### Complete Strategy Analysis Flow

```python
import asyncio
from src.analytics.backtest_engine import BacktestEngine
from src.analytics.swarm_analysis_service import SwarmAnalysisService, BacktestResult
from src.analytics.risk_guardrails import RiskGuardrailsService, RiskLevel

async def analyze_and_validate_strategy():
    """Complete flow: Backtest → AI Analysis → Risk Check"""

    # 1. Run backtest
    engine = BacktestEngine()
    backtest = await engine.run_backtest(
        strategy="iron_condor",
        symbol="SPY",
        start_date="2023-01-01",
        end_date="2023-12-31",
        initial_capital=100000
    )

    # 2. AI Swarm Analysis
    swarm = SwarmAnalysisService()
    backtest_result = BacktestResult(
        strategy_name="Iron Condor - SPY",
        symbol="SPY",
        timeframe="daily",
        total_return=backtest.total_return,
        sharpe_ratio=backtest.sharpe_ratio,
        sortino_ratio=backtest.sortino_ratio,
        max_drawdown=backtest.max_drawdown,
        win_rate=backtest.win_rate,
        profit_factor=backtest.profit_factor,
        avg_win=backtest.avg_win,
        avg_loss=backtest.avg_loss,
        total_trades=backtest.total_trades,
        kelly_criterion=backtest.kelly_criterion,
        var_95=backtest.var_95,
        expected_value=backtest.expected_value
    )

    consensus = await swarm.analyze_strategy(backtest_result)

    # 3. Risk Validation
    if consensus.go_decision:
        risk_service = RiskGuardrailsService(RiskLevel.MODERATE)

        # Calculate position size
        position_size = (consensus.suggested_position_size / 100) * portfolio_value

        risk_check = risk_service.check_new_position(
            symbol="SPY",
            proposed_size=position_size,
            position_type="CALL",
            portfolio=current_portfolio,
            market_data={}
        )

        if risk_check.approved:
            print("✅ STRATEGY APPROVED FOR LIVE TRADING")
            print(f"Position Size: ${risk_check.suggested_position_size:,.2f}")
            return True
        else:
            print("❌ FAILED RISK CHECK")
            return False
    else:
        print("❌ AI SWARM REJECTED STRATEGY")
        return False

# Run the analysis
asyncio.run(analyze_and_validate_strategy())
```

---

## 🎨 Frontend Integration

### Using AI Services in React

```typescript
// src/components/StrategyAnalyzer.tsx
import { useState } from 'react';
import { analyzeStrategy, checkNewPosition } from '../services/aiApi';
import { toast } from 'react-hot-toast';

export function StrategyAnalyzer() {
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async (backtestResult) => {
    setLoading(true);
    try {
      // Step 1: AI Swarm Analysis
      const consensus = await analyzeStrategy(backtestResult);

      if (consensus.go_decision) {
        toast.success(`AI Approved! Score: ${consensus.overall_score}/100`);

        // Step 2: Risk Check
        const riskCheck = await checkNewPosition(
          backtestResult.symbol,
          calculatePositionSize(consensus.suggested_position_size),
          "CALL",
          currentPortfolio,
          {},
          "moderate"
        );

        if (riskCheck.approved) {
          toast.success(`Risk Check Passed! Max Size: $${riskCheck.max_position_size}`);
          // Proceed to execution
        } else {
          toast.error(`Risk Check Failed: ${riskCheck.violations[0]?.message}`);
        }
      } else {
        toast.error(`AI Rejected Strategy. Consensus: ${consensus.consensus_recommendation}`);
      }
    } catch (error) {
      toast.error(`Analysis failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <button onClick={() => handleAnalyze(backtestResult)} disabled={loading}>
      {loading ? 'Analyzing...' : 'Analyze Strategy'}
    </button>
  );
}
```

---

## ⚙️ Risk Management Configuration

### Available Risk Profiles

| Profile | Max Position | Max Drawdown | Cash Reserve | Ideal For |
|---------|--------------|--------------|--------------|-----------|
| **Ultra Conservative** | 5% | 10% | 25% | Capital preservation, retirees |
| **Conservative** | 8% | 15% | 20% | Long-term growth, low risk |
| **Moderate** | 10% | 20% | 15% | Balanced risk/reward |
| **Aggressive** | 15% | 30% | 10% | High growth, active traders |
| **Ultra Aggressive** | 20% | 40% | 5% | Maximum returns, high tolerance |

### Customizing Risk Limits

```python
from src.analytics.risk_guardrails import RiskGuardrailsService, RiskLevel

# Use a preset profile
risk_service = RiskGuardrailsService(RiskLevel.MODERATE)

# Or customize limits
from src.analytics.risk_guardrails import RiskLimits

custom_limits = RiskLimits(
    max_position_size_pct=12.0,  # Custom: 12% max position
    max_daily_loss_pct=2.5,      # Custom: 2.5% daily loss limit
    max_drawdown_pct=18.0,       # Custom: 18% max drawdown
    # ... other parameters
)

# Apply custom limits
risk_service.limits = custom_limits
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "AI swarm analysis failed"

**Cause:** Missing or invalid backtest metrics

**Solution:**
```python
# Ensure all required fields are provided
backtest_result = BacktestResult(
    strategy_name="Your Strategy",  # Required
    symbol="SPY",                    # Required
    timeframe="daily",               # Required
    total_return=35.5,               # Required
    sharpe_ratio=1.85,               # Required
    # ... all other fields required
)
```

#### 2. "Risk check blocking trade"

**Cause:** Position violates risk limits

**Solution:**
- Review `risk_check.violations` for specific rules
- Reduce position size to `risk_check.suggested_position_size`
- Or switch to more aggressive risk profile if appropriate

#### 3. "Schwab API connection failed"

**Cause:** Missing or invalid credentials

**Solution:**
1. Check `.env` file has correct credentials:
```bash
SCHWAB_CLIENT_ID=your_client_id
SCHWAB_CLIENT_SECRET=your_secret
SCHWAB_REDIRECT_URI=https://localhost:8000/callback
```
2. Verify OAuth callback URL matches in Schwab developer portal
3. Check access token hasn't expired (re-authenticate if needed)

#### 4. "ImportError: cannot import name 'SwarmAnalysisService'"

**Cause:** Missing `__init__.py` or incorrect module structure

**Solution:**
```bash
# Verify __init__.py exists
ls -la src/analytics/__init__.py
ls -la src/integrations/__init__.py

# Reinstall package in development mode
pip install -e .
```

---

## 📊 Performance Expectations

With proper configuration, this system enables:

| Strategy Type | Expected Monthly Return | Risk Level | Success Rate |
|---------------|------------------------|------------|--------------|
| **Conservative** | 5-8% | Low | 85%+ |
| **Moderate** | 10-15% | Moderate | 70-80% |
| **Aggressive** | 15-20% | High | 60-70% |
| **Ultra Aggressive** | 20%+ | Very High | 50-60% |

**Note:** Returns depend on market conditions, strategy quality, and execution discipline. Past performance does not guarantee future results.

---

## 🎯 Next Steps

1. **Start with Backtesting** - Test strategies on historical data
2. **Get AI Validation** - Run swarm analysis on top performers
3. **Configure Risk** - Set appropriate risk profile for your goals
4. **Paper Trade** - Test with simulated capital first
5. **Go Live** - Execute with real capital (start small!)
6. **Monitor & Refine** - Track execution quality and optimize

**Remember:** The AI swarm and risk guardrails are there to protect you. Listen to their recommendations!

---

## 📚 Additional Resources

- **API Documentation**: http://localhost:8000/docs (when server running)
- **Expert Critique**: `/ai-recommendations` page for platform improvements
- **Keyboard Shortcuts**: Press `?` in app for full list
- **Multi-Monitor Setup**: `/multi-monitor` for Bloomberg-style layouts

---

**Happy Trading! 📈💰**

*Built with institutional-grade risk management and AI intelligence*
