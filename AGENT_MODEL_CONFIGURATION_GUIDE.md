# 🤖 Agent Model Configuration Guide

**How to assign different LLM models to different agents in the swarm**

---

## 📍 **WHERE MODELS ARE CONFIGURED**

### **File 1: Model Definitions** (`src/agents/multi_model_config.py`)
Defines available models and default assignments.

### **File 2: Agent Creation** (`src/api/swarm_routes.py`)
Where agents are instantiated with specific models.

---

## 🎯 **CURRENT CONFIGURATION**

### **Active Agents** (8 total)

| Agent | Type | Current Model | LLM-Powered? |
|-------|------|---------------|--------------|
| **Market Analyst** | LLMMarketAnalystAgent | `anthropic` (Claude) | ✅ Yes |
| **Risk Manager** | LLMRiskManagerAgent | `anthropic` (Claude) | ✅ Yes |
| **Sentiment Analyst** | LLMSentimentAnalystAgent | `anthropic` (Claude) | ✅ Yes |
| **Options Strategist** | OptionsStrategistAgent | None (rule-based) | ❌ No |
| **Technical Analyst** | TechnicalAnalystAgent | None (rule-based) | ❌ No |
| **Portfolio Optimizer** | PortfolioOptimizerAgent | None (rule-based) | ❌ No |
| **Trade Executor** | TradeExecutorAgent | None (rule-based) | ❌ No |
| **Compliance Officer** | ComplianceOfficerAgent | None (rule-based) | ❌ No |

**Summary**:
- ✅ **3 agents** use LLMs (Claude)
- ❌ **5 agents** use rule-based logic

---

## 🔧 **HOW TO CHANGE AGENT MODELS**

### **Option 1: Change Individual Agent Models**

Edit `src/api/swarm_routes.py` (lines 66-113):

```python
# Current configuration
LLMMarketAnalystAgent(
    agent_id="market_analyst_1",
    shared_context=_swarm_coordinator.shared_context,
    consensus_engine=_swarm_coordinator.consensus_engine,
    preferred_model="anthropic"  # ← CHANGE THIS
)
```

**Available models**:
- `"openai"` - GPT-4
- `"anthropic"` - Claude Sonnet 4.5
- `"lmstudio"` - Local model via LM Studio

**Example - Use GPT-4 for Market Analyst**:
```python
LLMMarketAnalystAgent(
    agent_id="market_analyst_1",
    shared_context=_swarm_coordinator.shared_context,
    consensus_engine=_swarm_coordinator.consensus_engine,
    preferred_model="openai"  # Changed to GPT-4
)
```

---

### **Option 2: Add More LLM-Powered Agents**

Currently, only 3 agents use LLMs. You can add more by creating multiple instances:

```python
# Add a second Market Analyst with different model
LLMMarketAnalystAgent(
    agent_id="market_analyst_2",  # Different ID
    shared_context=_swarm_coordinator.shared_context,
    consensus_engine=_swarm_coordinator.consensus_engine,
    preferred_model="openai"  # Different model
),

# Add a third Market Analyst with local model
LLMMarketAnalystAgent(
    agent_id="market_analyst_3",
    shared_context=_swarm_coordinator.shared_context,
    consensus_engine=_swarm_coordinator.consensus_engine,
    preferred_model="lmstudio"  # Local model
)
```

---

### **Option 3: Diversify Agent Types**

Create multiple agents of different types with different models:

```python
agents = [
    # Market Analysis - 3 agents with different models
    LLMMarketAnalystAgent(
        agent_id="market_analyst_claude",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine,
        preferred_model="anthropic"
    ),
    LLMMarketAnalystAgent(
        agent_id="market_analyst_gpt4",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine,
        preferred_model="openai"
    ),
    LLMMarketAnalystAgent(
        agent_id="market_analyst_local",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine,
        preferred_model="lmstudio"
    ),
    
    # Risk Management - 2 agents
    LLMRiskManagerAgent(
        agent_id="risk_manager_claude",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine,
        max_portfolio_delta=100.0,
        max_position_size_pct=0.10,
        max_drawdown_pct=0.15,
        preferred_model="anthropic"
    ),
    LLMRiskManagerAgent(
        agent_id="risk_manager_gpt4",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine,
        max_portfolio_delta=100.0,
        max_position_size_pct=0.10,
        max_drawdown_pct=0.15,
        preferred_model="openai"
    ),
    
    # Sentiment Analysis - 2 agents
    LLMSentimentAnalystAgent(
        agent_id="sentiment_analyst_claude",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine,
        preferred_model="anthropic"
    ),
    LLMSentimentAnalystAgent(
        agent_id="sentiment_analyst_gpt4",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine,
        preferred_model="openai"
    ),
    
    # Keep rule-based agents
    OptionsStrategistAgent(
        agent_id="options_strategist_1",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine
    ),
    TechnicalAnalystAgent(
        agent_id="technical_analyst_1",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine
    ),
    PortfolioOptimizerAgent(
        agent_id="portfolio_optimizer_1",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine
    ),
    TradeExecutorAgent(
        agent_id="trade_executor_1",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine
    ),
    ComplianceOfficerAgent(
        agent_id="compliance_officer_1",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine
    )
]
```

**This gives you**:
- 3 Market Analysts (Claude, GPT-4, LMStudio)
- 2 Risk Managers (Claude, GPT-4)
- 2 Sentiment Analysts (Claude, GPT-4)
- 5 Rule-based agents
- **Total: 12 agents** with diverse perspectives

---

## 🎨 **RECOMMENDED CONFIGURATIONS**

### **Configuration 1: Balanced (Current)**
- 3 LLM agents (all Claude)
- 5 rule-based agents
- **Total: 8 agents**
- **Cost**: Low (only 3 LLM calls)
- **Diversity**: Medium

### **Configuration 2: High Diversity**
- 3 Market Analysts (Claude, GPT-4, LMStudio)
- 2 Risk Managers (Claude, GPT-4)
- 2 Sentiment Analysts (Claude, GPT-4)
- 5 rule-based agents
- **Total: 12 agents**
- **Cost**: Medium (7 LLM calls)
- **Diversity**: High

### **Configuration 3: Maximum Intelligence**
- 3 Market Analysts (Claude, GPT-4, LMStudio)
- 3 Risk Managers (Claude, GPT-4, LMStudio)
- 3 Sentiment Analysts (Claude, GPT-4, LMStudio)
- 3 Options Strategists (if LLM version exists)
- 5 rule-based agents
- **Total: 17 agents**
- **Cost**: High (12+ LLM calls)
- **Diversity**: Maximum

### **Configuration 4: Cost-Effective**
- 1 Market Analyst (LMStudio - free)
- 1 Risk Manager (LMStudio - free)
- 1 Sentiment Analyst (LMStudio - free)
- 5 rule-based agents
- **Total: 8 agents**
- **Cost**: Free (local models only)
- **Diversity**: Low

---

## 🔑 **MODEL FALLBACK CHAIN**

Each LLM agent has automatic fallback:

1. **Try preferred model** (e.g., `anthropic`)
2. **If fails, try OpenAI** (if API key available)
3. **If fails, try Anthropic** (if API key available)
4. **If fails, try LMStudio** (always available)

This ensures agents always get a response even if one provider is down.

---

## 📝 **STEP-BY-STEP: ADD MORE AGENTS**

### **Step 1: Open the file**
```bash
code src/api/swarm_routes.py
```

### **Step 2: Find the agents list** (line 66)
```python
agents = [
    # Current agents here
]
```

### **Step 3: Add new agents**
```python
agents = [
    # Existing agents...
    
    # ADD NEW AGENTS HERE
    LLMMarketAnalystAgent(
        agent_id="market_analyst_gpt4",  # Unique ID
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine,
        preferred_model="openai"  # GPT-4
    ),
    LLMMarketAnalystAgent(
        agent_id="market_analyst_local",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine,
        preferred_model="lmstudio"  # Local
    ),
]
```

### **Step 4: Restart the backend**
```bash
# Kill the current server (Ctrl+C)
# Restart
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### **Step 5: Test**
```bash
python test_csv_swarm_integration.py
```

---

## 🎯 **EXAMPLE: 12-AGENT SWARM**

Here's a complete example with 12 agents using various models:

```python
agents = [
    # Market Analysis Team (3 agents)
    LLMMarketAnalystAgent(
        agent_id="market_claude",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine,
        preferred_model="anthropic"
    ),
    LLMMarketAnalystAgent(
        agent_id="market_gpt4",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine,
        preferred_model="openai"
    ),
    LLMMarketAnalystAgent(
        agent_id="market_local",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine,
        preferred_model="lmstudio"
    ),
    
    # Risk Management Team (2 agents)
    LLMRiskManagerAgent(
        agent_id="risk_claude",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine,
        max_portfolio_delta=100.0,
        max_position_size_pct=0.10,
        max_drawdown_pct=0.15,
        preferred_model="anthropic"
    ),
    LLMRiskManagerAgent(
        agent_id="risk_gpt4",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine,
        max_portfolio_delta=100.0,
        max_position_size_pct=0.10,
        max_drawdown_pct=0.15,
        preferred_model="openai"
    ),
    
    # Sentiment Team (2 agents)
    LLMSentimentAnalystAgent(
        agent_id="sentiment_claude",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine,
        preferred_model="anthropic"
    ),
    LLMSentimentAnalystAgent(
        agent_id="sentiment_gpt4",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine,
        preferred_model="openai"
    ),
    
    # Specialist Agents (5 rule-based)
    OptionsStrategistAgent(
        agent_id="options_strategist_1",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine
    ),
    TechnicalAnalystAgent(
        agent_id="technical_analyst_1",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine
    ),
    PortfolioOptimizerAgent(
        agent_id="portfolio_optimizer_1",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine
    ),
    TradeExecutorAgent(
        agent_id="trade_executor_1",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine
    ),
    ComplianceOfficerAgent(
        agent_id="compliance_officer_1",
        shared_context=_swarm_coordinator.shared_context,
        consensus_engine=_swarm_coordinator.consensus_engine
    )
]
```

---

## 📊 **CONSENSUS WITH MORE AGENTS**

With more agents, the consensus becomes more robust:

**8 agents** (current):
- 3 LLM votes
- 5 rule-based votes
- Consensus threshold: 67% (5/8 agents)

**12 agents** (recommended):
- 7 LLM votes (diverse models)
- 5 rule-based votes
- Consensus threshold: 67% (8/12 agents)

**More agents = More diverse perspectives = Better consensus**

---

## 📍 **WHERE TO FIND RESULTS**

**Configuration File**: `src/api/swarm_routes.py` (lines 66-113)

**Model Definitions**: `src/agents/multi_model_config.py`

**Test Script**: `test_csv_swarm_integration.py`

**Documentation**: This file (`AGENT_MODEL_CONFIGURATION_GUIDE.md`)

---

**Ready to customize your agent swarm! Add more agents with different models for better analysis!** 🚀

