# LLM Upgrade Implementation Plan - Institutional-Grade Distillation

## Executive Summary

This plan upgrades the Options Probability system to institutional-grade LLM-powered analysis by:
1. **Enforcing Structured Outputs** via JSON Schema validation (InvestorReport.v1)
2. **Integrating MCP tools** (jarvis + Firecrawl) for data retrieval and computation
3. **Adding Phase 4 metrics** (options flow, residual momentum, seasonality, breadth)
4. **Implementing provenance tracking** with authoritative sources
5. **Building evaluation infrastructure** (OpenAI Evals, Ragas, LangSmith)

**Target**: Transform soft LLM narratives into repeatable, measurable, institutional-quality reports.

---

## Phase 1: Structured Outputs & Schema Enforcement

### 1.1 Create JSON Schema for InvestorReport.v1

**File**: `src/schemas/investor_report_schema.json` (NEW)

**Action**: Create the complete JSON Schema from the blueprint (see section 3 of original report).

**Key Fields**:
- `as_of`, `universe`, `executive_summary`, `risk_panel`, `signals`, `actions`, `sources`, `confidence`
- Risk Panel: `omega`, `gh1`, `pain_index`, `upside_capture`, `downside_capture`, `cvar_95`, `max_drawdown`
- Phase 4 Tech: `options_flow_composite`, `residual_momentum`, `seasonality_score`, `breadth_liquidity`

### 1.2 Update PortfolioMetrics Dataclass

**File**: `src/analytics/portfolio_metrics.py`

**Changes**:
```python
@dataclass
class PortfolioMetrics:
    # ... existing fields ...
    
    # Phase 4: Technical & Cross-Asset Metrics (NEW)
    options_flow_composite: Optional[float] = None  # PCR + IV skew + volume
    residual_momentum: Optional[float] = None       # Asset vs market/sector
    seasonality_score: Optional[float] = None       # Calendar patterns
    breadth_liquidity: Optional[float] = None       # Market breadth + liquidity
    
    # Enhanced metadata
    data_sources: List[Dict[str, str]] = field(default_factory=list)  # Provenance
    as_of: str = ""  # ISO timestamp
```

### 1.3 Add Schema Validation to DistillationAgent

**File**: `src/agents/swarm/agents/distillation_agent.py`

**Changes**:
```python
import json
import jsonschema
from pathlib import Path

class DistillationAgent(LLMAgentBase):
    def __init__(self, ...):
        # ... existing init ...
        
        # Load JSON Schema
        schema_path = Path(__file__).parent.parent.parent.parent / "src/schemas/investor_report_schema.json"
        with open(schema_path) as f:
            self.report_schema = json.load(f)
    
    def _generate_narrative(self, insights, position_data) -> Dict[str, Any]:
        """Generate investor-friendly narrative with schema validation"""
        
        # Build synthesis prompt with schema
        prompt = self._build_synthesis_prompt_v2(insights, position_data)
        
        # Try LLM call with retry on schema failure
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self.call_llm(prompt, temperature=self.temperature)
                narrative = self._parse_narrative_response(response)
                
                # Validate against schema
                jsonschema.validate(instance=narrative, schema=self.report_schema)
                
                logger.info("✅ Report validated against InvestorReport.v1 schema")
                return narrative
                
            except jsonschema.ValidationError as e:
                logger.warning(f"⚠️ Schema validation failed (attempt {attempt+1}): {e.message}")
                
                if attempt < max_retries - 1:
                    # Retry with schema + example
                    prompt = self._build_retry_prompt(prompt, e.message)
                else:
                    # Final fallback
                    return self._generate_fallback_narrative(insights, position_data)
        
        return self._generate_fallback_narrative(insights, position_data)
```

---

## Phase 2: MCP Tool Integration (jarvis + Firecrawl)

### 2.1 Register jarvis MCP Tools

**File**: `src/agents/swarm/mcp_tools.py` (NEW)

**Purpose**: Expose jarvis tools for LLM function calling.

**Tools to Expose**:
```python
# Computation tools
- compute_portfolio_metrics(positions, benchmark_returns) -> PortfolioMetrics
- compute_options_flow(symbol, pcr, iv_skew, volume) -> float
- compute_residual_momentum(asset_returns, market_returns, sector_returns) -> float
- compute_seasonality_score(returns_series) -> float

# Data retrieval tools
- get_price_history(symbol, days) -> pd.DataFrame
- get_options_chain(symbol) -> Dict
- grep_repo(paths, patterns) -> List[str]

# File operations
- create_file(path, content) -> bool
- run_tests(test_path) -> Dict
```

**Implementation**:
```python
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class JarvisMCPTools:
    """Wrapper for jarvis MCP tools to be called by LLM agents"""
    
    @staticmethod
    def compute_portfolio_metrics(positions: List[Dict], benchmark_returns: List[float]) -> Dict[str, Any]:
        """Compute institutional-grade portfolio metrics"""
        from src.analytics.portfolio_metrics import PortfolioAnalytics
        
        analytics = PortfolioAnalytics()
        metrics = analytics.calculate_all_metrics(
            portfolio_returns=...,  # Extract from positions
            benchmark_returns=benchmark_returns,
            position_returns=...,
            position_weights=...
        )
        
        return metrics.__dict__
    
    @staticmethod
    def compute_options_flow(symbol: str, pcr: float, iv_skew: float, volume: int) -> float:
        """Compute options flow composite score"""
        # TODO: Implement in Phase 4
        return 0.0
```

### 2.2 Integrate Firecrawl MCP

**File**: `src/agents/swarm/llm_agent_base.py`

**Update FirecrawlMixin**:
```python
class FirecrawlMixin:
    """Mixin class with REAL Firecrawl MCP integration"""
    
    def search_web(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search web using Firecrawl MCP (REAL implementation)"""
        try:
            # Call actual Firecrawl MCP tool
            from firecrawl_search_firecrawl_mcp import firecrawl_search
            
            results = firecrawl_search(
                query=query,
                limit=max_results,
                scrapeOptions={
                    "formats": ["markdown"],
                    "onlyMainContent": True
                }
            )
            
            return {
                'query': query,
                'results': results,
                'timestamp': datetime.utcnow().isoformat(),
                'source': 'firecrawl'
            }
        except Exception as e:
            logger.error(f"Firecrawl search failed: {e}")
            return {'query': query, 'results': [], 'error': str(e)}
    
    def fetch_provider_fact_sheet(self, provider: str, signal: str) -> Dict[str, Any]:
        """Fetch authoritative fact sheets from providers"""
        queries = {
            'extractalpha_cam': 'site:extractalpha.com "Cross-Asset Model" fact sheet',
            'cboe_pcr': 'site:cboe.com "put/call ratio" historical data',
            'sec_13f': 'site:sec.gov "Form 13F Data Sets"',
            'fred_api': 'site:fred.stlouisfed.org "series_observations"',
            'alphasense': 'site:alphasense.com sentiment transcripts API',
            'lseg_marketpsych': 'site:lseg.com MarketPsych Analytics fact sheet'
        }
        
        query_key = f"{provider}_{signal}".lower()
        query = queries.get(query_key, f"site:{provider}.com {signal}")
        
        return self.search_web(query, max_results=3)
```

### 2.3 Update DistillationAgent System Prompt

**File**: `src/agents/swarm/prompt_templates.py`

**Add new function**:
```python
def get_distillation_prompt_v2(
    insights: Dict[str, List[Dict]],
    position_data: Dict[str, Any],
    schema: Dict[str, Any]
) -> str:
    """
    Generate prompt for Distillation Agent v2 with:
    - Structured Outputs enforcement
    - Tool calling instructions
    - Provenance requirements
    """
    
    return f"""
[ROLE] Senior Quant Distillation Agent

[MISSION] Produce an **InvestorReport (JSON)** strictly matching the schema below.

[BEHAVIORAL RULES - NON-NEGOTIABLE]
1. Use ONLY provided metrics and tool results. If missing, CALL TOOLS (jarvis/firecrawl).
2. Return STRICTLY in JSON Schema format. No free-text outside `explanations`.
3. CITE SOURCES: add `sources[]` with title/url/provider/as_of.
4. Short explanations (2-4 lines) per metric: what it means, why it matters.
5. Regime awareness: if High-Vol/Crisis, up-weight downside risk and hedging.
6. Degrade gracefully: if metric unavailable, (a) state it, (b) request via tools, (c) note confidence penalty.

[TOOLBOX - MCP]
- jarvis.compute_portfolio_metrics(positions, benchmark_returns)
- jarvis.compute_options_flow(symbol, pcr, iv_skew, volume)
- firecrawl.search_web(query, max_results)
- firecrawl.fetch_provider_fact_sheet(provider, signal)

[JSON SCHEMA]
{json.dumps(schema, indent=2)}

[CONTEXT]
{json.dumps(insights, indent=2)}

[POSITION DATA]
{json.dumps(position_data, indent=2)}

Generate the InvestorReport JSON now:
"""
```

---

## Phase 3: Phase 4 Metrics Implementation

### 3.1 Create technical_cross_asset.py Module

**File**: `src/analytics/technical_cross_asset.py` (NEW)

**Purpose**: Implement Phase 4 short-horizon metrics.

**Functions** (see full implementation in next file edit):
- `options_flow_composite(pcr, iv_skew, volume_ratio) -> float`
- `residual_momentum(asset_returns, market_returns, sector_returns) -> float`
- `seasonality_score(returns_series, calendar_effects) -> float`
- `breadth_liquidity(advancing, declining, volume) -> float`

### 3.2 Update calculate_all_metrics()

**File**: `src/analytics/portfolio_metrics.py`

**Add Phase 4 calculations**:
```python
def calculate_all_metrics(self, ...) -> PortfolioMetrics:
    # ... existing calculations ...
    
    # Phase 4: Technical & Cross-Asset Metrics
    from .technical_cross_asset import (
        options_flow_composite,
        residual_momentum,
        seasonality_score,
        breadth_liquidity
    )
    
    # Compute if data available
    opt_flow = None
    if 'pcr' in position_data and 'iv_skew' in position_data:
        opt_flow = options_flow_composite(
            pcr=position_data['pcr'],
            iv_skew=position_data['iv_skew'],
            volume_ratio=position_data.get('volume_ratio', 1.0)
        )
    
    # ... similar for other Phase 4 metrics ...
    
    metrics = PortfolioMetrics(
        # ... existing fields ...
        options_flow_composite=opt_flow,
        residual_momentum=res_mom,
        seasonality_score=season,
        breadth_liquidity=breadth,
        data_sources=sources,
        as_of=datetime.utcnow().isoformat()
    )
```

---

## Phase 4: Evaluation & Observability

### 4.1 Create OpenAI Evals Test Suite

**File**: `tests/evals/investor_report_evals.yaml` (NEW)

**Purpose**: Regression tests for report quality.

**Test Cases**:
```yaml
- name: report_completeness
  description: All required fields present
  input:
    positions: [...]
    metrics: {...}
  expected:
    has_fields: [omega, gh1, pain_index, upside_capture, downside_capture, cvar_95, max_drawdown]
    has_sources: true
    min_sources: 2

- name: risk_clarity
  description: Risk explanations are clear and actionable
  input: {...}
  criteria:
    - explanations_length: [50, 500]
    - mentions_metrics: [omega, gh1, pain]
    - actionable: true
```

### 4.2 Add Tracing with LangSmith

**File**: `src/agents/swarm/agents/distillation_agent.py`

**Add tracing**:
```python
from langsmith import traceable

class DistillationAgent(LLMAgentBase):
    @traceable(name="distillation_synthesis")
    def synthesize_swarm_output(self, swarm_output: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize with tracing"""
        # ... existing code ...
```

---

## Implementation Checklist

### Backend
- [ ] Create `src/schemas/investor_report_schema.json`
- [ ] Update `PortfolioMetrics` dataclass with Phase 4 fields
- [ ] Add schema validation to `DistillationAgent`
- [ ] Create `src/agents/swarm/mcp_tools.py` (jarvis wrapper)
- [ ] Update `FirecrawlMixin` with real Firecrawl MCP calls
- [ ] Create `src/analytics/technical_cross_asset.py`
- [ ] Update `calculate_all_metrics()` with Phase 4
- [ ] Add unit tests for Phase 4 metrics

### LLM Prompts
- [ ] Create `get_distillation_prompt_v2()` with schema
- [ ] Add tool calling instructions
- [ ] Add retry logic on schema validation failure

### Evaluation
- [ ] Create `tests/evals/investor_report_evals.yaml`
- [ ] Add LangSmith tracing decorators
- [ ] Set up nightly eval job

### Documentation
- [ ] Update README with Phase 4 metrics
- [ ] Document MCP tool usage
- [ ] Add metric tooltips with provider links

---

## Where to Find Results

After implementation:
- **Schema**: `src/schemas/investor_report_schema.json`
- **Phase 4 Module**: `src/analytics/technical_cross_asset.py`
- **MCP Tools**: `src/agents/swarm/mcp_tools.py`
- **Updated Agent**: `src/agents/swarm/agents/distillation_agent.py`
- **Tests**: `tests/test_technical_cross_asset.py`, `tests/evals/investor_report_evals.yaml`
- **Prompts**: `src/agents/swarm/prompt_templates.py` (v2 functions)

**Commands**:
- Run tests: `python -m pytest tests/test_technical_cross_asset.py -v`
- Validate schema: `python scripts/validate_investor_report.py`
- Run evals: `python scripts/run_evals.py`

