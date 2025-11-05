# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Common Development Commands

### Building and Running

#### Backend (Python FastAPI)
```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the main API server with auto-reload
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Start with rate limiting
python start_server_with_rate_limiting.py

# Start with debugging
python start_server_debug.py
```

#### Frontend (React + TypeScript)
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Run linting
npm run lint
```

### Testing

#### Backend Testing (Python)
```bash
# Run all Python tests with verbose output
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_distillation_e2e.py -v

# Run tests with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Test specific components
python test_llm_connectivity.py          # Test LLM connections
python test_llm_portfolio_analysis.py    # Test portfolio analysis
python test_agentic_research.py          # Test AI research agents
```

#### Frontend Testing (React)
```bash
cd frontend

# Run unit tests
npm test

# Run tests with UI
npm run test:ui

# Run tests with coverage
npm run test:coverage
```

#### E2E Testing (Playwright)
```bash
# Run all E2E tests
npx playwright test

# Run specific test file
npx playwright test e2e/agent-transparency.spec.ts

# Run tests in UI mode for debugging
npx playwright test --ui

# Debug mode with breakpoints
npx playwright test --debug

# View test report
npx playwright show-report
```

### Validation Scripts
```bash
# Validate InvestorReport JSON schema
python scripts/validate_investor_report.py --file <file.json>

# Run performance benchmarks
python scripts/performance_benchmark.py

# Generate eval inputs for testing
python scripts/generate_eval_inputs.py

# Run evaluation suite
python scripts/run_evals.py

# Run nightly eval job
python scripts/nightly_eval_job.py
```

## High-Level Architecture

### System Overview
This is an institutional-grade options and stock analysis platform inspired by Renaissance Technologies (quantitative rigor) and BlackRock Aladdin (risk management). It combines AI-powered multi-agent analysis with real-time sentiment analysis, advanced risk management, and machine learning predictions.

### Core Components

#### 1. Multi-Agent Swarm System (`src/agents/swarm/`)
- **SwarmCoordinator** (`swarm_coordinator.py`): Orchestrates 17 specialized agents across 8 tiers
- **DistillationAgent V2** (`agents/distillation_agent.py`): Synthesizes outputs into investor reports using structured JSON schema
- **Agent Instrumentation** (`agent_instrumentation.py`): Real-time agent transparency with WebSocket streaming
- **MCP Tools** (`mcp_tools.py`): Firecrawl integration for web scraping and fact sheets

#### 2. Analytics Engine (`src/analytics/`)
- **PortfolioMetrics** (`portfolio_metrics.py`): Advanced risk metrics including Phase 4 fields (options flow, momentum, seasonality, liquidity)
- **ML Alpha Engine** (`ml_alpha_engine.py`): Machine learning-based alpha generation with regime detection
- **Sentiment Engine** (`sentiment_engine.py`): NLP-based sentiment scoring across multiple sources
- **Technical Cross-Asset** (`technical_cross_asset.py`): Phase 4 technical indicators

#### 3. API Layer (`src/api/`)
- **FastAPI Main** (`main.py`): Core API with WebSocket support, rate limiting, monitoring
- **Agent Stream** (`agent_stream.py`): WebSocket manager for real-time agent events
- **Investor Report Routes** (`investor_report_routes.py`): GET /api/investor-report with L1/L2 caching
- **Phase 4 WebSocket** (`phase4_websocket.py`): WS /ws/phase4-metrics/{user_id} for real-time updates

#### 4. Frontend (`frontend/src/`)
- **React 18 + TypeScript**: Modern component-based UI
- **Agent Transparency**: Real-time display of agent thinking and progress
- **Risk Panel Dashboard**: 7 institutional-grade risk metrics with regime awareness
- **Phase 4 Signals Panel**: Options flow, momentum, seasonality, liquidity indicators
- **WebSocket Hooks**: `useAgentStream.ts`, `usePhase4Stream.ts` for real-time updates

### Data Flow

1. **CSV Import** → Position Management → Portfolio Analysis
2. **Market Data** → Real-time pricing via yfinance/Polygon/etc
3. **Multi-Agent Analysis** → 17 agents analyze positions → DistillationAgent synthesizes
4. **InvestorReport.v1** → Structured JSON output → Frontend display
5. **WebSocket Streaming** → Real-time agent events + Phase 4 metrics

### Key Design Patterns

#### Agent Diversity & Deduplication
- Temperature variation (0.2-0.7) across tiers for diverse perspectives
- SharedContext prevents redundant analysis (>90% deduplication rate)
- Role-specific prompts for differentiated analysis

#### Graceful Degradation
- LLM fallback chain: LMStudio → Anthropic → OpenAI
- Cache fallback when services unavailable
- Null handling for missing Phase 4 data

#### Performance Optimization
- L1 (TTLCache 15min) + L2 (Redis) caching
- Stale-while-revalidate pattern
- Target: <500ms cached, <10min uncached (multi-agent analysis)

### Critical Files to Understand

1. **Agent System**
   - `src/agents/swarm/swarm_coordinator.py` - Main orchestration
   - `src/agents/swarm/agents/distillation_agent.py` - Report synthesis
   - `src/agents/swarm/shared_context.py` - Deduplication logic

2. **Analytics**
   - `src/analytics/portfolio_metrics.py` - Core metrics dataclass
   - `src/schemas/investor_report_schema.json` - InvestorReport.v1 schema

3. **API**
   - `src/api/main.py` - FastAPI setup with middleware
   - `src/api/investor_report_routes.py` - Main investor report endpoint

4. **Frontend**
   - `frontend/src/types/investor-report.ts` - TypeScript types
   - `frontend/src/hooks/useAgentStream.ts` - Agent WebSocket hook
   - `frontend/src/components/AgentProgressPanel.tsx` - Progress display

## Environment Variables

Key variables from `.env.example`:
- `OPENAI_API_KEY` - For GPT-4 analysis
- `ANTHROPIC_API_KEY` - For Claude analysis  
- `POLYGON_API_KEY` - Market data provider
- `REDIS_URL` - For L2 caching
- `SENTRY_DSN` - Error tracking
- `PHASE4_UPDATE_INTERVAL` - WebSocket update frequency (default 30s)

## Performance Targets

- API Response: <500ms (cached), <10min (uncached multi-agent)
- WebSocket Latency: <50ms for agent events
- Phase 4 Computation: <200ms per asset
- Frontend Render: <100ms
- Page Load: <2s
- Cache Hit Rate: >80% after warmup

## Testing Strategy

The system has comprehensive test coverage across multiple levels:
- **Unit Tests**: 255+ tests across Python and TypeScript
- **Integration Tests**: API endpoints, WebSocket connections, cache behavior
- **E2E Tests**: 55 Playwright tests covering critical user paths
- **Eval System**: 10 evaluation scenarios with automated scoring

Always run relevant tests before committing changes. Use the validation scripts to ensure schema compliance and performance targets are met.