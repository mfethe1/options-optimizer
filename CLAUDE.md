# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

World-class institutional-grade options and stock analysis system combining Renaissance Technologies' quantitative rigor with BlackRock Aladdin's risk management framework. Multi-agent AI system powered by LLMs (Claude, GPT-4, LMStudio) with advanced ML forecasting (TFT, GNN, PINN, Mamba, Epidemic models).

**Tech Stack:**
- Backend: FastAPI + Python 3.11+ (TensorFlow, NumPy, SciPy, Pandas, QuantLib)
- Frontend: React 18 + TypeScript + Vite + MUI v7 + TailwindCSS
- Testing: Pytest (backend), Vitest (frontend), Playwright (E2E)
- ML: TensorFlow 2.16+, Graph Neural Networks, Physics-Informed Neural Networks, State-Space Models
- Real-time: WebSocket streaming for agent events and Phase 4 metrics

## Common Commands

### Backend Development

```bash
# Start development server (with hot reload)
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Run all backend tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_portfolio_phase4_integration.py -v

# Run tests with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Code formatting
black src/ tests/

# Type checking
mypy src/
```

### Frontend Development

```bash
# Install dependencies
cd frontend && npm install

# Start dev server (port 3000 by default)
npm run dev

# Build for production
npm run build

# Run all frontend tests
npm test

# Run tests with UI
npm run test:ui

# Run tests with coverage
npm run test:coverage

# Lint code
npm run lint
```

### End-to-End Testing

```bash
# Run all Playwright tests
npx playwright test

# Run specific test file
npx playwright test e2e/unified-analysis.spec.ts

# Run tests in UI mode (interactive debugging)
npx playwright test --ui

# Run tests in debug mode
npx playwright test --debug

# View test report
npx playwright show-report

# Run tests for single browser
npx playwright test --project=chromium
```

### Database & Infrastructure

```bash
# Start PostgreSQL (if using)
psql -U postgres -f src/database/schema.sql

# Start Redis (required for caching and WebSocket)
docker run -d -p 6379:6379 redis:7

# Run with Docker Compose (full stack)
docker-compose up
```

## Architecture Overview

### Backend Architecture (3-Layer)

1. **API Layer** (`src/api/`)
   - FastAPI application with CORS, rate limiting, authentication
   - Routes: `main.py`, `unified_routes.py`, `gnn_routes.py`, model management routes
   - Middleware: TimingMiddleware, PrometheusMiddleware, rate limiting
   - WebSocket endpoints: agent streaming, Phase 4 metrics streaming

2. **Multi-Agent Swarm System** (`src/agents/swarm/`)
   - 17 specialized agents (8 LLM-powered tiers + Distillation Agent)
   - Agent types: Market Analyst, Risk Manager, Sentiment Analyst, Fundamental Analyst, Macro Economist, Volatility Specialist, Technical Analyst, Options Strategist
   - Distillation Agent (Tier 8): Synthesizes all agent outputs into investor-friendly reports
   - Consensus Engine: Weighted, majority, unanimous, quorum methods
   - Shared Context: Agent communication and coordination
   - Agent Instrumentation: Real-time event streaming with progress tracking

3. **ML & Analytics Engine** (`src/ml/`, `src/analytics/`)
   - Advanced Forecasting: Temporal Fusion Transformer (TFT), Conformal Prediction
   - Graph Neural Networks: Stock correlation networks
   - Physics-Informed Neural Networks: Black-Scholes PDE constraints
   - State-Space Models: Mamba for long-range dependencies
   - Bio-Financial: Epidemic volatility modeling (COVID-style contagion)
   - Ensemble: Weighted combination of all models
   - Portfolio Metrics: Phase 1-4 signals (Omega, GH1, Pain Index, CVaR, Options Flow, Residual Momentum, Seasonality, Breadth/Liquidity)

### Frontend Architecture (React + TypeScript)

**Key Pages:**
- `UnifiedAnalysis.tsx`: Overlay chart for all ML models with NavigationSidebar
- `GNNPage.tsx`: Graph Neural Network analysis with network visualization
- `PINNPage.tsx`, `MambaPage.tsx`: Advanced forecasting pages
- `EpidemicVolatilityPage.tsx`: Bio-financial contagion modeling
- `SwarmAnalysisPage.tsx`: Multi-agent swarm analysis with investor reports
- `Phase4DemoPage.tsx`, `RiskPanelDemoPage.tsx`: Demo pages for testing components

**Key Components:**
- `Phase4SignalsPanel.tsx`: 2×2 grid of Phase 4 signals (Options Flow, Residual Momentum, Seasonality, Breadth/Liquidity)
- `RiskPanelDashboard.tsx`: 7-metric risk dashboard (Omega, GH1, Pain Index, CVaR, Upside/Downside Capture, Max Drawdown)
- `AgentProgressPanel.tsx`, `AgentConversationDisplay.tsx`: Real-time agent transparency
- `TradingViewChart.tsx`: Professional charting with lightweight-charts
- `NavigationSidebar.tsx`: Organized navigation structure
- `InvestorReportSynopsis.tsx`: Investor-friendly synopsis (no JSON)

**State Management:**
- Zustand for global state
- React hooks for local state and WebSocket connections
- `usePhase4Stream.ts`: Real-time Phase 4 metrics streaming

### ML Model Architecture

**5 Advanced Models:**
1. **TFT (Temporal Fusion Transformer)**: Multi-horizon forecasting with attention
2. **GNN (Graph Neural Network)**: Stock correlation networks for cross-asset signals
3. **PINN (Physics-Informed Neural Network)**: Black-Scholes PDE constraints
4. **Mamba**: State-space model for long-range dependencies
5. **Epidemic Volatility**: Bio-financial contagion modeling

**Ensemble Service:**
- Combines all models with weighted voting
- Confidence-based weighting
- Graceful degradation when models unavailable

### Agent System Architecture

**17-Agent Swarm (8 Tiers):**
- Tier 1-7: Specialized agents (Market, Risk, Sentiment, Fundamental, Macro, Volatility, Technical)
- Tier 8: **Distillation Agent** - Synthesizes all outputs into cohesive investor narratives
- Zero redundancy via context engineering
- Temperature diversity (0.2-0.7 range) for varied perspectives
- Real-time event streaming via WebSocket

**Agent Transparency System:**
- WebSocket streaming at `/ws/agent-stream/{user_id}`
- Event types: STARTED, THINKING, TOOL_CALL, TOOL_RESULT, PROGRESS, COMPLETED, ERROR, HEARTBEAT
- AgentStreamManager handles connection lifecycle, buffering, heartbeats
- AgentInstrumentation wraps agents with automatic event emission

## Critical Implementation Notes

### MUI v7 Grid Migration (IMPORTANT!)

**MUI Grid v7 Breaking Changes:**
- Remove `item` prop entirely
- Replace `xs={6}` with `size={{ xs: 6 }}`
- Replace `sm/md/lg/xl` with `size={{ sm: 4, md: 3 }}`
- Import `Grid` from `@mui/material/Grid` (NOT `@mui/material`)
- Use `Grid2` for new code

**Example Migration:**
```tsx
// OLD (MUI v5)
<Grid container spacing={2}>
  <Grid item xs={12} md={6}>Content</Grid>
</Grid>

// NEW (MUI v7)
<Grid container spacing={2}>
  <Grid size={{ xs: 12, md: 6 }}>Content</Grid>
</Grid>
```

### WebSocket Streaming

**Agent Stream:**
- Endpoint: `ws://localhost:8000/ws/agent-stream/{user_id}`
- Heartbeat every 30s to keep connections alive
- Event buffering (max 100 per user) when no active connections
- Auto-reconnect with exponential backoff (frontend hook)

**Phase 4 Metrics Stream:**
- Endpoint: `ws://localhost:8000/ws/phase4-metrics/{user_id}`
- Update interval: 30s (env-configurable via `PHASE4_UPDATE_INTERVAL_SECONDS`)
- Schema-validated updates (InvestorReport.v1)

### Schema Validation

**InvestorReport.v1 JSON Schema:**
- Location: `src/schemas/investor_report_schema.json`
- Validates: executive_summary, risk_panel, signals (Phase 1-4), actions, sources, confidence
- Required fields: as_of, universe, executive_summary, risk_panel, signals, actions, sources, confidence, metadata
- Validation script: `scripts/validate_investor_report.py`

### Performance Targets

**API:**
- Cached: <500ms (p95)
- Uncached: <10 minutes acceptable for multi-agent analysis
- WebSocket latency: <50ms

**Frontend:**
- Page load: <2s
- Component render: <100ms
- Chart update: <50ms
- Frame rate: ≥55fps

**ML Models:**
- Phase 4 computation: <200ms per asset
- Model inference: <1s per prediction

### Testing Best Practices

**Backend Tests:**
- Use pytest fixtures for test data
- Mock external API calls (LLM providers, market data)
- Test both success and error paths
- Verify schema compliance for all API responses
- Use `pytest-asyncio` for async tests

**Frontend Tests:**
- Use Vitest + React Testing Library
- Test user interactions, not implementation details
- Mock API calls and WebSocket connections
- Verify accessibility (ARIA labels, keyboard navigation)
- Test responsive behavior (mobile, tablet, desktop)

**E2E Tests (Playwright):**
- Run on multiple browsers (Chromium, Firefox, WebKit)
- Test critical user flows end-to-end
- Verify API integration and WebSocket streaming
- Performance benchmarks (page load, API latency)
- Take screenshots/videos on failure only

### Environment Configuration

**Required API Keys (for production):**
- `OPENAI_API_KEY`: GPT-4 for multi-agent system
- `ANTHROPIC_API_KEY`: Claude Sonnet 4.5 for distillation agent
- `POLYGON_API_KEY` or `INTRINIO_API_KEY`: Real-time market data

**Optional API Keys:**
- `LMSTUDIO_BASE_URL`: Local LLM inference
- `BENZINGA_API_KEY`, `NEWSAPI_KEY`: News feeds
- `SCHWAB_CLIENT_ID`, `SCHWAB_CLIENT_SECRET`: Live trading
- `FINNHUB_API_KEY`, `FMP_API_KEY`, `FRED_API_KEY`: Additional data sources

**Development:**
- Copy `.env.example` to `.env`
- System works with fallback data when API keys not configured
- yfinance used as fallback data provider (retail-grade, no SLA)

### Git Workflow

**Branches:**
- `main`: Production-ready code
- Feature branches: `claude/feature-name-*`

**Commits:**
- Write clear, descriptive commit messages
- Reference issue numbers if applicable
- Use conventional commits format preferred

**Creating PRs:**
```bash
# Push current branch
git push -u origin branch-name

# Create PR using gh CLI
gh pr create --title "feat: Add feature" --body "Description"

# Or create PR manually on GitHub
```

## Code Organization

### Backend Module Structure

```
src/
├── api/                   # FastAPI routes and middleware
│   ├── main.py           # Main application with CORS, rate limiting, monitoring
│   ├── unified_routes.py # Unified analysis endpoints (all models)
│   ├── gnn_routes.py     # GNN-specific endpoints
│   ├── agent_stream.py   # WebSocket agent streaming
│   └── model_management_routes.py  # Model lifecycle management
├── agents/swarm/         # Multi-agent system
│   ├── swarm_coordinator.py        # Agent orchestration
│   ├── consensus_engine.py         # Consensus algorithms
│   ├── shared_context.py           # Agent communication
│   ├── agent_instrumentation.py    # Real-time event streaming
│   └── agents/                     # 17 specialized agents
│       ├── distillation_agent.py   # Tier 8: Report synthesis
│       ├── llm_market_analyst.py   # Market analysis
│       └── ...
├── ml/                   # Machine learning models
│   ├── advanced_forecasting/  # TFT + Conformal Prediction
│   ├── graph_neural_network/  # GNN for stock correlations
│   ├── physics_informed/      # PINN with Black-Scholes
│   ├── state_space/           # Mamba model
│   ├── bio_financial/         # Epidemic volatility
│   └── ensemble/              # Ensemble predictor
├── analytics/            # Portfolio analytics
│   ├── portfolio_metrics.py  # Phase 1-4 metrics
│   ├── signals.py            # Signal generation
│   └── ...
└── schemas/              # JSON schemas
    └── investor_report_schema.json
```

### Frontend Module Structure

```
frontend/src/
├── pages/                # Route components
│   ├── UnifiedAnalysis.tsx       # Home page (overlay chart)
│   ├── SwarmAnalysisPage.tsx     # Multi-agent analysis
│   ├── AdvancedForecasting/      # ML model pages
│   │   ├── GNNPage.tsx
│   │   ├── PINNPage.tsx
│   │   └── MambaPage.tsx
│   └── BioFinancial/
│       └── EpidemicVolatilityPage.tsx
├── components/           # Reusable components
│   ├── charts/          # Chart components
│   │   ├── TradingViewChart.tsx
│   │   ├── GNNNetworkChart.tsx
│   │   └── MLPredictionChart.tsx
│   ├── Phase4SignalsPanel.tsx
│   ├── RiskPanelDashboard.tsx
│   ├── NavigationSidebar.tsx
│   └── __tests__/       # Component tests
├── hooks/               # Custom React hooks
│   ├── usePhase4Stream.ts     # Phase 4 WebSocket
│   └── useAgentStream.ts      # Agent event streaming
├── services/            # API services
│   └── investor-report-api.ts
└── types/               # TypeScript types
    └── investor-report.ts
```

## Common Issues & Solutions

### Issue: "X-RateLimit-Remaining: 0" 429 Errors

Rate limiting is enabled by default. Limits:
- Swarm analysis: 5/minute
- Analysis endpoints: 10/minute
- Read operations: 100/minute
- Write operations: 30/minute

**Solution:** Wait for rate limit reset or pass `user_id` query param for per-user limits.

### Issue: TensorFlow DLL Initialization Error (Windows)

TensorFlow is preloaded in `src/api/main.py` to avoid Windows DLL conflicts.

**Solution:** Import order matters - TensorFlow must be imported before other ML libraries.

### Issue: WebSocket Connection Refused

Backend must be running for WebSocket endpoints.

**Solution:**
```bash
# Start backend first
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Then start frontend
cd frontend && npm run dev
```

### Issue: Playwright Tests Fail to Start

Playwright config starts dev servers automatically on ports 3010 (frontend) and 8017 (backend).

**Solution:** Ensure ports are available or change in `playwright.config.ts`.

### Issue: MUI Grid Type Errors After v7 Upgrade

MUI v7 has breaking Grid changes.

**Solution:** Use migration guide in "Critical Implementation Notes" section above.

### Issue: Missing API Keys

System uses fallback data when API keys not configured.

**Solution:** Copy `.env.example` to `.env` and add keys, or continue with fallback (retail-grade) data.

## Documentation References

**Key Documentation Files:**
- `README.md`: System overview, features, quick start
- `LIVE_DATA_INTEGRATION.md`: Real-time data integration guide
- `docs/FRONTEND_LLM_V2_INTEGRATION_PLAN.md`: Frontend architecture
- `docs/LLM_UPGRADE_SUMMARY.md`: Backend agent system
- `docs/UI_UX_DESIGN_SPECIFICATION.md`: Design system
- `e2e/README.md`: E2E testing guide (if exists)

**Test Coverage:**
- Backend: 255 tests (100% passing)
- Frontend: 93 tests (100% passing)
- E2E: 55 tests across 4 spec files

## Development Workflow

**Starting Fresh Development Session:**
1. Pull latest changes: `git pull origin main`
2. Start backend: `python -m uvicorn src.api.main:app --reload`
3. Start frontend: `cd frontend && npm run dev`
4. Run tests to verify: `python -m pytest tests/ -v` and `cd frontend && npm test`

**Adding New Features:**
1. Create feature branch: `git checkout -b claude/feature-name-*`
2. Implement feature with tests
3. Run full test suite (backend, frontend, E2E)
4. Commit changes with clear messages
5. Create PR: `gh pr create --title "feat: Feature name" --body "Description"`

**Debugging Issues:**
1. Check logs: Backend logs show request latency, errors
2. Check health: `curl http://localhost:8000/health/detailed`
3. Check metrics: `curl http://localhost:8000/metrics`
4. Use Playwright UI mode for E2E debugging: `npx playwright test --ui`

## Production Deployment Notes

- Use environment-specific `.env` files
- Configure production API keys (Polygon, Intrinio for real-time data)
- Set up PostgreSQL for production (currently in-memory)
- Configure Redis for caching and WebSocket scaling
- Enable Sentry error tracking with `SENTRY_DSN`
- Set up Prometheus + Grafana for monitoring
- Use `npm run build` for frontend production build
- Run backend with production ASGI server (Gunicorn + Uvicorn workers)

## AI/LLM Integration

**Supported Providers:**
- OpenAI (GPT-4): Primary for multi-agent system
- Anthropic (Claude Sonnet 4.5): Distillation agent
- LMStudio: Local inference (privacy-focused)

**Configuration:**
- Multi-model discussion: 5 rounds with different models
- Temperature range: 0.2-0.7 (varied by agent role)
- Retry logic: Exponential backoff for rate limits (429 errors)
- Graceful degradation: System continues with fallback when LLM unavailable

**Agent Roles:**
- Market Analyst: GPT-4 (temperature 0.3)
- Risk Manager: Claude Sonnet (temperature 0.2)
- Sentiment Analyst: GPT-4 (temperature 0.5)
- Distillation Agent: Claude Sonnet (temperature 0.4)
