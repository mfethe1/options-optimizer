# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Institutional-grade options and stock analysis system with multi-agent AI (Claude, GPT-4, LMStudio) and advanced ML forecasting (TFT, GNN, PINN, Mamba, Epidemic models).

**Tech Stack:**
- Backend: FastAPI + Python 3.9-3.12 (TensorFlow 2.16+, NumPy, SciPy, Pandas, QuantLib)
- Frontend: React 18 + TypeScript + Vite + MUI v7 + TailwindCSS
- Testing: Pytest (backend), Vitest (frontend), Playwright (E2E)
- Real-time: WebSocket streaming for agent events and Phase 4 metrics

## Common Commands

### Backend
```bash
# Start dev server
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Run tests
python -m pytest tests/ -v                    # All tests
python -m pytest tests/test_file.py -v        # Single file
python -m pytest tests/ -k "phase4" -v        # Pattern match
python -m pytest tests/ --cov=src             # With coverage

# Formatting and type checking
black src/ tests/
mypy src/
```

### Frontend
```bash
cd frontend
npm install                    # Install dependencies
npm run dev                    # Start dev server (port 5173)
npm test                       # Run tests
npm test -- SignalCard.test.tsx  # Single test file
npm run build                  # Production build
npm run lint                   # Lint code
```

### E2E Testing (Playwright)
```bash
npx playwright test                              # Run all (auto-starts servers on 3010/8017)
npx playwright test e2e/unified-analysis.spec.ts # Single file
npx playwright test --ui                         # Interactive UI mode
npx playwright test --debug                      # Debug mode
npx playwright show-report                       # View report
```

### Infrastructure
```bash
docker run -d -p 6379:6379 redis:7    # Start Redis
docker-compose up                      # Full stack
```

## Architecture Overview

### Backend (3-Layer)

1. **API Layer** (`src/api/`)
   - `main.py` - Main entry point (CRITICAL: TensorFlow imported first for Windows DLL fix)
   - `unified_routes.py` - ML forecasting (`/api/forecast/all`, `/api/models/status`)
   - `phase4_websocket.py` - Phase 4 metrics streaming
   - WebSocket: `/ws/agent-stream/{user_id}`, `/ws/phase4-metrics/{user_id}`

2. **Multi-Agent Swarm** (`src/agents/swarm/`)
   - 17 specialized agents with Distillation Agent (Tier 8) for report synthesis
   - Key: `swarm_coordinator.py`, `consensus_engine.py`, `agent_instrumentation.py`

3. **ML Engine** (`src/ml/`)
   - `advanced_forecasting/` - TFT + Conformal Prediction
   - `graph_neural_network/` - Stock correlation networks
   - `physics_informed/` - PINN with Black-Scholes
   - `state_space/` - Mamba model
   - `bio_financial/` - Epidemic volatility
   - `ensemble/` - Weighted combination

### Frontend

- **Home**: `UnifiedAnalysis.tsx` - Overlay chart for all 5 ML models
- **Key Components**: `Phase4SignalsPanel.tsx`, `RiskPanelDashboard.tsx`, `NavigationSidebar.tsx`
- **Hooks**: `usePhase4Stream.ts`, `useAgentStream.ts` for WebSocket streaming
- **State**: Zustand for global state

## Critical Gotchas

1. **TensorFlow Import Order (Windows)**: TensorFlow MUST be imported before other ML libraries in `src/api/main.py`. Never change this order.

2. **MUI v7 Grid Migration**:
   ```tsx
   // OLD - Don't use
   <Grid item xs={12} md={6}>

   // NEW - Use this
   <Grid size={{ xs: 12, md: 6 }}>  // Remove 'item', use 'size'
   ```
   Import from `@mui/material/Grid` (not `@mui/material`). Use `Grid2` for new code.

3. **Playwright Ports**: Tests auto-start servers on ports 3010 (frontend) and 8017 (backend), NOT 8000/5173.

4. **Python Version**: Must be 3.9-3.12 (TensorFlow doesn't support 3.13+)

5. **Node Version**: Requires 18+

6. **Package Manager**: Frontend uses npm only (not yarn/pnpm)

7. **Rate Limiting**: Swarm analysis is 5/min. Use `user_id` query param for per-user limits.

8. **JSX Escaping**: Escape `>` as `{'>'}` in JSX text.

9. **Plotly**: Use `plotly.js-dist-min` package for ML charts.

10. **WebSocket**: Backend must be running first for WebSocket features.

## Key API Endpoints

```bash
# Health
GET  /health                    # Simple health check
GET  /health/detailed           # Component health

# ML Forecasting
GET  /api/forecast/all          # All 5 model forecasts
GET  /api/models/status         # Model availability

# Investor Reports
GET  /api/investor-report       # InvestorReport.v1 JSON (query: symbols, user_id, fresh)

# Swarm Analysis
POST /api/swarm/analyze         # Run analysis (rate limited: 5/min)

# WebSocket
WS   /ws/agent-stream/{user_id}       # Real-time agent events
WS   /ws/phase4-metrics/{user_id}     # Phase 4 metrics (30s interval)

# Cache
GET  /cache/stats               # Cache statistics
POST /cache/clear               # Clear all cache
```

## Environment Setup

```bash
cp .env.example .env
# System works with fallback data when API keys not configured
# Optional: OPENAI_API_KEY, ANTHROPIC_API_KEY, POLYGON_API_KEY
```

## Common Issues

### Port Already in Use (Windows)
```bash
netstat -ano | findstr :8000
taskkill /PID <PID> /F
# Or use: .\kill_port8000.ps1
```

### TensorFlow DLL Error (Windows)
TensorFlow is preloaded in `src/api/main.py`. Never change import order.

### Frontend Build Fails
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

## Performance Targets

- API: <500ms (cached p95)
- WebSocket: <50ms latency
- Frontend: <2s page load, <100ms render
- ML Models: <200ms Phase 4 computation per asset

## Git Workflow

- Main branch: `main`
- Feature branches: `claude/feature-name-*`
- Commits: Use conventional format (feat:, fix:, docs:)
