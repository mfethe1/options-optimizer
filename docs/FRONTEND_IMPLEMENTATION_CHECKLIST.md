# Frontend LLM V2 Integration - Implementation Checklist

**Goal**: Bloomberg Terminal-quality UI for InvestorReport.v1  
**Timeline**: 3 weeks (120-150 hours)  
**Status**: 📋 Ready to start

---

## Week 1: Backend + Data Layer (40-50 hours)

### Day 1-2: API Endpoints (16 hours)

- [ ] **Create `src/api/investor_report_routes.py`**
  - [ ] `GET /api/investor-report` endpoint
  - [ ] Query params: `user_id`, `symbols`, `fresh`
  - [ ] Call `DistillationAgent.synthesize_swarm_output()`
  - [ ] Return InvestorReport.v1 JSON
  - [ ] Add to `main.py` router
  - [ ] Test with Postman/curl

- [ ] **Create `src/api/phase4_routes.py`**
  - [ ] `POST /api/phase4-metrics` endpoint (snapshot)
  - [ ] Call `compute_phase4_metrics()` from `technical_cross_asset.py`
  - [ ] Return Phase4Tech JSON
  - [ ] Add to `main.py` router
  - [ ] Test with sample data

- [ ] **Update `src/analytics/portfolio_metrics.py`**
  - [ ] Add Phase 4 fields to `PortfolioMetrics` dataclass
  - [ ] Update `calculate_all_metrics()` to compute Phase 4
  - [ ] Add `data_sources` and `as_of` fields
  - [ ] Test with unit tests

### Day 3-4: WebSocket Streaming (16 hours)

- [ ] **Create `src/api/websocket_routes.py` (or modify existing)**
  - [ ] `WS /ws/phase4-metrics/{user_id}` endpoint
  - [ ] Stream Phase 4 updates every 30 seconds
  - [ ] Handle connection/disconnection
  - [ ] Add heartbeat mechanism
  - [ ] Test with WebSocket client

- [ ] **Create background task for Phase 4 computation**
  - [ ] Async task to compute Phase 4 every 30s
  - [ ] Broadcast to connected WebSocket clients
  - [ ] Handle errors gracefully
  - [ ] Log performance metrics

### Day 5: TypeScript Types + API Service (8 hours)

- [ ] **Create `frontend/src/types/investor-report.ts`**
  - [ ] `InvestorReport` interface
  - [ ] `RiskPanel` interface
  - [ ] `Phase4Tech` interface
  - [ ] `Signals` interface
  - [ ] `ActionItem` interface
  - [ ] `Source` interface
  - [ ] `Confidence` interface
  - [ ] Export all types

- [ ] **Update `frontend/src/services/api.ts`**
  - [ ] `getInvestorReport()` method
  - [ ] `getPhase4Metrics()` method
  - [ ] Type annotations
  - [ ] Error handling

- [ ] **Create `frontend/src/hooks/usePhase4Stream.ts`**
  - [ ] WebSocket connection hook
  - [ ] State management for Phase 4 data
  - [ ] Reconnection logic
  - [ ] Cleanup on unmount

---

## Week 2: Core UI Components (50-60 hours)

### Day 1-2: RiskPanelDashboard (16 hours)

- [ ] **Create `frontend/src/components/RiskPanelDashboard.tsx`**
  - [ ] 7-metric grid layout (2x4 on desktop, 1x7 on mobile)
  - [ ] Regime-aware styling (emphasize downside in High-Vol/Crisis)
  - [ ] Props: `riskPanel`, `regime`
  - [ ] Responsive design

- [ ] **Create `frontend/src/components/MetricCard.tsx`**
  - [ ] Color-coded thresholds (green/yellow/orange/red)
  - [ ] Tooltip on hover
  - [ ] Sparkline chart (optional)
  - [ ] Format helpers (decimal, percentage)
  - [ ] Inverted mode (lower is better)
  - [ ] Emphasized mode (regime-aware)
  - [ ] Smooth transitions

- [ ] **Create metric tooltip content**
  - [ ] `METRIC_TOOLTIPS` constant with explanations
  - [ ] Omega, GH1, Pain Index, Upside/Downside Capture, CVaR, MaxDD
  - [ ] Include interpretation thresholds
  - [ ] Add source citations

### Day 3-4: Phase4SignalsPanel (20 hours)

- [ ] **Create `frontend/src/components/Phase4SignalsPanel.tsx`**
  - [ ] 2x2 grid layout
  - [ ] Real-time indicator (green dot when connected)
  - [ ] Explanations section
  - [ ] Props: `userId`, `initialData`

- [ ] **Create `frontend/src/components/OptionsFlowGauge.tsx`**
  - [ ] SVG gauge/speedometer (-1 to +1)
  - [ ] Color-coded (green=bullish, red=bearish)
  - [ ] Breakdown: PCR, IV skew, volume
  - [ ] Tooltip
  - [ ] Smooth animations

- [ ] **Create `frontend/src/components/ResidualMomentumChart.tsx`**
  - [ ] Bar chart (Recharts or TradingView)
  - [ ] Z-score interpretation
  - [ ] Historical 5-day trend
  - [ ] Color-coded bars
  - [ ] Tooltip

- [ ] **Create `frontend/src/components/SeasonalityCalendar.tsx`**
  - [ ] Day-of-week heatmap
  - [ ] Turn-of-month indicator
  - [ ] Current day highlight
  - [ ] Tooltip

- [ ] **Create `frontend/src/components/BreadthLiquidityHeatmap.tsx`**
  - [ ] Component breakdown (A/D, volume, spreads)
  - [ ] Progress bars
  - [ ] Color-coded
  - [ ] Tooltip

### Day 5: ExecutiveSummaryPanel + ActionsTable (8 hours)

- [ ] **Create `frontend/src/components/ExecutiveSummaryPanel.tsx`**
  - [ ] Top picks cards (3-column grid)
  - [ ] Key risks alert (red warning box)
  - [ ] Thesis statement (large text)
  - [ ] Props: `executiveSummary`

- [ ] **Create `frontend/src/components/ActionsTable.tsx`**
  - [ ] Table with columns: Ticker, Action, Sizing, Risk Controls
  - [ ] Color-coded actions (buy=green, sell=red, hold=yellow, watch=blue)
  - [ ] Bulk action buttons
  - [ ] Props: `actions`

---

## Week 3: Polish + Integration (30-40 hours)

### Day 1-2: Additional Components (16 hours)

- [ ] **Create `frontend/src/components/ReportHeader.tsx`**
  - [ ] Universe chips (tickers)
  - [ ] Timestamp badge
  - [ ] Schema validation indicator (✓ InvestorReport.v1)
  - [ ] Props: `universe`, `as_of`, `metadata`

- [ ] **Create `frontend/src/components/SignalsOverviewPanel.tsx`**
  - [ ] ML Alpha score
  - [ ] Regime badge (Low-Vol/Normal/High-Vol/Crisis)
  - [ ] Sentiment gauge
  - [ ] Smart money indicators (13F, insider, options)
  - [ ] Alt data metrics
  - [ ] Props: `signals`

- [ ] **Create `frontend/src/components/ProvenanceSidebar.tsx`**
  - [ ] Source cards (Cboe, SEC, FRED, ExtractAlpha, etc.)
  - [ ] Clickable links
  - [ ] Provider logos (optional)
  - [ ] As-of dates
  - [ ] Props: `sources`

- [ ] **Create `frontend/src/components/ConfidenceFooter.tsx`**
  - [ ] Confidence gauge (0-1 scale, circular progress)
  - [ ] Drivers list (expandable)
  - [ ] Props: `confidence`

### Day 3: Main Dashboard Page (8 hours)

- [ ] **Create `frontend/src/pages/InvestorReportDashboard.tsx`**
  - [ ] Fetch InvestorReport.v1 on mount
  - [ ] Integrate all sub-components
  - [ ] Loading states
  - [ ] Error handling
  - [ ] Refresh button
  - [ ] Export to PDF button (optional)

- [ ] **Update `frontend/src/App.tsx`**
  - [ ] Add `/investor-report` route
  - [ ] Add navigation link
  - [ ] Update nav styling

- [ ] **Update `frontend/src/store/index.ts`**
  - [ ] Add `investorReport` state
  - [ ] Add `phase4Metrics` state
  - [ ] Add `selectedMetric` state
  - [ ] Add setters

### Day 4: Advanced Features (8 hours)

- [ ] **Create `frontend/src/components/MetricDetailModal.tsx`**
  - [ ] Historical chart (30/90/365 days)
  - [ ] Peer comparison (vs SPY, QQQ, sector)
  - [ ] Calculation breakdown
  - [ ] Related metrics
  - [ ] Close button

- [ ] **Add drill-down capability**
  - [ ] Click MetricCard → open MetricDetailModal
  - [ ] Pass metric name and historical data
  - [ ] Smooth modal transitions

- [ ] **Add regime-aware styling**
  - [ ] `getRegimeStyles()` helper
  - [ ] Auto-emphasize downside metrics in High-Vol/Crisis
  - [ ] Regime alert banner

- [ ] **Add schema validation indicator**
  - [ ] Green checkmark when validated
  - [ ] Yellow warning on fallback mode
  - [ ] Tooltip with details

### Day 5: Testing + Documentation (8 hours)

- [ ] **Unit tests**
  - [ ] `RiskPanelDashboard.test.tsx`
  - [ ] `Phase4SignalsPanel.test.tsx`
  - [ ] `MetricCard.test.tsx`
  - [ ] `usePhase4Stream.test.ts`
  - [ ] Target: >80% coverage

- [ ] **Integration tests**
  - [ ] Full dashboard render
  - [ ] API integration
  - [ ] WebSocket connection
  - [ ] Error states

- [ ] **Performance testing**
  - [ ] Lighthouse audit (target: >90 performance score)
  - [ ] Bundle size analysis (<500KB gzipped)
  - [ ] Render time (<100ms)
  - [ ] WebSocket latency (<50ms)

- [ ] **Documentation**
  - [ ] Component API docs (JSDoc)
  - [ ] Usage examples
  - [ ] Storybook stories (optional)
  - [ ] README updates

---

## Post-Implementation: Enhancements (Optional)

### Immediate Enhancements
- [ ] Export to PDF (jsPDF + html2canvas)
- [ ] Historical report comparison
- [ ] Email/SMS alerts for critical metrics
- [ ] Dark/light theme toggle

### Future Features
- [ ] Real-time chart annotations
- [ ] Custom metric thresholds
- [ ] Portfolio scenario analysis
- [ ] Backtesting recommendations
- [ ] Mobile app (React Native)

---

## Quality Gates

### Before Week 2
- [ ] All API endpoints return valid InvestorReport.v1 JSON
- [ ] WebSocket streams Phase 4 updates every 30s
- [ ] TypeScript types match JSON Schema exactly
- [ ] API service has 100% type coverage

### Before Week 3
- [ ] RiskPanelDashboard renders all 7 metrics correctly
- [ ] Phase4SignalsPanel shows all 4 visualizations
- [ ] Tooltips explain all metrics clearly
- [ ] Real-time updates work smoothly (60fps)

### Before Production
- [ ] All unit tests pass (>80% coverage)
- [ ] Lighthouse performance score >90
- [ ] Bundle size <500KB gzipped
- [ ] No console errors or warnings
- [ ] Responsive design works on tablet (768px+)
- [ ] Schema validation indicator works
- [ ] Provenance links lead to correct URLs

---

## Risk Mitigation

### High-Risk Items
1. **Bloomberg-level polish** - Allocate extra time for visual refinement
2. **Real-time WebSocket** - Test thoroughly with network issues
3. **Chart performance** - Use TradingView Lightweight Charts for 60fps
4. **Bundle size** - Code-split large components

### Contingency Plans
- **If behind schedule**: Cut optional features (PDF export, drill-down modals)
- **If WebSocket fails**: Fall back to polling every 30s
- **If charts lag**: Use simpler visualizations (Recharts instead of TradingView)
- **If bundle too large**: Lazy-load Phase4SignalsPanel

---

## Success Metrics

- [ ] **Visual Quality**: Matches Bloomberg Terminal / TradingView
- [ ] **Performance**: <100ms render, <500ms API, 60fps animations
- [ ] **Completeness**: All InvestorReport.v1 fields displayed
- [ ] **Usability**: Users understand metrics without external research
- [ ] **Reliability**: Handles missing data gracefully
- [ ] **Accessibility**: WCAG 2.1 AA compliance (optional)

---

## Where to Find Results

### Files Created (Backend)
- `src/api/investor_report_routes.py`
- `src/api/phase4_routes.py`
- `src/api/websocket_routes.py` (or modified)

### Files Created (Frontend)
- `frontend/src/types/investor-report.ts`
- `frontend/src/hooks/usePhase4Stream.ts`
- `frontend/src/pages/InvestorReportDashboard.tsx`
- `frontend/src/components/RiskPanelDashboard.tsx`
- `frontend/src/components/MetricCard.tsx`
- `frontend/src/components/Phase4SignalsPanel.tsx`
- `frontend/src/components/OptionsFlowGauge.tsx`
- `frontend/src/components/ResidualMomentumChart.tsx`
- `frontend/src/components/SeasonalityCalendar.tsx`
- `frontend/src/components/BreadthLiquidityHeatmap.tsx`
- `frontend/src/components/ExecutiveSummaryPanel.tsx`
- `frontend/src/components/ActionsTable.tsx`
- `frontend/src/components/ReportHeader.tsx`
- `frontend/src/components/SignalsOverviewPanel.tsx`
- `frontend/src/components/ProvenanceSidebar.tsx`
- `frontend/src/components/ConfidenceFooter.tsx`
- `frontend/src/components/MetricDetailModal.tsx`

### Files Modified
- `frontend/src/App.tsx` (routing + nav)
- `frontend/src/store/index.ts` (state management)
- `frontend/src/services/api.ts` (API methods)
- `src/analytics/portfolio_metrics.py` (Phase 4 fields)

### Tests Created
- `frontend/src/components/__tests__/RiskPanelDashboard.test.tsx`
- `frontend/src/components/__tests__/Phase4SignalsPanel.test.tsx`
- `frontend/src/components/__tests__/MetricCard.test.tsx`
- `frontend/src/hooks/__tests__/usePhase4Stream.test.ts`

---

**Status**: 📋 Ready to start implementation  
**Next Step**: Begin Week 1, Day 1 - Create API endpoints  
**Estimated Completion**: 3 weeks from start date

