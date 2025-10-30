# Frontend Integration Plan: LLM V2 Upgrade (InvestorReport.v1)

**Goal**: Deliver Bloomberg Terminal / TradingView / Aladdin-quality UX for institutional-grade analytics

**Status**: 📋 Planning Phase  
**Target Completion**: 2-3 weeks  
**Complexity**: High (Bloomberg-level polish required)

---

## 1. Data Integration Layer

### 1.1 New API Endpoints

#### **Primary Endpoint: InvestorReport.v1**
```typescript
// GET /api/investor-report
interface InvestorReportResponse {
  as_of: string;                    // ISO 8601 timestamp
  universe: string[];                // Tickers analyzed
  executive_summary: ExecutiveSummary;
  risk_panel: RiskPanel;
  signals: Signals;
  actions: ActionItem[];
  sources: Source[];
  confidence: Confidence;
  metadata?: {
    schema_version: string;
    validated: boolean;
    fallback: boolean;
  };
}
```

**Backend Implementation** (FastAPI):
```python
# src/api/investor_report_routes.py (NEW)
from fastapi import APIRouter, HTTPException
from src.agents.swarm.agents.distillation_agent import DistillationAgent
from src.analytics.portfolio_metrics import PortfolioAnalytics

router = APIRouter(prefix="/api/investor-report", tags=["investor-report"])

@router.get("/")
async def get_investor_report(
    user_id: str,
    symbols: Optional[List[str]] = None,
    fresh: bool = False  # Bypass cache
):
    """
    Generate InvestorReport.v1 for user's portfolio.
    
    Returns:
        InvestorReport.v1 JSON with schema validation
    """
    try:
        # Get positions
        positions = await get_user_positions(user_id)
        
        # Get market data
        market_data = await get_market_data(symbols or extract_symbols(positions))
        
        # Compute portfolio metrics (including Phase 4)
        analytics = PortfolioAnalytics()
        metrics = analytics.calculate_all_metrics(positions, market_data)
        
        # Run distillation agent
        agent = DistillationAgent()
        report = agent.synthesize_swarm_output(
            categorized_insights=categorize_insights(positions, metrics),
            position_data={'symbols': symbols, 'metrics': metrics}
        )
        
        return report
    except Exception as e:
        logger.error(f"Error generating investor report: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

#### **WebSocket: Real-Time Phase 4 Metrics**
```typescript
// WS /ws/phase4-metrics/{user_id}
interface Phase4MetricsUpdate {
  type: 'phase4_update';
  timestamp: string;
  data: {
    options_flow_composite: number | null;
    residual_momentum: number | null;
    seasonality_score: number | null;
    breadth_liquidity: number | null;
    interpretations: {
      options_flow: string;
      residual_momentum: string;
      seasonality: string;
      breadth_liquidity: string;
    };
  };
}
```

**Backend Implementation**:
```python
# src/api/websocket_routes.py (MODIFY)
@app.websocket("/ws/phase4-metrics/{user_id}")
async def phase4_metrics_stream(websocket: WebSocket, user_id: str):
    """Stream real-time Phase 4 metrics updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Compute Phase 4 metrics every 30 seconds
            await asyncio.sleep(30)
            
            metrics = compute_phase4_metrics(
                # ... fetch latest data
            )
            
            await websocket.send_json({
                'type': 'phase4_update',
                'timestamp': datetime.utcnow().isoformat(),
                'data': metrics
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### 1.2 TypeScript Type Definitions

**File**: `frontend/src/types/investor-report.ts` (NEW)

```typescript
// Mirror InvestorReport.v1 JSON Schema
export interface InvestorReport {
  as_of: string;
  universe: string[];
  executive_summary: ExecutiveSummary;
  risk_panel: RiskPanel;
  signals: Signals;
  actions: ActionItem[];
  sources: Source[];
  confidence: Confidence;
  metadata?: ReportMetadata;
}

export interface RiskPanel {
  omega: number;
  gh1: number;
  pain_index: number;
  upside_capture: number;
  downside_capture: number;
  cvar_95: number;
  max_drawdown: number;
  explanations?: string[];
}

export interface Phase4Tech {
  options_flow_composite: number | null;
  residual_momentum: number | null;
  seasonality_score: number | null;
  breadth_liquidity: number | null;
  explanations?: string[];
}

export interface Signals {
  ml_alpha: { score: number; explanations?: string[] };
  regime: 'Low-Vol' | 'Normal' | 'High-Vol' | 'Crisis';
  sentiment: { level: number; delta: number; explanations?: string[] };
  smart_money: { thirteenF: number; insider_bias: number; options_bias: number; explanations?: string[] };
  alt_data: { digital_demand: number; earnings_surprise_pred: number; explanations?: string[] };
  phase4_tech: Phase4Tech;
}

export interface ActionItem {
  ticker: string;
  action: 'buy' | 'hold' | 'sell' | 'watch';
  sizing: string;
  risk_controls: string;
}

export interface Source {
  title: string;
  url: string;
  provider: 'Cboe' | 'SEC' | 'FRED' | 'ExtractAlpha' | 'AlphaSense' | 'LSEG' | 'local' | 'computed';
  as_of: string;
}

export interface Confidence {
  overall: number;  // 0-1
  drivers: string[];
}
```

### 1.3 API Service Updates

**File**: `frontend/src/services/api.ts` (MODIFY)

```typescript
class ApiService {
  // ... existing methods ...
  
  // InvestorReport.v1
  async getInvestorReport(userId: string, symbols?: string[], fresh = false): Promise<InvestorReport> {
    const params = new URLSearchParams({ user_id: userId, fresh: fresh.toString() });
    if (symbols) params.append('symbols', symbols.join(','));
    
    return this.request<InvestorReport>(`/investor-report?${params}`);
  }
  
  // Phase 4 metrics (snapshot)
  async getPhase4Metrics(symbols: string[]): Promise<Phase4Tech> {
    return this.request<Phase4Tech>('/phase4-metrics', {
      method: 'POST',
      body: JSON.stringify({ symbols })
    });
  }
}
```

### 1.4 WebSocket Hook for Phase 4

**File**: `frontend/src/hooks/usePhase4Stream.ts` (NEW)

```typescript
import { useEffect, useState } from 'react';
import { Phase4Tech } from '../types/investor-report';

export const usePhase4Stream = (userId: string) => {
  const [phase4Data, setPhase4Data] = useState<Phase4Tech | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  
  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws/phase4-metrics/${userId}`);
    
    ws.onopen = () => setIsConnected(true);
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === 'phase4_update') {
        setPhase4Data(message.data);
      }
    };
    ws.onclose = () => setIsConnected(false);
    
    return () => ws.close();
  }, [userId]);
  
  return { phase4Data, isConnected };
};
```

---

## 2. UI Component Architecture

### 2.1 Component Hierarchy

```
InvestorReportDashboard (NEW PAGE)
├── ReportHeader
│   ├── UniverseChips (tickers)
│   ├── TimestampBadge (as_of)
│   └── SchemaValidationIndicator (✓ InvestorReport.v1)
│
├── ExecutiveSummaryPanel
│   ├── TopPicksCards (3-column grid)
│   ├── KeyRisksAlert (red warning box)
│   └── ThesisStatement (large text)
│
├── RiskPanelDashboard ⭐ CORE COMPONENT
│   ├── MetricCard (Omega) - green/yellow/red thresholds
│   ├── MetricCard (GH1)
│   ├── MetricCard (Pain Index)
│   ├── MetricCard (Upside Capture)
│   ├── MetricCard (Downside Capture)
│   ├── MetricCard (CVaR 95%)
│   ├── MetricCard (Max Drawdown)
│   └── HistoricalTrendChart (sparklines for each metric)
│
├── Phase4SignalsPanel ⭐ CORE COMPONENT
│   ├── OptionsFlowGauge (real-time)
│   ├── ResidualMomentumChart
│   ├── SeasonalityCalendar
│   └── BreadthLiquidityHeatmap
│
├── SignalsOverviewPanel
│   ├── MLAlphaScore
│   ├── RegimeBadge (Low-Vol/Normal/High-Vol/Crisis)
│   ├── SentimentGauge
│   ├── SmartMoneyIndicators (13F, insider, options)
│   └── AltDataMetrics
│
├── ActionsTable
│   ├── ActionRow (ticker, buy/hold/sell/watch, sizing, risk controls)
│   └── BulkActionButtons
│
├── ProvenanceSidebar
│   ├── SourceCard (Cboe) - clickable link
│   ├── SourceCard (SEC)
│   ├── SourceCard (FRED)
│   └── SourceCard (ExtractAlpha)
│
└── ConfidenceFooter
    ├── ConfidenceGauge (0-1 scale)
    └── DriversList (expandable)
```

### 2.2 Core Component Specifications

#### **RiskPanelDashboard.tsx** (NEW)

```typescript
import React from 'react';
import { RiskPanel } from '../types/investor-report';
import { MetricCard } from './MetricCard';
import { Tooltip } from './Tooltip';

interface Props {
  riskPanel: RiskPanel;
  regime: 'Low-Vol' | 'Normal' | 'High-Vol' | 'Crisis';
}

export const RiskPanelDashboard: React.FC<Props> = ({ riskPanel, regime }) => {
  // Regime-aware styling: emphasize downside metrics in High-Vol/Crisis
  const emphasizeDownside = ['High-Vol', 'Crisis'].includes(regime);
  
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <MetricCard
        title="Omega Ratio"
        value={riskPanel.omega}
        tooltip="Probability-weighted ratio of gains to losses. >1.0 is good, >2.0 is excellent (Renaissance-level)."
        thresholds={{ excellent: 2.0, good: 1.0, warning: 0.5 }}
        format="decimal"
      />
      
      <MetricCard
        title="GH1 Ratio"
        value={riskPanel.gh1}
        tooltip="Graham-Harvey 1: Return enhancement + risk reduction vs vol-matched benchmark. Positive = outperformance."
        thresholds={{ excellent: 0.5, good: 0.2, warning: 0 }}
        format="decimal"
      />
      
      <MetricCard
        title="Pain Index"
        value={riskPanel.pain_index}
        tooltip="Ulcer Index: Depth × duration of drawdowns. Lower is better (investor 'stomachability')."
        thresholds={{ excellent: 5, good: 10, warning: 20 }}
        inverted  // Lower is better
        emphasized={emphasizeDownside}
      />
      
      <MetricCard
        title="Upside Capture"
        value={riskPanel.upside_capture}
        tooltip="% of benchmark gains captured. >100% = outperformance on up days."
        thresholds={{ excellent: 110, good: 100, warning: 90 }}
        format="percentage"
      />
      
      <MetricCard
        title="Downside Capture"
        value={riskPanel.downside_capture}
        tooltip="% of benchmark losses captured. <100% = protection on down days."
        thresholds={{ excellent: 80, good: 90, warning: 100 }}
        format="percentage"
        inverted
        emphasized={emphasizeDownside}
      />
      
      <MetricCard
        title="CVaR 95%"
        value={riskPanel.cvar_95}
        tooltip="Conditional Value at Risk: Expected loss in worst 5% of scenarios."
        thresholds={{ excellent: -5, good: -10, warning: -20 }}
        format="percentage"
        emphasized={emphasizeDownside}
      />
      
      <MetricCard
        title="Max Drawdown"
        value={riskPanel.max_drawdown}
        tooltip="Maximum peak-to-trough decline. Lower is better."
        thresholds={{ excellent: -10, good: -20, warning: -30 }}
        format="percentage"
        inverted
        emphasized={emphasizeDownside}
      />
    </div>
  );
};
```

#### **MetricCard.tsx** (NEW - Reusable)

```typescript
import React, { useState } from 'react';
import { TrendingUp, TrendingDown, Info } from 'lucide-react';

interface Props {
  title: string;
  value: number;
  tooltip: string;
  thresholds: { excellent: number; good: number; warning: number };
  format?: 'decimal' | 'percentage';
  inverted?: boolean;  // Lower is better
  emphasized?: boolean;  // Regime-aware emphasis
  sparklineData?: number[];  // Historical trend
}

export const MetricCard: React.FC<Props> = ({
  title,
  value,
  tooltip,
  thresholds,
  format = 'decimal',
  inverted = false,
  emphasized = false,
  sparklineData
}) => {
  const [showTooltip, setShowTooltip] = useState(false);
  
  // Determine color based on thresholds
  const getColor = () => {
    const { excellent, good, warning } = thresholds;
    
    if (inverted) {
      if (value <= excellent) return 'green';
      if (value <= good) return 'yellow';
      if (value <= warning) return 'orange';
      return 'red';
    } else {
      if (value >= excellent) return 'green';
      if (value >= good) return 'yellow';
      if (value >= warning) return 'orange';
      return 'red';
    }
  };
  
  const color = getColor();
  const colorClasses = {
    green: 'bg-green-900/20 border-green-500 text-green-400',
    yellow: 'bg-yellow-900/20 border-yellow-500 text-yellow-400',
    orange: 'bg-orange-900/20 border-orange-500 text-orange-400',
    red: 'bg-red-900/20 border-red-500 text-red-400'
  };
  
  const formattedValue = format === 'percentage' 
    ? `${value.toFixed(1)}%` 
    : value.toFixed(2);
  
  return (
    <div 
      className={`
        relative p-4 rounded-lg border-2 transition-all
        ${colorClasses[color]}
        ${emphasized ? 'ring-2 ring-white/50 shadow-lg' : ''}
        hover:scale-105 cursor-pointer
      `}
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-300">{title}</h3>
        <Info className="w-4 h-4 text-gray-400" />
      </div>
      
      {/* Value */}
      <div className="text-3xl font-bold mb-2">{formattedValue}</div>
      
      {/* Sparkline */}
      {sparklineData && (
        <div className="h-8">
          {/* Mini chart using recharts or custom SVG */}
        </div>
      )}
      
      {/* Tooltip */}
      {showTooltip && (
        <div className="absolute z-10 bottom-full left-0 mb-2 p-3 bg-gray-800 border border-gray-700 rounded-lg shadow-xl max-w-xs">
          <p className="text-sm text-gray-200">{tooltip}</p>
        </div>
      )}
    </div>
  );
};
```

---

## 3. Visual Design Standards

### 3.1 Color Palette (Bloomberg/TradingView-inspired)

```css
/* Dark Theme */
:root {
  /* Backgrounds */
  --bg-primary: #1a1a1a;
  --bg-secondary: #2a2a2a;
  --bg-tertiary: #3a3a3a;
  
  /* Text */
  --text-primary: #ffffff;
  --text-secondary: #b0b0b0;
  --text-tertiary: #808080;
  
  /* Metrics Colors */
  --color-bullish: #10b981;      /* Green */
  --color-bearish: #ef4444;      /* Red */
  --color-warning: #f59e0b;      /* Yellow/Orange */
  --color-neutral: #3b82f6;      /* Blue */
  
  /* Risk Levels */
  --risk-critical: #dc2626;      /* Red */
  --risk-high: #f97316;          /* Orange */
  --risk-medium: #eab308;        /* Yellow */
  --risk-low: #22c55e;           /* Green */
  
  /* Borders */
  --border-primary: #404040;
  --border-secondary: #505050;
}
```

### 3.2 Typography

```css
/* Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 14px;
  line-height: 1.5;
  color: var(--text-primary);
  background: var(--bg-primary);
}

/* Headings */
h1 { font-size: 32px; font-weight: 700; }
h2 { font-size: 24px; font-weight: 600; }
h3 { font-size: 18px; font-weight: 600; }
h4 { font-size: 16px; font-weight: 500; }

/* Metric Values */
.metric-value {
  font-size: 36px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}
```

### 3.3 Component Styling Guidelines

- **Cards**: `bg-[#2a2a2a]` with `border-[#404040]`, rounded corners `rounded-lg`
- **Hover States**: `hover:bg-[#3a3a3a]` with smooth transitions
- **Shadows**: Subtle `shadow-lg` for depth
- **Spacing**: Consistent 4px grid (p-4, gap-4, etc.)
- **Animations**: 60fps transitions, `transition-all duration-200 ease-in-out`

---

## 4. Advanced Features

### 4.1 Metric Tooltips

**Implementation**: Hover-triggered popovers with detailed explanations

```typescript
const METRIC_TOOLTIPS = {
  omega: {
    title: 'Omega Ratio',
    description: 'Probability-weighted ratio of gains to losses above a threshold (typically 0%).',
    interpretation: {
      '>2.0': 'Excellent - Renaissance Technologies level',
      '1.0-2.0': 'Good - Consistent positive edge',
      '0.5-1.0': 'Warning - Marginal performance',
      '<0.5': 'Critical - Losses exceed gains'
    },
    source: 'Keating & Shadwick (2002)',
    link: 'https://en.wikipedia.org/wiki/Omega_ratio'
  },
  gh1: {
    title: 'Graham-Harvey 1 Ratio',
    description: 'Measures return enhancement + risk reduction vs a volatility-matched benchmark.',
    interpretation: {
      '>0.5': 'Excellent - Strong alpha generation',
      '0.2-0.5': 'Good - Positive risk-adjusted returns',
      '0-0.2': 'Neutral - Marginal outperformance',
      '<0': 'Underperformance vs benchmark'
    },
    source: 'Graham & Harvey (1997)',
    link: 'https://faculty.fuqua.duke.edu/~charvey/'
  },
  pain_index: {
    title: 'Pain Index (Ulcer Index)',
    description: 'Measures depth and duration of drawdowns. Lower is better.',
    interpretation: {
      '<5': 'Excellent - Minimal drawdown pain',
      '5-10': 'Good - Acceptable volatility',
      '10-20': 'Warning - Significant drawdowns',
      '>20': 'Critical - High investor discomfort'
    },
    source: 'Peter Martin (1987)',
    link: 'https://en.wikipedia.org/wiki/Ulcer_index'
  }
  // ... other metrics
};
```

### 4.2 Drill-Down Capability

**Click any metric → Modal with**:
- Historical chart (30/90/365 days)
- Peer comparison (vs SPY, QQQ, sector ETF)
- Calculation breakdown
- Related metrics

```typescript
const MetricDetailModal: React.FC<{ metric: string; value: number }> = ({ metric, value }) => {
  return (
    <Modal>
      <h2>{METRIC_TOOLTIPS[metric].title}</h2>
      
      {/* Historical Chart */}
      <LineChart data={historicalData} />
      
      {/* Peer Comparison */}
      <BarChart data={peerComparison} />
      
      {/* Calculation */}
      <CodeBlock>{calculationFormula}</CodeBlock>
      
      {/* Related Metrics */}
      <RelatedMetrics metrics={['omega', 'sharpe', 'sortino']} />
    </Modal>
  );
};
```

### 4.3 Regime-Aware Styling

**Auto-adjust UI based on detected regime**:

```typescript
const getRegimeStyles = (regime: string) => {
  switch (regime) {
    case 'Crisis':
      return {
        emphasize: ['pain_index', 'downside_capture', 'cvar_95', 'max_drawdown'],
        alertColor: 'red',
        message: '⚠️ Crisis Mode: Focus on downside protection'
      };
    case 'High-Vol':
      return {
        emphasize: ['pain_index', 'cvar_95'],
        alertColor: 'orange',
        message: '⚡ High Volatility: Monitor risk metrics closely'
      };
    default:
      return {
        emphasize: [],
        alertColor: 'blue',
        message: '✓ Normal Market Conditions'
      };
  }
};
```

### 4.4 Schema Validation Indicator

**Visual feedback on report quality**:

```typescript
const SchemaValidationBadge: React.FC<{ metadata: ReportMetadata }> = ({ metadata }) => {
  if (metadata.validated && !metadata.fallback) {
    return (
      <div className="flex items-center gap-2 px-3 py-1 bg-green-900/20 border border-green-500 rounded-full">
        <CheckCircle className="w-4 h-4 text-green-400" />
        <span className="text-sm text-green-400">InvestorReport.v1 ✓</span>
      </div>
    );
  } else if (metadata.fallback) {
    return (
      <div className="flex items-center gap-2 px-3 py-1 bg-yellow-900/20 border border-yellow-500 rounded-full">
        <AlertTriangle className="w-4 h-4 text-yellow-400" />
        <span className="text-sm text-yellow-400">Fallback Mode</span>
      </div>
    );
  }
  
  return null;
};
```

---

## 5. Technical Implementation

### 5.1 Technology Stack

**Frontend Framework**: React 18 + TypeScript  
**State Management**: Zustand (already in use)  
**Chart Library**: **TradingView Lightweight Charts** (best for financial data)  
  - Alternative: ECharts (more customizable)  
  - Fallback: Recharts (simpler, less performant)

**Styling**: Tailwind CSS (already in use)  
**Icons**: Lucide React (already in use)  
**Real-time**: WebSocket (already implemented)

### 5.2 State Management

**Zustand Store Updates** (`frontend/src/store/index.ts`):

```typescript
interface StoreState {
  // ... existing state ...
  
  // InvestorReport.v1
  investorReport: InvestorReport | null;
  setInvestorReport: (report: InvestorReport) => void;
  
  // Phase 4 Metrics (real-time)
  phase4Metrics: Phase4Tech | null;
  setPhase4Metrics: (metrics: Phase4Tech) => void;
  
  // UI State
  selectedMetric: string | null;
  setSelectedMetric: (metric: string | null) => void;
  
  showMetricDetail: boolean;
  setShowMetricDetail: (show: boolean) => void;
}
```

### 5.3 Chart Library Selection

**Recommendation**: **TradingView Lightweight Charts**

**Rationale**:
- ✅ Built for financial data (OHLC, volume, indicators)
- ✅ 60fps performance with large datasets
- ✅ Professional Bloomberg-style aesthetics
- ✅ Real-time updates via WebSocket
- ✅ Mobile-responsive
- ✅ Free & open-source

**Installation**:
```bash
npm install lightweight-charts
```

**Example Usage**:
```typescript
import { createChart } from 'lightweight-charts';

const chart = createChart(container, {
  layout: {
    background: { color: '#1a1a1a' },
    textColor: '#b0b0b0',
  },
  grid: {
    vertLines: { color: '#2a2a2a' },
    horzLines: { color: '#2a2a2a' },
  },
});

const lineSeries = chart.addLineSeries({
  color: '#10b981',
  lineWidth: 2,
});

lineSeries.setData(historicalOmegaData);
```

### 5.4 Performance Targets

- **Initial Load**: <2s (including API call)
- **Render Time**: <100ms for full dashboard
- **API Response**: <500ms for InvestorReport.v1
- **WebSocket Latency**: <50ms for Phase 4 updates
- **Animations**: 60fps (16.67ms per frame)
- **Bundle Size**: <500KB (gzipped)

---

## 6. Integration with Existing System

### 6.1 Routing

**Add new route** (`frontend/src/App.tsx`):

```typescript
<Routes>
  <Route path="/" element={<Dashboard />} />
  <Route path="/positions" element={<PositionsPage />} />
  <Route path="/swarm-analysis" element={<SwarmAnalysisPage />} />
  <Route path="/investor-report" element={<InvestorReportDashboard />} />  {/* NEW */}
</Routes>
```

### 6.2 Navigation

**Update nav** (`frontend/src/App.tsx`):

```typescript
<nav className="bg-[#2a2a2a] shadow-lg border-b border-[#404040]">
  <div className="container mx-auto px-4 py-4">
    <div className="flex gap-6">
      <NavLink to="/">Dashboard</NavLink>
      <NavLink to="/positions">Positions</NavLink>
      <NavLink to="/swarm-analysis">AI Swarm</NavLink>
      <NavLink to="/investor-report" className="text-green-400">  {/* NEW */}
        📊 Investor Report
      </NavLink>
    </div>
  </div>
</nav>
```

### 6.3 Graceful Degradation

**Handle missing Phase 4 data**:

```typescript
const Phase4MetricCard: React.FC<{ value: number | null; label: string }> = ({ value, label }) => {
  if (value === null) {
    return (
      <div className="p-4 bg-gray-800/50 border border-gray-700 rounded-lg">
        <h4 className="text-sm text-gray-400">{label}</h4>
        <div className="mt-2 flex items-center gap-2">
          <Loader2 className="w-4 h-4 animate-spin text-gray-500" />
          <span className="text-sm text-gray-500">Computing...</span>
        </div>
      </div>
    );
  }
  
  return (
    <MetricCard title={label} value={value} {...props} />
  );
};
```

---

## 7. Deliverables

### 7.1 Component Hierarchy Diagram

```
[See Mermaid diagram in separate file: component-hierarchy.mmd]
```

### 7.2 API Endpoint Specifications

| Endpoint | Method | Purpose | Response Time |
|----------|--------|---------|---------------|
| `/api/investor-report` | GET | Full InvestorReport.v1 | <500ms |
| `/api/phase4-metrics` | POST | Phase 4 snapshot | <200ms |
| `/ws/phase4-metrics/{user_id}` | WS | Real-time Phase 4 stream | <50ms |

### 7.3 Wireframes

**[See separate file: wireframes/investor-report-dashboard.png]**

Key views:
1. Full dashboard (desktop 1920x1080)
2. Risk Panel detail modal
3. Phase 4 Signals panel
4. Mobile responsive (375x667)

### 7.4 Implementation Timeline

**Week 1: Backend + Data Layer**
- Day 1-2: Create `/api/investor-report` endpoint
- Day 3-4: Implement Phase 4 WebSocket stream
- Day 5: TypeScript types + API service updates

**Week 2: Core UI Components**
- Day 1-2: RiskPanelDashboard + MetricCard
- Day 3-4: Phase4SignalsPanel
- Day 5: ExecutiveSummaryPanel + ActionsTable

**Week 3: Polish + Integration**
- Day 1-2: Tooltips, drill-down modals, regime-aware styling
- Day 3-4: ProvenanceSidebar, ConfidenceFooter
- Day 5: Testing, performance optimization, documentation

### 7.5 Technology Stack Recommendations

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Frontend Framework | React 18 + TypeScript | Already in use, mature ecosystem |
| State Management | Zustand | Already in use, lightweight |
| Charts | TradingView Lightweight Charts | Best for financial data, 60fps |
| Styling | Tailwind CSS | Already in use, rapid development |
| Icons | Lucide React | Already in use, comprehensive |
| Real-time | WebSocket | Already implemented |
| Testing | Vitest + React Testing Library | Fast, modern |

---

## 8. Success Criteria

✅ **Visual Quality**: Matches Bloomberg Terminal / TradingView polish  
✅ **Schema Compliance**: 100% InvestorReport.v1 validation  
✅ **Performance**: <100ms render, <500ms API, 60fps animations  
✅ **Tooltips**: All 7 risk metrics have clear explanations  
✅ **Real-time**: Phase 4 updates every 30s via WebSocket  
✅ **Provenance**: All sources clickable and lead to authoritative URLs  
✅ **Graceful Degradation**: UI handles null Phase 4 data without breaking  
✅ **Regime-Aware**: UI emphasizes downside risk in High-Vol/Crisis modes  
✅ **Mobile**: Responsive design works on tablet (768px+)  

---

## 9. Next Steps

1. **Review & Approve** this plan
2. **Create wireframes** (Figma/Sketch)
3. **Backend implementation** (Week 1)
4. **Frontend components** (Week 2)
5. **Integration & polish** (Week 3)
6. **User testing** with sample portfolios
7. **Production deployment**

---

**Status**: 📋 Awaiting approval to proceed with implementation  
**Estimated Effort**: 120-150 hours (3 weeks, 1 developer)  
**Risk Level**: Medium (Bloomberg-level polish is challenging)  
**Dependencies**: Phase 4 metrics backend, InvestorReport.v1 schema validation

