# InvestorReport.v1 Dashboard - UI Mockup Specification

**Purpose**: Visual specification for Bloomberg Terminal-quality dashboard  
**Target Resolution**: 1920x1080 (desktop primary), 768px+ (tablet secondary)  
**Theme**: Dark mode (Bloomberg/TradingView-inspired)

---

## Layout Structure (Desktop 1920x1080)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ HEADER (80px height)                                                        │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ 📊 Investor Report                    [AAPL, MSFT, GOOGL]  ✓ v1  🔴 Live│ │
│ │ Generated: 2025-10-19 14:32 ET                                          │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│ EXECUTIVE SUMMARY (200px height)                                            │
│ ┌──────────────────────┬──────────────────────┬──────────────────────────┐ │
│ │ TOP PICK #1          │ TOP PICK #2          │ TOP PICK #3              │ │
│ │ AAPL                 │ MSFT                 │ GOOGL                    │ │
│ │ Strong momentum +    │ Cloud growth +       │ AI leadership +          │ │
│ │ options flow         │ enterprise demand    │ search dominance         │ │
│ │ 30-day horizon       │ 60-day horizon       │ 45-day horizon           │ │
│ └──────────────────────┴──────────────────────┴──────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ ⚠️ KEY RISKS: Fed rate uncertainty, Tech sector rotation, Macro headwinds│ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│ Thesis: Tech leaders show strong fundamentals with options flow support... │
├─────────────────────────────────────────────────────────────────────────────┤
│ RISK PANEL (300px height)                                                   │
│ ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐ │
│ │ Omega  │  GH1   │  Pain  │ Upside │ Down-  │ CVaR   │  Max   │ Trend  │ │
│ │ Ratio  │ Ratio  │ Index  │Capture │ side   │  95%   │Drawdown│ Chart  │ │
│ │        │        │        │        │Capture │        │        │        │ │
│ │  2.15  │  0.42  │  8.3   │ 112%   │  85%   │ -7.2%  │ -12.5% │ ▁▂▃▅▇  │ │
│ │ 🟢 EXC │ 🟢 GOOD│ 🟢 EXC │ 🟢 OUT │ 🟢 PROT│ 🟢 LOW │ 🟢 MOD │        │ │
│ │        │        │        │        │        │        │        │        │ │
│ │ [ℹ️]   │ [ℹ️]   │ [ℹ️]   │ [ℹ️]   │ [ℹ️]   │ [ℹ️]   │ [ℹ️]   │        │ │
│ └────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│ PHASE 4 SIGNALS (400px height)                                              │
│ ┌──────────────────────────────┬──────────────────────────────────────────┐ │
│ │ OPTIONS FLOW COMPOSITE       │ RESIDUAL MOMENTUM                        │ │
│ │                              │                                          │ │
│ │        ╱───────╲             │  2.5 ┤                                   │ │
│ │       ╱    ●    ╲            │  2.0 ┤                                   │ │
│ │      ╱     │     ╲           │  1.5 ┤              ███                  │ │
│ │     ╱      │      ╲          │  1.0 ┤         ███  ███                  │ │
│ │    ╱       │       ╲         │  0.5 ┤    ███  ███  ███                  │ │
│ │   ╱        │        ╲        │  0.0 ┼────███──███──███──────────────    │ │
│ │  ╱         │         ╲       │ -0.5 ┤                                   │ │
│ │ ╱          │          ╲      │      └─────────────────────────────────  │ │
│ │ -1    0.65 (BULLISH)   +1    │       -5d  -4d  -3d  -2d  -1d  Today     │ │
│ │                              │                                          │ │
│ │ PCR: 0.72  Skew: -3.2%       │ Z-Score: 1.85σ (Mild Outperformance)     │ │
│ │ Volume: 1.8x                 │                                          │ │
│ ├──────────────────────────────┼──────────────────────────────────────────┤ │
│ │ SEASONALITY SCORE            │ BREADTH & LIQUIDITY                      │ │
│ │                              │                                          │ │
│ │ Day-of-Week Pattern:         │ Advance/Decline:  ████████░░ 80%         │ │
│ │ ┌───┬───┬───┬───┬───┐        │ Volume Ratio:     ██████░░░░ 60%         │ │
│ │ │Mon│Tue│Wed│Thu│Fri│        │ Spread Tightness: ██████████ 95%         │ │
│ │ │🟢 │🔴 │🟢 │🟢 │🔴 │        │                                          │ │
│ │ │+15│-5 │+8 │+12│-10│        │ Overall: 78% (Strong)                    │ │
│ │ └───┴───┴───┴───┴───┘        │                                          │ │
│ │                              │                                          │ │
│ │ Turn-of-Month: ✓ Active      │                                          │ │
│ │ Score: 0.42 (Mild Positive)  │                                          │ │
│ └──────────────────────────────┴──────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│ SIGNALS OVERVIEW (200px height)                                             │
│ ┌────────┬────────┬────────┬────────────────────┬────────────────────────┐ │
│ │ML Alpha│ Regime │Sentiment│ Smart Money        │ Alt Data               │ │
│ │        │        │         │                    │                        │ │
│ │  0.72  │ Normal │  0.45   │ 13F: +0.35         │ Digital Demand: 0.68   │ │
│ │ 🟢 HIGH│ 🔵 NORM│ 🟢 POS  │ Insider: +0.12     │ Earnings Pred: 0.55    │ │
│ │        │        │         │ Options: +0.28     │                        │ │
│ └────────┴────────┴────────┴────────────────────┴────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│ ACTIONS TABLE (200px height)                                                │
│ ┌────────┬────────┬──────────────────┬────────────────────────────────────┐ │
│ │ Ticker │ Action │ Sizing           │ Risk Controls                      │ │
│ ├────────┼────────┼──────────────────┼────────────────────────────────────┤ │
│ │ AAPL   │ 🟢 BUY │ +200 bps         │ Stop: $175, Target: $195           │ │
│ │ MSFT   │ 🟡 HOLD│ Maintain current │ Trailing stop: 8%                  │ │
│ │ GOOGL  │ 🟢 BUY │ +150 bps         │ Stop: $135, Target: $155           │ │
│ └────────┴────────┴──────────────────┴────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│ FOOTER (100px height)                                                       │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ CONFIDENCE: 0.82 (High)                                                 │ │
│ │ ████████████████░░░░                                                    │ │
│ │ Drivers: Strong options flow, Positive momentum, Favorable seasonality │ │
│ │                                                                         │ │
│ │ SOURCES: [Cboe] [SEC] [FRED] [ExtractAlpha] [AlphaSense] [LSEG]       │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Color Specifications

### Background Colors
```css
--bg-primary: #1a1a1a;      /* Main background */
--bg-secondary: #2a2a2a;    /* Cards, panels */
--bg-tertiary: #3a3a3a;     /* Hover states */
--bg-header: #2a2a2a;       /* Header background */
--bg-footer: #2a2a2a;       /* Footer background */
```

### Text Colors
```css
--text-primary: #ffffff;    /* Main text */
--text-secondary: #b0b0b0;  /* Secondary text */
--text-tertiary: #808080;   /* Muted text */
--text-label: #a0a0a0;      /* Labels */
```

### Metric Colors
```css
--metric-excellent: #10b981;  /* Green - Excellent */
--metric-good: #84cc16;       /* Light green - Good */
--metric-warning: #f59e0b;    /* Orange - Warning */
--metric-critical: #ef4444;   /* Red - Critical */
--metric-neutral: #3b82f6;    /* Blue - Neutral */
```

### Action Colors
```css
--action-buy: #10b981;      /* Green */
--action-sell: #ef4444;     /* Red */
--action-hold: #f59e0b;     /* Yellow */
--action-watch: #3b82f6;    /* Blue */
```

### Border Colors
```css
--border-primary: #404040;
--border-secondary: #505050;
--border-accent: #10b981;   /* Green accent */
```

---

## Typography Specifications

### Font Family
```css
font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
```

### Font Sizes
```css
--font-xs: 10px;      /* Labels, footnotes */
--font-sm: 12px;      /* Secondary text */
--font-base: 14px;    /* Body text */
--font-lg: 16px;      /* Subheadings */
--font-xl: 18px;      /* Section titles */
--font-2xl: 24px;     /* Panel titles */
--font-3xl: 32px;     /* Page title */
--font-metric: 36px;  /* Metric values */
```

### Font Weights
```css
--font-regular: 400;
--font-medium: 500;
--font-semibold: 600;
--font-bold: 700;
```

---

## Component Dimensions

### Header
- Height: 80px
- Padding: 16px 24px
- Border-bottom: 1px solid #404040

### Executive Summary
- Height: 200px
- Top Picks: 3-column grid, gap: 16px
- Key Risks: Full-width alert box, height: 60px
- Thesis: Full-width text, height: 40px

### Risk Panel
- Height: 300px
- Metric Cards: 8-column grid (7 metrics + 1 trend chart)
- Card padding: 16px
- Gap: 12px

### Phase 4 Signals
- Height: 400px
- 2x2 grid layout
- Gap: 16px
- Each panel: 192px height

### Signals Overview
- Height: 200px
- 5-column grid
- Gap: 12px

### Actions Table
- Height: 200px (variable based on rows)
- Row height: 48px
- Header height: 40px

### Footer
- Height: 100px
- Confidence gauge: 60px height
- Sources: 40px height

---

## Interactive States

### Hover States
```css
/* Metric Card Hover */
.metric-card:hover {
  transform: scale(1.05);
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
  transition: all 200ms ease-in-out;
}

/* Button Hover */
.button:hover {
  background: #3a3a3a;
  border-color: #10b981;
}

/* Link Hover */
.link:hover {
  color: #10b981;
  text-decoration: underline;
}
```

### Active States
```css
/* Metric Card Active (clicked) */
.metric-card:active {
  transform: scale(0.98);
}

/* Selected State */
.selected {
  border: 2px solid #10b981;
  box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2);
}
```

### Loading States
```css
/* Skeleton Loader */
.skeleton {
  background: linear-gradient(
    90deg,
    #2a2a2a 25%,
    #3a3a3a 50%,
    #2a2a2a 75%
  );
  background-size: 200% 100%;
  animation: loading 1.5s ease-in-out infinite;
}

@keyframes loading {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
```

---

## Responsive Breakpoints

### Desktop (1920x1080)
- Full 8-column grid for Risk Panel
- 2x2 grid for Phase 4 Signals
- 5-column grid for Signals Overview

### Laptop (1366x768)
- 4x2 grid for Risk Panel (4 metrics per row)
- 2x2 grid for Phase 4 Signals (unchanged)
- 3-column grid for Signals Overview (ML Alpha + Regime + Sentiment on row 1, Smart Money + Alt Data on row 2)

### Tablet (768px)
- 2x4 grid for Risk Panel (2 metrics per row)
- 1x4 grid for Phase 4 Signals (stacked vertically)
- 1-column grid for Signals Overview (stacked)

### Mobile (<768px)
- Not primary target, but should degrade gracefully
- 1-column layout for all components
- Horizontal scroll for tables

---

## Animation Specifications

### Transitions
```css
/* Smooth transitions for all interactive elements */
transition: all 200ms ease-in-out;

/* Specific transitions */
transition-property: transform, box-shadow, background-color, border-color;
transition-duration: 200ms;
transition-timing-function: ease-in-out;
```

### Keyframe Animations
```css
/* Pulse animation for live indicator */
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.live-indicator {
  animation: pulse 2s ease-in-out infinite;
}

/* Fade in animation for new data */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.fade-in {
  animation: fadeIn 300ms ease-out;
}
```

---

## Accessibility (WCAG 2.1 AA)

### Color Contrast
- Text on dark background: Minimum 4.5:1 ratio
- Large text (18px+): Minimum 3:1 ratio
- Interactive elements: Minimum 3:1 ratio

### Focus States
```css
/* Keyboard focus indicator */
:focus-visible {
  outline: 2px solid #10b981;
  outline-offset: 2px;
}

/* Skip to content link */
.skip-link {
  position: absolute;
  top: -40px;
  left: 0;
  background: #10b981;
  color: #1a1a1a;
  padding: 8px;
  z-index: 100;
}

.skip-link:focus {
  top: 0;
}
```

### ARIA Labels
```html
<!-- Metric Card -->
<div role="article" aria-label="Omega Ratio: 2.15 (Excellent)">
  <h3>Omega Ratio</h3>
  <div aria-live="polite">2.15</div>
</div>

<!-- Live Region for Real-Time Updates -->
<div aria-live="polite" aria-atomic="true">
  Phase 4 metrics updated
</div>
```

---

## Icon Usage

### Lucide React Icons
```typescript
import {
  TrendingUp,      // Bullish indicator
  TrendingDown,    // Bearish indicator
  Info,            // Tooltip trigger
  AlertTriangle,   // Warning/risk
  CheckCircle,     // Success/validation
  XCircle,         // Error
  Activity,        // Market activity
  Calendar,        // Seasonality
  BarChart3,       // Charts
  Loader2          // Loading spinner
} from 'lucide-react';
```

### Icon Sizes
- Small: 16px (inline with text)
- Medium: 20px (buttons, cards)
- Large: 24px (section headers)
- XLarge: 32px (page title)

---

## Performance Considerations

### Lazy Loading
```typescript
// Lazy load Phase 4 panel (below fold)
const Phase4SignalsPanel = lazy(() => import('./Phase4SignalsPanel'));

// Render with Suspense
<Suspense fallback={<SkeletonLoader />}>
  <Phase4SignalsPanel />
</Suspense>
```

### Memoization
```typescript
// Memoize expensive components
const MetricCard = React.memo(({ value, title, ...props }) => {
  // Component logic
});

// Memoize selectors
const selectRiskPanel = useMemo(
  () => investorReport?.risk_panel,
  [investorReport]
);
```

### Debouncing
```typescript
// Debounce WebSocket updates
const debouncedUpdate = useMemo(
  () => debounce((data) => setPhase4Data(data), 1000),
  []
);
```

---

**Status**: 📋 Visual specification complete  
**Next Step**: Create Figma/Sketch mockups (optional)  
**Implementation**: Follow `FRONTEND_IMPLEMENTATION_CHECKLIST.md`

