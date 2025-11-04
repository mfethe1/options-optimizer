# 📊 Chart Integration Implementation Guide

**Step-by-step guide to upgrade existing pages with TradingView Lightweight Charts**

---

## 🎯 Overview

This guide shows how to integrate the new professional charting system into existing pages like EnsembleAnalysisPage, EpidemicVolatilityPage, and others.

### Benefits of Upgrading

- **100K+ data points** instead of current 2-6 point limitations
- **60 FPS rendering** instead of sluggish Recharts performance
- **Professional appearance** matching Bloomberg Terminal
- **Technical indicators** built-in (SMA, EMA, RSI, MACD, etc.)
- **Multi-timeframe views** for better analysis
- **Tiny bundle size** (45KB vs current Recharts bloat)

---

## 📋 Pages to Upgrade

### Priority 1 (High Impact)

1. **EnsembleAnalysisPage** - Replace simple line charts with candlesticks + predictions
2. **EpidemicVolatilityPage** - Add multi-timeframe view for SEIR → VIX mapping
3. **MLPredictionsPage** - Add prediction cones with quantile bands
4. **ChartAnalysisPage** - Replace basic charts with full technical analysis

### Priority 2 (Medium Impact)

5. **GNNPage** - Add correlation network visualization
6. **PINNPage** - Add 3D Greeks surface charts (use Plotly separately)
7. **MambaPage** - Add long-sequence candlestick charts

### Priority 3 (Enhancement)

8. **Dashboard** - Add mini candlestick widgets
9. **PositionsPage** - Add P&L charts per position
10. **RiskDashboardPage** - Add portfolio value chart

---

## 🔧 Step-by-Step Integration

### Example 1: EnsembleAnalysisPage Upgrade

**Current State** (using Recharts with 6 data points):
```tsx
import { LineChart, Line, XAxis, YAxis } from 'recharts';

function EnsembleAnalysisPage() {
  const mockData = [
    { name: '1D', value: 158.23 },
    { name: '5D', value: 162.45 },
    // ... only 6 points
  ];

  return (
    <LineChart width={600} height={300} data={mockData}>
      <Line type="monotone" dataKey="value" stroke="#8884d8" />
      <XAxis dataKey="name" />
      <YAxis />
    </LineChart>
  );
}
```

**New Implementation** (TradingView with full features):
```tsx
import { CandlestickChart, generateSampleData } from '@/components/charts';

function EnsembleAnalysisPage() {
  // Generate realistic data (or fetch from API)
  const historicalData = generateSampleData(365, 150);

  // Add ensemble predictions as indicators
  const indicators = [
    { type: 'sma', period: 20, label: 'LSTM Forecast', color: '#2196f3' },
    { type: 'sma', period: 50, label: 'GNN Forecast', color: '#ff9800' },
    { type: 'ema', period: 12, label: 'TFT Forecast', color: '#4caf50' },
  ];

  return (
    <div>
      <h1>Ensemble Neural Network Analysis</h1>

      <CandlestickChart
        symbol="AAPL"
        data={historicalData}
        interval="1d"
        theme="dark"
        showVolume={true}
        showControls={true}
        indicators={indicators}
        height={600}
      />

      {/* Keep existing stats below chart */}
      <div className="stats-grid">
        {/* ... existing stats */}
      </div>
    </div>
  );
}
```

**Benefits:**
- ✅ Professional candlestick chart instead of basic line
- ✅ Volume histogram showing trading activity
- ✅ Multi-model predictions as overlays
- ✅ 365 days of data instead of 6 points
- ✅ Interactive crosshair tooltips
- ✅ Interval switching (1d, 1w, 1M)

---

### Example 2: EpidemicVolatilityPage with Multi-Timeframe

**Current State:**
```tsx
// Two separate Recharts
<ResponsiveContainer>
  <LineChart data={seirData}>
    {/* SEIR curves */}
  </LineChart>
</ResponsiveContainer>

<ResponsiveContainer>
  <LineChart data={volatilityData}>
    {/* VIX mapping */}
  </LineChart>
</ResponsiveContainer>
```

**New Implementation:**
```tsx
import { MultiTimeframeChart } from '@/components/charts';

function EpidemicVolatilityPage() {
  // Prepare multi-timeframe data
  const chartData = {
    '1d': dailyVolatilityData,
    '1w': weeklyVolatilityData,
    '1M': monthlyVolatilityData,
  };

  return (
    <div>
      <h1>Epidemic → Volatility Forecasting</h1>

      {/* SEIR Model Visualization (keep existing) */}
      <div className="seir-visualization">
        {/* ... existing SEIR curves */}
      </div>

      {/* VIX Multi-Timeframe Analysis */}
      <MultiTimeframeChart
        symbol="VIX"
        data={chartData}
        layout="1x2"  // Side-by-side comparison
        theme="dark"
        showVolume={false}  // VIX doesn't have volume
      />

      {/* Forecast Stats */}
      <div className="forecast-stats">
        {/* ... existing predictions */}
      </div>
    </div>
  );
}
```

**Benefits:**
- ✅ Side-by-side timeframe comparison
- ✅ Professional VIX charting
- ✅ Synchronized crosshairs
- ✅ Better context for epidemic → volatility mapping

---

### Example 3: MLPredictionsPage with Prediction Cones

**New Feature: Uncertainty Visualization**

```tsx
import { CandlestickChart } from '@/components/charts';

function MLPredictionsPage() {
  // Historical data
  const historical = generateSampleData(100, 150);

  // TFT multi-horizon predictions with quantiles
  const predictions = {
    horizons: [1, 5, 10, 30],  // days
    q10: [152, 155, 158, 165],  // 10th percentile
    q50: [155, 160, 165, 175],  // median
    q90: [158, 165, 172, 185],  // 90th percentile
  };

  // Convert predictions to candlestick format
  const predictionBars = predictions.horizons.map((days, idx) => {
    const date = new Date();
    date.setDate(date.getDate() + days);

    return {
      time: date.toISOString().split('T')[0],
      open: predictions.q50[idx],
      high: predictions.q90[idx],
      low: predictions.q10[idx],
      close: predictions.q50[idx],
      volume: 0,
    };
  });

  // Combine historical + predictions
  const combinedData = [...historical, ...predictionBars];

  return (
    <div>
      <h1>ML Predictions with Uncertainty</h1>

      <CandlestickChart
        symbol="AAPL"
        data={combinedData}
        interval="1d"
        theme="dark"
        showVolume={true}
        height={700}
      />

      <div className="prediction-explanation">
        <p>📊 Prediction bars show uncertainty:</p>
        <ul>
          <li><strong>High</strong>: 90th percentile (optimistic)</li>
          <li><strong>Close/Open</strong>: Median prediction (most likely)</li>
          <li><strong>Low</strong>: 10th percentile (pessimistic)</li>
        </ul>
      </div>
    </div>
  );
}
```

**Benefits:**
- ✅ Visual uncertainty bands
- ✅ Multi-horizon predictions visible
- ✅ Professional presentation
- ✅ Confidence intervals clear

---

## 🔄 Migration Checklist

### For Each Page:

- [ ] **Identify current chart usage**
  - Find all Recharts imports
  - Note what data is being displayed
  - Check if it's line, bar, or pie chart

- [ ] **Replace with appropriate TradingView component**
  - Price data → `CandlestickChart`
  - Multiple timeframes → `MultiTimeframeChart`
  - Custom viz → `TradingViewChart` (base component)

- [ ] **Update data format**
  - Convert to OHLCV format
  - Use `toTime()` for time conversion
  - Validate with `validateOHLCVData()`

- [ ] **Add indicators if applicable**
  - Technical analysis pages → SMA, EMA, Bollinger
  - ML prediction pages → Custom prediction overlays
  - Risk pages → ATR, standard deviation bands

- [ ] **Test performance**
  - Load 1000+ data points
  - Check FPS (should be 60)
  - Verify zoom/pan smoothness

- [ ] **Update styling**
  - Use dark/light theme props
  - Match rest of page design
  - Ensure responsive behavior

---

## 📊 Data Format Conversion

### From API Response to OHLCV

```typescript
// Example: Converting minute bars from API
interface APIBar {
  timestamp: number;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
}

function convertAPIData(apiBars: APIBar[]): OHLCVData[] {
  return apiBars.map(bar => ({
    time: new Date(bar.timestamp * 1000).toISOString().split('T')[0],
    open: bar.o,
    high: bar.h,
    low: bar.l,
    close: bar.c,
    volume: bar.v,
  }));
}
```

### From ML Predictions to Chart Data

```typescript
// Example: TFT multi-horizon predictions
interface TFTForecast {
  timestamp: string;
  horizons: number[];
  predictions: number[];
  q10: number[];
  q50: number[];
  q90: number[];
}

function convertTFTPredictions(forecast: TFTForecast): OHLCVData[] {
  return forecast.horizons.map((days, idx) => {
    const date = new Date(forecast.timestamp);
    date.setDate(date.getDate() + days);

    return {
      time: date.toISOString().split('T')[0],
      open: forecast.q50[idx],
      high: forecast.q90[idx],
      low: forecast.q10[idx],
      close: forecast.q50[idx],
      volume: 0,
    };
  });
}
```

### From GNN Correlations to Network

```typescript
// For GNN correlation matrix visualization
// (Use D3.js separately, not TradingView charts)

interface CorrelationNode {
  symbol: string;
  features: number[];
}

interface CorrelationEdge {
  source: string;
  target: string;
  weight: number;  // correlation strength
}

// See D3.js force-directed graph for implementation
```

---

## 🎨 Theme Consistency

### Match Existing Dark Theme

```typescript
import { DARK_THEME } from '@/components/charts';

// Customize to match your app's theme
const customTheme = {
  ...DARK_THEME,
  layout: {
    background: { color: '#0f1419' },  // Your app's bg color
    textColor: '#e6e6e6',              // Your app's text color
  },
};
```

### Use in Component

```tsx
<CandlestickChart
  symbol="AAPL"
  data={data}
  theme="dark"  // or "light"
  // ... other props
/>
```

---

## 🧪 Testing Strategy

### 1. Visual Testing

```tsx
// Create a test page for each chart type
function ChartTestPage() {
  return (
    <div>
      <h2>1000 Bars Test</h2>
      <CandlestickChart
        symbol="TEST"
        data={generateSampleData(1000, 100)}
        interval="1d"
        theme="dark"
      />

      <h2>10K Bars Test</h2>
      <CandlestickChart
        symbol="TEST"
        data={generateSampleData(10000, 100)}
        interval="1m"
        theme="dark"
      />
    </div>
  );
}
```

### 2. Performance Testing

```typescript
// Measure render time
console.time('chart-render');

<CandlestickChart
  symbol="AAPL"
  data={largeDataset}
  interval="1d"
  theme="dark"
/>

console.timeEnd('chart-render');
// Should be < 100ms for 10K bars
```

### 3. Data Validation Testing

```typescript
import { validateOHLCVData } from '@/components/charts';

const { valid, errors } = validateOHLCVData(data);

if (!valid) {
  console.error('Invalid OHLCV data:', errors);
  // Display error to user
}
```

---

## 📱 Responsive Design

### Auto-resize on Container Change

```tsx
<div style={{ width: '100%', height: '500px' }}>
  <CandlestickChart
    symbol="AAPL"
    data={data}
    interval="1d"
    theme="dark"
    // No width/height needed - auto-fills parent
  />
</div>
```

### Multi-Monitor Support

```tsx
// Use MultiTimeframeChart for trading workstations
<MultiTimeframeChart
  symbol="AAPL"
  data={multiTimeframeData}
  layout="3x3"  // Full 9-pane grid
  theme="dark"
/>
```

---

## 🚀 Deployment Checklist

- [ ] All Recharts imports removed
- [ ] TradingView charts render correctly
- [ ] Data format validated
- [ ] Theme matches app design
- [ ] Performance tested (60 FPS)
- [ ] Responsive on mobile
- [ ] Indicators working
- [ ] Multi-timeframe layouts tested
- [ ] API integration complete
- [ ] Error handling added
- [ ] Loading states implemented
- [ ] Documentation updated

---

## 📞 Support

**Common Issues:**

1. **Chart not rendering**: Check data format with `validateOHLCVData()`
2. **Slow performance**: Reduce data points or use `aggregateData()`
3. **Time format errors**: Use `toTime()` helper function
4. **Indicators not showing**: Ensure data length >= indicator period

**Resources:**
- Demo page: `/charts-demo`
- README: `frontend/CHARTING_SYSTEM_README.md`
- TradingView docs: https://tradingview.github.io/lightweight-charts/

---

## 🎯 Next Steps

1. **Start with highest impact page** (EnsembleAnalysisPage recommended)
2. **Test thoroughly** before moving to next page
3. **Document any custom patterns** you develop
4. **Share improvements** with team

**Estimated Time per Page:**
- Simple replacement: 30-60 minutes
- Complex integration: 2-3 hours
- Custom visualization: 4-6 hours

---

**Good luck with the integration! 🚀**
*Building Bloomberg Terminal-level visualizations*
