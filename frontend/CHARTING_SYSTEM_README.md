# 📊 TradingView Lightweight Charts Integration

**Professional charting system for Options Optimizer**
Built on TradingView Lightweight Charts • Bloomberg Terminal quality • 100K+ data points at 60 FPS

---

## 🎯 Overview

This charting system provides institutional-grade financial visualization capabilities competitive with Bloomberg Terminal and TradingView. It's designed specifically for multi-timeframe stock analysis, options trading, and ML model visualization.

### Key Features

- ⚡ **High Performance**: 100,000+ data points at 60 FPS (WebGL accelerated)
- 💾 **Tiny Bundle**: Only 45KB gzipped (vs 200KB-1.2MB alternatives)
- 📈 **Finance Native**: Candlesticks, volume, time-scale, all built-in
- 🎨 **Professional Themes**: Bloomberg Terminal-style dark/light themes
- 🔄 **Real-time Ready**: WebSocket streaming support
- 📊 **Multi-timeframe**: 2x2, 3x3 grid layouts
- 📉 **Full Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, and more
- 🆓 **FREE**: Apache 2.0 license (no $36K-$120K/year costs)

---

## 📦 Installation

The package is already installed:

```bash
npm install lightweight-charts@5.0.9
```

---

## 🚀 Quick Start

### Basic Candlestick Chart

```tsx
import { CandlestickChart } from '@/components/charts';

function MyPage() {
  const data = [
    { time: '2024-01-01', open: 100, high: 105, low: 98, close: 103, volume: 1000000 },
    { time: '2024-01-02', open: 103, high: 107, low: 102, close: 106, volume: 1200000 },
    // ... more data
  ];

  return (
    <CandlestickChart
      symbol="AAPL"
      data={data}
      interval="1d"
      theme="dark"
      showVolume={true}
      showControls={true}
    />
  );
}
```

### Multi-Timeframe Grid

```tsx
import { MultiTimeframeChart, TRADING_PRESETS } from '@/components/charts';

function MultiChartPage() {
  const data = {
    '15m': fifteenMinuteData,
    '1h': hourlyData,
    '1d': dailyData,
    '1w': weeklyData,
  };

  return (
    <MultiTimeframeChart
      symbol="AAPL"
      data={data}
      layout="2x2"
      theme="dark"
      showVolume={true}
    />
  );
}
```

### With Technical Indicators

```tsx
import { CandlestickChart } from '@/components/charts';

function ChartWithIndicators() {
  const indicators = [
    { type: 'sma', period: 20, color: '#2196f3' },
    { type: 'sma', period: 50, color: '#ff9800' },
    { type: 'ema', period: 12, color: '#4caf50' },
  ];

  return (
    <CandlestickChart
      symbol="AAPL"
      data={historicalData}
      interval="1d"
      theme="dark"
      indicators={indicators}
    />
  );
}
```

---

## 📖 Component API

### `<CandlestickChart>`

Full-featured candlestick chart with price stats, volume, controls, and tooltips.

**Props:**

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `symbol` | `string` | required | Stock symbol (e.g., "AAPL") |
| `data` | `OHLCVData[]` | required | OHLCV data array |
| `interval` | `'1m' \| '5m' \| '15m' \| '1h' \| '4h' \| '1d' \| '1w' \| '1M'` | `'1d'` | Time interval |
| `theme` | `'dark' \| 'light'` | `'dark'` | Color theme |
| `height` | `number` | `500` | Chart height in pixels |
| `showVolume` | `boolean` | `true` | Show volume histogram |
| `showControls` | `boolean` | `true` | Show interval selector and stats |
| `indicators` | `IndicatorConfig[]` | `[]` | Technical indicators to display |
| `onIntervalChange` | `(interval: string) => void` | - | Callback when interval changes |
| `className` | `string` | `''` | Additional CSS classes |

**Example:**

```tsx
<CandlestickChart
  symbol="TSLA"
  data={dailyData}
  interval="1d"
  theme="dark"
  height={600}
  showVolume={true}
  showControls={true}
  indicators={[
    { type: 'sma', period: 20 },
    { type: 'bollinger', period: 20 }
  ]}
  onIntervalChange={(interval) => fetchData(interval)}
/>
```

---

### `<MultiTimeframeChart>`

Multi-pane chart layout for simultaneous timeframe viewing.

**Props:**

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `symbol` | `string` | required | Stock symbol |
| `data` | `Record<string, OHLCVData[]>` | required | Data keyed by interval |
| `layout` | `'1x1' \| '2x2' \| '3x3' \| '1x2' \| '2x1'` | `'2x2'` | Grid layout |
| `timeframes` | `TimeframeConfig[]` | - | Custom timeframe configs |
| `theme` | `'dark' \| 'light'` | `'dark'` | Color theme |
| `showVolume` | `boolean` | `true` | Show volume in each pane |
| `onLayoutChange` | `(layout: LayoutType) => void` | - | Callback when layout changes |
| `className` | `string` | `''` | Additional CSS classes |

**Example:**

```tsx
<MultiTimeframeChart
  symbol="NVDA"
  data={{
    '15m': fifteenMinData,
    '1h': hourlyData,
    '1d': dailyData,
    '1w': weeklyData
  }}
  layout="2x2"
  theme="dark"
  showVolume={true}
/>
```

**Trading Presets:**

```tsx
import { TRADING_PRESETS } from '@/components/charts';

// Pre-configured layouts for different trading styles
const presets = {
  scalping: {     // 1m, 5m, 15m, 1h
    layout: '2x2',
    timeframes: [...]
  },
  dayTrading: {   // 5m, 15m, 1h, 1d
    layout: '2x2',
    timeframes: [...]
  },
  swingTrading: { // 1h, 4h, 1d, 1w
    layout: '2x2',
    timeframes: [...]
  },
  longTerm: {     // 1d, 1w
    layout: '1x2',
    timeframes: [...]
  }
};
```

---

### `<TradingViewChart>`

Base wrapper component (use for custom implementations).

**Props:**

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `config` | `TradingViewChartConfig` | required | Chart configuration object |
| `data` | `OHLCVData[]` | required | OHLCV data array |
| `indicators` | `IndicatorConfig[]` | `[]` | Technical indicators |
| `onCrosshairMove` | `(data: any) => void` | - | Crosshair move callback |
| `onVisibleRangeChange` | `(range: any) => void` | - | Visible range change callback |
| `className` | `string` | `''` | Additional CSS classes |

---

## 📊 Data Format

### OHLCVData Interface

```typescript
interface OHLCVData {
  time: string | number | Date;  // ISO string, timestamp, or Date
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;  // Optional
}
```

**Example:**

```typescript
const data: OHLCVData[] = [
  {
    time: '2024-01-01',
    open: 100.50,
    high: 105.25,
    low: 98.75,
    close: 103.80,
    volume: 1234567
  },
  // ... more data
];
```

---

## 📈 Technical Indicators

### Available Indicators

```typescript
import {
  calculateSMA,
  calculateEMA,
  calculateBollingerBands,
  calculateRSI,
  calculateMACD,
  calculateATR,
  calculateStochastic,
  calculateVWAP,
  calculateOBV,
  calculateParabolicSAR,
} from '@/components/charts';
```

### Usage Examples

**Simple Moving Average (SMA):**

```typescript
const sma20 = calculateSMA(data, 20);
// Returns: [{ time, value }, ...]
```

**Exponential Moving Average (EMA):**

```typescript
const ema12 = calculateEMA(data, 12);
```

**Bollinger Bands:**

```typescript
const { upper, middle, lower } = calculateBollingerBands(data, 20, 2);
// Returns 3 line series
```

**Relative Strength Index (RSI):**

```typescript
const rsi = calculateRSI(data, 14);
// Returns values 0-100
```

**MACD:**

```typescript
const { macd, signal, histogram } = calculateMACD(data, 12, 26, 9);
```

**Average True Range (ATR):**

```typescript
const atr = calculateATR(data, 14);
```

---

## 🎨 Themes and Styling

### Built-in Themes

```typescript
import { DARK_THEME, LIGHT_THEME } from '@/components/charts';

// Bloomberg Terminal-style dark theme
const darkConfig = {
  ...DARK_THEME,
  layout: {
    background: { color: '#131722' },
    textColor: '#d1d4dc'
  },
  grid: {
    vertLines: { color: '#1e222d' },
    horzLines: { color: '#1e222d' }
  }
};
```

### Candlestick Colors

```typescript
import { CANDLESTICK_COLORS, CANDLESTICK_COLORS_ALT } from '@/components/charts';

// Default: Green/Red
CANDLESTICK_COLORS = {
  upColor: '#26a69a',
  downColor: '#ef5350',
  // ... borders and wicks
};

// Alternative: Blue/Orange
CANDLESTICK_COLORS_ALT = {
  upColor: '#089981',
  downColor: '#f23645',
  // ... borders and wicks
};
```

### Indicator Colors

```typescript
import { INDICATOR_COLORS } from '@/components/charts';

const colors = {
  SMA_20: '#2196f3',
  SMA_50: '#ff9800',
  SMA_200: '#f44336',
  EMA_12: '#4caf50',
  EMA_26: '#9c27b0',
  BOLLINGER_UPPER: '#3f51b5',
  RSI: '#9c27b0',
  MACD: '#2196f3',
  // ... more indicators
};
```

---

## 🔧 Utility Functions

### Data Conversion

```typescript
import {
  convertToCandlestickData,
  convertToVolumeData,
  toTime,
} from '@/components/charts';

// Convert OHLCV to candlestick format
const candleData = convertToCandlestickData(ohlcvData);

// Convert OHLCV to volume histogram
const volumeData = convertToVolumeData(ohlcvData);

// Convert various time formats to TradingView Time
const time = toTime('2024-01-01');  // or timestamp or Date
```

### Formatters

```typescript
import {
  formatPrice,
  formatVolume,
  formatPercentage,
  calculatePercentageChange,
} from '@/components/charts';

formatPrice(1234.56);        // "1234.56"
formatVolume(1234567);       // "1.23M"
formatPercentage(5.67);      // "+5.67%"
calculatePercentageChange(105, 100);  // 5.0
```

### Data Manipulation

```typescript
import {
  aggregateData,
  filterDataByRange,
  generateSampleData,
  validateOHLCVData,
} from '@/components/charts';

// Aggregate to different timeframes
const hourlyData = aggregateData(minuteData, 'hour');

// Filter by time range
const last30Days = filterDataByRange(data, '1M');

// Generate sample data for testing
const sampleData = generateSampleData(100, 150);  // 100 bars, start at $150

// Validate data integrity
const { valid, errors } = validateOHLCVData(data);
```

---

## 🔄 Real-time Updates

### WebSocket Integration

```typescript
import { CandlestickChart } from '@/components/charts';

function RealTimeChart() {
  const [data, setData] = useState<OHLCVData[]>([]);

  useEffect(() => {
    const ws = new WebSocket('wss://your-market-data-api.com');

    ws.onmessage = (event) => {
      const newBar = JSON.parse(event.data);

      setData(prev => {
        const updated = [...prev];
        const lastBar = updated[updated.length - 1];

        // Update last bar or append new one
        if (lastBar.time === newBar.time) {
          updated[updated.length - 1] = newBar;
        } else {
          updated.push(newBar);
        }

        return updated;
      });
    };

    return () => ws.close();
  }, []);

  return (
    <CandlestickChart
      symbol="AAPL"
      data={data}
      interval="1m"
      theme="dark"
    />
  );
}
```

---

## 🧪 Testing

### Demo Page

Visit `/charts-demo` to see all features in action:

```
http://localhost:3000/charts-demo
```

Features demonstrated:
- Single candlestick chart with full controls
- Multi-timeframe 2x2 grid layout
- All technical indicators
- Dark/light theme switching
- Multiple symbols
- Performance stats

### Generate Sample Data

```typescript
import { generateSampleData } from '@/components/charts';

// Generate 365 days of data starting at $150
const data = generateSampleData(365, 150);
```

---

## 📱 Responsive Design

Charts automatically resize on window resize with debouncing:

```typescript
// Automatic resize handling built-in
<CandlestickChart
  symbol="AAPL"
  data={data}
  height={500}  // or omit for auto-height
/>
```

For custom resize handling:

```typescript
import { debounce } from '@/components/charts';

const handleResize = debounce(() => {
  // Your custom logic
}, 150);
```

---

## 🎯 Integration with ML Models

### TFT (Temporal Fusion Transformer) Predictions

```typescript
import { CandlestickChart, calculateEMA } from '@/components/charts';

function TFTPredictionChart({ historical, predictions }) {
  // Combine historical + predictions
  const combinedData = [
    ...historical,
    ...predictions.map(p => ({
      time: p.timestamp,
      open: p.q50,
      high: p.q90,
      low: p.q10,
      close: p.q50,
      volume: 0
    }))
  ];

  return (
    <CandlestickChart
      symbol="AAPL"
      data={combinedData}
      interval="1d"
      theme="dark"
      indicators={[
        { type: 'ema', period: 12, label: 'EMA 12' }
      ]}
    />
  );
}
```

### Prediction Cones (Uncertainty Visualization)

```typescript
// Add quantile bands as indicators
const predictionBands = [
  { type: 'line', data: q10Data, color: 'rgba(41,98,255,0.3)' },
  { type: 'line', data: q50Data, color: '#2962ff' },
  { type: 'line', data: q90Data, color: 'rgba(41,98,255,0.3)' },
];
```

---

## 📚 Additional Resources

### Files Structure

```
frontend/src/components/charts/
├── index.ts                    # Main exports
├── chartTypes.ts              # TypeScript definitions
├── chartUtils.ts              # Utilities and helpers
├── indicators.ts              # Technical indicator calculations
├── TradingViewChart.tsx       # Base wrapper component
├── CandlestickChart.tsx       # Full-featured chart
└── MultiTimeframeChart.tsx    # Multi-pane layout

frontend/src/pages/
└── ChartsDemo.tsx             # Comprehensive demo page

frontend/
└── CHARTING_SYSTEM_README.md  # This file
```

### Documentation Links

- TradingView Lightweight Charts: https://tradingview.github.io/lightweight-charts/
- GitHub: https://github.com/tradingview/lightweight-charts
- Examples: https://tradingview.github.io/lightweight-charts/docs/examples

---

## ⚡ Performance Tips

1. **Limit Data Points**: For best performance, display 1,000-10,000 candles at a time
2. **Use Time-based Filtering**: Use `filterDataByRange()` to reduce data volume
3. **Aggregate Data**: Use `aggregateData()` for higher timeframes
4. **Debounce Updates**: Use `debounce()` for frequent data updates
5. **Lazy Load**: Load historical data on-demand as user scrolls

---

## 🆘 Troubleshooting

### Chart Not Rendering

```typescript
// Ensure data is not empty
if (data.length === 0) {
  return <div>No data available</div>;
}

// Validate data format
import { validateOHLCVData } from '@/components/charts';
const { valid, errors } = validateOHLCVData(data);
```

### Time Format Issues

```typescript
// TradingView requires Unix timestamps (seconds)
import { toTime } from '@/components/charts';

const data = rawData.map(item => ({
  ...item,
  time: toTime(item.timestamp)  // Handles string, number, Date
}));
```

### Indicators Not Showing

```typescript
// Ensure data length >= indicator period
const sma50 = calculateSMA(data, 50);  // Needs at least 50 bars

// Check indicator data format
console.log(sma50[0]);  // Should be: { time: ..., value: ... }
```

---

## 🎉 Next Steps

1. **Explore Demo**: Visit `/charts-demo` to see all features
2. **Integrate APIs**: Connect to your market data endpoints
3. **Add Custom Indicators**: Create your own indicator calculations
4. **Customize Themes**: Modify color schemes to match your brand
5. **Add ML Visualizations**: Overlay model predictions on charts

---

## 📞 Support

For questions or issues:
- Check the demo page: `/charts-demo`
- Review TradingView docs: https://tradingview.github.io/lightweight-charts/
- Submit issues on GitHub

---

**Built with ❤️ for Options Optimizer**
*Professional charting at zero cost*
