# Phase 4 Signals Panel - Detailed Specification

**Component**: `Phase4SignalsPanel.tsx`  
**Purpose**: Real-time visualization of short-horizon (1-5 day) technical & cross-asset signals  
**Complexity**: High (4 distinct visualizations + real-time updates)

---

## 1. Component Structure

```typescript
import React, { useEffect } from 'react';
import { usePhase4Stream } from '../hooks/usePhase4Stream';
import { Phase4Tech } from '../types/investor-report';
import { OptionsFlowGauge } from './OptionsFlowGauge';
import { ResidualMomentumChart } from './ResidualMomentumChart';
import { SeasonalityCalendar } from './SeasonalityCalendar';
import { BreadthLiquidityHeatmap } from './BreadthLiquidityHeatmap';

interface Props {
  userId: string;
  initialData?: Phase4Tech;
}

export const Phase4SignalsPanel: React.FC<Props> = ({ userId, initialData }) => {
  const { phase4Data, isConnected } = usePhase4Stream(userId);
  const data = phase4Data || initialData;
  
  return (
    <div className="bg-[#2a2a2a] border border-[#404040] rounded-lg p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-white">Phase 4 Signals</h2>
          <p className="text-sm text-gray-400">Short-horizon edge (1-5 day alpha)</p>
        </div>
        
        {/* Real-time indicator */}
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`} />
          <span className="text-xs text-gray-400">
            {isConnected ? 'Live' : 'Disconnected'}
          </span>
        </div>
      </div>
      
      {/* 2x2 Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <OptionsFlowGauge value={data?.options_flow_composite} />
        <ResidualMomentumChart value={data?.residual_momentum} />
        <SeasonalityCalendar value={data?.seasonality_score} />
        <BreadthLiquidityHeatmap value={data?.breadth_liquidity} />
      </div>
      
      {/* Explanations */}
      {data?.explanations && data.explanations.length > 0 && (
        <div className="mt-6 p-4 bg-blue-900/20 border border-blue-500/50 rounded-lg">
          <h3 className="text-sm font-semibold text-blue-400 mb-2">Interpretation</h3>
          <ul className="space-y-1">
            {data.explanations.map((exp, i) => (
              <li key={i} className="text-sm text-gray-300">• {exp}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
```

---

## 2. Sub-Component: OptionsFlowGauge

**Purpose**: Visualize options flow composite (-1 to +1) as a gauge/speedometer

```typescript
import React from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface Props {
  value: number | null;
}

export const OptionsFlowGauge: React.FC<Props> = ({ value }) => {
  if (value === null) {
    return <LoadingState label="Options Flow" />;
  }
  
  // Map -1 to +1 → 0 to 180 degrees
  const angle = ((value + 1) / 2) * 180;
  
  // Color based on value
  const getColor = () => {
    if (value > 0.5) return '#10b981';  // Strong bullish
    if (value > 0) return '#84cc16';    // Mild bullish
    if (value > -0.5) return '#f59e0b'; // Mild bearish
    return '#ef4444';                   // Strong bearish
  };
  
  const color = getColor();
  const sentiment = value > 0 ? 'Bullish' : 'Bearish';
  const icon = value > 0 ? <TrendingUp /> : <TrendingDown />;
  
  return (
    <div className="p-6 bg-[#1a1a1a] border border-[#404040] rounded-lg">
      <h3 className="text-lg font-semibold text-white mb-4">Options Flow Composite</h3>
      
      {/* Gauge SVG */}
      <div className="relative w-full h-48 flex items-center justify-center">
        <svg viewBox="0 0 200 120" className="w-full h-full">
          {/* Background arc */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="#404040"
            strokeWidth="12"
            strokeLinecap="round"
          />
          
          {/* Value arc */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke={color}
            strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray={`${angle * 2.5} 1000`}
            className="transition-all duration-500"
          />
          
          {/* Needle */}
          <line
            x1="100"
            y1="100"
            x2={100 + 70 * Math.cos((angle - 90) * Math.PI / 180)}
            y2={100 + 70 * Math.sin((angle - 90) * Math.PI / 180)}
            stroke={color}
            strokeWidth="3"
            strokeLinecap="round"
            className="transition-all duration-500"
          />
          
          {/* Center dot */}
          <circle cx="100" cy="100" r="5" fill={color} />
        </svg>
        
        {/* Value display */}
        <div className="absolute bottom-0 text-center">
          <div className="text-3xl font-bold" style={{ color }}>
            {value.toFixed(2)}
          </div>
          <div className="text-sm text-gray-400 flex items-center gap-1">
            {icon}
            {sentiment}
          </div>
        </div>
      </div>
      
      {/* Breakdown */}
      <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
        <div className="text-center">
          <div className="text-gray-400">PCR</div>
          <div className="text-white font-semibold">0.72</div>
        </div>
        <div className="text-center">
          <div className="text-gray-400">IV Skew</div>
          <div className="text-white font-semibold">-3.2%</div>
        </div>
        <div className="text-center">
          <div className="text-gray-400">Volume</div>
          <div className="text-white font-semibold">1.8x</div>
        </div>
      </div>
      
      {/* Tooltip */}
      <div className="mt-3 p-2 bg-blue-900/20 border border-blue-500/30 rounded text-xs text-gray-300">
        💡 Combines Put/Call Ratio, IV skew, and volume. Bullish when calls dominate with high volume.
      </div>
    </div>
  );
};
```

---

## 3. Sub-Component: ResidualMomentumChart

**Purpose**: Show idiosyncratic momentum (asset vs market/sector) as a bar chart

```typescript
import React from 'react';
import { BarChart, Bar, XAxis, YAxis, ReferenceLine, ResponsiveContainer } from 'recharts';

interface Props {
  value: number | null;
}

export const ResidualMomentumChart: React.FC<Props> = ({ value }) => {
  if (value === null) {
    return <LoadingState label="Residual Momentum" />;
  }
  
  // Z-score interpretation
  const getInterpretation = () => {
    if (value > 2) return { text: 'Strong Outperformance', color: '#10b981' };
    if (value > 1) return { text: 'Mild Outperformance', color: '#84cc16' };
    if (value > -1) return { text: 'Neutral', color: '#6b7280' };
    if (value > -2) return { text: 'Mild Underperformance', color: '#f59e0b' };
    return { text: 'Strong Underperformance', color: '#ef4444' };
  };
  
  const { text, color } = getInterpretation();
  
  // Historical data (mock - replace with real data)
  const historicalData = [
    { day: '-5d', value: -0.5 },
    { day: '-4d', value: 0.2 },
    { day: '-3d', value: 0.8 },
    { day: '-2d', value: 1.2 },
    { day: '-1d', value: 1.5 },
    { day: 'Today', value: value }
  ];
  
  return (
    <div className="p-6 bg-[#1a1a1a] border border-[#404040] rounded-lg">
      <h3 className="text-lg font-semibold text-white mb-4">Residual Momentum</h3>
      
      {/* Current Value */}
      <div className="mb-4">
        <div className="text-3xl font-bold" style={{ color }}>
          {value.toFixed(2)}σ
        </div>
        <div className="text-sm text-gray-400">{text}</div>
      </div>
      
      {/* Chart */}
      <ResponsiveContainer width="100%" height={150}>
        <BarChart data={historicalData}>
          <XAxis dataKey="day" stroke="#808080" style={{ fontSize: '10px' }} />
          <YAxis stroke="#808080" style={{ fontSize: '10px' }} />
          <ReferenceLine y={0} stroke="#404040" strokeDasharray="3 3" />
          <Bar 
            dataKey="value" 
            fill={color}
            radius={[4, 4, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
      
      {/* Tooltip */}
      <div className="mt-3 p-2 bg-blue-900/20 border border-blue-500/30 rounded text-xs text-gray-300">
        💡 Idiosyncratic returns after removing market/sector effects. >2σ = strong alpha.
      </div>
    </div>
  );
};
```

---

## 4. Sub-Component: SeasonalityCalendar

**Purpose**: Visualize calendar effects (turn-of-month, day-of-week)

```typescript
import React from 'react';
import { Calendar } from 'lucide-react';

interface Props {
  value: number | null;
}

export const SeasonalityCalendar: React.FC<Props> = ({ value }) => {
  if (value === null) {
    return <LoadingState label="Seasonality" />;
  }
  
  // Day-of-week pattern (mock data)
  const dayOfWeekPattern = [
    { day: 'Mon', avg: 0.15, current: value > 0 && new Date().getDay() === 1 },
    { day: 'Tue', avg: -0.05, current: value > 0 && new Date().getDay() === 2 },
    { day: 'Wed', avg: 0.08, current: value > 0 && new Date().getDay() === 3 },
    { day: 'Thu', avg: 0.12, current: value > 0 && new Date().getDay() === 4 },
    { day: 'Fri', avg: -0.10, current: value > 0 && new Date().getDay() === 5 }
  ];
  
  // Turn-of-month indicator
  const today = new Date().getDate();
  const isTurnOfMonth = today >= 28 || today <= 2;
  
  return (
    <div className="p-6 bg-[#1a1a1a] border border-[#404040] rounded-lg">
      <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
        <Calendar className="w-5 h-5" />
        Seasonality Score
      </h3>
      
      {/* Current Value */}
      <div className="mb-4">
        <div className="text-3xl font-bold text-white">
          {value.toFixed(2)}
        </div>
        <div className="text-sm text-gray-400">
          {value > 0.5 ? 'Strong Positive' : value > 0 ? 'Mild Positive' : value > -0.5 ? 'Mild Negative' : 'Strong Negative'}
        </div>
      </div>
      
      {/* Day-of-Week Heatmap */}
      <div className="mb-4">
        <div className="text-xs text-gray-400 mb-2">Day-of-Week Pattern</div>
        <div className="flex gap-1">
          {dayOfWeekPattern.map(({ day, avg, current }) => (
            <div 
              key={day}
              className={`
                flex-1 p-2 rounded text-center transition-all
                ${current ? 'ring-2 ring-white' : ''}
                ${avg > 0.1 ? 'bg-green-900/40' : avg > 0 ? 'bg-green-900/20' : avg > -0.1 ? 'bg-red-900/20' : 'bg-red-900/40'}
              `}
            >
              <div className="text-xs text-gray-300">{day}</div>
              <div className={`text-xs font-semibold ${avg > 0 ? 'text-green-400' : 'text-red-400'}`}>
                {avg > 0 ? '+' : ''}{(avg * 100).toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      </div>
      
      {/* Turn-of-Month Indicator */}
      <div className={`
        p-3 rounded-lg border
        ${isTurnOfMonth ? 'bg-green-900/20 border-green-500/50' : 'bg-gray-800/50 border-gray-700'}
      `}>
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-300">Turn-of-Month Effect</span>
          <span className={`text-sm font-semibold ${isTurnOfMonth ? 'text-green-400' : 'text-gray-500'}`}>
            {isTurnOfMonth ? 'Active' : 'Inactive'}
          </span>
        </div>
      </div>
      
      {/* Tooltip */}
      <div className="mt-3 p-2 bg-blue-900/20 border border-blue-500/30 rounded text-xs text-gray-300">
        💡 Calendar patterns: turn-of-month (days 28-2) and day-of-week effects.
      </div>
    </div>
  );
};
```

---

## 5. Sub-Component: BreadthLiquidityHeatmap

**Purpose**: Market internals (advance/decline, volume, spreads) as a heatmap

```typescript
import React from 'react';
import { Activity } from 'lucide-react';

interface Props {
  value: number | null;
}

export const BreadthLiquidityHeatmap: React.FC<Props> = ({ value }) => {
  if (value === null) {
    return <LoadingState label="Breadth & Liquidity" />;
  }
  
  // Component breakdown (mock data)
  const components = [
    { label: 'Advance/Decline', value: 0.65, weight: 0.5 },
    { label: 'Volume Ratio', value: 0.45, weight: 0.3 },
    { label: 'Spread Tightness', value: 0.80, weight: 0.2 }
  ];
  
  const getColor = (val: number) => {
    if (val > 0.6) return '#10b981';
    if (val > 0.4) return '#84cc16';
    if (val > 0.2) return '#f59e0b';
    return '#ef4444';
  };
  
  return (
    <div className="p-6 bg-[#1a1a1a] border border-[#404040] rounded-lg">
      <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
        <Activity className="w-5 h-5" />
        Breadth & Liquidity
      </h3>
      
      {/* Current Value */}
      <div className="mb-4">
        <div className="text-3xl font-bold" style={{ color: getColor(value) }}>
          {(value * 100).toFixed(0)}%
        </div>
        <div className="text-sm text-gray-400">
          {value > 0.6 ? 'Strong' : value > 0.4 ? 'Moderate' : 'Weak'}
        </div>
      </div>
      
      {/* Component Breakdown */}
      <div className="space-y-3">
        {components.map(({ label, value: compValue, weight }) => (
          <div key={label}>
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-gray-400">{label}</span>
              <span className="text-xs text-white font-semibold">
                {(compValue * 100).toFixed(0)}%
              </span>
            </div>
            
            {/* Progress bar */}
            <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
              <div 
                className="h-full transition-all duration-500"
                style={{ 
                  width: `${compValue * 100}%`,
                  backgroundColor: getColor(compValue)
                }}
              />
            </div>
            
            <div className="text-xs text-gray-500 mt-1">
              Weight: {(weight * 100).toFixed(0)}%
            </div>
          </div>
        ))}
      </div>
      
      {/* Tooltip */}
      <div className="mt-3 p-2 bg-blue-900/20 border border-blue-500/30 rounded text-xs text-gray-300">
        💡 Market internals: A/D ratio (50%), volume (30%), spreads (20%).
      </div>
    </div>
  );
};
```

---

## 6. Loading State Component

```typescript
const LoadingState: React.FC<{ label: string }> = ({ label }) => (
  <div className="p-6 bg-[#1a1a1a] border border-[#404040] rounded-lg">
    <h3 className="text-lg font-semibold text-white mb-4">{label}</h3>
    <div className="flex flex-col items-center justify-center h-48">
      <Loader2 className="w-8 h-8 animate-spin text-gray-500 mb-2" />
      <span className="text-sm text-gray-500">Computing...</span>
    </div>
  </div>
);
```

---

## 7. Real-Time Update Behavior

**WebSocket Message Handling**:

```typescript
// In usePhase4Stream hook
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  if (message.type === 'phase4_update') {
    // Smooth transition (no jarring updates)
    setPhase4Data(prev => ({
      ...prev,
      ...message.data,
      // Interpolate values for smooth animation
      options_flow_composite: interpolate(prev?.options_flow_composite, message.data.options_flow_composite),
      residual_momentum: interpolate(prev?.residual_momentum, message.data.residual_momentum),
      // ... other fields
    }));
  }
};

// Interpolation helper (smooth transitions)
const interpolate = (oldVal: number | null, newVal: number | null, steps = 10) => {
  if (oldVal === null || newVal === null) return newVal;
  
  // Animate over 500ms
  const delta = (newVal - oldVal) / steps;
  let current = oldVal;
  
  const interval = setInterval(() => {
    current += delta;
    setPhase4Data(prev => ({ ...prev, options_flow_composite: current }));
  }, 50);
  
  setTimeout(() => clearInterval(interval), 500);
  
  return newVal;
};
```

---

## 8. Performance Considerations

- **Memoization**: Use `React.memo()` for sub-components
- **Debouncing**: Debounce WebSocket updates (max 1 update per second)
- **Lazy Loading**: Load charts only when visible (Intersection Observer)
- **Canvas Rendering**: Use `<canvas>` for complex visualizations (60fps)

---

## 9. Testing

**Unit Tests**:
```typescript
describe('Phase4SignalsPanel', () => {
  it('renders all 4 sub-components', () => {
    render(<Phase4SignalsPanel userId="test" initialData={mockData} />);
    expect(screen.getByText('Options Flow Composite')).toBeInTheDocument();
    expect(screen.getByText('Residual Momentum')).toBeInTheDocument();
    expect(screen.getByText('Seasonality Score')).toBeInTheDocument();
    expect(screen.getByText('Breadth & Liquidity')).toBeInTheDocument();
  });
  
  it('shows loading state when data is null', () => {
    render(<Phase4SignalsPanel userId="test" initialData={{ options_flow_composite: null, ... }} />);
    expect(screen.getAllByText('Computing...')).toHaveLength(4);
  });
  
  it('updates in real-time via WebSocket', async () => {
    const { rerender } = render(<Phase4SignalsPanel userId="test" />);
    
    // Simulate WebSocket message
    act(() => {
      mockWebSocket.send({ type: 'phase4_update', data: { options_flow_composite: 0.75 } });
    });
    
    await waitFor(() => {
      expect(screen.getByText('0.75')).toBeInTheDocument();
    });
  });
});
```

---

**Status**: 📋 Ready for implementation  
**Complexity**: High (4 custom visualizations + real-time)  
**Estimated Effort**: 24-32 hours  
**Dependencies**: WebSocket infrastructure, Phase 4 backend metrics

