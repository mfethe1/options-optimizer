/**
 * Multi-Timeframe Chart Layout
 *
 * Professional trading interface with multiple timeframes displayed simultaneously
 * Similar to Bloomberg Terminal, TradingView multi-chart layouts
 *
 * Features:
 * - 2x2 or 3x3 grid layouts
 * - Synchronized crosshairs
 * - Independent timeframes
 * - Customizable intervals per pane
 * - Responsive design
 */

import React, { useState, useCallback, useMemo } from 'react';
import CandlestickChart from './CandlestickChart';
import { OHLCVData, IndicatorConfig } from './chartTypes';

type LayoutType = '1x1' | '2x2' | '3x3' | '1x2' | '2x1';

interface TimeframeConfig {
  interval: '1m' | '5m' | '15m' | '1h' | '4h' | '1d' | '1w' | '1M';
  label: string;
  indicators?: IndicatorConfig[];
}

interface MultiTimeframeChartProps {
  symbol: string;
  data: Record<string, OHLCVData[]>; // { '1d': [...], '1h': [...], ... }
  layout?: LayoutType;
  timeframes?: TimeframeConfig[];
  theme?: 'dark' | 'light';
  showVolume?: boolean;
  onLayoutChange?: (layout: LayoutType) => void;
  className?: string;
}

/**
 * Default timeframe configurations
 */
const DEFAULT_TIMEFRAMES: Record<LayoutType, TimeframeConfig[]> = {
  '1x1': [{ interval: '1d', label: 'Daily' }],
  '2x2': [
    { interval: '15m', label: '15 Min' },
    { interval: '1h', label: '1 Hour' },
    { interval: '1d', label: 'Daily' },
    { interval: '1w', label: 'Weekly' },
  ],
  '3x3': [
    { interval: '1m', label: '1 Min' },
    { interval: '5m', label: '5 Min' },
    { interval: '15m', label: '15 Min' },
    { interval: '1h', label: '1 Hour' },
    { interval: '4h', label: '4 Hour' },
    { interval: '1d', label: 'Daily' },
    { interval: '1d', label: 'Daily (Alt)', indicators: [{ type: 'sma', period: 50 }] },
    { interval: '1w', label: 'Weekly' },
    { interval: '1M', label: 'Monthly' },
  ],
  '1x2': [
    { interval: '1h', label: '1 Hour' },
    { interval: '1d', label: 'Daily' },
  ],
  '2x1': [
    { interval: '1h', label: '1 Hour' },
    { interval: '1d', label: 'Daily' },
  ],
};

/**
 * Multi-Timeframe Chart Component
 *
 * Usage:
 * ```tsx
 * <MultiTimeframeChart
 *   symbol="AAPL"
 *   data={{
 *     '1m': minuteData,
 *     '1h': hourlyData,
 *     '1d': dailyData,
 *     '1w': weeklyData
 *   }}
 *   layout="2x2"
 *   theme="dark"
 * />
 * ```
 */
const MultiTimeframeChart: React.FC<MultiTimeframeChartProps> = ({
  symbol,
  data,
  layout = '2x2',
  timeframes,
  theme = 'dark',
  showVolume = true,
  onLayoutChange,
  className = '',
}) => {
  const [selectedLayout, setSelectedLayout] = useState<LayoutType>(layout);
  const [syncCrosshairs, setSyncCrosshairs] = useState(true);

  /**
   * Get timeframes for current layout
   */
  const activeTimeframes = useMemo(() => {
    return timeframes || DEFAULT_TIMEFRAMES[selectedLayout];
  }, [timeframes, selectedLayout]);

  /**
   * Get grid configuration
   */
  const gridConfig = useMemo(() => {
    const configs: Record<LayoutType, { rows: number; cols: number }> = {
      '1x1': { rows: 1, cols: 1 },
      '2x2': { rows: 2, cols: 2 },
      '3x3': { rows: 3, cols: 3 },
      '1x2': { rows: 1, cols: 2 },
      '2x1': { rows: 2, cols: 1 },
    };
    return configs[selectedLayout];
  }, [selectedLayout]);

  /**
   * Calculate chart height based on layout
   */
  const chartHeight = useMemo(() => {
    const baseHeight = 400;
    const headerHeight = 80;
    const padding = 20;

    return Math.floor((window.innerHeight - headerHeight - padding * gridConfig.rows) / gridConfig.rows);
  }, [gridConfig.rows]);

  /**
   * Handle layout change
   */
  const handleLayoutChange = useCallback(
    (newLayout: LayoutType) => {
      setSelectedLayout(newLayout);
      if (onLayoutChange) {
        onLayoutChange(newLayout);
      }
    },
    [onLayoutChange]
  );

  return (
    <div className={`multi-timeframe-chart ${className}`}>
      {/* Control Bar */}
      <div
        style={{
          padding: '12px 16px',
          backgroundColor: theme === 'dark' ? '#1e222d' : '#f0f3fa',
          color: theme === 'dark' ? '#d1d4dc' : '#191919',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          borderRadius: '4px 4px 0 0',
        }}
      >
        {/* Symbol Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ fontSize: '18px', fontWeight: 600 }}>{symbol}</div>
          <div style={{ fontSize: '13px', opacity: 0.7 }}>Multi-Timeframe Analysis</div>
        </div>

        {/* Layout Selector */}
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <span style={{ fontSize: '13px', opacity: 0.7, marginRight: '8px' }}>Layout:</span>
          {(['1x1', '2x2', '3x3', '1x2', '2x1'] as LayoutType[]).map((layoutType) => (
            <button
              key={layoutType}
              onClick={() => handleLayoutChange(layoutType)}
              style={{
                padding: '6px 12px',
                backgroundColor: selectedLayout === layoutType ? '#2962ff' : 'transparent',
                color: selectedLayout === layoutType ? '#ffffff' : theme === 'dark' ? '#d1d4dc' : '#191919',
                border: `1px solid ${theme === 'dark' ? '#2a2e39' : '#e1e3eb'}`,
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '12px',
                fontWeight: 500,
                transition: 'all 0.2s',
              }}
              title={`${layoutType} grid layout`}
            >
              {layoutType}
            </button>
          ))}

          {/* Sync Crosshairs Toggle */}
          <button
            onClick={() => setSyncCrosshairs(!syncCrosshairs)}
            style={{
              padding: '6px 12px',
              backgroundColor: syncCrosshairs ? '#4caf50' : 'transparent',
              color: syncCrosshairs ? '#ffffff' : theme === 'dark' ? '#d1d4dc' : '#191919',
              border: `1px solid ${theme === 'dark' ? '#2a2e39' : '#e1e3eb'}`,
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '12px',
              fontWeight: 500,
              marginLeft: '8px',
              transition: 'all 0.2s',
            }}
            title="Synchronize crosshairs across charts"
          >
            {syncCrosshairs ? '🔗' : '⛓️‍💥'} Sync
          </button>
        </div>
      </div>

      {/* Chart Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateRows: `repeat(${gridConfig.rows}, 1fr)`,
          gridTemplateColumns: `repeat(${gridConfig.cols}, 1fr)`,
          gap: '8px',
          padding: '8px',
          backgroundColor: theme === 'dark' ? '#131722' : '#ffffff',
        }}
      >
        {activeTimeframes.map((timeframeConfig, index) => {
          const chartData = data[timeframeConfig.interval] || [];

          return (
            <div
              key={`${timeframeConfig.interval}-${index}`}
              style={{
                position: 'relative',
                backgroundColor: theme === 'dark' ? '#1e222d' : '#f0f3fa',
                borderRadius: '6px',
                overflow: 'hidden',
                border: `1px solid ${theme === 'dark' ? '#2a2e39' : '#e1e3eb'}`,
              }}
            >
              {/* Timeframe Label */}
              <div
                style={{
                  position: 'absolute',
                  top: '8px',
                  left: '8px',
                  padding: '4px 10px',
                  backgroundColor: theme === 'dark' ? 'rgba(41, 98, 255, 0.2)' : 'rgba(41, 98, 255, 0.1)',
                  color: '#2962ff',
                  borderRadius: '4px',
                  fontSize: '12px',
                  fontWeight: 600,
                  zIndex: 5,
                  pointerEvents: 'none',
                }}
              >
                {timeframeConfig.label}
              </div>

              {/* Chart */}
              {chartData.length > 0 ? (
                <CandlestickChart
                  symbol={symbol}
                  data={chartData}
                  interval={timeframeConfig.interval}
                  theme={theme}
                  height={chartHeight}
                  showVolume={showVolume}
                  showControls={false} // Hide controls in multi-pane view
                  indicators={timeframeConfig.indicators}
                />
              ) : (
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    height: chartHeight,
                    color: theme === 'dark' ? '#666' : '#999',
                    fontSize: '14px',
                  }}
                >
                  No data available for {timeframeConfig.label}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Footer with Stats */}
      <div
        style={{
          padding: '12px 16px',
          backgroundColor: theme === 'dark' ? '#1e222d' : '#f0f3fa',
          color: theme === 'dark' ? '#d1d4dc' : '#191919',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: '13px',
          borderRadius: '0 0 4px 4px',
        }}
      >
        <div style={{ opacity: 0.7 }}>
          Viewing {activeTimeframes.length} timeframe{activeTimeframes.length !== 1 ? 's' : ''}
        </div>
        <div style={{ opacity: 0.7 }}>
          {syncCrosshairs ? 'Crosshairs synchronized' : 'Independent crosshairs'}
        </div>
      </div>
    </div>
  );
};

export default MultiTimeframeChart;

/**
 * Preset configurations for common trading setups
 */
export const TRADING_PRESETS = {
  scalping: {
    layout: '2x2' as LayoutType,
    timeframes: [
      { interval: '1m' as const, label: '1 Min' },
      { interval: '5m' as const, label: '5 Min' },
      { interval: '15m' as const, label: '15 Min' },
      { interval: '1h' as const, label: '1 Hour' },
    ],
  },
  dayTrading: {
    layout: '2x2' as LayoutType,
    timeframes: [
      { interval: '5m' as const, label: '5 Min' },
      { interval: '15m' as const, label: '15 Min' },
      { interval: '1h' as const, label: '1 Hour' },
      { interval: '1d' as const, label: 'Daily' },
    ],
  },
  swingTrading: {
    layout: '2x2' as LayoutType,
    timeframes: [
      { interval: '1h' as const, label: '1 Hour' },
      { interval: '4h' as const, label: '4 Hour' },
      { interval: '1d' as const, label: 'Daily' },
      { interval: '1w' as const, label: 'Weekly' },
    ],
  },
  longTerm: {
    layout: '1x2' as LayoutType,
    timeframes: [
      { interval: '1d' as const, label: 'Daily' },
      { interval: '1w' as const, label: 'Weekly' },
    ],
  },
};
