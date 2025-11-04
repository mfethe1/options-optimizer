/**
 * CandlestickChart - Specialized component for trading candlestick charts
 *
 * High-level component with built-in features:
 * - Automatic data fetching (optional)
 * - Price/volume display
 * - Technical indicators
 * - Time range selectors
 * - Real-time updates
 * - Crosshair tooltips
 */

import React, { useState, useCallback, useMemo } from 'react';
import TradingViewChart from './TradingViewChart';
import { OHLCVData, TradingViewChartConfig, IndicatorConfig } from './chartTypes';
import { formatPrice, formatVolume, formatPercentage, calculatePercentageChange } from './chartUtils';

interface CandlestickChartProps {
  symbol: string;
  data: OHLCVData[];
  interval?: '1m' | '5m' | '15m' | '1h' | '4h' | '1d' | '1w' | '1M';
  theme?: 'dark' | 'light';
  height?: number;
  showVolume?: boolean;
  showControls?: boolean;
  indicators?: IndicatorConfig[];
  onIntervalChange?: (interval: string) => void;
  className?: string;
}

interface CrosshairInfo {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
  change: number;
  changePercent: number;
}

/**
 * CandlestickChart - Professional candlestick chart with full features
 *
 * Usage:
 * ```tsx
 * <CandlestickChart
 *   symbol="AAPL"
 *   data={historicalData}
 *   interval="1d"
 *   theme="dark"
 *   showVolume={true}
 *   showControls={true}
 *   indicators={[
 *     { type: 'sma', period: 20 },
 *     { type: 'sma', period: 50 }
 *   ]}
 * />
 * ```
 */
const CandlestickChart: React.FC<CandlestickChartProps> = ({
  symbol,
  data,
  interval = '1d',
  theme = 'dark',
  height = 500,
  showVolume = true,
  showControls = true,
  indicators = [],
  onIntervalChange,
  className = '',
}) => {
  const [selectedInterval, setSelectedInterval] = useState(interval);
  const [crosshairInfo, setCrosshairInfo] = useState<CrosshairInfo | null>(null);
  const [activeIndicators, setActiveIndicators] = useState<IndicatorConfig[]>(indicators);

  /**
   * Calculate current price stats
   */
  const priceStats = useMemo(() => {
    if (data.length === 0) return null;

    const latest = data[data.length - 1];
    const previous = data.length > 1 ? data[data.length - 2] : latest;

    return {
      current: latest.close,
      change: latest.close - previous.close,
      changePercent: calculatePercentageChange(latest.close, previous.close),
      high24h: Math.max(...data.slice(-24).map((d) => d.high)),
      low24h: Math.min(...data.slice(-24).map((d) => d.low)),
      volume: latest.volume || 0,
    };
  }, [data]);

  /**
   * Handle crosshair move to show tooltip
   */
  const handleCrosshairMove = useCallback((param: any) => {
    if (!param || !param.time || !param.seriesData || param.seriesData.size === 0) {
      setCrosshairInfo(null);
      return;
    }

    // Get candlestick data at crosshair
    const candleData = Array.from(param.seriesData.values())[0];

    if (candleData && 'open' in candleData) {
      setCrosshairInfo({
        time: new Date(param.time * 1000).toLocaleString(),
        open: candleData.open,
        high: candleData.high,
        low: candleData.low,
        close: candleData.close,
        volume: candleData.volume,
        change: candleData.close - candleData.open,
        changePercent: calculatePercentageChange(candleData.close, candleData.open),
      });
    }
  }, []);

  /**
   * Handle interval change
   */
  const handleIntervalChange = useCallback(
    (newInterval: string) => {
      setSelectedInterval(newInterval as any);
      if (onIntervalChange) {
        onIntervalChange(newInterval);
      }
    },
    [onIntervalChange]
  );

  /**
   * Toggle indicator
   */
  const toggleIndicator = useCallback((indicator: IndicatorConfig) => {
    setActiveIndicators((prev) => {
      const exists = prev.find((i) => i.type === indicator.type && i.period === indicator.period);
      if (exists) {
        return prev.filter((i) => !(i.type === indicator.type && i.period === indicator.period));
      } else {
        return [...prev, indicator];
      }
    });
  }, []);

  /**
   * Chart configuration
   */
  const chartConfig: TradingViewChartConfig = {
    symbol,
    interval: selectedInterval,
    theme,
    height,
    showVolume,
  };

  return (
    <div className={`candlestick-chart-container ${className}`}>
      {/* Header with symbol and current price */}
      {showControls && priceStats && (
        <div
          className="chart-header"
          style={{
            padding: '12px 16px',
            backgroundColor: theme === 'dark' ? '#1e222d' : '#f0f3fa',
            color: theme === 'dark' ? '#d1d4dc' : '#191919',
            borderRadius: '4px 4px 0 0',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '12px',
          }}
        >
          {/* Symbol and Price */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <div>
              <div style={{ fontSize: '20px', fontWeight: 600 }}>{symbol}</div>
              <div style={{ fontSize: '14px', opacity: 0.7 }}>{selectedInterval.toUpperCase()}</div>
            </div>
            <div>
              <div style={{ fontSize: '24px', fontWeight: 500 }}>{formatPrice(priceStats.current)}</div>
              <div
                style={{
                  fontSize: '14px',
                  color: priceStats.change >= 0 ? '#26a69a' : '#ef5350',
                }}
              >
                {formatPercentage(priceStats.changePercent)} ({formatPrice(priceStats.change)})
              </div>
            </div>
          </div>

          {/* Stats */}
          <div style={{ display: 'flex', gap: '20px', fontSize: '13px' }}>
            <div>
              <div style={{ opacity: 0.7 }}>High 24h</div>
              <div>{formatPrice(priceStats.high24h)}</div>
            </div>
            <div>
              <div style={{ opacity: 0.7 }}>Low 24h</div>
              <div>{formatPrice(priceStats.low24h)}</div>
            </div>
            <div>
              <div style={{ opacity: 0.7 }}>Volume</div>
              <div>{formatVolume(priceStats.volume)}</div>
            </div>
          </div>
        </div>
      )}

      {/* Interval Selector */}
      {showControls && (
        <div
          className="interval-selector"
          style={{
            padding: '8px 16px',
            backgroundColor: theme === 'dark' ? '#131722' : '#ffffff',
            borderBottom: `1px solid ${theme === 'dark' ? '#2a2e39' : '#e1e3eb'}`,
            display: 'flex',
            gap: '8px',
          }}
        >
          {['1m', '5m', '15m', '1h', '4h', '1d', '1w', '1M'].map((int) => (
            <button
              key={int}
              onClick={() => handleIntervalChange(int)}
              style={{
                padding: '6px 12px',
                backgroundColor: selectedInterval === int ? '#2962ff' : 'transparent',
                color: selectedInterval === int ? '#ffffff' : theme === 'dark' ? '#d1d4dc' : '#191919',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '13px',
                fontWeight: 500,
                transition: 'all 0.2s',
              }}
            >
              {int.toUpperCase()}
            </button>
          ))}
        </div>
      )}

      {/* Indicator Selector */}
      {showControls && (
        <div
          className="indicator-selector"
          style={{
            padding: '8px 16px',
            backgroundColor: theme === 'dark' ? '#131722' : '#ffffff',
            borderBottom: `1px solid ${theme === 'dark' ? '#2a2e39' : '#e1e3eb'}`,
            display: 'flex',
            gap: '8px',
            flexWrap: 'wrap',
          }}
        >
          <span style={{ fontSize: '13px', opacity: 0.7, marginRight: '8px' }}>Indicators:</span>
          {[
            { type: 'sma', period: 20, label: 'SMA 20' },
            { type: 'sma', period: 50, label: 'SMA 50' },
            { type: 'ema', period: 12, label: 'EMA 12' },
            { type: 'bollinger', period: 20, label: 'BB 20' },
          ].map((ind) => (
            <button
              key={`${ind.type}-${ind.period}`}
              onClick={() => toggleIndicator({ type: ind.type as any, period: ind.period })}
              style={{
                padding: '4px 10px',
                backgroundColor: activeIndicators.some(
                  (i) => i.type === ind.type && i.period === ind.period
                )
                  ? '#4caf50'
                  : 'transparent',
                color:
                  activeIndicators.some((i) => i.type === ind.type && i.period === ind.period)
                    ? '#ffffff'
                    : theme === 'dark'
                    ? '#d1d4dc'
                    : '#191919',
                border: `1px solid ${theme === 'dark' ? '#2a2e39' : '#e1e3eb'}`,
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '12px',
                transition: 'all 0.2s',
              }}
            >
              {ind.label}
            </button>
          ))}
        </div>
      )}

      {/* Crosshair Tooltip */}
      {crosshairInfo && (
        <div
          className="crosshair-tooltip"
          style={{
            position: 'absolute',
            top: showControls ? '150px' : '10px',
            left: '16px',
            padding: '12px',
            backgroundColor: theme === 'dark' ? 'rgba(30, 34, 45, 0.95)' : 'rgba(255, 255, 255, 0.95)',
            color: theme === 'dark' ? '#d1d4dc' : '#191919',
            borderRadius: '6px',
            fontSize: '13px',
            zIndex: 10,
            boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
            pointerEvents: 'none',
          }}
        >
          <div style={{ marginBottom: '8px', fontWeight: 600 }}>{crosshairInfo.time}</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '4px 12px' }}>
            <span style={{ opacity: 0.7 }}>O:</span>
            <span>{formatPrice(crosshairInfo.open)}</span>
            <span style={{ opacity: 0.7 }}>H:</span>
            <span>{formatPrice(crosshairInfo.high)}</span>
            <span style={{ opacity: 0.7 }}>L:</span>
            <span>{formatPrice(crosshairInfo.low)}</span>
            <span style={{ opacity: 0.7 }}>C:</span>
            <span>{formatPrice(crosshairInfo.close)}</span>
            {crosshairInfo.volume && (
              <>
                <span style={{ opacity: 0.7 }}>V:</span>
                <span>{formatVolume(crosshairInfo.volume)}</span>
              </>
            )}
            <span style={{ opacity: 0.7 }}>Δ:</span>
            <span
              style={{
                color: crosshairInfo.change >= 0 ? '#26a69a' : '#ef5350',
              }}
            >
              {formatPercentage(crosshairInfo.changePercent)} ({formatPrice(crosshairInfo.change)})
            </span>
          </div>
        </div>
      )}

      {/* Chart */}
      <TradingViewChart
        config={chartConfig}
        data={data}
        indicators={activeIndicators}
        onCrosshairMove={handleCrosshairMove}
        className="main-chart"
      />
    </div>
  );
};

export default CandlestickChart;
