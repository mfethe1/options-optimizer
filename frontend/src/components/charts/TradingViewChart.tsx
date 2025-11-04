/**
 * TradingView Lightweight Charts - Base React Wrapper Component
 *
 * Professional-grade wrapper for TradingView Lightweight Charts library
 * Provides React lifecycle management, theme support, and real-time updates
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  createChart,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  LineData,
  HistogramData,
  Time,
} from 'lightweight-charts';
import {
  TradingViewChartConfig,
  OHLCVData,
  ChartTheme,
  LineSeriesConfig,
  IndicatorConfig,
} from './chartTypes';
import {
  DARK_THEME,
  LIGHT_THEME,
  CANDLESTICK_COLORS,
  convertToCandlestickData,
  convertToVolumeData,
  debounce,
} from './chartUtils';

interface TradingViewChartProps {
  config: TradingViewChartConfig;
  data: OHLCVData[];
  indicators?: IndicatorConfig[];
  onCrosshairMove?: (data: any) => void;
  onVisibleRangeChange?: (range: any) => void;
  className?: string;
}

/**
 * TradingViewChart - Base wrapper component
 *
 * Features:
 * - Automatic theme application (dark/light)
 * - Candlestick price series
 * - Optional volume histogram pane
 * - Real-time data updates
 * - Responsive resize handling
 * - Proper cleanup on unmount
 *
 * Usage:
 * ```tsx
 * <TradingViewChart
 *   config={{
 *     symbol: 'AAPL',
 *     interval: '1d',
 *     theme: 'dark',
 *     showVolume: true
 *   }}
 *   data={ohlcvData}
 *   indicators={[
 *     { type: 'sma', period: 20, color: '#2196f3' }
 *   ]}
 * />
 * ```
 */
const TradingViewChart: React.FC<TradingViewChartProps> = ({
  config,
  data,
  indicators = [],
  onCrosshairMove,
  onVisibleRangeChange,
  className = '',
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const indicatorSeriesRefs = useRef<Map<string, ISeriesApi<'Line'>>>(new Map());

  const [isInitialized, setIsInitialized] = useState(false);

  /**
   * Get theme configuration based on config
   */
  const getTheme = useCallback((): ChartTheme => {
    return config.theme === 'light' ? LIGHT_THEME : DARK_THEME;
  }, [config.theme]);

  /**
   * Initialize chart instance
   */
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const container = chartContainerRef.current;
    const theme = getTheme();

    // Create chart with theme
    const chart = createChart(container, {
      width: config.width || container.clientWidth,
      height: config.height || (config.showVolume ? 500 : 400),
      layout: theme.layout,
      grid: theme.grid,
      crosshair: theme.crosshair,
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        borderColor: theme.grid.vertLines.color,
      },
      rightPriceScale: {
        borderColor: theme.grid.vertLines.color,
        scaleMargins: {
          top: 0.1,
          bottom: config.showVolume ? 0.3 : 0.1,
        },
      },
      watermark: {
        ...theme.watermark,
        text: config.symbol,
      },
    });

    chartRef.current = chart;

    // Create candlestick series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: CANDLESTICK_COLORS.upColor,
      downColor: CANDLESTICK_COLORS.downColor,
      borderUpColor: CANDLESTICK_COLORS.borderUpColor,
      borderDownColor: CANDLESTICK_COLORS.borderDownColor,
      wickUpColor: CANDLESTICK_COLORS.wickUpColor,
      wickDownColor: CANDLESTICK_COLORS.wickDownColor,
    });

    candlestickSeriesRef.current = candlestickSeries;

    // Create volume series if enabled
    if (config.showVolume) {
      const volumeSeries = chart.addHistogramSeries({
        priceFormat: {
          type: 'volume',
        },
        priceScaleId: '', // Use separate scale
        scaleMargins: {
          top: 0.7,
          bottom: 0,
        },
      });

      volumeSeriesRef.current = volumeSeries;
    }

    // Set up event handlers
    if (onCrosshairMove) {
      chart.subscribeCrosshairMove((param) => {
        onCrosshairMove(param);
      });
    }

    if (onVisibleRangeChange) {
      chart.timeScale().subscribeVisibleTimeRangeChange((range) => {
        onVisibleRangeChange(range);
      });
    }

    setIsInitialized(true);

    // Cleanup on unmount
    return () => {
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
      candlestickSeriesRef.current = null;
      volumeSeriesRef.current = null;
      indicatorSeriesRefs.current.clear();
    };
  }, [config.symbol, config.theme, config.showVolume, config.width, config.height]);

  /**
   * Update chart data when data prop changes
   */
  useEffect(() => {
    if (!isInitialized || !candlestickSeriesRef.current || data.length === 0) return;

    try {
      // Update candlestick data
      const candlestickData = convertToCandlestickData(data);
      candlestickSeriesRef.current.setData(candlestickData);

      // Update volume data if enabled
      if (config.showVolume && volumeSeriesRef.current) {
        const volumeData = convertToVolumeData(data);
        volumeSeriesRef.current.setData(volumeData);
      }

      // Fit content to visible range
      if (chartRef.current) {
        chartRef.current.timeScale().fitContent();
      }
    } catch (error) {
      console.error('Error updating chart data:', error);
    }
  }, [data, isInitialized, config.showVolume]);

  /**
   * Handle window resize
   */
  useEffect(() => {
    if (!chartRef.current || !chartContainerRef.current) return;

    const handleResize = debounce(() => {
      if (chartRef.current && chartContainerRef.current) {
        const { clientWidth, clientHeight } = chartContainerRef.current;
        chartRef.current.applyOptions({
          width: clientWidth,
          height: config.height || clientHeight,
        });
      }
    }, 150);

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [config.height]);

  /**
   * Update indicators
   */
  useEffect(() => {
    if (!isInitialized || !chartRef.current || indicators.length === 0) return;

    // Remove old indicators
    indicatorSeriesRefs.current.forEach((series) => {
      if (chartRef.current) {
        chartRef.current.removeSeries(series);
      }
    });
    indicatorSeriesRefs.current.clear();

    // Add new indicators (placeholder - full implementation in separate component)
    indicators.forEach((indicator) => {
      if (!chartRef.current) return;

      const lineSeries = chartRef.current.addLineSeries({
        color: indicator.color || '#2196f3',
        lineWidth: indicator.lineWidth || 2,
        title: `${indicator.type.toUpperCase()}(${indicator.period || ''})`,
      });

      indicatorSeriesRefs.current.set(indicator.id || indicator.type, lineSeries);

      // Note: Actual indicator calculation will be done in a separate helper
      // For now, this is just the series setup
    });
  }, [indicators, isInitialized]);

  return (
    <div
      ref={chartContainerRef}
      className={`trading-view-chart ${className}`}
      style={{
        position: 'relative',
        width: config.width || '100%',
        height: config.height || (config.showVolume ? 500 : 400),
      }}
    />
  );
};

export default TradingViewChart;

/**
 * Hook for accessing chart instance externally
 * Allows parent components to control chart programmatically
 */
export const useChartRef = () => {
  const chartRef = useRef<IChartApi | null>(null);

  const setChartRef = useCallback((chart: IChartApi | null) => {
    chartRef.current = chart;
  }, []);

  return { chartRef, setChartRef };
};
