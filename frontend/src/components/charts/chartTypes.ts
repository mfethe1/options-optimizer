/**
 * Type definitions for TradingView Lightweight Charts integration
 * Provides type safety for our charting components
 */

import {
  IChartApi,
  ISeriesApi,
  CandlestickData,
  LineData,
  HistogramData,
  Time,
  SeriesMarker,
  DeepPartial,
  ChartOptions,
  CandlestickSeriesPartialOptions,
  HistogramSeriesPartialOptions,
  LineSeriesPartialOptions
} from 'lightweight-charts';

// Re-export types from lightweight-charts for convenience
export type {
  IChartApi,
  ISeriesApi,
  CandlestickData,
  LineData,
  HistogramData,
  Time,
  SeriesMarker,
  DeepPartial,
  ChartOptions,
  CandlestickSeriesPartialOptions,
  HistogramSeriesPartialOptions,
  LineSeriesPartialOptions
};

/**
 * OHLCV data structure (standard financial data format)
 */
export interface OHLCVData {
  time: string | number | Date; // Timestamp
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

/**
 * Technical indicator data structure
 */
export interface IndicatorData {
  time: Time;
  value: number;
}

/**
 * Bollinger Bands data structure
 */
export interface BollingerBandsData {
  time: Time;
  upper: number;
  middle: number;
  lower: number;
}

/**
 * Chart theme configuration
 */
export interface ChartTheme {
  layout: {
    background: { color: string };
    textColor: string;
  };
  grid: {
    vertLines: { color: string };
    horzLines: { color: string };
  };
  crosshair: {
    mode: number;
    vertLine: {
      width: number;
      color: string;
      style: number;
      labelBackgroundColor: string;
    };
    horzLine: {
      width: number;
      color: string;
      style: number;
      labelBackgroundColor: string;
    };
  };
  watermark: {
    visible: boolean;
    fontSize: number;
    horzAlign: 'left' | 'center' | 'right';
    vertAlign: 'top' | 'center' | 'bottom';
    color: string;
    text: string;
  };
}

/**
 * Technical indicator configuration
 */
export interface IndicatorConfig {
  id?: string;
  type: string;
  period?: number;
  color?: string;
  lineWidth?: number;
  visible?: boolean;
}

/**
 * Prediction line series configuration with data
 */
export interface PredictionSeriesConfig {
  id: string;
  name: string;
  color: string;
  lineWidth?: number;
  lineStyle?: 'solid' | 'dashed' | 'dotted';
  data: LineData[];
  visible?: boolean;
}

/**
 * Chart configuration for our wrapper components
 */
export interface TradingViewChartConfig {
  symbol: string;
  interval: '1m' | '5m' | '15m' | '1h' | '4h' | '1d' | '1w' | '1M';
  theme?: 'dark' | 'light';
  height?: number;
  width?: number;
  showVolume?: boolean;
  showWatermark?: boolean;
  indicators?: IndicatorConfig[];
  autoFit?: boolean;
  priceScaleWidth?: number;
  timeScaleHeight?: number;
}

/**
 * Price scale modes
 */
export type PriceScaleMode = 'normal' | 'logarithmic' | 'percentage' | 'indexed100';

/**
 * Chart event types
 */
export interface ChartEvents {
  onCrosshairMove?: (data: any) => void;
  onVisibleRangeChange?: (range: { from: Time; to: Time }) => void;
  onClick?: (data: any) => void;
  onDoubleClick?: (data: any) => void;
}

/**
 * Marker types for important events
 */
export interface ChartMarker {
  time: Time;
  position: 'aboveBar' | 'belowBar' | 'inBar';
  color: string;
  shape: 'circle' | 'square' | 'arrowUp' | 'arrowDown';
  text?: string;
  size?: number;
}

/**
 * Price line configuration
 */
export interface PriceLineConfig {
  price: number;
  color: string;
  lineWidth?: number;
  lineStyle?: number;
  axisLabelVisible?: boolean;
  title?: string;
}

/**
 * Time range selector options
 */
export type TimeRange = '1D' | '5D' | '1M' | '3M' | '6M' | '1Y' | 'YTD' | 'ALL';

/**
 * Data granularity options
 */
export interface DataGranularity {
  interval: string;
  displayName: string;
  seconds: number;
}

/**
 * Chart state for persistence
 */
export interface ChartState {
  symbol: string;
  interval: string;
  timeRange: TimeRange;
  indicators: IndicatorConfig[];
  priceLines: PriceLineConfig[];
  markers: ChartMarker[];
  theme: 'dark' | 'light';
  visibleRange?: { from: Time; to: Time };
}

/**
 * Real-time update data
 */
export interface RealtimeUpdate {
  time: Time;
  open?: number;
  high?: number;
  low?: number;
  close: number;
  volume?: number;
}

/**
 * Chart series types
 */
export type SeriesType = 'candlestick' | 'bar' | 'line' | 'area' | 'histogram';

/**
 * Multi-pane configuration
 */
export interface ChartPane {
  id: string;
  height: number;
  seriesType: SeriesType;
  data: any[];
  title?: string;
}
