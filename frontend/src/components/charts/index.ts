/**
 * TradingView Lightweight Charts Integration
 *
 * Professional charting system for Options Optimizer
 * Built on TradingView Lightweight Charts library
 *
 * Components:
 * - TradingViewChart: Base wrapper component
 * - CandlestickChart: Full-featured candlestick chart with controls
 * - MultiTimeframeChart: Multi-pane layout for simultaneous timeframes
 *
 * Utilities:
 * - chartUtils: Theme presets, formatters, data converters
 * - indicators: Technical indicator calculations (SMA, EMA, RSI, MACD, etc.)
 * - chartTypes: TypeScript type definitions
 *
 * Usage:
 * ```tsx
 * import { CandlestickChart } from '@/components/charts';
 *
 * <CandlestickChart
 *   symbol="AAPL"
 *   data={ohlcvData}
 *   interval="1d"
 *   theme="dark"
 *   showVolume={true}
 * />
 * ```
 */

// Components
export { default as TradingViewChart } from './TradingViewChart';
export { default as CandlestickChart } from './CandlestickChart';
export { default as MultiTimeframeChart, TRADING_PRESETS } from './MultiTimeframeChart';

// Types
export type {
  OHLCVData,
  TradingViewChartConfig,
  ChartTheme,
  LineSeriesConfig,
  IndicatorConfig,
  ChartMarker,
} from './chartTypes';

// Utilities
export {
  // Themes
  DARK_THEME,
  LIGHT_THEME,
  CANDLESTICK_COLORS,
  CANDLESTICK_COLORS_ALT,
  INDICATOR_COLORS,
  // Data conversion
  convertToCandlestickData,
  convertToVolumeData,
  // Formatters
  formatPrice,
  formatVolume,
  formatPercentage,
  calculatePercentageChange,
  // Time utilities
  toTime,
  getTimeRangeSeconds,
  // Data manipulation
  aggregateData,
  filterDataByRange,
  generateSampleData,
  // Validation
  validateOHLCVData,
  // Helpers
  debounce,
  mergeChartOptions,
} from './chartUtils';

// Technical Indicators
export {
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
  AVAILABLE_INDICATORS,
} from './indicators';

export type {
  BollingerBands,
  MACDData,
  StochasticData,
  IndicatorType,
} from './indicators';
