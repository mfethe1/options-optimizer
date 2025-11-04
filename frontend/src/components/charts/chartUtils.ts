/**
 * Utility functions for TradingView Lightweight Charts
 * Includes theme presets, formatters, and helper functions
 */

import { Time, UTCTimestamp } from 'lightweight-charts';
import { ChartTheme, OHLCVData } from './chartTypes';

/**
 * Dark theme preset (Bloomberg Terminal-like)
 */
export const DARK_THEME: ChartTheme = {
  layout: {
    background: { color: '#131722' },
    textColor: '#d1d4dc',
  },
  grid: {
    vertLines: { color: '#1e222d' },
    horzLines: { color: '#1e222d' },
  },
  crosshair: {
    mode: 1,
    vertLine: {
      width: 1,
      color: '#758696',
      style: 3,
      labelBackgroundColor: '#363c4e',
    },
    horzLine: {
      width: 1,
      color: '#758696',
      style: 3,
      labelBackgroundColor: '#363c4e',
    },
  },
  watermark: {
    visible: true,
    fontSize: 48,
    horzAlign: 'center',
    vertAlign: 'center',
    color: 'rgba(255, 255, 255, 0.05)',
    text: '',
  },
};

/**
 * Light theme preset (TradingView-like)
 */
export const LIGHT_THEME: ChartTheme = {
  layout: {
    background: { color: '#ffffff' },
    textColor: '#191919',
  },
  grid: {
    vertLines: { color: '#e1e3eb' },
    horzLines: { color: '#e1e3eb' },
  },
  crosshair: {
    mode: 1,
    vertLine: {
      width: 1,
      color: '#9598a1',
      style: 3,
      labelBackgroundColor: '#f0f3fa',
    },
    horzLine: {
      width: 1,
      color: '#9598a1',
      style: 3,
      labelBackgroundColor: '#f0f3fa',
    },
  },
  watermark: {
    visible: true,
    fontSize: 48,
    horzAlign: 'center',
    vertAlign: 'center',
    color: 'rgba(0, 0, 0, 0.05)',
    text: '',
  },
};

/**
 * Candlestick color scheme (green/red)
 */
export const CANDLESTICK_COLORS = {
  upColor: '#26a69a',
  downColor: '#ef5350',
  borderUpColor: '#26a69a',
  borderDownColor: '#ef5350',
  wickUpColor: '#26a69a',
  wickDownColor: '#ef5350',
};

/**
 * Candlestick color scheme (blue/orange - alternative)
 */
export const CANDLESTICK_COLORS_ALT = {
  upColor: '#089981',
  downColor: '#f23645',
  borderUpColor: '#089981',
  borderDownColor: '#f23645',
  wickUpColor: '#089981',
  wickDownColor: '#f23645',
};

/**
 * Technical indicator color palette
 */
export const INDICATOR_COLORS = {
  SMA_20: '#2196f3',
  SMA_50: '#ff9800',
  SMA_200: '#f44336',
  EMA_12: '#4caf50',
  EMA_26: '#9c27b0',
  BOLLINGER_UPPER: '#3f51b5',
  BOLLINGER_MIDDLE: '#2196f3',
  BOLLINGER_LOWER: '#3f51b5',
  RSI: '#9c27b0',
  MACD: '#2196f3',
  MACD_SIGNAL: '#ff9800',
  MACD_HISTOGRAM: '#4caf50',
  VOLUME: '#26a69a',
};

/**
 * Convert various time formats to TradingView Time type
 */
export function toTime(timestamp: string | number | Date): Time {
  if (typeof timestamp === 'string') {
    return Date.parse(timestamp) / 1000 as UTCTimestamp;
  } else if (timestamp instanceof Date) {
    return Math.floor(timestamp.getTime() / 1000) as UTCTimestamp;
  } else {
    // Assume it's already a Unix timestamp
    return (timestamp > 10000000000 ? Math.floor(timestamp / 1000) : timestamp) as UTCTimestamp;
  }
}

/**
 * Convert OHLCV data to TradingView candlestick format
 */
export function convertToCandlestickData(data: OHLCVData[]) {
  return data.map((item) => ({
    time: toTime(item.time),
    open: item.open,
    high: item.high,
    low: item.low,
    close: item.close,
  }));
}

/**
 * Convert OHLCV data to volume histogram format
 */
export function convertToVolumeData(data: OHLCVData[]) {
  return data
    .filter((item) => item.volume !== undefined && item.volume !== null)
    .map((item) => ({
      time: toTime(item.time),
      value: item.volume!,
      color: item.close >= item.open ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)',
    }));
}

/**
 * Format price with appropriate decimal places
 */
export function formatPrice(price: number): string {
  if (price >= 1000) {
    return price.toFixed(2);
  } else if (price >= 1) {
    return price.toFixed(2);
  } else if (price >= 0.01) {
    return price.toFixed(4);
  } else {
    return price.toFixed(6);
  }
}

/**
 * Format volume with K/M/B suffixes
 */
export function formatVolume(volume: number): string {
  if (volume >= 1e9) {
    return (volume / 1e9).toFixed(2) + 'B';
  } else if (volume >= 1e6) {
    return (volume / 1e6).toFixed(2) + 'M';
  } else if (volume >= 1e3) {
    return (volume / 1e3).toFixed(2) + 'K';
  } else {
    return volume.toString();
  }
}

/**
 * Format percentage change
 */
export function formatPercentage(value: number, decimals: number = 2): string {
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(decimals)}%`;
}

/**
 * Calculate percentage change
 */
export function calculatePercentageChange(current: number, previous: number): number {
  if (previous === 0) return 0;
  return ((current - previous) / previous) * 100;
}

/**
 * Get time range in seconds
 */
export function getTimeRangeSeconds(range: string): number {
  const ranges: Record<string, number> = {
    '1D': 86400,
    '5D': 432000,
    '1M': 2592000,
    '3M': 7776000,
    '6M': 15552000,
    '1Y': 31536000,
    'YTD': Math.floor((Date.now() - new Date(new Date().getFullYear(), 0, 1).getTime()) / 1000),
    'ALL': Infinity,
  };
  return ranges[range] || 86400;
}

/**
 * Generate sample OHLCV data for testing
 */
export function generateSampleData(count: number, startPrice: number = 100): OHLCVData[] {
  const data: OHLCVData[] = [];
  let currentPrice = startPrice;
  const startTime = Date.now() - count * 86400000; // count days ago

  for (let i = 0; i < count; i++) {
    const open = currentPrice;
    const close = currentPrice * (1 + (Math.random() - 0.5) * 0.04); // ±2% change
    const high = Math.max(open, close) * (1 + Math.random() * 0.02); // +0-2%
    const low = Math.min(open, close) * (1 - Math.random() * 0.02); // -0-2%
    const volume = Math.floor(1000000 + Math.random() * 5000000);

    data.push({
      time: new Date(startTime + i * 86400000).toISOString().split('T')[0],
      open,
      high,
      low,
      close,
      volume,
    });

    currentPrice = close;
  }

  return data;
}

/**
 * Detect and format time range based on data density
 */
export function detectTimeFormat(data: OHLCVData[]): string {
  if (data.length < 2) return 'day';

  const firstTime = new Date(data[0].time).getTime();
  const secondTime = new Date(data[1].time).getTime();
  const diff = Math.abs(secondTime - firstTime);

  if (diff < 3600000) return 'minute'; // < 1 hour
  if (diff < 86400000) return 'hour'; // < 1 day
  if (diff < 604800000) return 'day'; // < 1 week
  if (diff < 2592000000) return 'week'; // < 30 days
  return 'month';
}

/**
 * Group data by time period (for aggregation)
 */
export function aggregateData(
  data: OHLCVData[],
  period: 'hour' | 'day' | 'week' | 'month'
): OHLCVData[] {
  if (data.length === 0) return [];

  const groups = new Map<string, OHLCVData[]>();

  data.forEach((item) => {
    const date = new Date(item.time);
    let key: string;

    switch (period) {
      case 'hour':
        key = `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}-${date.getHours()}`;
        break;
      case 'day':
        key = `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}`;
        break;
      case 'week':
        const weekNum = Math.floor(date.getTime() / (7 * 86400000));
        key = `${weekNum}`;
        break;
      case 'month':
        key = `${date.getFullYear()}-${date.getMonth()}`;
        break;
    }

    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key)!.push(item);
  });

  const aggregated: OHLCVData[] = [];

  groups.forEach((groupData, key) => {
    const open = groupData[0].open;
    const close = groupData[groupData.length - 1].close;
    const high = Math.max(...groupData.map((d) => d.high));
    const low = Math.min(...groupData.map((d) => d.low));
    const volume = groupData.reduce((sum, d) => sum + (d.volume || 0), 0);
    const time = groupData[0].time;

    aggregated.push({ time, open, high, low, close, volume });
  });

  return aggregated.sort((a, b) => new Date(a.time).getTime() - new Date(b.time).getTime());
}

/**
 * Filter data by time range
 */
export function filterDataByRange(data: OHLCVData[], range: string): OHLCVData[] {
  const now = Date.now();
  const rangeSeconds = getTimeRangeSeconds(range);

  if (rangeSeconds === Infinity) return data;

  const cutoff = now - rangeSeconds * 1000;

  return data.filter((item) => new Date(item.time).getTime() >= cutoff);
}

/**
 * Calculate chart dimensions based on container and options
 */
export function calculateChartDimensions(
  container: HTMLElement,
  volumePane: boolean = false
): { chartHeight: number; volumeHeight: number } {
  const totalHeight = container.clientHeight;
  const volumeHeight = volumePane ? Math.floor(totalHeight * 0.2) : 0;
  const chartHeight = totalHeight - volumeHeight;

  return { chartHeight, volumeHeight };
}

/**
 * Debounce function for resize events
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: ReturnType<typeof setTimeout> | null = null;

  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

/**
 * Deep merge chart options
 */
export function mergeChartOptions<T extends Record<string, any>>(
  defaults: T,
  overrides: Partial<T>
): T {
  const result = { ...defaults };

  Object.keys(overrides).forEach((key) => {
    const value = overrides[key];
    if (value !== undefined) {
      if (typeof value === 'object' && !Array.isArray(value) && value !== null) {
        result[key] = mergeChartOptions(result[key] || {}, value);
      } else {
        result[key] = value;
      }
    }
  });

  return result;
}

/**
 * Export chart to PNG (browser download)
 */
export function exportChartToPNG(chart: any, filename: string = 'chart.png'): void {
  // This would require the chart API to support canvas extraction
  // TradingView Lightweight Charts has limited export capabilities
  console.warn('Chart export not yet implemented');
  // TODO: Implement using html2canvas or similar library
}

/**
 * Get appropriate line width based on data density
 */
export function getAdaptiveLineWidth(dataPoints: number): number {
  if (dataPoints < 100) return 2;
  if (dataPoints < 500) return 1;
  return 1;
}

/**
 * Validate OHLCV data
 */
export function validateOHLCVData(data: OHLCVData[]): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  data.forEach((item, index) => {
    if (item.high < item.low) {
      errors.push(`Item ${index}: High (${item.high}) is less than Low (${item.low})`);
    }
    if (item.high < item.open || item.high < item.close) {
      errors.push(`Item ${index}: High must be >= Open and Close`);
    }
    if (item.low > item.open || item.low > item.close) {
      errors.push(`Item ${index}: Low must be <= Open and Close`);
    }
    if (item.volume && item.volume < 0) {
      errors.push(`Item ${index}: Volume cannot be negative`);
    }
  });

  return {
    valid: errors.length === 0,
    errors,
  };
}
