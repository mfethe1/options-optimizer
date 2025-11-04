/**
 * Technical Indicator Calculations
 *
 * Pure functions for calculating common trading indicators:
 * - Simple Moving Average (SMA)
 * - Exponential Moving Average (EMA)
 * - Bollinger Bands
 * - Relative Strength Index (RSI)
 * - MACD (Moving Average Convergence Divergence)
 * - Average True Range (ATR)
 * - Stochastic Oscillator
 *
 * All functions return data in TradingView format (time, value)
 */

import { Time, LineData } from 'lightweight-charts';
import { OHLCVData } from './chartTypes';
import { toTime } from './chartUtils';

/**
 * Simple Moving Average (SMA)
 *
 * SMA = (Sum of prices over N periods) / N
 */
export function calculateSMA(data: OHLCVData[], period: number): LineData[] {
  if (data.length < period) return [];

  const result: LineData[] = [];

  for (let i = period - 1; i < data.length; i++) {
    const slice = data.slice(i - period + 1, i + 1);
    const sum = slice.reduce((acc, item) => acc + item.close, 0);
    const average = sum / period;

    result.push({
      time: toTime(data[i].time),
      value: average,
    });
  }

  return result;
}

/**
 * Exponential Moving Average (EMA)
 *
 * EMA = Price(t) × k + EMA(y) × (1 − k)
 * where k = 2 / (N + 1)
 */
export function calculateEMA(data: OHLCVData[], period: number): LineData[] {
  if (data.length < period) return [];

  const result: LineData[] = [];
  const multiplier = 2 / (period + 1);

  // Start with SMA for first value
  let ema = data.slice(0, period).reduce((acc, item) => acc + item.close, 0) / period;

  result.push({
    time: toTime(data[period - 1].time),
    value: ema,
  });

  // Calculate EMA for remaining values
  for (let i = period; i < data.length; i++) {
    ema = (data[i].close - ema) * multiplier + ema;
    result.push({
      time: toTime(data[i].time),
      value: ema,
    });
  }

  return result;
}

/**
 * Bollinger Bands
 *
 * Middle Band = SMA(N)
 * Upper Band = SMA(N) + (k × stddev)
 * Lower Band = SMA(N) − (k × stddev)
 */
export interface BollingerBands {
  upper: LineData[];
  middle: LineData[];
  lower: LineData[];
}

export function calculateBollingerBands(
  data: OHLCVData[],
  period: number = 20,
  stdDevMultiplier: number = 2
): BollingerBands {
  if (data.length < period) {
    return { upper: [], middle: [], lower: [] };
  }

  const upper: LineData[] = [];
  const middle: LineData[] = [];
  const lower: LineData[] = [];

  for (let i = period - 1; i < data.length; i++) {
    const slice = data.slice(i - period + 1, i + 1);
    const prices = slice.map((item) => item.close);

    // Calculate SMA (middle band)
    const sum = prices.reduce((acc, price) => acc + price, 0);
    const sma = sum / period;

    // Calculate standard deviation
    const squaredDiffs = prices.map((price) => Math.pow(price - sma, 2));
    const variance = squaredDiffs.reduce((acc, val) => acc + val, 0) / period;
    const stdDev = Math.sqrt(variance);

    const time = toTime(data[i].time);

    upper.push({ time, value: sma + stdDevMultiplier * stdDev });
    middle.push({ time, value: sma });
    lower.push({ time, value: sma - stdDevMultiplier * stdDev });
  }

  return { upper, middle, lower };
}

/**
 * Relative Strength Index (RSI)
 *
 * RSI = 100 - (100 / (1 + RS))
 * where RS = Average Gain / Average Loss
 */
export function calculateRSI(data: OHLCVData[], period: number = 14): LineData[] {
  if (data.length < period + 1) return [];

  const result: LineData[] = [];
  const changes: number[] = [];

  // Calculate price changes
  for (let i = 1; i < data.length; i++) {
    changes.push(data[i].close - data[i - 1].close);
  }

  // Calculate initial average gain and loss
  let avgGain = 0;
  let avgLoss = 0;

  for (let i = 0; i < period; i++) {
    if (changes[i] > 0) {
      avgGain += changes[i];
    } else {
      avgLoss += Math.abs(changes[i]);
    }
  }

  avgGain /= period;
  avgLoss /= period;

  // Calculate RSI for each subsequent period
  for (let i = period; i < changes.length; i++) {
    const change = changes[i];

    if (change > 0) {
      avgGain = (avgGain * (period - 1) + change) / period;
      avgLoss = (avgLoss * (period - 1)) / period;
    } else {
      avgGain = (avgGain * (period - 1)) / period;
      avgLoss = (avgLoss * (period - 1) + Math.abs(change)) / period;
    }

    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    const rsi = 100 - 100 / (1 + rs);

    result.push({
      time: toTime(data[i + 1].time),
      value: rsi,
    });
  }

  return result;
}

/**
 * MACD (Moving Average Convergence Divergence)
 *
 * MACD Line = EMA(12) - EMA(26)
 * Signal Line = EMA(9) of MACD Line
 * Histogram = MACD Line - Signal Line
 */
export interface MACDData {
  macd: LineData[];
  signal: LineData[];
  histogram: LineData[];
}

export function calculateMACD(
  data: OHLCVData[],
  fastPeriod: number = 12,
  slowPeriod: number = 26,
  signalPeriod: number = 9
): MACDData {
  if (data.length < slowPeriod) {
    return { macd: [], signal: [], histogram: [] };
  }

  // Calculate EMAs
  const emaFast = calculateEMA(data, fastPeriod);
  const emaSlow = calculateEMA(data, slowPeriod);

  // Calculate MACD line
  const macdLine: LineData[] = [];
  const startIdx = slowPeriod - fastPeriod;

  for (let i = 0; i < emaSlow.length; i++) {
    const fastValue = emaFast[i + startIdx].value;
    const slowValue = emaSlow[i].value;

    macdLine.push({
      time: emaSlow[i].time,
      value: fastValue - slowValue,
    });
  }

  // Calculate signal line (EMA of MACD)
  const signal: LineData[] = [];
  if (macdLine.length >= signalPeriod) {
    const multiplier = 2 / (signalPeriod + 1);
    let emaSignal = macdLine.slice(0, signalPeriod).reduce((acc, item) => acc + item.value, 0) / signalPeriod;

    signal.push({
      time: macdLine[signalPeriod - 1].time,
      value: emaSignal,
    });

    for (let i = signalPeriod; i < macdLine.length; i++) {
      emaSignal = (macdLine[i].value - emaSignal) * multiplier + emaSignal;
      signal.push({
        time: macdLine[i].time,
        value: emaSignal,
      });
    }
  }

  // Calculate histogram
  const histogram: LineData[] = [];
  const signalStartIdx = macdLine.length - signal.length;

  for (let i = 0; i < signal.length; i++) {
    histogram.push({
      time: signal[i].time,
      value: macdLine[i + signalStartIdx].value - signal[i].value,
    });
  }

  return { macd: macdLine, signal, histogram };
}

/**
 * Average True Range (ATR)
 *
 * Used for volatility measurement and position sizing
 * TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
 * ATR = EMA of TR
 */
export function calculateATR(data: OHLCVData[], period: number = 14): LineData[] {
  if (data.length < period + 1) return [];

  const trueRanges: number[] = [];

  // Calculate True Range for each period
  for (let i = 1; i < data.length; i++) {
    const high = data[i].high;
    const low = data[i].low;
    const prevClose = data[i - 1].close;

    const tr = Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));

    trueRanges.push(tr);
  }

  // Calculate ATR (EMA of TR)
  const result: LineData[] = [];
  const multiplier = 1 / period;

  let atr = trueRanges.slice(0, period).reduce((acc, tr) => acc + tr, 0) / period;

  result.push({
    time: toTime(data[period].time),
    value: atr,
  });

  for (let i = period; i < trueRanges.length; i++) {
    atr = (trueRanges[i] - atr) * multiplier + atr;
    result.push({
      time: toTime(data[i + 1].time),
      value: atr,
    });
  }

  return result;
}

/**
 * Stochastic Oscillator
 *
 * %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) × 100
 * %D = SMA of %K
 */
export interface StochasticData {
  k: LineData[];
  d: LineData[];
}

export function calculateStochastic(
  data: OHLCVData[],
  kPeriod: number = 14,
  dPeriod: number = 3
): StochasticData {
  if (data.length < kPeriod) {
    return { k: [], d: [] };
  }

  const kValues: LineData[] = [];

  // Calculate %K
  for (let i = kPeriod - 1; i < data.length; i++) {
    const slice = data.slice(i - kPeriod + 1, i + 1);
    const lowestLow = Math.min(...slice.map((item) => item.low));
    const highestHigh = Math.max(...slice.map((item) => item.high));
    const currentClose = data[i].close;

    const k = ((currentClose - lowestLow) / (highestHigh - lowestLow)) * 100;

    kValues.push({
      time: toTime(data[i].time),
      value: k,
    });
  }

  // Calculate %D (SMA of %K)
  const dValues: LineData[] = [];

  for (let i = dPeriod - 1; i < kValues.length; i++) {
    const slice = kValues.slice(i - dPeriod + 1, i + 1);
    const sum = slice.reduce((acc, item) => acc + item.value, 0);
    const avg = sum / dPeriod;

    dValues.push({
      time: kValues[i].time,
      value: avg,
    });
  }

  return { k: kValues, d: dValues };
}

/**
 * Volume Weighted Average Price (VWAP)
 *
 * VWAP = Σ(Price × Volume) / Σ(Volume)
 */
export function calculateVWAP(data: OHLCVData[]): LineData[] {
  const result: LineData[] = [];
  let cumulativePV = 0;
  let cumulativeVolume = 0;

  for (let i = 0; i < data.length; i++) {
    if (!data[i].volume) continue;

    const typicalPrice = (data[i].high + data[i].low + data[i].close) / 3;
    cumulativePV += typicalPrice * data[i].volume!;
    cumulativeVolume += data[i].volume!;

    result.push({
      time: toTime(data[i].time),
      value: cumulativePV / cumulativeVolume,
    });
  }

  return result;
}

/**
 * On-Balance Volume (OBV)
 *
 * Momentum indicator that uses volume flow
 */
export function calculateOBV(data: OHLCVData[]): LineData[] {
  if (data.length === 0) return [];

  const result: LineData[] = [];
  let obv = 0;

  for (let i = 0; i < data.length; i++) {
    if (!data[i].volume) continue;

    if (i > 0) {
      if (data[i].close > data[i - 1].close) {
        obv += data[i].volume!;
      } else if (data[i].close < data[i - 1].close) {
        obv -= data[i].volume!;
      }
    }

    result.push({
      time: toTime(data[i].time),
      value: obv,
    });
  }

  return result;
}

/**
 * Parabolic SAR (Stop and Reverse)
 *
 * Trend-following indicator
 */
export function calculateParabolicSAR(
  data: OHLCVData[],
  acceleration: number = 0.02,
  maximum: number = 0.2
): LineData[] {
  if (data.length < 2) return [];

  const result: LineData[] = [];
  let isUptrend = data[1].close > data[0].close;
  let sar = isUptrend ? data[0].low : data[0].high;
  let ep = isUptrend ? data[1].high : data[1].low;
  let af = acceleration;

  for (let i = 1; i < data.length; i++) {
    // Calculate new SAR
    sar = sar + af * (ep - sar);

    // Check for trend reversal
    if (isUptrend) {
      if (data[i].low < sar) {
        isUptrend = false;
        sar = ep;
        ep = data[i].low;
        af = acceleration;
      } else {
        if (data[i].high > ep) {
          ep = data[i].high;
          af = Math.min(af + acceleration, maximum);
        }
      }
    } else {
      if (data[i].high > sar) {
        isUptrend = true;
        sar = ep;
        ep = data[i].high;
        af = acceleration;
      } else {
        if (data[i].low < ep) {
          ep = data[i].low;
          af = Math.min(af + acceleration, maximum);
        }
      }
    }

    result.push({
      time: toTime(data[i].time),
      value: sar,
    });
  }

  return result;
}

/**
 * Helper: Get all available indicators
 */
export const AVAILABLE_INDICATORS = [
  'sma',
  'ema',
  'bollinger',
  'rsi',
  'macd',
  'atr',
  'stochastic',
  'vwap',
  'obv',
  'sar',
] as const;

export type IndicatorType = typeof AVAILABLE_INDICATORS[number];
