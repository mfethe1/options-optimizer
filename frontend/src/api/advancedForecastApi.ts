/**
 * Advanced Forecasting API Client - Priority #1
 *
 * TFT + TimesFM + Conformal Prediction
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface ForecastResponse {
  symbol: string;
  timestamp: string;
  current_price: number;
  horizons: number[];
  coverage_level: number;
  predictions: number[];
  tft_q10: number[];
  tft_q50: number[];
  tft_q90: number[];
  conformal_lower: number[];
  conformal_upper: number[];
  conformal_width: number[];
  expected_returns: number[];
  feature_importance: Record<string, number>;
  model: string;
  is_calibrated: boolean;
}

export interface TradingSignal {
  symbol: string;
  timestamp: string;
  action: string;
  confidence: number;
  strength: number;
  expected_return_1d: number;
  interval_width_pct: number;
  reasoning: string;
  forecast: ForecastResponse;
}

export async function getStatus() {
  const response = await axios.get(`${API_BASE_URL}/advanced-forecast/status`);
  return response.data;
}

export async function getForecast(symbol: string, useCache: boolean = true): Promise<ForecastResponse> {
  const response = await axios.post(`${API_BASE_URL}/advanced-forecast/forecast`, {
    symbol,
    use_cache: useCache
  });
  return response.data;
}

export async function getTradingSignal(symbol: string, useCache: boolean = true): Promise<TradingSignal> {
  const response = await axios.post(`${API_BASE_URL}/advanced-forecast/signal`, {
    symbol,
    use_cache: useCache
  });
  return response.data;
}

export async function trainModel(symbols: string[], epochs: number = 50, batchSize: number = 32) {
  const response = await axios.post(`${API_BASE_URL}/advanced-forecast/train`, {
    symbols,
    epochs,
    batch_size: batchSize
  });
  return response.data;
}

export async function evaluateModel(symbols: string[]) {
  const response = await axios.post(`${API_BASE_URL}/advanced-forecast/evaluate`, symbols);
  return response.data;
}

export async function getExplanation() {
  const response = await axios.get(`${API_BASE_URL}/advanced-forecast/explanation`);
  return response.data;
}
