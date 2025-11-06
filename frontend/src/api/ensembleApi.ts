/**
 * Ensemble Neural Network API Client
 * Multi-model analysis and predictions
 */

import axios from 'axios';

const API_BASE_URL = (import.meta.env.VITE_API_URL as string | undefined) || '/api';
// DEBUG
console.info('[EnsembleAPI] API_BASE_URL =', API_BASE_URL);


export interface ModelPrediction {
  model_name: string;
  price_prediction: number;
  confidence: number;
  signal: string;
  lower_bound?: number;
  upper_bound?: number;
  metadata?: Record<string, any>;
}

export interface EnsembleAnalysis {
  timestamp: string;
  symbol: string;
  current_price: number;
  time_horizon: string;

  // Ensemble results
  ensemble_prediction: number;
  ensemble_signal: string;
  ensemble_confidence: number;

  // Individual model predictions
  model_predictions: ModelPrediction[];

  // Model weights
  model_weights: Record<string, number>;

  // Uncertainty metrics
  prediction_std: number;
  model_agreement: number;

  // Trading recommendation
  position_size: number;
  stop_loss?: number;
  take_profit?: number;
  expected_return: number;
  risk_reward_ratio?: number;
}

export interface ModelPerformanceMetric {
  model_name: string;
  accuracy: number;
  sharpe_ratio: number;
  current_weight: number;
}

export async function getStatus() {
  const response = await axios.get(`${API_BASE_URL}/ensemble/status`);
  return response.data;
}

export async function getEnsembleAnalysis(
  symbol: string,
  timeHorizon: string = 'short_term',
  useCache: boolean = true
): Promise<EnsembleAnalysis> {
  const response = await axios.post(`${API_BASE_URL}/ensemble/analyze`, {
    symbol,
    time_horizon: timeHorizon,
    use_cache: useCache
  });
  return response.data;
}

export async function getModelPerformance() {
  const response = await axios.get(`${API_BASE_URL}/ensemble/performance`);
  return response.data;
}

export async function getExplanation() {
  const response = await axios.get(`${API_BASE_URL}/ensemble/explanation`);
  return response.data;
}
