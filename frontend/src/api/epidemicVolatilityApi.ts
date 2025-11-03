/**
 * Epidemic Volatility API Client
 *
 * Bio-Financial Breakthrough: Disease dynamics for market fear prediction
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface EpidemicForecast {
  timestamp: string;
  horizon_days: number;
  predicted_vix: number;
  predicted_regime: string;
  confidence: number;
  current_vix: number;
  current_sentiment: number;
  trading_signal: {
    action: string;
    confidence: number;
    reasoning: string;
  };
  interpretation: string;
}

export interface EpidemicState {
  timestamp: string;
  regime: string;
  susceptible: number;
  infected: number;
  recovered: number;
  exposed?: number;
  beta: number;
  gamma: number;
  current_vix: number;
  current_sentiment: number;
}

export interface EpidemicEpisode {
  start_date: string;
  end_date: string;
  duration_days: number;
  peak_vix: number;
  start_vix: number;
  end_vix: number;
  severity: string;
}

export interface HistoricalEpisodes {
  episodes: EpidemicEpisode[];
  total_episodes: number;
}

export interface ServiceStatus {
  status: string;
  predictor_ready: boolean;
  data_service_ready: boolean;
  trainer_ready: boolean;
  model_type: string;
  description: string;
}

export interface TrainingResults {
  status: string;
  message: string;
  results: {
    model_type: string;
    epochs_trained: number;
    final_loss: number;
    final_val_loss: number;
    final_mae: number;
    final_val_mae: number;
    training_samples: number;
    model_path: string;
  };
}

export interface EvaluationResults {
  test_samples: number;
  mse: number;
  mae: number;
  rmse: number;
  directional_accuracy: number;
}

export interface ModelExplanation {
  title: string;
  concept: string;
  model: {
    type: string;
    states: Record<string, string>;
    parameters: Record<string, string>;
  };
  equations: {
    SIR: string[];
    SEIR: string[];
  };
  innovation: string;
  advantages: string[];
  use_cases: string[];
}

/**
 * Get service status
 */
export async function getEpidemicStatus(): Promise<ServiceStatus> {
  const response = await axios.get(`${API_BASE_URL}/epidemic/status`);
  return response.data;
}

/**
 * Get epidemic volatility forecast
 */
export async function getEpidemicForecast(
  horizonDays: number = 30,
  modelType: string = 'SEIR'
): Promise<EpidemicForecast> {
  const response = await axios.post(`${API_BASE_URL}/epidemic/forecast`, {
    horizon_days: horizonDays,
    model_type: modelType
  });
  return response.data;
}

/**
 * Get current epidemic state of the market
 */
export async function getCurrentEpidemicState(): Promise<EpidemicState> {
  const response = await axios.get(`${API_BASE_URL}/epidemic/current-state`);
  return response.data;
}

/**
 * Get historical epidemic episodes
 */
export async function getHistoricalEpisodes(): Promise<HistoricalEpisodes> {
  const response = await axios.get(`${API_BASE_URL}/epidemic/historical-episodes`);
  return response.data;
}

/**
 * Train epidemic volatility model
 */
export async function trainEpidemicModel(
  modelType: string = 'SEIR',
  epochs: number = 100,
  batchSize: number = 32,
  physicsWeight: number = 0.1
): Promise<TrainingResults> {
  const response = await axios.post(`${API_BASE_URL}/epidemic/train`, {
    model_type: modelType,
    epochs: epochs,
    batch_size: batchSize,
    physics_weight: physicsWeight
  });
  return response.data;
}

/**
 * Evaluate trained model
 */
export async function evaluateEpidemicModel(): Promise<EvaluationResults> {
  const response = await axios.get(`${API_BASE_URL}/epidemic/evaluate`);
  return response.data;
}

/**
 * Get model explanation
 */
export async function getEpidemicExplanation(): Promise<ModelExplanation> {
  const response = await axios.get(`${API_BASE_URL}/epidemic/explanation`);
  return response.data;
}
