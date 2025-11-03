/**
 * ML Prediction API Client
 *
 * TypeScript client for ML-based price predictions.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ============================================================================
// Types
// ============================================================================

export interface MLPrediction {
  symbol: string;
  timestamp: string;
  predicted_direction: 'UP' | 'DOWN' | 'NEUTRAL';
  recommendation: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  target_price_1d: number;
  target_price_5d: number;
  expected_return_5d: number;
  downside_risk: number;
  current_price: number;
  models_used: string[];
}

export interface ModelInfo {
  symbol: string;
  model_type: string;
  model_exists: boolean;
  last_modified?: string;
  sequence_length?: number;
  prediction_horizon?: number;
}

export interface ModelTrainRequest {
  symbol: string;
  years?: number;
  epochs?: number;
  force_retrain?: boolean;
}

export interface MLStrategy {
  name: string;
  display_name: string;
  description: string;
  model: string;
  expected_accuracy: string;
  expected_sharpe: string;
  holding_period: string;
  best_for: string;
}

// ============================================================================
// API Functions
// ============================================================================

/**
 * Get ML price prediction for a symbol
 */
export async function getMLPrediction(symbol: string, forceRefresh: boolean = false): Promise<MLPrediction> {
  const response = await fetch(`${API_BASE_URL}/api/ml/predict/${symbol}?force_refresh=${forceRefresh}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get ML prediction');
  }

  return response.json();
}

/**
 * Get batch predictions for multiple symbols
 */
export async function getBatchPredictions(symbols: string[]): Promise<MLPrediction[]> {
  const response = await fetch(`${API_BASE_URL}/api/ml/predict/batch`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ symbols }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get batch predictions');
  }

  const data = await response.json();
  return data.predictions;
}

/**
 * Train ML model for a symbol
 */
export async function trainModel(request: ModelTrainRequest): Promise<{ message: string; status: string }> {
  const response = await fetch(`${API_BASE_URL}/api/ml/train`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to start model training');
  }

  return response.json();
}

/**
 * Get model information
 */
export async function getModelInfo(symbol: string): Promise<ModelInfo> {
  const response = await fetch(`${API_BASE_URL}/api/ml/model/info/${symbol}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get model info');
  }

  return response.json();
}

/**
 * Get available ML strategies
 */
export async function getMLStrategies(): Promise<{ strategies: MLStrategy[] }> {
  const response = await fetch(`${API_BASE_URL}/api/ml/strategies`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get ML strategies');
  }

  return response.json();
}

/**
 * Check ML service health
 */
export async function getMLHealth(): Promise<{ status: string; tensorflow_available: boolean }> {
  const response = await fetch(`${API_BASE_URL}/api/ml/health`);

  if (!response.ok) {
    throw new Error('Failed to check ML service health');
  }

  return response.json();
}
