/**
 * Truth Dashboard API Client
 *
 * Handles communication with the /api/truth endpoints for model accuracy tracking.
 */

import type { ModelAccuracy } from '../components/ModelAccuracyCard';

// Backend URL
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * API Response interface for daily accuracy endpoint
 */
export interface DailyAccuracyResponse {
  models: ModelAccuracy[];
  updated_at: string;
}

/**
 * Fetch daily accuracy metrics for all models
 *
 * @returns Promise<DailyAccuracyResponse> - Model accuracy data with timestamp
 * @throws Error if the request fails
 *
 * @example
 * const data = await getDailyAccuracy();
 * console.log(data.models); // Array of ModelAccuracy objects
 * console.log(data.updated_at); // ISO timestamp string
 */
export async function getDailyAccuracy(): Promise<DailyAccuracyResponse> {
  const response = await fetch(`${API_BASE_URL}/api/truth/daily-accuracy`);

  if (!response.ok) {
    const errorText = await response.text().catch(() => 'Unknown error');
    throw new Error(`Failed to fetch accuracy data (${response.status}): ${errorText}`);
  }

  return response.json();
}

/**
 * Fetch accuracy metrics for a specific model
 *
 * @param modelName - Name of the model (e.g., 'TFT', 'GNN', 'PINN')
 * @returns Promise<ModelAccuracy> - Accuracy data for the specified model
 * @throws Error if the request fails or model not found
 */
export async function getModelAccuracy(modelName: string): Promise<ModelAccuracy> {
  const response = await fetch(`${API_BASE_URL}/api/truth/model/${encodeURIComponent(modelName)}`);

  if (!response.ok) {
    if (response.status === 404) {
      throw new Error(`Model "${modelName}" not found`);
    }
    const errorText = await response.text().catch(() => 'Unknown error');
    throw new Error(`Failed to fetch model accuracy (${response.status}): ${errorText}`);
  }

  return response.json();
}

/**
 * Get historical accuracy data for trend analysis
 *
 * @param modelName - Optional: specific model name, or all models if omitted
 * @param days - Number of days of history (default: 30)
 * @returns Promise<HistoricalAccuracyResponse> - Historical accuracy data
 */
export interface HistoricalAccuracyPoint {
  date: string;
  direction_accuracy: number;
  mape: number;
  prediction_count: number;
}

export interface HistoricalAccuracyResponse {
  model_name: string;
  history: HistoricalAccuracyPoint[];
}

export async function getHistoricalAccuracy(
  modelName?: string,
  days: number = 30
): Promise<HistoricalAccuracyResponse[]> {
  const params = new URLSearchParams({ days: days.toString() });
  if (modelName) {
    params.set('model', modelName);
  }

  const response = await fetch(`${API_BASE_URL}/api/truth/history?${params}`);

  if (!response.ok) {
    const errorText = await response.text().catch(() => 'Unknown error');
    throw new Error(`Failed to fetch historical accuracy (${response.status}): ${errorText}`);
  }

  return response.json();
}

/**
 * Check if the Truth API is available
 *
 * @returns Promise<boolean> - True if the API is reachable
 */
export async function checkTruthApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/truth/health`, {
      method: 'GET',
      // Short timeout for health check
      signal: AbortSignal.timeout(5000)
    });
    return response.ok;
  } catch {
    return false;
  }
}

export default {
  getDailyAccuracy,
  getModelAccuracy,
  getHistoricalAccuracy,
  checkTruthApiHealth
};
