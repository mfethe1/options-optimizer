/**
 * Mamba State Space Model API Client
 * Priority #3: Linear-time sequence modeling
 */

import axios from 'axios';

// Backend URL - routes don't have /api prefix
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
// DEBUG
console.info('[MambaAPI] API_BASE_URL =', API_BASE_URL);


export interface MambaForecast {
  timestamp: string;
  symbol: string;
  current_price: number;
  predictions: Record<string, number>;
  efficiency_stats: {
    sequence_length: number;
    mamba_complexity: string;
    transformer_complexity: string;
    mamba_ops: number;
    transformer_ops: number;
    theoretical_speedup: string;
    can_process_ticks: boolean;
    memory_efficient: boolean;
  };
  signal: string;
  confidence: number;
}

export interface EfficiencyComparison {
  sequence_length: number;
  mamba_complexity: string;
  transformer_complexity: string;
  mamba_ops: number;
  transformer_ops: number;
  theoretical_speedup: string;
  can_process_ticks: boolean;
  memory_efficient: boolean;
}

export async function getStatus() {
  const response = await axios.get(`${API_BASE_URL}/mamba/status`);
  return response.data;
}

export async function getMambaForecast(
  symbol: string,
  sequenceLength: number = 1000,
  useCache: boolean = true
): Promise<MambaForecast> {
  const response = await axios.post(`${API_BASE_URL}/mamba/forecast`, {
    symbol,
    sequence_length: sequenceLength,
    use_cache: useCache
  });
  return response.data;
}

export async function trainMambaModel(
  symbols: string[],
  epochs: number = 50,
  sequenceLength: number = 1000
) {
  const response = await axios.post(`${API_BASE_URL}/mamba/train`, {
    symbols,
    epochs,
    sequence_length: sequenceLength
  });
  return response.data;
}

export async function analyzeEfficiency(
  sequenceLengths: number[]
): Promise<{ comparisons: EfficiencyComparison[]; summary: any }> {
  const response = await axios.post(`${API_BASE_URL}/mamba/efficiency-analysis`, {
    sequence_lengths: sequenceLengths
  });
  return response.data;
}

export async function getExplanation() {
  const response = await axios.get(`${API_BASE_URL}/mamba/explanation`);
  return response.data;
}

export async function getDemoScenarios() {
  const response = await axios.get(`${API_BASE_URL}/mamba/demo-scenarios`);
  return response.data;
}
