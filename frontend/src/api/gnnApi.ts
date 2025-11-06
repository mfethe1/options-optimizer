/**
 * Graph Neural Network API Client
 */

import axios from 'axios';

const API_BASE_URL = (import.meta.env.VITE_API_URL as string | undefined) || '/api';
// DEBUG
console.info('[GNNAPI] API_BASE_URL =', API_BASE_URL);


export interface GNNForecast {
  timestamp: string;
  symbols: string[];
  predictions: Record<string, number>;
  correlations: Record<string, Record<string, number>>;
  graph_stats: {
    num_nodes: number;
    num_edges: number;
    avg_correlation: number;
    max_correlation: number;
  };
  top_correlations: Array<{
    symbol1: string;
    symbol2: string;
    correlation: number;
  }>;
}

export async function getStatus() {
  const response = await axios.get(`${API_BASE_URL}/gnn/status`);
  return response.data;
}

export async function getGNNForecast(symbols: string[], lookbackDays: number = 20): Promise<GNNForecast> {
  const response = await axios.post(`${API_BASE_URL}/gnn/forecast`, {
    symbols,
    lookback_days: lookbackDays
  });
  return response.data;
}

export async function getCorrelationGraph(symbols: string[], lookbackDays: number = 20) {
  const symbolStr = symbols.join(',');
  const response = await axios.get(`${API_BASE_URL}/gnn/graph/${symbolStr}`, {
    params: { lookback_days: lookbackDays }
  });
  return response.data;
}

export async function getExplanation() {
  const response = await axios.get(`${API_BASE_URL}/gnn/explanation`);
  return response.data;
}
