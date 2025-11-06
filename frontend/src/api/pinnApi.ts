/**
 * Physics-Informed Neural Networks API Client
 * Priority #4: Option pricing & portfolio optimization
 */

import axios from 'axios';

const API_BASE_URL = (import.meta.env.VITE_API_URL as string | undefined) || '/api';
// DEBUG
console.info('[PINNAPI] API_BASE_URL =', API_BASE_URL);


export interface OptionPriceRequest {
  stock_price: number;
  strike_price: number;
  time_to_maturity: number;
  option_type: 'call' | 'put';
  risk_free_rate?: number;
  volatility?: number;
}

export interface OptionPriceResponse {
  timestamp: string;
  option_type: string;
  stock_price: number;
  strike_price: number;
  time_to_maturity: number;
  price: number;
  method: string;
  greeks: {
    delta: number | null;
    gamma: number | null;
    theta: number | null;
  };
}

export interface PortfolioOptimizationRequest {
  symbols: string[];
  target_return?: number;
  lookback_days?: number;
}

export interface PortfolioOptimizationResponse {
  timestamp: string;
  symbols: string[];
  weights: number[];
  expected_return: number;
  risk: number;
  sharpe_ratio: number;
  method: string;
}

export async function getStatus() {
  const response = await axios.get(`${API_BASE_URL}/pinn/status`);
  return response.data;
}

export async function priceOption(request: OptionPriceRequest): Promise<OptionPriceResponse> {
  const response = await axios.post(`${API_BASE_URL}/pinn/option-price`, request);
  return response.data;
}

export async function optimizePortfolio(
  request: PortfolioOptimizationRequest
): Promise<PortfolioOptimizationResponse> {
  const response = await axios.post(`${API_BASE_URL}/pinn/portfolio-optimize`, request);
  return response.data;
}

export async function trainPINNModel(
  modelType: string,
  optionType?: string,
  riskFreeRate: number = 0.05,
  volatility: number = 0.2,
  epochs: number = 1000
) {
  const response = await axios.post(`${API_BASE_URL}/pinn/train`, {
    model_type: modelType,
    option_type: optionType,
    risk_free_rate: riskFreeRate,
    volatility: volatility,
    epochs: epochs
  });
  return response.data;
}

export async function getExplanation() {
  const response = await axios.get(`${API_BASE_URL}/pinn/explanation`);
  return response.data;
}

export async function getDemoExamples() {
  const response = await axios.get(`${API_BASE_URL}/pinn/demo-examples`);
  return response.data;
}
