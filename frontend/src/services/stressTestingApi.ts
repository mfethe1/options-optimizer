/**
 * Stress Testing & Scenario Analysis API Client
 *
 * TypeScript client for portfolio stress testing and Monte Carlo simulation.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ============================================================================
// Type Definitions
// ============================================================================

export interface MarketShock {
  equity_return: number;
  volatility_change: number;
  interest_rate_change: number;
  time_horizon_days: number;
  correlation_shock: number;
}

export interface PortfolioPosition {
  symbol: string;
  type: 'stock' | 'call' | 'put';
  quantity: number;
  current_price: number;
  strike?: number;
  expiration?: string;
  time_value?: number;
}

export interface Portfolio {
  positions: PortfolioPosition[];
}

export interface PositionStressResult {
  symbol: string;
  position_type: string;
  current_value: number;
  stressed_value: number;
  pnl: number;
  pnl_pct: number;
  contribution_to_total_pnl: number;
}

export interface PortfolioStressResult {
  scenario_name: string;
  scenario_type: string;
  timestamp: string;
  current_portfolio_value: number;
  stressed_portfolio_value: number;
  total_pnl: number;
  total_pnl_pct: number;
  max_drawdown: number;
  var_95: number;
  cvar_95: number;
  position_results: PositionStressResult[];
  market_shock: MarketShock;
}

export interface MonteCarloResult {
  timestamp: string;
  num_simulations: number;
  time_horizon_days: number;
  mean_pnl: number;
  median_pnl: number;
  std_pnl: number;
  pnl_5th: number;
  pnl_25th: number;
  pnl_75th: number;
  pnl_95th: number;
  var_95: number;
  cvar_95: number;
  max_drawdown_mean: number;
  max_drawdown_95th: number;
  prob_loss_10pct: number;
  prob_loss_20pct: number;
  prob_gain_10pct: number;
  prob_gain_20pct: number;
  pnl_distribution?: number[];
}

export interface ScenarioInfo {
  scenario_type: string;
  name: string;
  description: string;
  date_range: string;
  probability: number;
  market_shock: MarketShock;
}

export interface ScenarioRunRequest {
  scenario_type: string;
  portfolio: Portfolio;
  custom_shock?: MarketShock;
}

export interface MonteCarloRequest {
  portfolio: Portfolio;
  num_simulations?: number;
  time_horizon_days?: number;
  confidence_level?: number;
}

// ============================================================================
// API Functions
// ============================================================================

/**
 * Run a specific stress test scenario
 */
export async function runScenario(request: ScenarioRunRequest): Promise<PortfolioStressResult> {
  const response = await fetch(`${API_BASE_URL}/api/stress-testing/scenario/run`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`Failed to run scenario: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Run all historical scenarios
 */
export async function runAllScenarios(portfolio: Portfolio): Promise<{ scenarios: PortfolioStressResult[]; count: number }> {
  const response = await fetch(`${API_BASE_URL}/api/stress-testing/scenario/run-all`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(portfolio),
  });

  if (!response.ok) {
    throw new Error(`Failed to run all scenarios: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Run Monte Carlo simulation
 */
export async function runMonteCarlo(request: MonteCarloRequest): Promise<MonteCarloResult> {
  const response = await fetch(`${API_BASE_URL}/api/stress-testing/monte-carlo`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`Failed to run Monte Carlo: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get all available scenarios
 */
export async function getScenarios(): Promise<ScenarioInfo[]> {
  const response = await fetch(`${API_BASE_URL}/api/stress-testing/scenarios`);

  if (!response.ok) {
    throw new Error(`Failed to get scenarios: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get information about a specific scenario
 */
export async function getScenarioInfo(scenarioType: string): Promise<ScenarioInfo> {
  const response = await fetch(`${API_BASE_URL}/api/stress-testing/scenarios/${scenarioType}`);

  if (!response.ok) {
    throw new Error(`Failed to get scenario info: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Health check for stress testing service
 */
export async function checkHealth(): Promise<{ status: string; message: string }> {
  const response = await fetch(`${API_BASE_URL}/api/stress-testing/health`);

  if (!response.ok) {
    throw new Error(`Health check failed: ${response.statusText}`);
  }

  return response.json();
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Create a sample portfolio for testing
 */
export function createSamplePortfolio(): Portfolio {
  return {
    positions: [
      {
        symbol: 'SPY',
        type: 'stock',
        quantity: 100,
        current_price: 450.00,
      },
      {
        symbol: 'AAPL',
        type: 'stock',
        quantity: 50,
        current_price: 180.50,
      },
      {
        symbol: 'QQQ',
        type: 'call',
        quantity: 10,
        current_price: 5.50,
        strike: 380,
        time_value: 2.00,
      },
    ],
  };
}

/**
 * Format currency
 */
export function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

/**
 * Format percentage
 */
export function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

/**
 * Get scenario color based on P&L
 */
export function getScenarioColor(pnlPct: number): string {
  if (pnlPct >= 0) return 'text-green-600';
  if (pnlPct > -0.05) return 'text-yellow-600';
  if (pnlPct > -0.10) return 'text-orange-600';
  return 'text-red-600';
}

/**
 * Get risk level based on VaR
 */
export function getRiskLevel(var95: number, portfolioValue: number): string {
  const varPct = var95 / portfolioValue;
  if (varPct < 0.05) return 'Low';
  if (varPct < 0.10) return 'Medium';
  if (varPct < 0.20) return 'High';
  return 'Critical';
}
