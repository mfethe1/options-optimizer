/**
 * Risk Dashboard API Service
 *
 * Bloomberg PORT equivalent - institutional-grade risk management
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ============================================================================
// Type Definitions
// ============================================================================

export interface PortfolioGreeks {
  total_delta: number;
  total_gamma: number;
  total_theta: number;
  total_vega: number;
  total_rho: number;
  delta_by_symbol: Record<string, number>;
  net_delta_exposure: number;
}

export interface VaRMetrics {
  var_1day_95: number;
  var_1day_99: number;
  var_10day_95: number;
  var_10day_99: number;
  cvar_1day_95: number;
  cvar_1day_99: number;
  var_as_pct_of_portfolio: number;
  method: string;
  confidence_level: number;
}

export interface StressTestResult {
  scenario: string;
  portfolio_change: number;
  portfolio_change_pct: number;
  new_portfolio_value: number;
  breaches_margin: boolean;
  delta_impact: number;
  gamma_impact: number;
  vega_impact: number;
}

export interface ConcentrationRisk {
  largest_position_pct: number;
  top_5_concentration_pct: number;
  position_count: number;
  herfindahl_index: number;
  effective_positions: number;
  sector_exposure: Record<string, number>;
}

export interface PerformanceAttribution {
  total_pnl: number;
  alpha_pnl: number;
  beta_pnl: number;
  theta_pnl: number;
  vega_pnl: number;
  gamma_pnl: number;
  realized_pnl: number;
  unrealized_pnl: number;
}

export interface RiskDashboard {
  user_id: string;
  timestamp: string;
  portfolio_value: number;
  cash: number;
  positions_value: number;
  margin_used: number;
  margin_available: number;
  buying_power: number;
  greeks: PortfolioGreeks;
  var_metrics: VaRMetrics;
  stress_tests: StressTestResult[];
  concentration: ConcentrationRisk;
  attribution: PerformanceAttribution;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  max_drawdown_pct: number;
  calculation_time_ms: number;
  data_source: string;
}

export interface GreeksResponse {
  user_id: string;
  timestamp: string;
  portfolio_value: number;
  greeks: PortfolioGreeks;
}

export interface VaRResponse {
  user_id: string;
  timestamp: string;
  portfolio_value: number;
  var_metrics: VaRMetrics;
  interpretation: {
    var_1day_95_meaning: string;
    var_1day_99_meaning: string;
    var_10day_95_meaning: string;
    cvar_explanation: string;
  };
}

export interface StressTestResponse {
  user_id: string;
  timestamp: string;
  portfolio_value: number;
  margin_available: number;
  stress_tests: Array<StressTestResult & { severity: string }>;
  worst_case: StressTestResult;
}

export interface ConcentrationResponse {
  user_id: string;
  timestamp: string;
  portfolio_value: number;
  concentration: ConcentrationRisk;
  warnings: string[];
  risk_level: string;
}

export interface AttributionResponse {
  user_id: string;
  timestamp: string;
  portfolio_value: number;
  attribution: PerformanceAttribution;
  attribution_pct: {
    alpha_pct: number;
    beta_pct: number;
    theta_pct: number;
    vega_pct: number;
    gamma_pct: number;
  };
  realized_vs_unrealized: {
    realized_pct: number;
    unrealized_pct: number;
  };
}

export interface PerformanceMetricsResponse {
  user_id: string;
  timestamp: string;
  portfolio_value: number;
  metrics: {
    sharpe_ratio: number;
    sortino_ratio: number;
    max_drawdown: number;
    max_drawdown_pct: number;
  };
  interpretation: {
    sharpe_rating: string;
    sharpe_explanation: string;
    sortino_explanation: string;
    max_drawdown_explanation: string;
  };
  lookback_days: number;
}

// ============================================================================
// API Functions
// ============================================================================

/**
 * Get complete risk dashboard
 */
export async function getRiskDashboard(
  userId: string,
  lookbackDays: number = 252
): Promise<RiskDashboard> {
  const response = await fetch(
    `${API_BASE_URL}/api/risk-dashboard/${userId}?lookback_days=${lookbackDays}`
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch risk dashboard: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get portfolio Greeks only (faster endpoint)
 */
export async function getPortfolioGreeks(userId: string): Promise<GreeksResponse> {
  const response = await fetch(`${API_BASE_URL}/api/risk-dashboard/${userId}/greeks`);

  if (!response.ok) {
    throw new Error(`Failed to fetch portfolio Greeks: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get VaR metrics only
 */
export async function getVaRMetrics(
  userId: string,
  lookbackDays: number = 252
): Promise<VaRResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/risk-dashboard/${userId}/var?lookback_days=${lookbackDays}`
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch VaR metrics: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get stress test results
 */
export async function getStressTests(userId: string): Promise<StressTestResponse> {
  const response = await fetch(`${API_BASE_URL}/api/risk-dashboard/${userId}/stress-tests`);

  if (!response.ok) {
    throw new Error(`Failed to fetch stress tests: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get concentration risk metrics
 */
export async function getConcentrationRisk(userId: string): Promise<ConcentrationResponse> {
  const response = await fetch(`${API_BASE_URL}/api/risk-dashboard/${userId}/concentration`);

  if (!response.ok) {
    throw new Error(`Failed to fetch concentration risk: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get performance attribution breakdown
 */
export async function getPerformanceAttribution(userId: string): Promise<AttributionResponse> {
  const response = await fetch(`${API_BASE_URL}/api/risk-dashboard/${userId}/attribution`);

  if (!response.ok) {
    throw new Error(`Failed to fetch performance attribution: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get risk-adjusted performance metrics
 */
export async function getPerformanceMetrics(
  userId: string,
  lookbackDays: number = 252
): Promise<PerformanceMetricsResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/risk-dashboard/${userId}/performance-metrics?lookback_days=${lookbackDays}`
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch performance metrics: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Health check for risk dashboard service
 */
export async function checkRiskDashboardHealth(): Promise<{ status: string; capabilities: string[] }> {
  const response = await fetch(`${API_BASE_URL}/api/risk-dashboard/health`);

  if (!response.ok) {
    throw new Error(`Risk dashboard health check failed: ${response.statusText}`);
  }

  return response.json();
}
