/**
 * AI Trading Services API
 *
 * Client for swarm analysis, risk management, and expert critique services.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// ============================================================================
// Type Definitions
// ============================================================================

export interface BacktestResult {
  strategy_name: string;
  symbol: string;
  timeframe: string;
  total_return: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  win_rate: number;
  profit_factor: number;
  avg_win: number;
  avg_loss: number;
  total_trades: number;
  kelly_criterion: number;
  var_95: number;
  expected_value: number;
}

export interface AgentAnalysis {
  agent_name: string;
  agent_type: string;
  score: number;
  recommendation: string;
  confidence: number;
  reasoning: string;
  risk_concerns: string[];
  opportunity_highlights: string[];
}

export interface SwarmConsensus {
  strategy: string;
  symbol: string;
  overall_score: number;
  consensus_recommendation: string;
  consensus_confidence: number;
  expected_value: number;
  risk_adjusted_return: number;
  suggested_position_size: number;
  stop_loss: number;
  take_profit: number;
  max_loss_per_trade: number;
  agent_votes: Record<string, number>;
  agent_analyses: AgentAnalysis[];
  risk_warnings: string[];
  go_decision: boolean;
  reasoning_summary: string;
}

export interface RankedStrategy {
  rank: number;
  strategy: string;
  symbol: string;
  score: number;
  recommendation: string;
  confidence: number;
  go_decision: boolean;
  expected_value: number;
  suggested_position_size: number;
}

export interface RiskViolation {
  severity: string;
  rule_name: string;
  current_value: number;
  limit_value: number;
  message: string;
  blocking: boolean;
}

export interface RiskCheckResult {
  approved: boolean;
  violations: RiskViolation[];
  warnings: RiskViolation[];
  max_position_size: number;
  suggested_position_size: number;
  risk_score: number;
  risk_level: string;
  detailed_report: string;
}

export interface Recommendation {
  priority: string;
  category: string;
  title: string;
  current_state: string;
  desired_state: string;
  rationale: string;
  expected_impact: string;
  implementation_complexity: string;
  estimated_value: string;
}

export interface ExpertCritiqueReport {
  overall_rating: string;
  overall_score: number;
  executive_summary: string;
  competitive_positioning: string;
  competitive_scores: {
    vs_bloomberg: number;
    vs_refinitiv: number;
    vs_factset: number;
  };
  category_scores: {
    data_quality: number;
    analytics: number;
    execution: number;
    risk_management: number;
    user_experience: number;
    technology: number;
  };
  strengths: string[];
  critical_gaps: string[];
  recommendations: Recommendation[];
  generated_at: string;
}

// ============================================================================
// Swarm Analysis
// ============================================================================

export async function analyzeStrategy(backtest: BacktestResult): Promise<SwarmConsensus> {
  const response = await fetch(`${API_BASE_URL}/api/ai/swarm/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(backtest),
  });

  if (!response.ok) {
    throw new Error(`Swarm analysis failed: ${response.statusText}`);
  }

  return response.json();
}

export async function compareStrategies(backtests: BacktestResult[]): Promise<{
  total_strategies: number;
  best_strategy: string | null;
  ranked_strategies: RankedStrategy[];
}> {
  const response = await fetch(`${API_BASE_URL}/api/ai/swarm/compare`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(backtests),
  });

  if (!response.ok) {
    throw new Error(`Strategy comparison failed: ${response.statusText}`);
  }

  return response.json();
}

// ============================================================================
// Risk Management
// ============================================================================

export async function checkNewPosition(
  symbol: string,
  proposedSize: number,
  positionType: string,
  portfolio: any,
  marketData: Record<string, any> = {},
  riskLevel: string = 'moderate'
): Promise<RiskCheckResult> {
  const response = await fetch(
    `${API_BASE_URL}/api/ai/risk/check-position?risk_level=${riskLevel}`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        symbol,
        proposed_size: proposedSize,
        position_type: positionType,
        portfolio,
        market_data: marketData,
      }),
    }
  );

  if (!response.ok) {
    throw new Error(`Risk check failed: ${response.statusText}`);
  }

  return response.json();
}

export async function getRiskLimits(riskLevel: string): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/api/ai/risk/limits/${riskLevel}`);

  if (!response.ok) {
    throw new Error(`Failed to get risk limits: ${response.statusText}`);
  }

  return response.json();
}

export async function getPositionSizing(
  kellyFraction: number,
  winRate: number,
  avgWin: number,
  avgLoss: number,
  portfolioValue: number,
  riskLevel: string = 'moderate'
): Promise<any> {
  const params = new URLSearchParams({
    kelly_fraction: kellyFraction.toString(),
    win_rate: winRate.toString(),
    avg_win: avgWin.toString(),
    avg_loss: avgLoss.toString(),
    portfolio_value: portfolioValue.toString(),
    risk_level: riskLevel,
  });

  const response = await fetch(
    `${API_BASE_URL}/api/ai/risk/position-sizing?${params.toString()}`,
    {
      method: 'POST',
    }
  );

  if (!response.ok) {
    throw new Error(`Position sizing calculation failed: ${response.statusText}`);
  }

  return response.json();
}

// ============================================================================
// Expert Critique
// ============================================================================

export async function getPlatformCritique(): Promise<ExpertCritiqueReport> {
  const response = await fetch(`${API_BASE_URL}/api/ai/critique/platform`);

  if (!response.ok) {
    throw new Error(`Platform critique failed: ${response.statusText}`);
  }

  return response.json();
}

// ============================================================================
// Helper Functions
// ============================================================================

export function getRecommendationColor(recommendation: string): string {
  switch (recommendation) {
    case 'STRONG_BUY':
      return 'text-green-700 bg-green-100';
    case 'BUY':
      return 'text-green-600 bg-green-50';
    case 'HOLD':
      return 'text-yellow-600 bg-yellow-50';
    case 'SELL':
      return 'text-red-600 bg-red-50';
    case 'STRONG_SELL':
      return 'text-red-700 bg-red-100';
    default:
      return 'text-gray-600 bg-gray-50';
  }
}

export function getRiskLevelColor(riskLevel: string): string {
  switch (riskLevel) {
    case 'MINIMAL':
    case 'LOW':
      return 'text-green-700 bg-green-100';
    case 'MODERATE':
      return 'text-yellow-600 bg-yellow-100';
    case 'HIGH':
      return 'text-orange-600 bg-orange-100';
    case 'EXTREME':
      return 'text-red-700 bg-red-100';
    default:
      return 'text-gray-600 bg-gray-100';
  }
}

export function getPriorityColor(priority: string): string {
  switch (priority) {
    case 'CRITICAL':
      return 'text-red-700 bg-red-100 border-red-500';
    case 'HIGH':
      return 'text-orange-700 bg-orange-100 border-orange-500';
    case 'MEDIUM':
      return 'text-yellow-700 bg-yellow-100 border-yellow-500';
    case 'LOW':
      return 'text-blue-700 bg-blue-100 border-blue-500';
    default:
      return 'text-gray-700 bg-gray-100 border-gray-500';
  }
}

export function formatScore(score: number): string {
  return `${score.toFixed(1)}/100`;
}

export function formatPercentage(value: number): string {
  return `${value.toFixed(2)}%`;
}

export function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}
