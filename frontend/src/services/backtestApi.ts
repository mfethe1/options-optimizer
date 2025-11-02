/**
 * Backtesting API Service
 *
 * Historical strategy testing with institutional-grade metrics.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface BacktestConfig {
  symbol: string;
  strategy_type: string;
  start_date: string;
  end_date: string;
  entry_dte_min?: number;
  entry_dte_max?: number;
  profit_target_pct?: number;
  stop_loss_pct?: number;
  exit_dte?: number;
  capital_per_trade?: number;
  max_positions?: number;
  commission_per_contract?: number;
  slippage_pct?: number;
  spread_width?: number;
  iv_rank_min?: number;
}

export interface BacktestMetrics {
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  total_pnl: number;
  total_pnl_pct: number;
  avg_win: number;
  avg_loss: number;
  profit_factor: number;
  max_drawdown: number;
  max_drawdown_pct: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  avg_days_held: number;
  expectancy: number;
  kelly_criterion: number;
  calmar_ratio: number;
  best_trade_pnl: number;
  worst_trade_pnl: number;
  consecutive_wins: number;
  consecutive_losses: number;
}

export interface BacktestTrade {
  entry_date: string;
  entry_price: number;
  entry_cost: number;
  exit_date: string | null;
  exit_price: number | null;
  pnl: number | null;
  pnl_pct: number | null;
  days_held: number | null;
  exit_reason: string | null;
  status: string;
  legs: Array<{
    option_type: string;
    action: string;
    strike: number;
    quantity: number;
    entry_price: number;
  }>;
}

export interface BacktestResult {
  symbol: string;
  strategy_type: string;
  start_date: string;
  end_date: string;
  config: {
    entry_dte_range: string;
    profit_target_pct: number;
    stop_loss_pct: number;
    exit_dte: number;
    capital_per_trade: number;
    max_positions: number;
  };
  metrics: BacktestMetrics;
  trades: BacktestTrade[];
}

export interface StrategyInfo {
  name: string;
  description: string;
  risk: string;
  reward: string;
  best_for: string;
}

export interface AvailableStrategies {
  strategies: Record<string, StrategyInfo>;
  count: number;
}

export interface ComparisonResult {
  symbol: string;
  start_date: string;
  end_date: string;
  strategies: Array<{
    strategy_type: string;
    total_pnl: number;
    win_rate: number;
    profit_factor: number;
    sharpe_ratio: number;
    max_drawdown: number;
    total_trades: number;
  }>;
  best_strategy: string | null;
  detailed_results: BacktestResult[];
}

/**
 * Run a backtest for a strategy
 */
export async function runBacktest(config: BacktestConfig): Promise<BacktestResult> {
  const response = await fetch(`${API_BASE_URL}/api/backtest/run`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(config),
  });

  if (!response.ok) {
    throw new Error(`Failed to run backtest: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get available strategy types
 */
export async function getAvailableStrategies(): Promise<AvailableStrategies> {
  const response = await fetch(`${API_BASE_URL}/api/backtest/strategies`);

  if (!response.ok) {
    throw new Error(`Failed to fetch strategies: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Compare multiple strategies
 */
export async function compareStrategies(
  symbol: string,
  strategies: string[],
  startDate: string,
  endDate: string,
  baseConfig: Partial<BacktestConfig> = {}
): Promise<ComparisonResult> {
  const response = await fetch(`${API_BASE_URL}/api/backtest/compare`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      symbol,
      strategies,
      start_date: startDate,
      end_date: endDate,
      base_config: baseConfig,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to compare strategies: ${response.statusText}`);
  }

  return response.json();
}
