/**
 * API service for AI-Powered Paper Trading
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface TradeRecommendation {
  symbol: string;
  action: 'buy' | 'sell' | 'open' | 'close';
  quantity: number;
  price?: number;
  trade_type: 'stock' | 'option';
  option_details?: Record<string, any>;
  confidence: number;
  reasoning?: string;
}

export interface ExecuteTradeRequest {
  recommendation: TradeRecommendation;
  user_id: string;
  auto_approve?: boolean;
  timeout_seconds?: number;
}

export interface ExecuteTradeResponse {
  status: 'executed' | 'rejected' | 'pending';
  trade?: any;
  consensus?: {
    result: string;
    confidence: number;
    votes: Record<string, number>;
  };
  risk_check?: {
    approved: boolean;
    reason?: string;
    position_size_pct?: number;
    cash_available?: number;
  };
  reason?: string;
  portfolio?: any;
  timestamp: string;
}

export interface PortfolioResponse {
  cash: number;
  positions_count: number;
  positions: Array<{
    symbol: string;
    quantity: number;
    avg_price: number;
    current_price: number;
    pnl: number;
  }>;
  performance: {
    total_pnl: number;
    realized_pnl: number;
    unrealized_pnl: number;
    total_return_pct: number;
    current_value: number;
    win_rate: number;
    total_trades: number;
    winning_trades: number;
  };
  timestamp: string;
}

/**
 * Execute a trade recommendation
 */
export async function executeTrade(
  request: ExecuteTradeRequest
): Promise<ExecuteTradeResponse> {
  const response = await fetch(`${API_BASE_URL}/api/paper-trading/execute`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Get portfolio for a user
 */
export async function getPortfolio(userId: string): Promise<PortfolioResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/paper-trading/portfolio/${userId}`,
    {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Get trade history
 */
export async function getTradeHistory(
  userId: string,
  limit = 50
): Promise<any> {
  const params = new URLSearchParams({
    limit: limit.toString(),
  });

  const response = await fetch(
    `${API_BASE_URL}/api/paper-trading/history/${userId}?${params}`,
    {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Get risk limits
 */
export async function getRiskLimits(userId: string): Promise<any> {
  const response = await fetch(
    `${API_BASE_URL}/api/paper-trading/risk-limits/${userId}`,
    {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Update risk limits
 */
export async function updateRiskLimits(
  userId: string,
  riskLimits: Record<string, number>
): Promise<any> {
  const response = await fetch(
    `${API_BASE_URL}/api/paper-trading/risk-limits/${userId}`,
    {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(riskLimits),
    }
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Get pending approvals
 */
export async function getPendingApprovals(userId: string): Promise<any> {
  const response = await fetch(
    `${API_BASE_URL}/api/paper-trading/approvals/${userId}`,
    {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Approve a trade
 */
export async function approveTrade(
  userId: string,
  tradeId: string
): Promise<any> {
  const response = await fetch(
    `${API_BASE_URL}/api/paper-trading/approvals/${userId}/${tradeId}/approve`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Reject a trade
 */
export async function rejectTrade(
  userId: string,
  tradeId: string
): Promise<any> {
  const response = await fetch(
    `${API_BASE_URL}/api/paper-trading/approvals/${userId}/${tradeId}/reject`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Reset portfolio
 */
export async function resetPortfolio(userId: string): Promise<any> {
  const response = await fetch(
    `${API_BASE_URL}/api/paper-trading/portfolio/${userId}/reset`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}
