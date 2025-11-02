/**
 * Smart Order Routing API Client
 *
 * TypeScript client for intelligent order execution.
 * Integrates with institutional data feeds and Schwab API.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ============================================================================
// Types
// ============================================================================

export interface SmartOrderRequest {
  account_id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  order_type?: 'market' | 'limit';
  limit_price?: number;
  strategy?: 'twap' | 'vwap' | 'iceberg' | 'immediate';
  execution_duration_minutes?: number;
  num_slices?: number;
  min_slice_size?: number;
  max_participation_rate?: number;
  display_size?: number;
}

export interface SmartOrderResponse {
  order_id: string;
  symbol: string;
  side: string;
  total_quantity: number;
  strategy: string;
  num_slices: number;
  status: string;
  message: string;
}

export interface OrderStatus {
  order_id: string;
  symbol: string;
  side: string;
  total_quantity: number;
  filled_quantity: number;
  fill_percentage: number;
  avg_fill_price: number | null;
  status: string;
  strategy: string;
  num_slices: number;
  slices_filled: number;
  created_at: string;
  slippage_bps: number | null;
}

export interface ExecutionStats {
  total_orders: number;
  avg_slippage_bps: number;
  median_slippage_bps: number;
  total_cost_saved_usd: number;
  avg_fill_rate: number;
}

export interface ExecutionReport {
  order_id: string;
  symbol: string;
  side: string;
  total_quantity: number;
  filled_quantity: number;
  avg_fill_price: number;
  arrival_price: number;
  vwap_price: number;
  slippage_vs_arrival_bps: number;
  slippage_vs_vwap_bps: number;
  execution_duration_seconds: number;
  num_slices: number;
  fill_rate: number;
  estimated_cost_saved_usd: number;
  timestamp: string;
}

export interface Strategy {
  name: string;
  display_name: string;
  description: string;
  best_for: string;
  parameters: string[];
  typical_slippage_bps: string;
}

export interface StrategiesResponse {
  strategies: Strategy[];
  comparison: {
    naive_execution: {
      typical_slippage_bps: string;
      use_case: string;
    };
    smart_routing_benefit: {
      slippage_reduction: string;
      monthly_return_improvement: string;
    };
  };
}

// ============================================================================
// API Functions
// ============================================================================

/**
 * Submit a smart order for intelligent execution
 */
export async function submitSmartOrder(request: SmartOrderRequest): Promise<SmartOrderResponse> {
  const response = await fetch(`${API_BASE_URL}/api/smart-routing/submit`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to submit smart order');
  }

  return response.json();
}

/**
 * Get status of a smart order
 */
export async function getOrderStatus(orderId: string): Promise<OrderStatus> {
  const response = await fetch(`${API_BASE_URL}/api/smart-routing/status/${orderId}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get order status');
  }

  return response.json();
}

/**
 * Cancel a smart order
 */
export async function cancelOrder(orderId: string): Promise<{ order_id: string; status: string; message: string }> {
  const response = await fetch(`${API_BASE_URL}/api/smart-routing/cancel/${orderId}`, {
    method: 'POST',
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to cancel order');
  }

  return response.json();
}

/**
 * Get execution statistics
 */
export async function getExecutionStats(): Promise<ExecutionStats> {
  const response = await fetch(`${API_BASE_URL}/api/smart-routing/stats`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get execution stats');
  }

  return response.json();
}

/**
 * Get execution reports
 */
export async function getExecutionReports(limit: number = 10): Promise<ExecutionReport[]> {
  const response = await fetch(`${API_BASE_URL}/api/smart-routing/reports?limit=${limit}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get execution reports');
  }

  return response.json();
}

/**
 * Get available strategies
 */
export async function getAvailableStrategies(): Promise<StrategiesResponse> {
  const response = await fetch(`${API_BASE_URL}/api/smart-routing/strategies`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get strategies');
  }

  return response.json();
}
