/**
 * Execution Quality API Service
 *
 * Track and analyze trade execution quality to minimize slippage.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface ExecutionMetrics {
  total_executions: number;
  total_volume: number;
  avg_slippage_bps: number;
  median_slippage_bps: number;
  fill_rate: number;
  avg_time_to_fill_ms: number;
  partial_fill_rate: number;
  price_improvement_rate: number;
  avg_price_improvement_bps: number;
  adverse_selection_rate: number;
  avg_adverse_selection_bps: number;
  slippage_25th_percentile: number;
  slippage_75th_percentile: number;
  slippage_95th_percentile: number;
  worst_slippage_bps: number;
  best_slippage_bps: number;
  total_slippage_cost: number;
  estimated_annual_drag: number;
}

export interface BrokerMetrics {
  avg_slippage_bps: number;
  total_executions: number;
  fill_rate: number;
}

export interface TimeOfDayMetrics {
  avg_slippage_bps: number;
  total_executions: number;
}

export interface ExecutionAnalysis {
  overall_metrics: ExecutionMetrics;
  by_broker: Record<string, BrokerMetrics>;
  by_time_of_day: Record<string, TimeOfDayMetrics>;
  by_order_type: Record<string, TimeOfDayMetrics>;
  by_symbol: Record<string, TimeOfDayMetrics>;
  recommendations: string[];
}

/**
 * Record a new order
 */
export async function recordOrder(
  orderId: string,
  symbol: string,
  orderType: string,
  orderSide: string,
  quantity: number,
  expectedPrice: number,
  broker: string,
  limitPrice?: number,
  bidAskSpread?: number
): Promise<{ status: string; order_id: string; order_time: string }> {
  const response = await fetch(`${API_BASE_URL}/api/execution/record-order`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      order_id: orderId,
      symbol,
      order_type: orderType,
      order_side: orderSide,
      quantity,
      expected_price: expectedPrice,
      broker,
      limit_price: limitPrice,
      bid_ask_spread: bidAskSpread,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to record order: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Record order fill
 */
export async function recordFill(
  orderId: string,
  fillPrice: number,
  filledQuantity: number,
  venue?: string,
  partial: boolean = false
): Promise<{
  status: string;
  order_id: string;
  fill_price: number;
  filled_quantity: number;
  slippage_bps: number;
  slippage_dollars: number;
  time_to_fill_ms: number;
  price_improvement: number | null;
}> {
  const response = await fetch(`${API_BASE_URL}/api/execution/record-fill`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      order_id: orderId,
      fill_price: fillPrice,
      filled_quantity: filledQuantity,
      venue,
      partial,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to record fill: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get execution quality analysis
 */
export async function getExecutionAnalysis(
  startDate?: string,
  endDate?: string,
  symbol?: string,
  broker?: string
): Promise<ExecutionAnalysis> {
  const params = new URLSearchParams();
  if (startDate) params.append('start_date', startDate);
  if (endDate) params.append('end_date', endDate);
  if (symbol) params.append('symbol', symbol);
  if (broker) params.append('broker', broker);

  const response = await fetch(
    `${API_BASE_URL}/api/execution/analysis?${params.toString()}`
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch execution analysis: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Generate mock execution data for demo
 */
export async function generateMockData(numOrders: number = 50): Promise<{ status: string; num_orders: number; message: string }> {
  const response = await fetch(`${API_BASE_URL}/api/execution/generate-mock-data`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ num_orders: numOrders }),
  });

  if (!response.ok) {
    throw new Error(`Failed to generate mock data: ${response.statusText}`);
  }

  return response.json();
}
