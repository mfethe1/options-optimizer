/**
 * Charles Schwab API Service
 *
 * Live trading and market data integration with Charles Schwab.
 * Enables real order execution for validated strategies.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// ============================================================================
// Type Definitions
// ============================================================================

export interface SchwabAccount {
  account_id: string;
  account_number: string;
  account_type: string;
  current_balances: {
    cash_balance: number;
    market_value: number;
    buying_power: number;
    total_value: number;
  };
  positions: SchwabPosition[];
}

export interface SchwabPosition {
  symbol: string;
  instrument_type: string;
  quantity: number;
  average_price: number;
  current_price: number;
  market_value: number;
  pnl: number;
  pnl_pct: number;
  day_pnl: number;
}

export interface SchwabQuote {
  symbol: string;
  bid: number;
  ask: number;
  last: number;
  volume: number;
  bid_size: number;
  ask_size: number;
  high: number;
  low: number;
  open: number;
  close_price: number;
  quote_time: string;
  trade_time: string;
}

export interface SchwabOptionContract {
  symbol: string;
  description: string;
  bid: number;
  ask: number;
  last: number;
  volume: number;
  open_interest: number;
  strike: number;
  expiration_date: string;
  days_to_expiration: number;
  option_type: 'CALL' | 'PUT';
  in_the_money: boolean;
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  rho: number;
  implied_volatility: number;
}

export interface SchwabOptionsChain {
  symbol: string;
  underlying_price: number;
  expirations: string[];
  calls: Record<string, SchwabOptionContract[]>;  // Keyed by expiration
  puts: Record<string, SchwabOptionContract[]>;   // Keyed by expiration
}

export interface SchwabOrderRequest {
  account_id: string;
  symbol: string;
  quantity: number;
  order_type: 'MARKET' | 'LIMIT' | 'STOP' | 'STOP_LIMIT';
  order_action: 'BUY' | 'SELL' | 'BUY_TO_OPEN' | 'SELL_TO_OPEN' | 'BUY_TO_CLOSE' | 'SELL_TO_CLOSE';
  duration: 'DAY' | 'GTC' | 'FILL_OR_KILL';
  price?: number;
  stop_price?: number;
}

export interface SchwabOrderResponse {
  order_id: string;
  status: string;
  message: string;
}

export interface SchwabOrderStatus {
  order_id: string;
  status: string;
  filled_quantity: number;
  remaining_quantity: number;
  average_fill_price: number;
  order_time: string;
  fills: Array<{
    quantity: number;
    price: number;
    timestamp: string;
  }>;
}

// ============================================================================
// Authentication
// ============================================================================

/**
 * Get Schwab OAuth authorization URL
 * User should be redirected to this URL to authorize the application
 */
export async function getSchwabAuthUrl(): Promise<{ auth_url: string }> {
  const response = await fetch(`${API_BASE_URL}/api/schwab/auth/url`);

  if (!response.ok) {
    throw new Error(`Failed to get auth URL: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Exchange authorization code for access token
 * Call this after user is redirected back from Schwab OAuth
 */
export async function exchangeAuthCode(authorizationCode: string): Promise<{ success: boolean; message: string }> {
  const response = await fetch(`${API_BASE_URL}/api/schwab/auth/token`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ authorization_code: authorizationCode }),
  });

  if (!response.ok) {
    throw new Error(`Failed to exchange auth code: ${response.statusText}`);
  }

  return response.json();
}

// ============================================================================
// Account Data
// ============================================================================

/**
 * Get all Schwab accounts with balances and positions
 */
export async function getSchwabAccounts(): Promise<SchwabAccount[]> {
  const response = await fetch(`${API_BASE_URL}/api/schwab/accounts`);

  if (!response.ok) {
    throw new Error(`Failed to fetch accounts: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get positions for specific account
 */
export async function getSchwabPositions(accountId: string): Promise<SchwabPosition[]> {
  const response = await fetch(`${API_BASE_URL}/api/schwab/accounts/${accountId}/positions`);

  if (!response.ok) {
    throw new Error(`Failed to fetch positions: ${response.statusText}`);
  }

  return response.json();
}

// ============================================================================
// Market Data
// ============================================================================

/**
 * Get real-time quote for a symbol
 */
export async function getSchwabQuote(symbol: string): Promise<SchwabQuote> {
  const response = await fetch(`${API_BASE_URL}/api/schwab/quote/${symbol}`);

  if (!response.ok) {
    throw new Error(`Failed to fetch quote: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get complete options chain for a symbol
 */
export async function getSchwabOptionsChain(
  symbol: string,
  strikeCount?: number,
  includeQuotes?: boolean,
  strategy?: string,
  range?: string
): Promise<SchwabOptionsChain> {
  const params = new URLSearchParams();
  if (strikeCount !== undefined) params.append('strike_count', strikeCount.toString());
  if (includeQuotes !== undefined) params.append('include_quotes', includeQuotes.toString());
  if (strategy) params.append('strategy', strategy);
  if (range) params.append('range', range);

  const url = `${API_BASE_URL}/api/schwab/options/${symbol}${params.toString() ? '?' + params.toString() : ''}`;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Failed to fetch options chain: ${response.statusText}`);
  }

  return response.json();
}

// ============================================================================
// Order Management
// ============================================================================

/**
 * Place an order
 * ⚠️ WARNING: This places a REAL order with REAL money!
 */
export async function placeSchwabOrder(order: SchwabOrderRequest): Promise<SchwabOrderResponse> {
  const response = await fetch(`${API_BASE_URL}/api/schwab/accounts/${order.account_id}/orders`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      symbol: order.symbol,
      quantity: order.quantity,
      order_type: order.order_type,
      order_action: order.order_action,
      duration: order.duration,
      price: order.price,
      stop_price: order.stop_price,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(`Failed to place order: ${errorData.detail || response.statusText}`);
  }

  return response.json();
}

/**
 * Cancel an order
 */
export async function cancelSchwabOrder(accountId: string, orderId: string): Promise<{ success: boolean; message: string }> {
  const response = await fetch(`${API_BASE_URL}/api/schwab/accounts/${accountId}/orders/${orderId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error(`Failed to cancel order: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get order status
 */
export async function getSchwabOrderStatus(accountId: string, orderId: string): Promise<SchwabOrderStatus> {
  const response = await fetch(`${API_BASE_URL}/api/schwab/accounts/${accountId}/orders/${orderId}`);

  if (!response.ok) {
    throw new Error(`Failed to fetch order status: ${response.statusText}`);
  }

  return response.json();
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Format currency value
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
export function formatPercent(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
}

/**
 * Get color class for P&L
 */
export function getPnLColor(value: number): string {
  if (value > 0) return 'text-green-600';
  if (value < 0) return 'text-red-600';
  return 'text-gray-600';
}
