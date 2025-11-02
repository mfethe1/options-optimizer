/**
 * Market Data API Client
 *
 * TypeScript client for institutional-grade market data with <200ms latency.
 * Features:
 * - Real-time quotes aggregated from multiple providers
 * - Level 2 order book depth
 * - WebSocket streaming for sub-second updates
 * - Latency monitoring and provider status
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_BASE_URL = API_BASE_URL.replace('http', 'ws');

// ============================================================================
// Types
// ============================================================================

export interface Quote {
  symbol: string;
  best_bid: number;
  best_ask: number;
  best_bid_provider: string;
  best_ask_provider: string;
  bid_size: number;
  ask_size: number;
  last: number;
  mid_price: number;
  spread: number;
  spread_bps: number;
  timestamp: string;
  num_providers: number;
  avg_latency_ms: number;
}

export interface BatchQuotesResponse {
  quotes: Record<string, {
    best_bid: number;
    best_ask: number;
    mid_price: number;
    spread_bps: number;
    last: number;
    timestamp: string;
    num_providers: number;
    avg_latency_ms: number;
  }>;
  count: number;
  timestamp: string;
}

export interface OrderBookLevel {
  price: number;
  size: number;
}

export interface OrderBook {
  symbol: string;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  best_bid: number;
  best_ask: number;
  spread: number;
  spread_bps: number;
  imbalance: number;
  timestamp: string;
}

export interface LatencyStats {
  [provider: string]: {
    avg_ms: number;
    p50_ms: number;
    p95_ms: number;
    p99_ms: number;
    min_ms: number;
    max_ms: number;
  };
}

export interface ProviderStatus {
  providers: Record<string, boolean>;
  connected_count: number;
  total_count: number;
  timestamp: string;
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'critical' | 'unavailable';
  message: string;
  providers_connected: number;
  providers_total: number;
  avg_p95_latency_ms: number | null;
  target_latency_ms: number;
  timestamp: string;
}

export interface StreamQuote {
  type: 'quote';
  symbol: string;
  best_bid: number;
  best_ask: number;
  mid_price: number;
  spread_bps: number;
  last: number;
  timestamp: string;
  num_providers: number;
  latency_ms: number;
}

// ============================================================================
// REST API Functions
// ============================================================================

/**
 * Get real-time aggregated quote for a symbol
 */
export async function getRealTimeQuote(symbol: string): Promise<Quote> {
  const response = await fetch(`${API_BASE_URL}/api/market-data/quote/${symbol}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to fetch quote');
  }

  return response.json();
}

/**
 * Get real-time quotes for multiple symbols
 */
export async function getBatchQuotes(symbols: string[]): Promise<BatchQuotesResponse> {
  const symbolsParam = symbols.join(',');
  const response = await fetch(`${API_BASE_URL}/api/market-data/quotes/batch?symbols=${symbolsParam}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to fetch batch quotes');
  }

  return response.json();
}

/**
 * Get Level 2 order book for a symbol
 */
export async function getOrderBook(symbol: string): Promise<OrderBook> {
  const response = await fetch(`${API_BASE_URL}/api/market-data/order-book/${symbol}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to fetch order book');
  }

  return response.json();
}

/**
 * Get latency statistics for all data providers
 */
export async function getLatencyStats(): Promise<LatencyStats> {
  const response = await fetch(`${API_BASE_URL}/api/market-data/latency-stats`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to fetch latency stats');
  }

  return response.json();
}

/**
 * Get connection status for all data providers
 */
export async function getProviderStatus(): Promise<ProviderStatus> {
  const response = await fetch(`${API_BASE_URL}/api/market-data/provider-status`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to fetch provider status');
  }

  return response.json();
}

/**
 * Get health status of market data service
 */
export async function getMarketDataHealth(): Promise<HealthStatus> {
  const response = await fetch(`${API_BASE_URL}/api/market-data/health`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to fetch health status');
  }

  return response.json();
}

// ============================================================================
// WebSocket Streaming
// ============================================================================

export type QuoteCallback = (quote: StreamQuote) => void;
export type ErrorCallback = (error: Error) => void;

/**
 * WebSocket client for real-time quote streaming
 */
export class MarketDataStream {
  private ws: WebSocket | null = null;
  private symbol: string;
  private onQuoteCallback: QuoteCallback;
  private onErrorCallback: ErrorCallback;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  constructor(symbol: string, onQuote: QuoteCallback, onError?: ErrorCallback) {
    this.symbol = symbol;
    this.onQuoteCallback = onQuote;
    this.onErrorCallback = onError || ((error) => console.error('Market data error:', error));
  }

  /**
   * Connect to WebSocket stream
   */
  connect(): void {
    const url = `${WS_BASE_URL}/api/market-data/ws/stream/${this.symbol}`;

    try {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        console.log(`Market data stream connected for ${this.symbol}`);
        this.reconnectAttempts = 0;
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'quote') {
            this.onQuoteCallback(data as StreamQuote);
          } else if (data.type === 'pong') {
            // Heartbeat response
            console.log('Heartbeat received');
          }
        } catch (error) {
          this.onErrorCallback(new Error(`Failed to parse message: ${error}`));
        }
      };

      this.ws.onerror = (error) => {
        this.onErrorCallback(new Error('WebSocket error'));
      };

      this.ws.onclose = () => {
        console.log(`Market data stream closed for ${this.symbol}`);
        this.attemptReconnect();
      };

      // Send periodic pings to keep connection alive
      setInterval(() => {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send('ping');
        }
      }, 30000);

    } catch (error) {
      this.onErrorCallback(new Error(`Failed to connect: ${error}`));
    }
  }

  /**
   * Attempt to reconnect with exponential backoff
   */
  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.onErrorCallback(new Error('Max reconnection attempts reached'));
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`);

    setTimeout(() => {
      this.connect();
    }, delay);
  }

  /**
   * Disconnect from WebSocket stream
   */
  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

/**
 * Multi-symbol WebSocket client
 */
export class MultiSymbolStream {
  private ws: WebSocket | null = null;
  private onQuoteCallback: QuoteCallback;
  private onErrorCallback: ErrorCallback;
  private subscribedSymbols: Set<string> = new Set();

  constructor(onQuote: QuoteCallback, onError?: ErrorCallback) {
    this.onQuoteCallback = onQuote;
    this.onErrorCallback = onError || ((error) => console.error('Market data error:', error));
  }

  /**
   * Connect to multi-symbol stream
   */
  connect(): void {
    const url = `${WS_BASE_URL}/api/market-data/ws/stream-multi`;

    try {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        console.log('Multi-symbol market data stream connected');
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'quote') {
            this.onQuoteCallback(data as StreamQuote);
          } else if (data.type === 'subscribed') {
            console.log(`Subscribed to ${data.count} symbols:`, data.symbols);
          } else if (data.type === 'unsubscribed') {
            console.log(`Unsubscribed from symbols:`, data.symbols);
          } else if (data.type === 'pong') {
            console.log('Heartbeat received');
          }
        } catch (error) {
          this.onErrorCallback(new Error(`Failed to parse message: ${error}`));
        }
      };

      this.ws.onerror = (error) => {
        this.onErrorCallback(new Error('WebSocket error'));
      };

      this.ws.onclose = () => {
        console.log('Multi-symbol market data stream closed');
      };

    } catch (error) {
      this.onErrorCallback(new Error(`Failed to connect: ${error}`));
    }
  }

  /**
   * Subscribe to symbols
   */
  subscribe(symbols: string[]): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected');
    }

    this.ws.send(JSON.stringify({
      action: 'subscribe',
      symbols: symbols
    }));

    symbols.forEach(s => this.subscribedSymbols.add(s));
  }

  /**
   * Unsubscribe from symbols
   */
  unsubscribe(symbols: string[]): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected');
    }

    this.ws.send(JSON.stringify({
      action: 'unsubscribe',
      symbols: symbols
    }));

    symbols.forEach(s => this.subscribedSymbols.delete(s));
  }

  /**
   * Disconnect from stream
   */
  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.subscribedSymbols.clear();
  }
}
