/**
 * API service for Options Chain data
 * Bloomberg OMON equivalent
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface OptionData {
  bid: number | null;
  ask: number | null;
  last: number | null;
  volume: number;
  open_interest: number;
  iv: number | null;
  delta: number | null;
  gamma: number | null;
  theta: number | null;
  vega: number | null;
  in_the_money: boolean;
  unusual_volume: boolean;
}

export interface OptionStrike {
  strike: number;
  expiration: string;
  call: OptionData;
  put: OptionData;
}

export interface OptionsChain {
  symbol: string;
  current_price: number;
  price_change: number;
  price_change_pct: number;
  iv_rank: number | null;
  iv_percentile: number | null;
  hv_20: number | null;
  hv_30: number | null;
  expirations: string[];
  strikes: Record<string, OptionStrike[]>;
  max_pain: number | null;
  put_call_ratio_volume: number | null;
  put_call_ratio_oi: number | null;
  last_updated: string;
  data_source: string;
}

export interface OptionsSummary {
  symbol: string;
  current_price: number;
  price_change: number;
  price_change_pct: number;
  iv_rank: number | null;
  iv_percentile: number | null;
  hv_20: number | null;
  hv_30: number | null;
  max_pain: number | null;
  put_call_ratio_volume: number | null;
  put_call_ratio_oi: number | null;
  last_updated: string;
  data_source: string;
}

/**
 * Get complete options chain
 */
export async function getOptionsChain(
  symbol: string,
  expiration?: string
): Promise<OptionsChain> {
  const params = new URLSearchParams();
  if (expiration) {
    params.append('expiration', expiration);
  }

  const response = await fetch(
    `${API_BASE_URL}/api/options-chain/${symbol}?${params}`,
    {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Get available expiration dates
 */
export async function getExpirations(symbol: string): Promise<string[]> {
  const response = await fetch(
    `${API_BASE_URL}/api/options-chain/${symbol}/expirations`,
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

  const data = await response.json();
  return data.expirations;
}

/**
 * Get options summary metrics
 */
export async function getOptionsSummary(symbol: string): Promise<OptionsSummary> {
  const response = await fetch(
    `${API_BASE_URL}/api/options-chain/${symbol}/summary`,
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
 * Get unusual activity
 */
export async function getUnusualActivity(
  symbol: string,
  expiration?: string,
  minVolume: number = 100
): Promise<any> {
  const params = new URLSearchParams({
    min_volume: minVolume.toString(),
  });

  if (expiration) {
    params.append('expiration', expiration);
  }

  const response = await fetch(
    `${API_BASE_URL}/api/options-chain/${symbol}/unusual-activity?${params}`,
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
 * Create WebSocket connection for real-time options chain updates
 */
export function createOptionsChainWebSocket(
  symbol: string,
  onMessage: (data: any) => void,
  expiration?: string,
  onError?: (error: Event) => void
): WebSocket {
  const wsUrl = API_BASE_URL.replace('http', 'ws');
  const params = expiration ? `?expiration=${expiration}` : '';
  const ws = new WebSocket(`${wsUrl}/api/options-chain/ws/${symbol}${params}`);

  ws.onopen = () => {
    console.log(`Options chain WebSocket connected for ${symbol}`);
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  };

  ws.onerror = (error) => {
    console.error('Options chain WebSocket error:', error);
    if (onError) onError(error);
  };

  ws.onclose = () => {
    console.log('Options chain WebSocket disconnected');
  };

  return ws;
}
