/**
 * API service for Deep Sentiment Analysis
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export type SentimentSource = 'twitter' | 'reddit' | 'news' | 'stocktwits';

export interface SentimentAnalysisRequest {
  symbol: string;
  sources?: SentimentSource[];
  lookback_hours?: number;
}

export interface SentimentAnalysisResponse {
  symbol: string;
  sentiment: {
    score: number;
    bias: string;
    confidence: number;
    mention_volume: number;
  };
  by_source: Record<string, any>;
  influencer_sentiment?: {
    score: number;
    tier_1_count: number;
    tier_2_count: number;
    bias_vs_retail: string;
  };
  controversy_score: number;
  sentiment_velocity: number;
  echo_chamber_detected: boolean;
  trading_implication: string;
  timestamp: string;
}

/**
 * Analyze sentiment for a symbol
 */
export async function analyzeSentiment(
  request: SentimentAnalysisRequest
): Promise<SentimentAnalysisResponse> {
  const response = await fetch(`${API_BASE_URL}/api/sentiment/analyze`, {
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
 * Compare sentiment across multiple symbols
 */
export async function compareSentiment(
  symbols: string[],
  sources?: SentimentSource[]
): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/api/sentiment/compare`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      symbols,
      sources,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Get trending sentiment stocks
 */
export async function getTrendingSentiment(
  timeframe: '1h' | '4h' | '24h' = '1h',
  limit = 20
): Promise<any> {
  const params = new URLSearchParams({
    timeframe,
    limit: limit.toString(),
  });

  const response = await fetch(
    `${API_BASE_URL}/api/sentiment/trending?${params}`,
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
 * Get influencer sentiment for a symbol
 */
export async function getInfluencerSentiment(
  symbol: string,
  limit = 10
): Promise<any> {
  const params = new URLSearchParams({
    limit: limit.toString(),
  });

  const response = await fetch(
    `${API_BASE_URL}/api/sentiment/influencers/${symbol}?${params}`,
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
 * Get sentiment analysis metrics and methodology
 */
export async function getSentimentMetrics(): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/api/sentiment/metrics`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}
