/**
 * News API Service
 *
 * Bloomberg NEWS equivalent - real-time financial news feeds
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ============================================================================
// Type Definitions
// ============================================================================

export interface NewsArticle {
  id: string;
  title: string;
  summary: string | null;
  url: string;
  source: string;
  author: string | null;
  published_at: string;
  symbols: string[];
  categories: string[];
  sentiment: string | null;
  sentiment_score: number | null;
  image_url: string | null;
  provider: string;
}

export interface NewsFeedResponse {
  count: number;
  articles: NewsArticle[];
  filters?: {
    symbols: string[] | null;
    categories: string[] | null;
    hours: number;
  };
}

export interface NewsSearchResponse {
  query: string;
  count: number;
  articles: NewsArticle[];
}

export interface NewsCategory {
  value: string;
  name: string;
}

export interface ProviderStatus {
  name: string;
  available: boolean;
  rate_limit: number;
  last_error: string | null;
}

export interface NewsProviderStatusResponse {
  total_providers: number;
  available_providers: string[];
  provider_details: ProviderStatus[];
}

// ============================================================================
// API Functions
// ============================================================================

/**
 * Get latest financial news
 */
export async function getNews(
  symbols?: string[],
  categories?: string[],
  limit: number = 50,
  hours: number = 24
): Promise<NewsFeedResponse> {
  const params = new URLSearchParams();

  if (symbols && symbols.length > 0) {
    params.append('symbols', symbols.join(','));
  }

  if (categories && categories.length > 0) {
    params.append('categories', categories.join(','));
  }

  params.append('limit', limit.toString());
  params.append('hours', hours.toString());

  const response = await fetch(`${API_BASE_URL}/api/news?${params}`);

  if (!response.ok) {
    throw new Error(`Failed to fetch news: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Search financial news
 */
export async function searchNews(
  query: string,
  symbols?: string[],
  limit: number = 50
): Promise<NewsSearchResponse> {
  const params = new URLSearchParams();
  params.append('q', query);

  if (symbols && symbols.length > 0) {
    params.append('symbols', symbols.join(','));
  }

  params.append('limit', limit.toString());

  const response = await fetch(`${API_BASE_URL}/api/news/search?${params}`);

  if (!response.ok) {
    throw new Error(`Failed to search news: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get news for a specific symbol
 */
export async function getSymbolNews(
  symbol: string,
  limit: number = 50,
  hours: number = 24
): Promise<NewsFeedResponse> {
  const params = new URLSearchParams();
  params.append('limit', limit.toString());
  params.append('hours', hours.toString());

  const response = await fetch(
    `${API_BASE_URL}/api/news/symbols/${symbol}?${params}`
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch news for ${symbol}: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get available news categories
 */
export async function getCategories(): Promise<{ categories: NewsCategory[] }> {
  const response = await fetch(`${API_BASE_URL}/api/news/categories`);

  if (!response.ok) {
    throw new Error(`Failed to fetch categories: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get news provider status
 */
export async function getProviderStatus(): Promise<NewsProviderStatusResponse> {
  const response = await fetch(`${API_BASE_URL}/api/news/providers/status`);

  if (!response.ok) {
    throw new Error(`Failed to fetch provider status: ${response.statusText}`);
  }

  return response.json();
}

/**
 * WebSocket connection for real-time news streaming
 */
export class NewsStreamConnection {
  private ws: WebSocket | null = null;
  private reconnectTimeout: number | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  constructor(
    private onMessage: (articles: NewsArticle[]) => void,
    private onError?: (error: Event) => void,
    private onConnected?: () => void
  ) {}

  connect(config?: {
    symbols?: string[];
    categories?: string[];
    interval_seconds?: number;
  }): void {
    const wsUrl = API_BASE_URL.replace('http', 'ws');
    this.ws = new WebSocket(`${wsUrl}/api/news/ws/stream`);

    this.ws.onopen = () => {
      console.log('News stream connected');
      this.reconnectAttempts = 0;

      if (config) {
        this.sendConfig(config);
      }

      if (this.onConnected) {
        this.onConnected();
      }
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'news') {
          this.onMessage(data.articles);
        } else if (data.type === 'error') {
          console.error('News stream error:', data.message);
        } else if (data.type === 'connected') {
          console.log('News stream initialized:', data.message);
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (this.onError) {
        this.onError(error);
      }
    };

    this.ws.onclose = () => {
      console.log('News stream disconnected');
      this.attemptReconnect(config);
    };
  }

  private attemptReconnect(config?: any): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);

      console.log(
        `Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`
      );

      this.reconnectTimeout = window.setTimeout(() => {
        this.connect(config);
      }, delay);
    } else {
      console.error('Max reconnection attempts reached');
    }
  }

  sendConfig(config: {
    symbols?: string[];
    categories?: string[];
    interval_seconds?: number;
  }): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(config));
    }
  }

  disconnect(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.reconnectAttempts = 0;
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}
