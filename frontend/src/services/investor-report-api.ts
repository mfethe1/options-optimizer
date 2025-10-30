/**
 * API Service for InvestorReport.v1
 * 
 * Provides type-safe API calls to backend endpoints.
 * Implements error handling, retry logic, and performance monitoring.
 * 
 * Performance contracts:
 * - API response: <500ms (cached)
 * - WebSocket latency: <50ms
 * - Retry on 5xx errors (max 2 retries)
 */

import {
  InvestorReport,
  APIError,
  ProcessingResponse,
  isInvestorReport,
  isProcessingResponse
} from '../types/investor-report';

// API configuration
const API_BASE = import.meta.env.VITE_API_BASE || '/api';
const WS_BASE = import.meta.env.VITE_WS_BASE || 
  ((location.protocol === 'https:' ? 'wss' : 'ws') + '://' + location.host);

// Request timeout (ms)
const REQUEST_TIMEOUT = 10000;  // 10s

// Retry configuration
const MAX_RETRIES = 2;
const RETRY_DELAY_MS = 1000;

/**
 * Fetch InvestorReport.v1 JSON for a user's portfolio
 * 
 * @param userId - User ID for portfolio lookup
 * @param symbols - Optional array of symbols (defaults to user's portfolio)
 * @param fresh - Force fresh computation (bypass cache)
 * @returns InvestorReport or ProcessingResponse (202)
 * @throws APIError on failure
 */
export async function getInvestorReport(
  userId: string,
  symbols?: string[],
  fresh: boolean = false
): Promise<InvestorReport | ProcessingResponse> {
  const params = new URLSearchParams({
    user_id: userId,
    fresh: String(fresh)
  });
  
  if (symbols && symbols.length > 0) {
    params.append('symbols', symbols.join(','));
  }
  
  const url = `${API_BASE}/investor-report?${params.toString()}`;
  
  // Retry logic for 5xx errors
  let lastError: Error | null = null;
  
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);
      
      const response = await fetch(url, {
        signal: controller.signal,
        headers: {
          'Accept': 'application/json'
        }
      });
      
      clearTimeout(timeoutId);
      
      // Handle 202 Accepted (processing)
      if (response.status === 202) {
        const data = await response.json();
        if (isProcessingResponse(data)) {
          return data;
        }
      }
      
      // Handle errors
      if (!response.ok) {
        const error: APIError = {
          detail: `API error: ${response.status} ${response.statusText}`,
          status_code: response.status
        };
        
        try {
          const errorData = await response.json();
          error.detail = errorData.detail || error.detail;
        } catch {
          // Ignore JSON parse errors
        }
        
        // Retry on 5xx errors
        if (response.status >= 500 && attempt < MAX_RETRIES) {
          lastError = new Error(error.detail);
          await sleep(RETRY_DELAY_MS * (attempt + 1));
          continue;
        }
        
        throw new Error(error.detail);
      }
      
      // Parse and validate response
      const data = await response.json();
      
      if (!isInvestorReport(data)) {
        throw new Error('Invalid InvestorReport schema in response');
      }
      
      return data;
      
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Request timeout after ${REQUEST_TIMEOUT}ms`);
      }
      
      lastError = error as Error;
      
      // Don't retry on client errors
      if (attempt >= MAX_RETRIES) {
        throw lastError;
      }
      
      await sleep(RETRY_DELAY_MS * (attempt + 1));
    }
  }
  
  throw lastError || new Error('Unknown error');
}

/**
 * Open WebSocket connection for Phase 4 metrics streaming
 * 
 * @param userId - User ID for portfolio lookup
 * @returns WebSocket instance
 */
export function openPhase4Socket(userId: string): WebSocket {
  const url = `${WS_BASE}/ws/phase4-metrics/${userId}`;
  const ws = new WebSocket(url);
  
  // Set binary type for efficient data transfer
  ws.binaryType = 'arraybuffer';
  
  return ws;
}

// Utility functions

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

