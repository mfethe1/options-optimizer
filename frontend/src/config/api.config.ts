/**
 * Centralized API Configuration
 *
 * This module provides a single source of truth for all API and WebSocket URLs.
 * It automatically detects the environment and uses the appropriate base URLs.
 *
 * In development: Uses Vite proxy (/api -> http://localhost:9001)
 * In production: Uses environment variable or defaults
 */

// Get environment variables (Vite automatically loads from .env files)
const isDevelopment = import.meta.env.DEV;
const isProduction = import.meta.env.PROD;

/**
 * API Base URL
 * In development, we use relative paths that Vite will proxy to the backend
 * In production, use the full URL from environment variable
 */
export const API_BASE_URL = isDevelopment
  ? '' // Use relative paths in dev (Vite proxy handles this)
  : import.meta.env.VITE_API_BASE_URL || 'http://localhost:9001';

/**
 * WebSocket Base URL
 * Automatically uses ws:// or wss:// based on current protocol
 */
export const WS_BASE_URL = (() => {
  if (isDevelopment) {
    // In development, use current host (Vite dev server) with /ws prefix
    // Vite proxy will forward to backend WebSocket
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    return `${protocol}://${window.location.host}`;
  }

  // In production, use environment variable or construct from current host
  if (import.meta.env.VITE_WS_BASE_URL) {
    return import.meta.env.VITE_WS_BASE_URL;
  }

  // Fallback: use current host with appropriate protocol
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${protocol}://${window.location.host}`;
})();

/**
 * API Configuration
 */
export const apiConfig = {
  // Base URLs
  baseURL: API_BASE_URL,
  wsBaseURL: WS_BASE_URL,

  // Timeouts
  timeout: 30000, // 30 seconds for regular requests
  longTimeout: 600000, // 10 minutes for long-running operations

  // Retry configuration
  retry: {
    attempts: 3,
    delay: 1000, // 1 second initial delay
    backoff: 2, // Exponential backoff multiplier
  },

  // WebSocket configuration
  websocket: {
    reconnectAttempts: 10,
    reconnectDelay: 2000, // 2 seconds
    heartbeatInterval: 30000, // 30 seconds
  },

  // Logging
  logging: import.meta.env.VITE_API_LOGGING === 'true' || isDevelopment,
};

/**
 * Helper function to build full API URL
 * Handles both absolute and relative paths correctly
 */
export function buildApiUrl(path: string): string {
  // Remove leading slash if present (we'll add it)
  const cleanPath = path.startsWith('/') ? path.slice(1) : path;

  if (isDevelopment) {
    // In development, use /api prefix which Vite will proxy
    return `/api/${cleanPath}`;
  }

  // In production, use full URL
  return `${API_BASE_URL}/${cleanPath}`;
}

/**
 * Helper function to build WebSocket URL
 */
export function buildWsUrl(path: string): string {
  // Remove leading slash if present
  const cleanPath = path.startsWith('/') ? path.slice(1) : path;

  if (isDevelopment) {
    // In development, use /ws prefix for Vite proxy
    // If path already starts with 'ws/', keep it; otherwise add it
    const wsPath = cleanPath.startsWith('ws/') ? cleanPath : `ws/${cleanPath}`;
    return `${WS_BASE_URL}/${wsPath}`;
  }

  // In production, use full WebSocket URL
  return `${WS_BASE_URL}/${cleanPath}`;
}

/**
 * Log API configuration on startup (development only)
 */
if (apiConfig.logging) {
  console.log('[API Config] Initialized:', {
    environment: isDevelopment ? 'development' : 'production',
    apiBaseURL: API_BASE_URL,
    wsBaseURL: WS_BASE_URL,
    sampleApiUrl: buildApiUrl('health'),
    sampleWsUrl: buildWsUrl('ws/agent-stream/test'),
  });
}

export default apiConfig;
