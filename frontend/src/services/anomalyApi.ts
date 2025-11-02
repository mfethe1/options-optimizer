/**
 * API service for Real-Time Anomaly Detection
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export type DetectionType = 'volume' | 'price' | 'iv' | 'options_flow';

export interface AnomalyDetectionRequest {
  symbol: string;
  detection_types?: DetectionType[];
}

export interface Anomaly {
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  z_score?: number;
  multiplier?: number;
  current_value?: number;
  average_value?: number;
  trading_implication: string;
  detected_at: string;
}

export interface AnomalyDetectionResponse {
  symbol: string;
  anomalies: Anomaly[];
  count: number;
  timestamp: string;
}

/**
 * Detect anomalies for a symbol
 */
export async function detectAnomalies(
  request: AnomalyDetectionRequest
): Promise<AnomalyDetectionResponse> {
  const response = await fetch(`${API_BASE_URL}/api/anomalies/detect`, {
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
 * Scan multiple symbols for anomalies
 */
export async function scanAnomalies(
  symbols: string[],
  detectionTypes?: DetectionType[]
): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/api/anomalies/scan`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      symbols,
      detection_types: detectionTypes,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Get detection thresholds
 */
export async function getDetectionThresholds(): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/api/anomalies/thresholds`, {
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

/**
 * Create WebSocket connection for real-time anomaly alerts
 */
export function createAnomalyWebSocket(
  userId: string,
  onMessage: (data: any) => void,
  onError?: (error: Event) => void
): WebSocket {
  const wsUrl = API_BASE_URL.replace('http', 'ws');
  const ws = new WebSocket(`${wsUrl}/api/anomalies/ws/alerts/${userId}`);

  ws.onopen = () => {
    console.log('Anomaly WebSocket connected');
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
    console.error('Anomaly WebSocket error:', error);
    if (onError) onError(error);
  };

  ws.onclose = () => {
    console.log('Anomaly WebSocket disconnected');
  };

  return ws;
}

/**
 * Subscribe to anomaly alerts for specific symbols
 */
export function subscribeToSymbols(ws: WebSocket, symbols: string[]): void {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(
      JSON.stringify({
        action: 'subscribe',
        symbols,
      })
    );
  } else {
    console.warn('WebSocket not open, cannot subscribe');
  }
}
