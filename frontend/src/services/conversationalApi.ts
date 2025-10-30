/**
 * API service for Conversational Trading features
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface ConversationMessage {
  message: string;
  user_id: string;
  session_id?: string;
  context?: Record<string, any>;
}

export interface ConversationResponse {
  response: string;
  intent: string;
  confidence: number;
  actions?: Array<{
    type: string;
    parameters: Record<string, any>;
  }>;
  data?: Record<string, any>;
  session_id: string;
  turn_number: number;
  timestamp: string;
}

export interface ExplanationRequest {
  topic: string;
  complexity: 'beginner' | 'medium' | 'advanced';
  context?: Record<string, any>;
}

export interface ExplanationResponse {
  topic: string;
  simple_explanation: string;
  detailed_explanation: string;
  example?: string;
  misconceptions?: string[];
  related_topics?: string[];
}

/**
 * Send a conversational message to the trading agent
 */
export async function sendConversationMessage(
  message: ConversationMessage
): Promise<ConversationResponse> {
  const response = await fetch(`${API_BASE_URL}/api/conversation/message`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(message),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Get educational explanation of a topic
 */
export async function getExplanation(
  request: ExplanationRequest
): Promise<ExplanationResponse> {
  const response = await fetch(`${API_BASE_URL}/api/conversation/explain`, {
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
 * Get conversation history for a user
 */
export async function getConversationHistory(
  userId: string,
  sessionId?: string,
  limit = 50
): Promise<any> {
  const params = new URLSearchParams({
    ...(sessionId && { session_id: sessionId }),
    limit: limit.toString(),
  });

  const response = await fetch(
    `${API_BASE_URL}/api/conversation/history/${userId}?${params}`,
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
 * Clear conversation history
 */
export async function clearConversationHistory(
  userId: string,
  sessionId?: string
): Promise<void> {
  const params = new URLSearchParams({
    ...(sessionId && { session_id: sessionId }),
  });

  const response = await fetch(
    `${API_BASE_URL}/api/conversation/history/${userId}?${params}`,
    {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
}

/**
 * Get supported intents
 */
export async function getSupportedIntents(): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/api/conversation/intents`, {
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
