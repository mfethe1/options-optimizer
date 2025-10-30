/**
 * useAgentStream Hook - Real-time LLM Agent Event Streaming
 * 
 * Connects to WebSocket endpoint for streaming agent thoughts, tool calls, and progress.
 * Provides institutional-grade transparency into long-running analysis (5-10 minutes).
 * 
 * Features:
 * - Auto-reconnect on disconnect
 * - Event buffering and ordering
 * - Progress state management
 * - Conversation history
 * - Performance optimized for real-time streaming
 */

import { useEffect, useRef, useReducer, useCallback } from 'react';

// Event types from backend
export enum AgentEventType {
  STARTED = 'started',
  THINKING = 'thinking',
  TOOL_CALL = 'tool_call',
  TOOL_RESULT = 'tool_result',
  PROGRESS = 'progress',
  ERROR = 'error',
  COMPLETED = 'completed',
  HEARTBEAT = 'heartbeat',
}

// Agent event structure
export interface AgentEvent {
  timestamp: string; // ISO 8601
  agent_id: string;
  event_type: AgentEventType;
  content: string;
  metadata?: Record<string, any>;
  error_flag?: boolean;
}

// Progress tracking
export interface AgentProgress {
  task_id: string;
  agent_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress_pct: number; // 0.0 to 100.0
  time_elapsed_sec: number;
  estimated_time_remaining_sec?: number;
  current_step: string;
  total_steps: number;
  completed_steps: number;
}

// WebSocket message types
interface AgentEventMessage {
  type: 'agent_event';
  data: AgentEvent;
}

interface HeartbeatMessage {
  type: 'heartbeat';
  timestamp: string;
}

type WebSocketMessage = AgentEventMessage | HeartbeatMessage;

// Hook state
interface AgentStreamState {
  events: AgentEvent[];
  progress: AgentProgress | null;
  isConnected: boolean;
  error: string | null;
  lastHeartbeat: Date | null;
}

// State actions
type AgentStreamAction =
  | { type: 'ADD_EVENT'; payload: AgentEvent }
  | { type: 'UPDATE_PROGRESS'; payload: AgentProgress }
  | { type: 'SET_CONNECTED'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_HEARTBEAT'; payload: Date }
  | { type: 'CLEAR_EVENTS' };

// Reducer
function agentStreamReducer(state: AgentStreamState, action: AgentStreamAction): AgentStreamState {
  switch (action.type) {
    case 'ADD_EVENT':
      return {
        ...state,
        events: [...state.events, action.payload],
        error: null,
      };
    
    case 'UPDATE_PROGRESS':
      return {
        ...state,
        progress: action.payload,
      };
    
    case 'SET_CONNECTED':
      return {
        ...state,
        isConnected: action.payload,
        error: action.payload ? null : state.error,
      };
    
    case 'SET_ERROR':
      return {
        ...state,
        error: action.payload,
        isConnected: false,
      };
    
    case 'SET_HEARTBEAT':
      return {
        ...state,
        lastHeartbeat: action.payload,
      };
    
    case 'CLEAR_EVENTS':
      return {
        ...state,
        events: [],
        progress: null,
      };
    
    default:
      return state;
  }
}

// Hook options
export interface UseAgentStreamOptions {
  userId: string;
  autoConnect?: boolean;
  reconnectInterval?: number; // ms
  maxReconnectAttempts?: number;
  wsUrl?: string;
}

// Hook return type
export interface UseAgentStreamReturn {
  events: AgentEvent[];
  progress: AgentProgress | null;
  isConnected: boolean;
  error: string | null;
  lastHeartbeat: Date | null;
  connect: () => void;
  disconnect: () => void;
  clearEvents: () => void;
  sendPing: () => void;
}

/**
 * useAgentStream Hook
 * 
 * @param options - Configuration options
 * @returns Agent stream state and control functions
 */
export function useAgentStream(options: UseAgentStreamOptions): UseAgentStreamReturn {
  const {
    userId,
    autoConnect = true,
    reconnectInterval = 3000,
    maxReconnectAttempts = 10,
    wsUrl = `ws://localhost:8000/ws/agent-stream/${userId}`,
  } = options;

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [state, dispatch] = useReducer(agentStreamReducer, {
    events: [],
    progress: null,
    isConnected: false,
    error: null,
    lastHeartbeat: null,
  });

  // Handle incoming WebSocket messages
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);

      if (message.type === 'agent_event') {
        const agentEvent = message.data;
        
        // Add event to history
        dispatch({ type: 'ADD_EVENT', payload: agentEvent });

        // Update progress if it's a progress event
        if (agentEvent.event_type === AgentEventType.PROGRESS && agentEvent.metadata) {
          const progress: AgentProgress = {
            task_id: agentEvent.metadata.task_id,
            agent_id: agentEvent.agent_id,
            status: agentEvent.metadata.status,
            progress_pct: agentEvent.metadata.progress_pct,
            time_elapsed_sec: agentEvent.metadata.time_elapsed_sec,
            estimated_time_remaining_sec: agentEvent.metadata.estimated_time_remaining_sec,
            current_step: agentEvent.metadata.current_step,
            total_steps: agentEvent.metadata.total_steps,
            completed_steps: agentEvent.metadata.completed_steps,
          };
          dispatch({ type: 'UPDATE_PROGRESS', payload: progress });
        }
      } else if (message.type === 'heartbeat') {
        dispatch({ type: 'SET_HEARTBEAT', payload: new Date(message.timestamp) });
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  }, []);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    try {
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('Agent stream connected');
        dispatch({ type: 'SET_CONNECTED', payload: true });
        reconnectAttemptsRef.current = 0;
      };

      ws.onmessage = handleMessage;

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        dispatch({ type: 'SET_ERROR', payload: 'WebSocket connection error' });
      };

      ws.onclose = () => {
        console.log('Agent stream disconnected');
        dispatch({ type: 'SET_CONNECTED', payload: false });

        // Attempt reconnect
        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current += 1;
          console.log(`Reconnecting... (attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts})`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        } else {
          dispatch({ type: 'SET_ERROR', payload: 'Max reconnection attempts reached' });
        }
      };

      wsRef.current = ws;
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      dispatch({ type: 'SET_ERROR', payload: 'Failed to create WebSocket connection' });
    }
  }, [wsUrl, handleMessage, reconnectInterval, maxReconnectAttempts]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    dispatch({ type: 'SET_CONNECTED', payload: false });
  }, []);

  // Clear events
  const clearEvents = useCallback(() => {
    dispatch({ type: 'CLEAR_EVENTS' });
  }, []);

  // Send ping to server
  const sendPing = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send('ping');
    }
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  return {
    events: state.events,
    progress: state.progress,
    isConnected: state.isConnected,
    error: state.error,
    lastHeartbeat: state.lastHeartbeat,
    connect,
    disconnect,
    clearEvents,
    sendPing,
  };
}

