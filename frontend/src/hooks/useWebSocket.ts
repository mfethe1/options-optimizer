/**
 * WebSocket hook for real-time updates
 */
import { useRef, useCallback } from 'react';
import { useStore } from '../store';
import toast from 'react-hot-toast';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';

export const useWebSocket = () => {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  const setWsConnected = useStore((state) => state.setWsConnected);
  const updatePosition = useStore((state) => state.updatePosition);
  const addPosition = useStore((state) => state.addPosition);
  const removePosition = useStore((state) => state.removePosition);


  const connect = useCallback((userId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    const ws = new WebSocket(`${WS_URL}/${userId}`);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setWsConnected(true);
      reconnectAttempts.current = 0;
      toast.success('Connected to real-time updates');
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        handleMessage(message);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      toast.error('Connection error');
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setWsConnected(false);

      // Attempt to reconnect
      if (reconnectAttempts.current < maxReconnectAttempts) {
        reconnectAttempts.current++;
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
        
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log(`Reconnecting... (attempt ${reconnectAttempts.current})`);
          connect(userId);
        }, delay);
      } else {
        toast.error('Connection lost. Please refresh the page.');
      }
    };

    wsRef.current = ws;
  }, [setWsConnected]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setWsConnected(false);
  }, [setWsConnected]);

  const handleMessage = (message: any) => {
    switch (message.type) {
      case 'position_created':
        addPosition(message.data);
        toast.success('New position added');
        break;

      case 'position_updated':
        updatePosition(message.data.id, message.data);
        toast.success('Position updated');
        break;

      case 'position_deleted':
        removePosition(message.data.position_id);
        toast.success('Position removed');
        break;

      case 'analysis_completed':
        toast.success('Analysis completed');
        break;

      case 'heartbeat':
        // Keep-alive message
        break;

      default:
        console.log('Unknown message type:', message.type);
    }
  };

  return { connect, disconnect };
};

