/**
 * AgentTransparencyDemoPage - Interactive Demo for Agent Transparency System
 * 
 * Showcases real-time LLM agent event streaming with:
 * - Live WebSocket connection to agent stream
 * - Progress panel with time tracking
 * - Conversation display with color-coded events
 * - Connection status and controls
 * - Mock data generator for testing
 * 
 * Designed to demonstrate Bloomberg Terminal / TradingView-level transparency.
 */

import React, { useState } from 'react';
import { Activity, Wifi, WifiOff, RefreshCw, Trash2, Zap } from 'lucide-react';
import { useAgentStream, AgentEvent, AgentEventType } from '../hooks/useAgentStream';
import AgentProgressPanel from '../components/AgentProgressPanel';
import AgentConversationDisplay from '../components/AgentConversationDisplay';

export const AgentTransparencyDemoPage: React.FC = () => {
  const [userId, setUserId] = useState('demo-user-123');
  const [useMockData, setUseMockData] = useState(false);

  const {
    events,
    progress,
    isConnected,
    error,
    lastHeartbeat,
    connect,
    disconnect,
    clearEvents,
    sendPing,
  } = useAgentStream({
    userId,
    autoConnect: !useMockData,
  });

  // Mock data for testing UI without backend
  const mockEvents: AgentEvent[] = useMockData
    ? [
        {
          timestamp: new Date().toISOString(),
          agent_id: 'distillation_v2',
          event_type: AgentEventType.STARTED,
          content: 'Agent distillation_v2 started task distillation_v2_1234567890',
          metadata: { total_steps: 8 },
        },
        {
          timestamp: new Date(Date.now() - 60000).toISOString(),
          agent_id: 'distillation_v2',
          event_type: AgentEventType.THINKING,
          content: 'Initializing DistillationAgent V2...',
        },
        {
          timestamp: new Date(Date.now() - 50000).toISOString(),
          agent_id: 'distillation_v2',
          event_type: AgentEventType.TOOL_CALL,
          content: 'Calling tool: compute_portfolio_metrics',
          metadata: {
            tool_name: 'compute_portfolio_metrics',
            args: { symbols: ['AAPL', 'GOOGL', 'MSFT'] },
          },
        },
        {
          timestamp: new Date(Date.now() - 40000).toISOString(),
          agent_id: 'distillation_v2',
          event_type: AgentEventType.TOOL_RESULT,
          content: 'Tool compute_portfolio_metrics succeeded',
          metadata: {
            tool_name: 'compute_portfolio_metrics',
            success: true,
            result_preview: 'Computed metrics for 3 positions: Omega=1.85, GH1=0.72, Pain Index=0.15',
          },
        },
        {
          timestamp: new Date(Date.now() - 30000).toISOString(),
          agent_id: 'distillation_v2',
          event_type: AgentEventType.THINKING,
          content: 'Computing Phase 4 technical signals...',
        },
        {
          timestamp: new Date(Date.now() - 20000).toISOString(),
          agent_id: 'distillation_v2',
          event_type: AgentEventType.TOOL_CALL,
          content: 'Calling tool: compute_phase4_metrics',
          metadata: {
            tool_name: 'compute_phase4_metrics',
            args: { symbols: ['AAPL', 'GOOGL', 'MSFT'] },
          },
        },
        {
          timestamp: new Date(Date.now() - 10000).toISOString(),
          agent_id: 'distillation_v2',
          event_type: AgentEventType.TOOL_RESULT,
          content: 'Tool compute_phase4_metrics succeeded',
          metadata: {
            tool_name: 'compute_phase4_metrics',
            success: true,
            result_preview:
              'Options Flow: 0.65, Residual Momentum: 0.42, Seasonality: 0.78, Breadth/Liquidity: 0.55',
          },
        },
        {
          timestamp: new Date(Date.now() - 5000).toISOString(),
          agent_id: 'distillation_v2',
          event_type: AgentEventType.ERROR,
          content: 'Failed to fetch options chain data for TSLA',
          error_flag: true,
          metadata: {
            error_type: 'DataFetchError',
            retry_count: 2,
          },
        },
      ]
    : events;

  const mockProgress = useMockData
    ? {
        task_id: 'distillation_v2_1234567890',
        agent_id: 'distillation_v2',
        status: 'running' as const,
        progress_pct: 62.5,
        time_elapsed_sec: 125,
        estimated_time_remaining_sec: 75,
        current_step: 'Generating executive summary with LLM...',
        total_steps: 8,
        completed_steps: 5,
      }
    : progress;

  return (
    <div className="min-h-screen bg-[#1a1a1a] text-white p-6">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2 flex items-center">
              <Activity className="w-8 h-8 mr-3 text-blue-400" />
              Agent Transparency System
            </h1>
            <p className="text-gray-400">
              Real-time LLM agent event streaming for institutional-grade transparency
            </p>
          </div>

          {/* Connection Status */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center">
              {isConnected ? (
                <>
                  <Wifi className="w-5 h-5 mr-2 text-green-400" />
                  <span className="text-sm text-green-400">Connected</span>
                </>
              ) : (
                <>
                  <WifiOff className="w-5 h-5 mr-2 text-red-400" />
                  <span className="text-sm text-red-400">Disconnected</span>
                </>
              )}
            </div>

            {lastHeartbeat && (
              <div className="text-xs text-gray-500">
                Last heartbeat: {lastHeartbeat.toLocaleTimeString()}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="max-w-7xl mx-auto mb-6">
        <div className="bg-[#2a2a2a] border border-[#404040] rounded-lg p-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* User ID Input */}
            <div>
              <label className="block text-sm text-gray-400 mb-2">User ID</label>
              <input
                type="text"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#404040] rounded text-white focus:outline-none focus:border-blue-500"
                placeholder="Enter user ID"
              />
            </div>

            {/* Mock Data Toggle */}
            <div>
              <label className="block text-sm text-gray-400 mb-2">Data Source</label>
              <button
                onClick={() => setUseMockData(!useMockData)}
                className={`w-full px-3 py-2 rounded font-semibold transition-colors ${
                  useMockData
                    ? 'bg-orange-500 hover:bg-orange-600 text-white'
                    : 'bg-blue-500 hover:bg-blue-600 text-white'
                }`}
              >
                {useMockData ? 'Using Mock Data' : 'Using Live WebSocket'}
              </button>
            </div>

            {/* Actions */}
            <div>
              <label className="block text-sm text-gray-400 mb-2">Actions</label>
              <div className="flex space-x-2">
                <button
                  onClick={isConnected ? disconnect : connect}
                  disabled={useMockData}
                  className="flex-1 px-3 py-2 bg-[#1a1a1a] border border-[#404040] rounded hover:bg-[#3a3a3a] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <RefreshCw className="w-4 h-4 mx-auto" />
                </button>
                <button
                  onClick={sendPing}
                  disabled={!isConnected || useMockData}
                  className="flex-1 px-3 py-2 bg-[#1a1a1a] border border-[#404040] rounded hover:bg-[#3a3a3a] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Zap className="w-4 h-4 mx-auto" />
                </button>
                <button
                  onClick={clearEvents}
                  className="flex-1 px-3 py-2 bg-[#1a1a1a] border border-[#404040] rounded hover:bg-[#3a3a3a] transition-colors"
                >
                  <Trash2 className="w-4 h-4 mx-auto" />
                </button>
              </div>
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="mt-4 p-3 bg-red-900/20 border border-red-500/50 rounded">
              <div className="text-sm text-red-400">{error}</div>
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Progress Panel */}
        <div className="lg:col-span-1">
          <AgentProgressPanel progress={mockProgress} />
        </div>

        {/* Conversation Display */}
        <div className="lg:col-span-2">
          <AgentConversationDisplay events={mockEvents} maxHeight="700px" />
        </div>
      </div>

      {/* Documentation */}
      <div className="max-w-7xl mx-auto mt-6">
        <div className="bg-[#2a2a2a] border border-[#404040] rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">How It Works</h2>
          <div className="space-y-4 text-sm text-gray-300">
            <div>
              <h3 className="font-semibold text-white mb-2">Real-Time Event Streaming</h3>
              <p>
                The agent transparency system streams events from LLM agents in real-time via WebSocket.
                Events include agent thoughts, tool calls, results, progress updates, and errors.
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-white mb-2">Event Types</h3>
              <ul className="list-disc list-inside space-y-1 ml-4">
                <li>
                  <span className="text-blue-400">THINKING</span> - Agent reasoning steps
                </li>
                <li>
                  <span className="text-purple-400">TOOL_CALL</span> - Tool invocations with arguments
                </li>
                <li>
                  <span className="text-green-400">TOOL_RESULT</span> - Tool results (success/failure)
                </li>
                <li>
                  <span className="text-orange-400">PROGRESS</span> - Progress updates with time estimates
                </li>
                <li>
                  <span className="text-red-400">ERROR</span> - Error messages with recovery info
                </li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold text-white mb-2">WebSocket Endpoint</h3>
              <code className="block p-2 bg-[#1a1a1a] rounded font-mono text-xs">
                ws://localhost:8000/ws/agent-stream/{'{user_id}'}
              </code>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgentTransparencyDemoPage;

