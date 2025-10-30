/**
 * AgentProgressPanel - Real-time Progress Display for LLM Agent Analysis
 * 
 * Bloomberg Terminal-style progress panel showing:
 * - Progress bar (0-100%)
 * - Time elapsed / estimated remaining
 * - Current step description
 * - Status indicator (pending/running/completed/failed)
 * - Color-coded status (green=completed, yellow=running, red=failed)
 * 
 * Designed to compete with Bloomberg Terminal and TradingView with superior UX.
 */

import React from 'react';
import { Loader2, CheckCircle2, XCircle, Clock, TrendingUp } from 'lucide-react';
import { AgentProgress } from '../hooks/useAgentStream';

interface AgentProgressPanelProps {
  progress: AgentProgress | null;
  className?: string;
}

export const AgentProgressPanel: React.FC<AgentProgressPanelProps> = ({
  progress,
  className = '',
}) => {
  if (!progress) {
    return (
      <div className={`bg-[#2a2a2a] border border-[#404040] rounded-lg p-6 ${className}`}>
        <div className="flex items-center justify-center text-gray-400">
          <Loader2 className="w-5 h-5 mr-2 animate-spin" />
          <span className="text-sm">Waiting for analysis to start...</span>
        </div>
      </div>
    );
  }

  // Format time (seconds to MM:SS)
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Get status color
  const getStatusColor = (): string => {
    switch (progress.status) {
      case 'completed':
        return '#10b981'; // Green
      case 'running':
        return '#f59e0b'; // Orange
      case 'failed':
        return '#ef4444'; // Red
      case 'pending':
      default:
        return '#3b82f6'; // Blue
    }
  };

  // Get status icon
  const getStatusIcon = () => {
    switch (progress.status) {
      case 'completed':
        return <CheckCircle2 className="w-5 h-5" style={{ color: getStatusColor() }} />;
      case 'running':
        return <Loader2 className="w-5 h-5 animate-spin" style={{ color: getStatusColor() }} />;
      case 'failed':
        return <XCircle className="w-5 h-5" style={{ color: getStatusColor() }} />;
      case 'pending':
      default:
        return <Clock className="w-5 h-5" style={{ color: getStatusColor() }} />;
    }
  };

  // Get status text
  const getStatusText = (): string => {
    switch (progress.status) {
      case 'completed':
        return 'Analysis Complete';
      case 'running':
        return 'Analysis In Progress';
      case 'failed':
        return 'Analysis Failed';
      case 'pending':
      default:
        return 'Analysis Pending';
    }
  };

  // Calculate progress percentage (clamped to 0-100)
  const progressPct = Math.min(Math.max(progress.progress_pct, 0), 100);

  // Estimate completion time
  const estimatedCompletion = progress.estimated_time_remaining_sec
    ? new Date(Date.now() + progress.estimated_time_remaining_sec * 1000)
    : null;

  return (
    <div className={`bg-[#2a2a2a] border border-[#404040] rounded-lg p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          {getStatusIcon()}
          <span className="ml-2 text-white font-semibold">{getStatusText()}</span>
        </div>
        <div className="flex items-center text-sm text-gray-400">
          <TrendingUp className="w-4 h-4 mr-1" />
          <span>{progress.agent_id}</span>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-gray-400">
            Step {progress.completed_steps} of {progress.total_steps}
          </span>
          <span className="text-sm font-semibold" style={{ color: getStatusColor() }}>
            {progressPct.toFixed(1)}%
          </span>
        </div>
        
        <div className="w-full h-2 bg-[#1a1a1a] rounded-full overflow-hidden">
          <div
            className="h-full transition-all duration-500 ease-out rounded-full"
            style={{
              width: `${progressPct}%`,
              background: `linear-gradient(90deg, ${getStatusColor()}, ${getStatusColor()}dd)`,
            }}
          />
        </div>
      </div>

      {/* Current Step */}
      <div className="mb-4 p-3 bg-[#1a1a1a] rounded border border-[#404040]">
        <div className="text-xs text-gray-400 mb-1">Current Step</div>
        <div className="text-sm text-white">{progress.current_step}</div>
      </div>

      {/* Time Tracking */}
      <div className="grid grid-cols-2 gap-4">
        {/* Time Elapsed */}
        <div className="p-3 bg-[#1a1a1a] rounded border border-[#404040]">
          <div className="flex items-center text-xs text-gray-400 mb-1">
            <Clock className="w-3 h-3 mr-1" />
            <span>Time Elapsed</span>
          </div>
          <div className="text-lg font-semibold text-white">
            {formatTime(progress.time_elapsed_sec)}
          </div>
        </div>

        {/* Time Remaining */}
        <div className="p-3 bg-[#1a1a1a] rounded border border-[#404040]">
          <div className="flex items-center text-xs text-gray-400 mb-1">
            <Clock className="w-3 h-3 mr-1" />
            <span>Est. Remaining</span>
          </div>
          <div className="text-lg font-semibold text-white">
            {progress.estimated_time_remaining_sec !== undefined
              ? formatTime(progress.estimated_time_remaining_sec)
              : '--:--'}
          </div>
        </div>
      </div>

      {/* Estimated Completion Time */}
      {estimatedCompletion && progress.status === 'running' && (
        <div className="mt-4 p-3 bg-[#1a1a1a] rounded border border-[#404040]">
          <div className="text-xs text-gray-400 mb-1">Estimated Completion</div>
          <div className="text-sm text-white">
            {estimatedCompletion.toLocaleTimeString('en-US', {
              hour: '2-digit',
              minute: '2-digit',
              second: '2-digit',
            })}
          </div>
        </div>
      )}

      {/* Task ID (for debugging) */}
      <div className="mt-4 pt-4 border-t border-[#404040]">
        <div className="text-xs text-gray-500">
          Task ID: <span className="font-mono">{progress.task_id}</span>
        </div>
      </div>
    </div>
  );
};

export default AgentProgressPanel;

