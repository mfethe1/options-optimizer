/**
 * AgentConversationDisplay - Real-time Agent Message Stream
 * 
 * Bloomberg Terminal-style conversation display showing:
 * - Scrolling message list (auto-scroll to bottom)
 * - Color-coded event types (thinking=blue, tool_call=purple, result=green/red, error=red)
 * - Timestamps for each message
 * - Expandable tool call details (args, results)
 * - Search/filter messages
 * - Export conversation log
 * 
 * Designed to surpass Bloomberg Terminal and TradingView with superior transparency.
 */

import React, { useEffect, useRef, useState } from 'react';
import {
  Brain,
  Wrench,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Play,
  Flag,
  ChevronDown,
  ChevronRight,
  Search,
  Download,
  Pause,
} from 'lucide-react';
import { AgentEvent, AgentEventType } from '../hooks/useAgentStream';

interface AgentConversationDisplayProps {
  events: AgentEvent[];
  className?: string;
  maxHeight?: string;
  autoScroll?: boolean;
}

export const AgentConversationDisplay: React.FC<AgentConversationDisplayProps> = ({
  events,
  className = '',
  maxHeight = '600px',
  autoScroll = true,
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [expandedEvents, setExpandedEvents] = useState<Set<number>>(new Set());
  const [searchTerm, setSearchTerm] = useState('');
  const [isPaused, setIsPaused] = useState(false);

  // Auto-scroll to bottom when new events arrive
  useEffect(() => {
    if (autoScroll && !isPaused && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [events, autoScroll, isPaused]);

  // Toggle event expansion
  const toggleExpand = (index: number) => {
    const newExpanded = new Set(expandedEvents);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedEvents(newExpanded);
  };

  // Get event color
  const getEventColor = (eventType: AgentEventType): string => {
    switch (eventType) {
      case AgentEventType.THINKING:
        return '#3b82f6'; // Blue
      case AgentEventType.TOOL_CALL:
        return '#8b5cf6'; // Purple
      case AgentEventType.TOOL_RESULT:
        return '#10b981'; // Green (will be red if error)
      case AgentEventType.ERROR:
        return '#ef4444'; // Red
      case AgentEventType.STARTED:
        return '#10b981'; // Green
      case AgentEventType.COMPLETED:
        return '#10b981'; // Green
      case AgentEventType.PROGRESS:
        return '#f59e0b'; // Orange
      default:
        return '#6b7280'; // Gray
    }
  };

  // Get event icon
  const getEventIcon = (event: AgentEvent) => {
    const color = event.error_flag ? '#ef4444' : getEventColor(event.event_type);
    const iconProps = { className: 'w-4 h-4', style: { color } };

    switch (event.event_type) {
      case AgentEventType.THINKING:
        return <Brain {...iconProps} />;
      case AgentEventType.TOOL_CALL:
        return <Wrench {...iconProps} />;
      case AgentEventType.TOOL_RESULT:
        return event.error_flag ? <XCircle {...iconProps} /> : <CheckCircle2 {...iconProps} />;
      case AgentEventType.ERROR:
        return <AlertTriangle {...iconProps} />;
      case AgentEventType.STARTED:
        return <Play {...iconProps} />;
      case AgentEventType.COMPLETED:
        return <Flag {...iconProps} />;
      default:
        return <CheckCircle2 {...iconProps} />;
    }
  };

  // Get event type label
  const getEventTypeLabel = (eventType: AgentEventType): string => {
    switch (eventType) {
      case AgentEventType.THINKING:
        return 'Thinking';
      case AgentEventType.TOOL_CALL:
        return 'Tool Call';
      case AgentEventType.TOOL_RESULT:
        return 'Tool Result';
      case AgentEventType.ERROR:
        return 'Error';
      case AgentEventType.STARTED:
        return 'Started';
      case AgentEventType.COMPLETED:
        return 'Completed';
      case AgentEventType.PROGRESS:
        return 'Progress';
      default:
        return eventType;
    }
  };

  // Format timestamp
  const formatTimestamp = (timestamp: string): string => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  // Filter events by search term
  const filteredEvents = events.filter((event) => {
    if (!searchTerm) return true;
    const searchLower = searchTerm.toLowerCase();
    return (
      event.content.toLowerCase().includes(searchLower) ||
      event.agent_id.toLowerCase().includes(searchLower) ||
      event.event_type.toLowerCase().includes(searchLower)
    );
  });

  // Export conversation log
  const exportLog = () => {
    const log = events
      .map((event) => {
        const timestamp = formatTimestamp(event.timestamp);
        const type = getEventTypeLabel(event.event_type);
        const content = event.content;
        const metadata = event.metadata ? JSON.stringify(event.metadata, null, 2) : '';
        return `[${timestamp}] ${type}: ${content}\n${metadata ? `Metadata: ${metadata}\n` : ''}`;
      })
      .join('\n---\n\n');

    const blob = new Blob([log], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `agent-log-${new Date().toISOString()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className={`bg-[#2a2a2a] border border-[#404040] rounded-lg ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-[#404040]">
        <div className="flex items-center">
          <Brain className="w-5 h-5 mr-2 text-blue-400" />
          <span className="text-white font-semibold">Agent Conversation</span>
          <span className="ml-2 text-sm text-gray-400">({filteredEvents.length} events)</span>
        </div>
        
        <div className="flex items-center space-x-2">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-2 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-8 pr-3 py-1 bg-[#1a1a1a] border border-[#404040] rounded text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
            />
          </div>

          {/* Pause Auto-scroll */}
          <button
            onClick={() => setIsPaused(!isPaused)}
            className="p-2 bg-[#1a1a1a] border border-[#404040] rounded hover:bg-[#3a3a3a] transition-colors"
            title={isPaused ? 'Resume auto-scroll' : 'Pause auto-scroll'}
          >
            <Pause className={`w-4 h-4 ${isPaused ? 'text-orange-400' : 'text-gray-400'}`} />
          </button>

          {/* Export */}
          <button
            onClick={exportLog}
            className="p-2 bg-[#1a1a1a] border border-[#404040] rounded hover:bg-[#3a3a3a] transition-colors"
            title="Export conversation log"
          >
            <Download className="w-4 h-4 text-gray-400" />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div
        ref={containerRef}
        className="overflow-y-auto p-4 space-y-3"
        style={{ maxHeight }}
      >
        {filteredEvents.length === 0 ? (
          <div className="flex items-center justify-center text-gray-400 py-8">
            <span className="text-sm">
              {searchTerm ? 'No events match your search' : 'No events yet'}
            </span>
          </div>
        ) : (
          filteredEvents.map((event, index) => {
            const isExpanded = expandedEvents.has(index);
            const hasMetadata = event.metadata && Object.keys(event.metadata).length > 0;
            const color = event.error_flag ? '#ef4444' : getEventColor(event.event_type);

            return (
              <div
                key={index}
                className="p-3 bg-[#1a1a1a] rounded border border-[#404040] hover:border-[#505050] transition-colors"
              >
                {/* Event Header */}
                <div className="flex items-start justify-between">
                  <div className="flex items-start flex-1">
                    {/* Icon */}
                    <div className="mt-0.5">{getEventIcon(event)}</div>

                    {/* Content */}
                    <div className="ml-3 flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <span className="text-xs font-semibold" style={{ color }}>
                          {getEventTypeLabel(event.event_type)}
                        </span>
                        <span className="text-xs text-gray-500">{formatTimestamp(event.timestamp)}</span>
                        <span className="text-xs text-gray-600">•</span>
                        <span className="text-xs text-gray-500">{event.agent_id}</span>
                      </div>
                      <div className="text-sm text-white">{event.content}</div>
                    </div>

                    {/* Expand Button */}
                    {hasMetadata && (
                      <button
                        onClick={() => toggleExpand(index)}
                        className="ml-2 p-1 hover:bg-[#2a2a2a] rounded transition-colors"
                      >
                        {isExpanded ? (
                          <ChevronDown className="w-4 h-4 text-gray-400" />
                        ) : (
                          <ChevronRight className="w-4 h-4 text-gray-400" />
                        )}
                      </button>
                    )}
                  </div>
                </div>

                {/* Expanded Metadata */}
                {isExpanded && hasMetadata && (
                  <div className="mt-3 p-3 bg-[#0a0a0a] rounded border border-[#404040]">
                    <div className="text-xs text-gray-400 mb-2">Metadata</div>
                    <pre className="text-xs text-gray-300 font-mono overflow-x-auto">
                      {JSON.stringify(event.metadata, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            );
          })
        )}
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
};

export default AgentConversationDisplay;

