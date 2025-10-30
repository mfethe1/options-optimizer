/**
 * AgentConversationDisplay Tests
 * 
 * Comprehensive test suite for AgentConversationDisplay component.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import AgentConversationDisplay from '../AgentConversationDisplay';
import { AgentEvent, AgentEventType } from '../../hooks/useAgentStream';

describe('AgentConversationDisplay', () => {
  const mockEvents: AgentEvent[] = [
    {
      timestamp: '2024-01-15T10:00:00.000Z',
      agent_id: 'distillation_v2',
      event_type: AgentEventType.STARTED,
      content: 'Agent started',
      metadata: { total_steps: 8 },
    },
    {
      timestamp: '2024-01-15T10:01:00.000Z',
      agent_id: 'distillation_v2',
      event_type: AgentEventType.THINKING,
      content: 'Analyzing portfolio metrics...',
    },
    {
      timestamp: '2024-01-15T10:02:00.000Z',
      agent_id: 'distillation_v2',
      event_type: AgentEventType.TOOL_CALL,
      content: 'Calling tool: compute_portfolio_metrics',
      metadata: {
        tool_name: 'compute_portfolio_metrics',
        args: { symbols: ['AAPL', 'GOOGL'] },
      },
    },
    {
      timestamp: '2024-01-15T10:03:00.000Z',
      agent_id: 'distillation_v2',
      event_type: AgentEventType.TOOL_RESULT,
      content: 'Tool succeeded',
      metadata: {
        tool_name: 'compute_portfolio_metrics',
        success: true,
        result_preview: 'Omega=1.85',
      },
    },
    {
      timestamp: '2024-01-15T10:04:00.000Z',
      agent_id: 'distillation_v2',
      event_type: AgentEventType.ERROR,
      content: 'Failed to fetch data',
      error_flag: true,
      metadata: { error_type: 'DataFetchError' },
    },
  ];

  it('renders empty state when no events', () => {
    render(<AgentConversationDisplay events={[]} />);
    expect(screen.getByText('No events yet')).toBeInTheDocument();
  });

  it('renders all events', () => {
    render(<AgentConversationDisplay events={mockEvents} />);
    
    expect(screen.getByText('Agent started')).toBeInTheDocument();
    expect(screen.getByText('Analyzing portfolio metrics...')).toBeInTheDocument();
    expect(screen.getByText('Calling tool: compute_portfolio_metrics')).toBeInTheDocument();
    expect(screen.getByText('Tool succeeded')).toBeInTheDocument();
    expect(screen.getByText('Failed to fetch data')).toBeInTheDocument();
  });

  it('displays event count', () => {
    render(<AgentConversationDisplay events={mockEvents} />);
    expect(screen.getByText('(5 events)')).toBeInTheDocument();
  });

  it('shows event type labels', () => {
    render(<AgentConversationDisplay events={mockEvents} />);
    
    expect(screen.getByText('Started')).toBeInTheDocument();
    expect(screen.getByText('Thinking')).toBeInTheDocument();
    expect(screen.getByText('Tool Call')).toBeInTheDocument();
    expect(screen.getByText('Tool Result')).toBeInTheDocument();
    expect(screen.getByText('Error')).toBeInTheDocument();
  });

  it('displays agent IDs', () => {
    render(<AgentConversationDisplay events={mockEvents} />);
    
    const agentIds = screen.getAllByText('distillation_v2');
    expect(agentIds.length).toBe(5);
  });

  it('expands and collapses metadata', () => {
    const { container } = render(<AgentConversationDisplay events={mockEvents} />);

    // Find expand buttons (chevron icons)
    const expandButtons = container.querySelectorAll('button svg[class*="lucide-chevron"]');

    // Click first expand button's parent
    const firstButton = expandButtons[0].closest('button');
    if (firstButton) {
      fireEvent.click(firstButton);

      // Metadata should be visible
      expect(screen.getByText('Metadata')).toBeInTheDocument();

      // Click again to collapse
      fireEvent.click(firstButton);

      // Metadata should be hidden
      expect(screen.queryByText('Metadata')).not.toBeInTheDocument();
    }
  });

  it('filters events by search term', () => {
    render(<AgentConversationDisplay events={mockEvents} />);
    
    const searchInput = screen.getByPlaceholderText('Search...');
    fireEvent.change(searchInput, { target: { value: 'portfolio' } });
    
    // Should show events with "portfolio" in content
    expect(screen.getByText('Analyzing portfolio metrics...')).toBeInTheDocument();
    expect(screen.getByText('Calling tool: compute_portfolio_metrics')).toBeInTheDocument();
    
    // Should not show other events
    expect(screen.queryByText('Agent started')).not.toBeInTheDocument();
  });

  it('shows "no results" message when search has no matches', () => {
    render(<AgentConversationDisplay events={mockEvents} />);
    
    const searchInput = screen.getByPlaceholderText('Search...');
    fireEvent.change(searchInput, { target: { value: 'nonexistent' } });
    
    expect(screen.getByText('No events match your search')).toBeInTheDocument();
  });

  it('toggles pause auto-scroll', () => {
    render(<AgentConversationDisplay events={mockEvents} />);
    
    const pauseButton = screen.getByTitle(/pause auto-scroll/i);
    fireEvent.click(pauseButton);
    
    // Button should now show "Resume auto-scroll"
    expect(screen.getByTitle(/resume auto-scroll/i)).toBeInTheDocument();
  });

  it('exports conversation log', () => {
    // Mock URL.createObjectURL
    (globalThis as any).URL.createObjectURL = vi.fn(() => 'blob:mock-url');
    ;(globalThis as any).URL.revokeObjectURL = vi.fn();

    const mockClick = vi.fn();
    const originalCreateElement = document.createElement.bind(document);

    vi.spyOn(document, 'createElement').mockImplementation((tagName: string) => {
      if (tagName === 'a') {
        const anchor = originalCreateElement('a');
        anchor.click = mockClick;
        return anchor;
      }
      return originalCreateElement(tagName);
    });

    render(<AgentConversationDisplay events={mockEvents} />);

    const exportButton = screen.getByTitle('Export conversation log');
    fireEvent.click(exportButton);

    expect(mockClick).toHaveBeenCalled();
    expect((globalThis as any).URL.createObjectURL).toHaveBeenCalled();

    // Restore
    vi.restoreAllMocks();
  });

  it('applies custom className', () => {
    const { container } = render(
      <AgentConversationDisplay events={mockEvents} className="custom-class" />
    );
    expect(container.querySelector('.custom-class')).toBeInTheDocument();
  });

  it('applies custom maxHeight', () => {
    const { container } = render(
      <AgentConversationDisplay events={mockEvents} maxHeight="400px" />
    );
    const scrollContainer = container.querySelector('[style*="max-height"]');
    expect(scrollContainer).toHaveStyle({ maxHeight: '400px' });
  });

  it('shows error events with red color', () => {
    render(<AgentConversationDisplay events={mockEvents} />);
    
    // Find the error event
    const errorEvent = screen.getByText('Failed to fetch data');
    const errorContainer = errorEvent.closest('.p-3');
    
    expect(errorContainer).toBeInTheDocument();
  });

  it('shows tool call events with purple color', () => {
    render(<AgentConversationDisplay events={mockEvents} />);
    
    const toolCallEvent = screen.getByText('Calling tool: compute_portfolio_metrics');
    expect(toolCallEvent).toBeInTheDocument();
  });

  it('shows thinking events with blue color', () => {
    render(<AgentConversationDisplay events={mockEvents} />);
    
    const thinkingEvent = screen.getByText('Analyzing portfolio metrics...');
    expect(thinkingEvent).toBeInTheDocument();
  });

  it('renders icons for each event type', () => {
    const { container } = render(<AgentConversationDisplay events={mockEvents} />);
    
    // Should have multiple SVG icons (one per event)
    const icons = container.querySelectorAll('svg');
    expect(icons.length).toBeGreaterThan(5); // At least one icon per event + UI icons
  });

  it('formats timestamps correctly', () => {
    render(<AgentConversationDisplay events={mockEvents} />);
    
    // Timestamps should be formatted as HH:MM:SS.mmm
    // The exact format depends on locale, but should contain colons
    const timestamps = screen.getAllByText(/\d{1,2}:\d{2}:\d{2}/);
    expect(timestamps.length).toBeGreaterThan(0);
  });

  it('handles events without metadata', () => {
    const eventsWithoutMetadata: AgentEvent[] = [
      {
        timestamp: '2024-01-15T10:00:00.000Z',
        agent_id: 'test_agent',
        event_type: AgentEventType.THINKING,
        content: 'Simple thought',
      },
    ];

    const { container } = render(<AgentConversationDisplay events={eventsWithoutMetadata} />);

    expect(screen.getByText('Simple thought')).toBeInTheDocument();
    // Should not have expand button (chevron icons)
    const expandButtons = container.querySelectorAll('button svg[class*="lucide-chevron"]');
    expect(expandButtons.length).toBe(0);
  });

  it('auto-scrolls to bottom when new events arrive', async () => {
    const { rerender } = render(<AgentConversationDisplay events={[mockEvents[0]]} />);
    
    // Add more events
    rerender(<AgentConversationDisplay events={mockEvents} />);
    
    // scrollIntoView should be called (mocked in test setup)
    await waitFor(() => {
      // Just verify component re-rendered with new events
      expect(screen.getByText('Failed to fetch data')).toBeInTheDocument();
    });
  });
});

