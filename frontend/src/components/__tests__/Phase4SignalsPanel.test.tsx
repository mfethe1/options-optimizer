import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { Phase4SignalsPanel } from '../Phase4SignalsPanel';
import { Phase4Tech } from '../../types/investor-report';

// Mock the usePhase4Stream hook
vi.mock('../../hooks/usePhase4Stream', () => ({
  usePhase4Stream: vi.fn(),
}));

import { usePhase4Stream } from '../../hooks/usePhase4Stream';

describe('Phase4SignalsPanel', () => {
  const mockPhase4Data: Phase4Tech = {
    options_flow_composite: 0.65,
    residual_momentum: 1.85,
    seasonality_score: 0.42,
    breadth_liquidity: 0.78,
    explanations: [
      'Strong options flow indicates bullish sentiment',
      'Residual momentum shows outperformance vs sector',
    ],

  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders all 4 sub-components with data', () => {
    (usePhase4Stream as any).mockReturnValue({
      phase4Data: mockPhase4Data,
      isConnected: true,
      error: null,
    });

    render(<Phase4SignalsPanel userId="test_user" />);

    // Check all 4 metric titles are present
    expect(screen.getByText('Options Flow Composite')).toBeInTheDocument();
    expect(screen.getByText('Residual Momentum')).toBeInTheDocument();
    expect(screen.getByText('Seasonality Score')).toBeInTheDocument();
    expect(screen.getByText('Breadth & Liquidity')).toBeInTheDocument();

    // Check header
    expect(screen.getByText('Phase 4 Signals')).toBeInTheDocument();
    expect(screen.getByText('Short-horizon edge (1-5 day alpha)')).toBeInTheDocument();
  });

  it('shows live indicator when connected', () => {
    (usePhase4Stream as any).mockReturnValue({
      phase4Data: mockPhase4Data,
      isConnected: true,
      error: null,
    });

    render(<Phase4SignalsPanel userId="test_user" />);

    expect(screen.getByText('Live')).toBeInTheDocument();
    expect(screen.getByLabelText('Connected')).toBeInTheDocument();
  });

  it('shows disconnected indicator when not connected', () => {
    (usePhase4Stream as any).mockReturnValue({
      phase4Data: mockPhase4Data,
      isConnected: false,
      error: null,
    });

    render(<Phase4SignalsPanel userId="test_user" />);

    expect(screen.getByText('Disconnected')).toBeInTheDocument();
    expect(screen.getByLabelText('Disconnected')).toBeInTheDocument();
  });

  it('shows loading state when data is null', () => {
    (usePhase4Stream as any).mockReturnValue({
      phase4Data: null,
      isConnected: false,
      error: null,
    });

    render(<Phase4SignalsPanel userId="test_user" />);

    expect(screen.getByText('Loading Phase 4 metrics...')).toBeInTheDocument();
  });

  it('displays error message when error occurs', () => {
    (usePhase4Stream as any).mockReturnValue({
      phase4Data: null,
      isConnected: false,
      error: 'WebSocket connection failed',
    });

    render(<Phase4SignalsPanel userId="test_user" />);

    expect(screen.getByText(/Connection error:/)).toBeInTheDocument();
    expect(screen.getByText(/WebSocket connection failed/)).toBeInTheDocument();
  });

  it('displays explanations when available', () => {
    (usePhase4Stream as any).mockReturnValue({
      phase4Data: mockPhase4Data,
      isConnected: true,
      error: null,
    });

    render(<Phase4SignalsPanel userId="test_user" />);

    expect(screen.getByText('Interpretation')).toBeInTheDocument();
    expect(screen.getByText(/Strong options flow indicates bullish sentiment/)).toBeInTheDocument();
    expect(screen.getByText(/Residual momentum shows outperformance vs sector/)).toBeInTheDocument();
  });



  it('uses initialData when WebSocket data is not available', () => {
    (usePhase4Stream as any).mockReturnValue({
      phase4Data: null,
      isConnected: false,
      error: null,
    });

    render(<Phase4SignalsPanel userId="test_user" initialData={mockPhase4Data} />);

    // Should display data from initialData prop
    expect(screen.getByText('Options Flow Composite')).toBeInTheDocument();
    expect(screen.getByText('0.65')).toBeInTheDocument();
  });

  it('updates in real-time when WebSocket data changes', async () => {
    const { rerender } = render(<Phase4SignalsPanel userId="test_user" />);

    // Initial state
    (usePhase4Stream as any).mockReturnValue({
      phase4Data: mockPhase4Data,
      isConnected: true,
      error: null,
    });

    rerender(<Phase4SignalsPanel userId="test_user" />);
    expect(screen.getByText('0.65')).toBeInTheDocument();

    // Simulate WebSocket update
    const updatedData = { ...mockPhase4Data, options_flow_composite: 0.85 };
    (usePhase4Stream as any).mockReturnValue({
      phase4Data: updatedData,
      isConnected: true,
      error: null,
    });

    rerender(<Phase4SignalsPanel userId="test_user" />);

    await waitFor(() => {
      expect(screen.getByText('0.85')).toBeInTheDocument();
    });
  });

  it('passes correct options to usePhase4Stream hook', () => {
    (usePhase4Stream as any).mockReturnValue({
      phase4Data: mockPhase4Data,
      isConnected: true,
      error: null,
    });

    render(<Phase4SignalsPanel userId="test_user_123" />);

    expect(usePhase4Stream).toHaveBeenCalledWith('test_user_123', {
      autoReconnect: true,
      maxReconnectAttempts: 5,
    });
  });

  it('renders responsive grid layout', () => {
    (usePhase4Stream as any).mockReturnValue({
      phase4Data: mockPhase4Data,
      isConnected: true,
      error: null,
    });

    const { container } = render(<Phase4SignalsPanel userId="test_user" />);

    // Check for grid layout classes
    const gridElement = container.querySelector('.grid.grid-cols-1.lg\\:grid-cols-2');
    expect(gridElement).toBeInTheDocument();
  });

  it('handles null values in Phase4Tech data gracefully', () => {
    const partialData: Phase4Tech = {
      options_flow_composite: 0.65,
      residual_momentum: null,
      seasonality_score: null,
      breadth_liquidity: 0.78,
      explanations: [],

    };

    (usePhase4Stream as any).mockReturnValue({
      phase4Data: partialData,
      isConnected: true,
      error: null,
    });

    render(<Phase4SignalsPanel userId="test_user" />);

    // Should show loading state for null values
    expect(screen.getAllByText('Computing...')).toHaveLength(2);
    // Should show actual values for non-null
    expect(screen.getByText('0.65')).toBeInTheDocument();
    expect(screen.getByText('78%')).toBeInTheDocument();
  });
});

