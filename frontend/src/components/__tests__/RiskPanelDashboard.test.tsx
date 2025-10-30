import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { RiskPanelDashboard } from '../RiskPanelDashboard';
import { RiskPanel } from '../../types/investor-report';

describe('RiskPanelDashboard', () => {
  const mockRiskPanel: RiskPanel = {
    omega: 2.15,
    gh1: 1.68,
    pain_index: 4.2,
    upside_capture: 105.3,
    downside_capture: 42.8,
    cvar_95: -6.5,
    max_drawdown: -12.3,
    explanations: [
      'Omega ratio >2.0 indicates Renaissance-level performance',
      'Downside capture <50% shows excellent protection',
    ],
  };

  it('renders all 7 risk metrics', () => {
    render(<RiskPanelDashboard riskPanel={mockRiskPanel} />);

    expect(screen.getByText('Omega Ratio')).toBeInTheDocument();
    expect(screen.getByText('GH1 Ratio')).toBeInTheDocument();
    expect(screen.getByText('Pain Index')).toBeInTheDocument();
    expect(screen.getByText('Upside Capture')).toBeInTheDocument();
    expect(screen.getByText('Downside Capture')).toBeInTheDocument();
    expect(screen.getByText('CVaR 95%')).toBeInTheDocument();
    expect(screen.getByText('Max Drawdown')).toBeInTheDocument();
  });

  it('renders header with title and description', () => {
    render(<RiskPanelDashboard riskPanel={mockRiskPanel} />);

    expect(screen.getByText('Risk Panel')).toBeInTheDocument();
    expect(screen.getByText('Institutional-grade risk metrics')).toBeInTheDocument();
  });

  it('displays neutral regime by default', () => {
    render(<RiskPanelDashboard riskPanel={mockRiskPanel} />);

    expect(screen.getByText('neutral Regime')).toBeInTheDocument();
  });

  it('displays bull regime when specified', () => {
    render(<RiskPanelDashboard riskPanel={mockRiskPanel} regime="bull" />);

    expect(screen.getByText('bull Regime')).toBeInTheDocument();
  });

  it('displays bear regime when specified', () => {
    render(<RiskPanelDashboard riskPanel={mockRiskPanel} regime="bear" />);

    expect(screen.getByText('bear Regime')).toBeInTheDocument();
  });

  it('displays explanations when available', () => {
    render(<RiskPanelDashboard riskPanel={mockRiskPanel} />);

    expect(screen.getByText('Risk Assessment')).toBeInTheDocument();
    expect(screen.getByText(/Omega ratio >2.0 indicates Renaissance-level performance/)).toBeInTheDocument();
    expect(screen.getByText(/Downside capture <50% shows excellent protection/)).toBeInTheDocument();
  });

  it('does not display explanations section when empty', () => {
    const riskPanelNoExplanations = { ...mockRiskPanel, explanations: [] };
    render(<RiskPanelDashboard riskPanel={riskPanelNoExplanations} />);

    expect(screen.queryByText('Risk Assessment')).not.toBeInTheDocument();
  });

  it('displays risk summary with correct assessments', () => {
    const { container } = render(<RiskPanelDashboard riskPanel={mockRiskPanel} />);

    expect(screen.getByText('Return Quality')).toBeInTheDocument();
    expect(screen.getByText('Downside Protection')).toBeInTheDocument();
    expect(screen.getByText('Tail Risk')).toBeInTheDocument();
    expect(screen.getByText('Moderate')).toBeInTheDocument(); // CVaR -6.5 is between -5 and -10

    // Check that risk summary section exists
    const riskSummary = container.querySelector('.grid.grid-cols-1.md\\:grid-cols-3');
    expect(riskSummary).toBeInTheDocument();
  });

  it('renders responsive grid layout', () => {
    const { container } = render(<RiskPanelDashboard riskPanel={mockRiskPanel} />);

    const gridElement = container.querySelector('.grid.grid-cols-1.lg\\:grid-cols-2');
    expect(gridElement).toBeInTheDocument();
  });

  it('passes correct props to RiskMetricCard components', () => {
    render(<RiskPanelDashboard riskPanel={mockRiskPanel} regime="bull" />);

    // Check that values are displayed (RiskMetricCard should render them)
    expect(screen.getByText('2.15')).toBeInTheDocument(); // Omega
    expect(screen.getByText('1.68')).toBeInTheDocument(); // GH1
    expect(screen.getByText('4.2%')).toBeInTheDocument(); // Pain Index
    expect(screen.getByText('105.3%')).toBeInTheDocument(); // Upside Capture
    expect(screen.getByText('42.8%')).toBeInTheDocument(); // Downside Capture
    expect(screen.getByText('-6.5%')).toBeInTheDocument(); // CVaR
    expect(screen.getByText('-12.3%')).toBeInTheDocument(); // Max Drawdown
  });

  it('shows loading state when loading prop is true', () => {
    render(<RiskPanelDashboard riskPanel={mockRiskPanel} loading={true} />);

    // Should show multiple "Computing..." texts (one per metric)
    const computingTexts = screen.getAllByText('Computing...');
    expect(computingTexts.length).toBe(7); // 7 metrics
  });

  it('displays correct return quality assessment for good omega', () => {
    const goodOmegaPanel = { ...mockRiskPanel, omega: 1.6 };
    const { container } = render(<RiskPanelDashboard riskPanel={goodOmegaPanel} />);

    expect(screen.getByText('Return Quality')).toBeInTheDocument();
    // Check in risk summary section specifically
    const riskSummary = container.querySelector('.grid.grid-cols-1.md\\:grid-cols-3');
    expect(riskSummary).toHaveTextContent('Good');
  });

  it('displays correct return quality assessment for fair omega', () => {
    const fairOmegaPanel = { ...mockRiskPanel, omega: 1.2 };
    const { container } = render(<RiskPanelDashboard riskPanel={fairOmegaPanel} />);

    expect(screen.getByText('Return Quality')).toBeInTheDocument();
    // Check in risk summary section specifically
    const riskSummary = container.querySelector('.grid.grid-cols-1.md\\:grid-cols-3');
    expect(riskSummary).toHaveTextContent('Fair');
  });

  it('displays correct downside protection assessment for good capture', () => {
    const goodDownsidePanel = { ...mockRiskPanel, downside_capture: 60 };
    const { container } = render(<RiskPanelDashboard riskPanel={goodDownsidePanel} />);

    expect(screen.getByText('Downside Protection')).toBeInTheDocument();
    // Check in risk summary section specifically
    const riskSummary = container.querySelector('.grid.grid-cols-1.md\\:grid-cols-3');
    expect(riskSummary).toHaveTextContent('Good');
  });

  it('displays correct downside protection assessment for fair capture', () => {
    const fairDownsidePanel = { ...mockRiskPanel, downside_capture: 85 };
    const { container } = render(<RiskPanelDashboard riskPanel={fairDownsidePanel} />);

    expect(screen.getByText('Downside Protection')).toBeInTheDocument();
    // Check in risk summary section specifically
    const riskSummary = container.querySelector('.grid.grid-cols-1.md\\:grid-cols-3');
    expect(riskSummary).toHaveTextContent('Fair');
  });

  it('displays correct tail risk assessment for low risk', () => {
    const lowRiskPanel = { ...mockRiskPanel, cvar_95: -3.5 };
    render(<RiskPanelDashboard riskPanel={lowRiskPanel} />);

    expect(screen.getByText('Tail Risk')).toBeInTheDocument();
    expect(screen.getByText('Low')).toBeInTheDocument();
  });

  it('displays correct tail risk assessment for high risk', () => {
    const highRiskPanel = { ...mockRiskPanel, cvar_95: -15.0 };
    render(<RiskPanelDashboard riskPanel={highRiskPanel} />);

    expect(screen.getByText('Tail Risk')).toBeInTheDocument();
    expect(screen.getByText('High')).toBeInTheDocument();
  });

  it('applies regime color to regime indicator', () => {
    const { container } = render(<RiskPanelDashboard riskPanel={mockRiskPanel} regime="bull" />);

    // Check for regime indicator with green color (bull)
    const regimeIndicator = container.querySelector('[style*="border-color"]');
    expect(regimeIndicator).toBeInTheDocument();
  });

  it('handles missing explanations gracefully', () => {
    const noExplanationsPanel = { ...mockRiskPanel, explanations: undefined };
    render(<RiskPanelDashboard riskPanel={noExplanationsPanel} />);

    expect(screen.queryByText('Risk Assessment')).not.toBeInTheDocument();
  });
});

