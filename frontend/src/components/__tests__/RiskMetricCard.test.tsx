import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { RiskMetricCard } from '../RiskMetricCard';

describe('RiskMetricCard', () => {
  const defaultProps = {
    title: 'Test Metric',
    value: 1.5,
    format: 'ratio' as const,
    tooltip: 'This is a test tooltip',
    thresholds: {
      excellent: 2.0,
      good: 1.5,
      warning: 1.0,
    },
    higherIsBetter: true,
  };

  it('renders title and value correctly', () => {
    render(<RiskMetricCard {...defaultProps} />);

    expect(screen.getByText('Test Metric')).toBeInTheDocument();
    expect(screen.getByText('1.50')).toBeInTheDocument();
  });

  it('shows loading state when loading prop is true', () => {
    render(<RiskMetricCard {...defaultProps} loading={true} />);

    expect(screen.getByText('Computing...')).toBeInTheDocument();
    expect(screen.getByText('Test Metric')).toBeInTheDocument();
  });

  it('displays tooltip on hover', () => {
    render(<RiskMetricCard {...defaultProps} />);

    const infoButton = screen.getByLabelText('More information');
    fireEvent.mouseEnter(infoButton);

    expect(screen.getByText('This is a test tooltip')).toBeInTheDocument();
  });

  it('hides tooltip on mouse leave', () => {
    render(<RiskMetricCard {...defaultProps} />);

    const infoButton = screen.getByLabelText('More information');
    fireEvent.mouseEnter(infoButton);
    expect(screen.getByText('This is a test tooltip')).toBeInTheDocument();

    fireEvent.mouseLeave(infoButton);
    expect(screen.queryByText('This is a test tooltip')).not.toBeInTheDocument();
  });

  describe('Higher is better logic', () => {
    it('shows Excellent for value >= excellent threshold', () => {
      const { container } = render(<RiskMetricCard {...defaultProps} value={2.5} />);
      // Check in performance level div specifically (not threshold labels)
      const performanceDiv = container.querySelector('.text-xs.text-gray-400.flex.items-center');
      expect(performanceDiv).toHaveTextContent('Excellent');
    });

    it('shows Good for value >= good threshold', () => {
      const { container } = render(<RiskMetricCard {...defaultProps} value={1.7} />);
      const performanceDiv = container.querySelector('.text-xs.text-gray-400.flex.items-center');
      expect(performanceDiv).toHaveTextContent('Good');
    });

    it('shows Fair for value >= warning threshold', () => {
      const { container } = render(<RiskMetricCard {...defaultProps} value={1.2} />);
      const performanceDiv = container.querySelector('.text-xs.text-gray-400.flex.items-center');
      expect(performanceDiv).toHaveTextContent('Fair');
    });

    it('shows Poor for value < warning threshold', () => {
      const { container } = render(<RiskMetricCard {...defaultProps} value={0.8} />);
      const performanceDiv = container.querySelector('.text-xs.text-gray-400.flex.items-center');
      expect(performanceDiv).toHaveTextContent('Poor');
    });
  });

  describe('Lower is better logic', () => {
    const lowerIsBetterProps = {
      ...defaultProps,
      higherIsBetter: false,
      thresholds: {
        excellent: 5,
        good: 10,
        warning: 20,
      },
    };

    it('shows Excellent for value <= excellent threshold', () => {
      render(<RiskMetricCard {...lowerIsBetterProps} value={3} />);
      expect(screen.getByText('Excellent')).toBeInTheDocument();
    });

    it('shows Good for value <= good threshold', () => {
      render(<RiskMetricCard {...lowerIsBetterProps} value={8} />);
      expect(screen.getByText('Good')).toBeInTheDocument();
    });

    it('shows Fair for value <= warning threshold', () => {
      render(<RiskMetricCard {...lowerIsBetterProps} value={15} />);
      expect(screen.getByText('Fair')).toBeInTheDocument();
    });

    it('shows Poor for value > warning threshold', () => {
      render(<RiskMetricCard {...lowerIsBetterProps} value={25} />);
      expect(screen.getByText('Poor')).toBeInTheDocument();
    });
  });

  describe('Format types', () => {
    it('formats ratio values correctly', () => {
      render(<RiskMetricCard {...defaultProps} format="ratio" value={1.234} />);
      expect(screen.getByText('1.23')).toBeInTheDocument();
    });

    it('formats percentage values correctly', () => {
      render(<RiskMetricCard {...defaultProps} format="percentage" value={45.678} />);
      expect(screen.getByText('45.7%')).toBeInTheDocument();
    });
  });

  describe('Regime styling', () => {
    it('applies bull regime styling', () => {
      const { container } = render(<RiskMetricCard {...defaultProps} regime="bull" />);
      const regimeBorder = container.querySelector('[style*="border-color"]');
      expect(regimeBorder).toBeInTheDocument();
    });

    it('applies bear regime styling', () => {
      const { container } = render(<RiskMetricCard {...defaultProps} regime="bear" />);
      const regimeBorder = container.querySelector('[style*="border-color"]');
      expect(regimeBorder).toBeInTheDocument();
    });

    it('applies neutral regime styling by default', () => {
      const { container } = render(<RiskMetricCard {...defaultProps} />);
      const regimeBorder = container.querySelector('[style*="border-color"]');
      expect(regimeBorder).toBeInTheDocument();
    });
  });

  describe('Color coding', () => {
    it('shows green color for excellent values (higher is better)', () => {
      const { container } = render(<RiskMetricCard {...defaultProps} value={2.5} />);
      const valueElement = container.querySelector('.text-2xl');
      expect(valueElement).toHaveStyle({ color: '#10b981' });
    });

    it('shows light green color for good values (higher is better)', () => {
      const { container } = render(<RiskMetricCard {...defaultProps} value={1.7} />);
      const valueElement = container.querySelector('.text-2xl');
      expect(valueElement).toHaveStyle({ color: '#84cc16' });
    });

    it('shows orange color for fair values (higher is better)', () => {
      const { container } = render(<RiskMetricCard {...defaultProps} value={1.2} />);
      const valueElement = container.querySelector('.text-2xl');
      expect(valueElement).toHaveStyle({ color: '#f59e0b' });
    });

    it('shows red color for poor values (higher is better)', () => {
      const { container } = render(<RiskMetricCard {...defaultProps} value={0.8} />);
      const valueElement = container.querySelector('.text-2xl');
      expect(valueElement).toHaveStyle({ color: '#ef4444' });
    });

    it('shows green color for excellent values (lower is better)', () => {
      const lowerIsBetterProps = {
        ...defaultProps,
        higherIsBetter: false,
        thresholds: { excellent: 5, good: 10, warning: 20 },
      };
      const { container } = render(<RiskMetricCard {...lowerIsBetterProps} value={3} />);
      const valueElement = container.querySelector('.text-2xl');
      expect(valueElement).toHaveStyle({ color: '#10b981' });
    });
  });

  describe('Trend icons', () => {
    it('shows trending up icon for good performance (higher is better)', () => {
      const { container } = render(<RiskMetricCard {...defaultProps} value={2.0} />);
      const icon = container.querySelector('svg.lucide-trending-up');
      expect(icon).toBeInTheDocument();
    });

    it('shows trending down icon for poor performance (higher is better)', () => {
      const { container } = render(<RiskMetricCard {...defaultProps} value={0.5} />);
      const icon = container.querySelector('svg.lucide-trending-down');
      expect(icon).toBeInTheDocument();
    });

    it('shows minus icon for fair performance (higher is better)', () => {
      const { container } = render(<RiskMetricCard {...defaultProps} value={1.2} />);
      const icon = container.querySelector('svg.lucide-minus');
      expect(icon).toBeInTheDocument();
    });

    it('shows trending up icon for good performance (lower is better)', () => {
      const lowerIsBetterProps = {
        ...defaultProps,
        higherIsBetter: false,
        thresholds: { excellent: 5, good: 10, warning: 20 },
      };
      const { container } = render(<RiskMetricCard {...lowerIsBetterProps} value={8} />);
      const icon = container.querySelector('svg.lucide-trending-up');
      expect(icon).toBeInTheDocument();
    });
  });

  describe('Progress bar', () => {
    it('renders progress bar', () => {
      const { container } = render(<RiskMetricCard {...defaultProps} />);
      const progressBar = container.querySelector('.h-1\\.5.bg-gray-700');
      expect(progressBar).toBeInTheDocument();
    });

    it('progress bar has correct color', () => {
      const { container } = render(<RiskMetricCard {...defaultProps} value={2.5} />);
      const progressFill = container.querySelector('.h-full.transition-all');
      expect(progressFill).toHaveStyle({ backgroundColor: '#10b981' });
    });
  });

  describe('Threshold labels', () => {
    it('shows correct labels for higher is better', () => {
      render(<RiskMetricCard {...defaultProps} higherIsBetter={true} />);
      expect(screen.getByText('Poor')).toBeInTheDocument();
      expect(screen.getByText('Excellent')).toBeInTheDocument();
    });

    it('shows correct labels for lower is better', () => {
      render(<RiskMetricCard {...defaultProps} higherIsBetter={false} />);
      expect(screen.getByText('High Risk')).toBeInTheDocument();
      expect(screen.getByText('Low Risk')).toBeInTheDocument();
    });
  });

  it('applies hover effect to card', () => {
    const { container } = render(<RiskMetricCard {...defaultProps} />);
    const card = container.querySelector('.hover\\:border-\\[\\#505050\\]');
    expect(card).toBeInTheDocument();
  });
});

