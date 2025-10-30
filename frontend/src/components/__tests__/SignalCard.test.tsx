import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { SignalCard } from '../SignalCard';

describe('SignalCard', () => {
  const defaultProps = {
    title: 'Test Signal',
    value: 0.5,
    type: 'gauge' as const,
    tooltip: 'This is a test tooltip',
    range: [-1, 1] as [number, number],
    thresholds: {
      excellent: 0.5,
      good: 0,
      warning: -0.5,
    },
  };

  it('renders title and value correctly', () => {
    render(<SignalCard {...defaultProps} />);

    expect(screen.getByText('Test Signal')).toBeInTheDocument();
    expect(screen.getByText('0.50')).toBeInTheDocument();
  });

  it('shows loading state when value is null', () => {
    render(<SignalCard {...defaultProps} value={null} />);

    expect(screen.getByText('Computing...')).toBeInTheDocument();
    expect(screen.getByText('Test Signal')).toBeInTheDocument();
  });

  it('shows loading state when value is undefined', () => {
    render(<SignalCard {...defaultProps} value={undefined} />);

    expect(screen.getByText('Computing...')).toBeInTheDocument();
  });

  it('displays tooltip on hover', () => {
    render(<SignalCard {...defaultProps} />);

    const infoButton = screen.getByLabelText('More information');
    fireEvent.mouseEnter(infoButton);

    expect(screen.getByText('This is a test tooltip')).toBeInTheDocument();
  });

  it('hides tooltip on mouse leave', () => {
    render(<SignalCard {...defaultProps} />);

    const infoButton = screen.getByLabelText('More information');
    fireEvent.mouseEnter(infoButton);
    expect(screen.getByText('This is a test tooltip')).toBeInTheDocument();

    fireEvent.mouseLeave(infoButton);
    expect(screen.queryByText('This is a test tooltip')).not.toBeInTheDocument();
  });

  describe('Gauge type', () => {
    it('renders gauge visualization', () => {
      const { container } = render(<SignalCard {...defaultProps} type="gauge" value={0.65} />);

      const svg = container.querySelector('svg');
      expect(svg).toBeInTheDocument();
    });

    it('shows correct interpretation for strong bullish', () => {
      render(<SignalCard {...defaultProps} type="gauge" value={0.75} />);
      expect(screen.getByText('Strong Bullish')).toBeInTheDocument();
    });

    it('shows correct interpretation for mild bullish', () => {
      render(<SignalCard {...defaultProps} type="gauge" value={0.25} />);
      expect(screen.getByText('Mild Bullish')).toBeInTheDocument();
    });

    it('shows correct interpretation for mild bearish', () => {
      render(<SignalCard {...defaultProps} type="gauge" value={-0.25} />);
      expect(screen.getByText('Mild Bearish')).toBeInTheDocument();
    });

    it('shows correct interpretation for strong bearish', () => {
      render(<SignalCard {...defaultProps} type="gauge" value={-0.75} />);
      expect(screen.getByText('Strong Bearish')).toBeInTheDocument();
    });
  });

  describe('Z-score type', () => {
    const zscoreProps = {
      ...defaultProps,
      type: 'zscore' as const,
      range: [-3, 3] as [number, number],
      thresholds: {
        excellent: 2,
        good: 1,
        warning: -1,
      },
    };

    it('formats value with sigma symbol', () => {
      render(<SignalCard {...zscoreProps} value={1.85} />);
      expect(screen.getByText('1.85σ')).toBeInTheDocument();
    });

    it('shows correct interpretation for strong outperformance', () => {
      render(<SignalCard {...zscoreProps} value={2.5} />);
      expect(screen.getByText('Strong Outperformance')).toBeInTheDocument();
    });

    it('shows correct interpretation for mild outperformance', () => {
      render(<SignalCard {...zscoreProps} value={1.5} />);
      expect(screen.getByText('Mild Outperformance')).toBeInTheDocument();
    });

    it('shows correct interpretation for neutral', () => {
      render(<SignalCard {...zscoreProps} value={0.5} />);
      expect(screen.getByText('Neutral')).toBeInTheDocument();
    });

    it('shows correct interpretation for mild underperformance', () => {
      render(<SignalCard {...zscoreProps} value={-1.5} />);
      expect(screen.getByText('Mild Underperformance')).toBeInTheDocument();
    });

    it('shows correct interpretation for strong underperformance', () => {
      render(<SignalCard {...zscoreProps} value={-2.5} />);
      expect(screen.getByText('Strong Underperformance')).toBeInTheDocument();
    });

    it('renders progress bar for zscore type', () => {
      const { container } = render(<SignalCard {...zscoreProps} value={1.5} />);
      const progressBar = container.querySelector('.h-2.bg-gray-700');
      expect(progressBar).toBeInTheDocument();
    });
  });

  describe('Percentage type', () => {
    const percentageProps = {
      ...defaultProps,
      type: 'percentage' as const,
      range: [0, 1] as [number, number],
      thresholds: {
        excellent: 0.6,
        good: 0.4,
        warning: 0.2,
      },
    };

    it('formats value as percentage', () => {
      render(<SignalCard {...percentageProps} value={0.78} />);
      expect(screen.getByText('78%')).toBeInTheDocument();
    });

    it('shows correct interpretation for strong', () => {
      render(<SignalCard {...percentageProps} value={0.75} />);
      expect(screen.getByText('Strong')).toBeInTheDocument();
    });

    it('shows correct interpretation for moderate', () => {
      render(<SignalCard {...percentageProps} value={0.5} />);
      expect(screen.getByText('Moderate')).toBeInTheDocument();
    });

    it('shows correct interpretation for weak', () => {
      render(<SignalCard {...percentageProps} value={0.3} />);
      expect(screen.getByText('Weak')).toBeInTheDocument();
    });

    it('shows correct interpretation for very weak', () => {
      render(<SignalCard {...percentageProps} value={0.1} />);
      expect(screen.getByText('Very Weak')).toBeInTheDocument();
    });

    it('renders progress bar for percentage type', () => {
      const { container } = render(<SignalCard {...percentageProps} value={0.78} />);
      const progressBar = container.querySelector('.h-2.bg-gray-700');
      expect(progressBar).toBeInTheDocument();
    });
  });

  describe('Color coding', () => {
    it('shows green color for excellent values', () => {
      const { container } = render(<SignalCard {...defaultProps} value={0.75} />);
      const valueElement = container.querySelector('.text-3xl');
      expect(valueElement).toHaveStyle({ color: '#10b981' });
    });

    it('shows light green color for good values', () => {
      const { container } = render(<SignalCard {...defaultProps} value={0.25} />);
      const valueElement = container.querySelector('.text-3xl');
      expect(valueElement).toHaveStyle({ color: '#84cc16' });
    });

    it('shows orange color for warning values', () => {
      const { container } = render(<SignalCard {...defaultProps} value={-0.25} />);
      const valueElement = container.querySelector('.text-3xl');
      expect(valueElement).toHaveStyle({ color: '#f59e0b' });
    });

    it('shows red color for critical values', () => {
      const { container } = render(<SignalCard {...defaultProps} value={-0.75} />);
      const valueElement = container.querySelector('.text-3xl');
      expect(valueElement).toHaveStyle({ color: '#ef4444' });
    });
  });

  describe('Icons', () => {
    it('shows trending up icon for positive values', () => {
      const { container } = render(<SignalCard {...defaultProps} value={0.5} />);
      const icon = container.querySelector('svg.lucide-trending-up');
      expect(icon).toBeInTheDocument();
    });

    it('shows trending down icon for negative values', () => {
      const { container } = render(<SignalCard {...defaultProps} value={-0.5} />);
      const icon = container.querySelector('svg.lucide-trending-down');
      expect(icon).toBeInTheDocument();
    });
  });

  it('displays tooltip text at bottom of card', () => {
    render(<SignalCard {...defaultProps} />);
    expect(screen.getByText(/💡 This is a test tooltip/)).toBeInTheDocument();
  });

  it('applies hover effect to card', () => {
    const { container } = render(<SignalCard {...defaultProps} />);
    const card = container.querySelector('.hover\\:border-\\[\\#505050\\]');
    expect(card).toBeInTheDocument();
  });
});

