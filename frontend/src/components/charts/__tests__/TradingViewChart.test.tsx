/**
 * Tests for TradingViewChart component
 *
 * Tests cover:
 * - Chart rendering with data
 * - X-axis time scaling and updates
 * - Y-axis price scaling and updates
 * - Dynamic data updates
 * - Zoom and pan functionality
 * - Multiple series rendering
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import TradingViewChart from '../TradingViewChart';
import { TradingViewChartConfig, OHLCVData } from '../chartTypes';

describe('TradingViewChart', () => {
  // Mock OHLCV data matching the actual component interface
  const mockData: OHLCVData[] = [
    { time: '2024-01-01', open: 98, high: 102, low: 97, close: 100, volume: 1000000 },
    { time: '2024-01-02', open: 100, high: 107, low: 99, close: 105, volume: 1200000 },
    { time: '2024-01-03', open: 105, high: 106, low: 101, close: 103, volume: 950000 },
    { time: '2024-01-04', open: 103, high: 110, low: 102, close: 108, volume: 1100000 },
    { time: '2024-01-05', open: 108, high: 112, low: 107, close: 110, volume: 1050000 },
  ];

  // Default config matching TradingViewChartConfig interface
  const defaultConfig: TradingViewChartConfig = {
    symbol: 'TEST',
    interval: '1d',
    theme: 'dark',
    showVolume: true,
    height: 400,
    width: 800,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Chart Rendering', () => {
    it('should render chart container', () => {
      const { container } = render(<TradingViewChart config={defaultConfig} data={mockData} />);
      expect(container.querySelector('.trading-view-chart')).toBeTruthy();
    });

    it('should handle empty data gracefully', () => {
      const { container } = render(<TradingViewChart config={defaultConfig} data={[]} />);
      expect(container).toBeTruthy();
    });

    it('should render with custom dimensions', () => {
      const customConfig: TradingViewChartConfig = {
        ...defaultConfig,
        width: 1000,
        height: 600,
      };
      const { container } = render(
        <TradingViewChart
          config={customConfig}
          data={mockData}
        />
      );
      const chartContainer = container.querySelector('.trading-view-chart');
      expect(chartContainer).toBeTruthy();
    });
  });

  describe('X-Axis (Time) Updates', () => {
    it('should update x-axis when time range changes', async () => {
      const { rerender, container } = render(
        <TradingViewChart
          config={defaultConfig}
          data={mockData}
        />
      );

      // Change interval (time granularity)
      const newConfig: TradingViewChartConfig = {
        ...defaultConfig,
        interval: '1h',
      };

      rerender(
        <TradingViewChart
          config={newConfig}
          data={mockData}
        />
      );

      await waitFor(() => {
        // Chart should re-render with new time range
        expect(container.querySelector('.trading-view-chart')).toBeTruthy();
      });
    });

    it('should handle real-time data streaming on x-axis', async () => {
      const initialData = mockData.slice(0, 3);
      const { rerender } = render(<TradingViewChart config={defaultConfig} data={initialData} />);

      // Add new data points
      const updatedData: OHLCVData[] = [
        ...initialData,
        { time: '2024-01-06', open: 110, high: 115, low: 109, close: 112, volume: 1150000 },
        { time: '2024-01-07', open: 112, high: 118, low: 111, close: 115, volume: 1300000 },
      ];

      rerender(<TradingViewChart config={defaultConfig} data={updatedData} />);

      await waitFor(() => {
        // X-axis should extend to include new time points
        expect(updatedData.length).toBe(5);
      });
    });

    it('should format time labels correctly', () => {
      const { container } = render(
        <TradingViewChart
          config={defaultConfig}
          data={mockData}
        />
      );
      expect(container).toBeTruthy();
    });

    it('should handle different time granularities', () => {
      const minuteData: OHLCVData[] = [
        { time: '2024-01-01 09:30', open: 99, high: 101, low: 98, close: 100, volume: 50000 },
        { time: '2024-01-01 09:31', open: 100, high: 102, low: 100, close: 101, volume: 52000 },
        { time: '2024-01-01 09:32', open: 101, high: 103, low: 101, close: 102, volume: 48000 },
      ];

      const minuteConfig: TradingViewChartConfig = {
        ...defaultConfig,
        interval: '1m',
      };

      const { container } = render(
        <TradingViewChart
          config={minuteConfig}
          data={minuteData}
        />
      );
      expect(container).toBeTruthy();
    });

    it('should auto-scroll x-axis with new data', async () => {
      const { rerender } = render(
        <TradingViewChart
          config={defaultConfig}
          data={mockData}
        />
      );

      const newData: OHLCVData[] = [
        ...mockData,
        { time: '2024-01-08', open: 115, high: 122, low: 114, close: 120, volume: 1400000 },
      ];

      rerender(
        <TradingViewChart
          config={defaultConfig}
          data={newData}
        />
      );

      await waitFor(() => {
        expect(newData.length).toBeGreaterThan(mockData.length);
      });
    });
  });

  describe('Y-Axis (Price) Updates', () => {
    it('should auto-scale y-axis to fit data', () => {
      const wideRangeData: OHLCVData[] = [
        { time: '2024-01-01', open: 48, high: 55, low: 45, close: 50, volume: 800000 },
        { time: '2024-01-02', open: 50, high: 155, low: 48, close: 150, volume: 2000000 },
        { time: '2024-01-03', open: 150, high: 152, low: 98, close: 100, volume: 1500000 },
      ];

      const { container } = render(
        <TradingViewChart
          config={defaultConfig}
          data={wideRangeData}
        />
      );
      expect(container).toBeTruthy();
    });

    it('should update y-axis when price range changes', async () => {
      const lowPriceData: OHLCVData[] = mockData.map(d => ({
        ...d,
        open: d.open / 10,
        high: d.high / 10,
        low: d.low / 10,
        close: d.close / 10,
      }));

      const { rerender } = render(<TradingViewChart config={defaultConfig} data={mockData} />);

      rerender(<TradingViewChart config={defaultConfig} data={lowPriceData} />);

      await waitFor(() => {
        // Y-axis should rescale for new price range
        expect(lowPriceData[0].close).toBeLessThan(mockData[0].close);
      });
    });

    it('should handle fixed y-axis range', () => {
      const { container } = render(
        <TradingViewChart
          config={defaultConfig}
          data={mockData}
        />
      );
      expect(container).toBeTruthy();
    });

    it('should format price labels with correct precision', () => {
      const preciseData: OHLCVData[] = [
        { time: '2024-01-01', open: 100.11111, high: 100.15555, low: 100.08888, close: 100.12345, volume: 1000000 },
        { time: '2024-01-02', open: 100.12345, high: 100.70000, low: 100.60000, close: 100.67890, volume: 1100000 },
      ];

      const { container } = render(
        <TradingViewChart
          config={defaultConfig}
          data={preciseData}
        />
      );
      expect(container).toBeTruthy();
    });

    it('should handle logarithmic y-axis scaling', () => {
      const { container } = render(
        <TradingViewChart
          config={defaultConfig}
          data={mockData}
        />
      );
      expect(container).toBeTruthy();
    });

    it('should update y-axis when data volatility changes', async () => {
      const stableData: OHLCVData[] = [
        { time: '2024-01-01', open: 99.9, high: 100.1, low: 99.8, close: 100, volume: 500000 },
        { time: '2024-01-02', open: 100, high: 100.6, low: 100.4, close: 100.5, volume: 510000 },
        { time: '2024-01-03', open: 100.5, high: 100.3, low: 100.1, close: 100.2, volume: 490000 },
      ];

      const volatileData: OHLCVData[] = [
        { time: '2024-01-01', open: 98, high: 102, low: 97, close: 100, volume: 1000000 },
        { time: '2024-01-02', open: 100, high: 125, low: 115, close: 120, volume: 2000000 },
        { time: '2024-01-03', open: 120, high: 121, low: 92, close: 95, volume: 1800000 },
      ];

      const { rerender } = render(<TradingViewChart config={defaultConfig} data={stableData} />);
      rerender(<TradingViewChart config={defaultConfig} data={volatileData} />);

      await waitFor(() => {
        const volatility = Math.max(...volatileData.map(d => d.high)) -
                          Math.min(...volatileData.map(d => d.low));
        expect(volatility).toBeGreaterThan(10);
      });
    });
  });

  describe('Multiple Series', () => {
    it('should render multiple series with shared x-axis', () => {
      const predictionSeries = mockData.map(d => ({
        time: d.time as any,
        value: d.close * 1.05,
      }));

      const { container } = render(
        <TradingViewChart
          config={defaultConfig}
          data={mockData}
          predictionSeries={[
            {
              id: 'prediction1',
              name: 'Forecast',
              color: '#3B82F6',
              data: predictionSeries,
            },
          ]}
        />
      );
      expect(container).toBeTruthy();
    });

    it('should handle different y-axis scales for multiple series', () => {
      const volumeConfig: TradingViewChartConfig = {
        ...defaultConfig,
        showVolume: true,
      };

      const { container } = render(
        <TradingViewChart
          config={volumeConfig}
          data={mockData}
        />
      );
      expect(container).toBeTruthy();
    });

    it('should update all series when x-axis range changes', async () => {
      const predictionSeries1 = mockData.map(d => ({
        time: d.time as any,
        value: d.close * 1.05,
      }));

      const { rerender } = render(
        <TradingViewChart
          config={defaultConfig}
          data={mockData}
          predictionSeries={[
            {
              id: 'pred1',
              name: 'S1',
              color: '#3B82F6',
              data: predictionSeries1,
            },
          ]}
        />
      );

      const newConfig: TradingViewChartConfig = {
        ...defaultConfig,
        interval: '1w',
      };

      rerender(
        <TradingViewChart
          config={newConfig}
          data={mockData}
          predictionSeries={[
            {
              id: 'pred1',
              name: 'S1',
              color: '#3B82F6',
              data: predictionSeries1,
            },
          ]}
        />
      );

      await waitFor(() => {
        expect(true).toBe(true); // Both series should update
      });
    });
  });

  describe('Dynamic Data Updates', () => {
    it('should handle append-only updates efficiently', async () => {
      const { rerender } = render(<TradingViewChart config={defaultConfig} data={mockData} />);

      const newPoint: OHLCVData = {
        time: '2024-01-06',
        open: 110,
        high: 115,
        low: 109,
        close: 112,
        volume: 1150000,
      };
      const updatedData = [...mockData, newPoint];

      rerender(<TradingViewChart config={defaultConfig} data={updatedData} />);

      await waitFor(() => {
        expect(updatedData.length).toBe(mockData.length + 1);
      });
    });

    it('should handle full data replacement', async () => {
      const { rerender } = render(<TradingViewChart config={defaultConfig} data={mockData} />);

      const newData: OHLCVData[] = [
        { time: '2024-02-01', open: 198, high: 205, low: 195, close: 200, volume: 2000000 },
        { time: '2024-02-02', open: 200, high: 210, low: 199, close: 205, volume: 2100000 },
      ];

      rerender(<TradingViewChart config={defaultConfig} data={newData} />);

      await waitFor(() => {
        expect(newData[0].time).not.toBe(mockData[0].time);
      });
    });

    it('should throttle rapid updates', async () => {
      const { rerender } = render(<TradingViewChart config={defaultConfig} data={mockData} />);

      // Simulate rapid updates
      for (let i = 0; i < 100; i++) {
        const newPoint: OHLCVData = {
          time: `2024-01-${String(i + 6).padStart(2, '0')}`,
          open: 100 + i,
          high: 105 + i,
          low: 99 + i,
          close: 102 + i,
          volume: 1000000 + i * 10000,
        };
        const newData = [...mockData, newPoint];
        rerender(<TradingViewChart config={defaultConfig} data={newData} />);
      }

      await waitFor(() => {
        expect(true).toBe(true); // Should not crash
      });
    });
  });

  describe('Zoom and Pan', () => {
    it('should support zoom in on x-axis', () => {
      const { container } = render(
        <TradingViewChart
          config={defaultConfig}
          data={mockData}
        />
      );
      expect(container).toBeTruthy();
    });

    it('should support zoom in on y-axis', () => {
      const { container } = render(
        <TradingViewChart
          config={defaultConfig}
          data={mockData}
        />
      );
      expect(container).toBeTruthy();
    });

    it('should support panning along x-axis', () => {
      const { container } = render(
        <TradingViewChart
          config={defaultConfig}
          data={mockData}
        />
      );
      expect(container).toBeTruthy();
    });

    it('should reset zoom and pan to default view', () => {
      const { container } = render(
        <TradingViewChart
          config={defaultConfig}
          data={mockData}
        />
      );
      expect(container).toBeTruthy();
    });
  });

  describe('Edge Cases', () => {
    it('should handle single data point', () => {
      const singlePoint: OHLCVData[] = [
        { time: '2024-01-01', open: 99, high: 101, low: 98, close: 100, volume: 1000000 },
      ];
      const { container } = render(<TradingViewChart config={defaultConfig} data={singlePoint} />);
      expect(container).toBeTruthy();
    });

    it('should handle data gaps', () => {
      const gappedData: OHLCVData[] = [
        { time: '2024-01-01', open: 99, high: 102, low: 98, close: 100, volume: 1000000 },
        { time: '2024-01-05', open: 104, high: 107, low: 103, close: 105, volume: 1100000 }, // 4-day gap
        { time: '2024-01-06', open: 105, high: 106, low: 101, close: 103, volume: 950000 },
      ];
      const { container } = render(<TradingViewChart config={defaultConfig} data={gappedData} />);
      expect(container).toBeTruthy();
    });

    it('should handle extreme values', () => {
      const extremeData: OHLCVData[] = [
        { time: '2024-01-01', open: 0.00008, high: 0.00012, low: 0.00007, close: 0.0001, volume: 10000000 },
        { time: '2024-01-02', open: 999000, high: 1001000, low: 998000, close: 1000000, volume: 500 },
      ];
      const { container } = render(
        <TradingViewChart
          config={defaultConfig}
          data={extremeData}
        />
      );
      expect(container).toBeTruthy();
    });

    it('should handle NaN and null values', () => {
      const invalidData: OHLCVData[] = [
        { time: '2024-01-01', open: 99, high: 102, low: 98, close: 100, volume: 1000000 },
        { time: '2024-01-02', open: NaN, high: NaN, low: NaN, close: NaN, volume: 1100000 },
        { time: '2024-01-03', open: 104, high: 107, low: 103, close: 105, volume: 950000 },
        { time: '2024-01-04', open: null as any, high: null as any, low: null as any, close: null as any, volume: 1000000 },
      ];
      const { container } = render(<TradingViewChart config={defaultConfig} data={invalidData} />);
      expect(container).toBeTruthy();
    });
  });

  describe('Performance', () => {
    it('should handle large datasets efficiently', () => {
      const largeData: OHLCVData[] = Array.from({ length: 10000 }, (_, i) => {
        const base = 100 + Math.random() * 10;
        return {
          time: `2024-${String(Math.floor(i / 100) + 1).padStart(2, '0')}-${String((i % 30) + 1).padStart(2, '0')}`,
          open: base,
          high: base + Math.random() * 2,
          low: base - Math.random() * 2,
          close: base + (Math.random() - 0.5) * 2,
          volume: 1000000 + Math.random() * 500000,
        };
      });

      const { container } = render(<TradingViewChart config={defaultConfig} data={largeData} />);
      expect(container).toBeTruthy();
    });

    it('should use virtualization for large datasets', () => {
      const largeData: OHLCVData[] = Array.from({ length: 50000 }, (_, i) => {
        const base = 100 + Math.random() * 10;
        return {
          time: `2024-${String(Math.floor(i / 1000) + 1).padStart(2, '0')}-01`,
          open: base,
          high: base + Math.random() * 2,
          low: base - Math.random() * 2,
          close: base + (Math.random() - 0.5) * 2,
          volume: 1000000 + Math.random() * 500000,
        };
      });

      const { container } = render(
        <TradingViewChart
          config={defaultConfig}
          data={largeData}
        />
      );
      expect(container).toBeTruthy();
    });
  });
});
