/**
 * Portfolio P&L and Drawdown Visualization
 *
 * Professional portfolio performance tracking with:
 * - Cumulative P&L over time
 * - Drawdown analysis
 * - Sharpe ratio calculation
 * - Win rate and max drawdown stats
 * - Benchmark comparison
 *
 * Phase 2 - Portfolio Analytics Visualization
 */

import React, { useMemo } from 'react';
import { CandlestickChart, OHLCVData } from './index';
import { Box, Typography, Grid, Chip } from '@mui/material';

export interface PortfolioSnapshot {
  timestamp: string | Date;
  portfolio_value: number;
  cash: number;
  positions_value: number;
  daily_pnl?: number;
  cumulative_pnl?: number;
  benchmark_value?: number; // S&P 500 or other benchmark
}

export interface PortfolioPnLChartProps {
  snapshots: PortfolioSnapshot[];
  initialValue: number;
  theme?: 'dark' | 'light';
  showDrawdown?: boolean;
  showBenchmark?: boolean;
  height?: number;
  className?: string;
}

/**
 * Portfolio P&L Chart
 *
 * Usage:
 * ```tsx
 * <PortfolioPnLChart
 *   snapshots={[
 *     { timestamp: '2024-01-01', portfolio_value: 100000, cash: 50000, positions_value: 50000 },
 *     { timestamp: '2024-01-02', portfolio_value: 101000, cash: 50000, positions_value: 51000 }
 *   ]}
 *   initialValue={100000}
 *   showDrawdown={true}
 *   showBenchmark={true}
 * />
 * ```
 */
const PortfolioPnLChart: React.FC<PortfolioPnLChartProps> = ({
  snapshots,
  initialValue,
  theme = 'dark',
  showDrawdown = true,
  showBenchmark = false,
  height = 600,
  className = '',
}) => {
  /**
   * Calculate performance metrics
   */
  const metrics = useMemo(() => {
    if (snapshots.length === 0) return null;

    const currentValue = snapshots[snapshots.length - 1].portfolio_value;
    const totalReturn = currentValue - initialValue;
    const totalReturnPct = (totalReturn / initialValue) * 100;

    // Calculate daily returns
    const dailyReturns: number[] = [];
    for (let i = 1; i < snapshots.length; i++) {
      const prevValue = snapshots[i - 1].portfolio_value;
      const currValue = snapshots[i].portfolio_value;
      const dailyReturn = (currValue - prevValue) / prevValue;
      dailyReturns.push(dailyReturn);
    }

    // Average daily return
    const avgDailyReturn = dailyReturns.reduce((sum, r) => sum + r, 0) / dailyReturns.length;

    // Standard deviation of returns
    const variance =
      dailyReturns.reduce((sum, r) => sum + Math.pow(r - avgDailyReturn, 2), 0) / dailyReturns.length;
    const stdDev = Math.sqrt(variance);

    // Sharpe ratio (annualized, assuming 0% risk-free rate)
    const sharpeRatio = (avgDailyReturn * Math.sqrt(252)) / (stdDev || 1);

    // Max drawdown
    let peak = initialValue;
    let maxDrawdown = 0;
    let maxDrawdownDate = '';

    snapshots.forEach((snapshot) => {
      const value = snapshot.portfolio_value;
      if (value > peak) {
        peak = value;
      }
      const drawdown = (peak - value) / peak;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
        maxDrawdownDate = typeof snapshot.timestamp === 'string'
          ? snapshot.timestamp
          : snapshot.timestamp.toISOString().split('T')[0];
      }
    });

    // Win rate
    const positiveReturns = dailyReturns.filter((r) => r > 0).length;
    const winRate = (positiveReturns / dailyReturns.length) * 100;

    // Benchmark comparison (if available)
    let benchmarkReturn = 0;
    if (showBenchmark && snapshots[0].benchmark_value && snapshots[snapshots.length - 1].benchmark_value) {
      const benchmarkInitial = snapshots[0].benchmark_value;
      const benchmarkFinal = snapshots[snapshots.length - 1].benchmark_value;
      benchmarkReturn = ((benchmarkFinal - benchmarkInitial) / benchmarkInitial) * 100;
    }

    return {
      currentValue,
      totalReturn,
      totalReturnPct,
      avgDailyReturn: avgDailyReturn * 100,
      sharpeRatio,
      maxDrawdown: maxDrawdown * 100,
      maxDrawdownDate,
      winRate,
      benchmarkReturn,
      outperformance: totalReturnPct - benchmarkReturn,
    };
  }, [snapshots, initialValue, showBenchmark]);

  /**
   * Convert portfolio snapshots to candlestick chart data
   */
  const chartData = useMemo<OHLCVData[]>(() => {
    return snapshots.map((snapshot, idx) => {
      const time = typeof snapshot.timestamp === 'string'
        ? snapshot.timestamp
        : snapshot.timestamp.toISOString().split('T')[0];

      // For portfolio, we use same value for open/close
      // High/low represent intraday if available, otherwise same as close
      const value = snapshot.portfolio_value;

      return {
        time,
        open: value,
        high: value * 1.005, // Placeholder: +0.5% intraday high
        low: value * 0.995, // Placeholder: -0.5% intraday low
        close: value,
        volume: Math.abs(snapshot.daily_pnl || 0), // Use daily P&L as "volume"
      };
    });
  }, [snapshots]);

  /**
   * Calculate drawdown series for visualization
   */
  const drawdownData = useMemo(() => {
    if (!showDrawdown) return [];

    let peak = initialValue;
    const drawdowns: { time: string; drawdown: number }[] = [];

    snapshots.forEach((snapshot) => {
      const value = snapshot.portfolio_value;
      if (value > peak) {
        peak = value;
      }
      const drawdown = ((peak - value) / peak) * 100;

      const time = typeof snapshot.timestamp === 'string'
        ? snapshot.timestamp
        : snapshot.timestamp.toISOString().split('T')[0];

      drawdowns.push({ time, drawdown });
    });

    return drawdowns;
  }, [snapshots, initialValue, showDrawdown]);

  if (!metrics) {
    return (
      <div className={`portfolio-pnl-chart ${className}`}>
        <Typography>No portfolio data available</Typography>
      </div>
    );
  }

  return (
    <div className={`portfolio-pnl-chart ${className}`}>
      {/* Performance Stats Header */}
      <Box
        sx={{
          padding: '16px',
          backgroundColor: theme === 'dark' ? '#1e222d' : '#f0f3fa',
          color: theme === 'dark' ? '#d1d4dc' : '#191919',
          borderRadius: '8px 8px 0 0',
          marginBottom: '2px',
        }}
      >
        <Typography variant="h6" gutterBottom>
          Portfolio Performance
        </Typography>

        <Grid container spacing={2}>
          {/* Current Value */}
          <Grid item xs={12} sm={6} md={3}>
            <div style={{ opacity: 0.7, fontSize: '12px' }}>Current Value</div>
            <div style={{ fontSize: '24px', fontWeight: 600 }}>
              ${metrics.currentValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
          </Grid>

          {/* Total Return */}
          <Grid item xs={12} sm={6} md={3}>
            <div style={{ opacity: 0.7, fontSize: '12px' }}>Total Return</div>
            <div
              style={{
                fontSize: '24px',
                fontWeight: 600,
                color: metrics.totalReturn >= 0 ? '#26a69a' : '#ef5350',
              }}
            >
              {metrics.totalReturn >= 0 ? '+' : ''}$
              {metrics.totalReturn.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
            <div style={{ fontSize: '12px', opacity: 0.8 }}>
              {metrics.totalReturnPct >= 0 ? '+' : ''}
              {metrics.totalReturnPct.toFixed(2)}%
            </div>
          </Grid>

          {/* Sharpe Ratio */}
          <Grid item xs={12} sm={6} md={3}>
            <div style={{ opacity: 0.7, fontSize: '12px' }}>Sharpe Ratio</div>
            <div style={{ fontSize: '24px', fontWeight: 600 }}>
              {metrics.sharpeRatio.toFixed(2)}
            </div>
            <div style={{ fontSize: '11px', opacity: 0.7 }}>
              {metrics.sharpeRatio > 2 ? 'Excellent' : metrics.sharpeRatio > 1 ? 'Good' : 'Below Target'}
            </div>
          </Grid>

          {/* Max Drawdown */}
          <Grid item xs={12} sm={6} md={3}>
            <div style={{ opacity: 0.7, fontSize: '12px' }}>Max Drawdown</div>
            <div style={{ fontSize: '24px', fontWeight: 600, color: '#ef5350' }}>
              -{metrics.maxDrawdown.toFixed(2)}%
            </div>
            <div style={{ fontSize: '11px', opacity: 0.7 }}>{metrics.maxDrawdownDate}</div>
          </Grid>

          {/* Win Rate */}
          <Grid item xs={12} sm={6} md={3}>
            <div style={{ opacity: 0.7, fontSize: '12px' }}>Win Rate</div>
            <div style={{ fontSize: '20px', fontWeight: 600 }}>{metrics.winRate.toFixed(1)}%</div>
          </Grid>

          {/* Avg Daily Return */}
          <Grid item xs={12} sm={6} md={3}>
            <div style={{ opacity: 0.7, fontSize: '12px' }}>Avg Daily Return</div>
            <div
              style={{
                fontSize: '20px',
                fontWeight: 600,
                color: metrics.avgDailyReturn >= 0 ? '#26a69a' : '#ef5350',
              }}
            >
              {metrics.avgDailyReturn >= 0 ? '+' : ''}
              {metrics.avgDailyReturn.toFixed(3)}%
            </div>
          </Grid>

          {/* Benchmark Comparison */}
          {showBenchmark && metrics.benchmarkReturn !== 0 && (
            <>
              <Grid item xs={12} sm={6} md={3}>
                <div style={{ opacity: 0.7, fontSize: '12px' }}>Benchmark Return</div>
                <div style={{ fontSize: '20px', fontWeight: 600 }}>
                  {metrics.benchmarkReturn >= 0 ? '+' : ''}
                  {metrics.benchmarkReturn.toFixed(2)}%
                </div>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <div style={{ opacity: 0.7, fontSize: '12px' }}>Outperformance</div>
                <div
                  style={{
                    fontSize: '20px',
                    fontWeight: 600,
                    color: metrics.outperformance >= 0 ? '#26a69a' : '#ef5350',
                  }}
                >
                  {metrics.outperformance >= 0 ? '+' : ''}
                  {metrics.outperformance.toFixed(2)}%
                </div>
              </Grid>
            </>
          )}
        </Grid>
      </Box>

      {/* P&L Chart */}
      <Box sx={{ marginBottom: '2px' }}>
        <CandlestickChart
          symbol="Portfolio Value"
          data={chartData}
          interval="1d"
          theme={theme}
          showVolume={true} // Show daily P&L as "volume"
          showControls={false}
          height={height}
        />
      </Box>

      {/* Drawdown Chart (if enabled) */}
      {showDrawdown && drawdownData.length > 0 && (
        <Box
          sx={{
            padding: '16px',
            backgroundColor: theme === 'dark' ? '#1e222d' : '#f0f3fa',
            color: theme === 'dark' ? '#d1d4dc' : '#191919',
            marginTop: '2px',
          }}
        >
          <Typography variant="subtitle2" gutterBottom style={{ fontWeight: 600 }}>
            📉 Drawdown Analysis
          </Typography>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '12px', fontSize: '12px', marginTop: '12px' }}>
            <div>
              <div style={{ opacity: 0.7 }}>Max Drawdown</div>
              <div style={{ fontSize: '20px', fontWeight: 600, color: '#ef5350' }}>
                -{metrics.maxDrawdown.toFixed(2)}%
              </div>
            </div>
            <div>
              <div style={{ opacity: 0.7 }}>Recovery Status</div>
              <Chip
                label={drawdownData[drawdownData.length - 1].drawdown < 1 ? 'Recovered' : 'In Drawdown'}
                size="small"
                color={drawdownData[drawdownData.length - 1].drawdown < 1 ? 'success' : 'error'}
              />
            </div>
          </div>
        </Box>
      )}

      {/* Info Footer */}
      <Box
        sx={{
          padding: '16px',
          backgroundColor: theme === 'dark' ? '#1e222d' : '#f0f3fa',
          color: theme === 'dark' ? '#d1d4dc' : '#191919',
          borderRadius: '0 0 8px 8px',
          marginTop: '2px',
        }}
      >
        <Typography variant="subtitle2" gutterBottom style={{ fontWeight: 600, fontSize: '13px' }}>
          📖 Metrics Guide:
        </Typography>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '12px', fontSize: '12px', marginTop: '8px' }}>
          <div>
            <strong>Sharpe Ratio:</strong> Risk-adjusted returns. &gt;2 is excellent, 1-2 is good, &lt;1 needs improvement.
          </div>
          <div>
            <strong>Max Drawdown:</strong> Largest peak-to-trough decline. Lower is better.
          </div>
          <div>
            <strong>Win Rate:</strong> Percentage of profitable days. 60%+ is strong.
          </div>
          <div>
            <strong>Volume Bars:</strong> Represent daily P&L magnitude (green=profit, red=loss).
          </div>
        </div>
      </Box>
    </div>
  );
};

export default PortfolioPnLChart;

/**
 * Helper: Generate sample portfolio snapshots
 */
export function generateSamplePortfolioSnapshots(
  initialValue: number = 100000,
  days: number = 365
): PortfolioSnapshot[] {
  const snapshots: PortfolioSnapshot[] = [];
  const currentDate = new Date();
  let portfolioValue = initialValue;
  let cumulativePnl = 0;

  for (let i = 0; i < days; i++) {
    const date = new Date(currentDate);
    date.setDate(date.getDate() - (days - i));

    // Random daily return with slight upward drift
    const dailyReturn = (Math.random() - 0.48) * 0.02; // -1% to +1%, slightly positive bias
    const dailyPnl = portfolioValue * dailyReturn;
    portfolioValue += dailyPnl;
    cumulativePnl += dailyPnl;

    // Split between cash and positions (assume 50/50 on average)
    const positionsValue = portfolioValue * (0.4 + Math.random() * 0.2);
    const cash = portfolioValue - positionsValue;

    // Benchmark (e.g., S&P 500) with 0.01% daily drift
    const benchmarkValue = initialValue * Math.pow(1.0001, i);

    snapshots.push({
      timestamp: date.toISOString().split('T')[0],
      portfolio_value: portfolioValue,
      cash,
      positions_value: positionsValue,
      daily_pnl: dailyPnl,
      cumulative_pnl: cumulativePnl,
      benchmark_value: benchmarkValue,
    });
  }

  return snapshots;
}
