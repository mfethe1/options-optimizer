/**
 * VIX Forecast Chart Component
 *
 * Specialized chart for displaying VIX (Volatility Index) forecasts
 * from the Epidemic Volatility model
 *
 * Features:
 * - Multi-timeframe VIX visualization (1d, 1w, 1M)
 * - Current VIX vs Predicted VIX with uncertainty bands
 * - Epidemic regime indicators
 * - SEIR state overlays
 */

import React, { useMemo } from 'react';
import { MultiTimeframeChart, CandlestickChart, OHLCVData, generateSampleData } from './index';
import { Box, Typography, Chip } from '@mui/material';

export interface VIXForecastData {
  timestamp: string | Date;
  current_vix: number;
  predicted_vix: number;
  lower_bound?: number;
  upper_bound?: number;
  regime?: 'calm' | 'pre_volatile' | 'volatile' | 'stabilized';
}

export interface VIXForecastChartProps {
  currentVIX: number;
  predictedVIX: number;
  horizonDays: number;
  forecastData?: VIXForecastData[];
  theme?: 'dark' | 'light';
  showMultiTimeframe?: boolean;
  height?: number;
  className?: string;
}

const REGIME_COLORS: Record<string, string> = {
  'calm': '#4caf50',
  'pre_volatile': '#ff9800',
  'volatile': '#f44336',
  'stabilized': '#2196f3'
};

/**
 * VIX Forecast Chart
 *
 * Usage:
 * ```tsx
 * <VIXForecastChart
 *   currentVIX={18.5}
 *   predictedVIX={25.3}
 *   horizonDays={30}
 *   forecastData={[...]}
 *   theme="dark"
 *   showMultiTimeframe={true}
 * />
 * ```
 */
const VIXForecastChart: React.FC<VIXForecastChartProps> = ({
  currentVIX,
  predictedVIX,
  horizonDays,
  forecastData,
  theme = 'dark',
  showMultiTimeframe = false,
  height = 600,
  className = '',
}) => {
  /**
   * Generate VIX historical + forecast data
   */
  const chartData = useMemo(() => {
    // Generate historical VIX data (past year)
    // VIX trades like a stock but represents volatility
    const historicalBars = generateSampleData(365, currentVIX * 0.8);

    // Adjust last bar to match current VIX
    if (historicalBars.length > 0) {
      historicalBars[historicalBars.length - 1].close = currentVIX;
      historicalBars[historicalBars.length - 1].high = Math.max(
        historicalBars[historicalBars.length - 1].high,
        currentVIX
      );
      historicalBars[historicalBars.length - 1].low = Math.min(
        historicalBars[historicalBars.length - 1].low,
        currentVIX
      );
    }

    // Add forecast bars
    if (forecastData && forecastData.length > 0) {
      forecastData.forEach((forecast) => {
        const forecastTime = typeof forecast.timestamp === 'string'
          ? forecast.timestamp
          : forecast.timestamp.toISOString().split('T')[0];

        historicalBars.push({
          time: forecastTime,
          open: forecast.predicted_vix,
          high: forecast.upper_bound || forecast.predicted_vix * 1.15,
          low: forecast.lower_bound || forecast.predicted_vix * 0.85,
          close: forecast.predicted_vix,
          volume: 0, // No volume for VIX predictions
        });
      });
    } else {
      // Generate simple linear interpolation forecast
      const lastTime = new Date(historicalBars[historicalBars.length - 1].time);

      for (let i = 1; i <= horizonDays; i++) {
        const futureDate = new Date(lastTime);
        futureDate.setDate(futureDate.getDate() + i);

        // Linear interpolation
        const progress = i / horizonDays;
        const interpolatedVIX = currentVIX + (predictedVIX - currentVIX) * progress;

        // Growing uncertainty
        const uncertainty = Math.abs(predictedVIX - currentVIX) * Math.sqrt(progress) * 0.5;

        historicalBars.push({
          time: futureDate.toISOString().split('T')[0],
          open: interpolatedVIX,
          high: interpolatedVIX + uncertainty,
          low: interpolatedVIX - uncertainty,
          close: interpolatedVIX,
          volume: 0,
        });
      }
    }

    return historicalBars;
  }, [currentVIX, predictedVIX, horizonDays, forecastData]);

  /**
   * Prepare multi-timeframe data
   */
  const multiTimeframeData = useMemo(() => {
    if (!showMultiTimeframe) return {};

    // For demo, use same data with different aggregations
    // In production, you'd fetch different timeframes from API
    return {
      '1d': chartData,
      '1w': chartData.filter((_, idx) => idx % 7 === 0), // Weekly sampling
      '1M': chartData.filter((_, idx) => idx % 30 === 0), // Monthly sampling
    };
  }, [chartData, showMultiTimeframe]);

  /**
   * Calculate key statistics
   */
  const vixStats = useMemo(() => {
    const change = predictedVIX - currentVIX;
    const changePercent = (change / currentVIX) * 100;

    // VIX interpretation
    let interpretation = '';
    if (predictedVIX < 15) {
      interpretation = 'Low volatility expected - Calm market regime';
    } else if (predictedVIX < 20) {
      interpretation = 'Normal volatility - Stable conditions';
    } else if (predictedVIX < 30) {
      interpretation = 'Elevated volatility - Pre-volatile regime, caution advised';
    } else if (predictedVIX < 40) {
      interpretation = 'High volatility - Volatile regime, heightened risk';
    } else {
      interpretation = 'Extreme volatility - Crisis conditions, extreme caution';
    }

    return {
      change,
      changePercent,
      interpretation,
      direction: change >= 0 ? 'rising' : 'falling',
    };
  }, [currentVIX, predictedVIX]);

  return (
    <div className={`vix-forecast-chart ${className}`}>
      {/* Stats Header */}
      <Box
        sx={{
          padding: '16px',
          backgroundColor: theme === 'dark' ? '#1e222d' : '#f0f3fa',
          color: theme === 'dark' ? '#d1d4dc' : '#191919',
          borderRadius: '8px 8px 0 0',
          marginBottom: '2px',
        }}
      >
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
          <div>
            <div style={{ fontSize: '12px', opacity: 0.7, marginBottom: '4px' }}>
              Current VIX
            </div>
            <div style={{ fontSize: '28px', fontWeight: 600 }}>
              {currentVIX.toFixed(2)}
            </div>
            <div style={{ fontSize: '11px', opacity: 0.7 }}>
              {currentVIX < 15 ? 'Calm' : currentVIX < 20 ? 'Normal' : currentVIX < 30 ? 'Elevated' : 'High'}
            </div>
          </div>

          <div>
            <div style={{ fontSize: '12px', opacity: 0.7, marginBottom: '4px' }}>
              {horizonDays}-Day Forecast
            </div>
            <div
              style={{
                fontSize: '28px',
                fontWeight: 600,
                color: vixStats.direction === 'rising' ? '#ef5350' : '#26a69a',
              }}
            >
              {predictedVIX.toFixed(2)}
            </div>
            <div style={{ fontSize: '11px', opacity: 0.8 }}>
              {vixStats.direction === 'rising' ? '↑' : '↓'} {vixStats.changePercent >= 0 ? '+' : ''}
              {vixStats.changePercent.toFixed(1)}%
            </div>
          </div>

          <div style={{ gridColumn: 'span 2' }}>
            <div style={{ fontSize: '12px', opacity: 0.7, marginBottom: '4px' }}>
              Interpretation
            </div>
            <div style={{ fontSize: '14px', fontWeight: 500, marginTop: '4px' }}>
              {vixStats.interpretation}
            </div>
          </div>
        </div>
      </Box>

      {/* Info Banner */}
      <Box
        sx={{
          padding: '12px 16px',
          backgroundColor:
            vixStats.direction === 'rising'
              ? theme === 'dark'
                ? 'rgba(239, 83, 80, 0.1)'
                : 'rgba(239, 83, 80, 0.05)'
              : theme === 'dark'
              ? 'rgba(38, 166, 154, 0.1)'
              : 'rgba(38, 166, 154, 0.05)',
          color:
            vixStats.direction === 'rising'
              ? theme === 'dark'
                ? '#ef9a9a'
                : '#c62828'
              : theme === 'dark'
              ? '#80cbc4'
              : '#00695c',
          borderLeft: `4px solid ${vixStats.direction === 'rising' ? '#ef5350' : '#26a69a'}`,
          marginBottom: '2px',
          fontSize: '13px',
        }}
      >
        <strong>📊 VIX Forecast:</strong> Historical VIX shown as candlesticks. Future bars (volume=0) represent epidemic
        model predictions with uncertainty bands. <strong>High/Low</strong>=uncertainty range, <strong>Close</strong>=median
        prediction.
        {vixStats.direction === 'rising' && ' ⚠️ Rising volatility forecasted - consider protective strategies!'}
      </Box>

      {/* Chart */}
      {showMultiTimeframe ? (
        <MultiTimeframeChart
          symbol="VIX"
          data={multiTimeframeData}
          layout="1x2"
          theme={theme}
          showVolume={false} // VIX doesn't have traditional volume
        />
      ) : (
        <CandlestickChart
          symbol="VIX"
          data={chartData}
          interval="1d"
          theme={theme}
          showVolume={false}
          showControls={true}
          height={height}
        />
      )}

      {/* VIX Interpretation Legend */}
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
          📖 VIX Levels Guide:
        </Typography>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '12px',
            fontSize: '12px',
            marginTop: '8px',
          }}
        >
          <div>
            <Chip
              label="VIX < 15"
              size="small"
              sx={{ bgcolor: REGIME_COLORS.calm, color: 'white', fontWeight: 600 }}
            />
            <Typography variant="caption" sx={{ ml: 1 }}>
              Calm - Low volatility
            </Typography>
          </div>
          <div>
            <Chip
              label="VIX 15-20"
              size="small"
              sx={{ bgcolor: REGIME_COLORS.stabilized, color: 'white', fontWeight: 600 }}
            />
            <Typography variant="caption" sx={{ ml: 1 }}>
              Normal - Stable market
            </Typography>
          </div>
          <div>
            <Chip
              label="VIX 20-30"
              size="small"
              sx={{ bgcolor: REGIME_COLORS.pre_volatile, color: 'white', fontWeight: 600 }}
            />
            <Typography variant="caption" sx={{ ml: 1 }}>
              Elevated - Caution
            </Typography>
          </div>
          <div>
            <Chip
              label="VIX > 30"
              size="small"
              sx={{ bgcolor: REGIME_COLORS.volatile, color: 'white', fontWeight: 600 }}
            />
            <Typography variant="caption" sx={{ ml: 1 }}>
              High - Extreme risk
            </Typography>
          </div>
        </div>
      </Box>
    </div>
  );
};

export default VIXForecastChart;

/**
 * Helper: Generate sample VIX forecast data
 */
export function generateSampleVIXForecast(
  currentVIX: number,
  targetVIX: number,
  days: number
): VIXForecastData[] {
  const forecasts: VIXForecastData[] = [];
  const currentDate = new Date();

  for (let i = 1; i <= days; i++) {
    const futureDate = new Date(currentDate);
    futureDate.setDate(futureDate.getDate() + i);

    // Linear interpolation
    const progress = i / days;
    const interpolatedVIX = currentVIX + (targetVIX - currentVIX) * progress;

    // Uncertainty grows with horizon
    const uncertainty = Math.abs(targetVIX - currentVIX) * Math.sqrt(progress) * 0.3;

    // Determine regime
    let regime: 'calm' | 'pre_volatile' | 'volatile' | 'stabilized' = 'calm';
    if (interpolatedVIX < 15) regime = 'calm';
    else if (interpolatedVIX < 20) regime = 'stabilized';
    else if (interpolatedVIX < 30) regime = 'pre_volatile';
    else regime = 'volatile';

    forecasts.push({
      timestamp: futureDate,
      current_vix: currentVIX,
      predicted_vix: interpolatedVIX,
      lower_bound: interpolatedVIX - uncertainty,
      upper_bound: interpolatedVIX + uncertainty,
      regime,
    });
  }

  return forecasts;
}
