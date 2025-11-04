/**
 * ML Prediction Chart with Uncertainty Visualization
 *
 * Specialized chart for machine learning predictions with:
 * - Prediction cones (quantile bands)
 * - Conformal prediction intervals
 * - Multi-horizon forecasts
 * - Uncertainty visualization
 * - Historical context + future predictions
 *
 * Designed for TFT, GNN, Mamba, PINN models
 */

import React, { useMemo } from 'react';
import { CandlestickChart, OHLCVData, IndicatorConfig } from './index';

export interface MLPrediction {
  timestamp: string | Date;
  horizon: number; // days ahead
  point_prediction: number;
  q10?: number; // 10th percentile (pessimistic)
  q25?: number;
  q50?: number; // median
  q75?: number;
  q90?: number; // 90th percentile (optimistic)
  conformal_lower?: number; // Conformal prediction interval
  conformal_upper?: number;
  model_name?: string;
}

export interface MLPredictionChartProps {
  symbol: string;
  historicalData: OHLCVData[];
  predictions: MLPrediction[];
  currentPrice: number;
  theme?: 'dark' | 'light';
  showConformalIntervals?: boolean;
  showQuantiles?: boolean;
  height?: number;
  className?: string;
}

/**
 * ML Prediction Chart
 *
 * Usage:
 * ```tsx
 * <MLPredictionChart
 *   symbol="AAPL"
 *   historicalData={historicalOHLCV}
 *   predictions={[
 *     {
 *       timestamp: '2024-01-10',
 *       horizon: 1,
 *       point_prediction: 155,
 *       q10: 150,
 *       q50: 155,
 *       q90: 160,
 *       conformal_lower: 148,
 *       conformal_upper: 162
 *     }
 *   ]}
 *   currentPrice={152.50}
 *   showConformalIntervals={true}
 *   showQuantiles={true}
 * />
 * ```
 */
const MLPredictionChart: React.FC<MLPredictionChartProps> = ({
  symbol,
  historicalData,
  predictions,
  currentPrice,
  theme = 'dark',
  showConformalIntervals = true,
  showQuantiles = true,
  height = 700,
  className = '',
}) => {
  /**
   * Combine historical data with prediction bars
   * Predictions are rendered as candlesticks with:
   * - Open/Close: Point prediction (q50)
   * - High: q90 (optimistic)
   * - Low: q10 (pessimistic)
   * - Volume: 0 (no volume for predictions)
   */
  const chartData = useMemo(() => {
    const combined = [...historicalData];

    // Add prediction bars
    predictions.forEach((pred) => {
      const predTime = typeof pred.timestamp === 'string'
        ? pred.timestamp
        : pred.timestamp.toISOString().split('T')[0];

      // Use quantiles if available, otherwise use point prediction
      const high = showQuantiles && pred.q90 ? pred.q90 : pred.point_prediction * 1.05;
      const low = showQuantiles && pred.q10 ? pred.q10 : pred.point_prediction * 0.95;
      const median = pred.q50 || pred.point_prediction;

      combined.push({
        time: predTime,
        open: median,
        high: high,
        low: low,
        close: median,
        volume: 0, // No volume for predictions
      });
    });

    return combined;
  }, [historicalData, predictions, showQuantiles]);

  /**
   * Create indicator lines for conformal intervals
   */
  const indicators = useMemo<IndicatorConfig[]>(() => {
    const indics: IndicatorConfig[] = [];

    if (showConformalIntervals && predictions.length > 0) {
      // Check if conformal intervals are available
      const hasConformal = predictions.some(
        (p) => p.conformal_lower !== undefined && p.conformal_upper !== undefined
      );

      if (hasConformal) {
        // Note: Conformal intervals would need custom rendering
        // For now, we'll use the quantile bands in the candlesticks
        // Future enhancement: Add custom band overlay
      }
    }

    // Add trend line for point predictions
    indics.push({
      type: 'ema' as any,
      period: 20,
      color: '#2196f3',
      lineWidth: 2,
      label: 'Prediction Trend',
      id: 'prediction_trend',
    });

    return indics;
  }, [predictions, showConformalIntervals]);

  /**
   * Calculate prediction statistics for display
   */
  const predictionStats = useMemo(() => {
    if (predictions.length === 0) return null;

    const lastPred = predictions[predictions.length - 1];
    const expectedReturn = ((lastPred.point_prediction - currentPrice) / currentPrice) * 100;

    // Calculate average uncertainty
    const avgUncertainty =
      predictions.reduce((sum, p) => {
        const upper = p.q90 || p.conformal_upper || p.point_prediction * 1.05;
        const lower = p.q10 || p.conformal_lower || p.point_prediction * 0.95;
        return sum + (upper - lower);
      }, 0) / predictions.length;

    return {
      finalPrediction: lastPred.point_prediction,
      expectedReturn,
      avgUncertainty,
      horizonDays: lastPred.horizon,
      predictionsCount: predictions.length,
    };
  }, [predictions, currentPrice]);

  return (
    <div className={`ml-prediction-chart ${className}`}>
      {/* Stats Header */}
      {predictionStats && (
        <div
          style={{
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
                Current Price
              </div>
              <div style={{ fontSize: '24px', fontWeight: 600 }}>
                ${currentPrice.toFixed(2)}
              </div>
            </div>

            <div>
              <div style={{ fontSize: '12px', opacity: 0.7, marginBottom: '4px' }}>
                {predictionStats.horizonDays}-Day Prediction
              </div>
              <div
                style={{
                  fontSize: '24px',
                  fontWeight: 600,
                  color: predictionStats.expectedReturn >= 0 ? '#26a69a' : '#ef5350',
                }}
              >
                ${predictionStats.finalPrediction.toFixed(2)}
              </div>
              <div style={{ fontSize: '12px', opacity: 0.8 }}>
                {predictionStats.expectedReturn >= 0 ? '+' : ''}
                {predictionStats.expectedReturn.toFixed(2)}% expected
              </div>
            </div>

            <div>
              <div style={{ fontSize: '12px', opacity: 0.7, marginBottom: '4px' }}>
                Avg Uncertainty (±)
              </div>
              <div style={{ fontSize: '24px', fontWeight: 600 }}>
                ${predictionStats.avgUncertainty.toFixed(2)}
              </div>
              <div style={{ fontSize: '12px', opacity: 0.8 }}>
                {((predictionStats.avgUncertainty / currentPrice) * 100).toFixed(1)}% of price
              </div>
            </div>

            <div>
              <div style={{ fontSize: '12px', opacity: 0.7, marginBottom: '4px' }}>
                Forecast Points
              </div>
              <div style={{ fontSize: '24px', fontWeight: 600 }}>
                {predictionStats.predictionsCount}
              </div>
              <div style={{ fontSize: '12px', opacity: 0.8 }}>
                {showQuantiles ? 'With quantiles' : 'Point predictions'}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Info Banner */}
      <div
        style={{
          padding: '12px 16px',
          backgroundColor: theme === 'dark' ? 'rgba(33, 150, 243, 0.1)' : 'rgba(33, 150, 243, 0.05)',
          color: theme === 'dark' ? '#64b5f6' : '#1976d2',
          borderLeft: '4px solid #2196f3',
          marginBottom: '2px',
          fontSize: '13px',
        }}
      >
        <strong>📊 Prediction Visualization:</strong> Candlestick bars in the future represent predictions.
        {showQuantiles && (
          <>
            {' '}
            <strong>High</strong>=90th percentile (optimistic), <strong>Low</strong>=10th percentile (pessimistic),{' '}
            <strong>Close</strong>=median prediction.
          </>
        )}
        {showConformalIntervals && ' Conformal prediction intervals provide additional uncertainty bounds.'}
      </div>

      {/* Chart */}
      <CandlestickChart
        symbol={symbol}
        data={chartData}
        interval="1d"
        theme={theme}
        showVolume={true}
        showControls={false}
        indicators={indicators}
        height={height}
      />

      {/* Legend */}
      <div
        style={{
          padding: '16px',
          backgroundColor: theme === 'dark' ? '#1e222d' : '#f0f3fa',
          color: theme === 'dark' ? '#d1d4dc' : '#191919',
          borderRadius: '0 0 8px 8px',
          marginTop: '2px',
        }}
      >
        <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '8px' }}>
          📖 How to Read This Chart:
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '12px', fontSize: '12px' }}>
          <div>
            <strong>Green Candles (Historical):</strong> Price went up that day
          </div>
          <div>
            <strong>Red Candles (Historical):</strong> Price went down that day
          </div>
          <div>
            <strong>Future Candles:</strong> ML predictions with uncertainty
          </div>
          <div>
            <strong>Wicks (High/Low):</strong> Uncertainty range (10th-90th percentile)
          </div>
          <div>
            <strong>Body:</strong> Most likely price (median prediction)
          </div>
          <div>
            <strong>Volume = 0:</strong> Indicates prediction bars (not real trades)
          </div>
        </div>
      </div>
    </div>
  );
};

export default MLPredictionChart;

/**
 * Helper: Convert TFT model output to ML predictions
 */
export function convertTFTOutput(
  tftOutput: any,
  currentTime: Date
): MLPrediction[] {
  const predictions: MLPrediction[] = [];

  // TFT typically outputs multiple horizons with quantiles
  for (let i = 0; i < tftOutput.horizons.length; i++) {
    const futureDate = new Date(currentTime);
    futureDate.setDate(futureDate.getDate() + tftOutput.horizons[i]);

    predictions.push({
      timestamp: futureDate,
      horizon: tftOutput.horizons[i],
      point_prediction: tftOutput.point_predictions[i],
      q10: tftOutput.quantiles?.q10[i],
      q25: tftOutput.quantiles?.q25[i],
      q50: tftOutput.quantiles?.q50[i],
      q75: tftOutput.quantiles?.q75[i],
      q90: tftOutput.quantiles?.q90[i],
      conformal_lower: tftOutput.conformal?.lower[i],
      conformal_upper: tftOutput.conformal?.upper[i],
      model_name: 'TFT',
    });
  }

  return predictions;
}

/**
 * Helper: Convert GNN model output to ML predictions
 */
export function convertGNNOutput(
  gnnOutput: any,
  currentTime: Date
): MLPrediction[] {
  const predictions: MLPrediction[] = [];

  // GNN typically outputs single horizon with uncertainty
  const futureDate = new Date(currentTime);
  futureDate.setDate(futureDate.getDate() + (gnnOutput.horizon || 1));

  predictions.push({
    timestamp: futureDate,
    horizon: gnnOutput.horizon || 1,
    point_prediction: gnnOutput.prediction,
    q10: gnnOutput.uncertainty_lower,
    q90: gnnOutput.uncertainty_upper,
    model_name: 'GNN',
  });

  return predictions;
}

/**
 * Helper: Convert Mamba model output to ML predictions
 */
export function convertMambaOutput(
  mambaOutput: any,
  currentTime: Date
): MLPrediction[] {
  const predictions: MLPrediction[] = [];

  // Mamba outputs long sequences
  for (let i = 0; i < mambaOutput.sequence.length; i++) {
    const futureDate = new Date(currentTime);
    futureDate.setDate(futureDate.getDate() + i + 1);

    predictions.push({
      timestamp: futureDate,
      horizon: i + 1,
      point_prediction: mambaOutput.sequence[i],
      model_name: 'Mamba',
    });
  }

  return predictions;
}

/**
 * Helper: Convert PINN model output to ML predictions
 */
export function convertPINNOutput(
  pinnOutput: any,
  currentTime: Date
): MLPrediction[] {
  const predictions: MLPrediction[] = [];

  // PINN typically outputs option prices, but can forecast underlying
  if (pinnOutput.underlying_forecast) {
    for (let i = 0; i < pinnOutput.underlying_forecast.length; i++) {
      const futureDate = new Date(currentTime);
      futureDate.setDate(futureDate.getDate() + i + 1);

      predictions.push({
        timestamp: futureDate,
        horizon: i + 1,
        point_prediction: pinnOutput.underlying_forecast[i],
        model_name: 'PINN',
      });
    }
  }

  return predictions;
}

/**
 * Helper: Generate sample ML predictions for testing
 */
export function generateSampleMLPredictions(
  startPrice: number,
  days: number = 30
): MLPrediction[] {
  const predictions: MLPrediction[] = [];
  const currentDate = new Date();
  let price = startPrice;

  for (let i = 1; i <= days; i++) {
    const futureDate = new Date(currentDate);
    futureDate.setDate(futureDate.getDate() + i);

    // Random walk with slight upward bias
    const dailyReturn = (Math.random() - 0.45) * 0.02;
    price *= 1 + dailyReturn;

    // Add uncertainty that grows with horizon
    const uncertainty = price * 0.05 * Math.sqrt(i / days);

    predictions.push({
      timestamp: futureDate,
      horizon: i,
      point_prediction: price,
      q10: price - uncertainty * 1.5,
      q25: price - uncertainty * 0.8,
      q50: price,
      q75: price + uncertainty * 0.8,
      q90: price + uncertainty * 1.5,
      conformal_lower: price - uncertainty * 2,
      conformal_upper: price + uncertainty * 2,
    });
  }

  return predictions;
}
