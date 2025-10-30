import React, { useState } from 'react';
import { TrendingUp, TrendingDown, Info, Loader2 } from 'lucide-react';

interface SignalCardProps {
  title: string;
  value: number | null | undefined;
  type: 'gauge' | 'zscore' | 'percentage';
  tooltip: string;
  range: [number, number];
  thresholds: {
    excellent: number;
    good: number;
    warning: number;
  };
}

/**
 * SignalCard - Displays a single Phase 4 metric with color-coded visualization
 * 
 * Supports three visualization types:
 * - gauge: -1 to +1 scale (Options Flow)
 * - zscore: Z-score scale (Residual Momentum)
 * - percentage: 0 to 100% scale (Seasonality, Breadth & Liquidity)
 */
export const SignalCard: React.FC<SignalCardProps> = ({
  title,
  value,
  type,
  tooltip,
  range,
  thresholds,
}) => {
  const [showTooltip, setShowTooltip] = useState(false);

  // Loading state
  if (value === null || value === undefined) {
    return (
      <div className="p-6 bg-[#1a1a1a] border border-[#404040] rounded-lg">
        <h3 className="text-lg font-semibold text-white mb-4">{title}</h3>
        <div className="flex flex-col items-center justify-center h-48">
          <Loader2 className="w-8 h-8 animate-spin text-gray-500 mb-2" />
          <span className="text-sm text-gray-500">Computing...</span>
        </div>
      </div>
    );
  }

  // Color based on thresholds
  const getColor = (): string => {
    if (value >= thresholds.excellent) return '#10b981'; // Green
    if (value >= thresholds.good) return '#84cc16'; // Light green
    if (value >= thresholds.warning) return '#f59e0b'; // Orange
    return '#ef4444'; // Red
  };

  // Interpretation text
  const getInterpretation = (): string => {
    if (type === 'gauge') {
      if (value > 0.5) return 'Strong Bullish';
      if (value > 0) return 'Mild Bullish';
      if (value > -0.5) return 'Mild Bearish';
      return 'Strong Bearish';
    }

    if (type === 'zscore') {
      if (value > 2) return 'Strong Outperformance';
      if (value > 1) return 'Mild Outperformance';
      if (value > -1) return 'Neutral';
      if (value > -2) return 'Mild Underperformance';
      return 'Strong Underperformance';
    }

    // percentage
    if (value > 0.6) return 'Strong';
    if (value > 0.4) return 'Moderate';
    if (value > 0.2) return 'Weak';
    return 'Very Weak';
  };

  // Format value for display
  const formatValue = (): string => {
    if (type === 'percentage') {
      return `${(value * 100).toFixed(0)}%`;
    }
    if (type === 'zscore') {
      return `${value.toFixed(2)}σ`;
    }
    return value.toFixed(2);
  };

  // Calculate gauge angle (for gauge type)
  const getGaugeAngle = (): number => {
    const [min, max] = range;
    const normalized = (value - min) / (max - min);
    return normalized * 180; // 0 to 180 degrees
  };

  const color = getColor();
  const interpretation = getInterpretation();
  const formattedValue = formatValue();

  const icon = value > 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />;

  return (
    <div className="p-6 bg-[#1a1a1a] border border-[#404040] rounded-lg hover:border-[#505050] transition-all">
      {/* Header with tooltip */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">{title}</h3>
        <div className="relative">
          <button
            className="text-gray-400 hover:text-white transition-colors"
            onMouseEnter={() => setShowTooltip(true)}
            onMouseLeave={() => setShowTooltip(false)}
            aria-label="More information"
          >
            <Info className="w-4 h-4" />
          </button>
          {showTooltip && (
            <div className="absolute right-0 top-6 w-64 p-3 bg-[#3a3a3a] border border-[#505050] rounded-lg shadow-lg z-10">
              <p className="text-xs text-gray-300">{tooltip}</p>
            </div>
          )}
        </div>
      </div>

      {/* Gauge Visualization (for gauge type) */}
      {type === 'gauge' && (
        <div className="relative w-full h-32 flex items-center justify-center mb-4">
          <svg viewBox="0 0 200 120" className="w-full h-full">
            {/* Background arc */}
            <path
              d="M 20 100 A 80 80 0 0 1 180 100"
              fill="none"
              stroke="#404040"
              strokeWidth="12"
              strokeLinecap="round"
            />

            {/* Value arc */}
            <path
              d="M 20 100 A 80 80 0 0 1 180 100"
              fill="none"
              stroke={color}
              strokeWidth="12"
              strokeLinecap="round"
              strokeDasharray={`${getGaugeAngle() * 2.5} 1000`}
              className="transition-all duration-500"
            />

            {/* Needle */}
            <line
              x1="100"
              y1="100"
              x2={100 + 70 * Math.cos((getGaugeAngle() - 90) * Math.PI / 180)}
              y2={100 + 70 * Math.sin((getGaugeAngle() - 90) * Math.PI / 180)}
              stroke={color}
              strokeWidth="3"
              strokeLinecap="round"
              className="transition-all duration-500"
            />

            {/* Center dot */}
            <circle cx="100" cy="100" r="5" fill={color} />
          </svg>
        </div>
      )}

      {/* Value Display */}
      <div className="mb-4">
        <div className="text-3xl font-bold transition-colors duration-500" style={{ color }}>
          {formattedValue}
        </div>
        <div className="text-sm text-gray-400 flex items-center gap-1 mt-1">
          {icon}
          {interpretation}
        </div>
      </div>

      {/* Progress Bar (for percentage and zscore types) */}
      {(type === 'percentage' || type === 'zscore') && (
        <div className="mb-4">
          <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full transition-all duration-500"
              style={{
                width: `${((value - range[0]) / (range[1] - range[0])) * 100}%`,
                backgroundColor: color,
              }}
            />
          </div>
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>{range[0]}</span>
            <span>{range[1]}</span>
          </div>
        </div>
      )}

      {/* Tooltip (always visible at bottom) */}
      <div className="p-2 bg-blue-900/20 border border-blue-500/30 rounded text-xs text-gray-300">
        💡 {tooltip}
      </div>
    </div>
  );
};

export default SignalCard;

