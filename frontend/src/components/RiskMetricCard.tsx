import React, { useState } from 'react';
import { Info, TrendingUp, TrendingDown, Minus, Loader2 } from 'lucide-react';

interface RiskMetricCardProps {
  title: string;
  value: number;
  format: 'ratio' | 'percentage';
  tooltip: string;
  thresholds: {
    excellent: number;
    good: number;
    warning: number;
  };
  higherIsBetter: boolean;
  regime?: 'bull' | 'bear' | 'neutral';
  loading?: boolean;
}

/**
 * RiskMetricCard - Displays a single risk metric with regime-aware color coding
 * 
 * Features:
 * - Color-coded based on thresholds and higherIsBetter flag
 * - Regime-aware styling (bull/bear/neutral)
 * - Hover tooltip with institutional explanation
 * - Trend indicator (up/down/neutral)
 * - Smooth transitions for value changes
 * - Loading state
 */
export const RiskMetricCard: React.FC<RiskMetricCardProps> = ({
  title,
  value,
  format,
  tooltip,
  thresholds,
  higherIsBetter,
  regime = 'neutral',
  loading = false,
}) => {
  const [showTooltip, setShowTooltip] = useState(false);

  // Loading state
  if (loading) {
    return (
      <div className="p-4 bg-[#1a1a1a] border border-[#404040] rounded-lg">
        <h3 className="text-sm font-semibold text-white mb-3">{title}</h3>
        <div className="flex flex-col items-center justify-center h-24">
          <Loader2 className="w-6 h-6 animate-spin text-gray-500 mb-2" />
          <span className="text-xs text-gray-500">Computing...</span>
        </div>
      </div>
    );
  }

  // Color based on thresholds and higherIsBetter
  const getColor = (): string => {
    if (higherIsBetter) {
      if (value >= thresholds.excellent) return '#10b981'; // Green
      if (value >= thresholds.good) return '#84cc16'; // Light green
      if (value >= thresholds.warning) return '#f59e0b'; // Orange
      return '#ef4444'; // Red
    } else {
      // Lower is better (inverted logic)
      if (value <= thresholds.excellent) return '#10b981'; // Green
      if (value <= thresholds.good) return '#84cc16'; // Light green
      if (value <= thresholds.warning) return '#f59e0b'; // Orange
      return '#ef4444'; // Red
    }
  };

  // Performance level text
  const getPerformanceLevel = (): string => {
    if (higherIsBetter) {
      if (value >= thresholds.excellent) return 'Excellent';
      if (value >= thresholds.good) return 'Good';
      if (value >= thresholds.warning) return 'Fair';
      return 'Poor';
    } else {
      if (value <= thresholds.excellent) return 'Excellent';
      if (value <= thresholds.good) return 'Good';
      if (value <= thresholds.warning) return 'Fair';
      return 'Poor';
    }
  };

  // Trend icon
  const getTrendIcon = () => {
    const color = getColor();
    
    if (higherIsBetter) {
      if (value >= thresholds.good) return <TrendingUp className="w-4 h-4" style={{ color }} />;
      if (value >= thresholds.warning) return <Minus className="w-4 h-4" style={{ color }} />;
      return <TrendingDown className="w-4 h-4" style={{ color }} />;
    } else {
      if (value <= thresholds.good) return <TrendingUp className="w-4 h-4" style={{ color }} />;
      if (value <= thresholds.warning) return <Minus className="w-4 h-4" style={{ color }} />;
      return <TrendingDown className="w-4 h-4" style={{ color }} />;
    }
  };

  // Format value for display
  const formatValue = (): string => {
    if (format === 'percentage') {
      return `${value.toFixed(1)}%`;
    }
    return value.toFixed(2);
  };

  // Regime border color
  const getRegimeBorderColor = (): string => {
    switch (regime) {
      case 'bull':
        return '#10b981';
      case 'bear':
        return '#ef4444';
      default:
        return '#3b82f6';
    }
  };

  const color = getColor();
  const performanceLevel = getPerformanceLevel();
  const formattedValue = formatValue();
  const trendIcon = getTrendIcon();
  const regimeBorderColor = getRegimeBorderColor();

  return (
    <div
      className="p-4 bg-[#1a1a1a] border rounded-lg hover:border-[#505050] transition-all"
      style={{ borderColor: '#404040' }}
    >
      {/* Header with tooltip */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-white">{title}</h3>
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
            <div className="absolute right-0 top-6 w-72 p-3 bg-[#3a3a3a] border border-[#505050] rounded-lg shadow-lg z-10">
              <p className="text-xs text-gray-300">{tooltip}</p>
            </div>
          )}
        </div>
      </div>

      {/* Value Display */}
      <div className="mb-3">
        <div
          className="text-2xl font-bold transition-colors duration-500"
          style={{ color }}
        >
          {formattedValue}
        </div>
        <div className="text-xs text-gray-400 flex items-center gap-1 mt-1">
          {trendIcon}
          {performanceLevel}
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mb-3">
        <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full transition-all duration-500"
            style={{
              width: `${getProgressWidth()}%`,
              backgroundColor: color,
            }}
          />
        </div>
      </div>

      {/* Threshold Indicators */}
      <div className="flex justify-between text-xs text-gray-500">
        <span>{higherIsBetter ? 'Poor' : 'High Risk'}</span>
        <span>{higherIsBetter ? 'Excellent' : 'Low Risk'}</span>
      </div>

      {/* Regime Indicator (subtle bottom border) */}
      <div
        className="mt-3 pt-2 border-t"
        style={{ borderColor: regimeBorderColor, opacity: 0.3 }}
      />
    </div>
  );

  // Helper function to calculate progress bar width
  function getProgressWidth(): number {
    const { excellent, warning } = thresholds;
    
    if (higherIsBetter) {
      // Map value to 0-100% range
      const min = warning * 0.5; // Assume min is half of warning threshold
      const max = excellent * 1.5; // Assume max is 1.5x excellent threshold
      const normalized = ((value - min) / (max - min)) * 100;
      return Math.max(0, Math.min(100, normalized));
    } else {
      // Inverted: lower values = higher progress
      const min = 0;
      const max = warning * 2; // Assume max is 2x warning threshold
      const normalized = ((max - value) / (max - min)) * 100;
      return Math.max(0, Math.min(100, normalized));
    }
  }
};

export default RiskMetricCard;

