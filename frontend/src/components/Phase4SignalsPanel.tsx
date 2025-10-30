import React from 'react';
import { Activity, Loader2 } from 'lucide-react';
import { usePhase4Stream } from '../hooks/usePhase4Stream';
import { Phase4Tech } from '../types/investor-report';
import { SignalCard } from './SignalCard';

interface Props {
  userId: string;
  initialData?: Phase4Tech;
}

/**
 * Phase4SignalsPanel - Real-time visualization of short-horizon (1-5 day) technical signals
 * 
 * Displays 4 key metrics in a 2×2 grid:
 * 1. Options Flow Composite - PCR + IV skew + volume
 * 2. Residual Momentum - Idiosyncratic returns (asset vs market/sector)
 * 3. Seasonality Score - Calendar patterns (turn-of-month, day-of-week)
 * 4. Breadth & Liquidity - Market internals (A/D ratio, volume, spreads)
 * 
 * Updates via WebSocket every 30s (configurable via PHASE4_WS_INTERVAL_SECONDS)
 */
export const Phase4SignalsPanel: React.FC<Props> = ({ userId, initialData }) => {
  const { phase4Data, isConnected, error } = usePhase4Stream(userId, {
    autoReconnect: true,
    maxReconnectAttempts: 5,
  });

  // Use WebSocket data if available, otherwise fall back to initial data
  const data = phase4Data || initialData;

  return (
    <div className="bg-[#2a2a2a] border border-[#404040] rounded-lg p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center gap-2">
            <Activity className="w-6 h-6" />
            Phase 4 Signals
          </h2>
          <p className="text-sm text-gray-400 mt-1">
            Short-horizon edge (1-5 day alpha)
          </p>
        </div>

        {/* Real-time indicator */}
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              isConnected ? 'bg-green-500 animate-pulse' : 'bg-gray-500'
            }`}
            aria-label={isConnected ? 'Connected' : 'Disconnected'}
          />
          <span className="text-xs text-gray-400">
            {isConnected ? 'Live' : 'Disconnected'}
          </span>
        </div>
      </div>

      {/* Error State */}
      {error && (
        <div className="mb-4 p-4 bg-red-900/20 border border-red-500/50 rounded-lg">
          <p className="text-sm text-red-400">
            ⚠️ Connection error: {typeof error === 'string' ? error : (error as any)?.message ?? String(error)}
          </p>
        </div>
      )}

      {/* 2×2 Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <SignalCard
          title="Options Flow Composite"
          value={data?.options_flow_composite}
          type="gauge"
          tooltip="Combines Put/Call Ratio, IV skew, and volume. Bullish when calls dominate with high volume."
          range={[-1, 1]}
          thresholds={{
            excellent: 0.5,
            good: 0,
            warning: -0.5,
          }}
        />

        <SignalCard
          title="Residual Momentum"
          value={data?.residual_momentum}
          type="zscore"
          tooltip="Idiosyncratic returns after removing market/sector effects. >2σ = strong alpha."
          range={[-3, 3]}
          thresholds={{
            excellent: 2,
            good: 1,
            warning: -1,
          }}
        />

        <SignalCard
          title="Seasonality Score"
          value={data?.seasonality_score}
          type="percentage"
          tooltip="Calendar patterns: turn-of-month (days 28-2) and day-of-week effects."
          range={[-1, 1]}
          thresholds={{
            excellent: 0.5,
            good: 0,
            warning: -0.5,
          }}
        />

        <SignalCard
          title="Breadth & Liquidity"
          value={data?.breadth_liquidity}
          type="percentage"
          tooltip="Market internals: A/D ratio (50%), volume (30%), spreads (20%)."
          range={[0, 1]}
          thresholds={{
            excellent: 0.6,
            good: 0.4,
            warning: 0.2,
          }}
        />
      </div>

      {/* Explanations */}
      {data?.explanations && data.explanations.length > 0 && (
        <div className="mt-6 p-4 bg-blue-900/20 border border-blue-500/50 rounded-lg">
          <h3 className="text-sm font-semibold text-blue-400 mb-2">
            Interpretation
          </h3>
          <ul className="space-y-1">
            {data.explanations.map((exp, i) => (
              <li key={i} className="text-sm text-gray-300">
                • {exp}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Loading State (when no data) */}
      {!data && !error && (
        <div className="flex flex-col items-center justify-center h-64">
          <Loader2 className="w-8 h-8 animate-spin text-gray-500 mb-2" />
          <span className="text-sm text-gray-500">
            Loading Phase 4 metrics...
          </span>
        </div>
      )}

    </div>
  );
};

export default Phase4SignalsPanel;

