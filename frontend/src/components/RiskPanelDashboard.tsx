import React from 'react';
import { Shield, TrendingUp, TrendingDown, AlertTriangle, Clock } from 'lucide-react';
import { RiskPanel } from '../types/investor-report';
import { RiskMetricCard } from './RiskMetricCard';
import { DataFreshnessIndicator } from './DataFreshnessIndicator';

interface Props {
  riskPanel: RiskPanel;
  regime?: 'bull' | 'bear' | 'neutral';
  loading?: boolean;
  lastUpdated?: Date | string | null;
}

/**
 * RiskPanelDashboard - Institutional-grade risk metrics visualization
 * 
 * Displays 7 key risk metrics in a responsive grid:
 * 1. Omega Ratio - Probability-weighted gains/losses (>2.0 = Renaissance-level)
 * 2. GH1 Ratio - Return enhancement + risk reduction vs benchmark
 * 3. Pain Index - Drawdown depth × duration (lower is better)
 * 4. Upside Capture - % of benchmark gains captured
 * 5. Downside Capture - % of benchmark losses captured (lower is better)
 * 6. CVaR 95% - Expected loss in worst 5% scenarios
 * 7. Max Drawdown - Maximum peak-to-trough decline
 * 
 * Layout: 2×4 grid (desktop) → 1×7 stack (mobile)
 * Color coding: Regime-aware (bull/bear/neutral)
 */
export const RiskPanelDashboard: React.FC<Props> = ({
  riskPanel,
  regime = 'neutral',
  loading = false,
  lastUpdated = null,
}) => {
  // Regime-based theme colors
  const getRegimeColor = (): string => {
    switch (regime) {
      case 'bull':
        return '#10b981'; // Green
      case 'bear':
        return '#ef4444'; // Red
      default:
        return '#3b82f6'; // Blue
    }
  };

  const getRegimeIcon = () => {
    switch (regime) {
      case 'bull':
        return <TrendingUp className="w-5 h-5" />;
      case 'bear':
        return <TrendingDown className="w-5 h-5" />;
      default:
        return <Shield className="w-5 h-5" />;
    }
  };

  const regimeColor = getRegimeColor();
  const regimeIcon = getRegimeIcon();

  return (
    <div className="bg-[#2a2a2a] border border-[#404040] rounded-lg p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center gap-2">
            <Shield className="w-6 h-6" />
            Risk Panel
          </h2>
          <p className="text-sm text-gray-400 mt-1">
            Institutional-grade risk metrics
          </p>
        </div>

        {/* Data Freshness and Regime Indicator */}
        <div className="flex items-center gap-4">
          {/* Data Freshness Indicator */}
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-gray-400" />
            <span className="text-xs text-gray-400">As of:</span>
            <DataFreshnessIndicator
              lastUpdated={lastUpdated}
              staleThresholdSeconds={300}   // 5 minutes = stale for risk metrics
              oldThresholdSeconds={900}     // 15 minutes = old
              showTimestamp={true}
            />
          </div>

          {/* Regime Indicator */}
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg border" style={{ borderColor: regimeColor }}>
            <div style={{ color: regimeColor }}>
              {regimeIcon}
            </div>
            <span className="text-sm font-semibold capitalize" style={{ color: regimeColor }}>
              {regime} Regime
            </span>
          </div>
        </div>
      </div>

      {/* 2×4 Grid (desktop) → 1×7 Stack (mobile) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <RiskMetricCard
          title="Omega Ratio"
          value={riskPanel.omega}
          format="ratio"
          tooltip="Probability-weighted ratio of gains to losses. >2.0 indicates Renaissance-level performance. Measures upside potential vs downside risk."
          thresholds={{
            excellent: 2.0,
            good: 1.5,
            warning: 1.0,
          }}
          higherIsBetter={true}
          regime={regime}
          loading={loading}
        />

        <RiskMetricCard
          title="GH1 Ratio"
          value={riskPanel.gh1}
          format="ratio"
          tooltip="Gain-to-Hurt ratio: return enhancement + risk reduction vs benchmark. >1.5 = strong alpha generation with controlled risk."
          thresholds={{
            excellent: 1.5,
            good: 1.0,
            warning: 0.5,
          }}
          higherIsBetter={true}
          regime={regime}
          loading={loading}
        />

        <RiskMetricCard
          title="Pain Index"
          value={riskPanel.pain_index}
          format="percentage"
          tooltip="Drawdown depth × duration. Measures investor suffering during losses. Lower is better. <5% = excellent risk management."
          thresholds={{
            excellent: 5,
            good: 10,
            warning: 20,
          }}
          higherIsBetter={false}
          regime={regime}
          loading={loading}
        />

        <RiskMetricCard
          title="Upside Capture"
          value={riskPanel.upside_capture}
          format="percentage"
          tooltip="% of benchmark gains captured during up markets. >100% = outperformance. Measures ability to participate in rallies."
          thresholds={{
            excellent: 100,
            good: 80,
            warning: 60,
          }}
          higherIsBetter={true}
          regime={regime}
          loading={loading}
        />

        <RiskMetricCard
          title="Downside Capture"
          value={riskPanel.downside_capture}
          format="percentage"
          tooltip="% of benchmark losses captured during down markets. <100% = protection. Lower is better. <50% = excellent downside protection."
          thresholds={{
            excellent: 50,
            good: 75,
            warning: 100,
          }}
          higherIsBetter={false}
          regime={regime}
          loading={loading}
        />

        <RiskMetricCard
          title="CVaR 95%"
          value={riskPanel.cvar_95}
          format="percentage"
          tooltip="Conditional Value at Risk: expected loss in worst 5% of scenarios. Tail risk measure. More conservative than VaR."
          thresholds={{
            excellent: -5,
            good: -10,
            warning: -20,
          }}
          higherIsBetter={true}
          regime={regime}
          loading={loading}
        />

        <RiskMetricCard
          title="Max Drawdown"
          value={riskPanel.max_drawdown}
          format="percentage"
          tooltip="Maximum peak-to-trough decline. Worst historical loss. <10% = excellent capital preservation. <20% = good risk control."
          thresholds={{
            excellent: -10,
            good: -20,
            warning: -30,
          }}
          higherIsBetter={true}
          regime={regime}
          loading={loading}
        />
      </div>

      {/* Explanations */}
      {riskPanel.explanations && riskPanel.explanations.length > 0 && (
        <div className="mt-6 p-4 bg-blue-900/20 border border-blue-500/50 rounded-lg">
          <h3 className="text-sm font-semibold text-blue-400 mb-2 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4" />
            Risk Assessment
          </h3>
          <ul className="space-y-1">
            {riskPanel.explanations.map((exp, i) => (
              <li key={i} className="text-sm text-gray-300">
                • {exp}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Risk Summary */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-3 bg-[#1a1a1a] border border-[#404040] rounded">
          <div className="text-xs text-gray-400 mb-1">Return Quality</div>
          <div className="text-lg font-bold text-white">
            {riskPanel.omega >= 2.0 ? 'Excellent' : riskPanel.omega >= 1.5 ? 'Good' : 'Fair'}
          </div>
        </div>

        <div className="p-3 bg-[#1a1a1a] border border-[#404040] rounded">
          <div className="text-xs text-gray-400 mb-1">Downside Protection</div>
          <div className="text-lg font-bold text-white">
            {riskPanel.downside_capture <= 50 ? 'Excellent' : riskPanel.downside_capture <= 75 ? 'Good' : 'Fair'}
          </div>
        </div>

        <div className="p-3 bg-[#1a1a1a] border border-[#404040] rounded">
          <div className="text-xs text-gray-400 mb-1">Tail Risk</div>
          <div className="text-lg font-bold text-white">
            {Math.abs(riskPanel.cvar_95) <= 5 ? 'Low' : Math.abs(riskPanel.cvar_95) <= 10 ? 'Moderate' : 'High'}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RiskPanelDashboard;

