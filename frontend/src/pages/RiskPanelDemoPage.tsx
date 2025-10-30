import React, { useState } from 'react';
import { RiskPanelDashboard } from '../components/RiskPanelDashboard';
import { RiskPanel } from '../types/investor-report';

/**
 * RiskPanelDemoPage - Demonstration page for RiskPanelDashboard component
 * 
 * Shows the Risk Panel with mock data and regime controls.
 * This page can be used for:
 * - Visual testing of the component
 * - UI/UX validation
 * - Regime-aware styling verification
 */
const RiskPanelDemoPage: React.FC = () => {
  const [regime, setRegime] = useState<'bull' | 'bear' | 'neutral'>('neutral');
  const [loading, setLoading] = useState(false);

  // Mock risk panel data (Renaissance-level performance)
  const excellentRiskPanel: RiskPanel = {
    omega: 2.15,
    gh1: 1.68,
    pain_index: 4.2,
    upside_capture: 105.3,
    downside_capture: 42.8,
    cvar_95: -6.5,
    max_drawdown: -12.3,
    explanations: [
      'Omega ratio >2.0 indicates Renaissance-level performance with strong risk-adjusted returns',
      'Downside capture <50% shows excellent protection during market declines',
      'GH1 ratio >1.5 demonstrates superior alpha generation with controlled risk',
      'Pain index <5% indicates minimal investor suffering during drawdowns',
    ],
  };

  // Mock risk panel data (Good performance)
  const goodRiskPanel: RiskPanel = {
    omega: 1.65,
    gh1: 1.25,
    pain_index: 8.5,
    upside_capture: 95.2,
    downside_capture: 68.5,
    cvar_95: -9.2,
    max_drawdown: -18.7,
    explanations: [
      'Omega ratio >1.5 shows good risk-adjusted returns',
      'Downside capture <75% provides reasonable protection',
      'Upside capture near 100% indicates strong participation in rallies',
    ],
  };

  // Mock risk panel data (Fair performance)
  const fairRiskPanel: RiskPanel = {
    omega: 1.25,
    gh1: 0.85,
    pain_index: 15.3,
    upside_capture: 78.5,
    downside_capture: 92.3,
    cvar_95: -14.8,
    max_drawdown: -25.6,
    explanations: [
      'Omega ratio >1.0 shows positive risk-adjusted returns but room for improvement',
      'Downside capture near 100% indicates limited protection during declines',
      'Pain index >10% suggests significant investor discomfort during drawdowns',
    ],
  };

  const [selectedPanel, setSelectedPanel] = useState<'excellent' | 'good' | 'fair'>('excellent');

  const getRiskPanel = (): RiskPanel => {
    switch (selectedPanel) {
      case 'excellent':
        return excellentRiskPanel;
      case 'good':
        return goodRiskPanel;
      case 'fair':
        return fairRiskPanel;
    }
  };

  const handleLoadingToggle = () => {
    setLoading(true);
    setTimeout(() => setLoading(false), 2000);
  };

  return (
    <div className="min-h-screen bg-[#1a1a1a] p-8">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-8">
        <h1 className="text-4xl font-bold text-white mb-2">
          Risk Panel Dashboard - Demo
        </h1>
        <p className="text-gray-400">
          Institutional-grade risk metrics with regime-aware styling
        </p>
      </div>

      {/* Controls */}
      <div className="max-w-7xl mx-auto mb-6 p-4 bg-[#2a2a2a] border border-[#404040] rounded-lg">
        <h2 className="text-lg font-semibold text-white mb-4">Controls</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Regime Selector */}
          <div>
            <label className="block text-sm text-gray-400 mb-2">
              Market Regime
            </label>
            <div className="flex gap-2">
              <button
                onClick={() => setRegime('bull')}
                className={`px-4 py-2 rounded font-medium transition-colors ${
                  regime === 'bull'
                    ? 'bg-green-600 hover:bg-green-700 text-white'
                    : 'bg-gray-600 hover:bg-gray-700 text-white'
                }`}
              >
                Bull
              </button>
              <button
                onClick={() => setRegime('neutral')}
                className={`px-4 py-2 rounded font-medium transition-colors ${
                  regime === 'neutral'
                    ? 'bg-blue-600 hover:bg-blue-700 text-white'
                    : 'bg-gray-600 hover:bg-gray-700 text-white'
                }`}
              >
                Neutral
              </button>
              <button
                onClick={() => setRegime('bear')}
                className={`px-4 py-2 rounded font-medium transition-colors ${
                  regime === 'bear'
                    ? 'bg-red-600 hover:bg-red-700 text-white'
                    : 'bg-gray-600 hover:bg-gray-700 text-white'
                }`}
              >
                Bear
              </button>
            </div>
          </div>

          {/* Performance Level Selector */}
          <div>
            <label className="block text-sm text-gray-400 mb-2">
              Performance Level
            </label>
            <div className="flex gap-2">
              <button
                onClick={() => setSelectedPanel('excellent')}
                className={`px-4 py-2 rounded font-medium transition-colors ${
                  selectedPanel === 'excellent'
                    ? 'bg-green-600 hover:bg-green-700 text-white'
                    : 'bg-gray-600 hover:bg-gray-700 text-white'
                }`}
              >
                Excellent
              </button>
              <button
                onClick={() => setSelectedPanel('good')}
                className={`px-4 py-2 rounded font-medium transition-colors ${
                  selectedPanel === 'good'
                    ? 'bg-blue-600 hover:bg-blue-700 text-white'
                    : 'bg-gray-600 hover:bg-gray-700 text-white'
                }`}
              >
                Good
              </button>
              <button
                onClick={() => setSelectedPanel('fair')}
                className={`px-4 py-2 rounded font-medium transition-colors ${
                  selectedPanel === 'fair'
                    ? 'bg-orange-600 hover:bg-orange-700 text-white'
                    : 'bg-gray-600 hover:bg-gray-700 text-white'
                }`}
              >
                Fair
              </button>
            </div>
          </div>

          {/* Loading Toggle */}
          <div>
            <label className="block text-sm text-gray-400 mb-2">
              Loading State
            </label>
            <button
              onClick={handleLoadingToggle}
              className="px-4 py-2 rounded font-medium bg-purple-600 hover:bg-purple-700 text-white transition-colors"
            >
              Simulate Loading (2s)
            </button>
          </div>
        </div>
      </div>

      {/* RiskPanelDashboard */}
      <div className="max-w-7xl mx-auto">
        <RiskPanelDashboard
          riskPanel={getRiskPanel()}
          regime={regime}
          loading={loading}
        />
      </div>

      {/* Documentation */}
      <div className="max-w-7xl mx-auto mt-8 p-6 bg-[#2a2a2a] border border-[#404040] rounded-lg">
        <h2 className="text-2xl font-bold text-white mb-4">
          Component Documentation
        </h2>

        <div className="space-y-4 text-gray-300">
          <div>
            <h3 className="text-lg font-semibold text-white mb-2">
              Risk Metrics Explained
            </h3>
            <ul className="space-y-2 text-sm">
              <li>
                <strong className="text-green-400">Omega Ratio:</strong>{' '}
                Probability-weighted gains/losses. &gt;2.0 = Renaissance-level. Measures upside potential vs downside risk.
              </li>
              <li>
                <strong className="text-green-400">GH1 Ratio:</strong>{' '}
                Return enhancement + risk reduction vs benchmark. &gt;1.5 = strong alpha with controlled risk.
              </li>
              <li>
                <strong className="text-green-400">Pain Index:</strong>{' '}
                Drawdown depth × duration. Lower is better. &lt;5% = excellent risk management.
              </li>
              <li>
                <strong className="text-green-400">Upside Capture:</strong>{' '}
                % of benchmark gains captured. &gt;100% = outperformance during rallies.
              </li>
              <li>
                <strong className="text-green-400">Downside Capture:</strong>{' '}
                % of benchmark losses captured. &lt;100% = protection. &lt;50% = excellent downside protection.
              </li>
              <li>
                <strong className="text-green-400">CVaR 95%:</strong>{' '}
                Expected loss in worst 5% scenarios. Tail risk measure. More conservative than VaR.
              </li>
              <li>
                <strong className="text-green-400">Max Drawdown:</strong>{' '}
                Maximum peak-to-trough decline. &lt;10% = excellent capital preservation.
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-2">
              Regime-Aware Styling
            </h3>
            <p className="text-sm">
              The component adapts its visual styling based on the market regime:
            </p>
            <ul className="space-y-1 text-sm mt-2">
              <li>• <strong className="text-green-400">Bull Regime:</strong> Green accents, optimistic tone</li>
              <li>• <strong className="text-blue-400">Neutral Regime:</strong> Blue accents, balanced tone</li>
              <li>• <strong className="text-red-400">Bear Regime:</strong> Red accents, defensive tone</li>
            </ul>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-2">
              Usage Example
            </h3>
            <pre className="p-4 bg-[#1a1a1a] border border-[#404040] rounded overflow-x-auto text-sm">
              <code className="text-gray-300">{`import { RiskPanelDashboard } from './components/RiskPanelDashboard';

function MyDashboard() {
  const riskPanel = {
    omega: 2.15,
    gh1: 1.68,
    pain_index: 4.2,
    upside_capture: 105.3,
    downside_capture: 42.8,
    cvar_95: -6.5,
    max_drawdown: -12.3,
    explanations: ['...'],
  };

  return (
    <RiskPanelDashboard
      riskPanel={riskPanel}
      regime="bull"
      loading={false}
    />
  );
}`}</code>
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RiskPanelDemoPage;

