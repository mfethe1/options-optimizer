import React, { useState } from 'react';
import { Phase4SignalsPanel } from '../components/Phase4SignalsPanel';
import { Phase4Tech } from '../types/investor-report';

/**
 * Phase4DemoPage - Demonstration page for Phase4SignalsPanel component
 * 
 * Shows the Phase 4 Signals panel with mock data and real-time WebSocket updates.
 * This page can be used for:
 * - Visual testing of the component
 * - Integration testing with WebSocket
 * - UI/UX validation
 */
const Phase4DemoPage: React.FC = () => {
  const [userId, setUserId] = useState('demo_user');
  const [useInitialData, setUseInitialData] = useState(true);

  // Mock initial data for testing
  const mockInitialData: Phase4Tech = {
    options_flow_composite: 0.65,
    residual_momentum: 1.85,
    seasonality_score: 0.42,
    breadth_liquidity: 0.78,
    explanations: [
      'Strong options flow indicates bullish sentiment with high call volume',
      'Residual momentum shows outperformance vs sector after removing market effects',
      'Seasonality score positive due to turn-of-month effect',
      'Breadth & liquidity strong with 80% advance/decline ratio',
    ],
  };

  return (
    <div className="min-h-screen bg-[#1a1a1a] p-8">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-8">
        <h1 className="text-4xl font-bold text-white mb-2">
          Phase 4 Signals - Demo
        </h1>
        <p className="text-gray-400">
          Real-time short-horizon (1-5 day) technical & cross-asset signals
        </p>
      </div>

      {/* Controls */}
      <div className="max-w-7xl mx-auto mb-6 p-4 bg-[#2a2a2a] border border-[#404040] rounded-lg">
        <h2 className="text-lg font-semibold text-white mb-4">Controls</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* User ID Input */}
          <div>
            <label className="block text-sm text-gray-400 mb-2">
              User ID (for WebSocket connection)
            </label>
            <input
              type="text"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              className="w-full px-3 py-2 bg-[#1a1a1a] border border-[#404040] rounded text-white focus:outline-none focus:border-[#10b981]"
              placeholder="Enter user ID"
            />
          </div>

          {/* Initial Data Toggle */}
          <div>
            <label className="block text-sm text-gray-400 mb-2">
              Use Initial Data (fallback when WebSocket unavailable)
            </label>
            <button
              onClick={() => setUseInitialData(!useInitialData)}
              className={`px-4 py-2 rounded font-medium transition-colors ${
                useInitialData
                  ? 'bg-green-600 hover:bg-green-700 text-white'
                  : 'bg-gray-600 hover:bg-gray-700 text-white'
              }`}
            >
              {useInitialData ? 'Enabled' : 'Disabled'}
            </button>
          </div>
        </div>

        {/* Info */}
        <div className="mt-4 p-3 bg-blue-900/20 border border-blue-500/30 rounded">
          <p className="text-sm text-gray-300">
            💡 <strong>Tip:</strong> The panel will attempt to connect to WebSocket at{' '}
            <code className="text-blue-400">ws://localhost:8000/ws/phase4-metrics/{userId}</code>.
            If the connection fails, it will fall back to initial data (if enabled).
          </p>
        </div>
      </div>

      {/* Phase4SignalsPanel */}
      <div className="max-w-7xl mx-auto">
        <Phase4SignalsPanel
          userId={userId}
          initialData={useInitialData ? mockInitialData : undefined}
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
              Metrics Explained
            </h3>
            <ul className="space-y-2 text-sm">
              <li>
                <strong className="text-green-400">Options Flow Composite:</strong>{' '}
                Combines Put/Call Ratio, IV skew, and volume. Range: -1 (bearish) to +1 (bullish).
              </li>
              <li>
                <strong className="text-green-400">Residual Momentum:</strong>{' '}
                Idiosyncratic returns after removing market/sector effects. Z-score scale. &gt;2σ = strong alpha.
              </li>
              <li>
                <strong className="text-green-400">Seasonality Score:</strong>{' '}
                Calendar patterns including turn-of-month and day-of-week effects. Range: -1 to +1.
              </li>
              <li>
                <strong className="text-green-400">Breadth & Liquidity:</strong>{' '}
                Market internals: A/D ratio (50%), volume (30%), spreads (20%). Range: 0 to 100%.
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-2">
              Color Coding
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
              <div className="p-2 bg-green-900/40 border border-green-500/50 rounded">
                <span className="text-green-400 font-semibold">Green:</span> Excellent
              </div>
              <div className="p-2 bg-green-900/20 border border-green-500/30 rounded">
                <span className="text-green-300 font-semibold">Light Green:</span> Good
              </div>
              <div className="p-2 bg-orange-900/40 border border-orange-500/50 rounded">
                <span className="text-orange-400 font-semibold">Orange:</span> Warning
              </div>
              <div className="p-2 bg-red-900/40 border border-red-500/50 rounded">
                <span className="text-red-400 font-semibold">Red:</span> Critical
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-2">
              Real-Time Updates
            </h3>
            <p className="text-sm">
              The panel connects to the WebSocket endpoint and receives updates every 30 seconds
              (configurable via <code className="text-blue-400">PHASE4_WS_INTERVAL_SECONDS</code>).
              The connection status is indicated by the live/disconnected indicator in the top-right corner.
            </p>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-2">
              Usage Example
            </h3>
            <pre className="p-4 bg-[#1a1a1a] border border-[#404040] rounded overflow-x-auto text-sm">
              <code className="text-gray-300">{`import { Phase4SignalsPanel } from './components/Phase4SignalsPanel';

function MyDashboard() {
  return (
    <Phase4SignalsPanel
      userId="user_123"
      initialData={mockData} // Optional fallback
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

export default Phase4DemoPage;

