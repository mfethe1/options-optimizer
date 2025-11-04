import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Dashboard from './pages/Dashboard';
import PositionsPage from './pages/PositionsPage';
import SwarmAnalysisPage from './pages/SwarmAnalysisPage';
import Phase4DemoPage from './pages/Phase4DemoPage';
import RiskPanelDemoPage from './pages/RiskPanelDemoPage';
import AgentTransparencyDemoPage from './pages/AgentTransparencyDemoPage';
import ConversationalTradingPage from './pages/ConversationalTradingPage';
import ChartAnalysisPage from './pages/ChartAnalysisPage';
import AnomalyDetectionPage from './pages/AnomalyDetectionPage';
import SentimentAnalysisPage from './pages/SentimentAnalysisPage';
import PaperTradingPage from './pages/PaperTradingPage';
import OptionsChainPage from './pages/OptionsChainPage';
import RiskDashboardPage from './pages/RiskDashboardPage';
import NewsFeedPage from './pages/NewsFeedPage';
import OptionsAnalyticsPage from './pages/OptionsAnalyticsPage';
import EconomicCalendarPage from './pages/EconomicCalendarPage';
import BacktestPage from './pages/BacktestPage';
import ExecutionQualityPage from './pages/ExecutionQualityPage';
import CustomDashboardPage from './pages/CustomDashboardPage';
import SchwabConnectionPage from './pages/SchwabConnectionPage';
import SchwabTradingPage from './pages/SchwabTradingPage';
import MultiMonitorPage from './pages/MultiMonitorPage';
import AIRecommendationsPage from './pages/AIRecommendationsPage';
import RealTimeQuotePage from './pages/RealTimeQuotePage';
import SmartRoutingPage from './pages/SmartRoutingPage';
import MLPredictionsPage from './pages/MLPredictionsPage';
import StressTestingPage from './pages/StressTestingPage';
import BrokerManagementPage from './pages/BrokerManagementPage';
import EpidemicVolatilityPage from './pages/BioFinancial/EpidemicVolatilityPage';
import GNNPage from './pages/AdvancedForecasting/GNNPage';
import MambaPage from './pages/AdvancedForecasting/MambaPage';
import PINNPage from './pages/AdvancedForecasting/PINNPage';
import CommandPalette from './components/CommandPalette';

function AppContent() {
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Command Palette (Ctrl+K or Cmd+K)
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        setCommandPaletteOpen(true);
      }

      // Navigation shortcuts (only when command palette is closed)
      if (!commandPaletteOpen && (e.ctrlKey || e.metaKey)) {
        switch (e.key.toLowerCase()) {
          case 'd':
            e.preventDefault();
            navigate('/');
            break;
          case 'o':
            e.preventDefault();
            navigate('/options-chain');
            break;
          case 'r':
            e.preventDefault();
            navigate('/risk-dashboard');
            break;
          case 'n':
            e.preventDefault();
            navigate('/news');
            break;
          case 'c':
            e.preventDefault();
            navigate('/chart-analysis');
            break;
          case 's':
            e.preventDefault();
            navigate('/sentiment');
            break;
          case 'a':
            e.preventDefault();
            navigate('/anomalies');
            break;
          case 't':
            e.preventDefault();
            navigate('/paper-trading');
            break;
          case 'p':
            e.preventDefault();
            navigate('/positions');
            break;
          case 'v':
            e.preventDefault();
            navigate('/options-analytics');
            break;
          case 'e':
            e.preventDefault();
            navigate('/calendar');
            break;
          case 'b':
            e.preventDefault();
            navigate('/backtest');
            break;
          case 'x':
            e.preventDefault();
            navigate('/execution');
            break;
          case 'w':
            e.preventDefault();
            navigate('/custom-dashboard');
            break;
          case 'l':
            e.preventDefault();
            navigate('/schwab-connection');
            break;
          case 'u':
            e.preventDefault();
            navigate('/schwab-trading');
            break;
          case 'm':
            e.preventDefault();
            navigate('/multi-monitor');
            break;
          case 'i':
            e.preventDefault();
            navigate('/ai-recommendations');
            break;
          case 'q':
            e.preventDefault();
            navigate('/market-data');
            break;
          case 'j':
            e.preventDefault();
            navigate('/smart-routing');
            break;
          case 'y':
            e.preventDefault();
            navigate('/ml-predictions');
            break;
          case 'z':
            e.preventDefault();
            navigate('/stress-testing');
            break;
          case 'f':
            e.preventDefault();
            navigate('/broker-management');
            break;
          case 'g':
            e.preventDefault();
            navigate('/epidemic-volatility');
            break;
          case 'h':
            e.preventDefault();
            navigate('/mamba');
            break;
          case '1':
            e.preventDefault();
            navigate('/pinn');
            break;
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [commandPaletteOpen, navigate]);

  return (
      <div className="min-h-screen bg-gray-50">
        {/* Command Palette */}
        <CommandPalette
          isOpen={commandPaletteOpen}
          onClose={() => setCommandPaletteOpen(false)}
        />

        <nav className="bg-white shadow-sm">
          <div className="container mx-auto px-4 py-4">
            <div className="flex flex-wrap gap-4 items-center justify-between">
              <div className="flex flex-wrap gap-4">
              <a href="/" className="text-blue-600 hover:text-blue-800 font-medium">Dashboard</a>
              <a href="/positions" className="text-blue-600 hover:text-blue-800">Positions</a>

              {/* New Features */}
              <div className="flex gap-4 border-l border-gray-300 pl-4">
                <a href="/custom-dashboard" className="text-green-600 hover:text-green-800 font-medium">📊 Dashboard</a>
                <a href="/options-chain" className="text-green-600 hover:text-green-800 font-medium">⚡ Options Chain</a>
                <a href="/options-analytics" className="text-green-600 hover:text-green-800 font-medium">📉 Analytics</a>
                <a href="/risk-dashboard" className="text-green-600 hover:text-green-800 font-medium">🛡️ Risk Dashboard</a>
                <a href="/calendar" className="text-green-600 hover:text-green-800 font-medium">📅 Calendar</a>
                <a href="/backtest" className="text-green-600 hover:text-green-800 font-medium">📊 Backtest</a>
                <a href="/execution" className="text-green-600 hover:text-green-800 font-medium">🎯 Execution</a>
                <a href="/schwab-connection" className="text-purple-600 hover:text-purple-800 font-bold">🔗 Schwab</a>
                <a href="/schwab-trading" className="text-purple-600 hover:text-purple-800 font-bold">💰 Trade</a>
                <a href="/multi-monitor" className="text-indigo-600 hover:text-indigo-800 font-bold">🖥️ Multi-Monitor</a>
                <a href="/ai-recommendations" className="text-pink-600 hover:text-pink-800 font-bold">🤖 AI Insights</a>
                <a href="/market-data" className="text-orange-600 hover:text-orange-800 font-bold">⚡ Live Data</a>
                <a href="/smart-routing" className="text-purple-600 hover:text-purple-800 font-bold">🎯 Smart Routing</a>
                <a href="/ml-predictions" className="text-pink-600 hover:text-pink-800 font-bold">🧠 ML Predictions</a>
                <a href="/stress-testing" className="text-red-600 hover:text-red-800 font-bold">🛡️ Stress Testing</a>
                <a href="/broker-management" className="text-indigo-600 hover:text-indigo-800 font-bold">🔗 Brokers</a>
                <a href="/epidemic-volatility" className="text-purple-600 hover:text-purple-800 font-bold">🦠 Epidemic Vol</a>
                <a href="/gnn" className="text-blue-600 hover:text-blue-800 font-bold">📊 GNN</a>
                <a href="/mamba" className="text-green-600 hover:text-green-800 font-bold">⚡ Mamba</a>
                <a href="/pinn" className="text-indigo-600 hover:text-indigo-800 font-bold">🧬 PINN</a>
                <a href="/news" className="text-green-600 hover:text-green-800 font-medium">📰 News</a>
                <a href="/conversational" className="text-green-600 hover:text-green-800 font-medium">💬 Chat</a>
                <a href="/chart-analysis" className="text-green-600 hover:text-green-800 font-medium">📊 Charts</a>
                <a href="/anomalies" className="text-green-600 hover:text-green-800 font-medium">🚨 Anomalies</a>
                <a href="/sentiment" className="text-green-600 hover:text-green-800 font-medium">📈 Sentiment</a>
                <a href="/paper-trading" className="text-green-600 hover:text-green-800 font-medium">🤖 Paper Trading</a>
              </div>

              {/* Original Features */}
              <div className="flex gap-4 border-l border-gray-300 pl-4">
                <a href="/swarm-analysis" className="text-blue-600 hover:text-blue-800">AI Swarm</a>
                <a href="/phase4-demo" className="text-blue-600 hover:text-blue-800">Phase 4</a>
                <a href="/risk-panel-demo" className="text-blue-600 hover:text-blue-800">Risk Panel</a>
                <a href="/agent-transparency" className="text-blue-600 hover:text-blue-800">Transparency</a>
              </div>
              </div>

              {/* Command Palette Button */}
              <button
                onClick={() => setCommandPaletteOpen(true)}
                className="px-4 py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-700 flex items-center gap-2"
                title="Command Palette (Ctrl+K)"
              >
                <span>⌘</span>
                <span>K</span>
              </button>
            </div>
          </div>
        </nav>

        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/positions" element={<PositionsPage />} />

            {/* New Feature Routes */}
            <Route path="/custom-dashboard" element={<CustomDashboardPage />} />
            <Route path="/options-chain" element={<OptionsChainPage />} />
            <Route path="/options-analytics" element={<OptionsAnalyticsPage />} />
            <Route path="/risk-dashboard" element={<RiskDashboardPage />} />
            <Route path="/calendar" element={<EconomicCalendarPage />} />
            <Route path="/backtest" element={<BacktestPage />} />
            <Route path="/execution" element={<ExecutionQualityPage />} />
            <Route path="/schwab-connection" element={<SchwabConnectionPage />} />
            <Route path="/schwab-trading" element={<SchwabTradingPage />} />
            <Route path="/multi-monitor" element={<MultiMonitorPage />} />
            <Route path="/ai-recommendations" element={<AIRecommendationsPage />} />
            <Route path="/market-data" element={<RealTimeQuotePage />} />
            <Route path="/smart-routing" element={<SmartRoutingPage />} />
            <Route path="/ml-predictions" element={<MLPredictionsPage />} />
            <Route path="/stress-testing" element={<StressTestingPage />} />
            <Route path="/broker-management" element={<BrokerManagementPage />} />
            <Route path="/epidemic-volatility" element={<EpidemicVolatilityPage />} />
            <Route path="/gnn" element={<GNNPage />} />
            <Route path="/mamba" element={<MambaPage />} />
            <Route path="/pinn" element={<PINNPage />} />
            <Route path="/news" element={<NewsFeedPage />} />
            <Route path="/conversational" element={<ConversationalTradingPage />} />
            <Route path="/chart-analysis" element={<ChartAnalysisPage />} />
            <Route path="/anomalies" element={<AnomalyDetectionPage />} />
            <Route path="/sentiment" element={<SentimentAnalysisPage />} />
            <Route path="/paper-trading" element={<PaperTradingPage />} />

            {/* Original Routes */}
            <Route path="/swarm-analysis" element={<SwarmAnalysisPage />} />
            <Route path="/phase4-demo" element={<Phase4DemoPage />} />
            <Route path="/risk-panel-demo" element={<RiskPanelDemoPage />} />
            <Route path="/agent-transparency" element={<AgentTransparencyDemoPage />} />
          </Routes>
        </main>

        {/* Toast Notifications */}
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#363636',
              color: '#fff',
            },
          }}
        />
      </div>
  );
}

function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

export default App;

