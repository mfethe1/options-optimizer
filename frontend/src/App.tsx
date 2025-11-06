import { useState, useEffect } from 'react';
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
// Individual ML model pages consolidated into UnifiedAnalysis
import ChartsDemo from './pages/ChartsDemo';
import UnifiedAnalysis from './pages/UnifiedAnalysis';
import UnifiedAnalysisEnhanced from './pages/UnifiedAnalysisEnhanced';
import CommandPalette from './components/CommandPalette';
import NavigationSidebar from './components/NavigationSidebar';

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
          // ML models integrated into unified analysis (/ or /unified)
          // Removed individual shortcuts for epidemic, gnn, mamba, pinn, ensemble
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [commandPaletteOpen, navigate]);

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Command Palette */}
      <CommandPalette
        isOpen={commandPaletteOpen}
        onClose={() => setCommandPaletteOpen(false)}
      />

      {/* Sidebar Navigation */}
      <NavigationSidebar />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto p-6 bg-gray-50">
          <Routes>
            <Route path="/" element={<UnifiedAnalysisEnhanced />} />
            <Route path="/unified" element={<UnifiedAnalysis />} />
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
            {/* All ML models now integrated into Unified Analysis page */}
            <Route path="/charts-demo" element={<ChartsDemo />} />
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
      </div>

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

