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
                <a href="/options-chain" className="text-green-600 hover:text-green-800 font-medium">⚡ Options Chain</a>
                <a href="/options-analytics" className="text-green-600 hover:text-green-800 font-medium">📉 Analytics</a>
                <a href="/risk-dashboard" className="text-green-600 hover:text-green-800 font-medium">🛡️ Risk Dashboard</a>
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
            <Route path="/options-chain" element={<OptionsChainPage />} />
            <Route path="/options-analytics" element={<OptionsAnalyticsPage />} />
            <Route path="/risk-dashboard" element={<RiskDashboardPage />} />
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

