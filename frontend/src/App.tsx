import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
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

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <nav className="bg-white shadow-sm">
          <div className="container mx-auto px-4 py-4">
            <div className="flex flex-wrap gap-4">
              <a href="/" className="text-blue-600 hover:text-blue-800 font-medium">Dashboard</a>
              <a href="/positions" className="text-blue-600 hover:text-blue-800">Positions</a>

              {/* New Features */}
              <div className="flex gap-4 border-l border-gray-300 pl-4">
                <a href="/options-chain" className="text-green-600 hover:text-green-800 font-medium">⚡ Options Chain</a>
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
          </div>
        </nav>

        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/positions" element={<PositionsPage />} />

            {/* New Feature Routes */}
            <Route path="/options-chain" element={<OptionsChainPage />} />
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
    </Router>
  );
}

export default App;

