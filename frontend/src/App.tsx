import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import PositionsPage from './pages/PositionsPage';
import SwarmAnalysisPage from './pages/SwarmAnalysisPage';
import Phase4DemoPage from './pages/Phase4DemoPage';
import RiskPanelDemoPage from './pages/RiskPanelDemoPage';
import AgentTransparencyDemoPage from './pages/AgentTransparencyDemoPage';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <nav className="bg-white shadow-sm">
          <div className="container mx-auto px-4 py-4">
            <div className="flex gap-4">
              <a href="/" className="text-blue-600 hover:text-blue-800">Dashboard</a>
              <a href="/positions" className="text-blue-600 hover:text-blue-800">Positions</a>
              <a href="/swarm-analysis" className="text-blue-600 hover:text-blue-800">AI Swarm Analysis</a>
              <a href="/phase4-demo" className="text-blue-600 hover:text-blue-800">Phase 4 Signals</a>
              <a href="/risk-panel-demo" className="text-blue-600 hover:text-blue-800">Risk Panel</a>
              <a href="/agent-transparency" className="text-blue-600 hover:text-blue-800">Agent Transparency</a>
            </div>
          </div>
        </nav>

        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/positions" element={<PositionsPage />} />
            <Route path="/swarm-analysis" element={<SwarmAnalysisPage />} />
            <Route path="/phase4-demo" element={<Phase4DemoPage />} />
            <Route path="/risk-panel-demo" element={<RiskPanelDemoPage />} />
            <Route path="/agent-transparency" element={<AgentTransparencyDemoPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;

