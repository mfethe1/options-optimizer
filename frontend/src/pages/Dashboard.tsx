import React, { useEffect, useState } from 'react';
import PortfolioSummary from '../components/PortfolioSummary';
import GreeksDisplay from '../components/GreeksDisplay';
import RiskScore from '../components/RiskScore';
import RecentAlerts from '../components/RecentAlerts';
import QuickActions from '../components/QuickActions';
import { Loader2 } from 'lucide-react';

const Dashboard: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [positions, setPositions] = useState<any[]>([]);
  const [portfolioGreeks, setPortfolioGreeks] = useState<any>(null);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setIsLoading(true);
    try {
      // Load positions from API
      const response = await fetch('http://localhost:8000/api/positions/options');
      if (response.ok) {
        const data = await response.json();
        setPositions(data);

        // Calculate portfolio Greeks
        const greeks = {
          delta: data.reduce((sum: number, p: any) => sum + (p.delta || 0), 0),
          gamma: data.reduce((sum: number, p: any) => sum + (p.gamma || 0), 0),
          theta: data.reduce((sum: number, p: any) => sum + (p.theta || 0), 0),
          vega: data.reduce((sum: number, p: any) => sum + (p.vega || 0), 0),
          rho: data.reduce((sum: number, p: any) => sum + (p.rho || 0), 0),
        };
        setPortfolioGreeks(greeks);
      }
    } catch (error) {
      console.error('Error loading dashboard data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600 mt-1">
            Welcome back! Here's your portfolio overview.
          </p>
        </div>
        <QuickActions />
      </div>

      {/* Portfolio Summary */}
      <PortfolioSummary positions={positions} />

      {/* Greeks and Risk Score */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <GreeksDisplay greeks={portfolioGreeks} />
        <RiskScore score={45} />
      </div>

      {/* Recent Alerts */}
      <RecentAlerts />
    </div>
  );
};

export default Dashboard;

