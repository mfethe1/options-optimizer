import React from 'react';
import { TrendingUp, TrendingDown, DollarSign, Target } from 'lucide-react';

interface Position {
  market_value?: number;
  pnl?: number;
  pnl_pct?: number;
}

interface Props {
  positions: Position[];
}

const PortfolioSummary: React.FC<Props> = ({ positions }) => {
  const totalValue = positions.reduce((sum, p) => sum + (p.market_value || 0), 0);
  const totalPnL = positions.reduce((sum, p) => sum + (p.pnl || 0), 0);
  const avgPnLPct = positions.length > 0
    ? positions.reduce((sum, p) => sum + (p.pnl_pct || 0), 0) / positions.length
    : 0;
  const winningPositions = positions.filter(p => (p.pnl || 0) > 0).length;

  const stats = [
    {
      label: 'Total Portfolio Value',
      value: `$${totalValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
      icon: DollarSign,
      color: 'blue',
    },
    {
      label: 'Total P&L',
      value: `$${totalPnL.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
      icon: totalPnL >= 0 ? TrendingUp : TrendingDown,
      color: totalPnL >= 0 ? 'green' : 'red',
      change: `${avgPnLPct.toFixed(2)}%`,
    },
    {
      label: 'Active Positions',
      value: positions.length.toString(),
      icon: Target,
      color: 'purple',
    },
    {
      label: 'Win Rate',
      value: positions.length > 0
        ? `${((winningPositions / positions.length) * 100).toFixed(1)}%`
        : '0%',
      icon: TrendingUp,
      color: 'green',
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {stats.map((stat, index) => {
        const Icon = stat.icon;
        const colorClasses = {
          blue: 'bg-blue-100 text-blue-600',
          green: 'bg-green-100 text-green-600',
          red: 'bg-red-100 text-red-600',
          purple: 'bg-purple-100 text-purple-600',
        };

        return (
          <div key={index} className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <p className="text-sm font-medium text-gray-600">{stat.label}</p>
                <p className="text-2xl font-bold text-gray-900 mt-2">
                  {stat.value}
                </p>
                {stat.change && (
                  <p className={`text-sm mt-1 ${
                    totalPnL >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {stat.change}
                  </p>
                )}
              </div>
              <div className={`p-3 rounded-full ${colorClasses[stat.color as keyof typeof colorClasses]}`}>
                <Icon className="w-6 h-6" />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default PortfolioSummary;

