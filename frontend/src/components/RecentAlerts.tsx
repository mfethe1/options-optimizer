import React from 'react';
import { AlertCircle, TrendingUp, TrendingDown } from 'lucide-react';

const RecentAlerts: React.FC = () => {
  // Mock alerts - in production, fetch from API
  const alerts = [
    {
      id: 1,
      type: 'warning',
      message: 'NVDA option expiring in 3 days',
      timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
    },
    {
      id: 2,
      type: 'info',
      message: 'Portfolio delta increased by 15%',
      timestamp: new Date(Date.now() - 5 * 60 * 60 * 1000),
    },
    {
      id: 3,
      type: 'success',
      message: 'AAPL position up 8% today',
      timestamp: new Date(Date.now() - 8 * 60 * 60 * 1000),
    },
  ];

  const getIcon = (type: string) => {
    switch (type) {
      case 'warning':
        return <AlertCircle className="w-5 h-5 text-yellow-600" />;
      case 'success':
        return <TrendingUp className="w-5 h-5 text-green-600" />;
      default:
        return <TrendingDown className="w-5 h-5 text-blue-600" />;
    }
  };

  const getTimeAgo = (date: Date) => {
    const hours = Math.floor((Date.now() - date.getTime()) / (1000 * 60 * 60));
    if (hours < 1) return 'Just now';
    if (hours === 1) return '1 hour ago';
    return `${hours} hours ago`;
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold mb-4">Recent Alerts</h2>
      <div className="space-y-3">
        {alerts.map((alert) => (
          <div key={alert.id} className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
            {getIcon(alert.type)}
            <div className="flex-1">
              <p className="text-sm text-gray-900">{alert.message}</p>
              <p className="text-xs text-gray-500 mt-1">{getTimeAgo(alert.timestamp)}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default RecentAlerts;

