import React, { useState, useEffect } from 'react';
import { getUpcomingWeek } from '../services/calendarApi';
import { getExecutionAnalysis } from '../services/executionApi';
import toast from 'react-hot-toast';

interface Widget {
  id: string;
  type: 'upcoming_events' | 'execution_summary' | 'quick_stats' | 'watchlist' | 'recent_news';
  title: string;
  size: 'small' | 'medium' | 'large';
  enabled: boolean;
}

const CustomDashboardPage: React.FC = () => {
  const [widgets, setWidgets] = useState<Widget[]>([
    { id: 'w1', type: 'quick_stats', title: 'Quick Stats', size: 'medium', enabled: true },
    { id: 'w2', type: 'upcoming_events', title: 'Upcoming Events', size: 'large', enabled: true },
    { id: 'w3', type: 'execution_summary', title: 'Execution Quality', size: 'medium', enabled: true },
    { id: 'w4', type: 'watchlist', title: 'Watchlist', size: 'small', enabled: true },
  ]);

  const [upcomingEvents, setUpcomingEvents] = useState<any>(null);
  const [executionSummary, setExecutionSummary] = useState<any>(null);
  const [watchlist, setWatchlist] = useState<string[]>(['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA']);

  useEffect(() => {
    loadWidgetData();
  }, []);

  const loadWidgetData = async () => {
    try {
      // Load upcoming events
      const events = await getUpcomingWeek();
      setUpcomingEvents(events);

      // Load execution summary
      const execution = await getExecutionAnalysis();
      setExecutionSummary(execution);
    } catch (error) {
      console.error('Failed to load widget data:', error);
    }
  };

  const toggleWidget = (id: string) => {
    setWidgets(widgets.map(w =>
      w.id === id ? { ...w, enabled: !w.enabled } : w
    ));
  };

  const removeSymbolFromWatchlist = (symbol: string) => {
    setWatchlist(watchlist.filter(s => s !== symbol));
  };

  const addSymbolToWatchlist = () => {
    const symbol = prompt('Enter symbol to add:');
    if (symbol && !watchlist.includes(symbol.toUpperCase())) {
      setWatchlist([...watchlist, symbol.toUpperCase()]);
    }
  };

  const renderWidget = (widget: Widget) => {
    if (!widget.enabled) return null;

    const sizeClasses = {
      small: 'col-span-1',
      medium: 'col-span-2',
      large: 'col-span-3'
    };

    switch (widget.type) {
      case 'quick_stats':
        return (
          <div key={widget.id} className={`${sizeClasses[widget.size]} bg-white rounded-lg shadow p-6`}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-gray-900">{widget.title}</h3>
              <button
                onClick={() => toggleWidget(widget.id)}
                className="text-gray-400 hover:text-gray-600"
              >
                ✕
              </button>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 bg-blue-50 rounded">
                <div className="text-sm text-gray-600">Active Positions</div>
                <div className="text-2xl font-bold text-blue-700">12</div>
              </div>
              <div className="p-3 bg-green-50 rounded">
                <div className="text-sm text-gray-600">Today's P&L</div>
                <div className="text-2xl font-bold text-green-700">+$1,245</div>
              </div>
              <div className="p-3 bg-purple-50 rounded">
                <div className="text-sm text-gray-600">Win Rate (30d)</div>
                <div className="text-2xl font-bold text-purple-700">68%</div>
              </div>
              <div className="p-3 bg-yellow-50 rounded">
                <div className="text-sm text-gray-600">Total Value</div>
                <div className="text-2xl font-bold text-yellow-700">$125K</div>
              </div>
            </div>
          </div>
        );

      case 'upcoming_events':
        return (
          <div key={widget.id} className={`${sizeClasses[widget.size]} bg-white rounded-lg shadow p-6`}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-gray-900">{widget.title}</h3>
              <button
                onClick={() => toggleWidget(widget.id)}
                className="text-gray-400 hover:text-gray-600"
              >
                ✕
              </button>
            </div>
            {upcomingEvents ? (
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {upcomingEvents.days?.slice(0, 5).map((day: any) => (
                  <div key={day.date} className="p-3 bg-gray-50 rounded border-l-4 border-blue-500">
                    <div className="font-semibold text-sm">{day.date}</div>
                    <div className="text-xs text-gray-600 mt-1">
                      {day.economic_events?.length || 0} economic events, {day.earnings_events?.length || 0} earnings
                    </div>
                    {day.high_importance_count > 0 && (
                      <span className="text-xs px-2 py-0.5 bg-red-100 text-red-800 rounded mt-1 inline-block">
                        {day.high_importance_count} high impact
                      </span>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-gray-500 text-sm">Loading events...</div>
            )}
          </div>
        );

      case 'execution_summary':
        return (
          <div key={widget.id} className={`${sizeClasses[widget.size]} bg-white rounded-lg shadow p-6`}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-gray-900">{widget.title}</h3>
              <button
                onClick={() => toggleWidget(widget.id)}
                className="text-gray-400 hover:text-gray-600"
              >
                ✕
              </button>
            </div>
            {executionSummary && executionSummary.overall_metrics.total_executions > 0 ? (
              <div className="space-y-3">
                <div className="flex justify-between items-center p-2 bg-gray-50 rounded">
                  <span className="text-sm text-gray-600">Avg Slippage</span>
                  <span className={`font-bold ${
                    executionSummary.overall_metrics.avg_slippage_bps < 3
                      ? 'text-green-700'
                      : executionSummary.overall_metrics.avg_slippage_bps < 7
                      ? 'text-yellow-700'
                      : 'text-red-700'
                  }`}>
                    {executionSummary.overall_metrics.avg_slippage_bps.toFixed(1)} bps
                  </span>
                </div>
                <div className="flex justify-between items-center p-2 bg-gray-50 rounded">
                  <span className="text-sm text-gray-600">Fill Rate</span>
                  <span className="font-bold text-blue-700">
                    {executionSummary.overall_metrics.fill_rate.toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between items-center p-2 bg-gray-50 rounded">
                  <span className="text-sm text-gray-600">Total Cost</span>
                  <span className="font-bold text-red-700">
                    ${executionSummary.overall_metrics.total_slippage_cost.toFixed(2)}
                  </span>
                </div>
                <a
                  href="/execution"
                  className="block text-center text-sm text-blue-600 hover:text-blue-800 mt-2"
                >
                  View Full Analysis →
                </a>
              </div>
            ) : (
              <div className="text-gray-500 text-sm">No execution data available</div>
            )}
          </div>
        );

      case 'watchlist':
        return (
          <div key={widget.id} className={`${sizeClasses[widget.size]} bg-white rounded-lg shadow p-6`}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-gray-900">{widget.title}</h3>
              <button
                onClick={() => toggleWidget(widget.id)}
                className="text-gray-400 hover:text-gray-600"
              >
                ✕
              </button>
            </div>
            <div className="space-y-2 mb-3">
              {watchlist.map(symbol => (
                <div key={symbol} className="flex items-center justify-between p-2 bg-gray-50 rounded hover:bg-gray-100">
                  <a
                    href={`/options-chain?symbol=${symbol}`}
                    className="font-mono font-bold text-blue-600 hover:text-blue-800"
                  >
                    {symbol}
                  </a>
                  <button
                    onClick={() => removeSymbolFromWatchlist(symbol)}
                    className="text-red-500 hover:text-red-700 text-xs"
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
            <button
              onClick={addSymbolToWatchlist}
              className="w-full px-3 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-700"
            >
              + Add Symbol
            </button>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">
          📊 Custom Dashboard
        </h1>
        <p className="text-gray-600 mt-2">
          Personalized trading dashboard - Add widgets for the metrics you care about
        </p>
      </div>

      {/* Widget Controls */}
      <div className="bg-white rounded-lg shadow p-4 mb-6">
        <div className="flex flex-wrap gap-2">
          <span className="text-sm font-medium text-gray-700 mr-2">Available Widgets:</span>
          {widgets.map(widget => (
            <button
              key={widget.id}
              onClick={() => toggleWidget(widget.id)}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                widget.enabled
                  ? 'bg-green-100 text-green-800 hover:bg-green-200'
                  : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
              }`}
            >
              {widget.title} {widget.enabled ? '✓' : '+'}
            </button>
          ))}
          <button
            onClick={loadWidgetData}
            className="ml-auto px-4 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700"
          >
            ↻ Refresh All
          </button>
        </div>
      </div>

      {/* Dashboard Grid */}
      <div className="grid grid-cols-3 gap-6">
        {widgets.map(widget => renderWidget(widget))}
      </div>

      {/* Helpful Info */}
      <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="text-sm text-gray-700">
          <span className="font-semibold">💡 Pro Tips:</span>
          <ul className="list-disc list-inside mt-2 space-y-1">
            <li>Click widget titles to hide/show them</li>
            <li>Watchlist links directly to options chains</li>
            <li>Dashboard refreshes automatically on page load</li>
            <li>More widget types coming soon (Charts, Positions, P&L)</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default CustomDashboardPage;
