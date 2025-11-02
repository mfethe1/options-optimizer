import React, { useState, useEffect } from 'react';
import { getExecutionAnalysis, generateMockData, ExecutionAnalysis } from '../services/executionApi';
import toast from 'react-hot-toast';

const ExecutionQualityPage: React.FC = () => {
  const [analysis, setAnalysis] = useState<ExecutionAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [symbolFilter, setSymbolFilter] = useState('');
  const [brokerFilter, setBrokerFilter] = useState('');

  useEffect(() => {
    loadAnalysis();
  }, []);

  const loadAnalysis = async () => {
    setLoading(true);
    try {
      const result = await getExecutionAnalysis(
        startDate || undefined,
        endDate || undefined,
        symbolFilter || undefined,
        brokerFilter || undefined
      );
      setAnalysis(result);

      if (result.overall_metrics.total_executions === 0) {
        toast('No execution data found. Generate mock data to see analysis.', {
          icon: 'ℹ️',
          duration: 5000
        });
      } else {
        toast.success(`Analyzed ${result.overall_metrics.total_executions} executions`);
      }
    } catch (error: any) {
      toast.error(error.message || 'Failed to load execution analysis');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateMockData = async () => {
    try {
      const result = await generateMockData(100);
      toast.success(result.message);
      // Reload analysis
      await loadAnalysis();
    } catch (error: any) {
      toast.error('Failed to generate mock data');
      console.error(error);
    }
  };

  const getSlippageColor = (bps: number) => {
    if (bps < 3) return 'text-green-700';
    if (bps < 7) return 'text-yellow-700';
    return 'text-red-700';
  };

  const getFillRateColor = (rate: number) => {
    if (rate >= 95) return 'text-green-700';
    if (rate >= 90) return 'text-yellow-700';
    return 'text-red-700';
  };

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">
          📊 Execution Quality Analysis
        </h1>
        <p className="text-gray-600 mt-2">
          Track fill quality and slippage - Even 1% slippage = 12% annual performance drag
        </p>
      </div>

      {/* Filters */}
      <div className="bg-white rounded-lg shadow p-4 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Start Date
            </label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              End Date
            </label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Symbol
            </label>
            <input
              type="text"
              placeholder="e.g., SPY"
              value={symbolFilter}
              onChange={(e) => setSymbolFilter(e.target.value.toUpperCase())}
              className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Broker
            </label>
            <input
              type="text"
              placeholder="e.g., Schwab"
              value={brokerFilter}
              onChange={(e) => setBrokerFilter(e.target.value)}
              className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="flex items-end">
            <button
              onClick={loadAnalysis}
              disabled={loading}
              className="w-full px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
            >
              {loading ? 'Loading...' : 'Analyze'}
            </button>
          </div>
        </div>

        <div className="mt-4 flex gap-3">
          <button
            onClick={handleGenerateMockData}
            className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
          >
            Generate Mock Data (100 orders)
          </button>
          <button
            onClick={() => {
              setStartDate('');
              setEndDate('');
              setSymbolFilter('');
              setBrokerFilter('');
            }}
            className="px-4 py-2 bg-gray-300 text-gray-700 rounded hover:bg-gray-400"
          >
            Clear Filters
          </button>
        </div>
      </div>

      {analysis && analysis.overall_metrics.total_executions > 0 ? (
        <div className="space-y-6">
          {/* Overall Metrics */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              Overall Performance
            </h2>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              {/* Avg Slippage */}
              <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg border border-blue-200">
                <div className="text-sm text-gray-600 mb-1">Avg Slippage</div>
                <div className={`text-2xl font-bold ${getSlippageColor(analysis.overall_metrics.avg_slippage_bps)}`}>
                  {analysis.overall_metrics.avg_slippage_bps.toFixed(1)} bps
                </div>
                <div className="text-xs text-gray-600 mt-1">
                  Median: {analysis.overall_metrics.median_slippage_bps.toFixed(1)} bps
                </div>
              </div>

              {/* Fill Rate */}
              <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 rounded-lg border border-green-200">
                <div className="text-sm text-gray-600 mb-1">Fill Rate</div>
                <div className={`text-2xl font-bold ${getFillRateColor(analysis.overall_metrics.fill_rate)}`}>
                  {analysis.overall_metrics.fill_rate.toFixed(1)}%
                </div>
                <div className="text-xs text-gray-600 mt-1">
                  {analysis.overall_metrics.total_executions} executions
                </div>
              </div>

              {/* Total Cost */}
              <div className="p-4 bg-gradient-to-br from-red-50 to-red-100 rounded-lg border border-red-200">
                <div className="text-sm text-gray-600 mb-1">Total Slippage Cost</div>
                <div className="text-2xl font-bold text-red-700">
                  ${analysis.overall_metrics.total_slippage_cost.toFixed(2)}
                </div>
                <div className="text-xs text-gray-600 mt-1">
                  Annual est: ${analysis.overall_metrics.estimated_annual_drag.toFixed(2)}
                </div>
              </div>

              {/* Avg Time to Fill */}
              <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg border border-purple-200">
                <div className="text-sm text-gray-600 mb-1">Avg Fill Time</div>
                <div className="text-2xl font-bold text-purple-700">
                  {analysis.overall_metrics.avg_time_to_fill_ms.toFixed(0)}ms
                </div>
                <div className="text-xs text-gray-600 mt-1">
                  Partial fills: {analysis.overall_metrics.partial_fill_rate.toFixed(1)}%
                </div>
              </div>
            </div>

            {/* Additional Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 text-sm">
              <div className="p-3 border rounded">
                <div className="text-gray-600">Price Improvement</div>
                <div className="font-bold text-lg text-green-700">
                  {analysis.overall_metrics.price_improvement_rate.toFixed(1)}%
                </div>
                <div className="text-xs text-gray-600">
                  Avg: {analysis.overall_metrics.avg_price_improvement_bps.toFixed(1)} bps
                </div>
              </div>

              <div className="p-3 border rounded">
                <div className="text-gray-600">Adverse Selection</div>
                <div className="font-bold text-lg text-red-700">
                  {analysis.overall_metrics.adverse_selection_rate.toFixed(1)}%
                </div>
                <div className="text-xs text-gray-600">
                  Avg: {analysis.overall_metrics.avg_adverse_selection_bps.toFixed(1)} bps
                </div>
              </div>

              <div className="p-3 border rounded">
                <div className="text-gray-600">25th Percentile</div>
                <div className="font-bold text-lg">
                  {analysis.overall_metrics.slippage_25th_percentile.toFixed(1)} bps
                </div>
              </div>

              <div className="p-3 border rounded">
                <div className="text-gray-600">75th Percentile</div>
                <div className="font-bold text-lg">
                  {analysis.overall_metrics.slippage_75th_percentile.toFixed(1)} bps
                </div>
              </div>

              <div className="p-3 border rounded">
                <div className="text-gray-600">95th Percentile</div>
                <div className="font-bold text-lg">
                  {analysis.overall_metrics.slippage_95th_percentile.toFixed(1)} bps
                </div>
              </div>

              <div className="p-3 border rounded">
                <div className="text-gray-600">Best Execution</div>
                <div className="font-bold text-lg text-green-700">
                  {analysis.overall_metrics.best_slippage_bps.toFixed(1)} bps
                </div>
              </div>

              <div className="p-3 border rounded">
                <div className="text-gray-600">Worst Execution</div>
                <div className="font-bold text-lg text-red-700">
                  {analysis.overall_metrics.worst_slippage_bps.toFixed(1)} bps
                </div>
              </div>

              <div className="p-3 border rounded">
                <div className="text-gray-600">Total Volume</div>
                <div className="font-bold text-lg">
                  ${(analysis.overall_metrics.total_volume / 1000).toFixed(0)}K
                </div>
              </div>
            </div>
          </div>

          {/* Broker Comparison */}
          {Object.keys(analysis.by_broker).length > 0 && (
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                Broker Comparison
              </h2>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-100 border-b-2 border-gray-300">
                    <tr>
                      <th className="px-4 py-3 text-left">Broker</th>
                      <th className="px-4 py-3 text-right">Avg Slippage (bps)</th>
                      <th className="px-4 py-3 text-right">Executions</th>
                      <th className="px-4 py-3 text-right">Fill Rate</th>
                      <th className="px-4 py-3 text-left">Rating</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(analysis.by_broker)
                      .sort((a, b) => a[1].avg_slippage_bps - b[1].avg_slippage_bps)
                      .map(([broker, metrics], idx) => (
                        <tr key={broker} className={`border-b ${idx === 0 ? 'bg-green-50' : 'hover:bg-gray-50'}`}>
                          <td className="px-4 py-3 font-semibold">
                            {broker}
                            {idx === 0 && <span className="ml-2 text-green-600">🏆</span>}
                          </td>
                          <td className={`px-4 py-3 text-right font-bold ${getSlippageColor(metrics.avg_slippage_bps)}`}>
                            {metrics.avg_slippage_bps.toFixed(1)}
                          </td>
                          <td className="px-4 py-3 text-right">{metrics.total_executions}</td>
                          <td className={`px-4 py-3 text-right ${getFillRateColor(metrics.fill_rate)}`}>
                            {metrics.fill_rate.toFixed(1)}%
                          </td>
                          <td className="px-4 py-3">
                            {metrics.avg_slippage_bps < 3 ? (
                              <span className="px-2 py-1 bg-green-100 text-green-800 text-xs font-semibold rounded">
                                Excellent
                              </span>
                            ) : metrics.avg_slippage_bps < 7 ? (
                              <span className="px-2 py-1 bg-yellow-100 text-yellow-800 text-xs font-semibold rounded">
                                Good
                              </span>
                            ) : (
                              <span className="px-2 py-1 bg-red-100 text-red-800 text-xs font-semibold rounded">
                                Poor
                              </span>
                            )}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Time of Day Analysis */}
          {Object.keys(analysis.by_time_of_day).length > 0 && (
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                Time of Day Analysis
              </h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(analysis.by_time_of_day)
                  .sort((a, b) => a[1].avg_slippage_bps - b[1].avg_slippage_bps)
                  .map(([period, metrics]) => (
                    <div key={period} className="p-4 border-2 rounded-lg hover:border-blue-400 transition-colors">
                      <div className="text-sm font-medium text-gray-700 mb-2">
                        {period.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </div>
                      <div className={`text-xl font-bold ${getSlippageColor(metrics.avg_slippage_bps)}`}>
                        {metrics.avg_slippage_bps.toFixed(1)} bps
                      </div>
                      <div className="text-xs text-gray-600 mt-1">
                        {metrics.total_executions} trades
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* Order Type Comparison */}
          {Object.keys(analysis.by_order_type).length > 1 && (
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                Order Type Comparison
              </h2>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {Object.entries(analysis.by_order_type).map(([orderType, metrics]) => (
                  <div key={orderType} className="p-4 border-2 rounded-lg">
                    <div className="text-sm font-medium text-gray-700 mb-2">
                      {orderType.toUpperCase()} Orders
                    </div>
                    <div className={`text-xl font-bold ${getSlippageColor(metrics.avg_slippage_bps)}`}>
                      {metrics.avg_slippage_bps.toFixed(1)} bps
                    </div>
                    <div className="text-xs text-gray-600 mt-1">
                      {metrics.total_executions} trades
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Recommendations */}
          {analysis.recommendations.length > 0 && (
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                💡 Actionable Recommendations
              </h2>
              <div className="space-y-3">
                {analysis.recommendations.map((recommendation, idx) => (
                  <div key={idx} className="p-4 bg-blue-50 border-l-4 border-blue-500 rounded">
                    <p className="text-sm">{recommendation}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Symbol Breakdown */}
          {Object.keys(analysis.by_symbol).length > 0 && (
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                Symbol Breakdown (5+ trades)
              </h2>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                {Object.entries(analysis.by_symbol)
                  .sort((a, b) => b[1].total_executions - a[1].total_executions)
                  .map(([symbol, metrics]) => (
                    <div key={symbol} className="p-3 border rounded">
                      <div className="font-bold text-gray-900">{symbol}</div>
                      <div className={`text-sm font-semibold ${getSlippageColor(metrics.avg_slippage_bps)}`}>
                        {metrics.avg_slippage_bps.toFixed(1)} bps
                      </div>
                      <div className="text-xs text-gray-600">
                        {metrics.total_executions} trades
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        !loading && (
          <div className="bg-white rounded-lg shadow p-12 text-center">
            <div className="text-gray-500 mb-4">No execution data available</div>
            <button
              onClick={handleGenerateMockData}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Generate Mock Data to Get Started
            </button>
          </div>
        )
      )}
    </div>
  );
};

export default ExecutionQualityPage;
