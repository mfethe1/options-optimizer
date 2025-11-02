import React, { useState, useEffect } from 'react';
import { getRiskDashboard, RiskDashboard } from '../services/riskDashboardApi';
import toast from 'react-hot-toast';

const RiskDashboardPage: React.FC = () => {
  const [dashboard, setDashboard] = useState<RiskDashboard | null>(null);
  const [loading, setLoading] = useState(false);
  const [lookbackDays, setLookbackDays] = useState(252);
  const [autoRefresh, setAutoRefresh] = useState(false);

  const userId = 'demo_user';

  useEffect(() => {
    loadDashboard();
  }, []);

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        loadDashboard();
      }, 30000); // Refresh every 30 seconds

      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const loadDashboard = async () => {
    setLoading(true);
    try {
      const data = await getRiskDashboard(userId, lookbackDays);
      setDashboard(data);
    } catch (error: any) {
      toast.error(error.message || 'Failed to load risk dashboard');
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  const formatNumber = (value: number, decimals: number = 2) => {
    return value.toFixed(decimals);
  };

  const getGreekColor = (greek: string, value: number) => {
    if (greek === 'delta') {
      if (Math.abs(value) > 0.5) return 'text-red-600';
      if (Math.abs(value) > 0.3) return 'text-orange-600';
      return 'text-green-600';
    }
    if (greek === 'theta') {
      if (value < -1000) return 'text-red-600';
      if (value < -500) return 'text-orange-600';
      return 'text-green-600';
    }
    return 'text-gray-900';
  };

  const getVaRColor = (varPct: number) => {
    if (varPct > 10) return 'text-red-600';
    if (varPct > 5) return 'text-orange-600';
    return 'text-green-600';
  };

  const getStressTestColor = (changePct: number, breachesMargin: boolean) => {
    if (breachesMargin) return 'bg-red-100 border-red-300';
    if (Math.abs(changePct) > 10) return 'bg-orange-100 border-orange-300';
    if (Math.abs(changePct) > 5) return 'bg-yellow-100 border-yellow-300';
    return 'bg-green-100 border-green-300';
  };

  const getSharpeColor = (sharpe: number) => {
    if (sharpe > 2) return 'text-green-600';
    if (sharpe > 1) return 'text-blue-600';
    if (sharpe > 0.5) return 'text-gray-600';
    if (sharpe > 0) return 'text-orange-600';
    return 'text-red-600';
  };

  const getSharpeRating = (sharpe: number) => {
    if (sharpe > 2) return 'Excellent';
    if (sharpe > 1) return 'Very Good';
    if (sharpe > 0.5) return 'Good';
    if (sharpe > 0) return 'Fair';
    return 'Poor';
  };

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">
              Risk Dashboard
            </h1>
            <p className="text-gray-600 mt-2">
              Bloomberg PORT equivalent - institutional-grade risk management
            </p>
          </div>
          <div className="flex items-center gap-3">
            <select
              value={lookbackDays}
              onChange={(e) => {
                setLookbackDays(parseInt(e.target.value));
                setTimeout(() => loadDashboard(), 100);
              }}
              className="border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value={30}>30 Days</option>
              <option value={90}>90 Days</option>
              <option value={252}>1 Year (252 days)</option>
              <option value={504}>2 Years</option>
            </select>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="w-4 h-4"
              />
              <span className="text-sm text-gray-700">Auto-refresh</span>
            </label>
            <button
              onClick={loadDashboard}
              disabled={loading}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
            >
              {loading ? 'Loading...' : 'Refresh'}
            </button>
          </div>
        </div>
      </div>

      {!dashboard ? (
        <div className="bg-white rounded-lg shadow p-12 text-center">
          <div className="text-gray-500">
            {loading ? 'Loading risk dashboard...' : 'No data available'}
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Portfolio Overview */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4">Portfolio Overview</h2>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-blue-50 rounded-lg p-4">
                <div className="text-sm text-gray-600 mb-1">Total Value</div>
                <div className="text-2xl font-bold text-gray-900">
                  {formatCurrency(dashboard.portfolio_value)}
                </div>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="text-sm text-gray-600 mb-1">Cash</div>
                <div className="text-2xl font-bold text-gray-900">
                  {formatCurrency(dashboard.cash)}
                </div>
              </div>
              <div className="bg-green-50 rounded-lg p-4">
                <div className="text-sm text-gray-600 mb-1">Buying Power</div>
                <div className="text-2xl font-bold text-green-600">
                  {formatCurrency(dashboard.buying_power)}
                </div>
              </div>
              <div className="bg-purple-50 rounded-lg p-4">
                <div className="text-sm text-gray-600 mb-1">Margin Available</div>
                <div className="text-2xl font-bold text-purple-600">
                  {formatCurrency(dashboard.margin_available)}
                </div>
              </div>
            </div>
            <div className="mt-3 text-xs text-gray-500 text-right">
              Updated: {new Date(dashboard.timestamp).toLocaleString()} •
              Calculation: {dashboard.calculation_time_ms}ms
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Portfolio Greeks */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold mb-4">Portfolio Greeks</h2>
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded">
                  <div>
                    <div className="font-semibold">Delta</div>
                    <div className="text-xs text-gray-600">Directional exposure</div>
                  </div>
                  <div className={`text-2xl font-bold ${getGreekColor('delta', dashboard.greeks.total_delta)}`}>
                    {formatNumber(dashboard.greeks.total_delta, 4)}
                  </div>
                </div>
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded">
                  <div>
                    <div className="font-semibold">Gamma</div>
                    <div className="text-xs text-gray-600">Delta sensitivity</div>
                  </div>
                  <div className="text-2xl font-bold text-gray-900">
                    {formatNumber(dashboard.greeks.total_gamma, 4)}
                  </div>
                </div>
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded">
                  <div>
                    <div className="font-semibold">Theta</div>
                    <div className="text-xs text-gray-600">Time decay (per day)</div>
                  </div>
                  <div className={`text-2xl font-bold ${getGreekColor('theta', dashboard.greeks.total_theta)}`}>
                    {formatCurrency(dashboard.greeks.total_theta)}
                  </div>
                </div>
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded">
                  <div>
                    <div className="font-semibold">Vega</div>
                    <div className="text-xs text-gray-600">IV sensitivity</div>
                  </div>
                  <div className="text-2xl font-bold text-gray-900">
                    {formatCurrency(dashboard.greeks.total_vega)}
                  </div>
                </div>
                <div className="flex items-center justify-between p-3 bg-gray-50 rounded">
                  <div>
                    <div className="font-semibold">Net Delta Exposure</div>
                    <div className="text-xs text-gray-600">Dollar delta</div>
                  </div>
                  <div className="text-2xl font-bold text-gray-900">
                    {formatCurrency(dashboard.greeks.net_delta_exposure)}
                  </div>
                </div>
              </div>
            </div>

            {/* Value at Risk */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold mb-4">Value at Risk (VaR)</h2>
              <div className="space-y-3">
                <div className="border border-gray-200 rounded-lg p-4">
                  <div className="text-sm text-gray-600 mb-2">1-Day VaR (95% confidence)</div>
                  <div className={`text-3xl font-bold ${getVaRColor(dashboard.var_metrics.var_as_pct_of_portfolio)}`}>
                    {formatCurrency(dashboard.var_metrics.var_1day_95)}
                  </div>
                  <div className="text-sm text-gray-600 mt-1">
                    {dashboard.var_metrics.var_as_pct_of_portfolio.toFixed(2)}% of portfolio
                  </div>
                </div>
                <div className="border border-gray-200 rounded-lg p-4">
                  <div className="text-sm text-gray-600 mb-2">1-Day VaR (99% confidence)</div>
                  <div className="text-2xl font-bold text-red-600">
                    {formatCurrency(dashboard.var_metrics.var_1day_99)}
                  </div>
                </div>
                <div className="border border-gray-200 rounded-lg p-4">
                  <div className="text-sm text-gray-600 mb-2">10-Day VaR (95% confidence)</div>
                  <div className="text-2xl font-bold text-orange-600">
                    {formatCurrency(dashboard.var_metrics.var_10day_95)}
                  </div>
                </div>
                <div className="bg-blue-50 rounded-lg p-3">
                  <div className="text-xs text-blue-800">
                    <strong>Expected Shortfall (CVaR):</strong> {formatCurrency(dashboard.var_metrics.cvar_1day_95)}
                  </div>
                  <div className="text-xs text-blue-700 mt-1">
                    Average loss when losses exceed VaR
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Stress Tests */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4">Stress Test Results</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {dashboard.stress_tests.map((test, idx) => (
                <div
                  key={idx}
                  className={`border-2 rounded-lg p-4 ${getStressTestColor(test.portfolio_change_pct, test.breaches_margin)}`}
                >
                  <div className="font-semibold mb-2">{test.scenario}</div>
                  <div className="flex items-baseline gap-2">
                    <span className={`text-2xl font-bold ${test.portfolio_change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {formatCurrency(test.portfolio_change)}
                    </span>
                    <span className="text-sm text-gray-600">
                      ({formatPercent(test.portfolio_change_pct)})
                    </span>
                  </div>
                  {test.breaches_margin && (
                    <div className="mt-2 text-xs font-semibold text-red-700">
                      ⚠️ BREACHES MARGIN
                    </div>
                  )}
                  <div className="mt-2 text-xs text-gray-600">
                    <div>Δ impact: {formatCurrency(test.delta_impact)}</div>
                    <div>Γ impact: {formatCurrency(test.gamma_impact)}</div>
                    <div>V impact: {formatCurrency(test.vega_impact)}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Concentration Risk */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold mb-4">Concentration Risk</h2>
              <div className="space-y-4">
                <div className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-600">Largest Position</span>
                    <span className={`text-xl font-bold ${dashboard.concentration.largest_position_pct > 20 ? 'text-red-600' : 'text-green-600'}`}>
                      {dashboard.concentration.largest_position_pct.toFixed(1)}%
                    </span>
                  </div>
                  {dashboard.concentration.largest_position_pct > 20 && (
                    <div className="text-xs text-red-600">⚠️ Over 20% concentration is risky</div>
                  )}
                </div>
                <div className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-600">Top 5 Concentration</span>
                    <span className={`text-xl font-bold ${dashboard.concentration.top_5_concentration_pct > 60 ? 'text-orange-600' : 'text-green-600'}`}>
                      {dashboard.concentration.top_5_concentration_pct.toFixed(1)}%
                    </span>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-50 rounded-lg p-3">
                    <div className="text-xs text-gray-600">Position Count</div>
                    <div className="text-lg font-bold text-gray-900">
                      {dashboard.concentration.position_count}
                    </div>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-3">
                    <div className="text-xs text-gray-600">Effective Positions</div>
                    <div className="text-lg font-bold text-gray-900">
                      {dashboard.concentration.effective_positions.toFixed(1)}
                    </div>
                  </div>
                </div>
                <div className="bg-blue-50 rounded-lg p-3">
                  <div className="text-xs font-semibold text-blue-900 mb-2">Sector Exposure</div>
                  <div className="space-y-1">
                    {Object.entries(dashboard.concentration.sector_exposure).map(([sector, pct]) => (
                      <div key={sector} className="flex items-center justify-between text-xs">
                        <span className="text-blue-800">{sector}</span>
                        <span className="font-semibold text-blue-900">{pct.toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Performance Attribution */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold mb-4">Performance Attribution</h2>
              <div className="space-y-3">
                <div className="border-b border-gray-200 pb-3">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold">Total P&L</span>
                    <span className={`text-3xl font-bold ${dashboard.attribution.total_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {formatCurrency(dashboard.attribution.total_pnl)}
                    </span>
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <span className="text-sm">Alpha P&L (stock selection)</span>
                    <span className={`font-bold ${dashboard.attribution.alpha_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {formatCurrency(dashboard.attribution.alpha_pnl)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <span className="text-sm">Beta P&L (market exposure)</span>
                    <span className={`font-bold ${dashboard.attribution.beta_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {formatCurrency(dashboard.attribution.beta_pnl)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <span className="text-sm">Theta P&L (time decay)</span>
                    <span className={`font-bold ${dashboard.attribution.theta_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {formatCurrency(dashboard.attribution.theta_pnl)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <span className="text-sm">Vega P&L (volatility)</span>
                    <span className={`font-bold ${dashboard.attribution.vega_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {formatCurrency(dashboard.attribution.vega_pnl)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <span className="text-sm">Gamma P&L (convexity)</span>
                    <span className={`font-bold ${dashboard.attribution.gamma_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {formatCurrency(dashboard.attribution.gamma_pnl)}
                    </span>
                  </div>
                </div>
                <div className="border-t border-gray-200 pt-3 mt-3">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-green-50 rounded p-2">
                      <div className="text-xs text-gray-600">Realized</div>
                      <div className="font-bold text-green-700">
                        {formatCurrency(dashboard.attribution.realized_pnl)}
                      </div>
                    </div>
                    <div className="bg-blue-50 rounded p-2">
                      <div className="text-xs text-gray-600">Unrealized</div>
                      <div className="font-bold text-blue-700">
                        {formatCurrency(dashboard.attribution.unrealized_pnl)}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Risk-Adjusted Performance Metrics */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4">Risk-Adjusted Performance</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="border border-gray-200 rounded-lg p-4">
                <div className="text-sm text-gray-600 mb-1">Sharpe Ratio</div>
                <div className={`text-3xl font-bold ${getSharpeColor(dashboard.sharpe_ratio)}`}>
                  {formatNumber(dashboard.sharpe_ratio, 2)}
                </div>
                <div className="text-sm mt-1">
                  <span className={`font-semibold ${getSharpeColor(dashboard.sharpe_ratio)}`}>
                    {getSharpeRating(dashboard.sharpe_ratio)}
                  </span>
                </div>
                <div className="text-xs text-gray-600 mt-2">
                  Measures excess return per unit of risk. &gt;1 is good, &gt;2 is excellent.
                </div>
              </div>
              <div className="border border-gray-200 rounded-lg p-4">
                <div className="text-sm text-gray-600 mb-1">Sortino Ratio</div>
                <div className={`text-3xl font-bold ${getSharpeColor(dashboard.sortino_ratio)}`}>
                  {formatNumber(dashboard.sortino_ratio, 2)}
                </div>
                <div className="text-xs text-gray-600 mt-2">
                  Like Sharpe but only penalizes downside volatility. Higher is better.
                </div>
              </div>
              <div className="border border-gray-200 rounded-lg p-4">
                <div className="text-sm text-gray-600 mb-1">Maximum Drawdown</div>
                <div className="text-3xl font-bold text-red-600">
                  {formatPercent(dashboard.max_drawdown_pct)}
                </div>
                <div className="text-sm mt-1 text-gray-900">
                  {formatCurrency(dashboard.max_drawdown)}
                </div>
                <div className="text-xs text-gray-600 mt-2">
                  Worst decline from peak
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RiskDashboardPage;
