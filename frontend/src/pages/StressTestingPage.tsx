/**
 * Stress Testing & Scenario Analysis Page
 *
 * Comprehensive risk management dashboard for portfolio stress testing.
 * Features historical scenarios, Monte Carlo simulation, and risk attribution.
 */

import React, { useState, useEffect } from 'react';
import {
  Portfolio,
  PortfolioPosition,
  PortfolioStressResult,
  MonteCarloResult,
  ScenarioInfo,
  runScenario,
  runAllScenarios,
  runMonteCarlo,
  getScenarios,
  createSamplePortfolio,
  formatCurrency,
  formatPercentage,
  getScenarioColor,
  getRiskLevel,
} from '../services/stressTestingApi';

const StressTestingPage: React.FC = () => {
  const [portfolio, setPortfolio] = useState<Portfolio>(createSamplePortfolio());
  const [scenarios, setScenarios] = useState<ScenarioInfo[]>([]);
  const [stressResults, setStressResults] = useState<PortfolioStressResult[]>([]);
  const [monteCarloResult, setMonteCarloResult] = useState<MonteCarloResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [mcLoading, setMcLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'scenarios' | 'montecarlo'>('scenarios');

  // Load available scenarios on mount
  useEffect(() => {
    loadScenarios();
  }, []);

  const loadScenarios = async () => {
    try {
      const data = await getScenarios();
      setScenarios(data);
    } catch (error) {
      console.error('Failed to load scenarios:', error);
    }
  };

  const handleRunAllScenarios = async () => {
    setLoading(true);
    try {
      const result = await runAllScenarios(portfolio);
      setStressResults(result.scenarios);
    } catch (error) {
      console.error('Failed to run scenarios:', error);
      alert('Failed to run stress tests. Check console for details.');
    } finally {
      setLoading(false);
    }
  };

  const handleRunMonteCarlo = async () => {
    setMcLoading(true);
    try {
      const result = await runMonteCarlo({
        portfolio,
        num_simulations: 10000,
        time_horizon_days: 30,
      });
      setMonteCarloResult(result);
    } catch (error) {
      console.error('Failed to run Monte Carlo:', error);
      alert('Failed to run Monte Carlo simulation. Check console for details.');
    } finally {
      setMcLoading(false);
    }
  };

  const handleAddPosition = () => {
    setPortfolio({
      ...portfolio,
      positions: [
        ...portfolio.positions,
        {
          symbol: '',
          type: 'stock',
          quantity: 0,
          current_price: 0,
        },
      ],
    });
  };

  const handleRemovePosition = (index: number) => {
    setPortfolio({
      ...portfolio,
      positions: portfolio.positions.filter((_, i) => i !== index),
    });
  };

  const handlePositionChange = (index: number, field: keyof PortfolioPosition, value: any) => {
    const newPositions = [...portfolio.positions];
    newPositions[index] = {
      ...newPositions[index],
      [field]: value,
    };
    setPortfolio({ ...portfolio, positions: newPositions });
  };

  const calculatePortfolioValue = () => {
    return portfolio.positions.reduce(
      (sum, pos) => sum + pos.quantity * pos.current_price,
      0
    );
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          🛡️ Stress Testing & Scenario Analysis
        </h1>
        <p className="text-gray-600">
          Test your portfolio against historical crises and Monte Carlo simulations.
          Understand tail risk and prevent catastrophic losses.
        </p>
      </div>

      {/* Portfolio Input Section */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold text-gray-900">Portfolio Positions</h2>
          <div className="space-x-2">
            <button
              onClick={() => setPortfolio(createSamplePortfolio())}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
            >
              Load Sample
            </button>
            <button
              onClick={handleAddPosition}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              + Add Position
            </button>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Symbol</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Quantity</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Price</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Value</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {portfolio.positions.map((position, index) => (
                <tr key={index}>
                  <td className="px-4 py-3">
                    <input
                      type="text"
                      value={position.symbol}
                      onChange={(e) => handlePositionChange(index, 'symbol', e.target.value)}
                      className="w-24 px-2 py-1 border rounded"
                      placeholder="AAPL"
                    />
                  </td>
                  <td className="px-4 py-3">
                    <select
                      value={position.type}
                      onChange={(e) => handlePositionChange(index, 'type', e.target.value)}
                      className="px-2 py-1 border rounded"
                    >
                      <option value="stock">Stock</option>
                      <option value="call">Call</option>
                      <option value="put">Put</option>
                    </select>
                  </td>
                  <td className="px-4 py-3">
                    <input
                      type="number"
                      value={position.quantity}
                      onChange={(e) => handlePositionChange(index, 'quantity', parseFloat(e.target.value))}
                      className="w-24 px-2 py-1 border rounded"
                    />
                  </td>
                  <td className="px-4 py-3">
                    <input
                      type="number"
                      value={position.current_price}
                      onChange={(e) => handlePositionChange(index, 'current_price', parseFloat(e.target.value))}
                      className="w-24 px-2 py-1 border rounded"
                      step="0.01"
                    />
                  </td>
                  <td className="px-4 py-3">
                    {formatCurrency(position.quantity * position.current_price)}
                  </td>
                  <td className="px-4 py-3">
                    <button
                      onClick={() => handleRemovePosition(index)}
                      className="text-red-600 hover:text-red-800"
                    >
                      Remove
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <div className="text-lg font-bold">
            Total Portfolio Value: {formatCurrency(calculatePortfolioValue())}
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-white rounded-lg shadow-sm">
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6" aria-label="Tabs">
            <button
              onClick={() => setActiveTab('scenarios')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'scenarios'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Historical Scenarios
            </button>
            <button
              onClick={() => setActiveTab('montecarlo')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'montecarlo'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Monte Carlo Simulation
            </button>
          </nav>
        </div>

        {/* Historical Scenarios Tab */}
        {activeTab === 'scenarios' && (
          <div className="p-6 space-y-6">
            <div className="flex justify-between items-center">
              <div>
                <h2 className="text-xl font-bold text-gray-900">Historical Crisis Scenarios</h2>
                <p className="text-sm text-gray-600">
                  Test your portfolio against 2008 Financial Crisis, COVID Crash, Flash Crash, and Volmageddon
                </p>
              </div>
              <button
                onClick={handleRunAllScenarios}
                disabled={loading || portfolio.positions.length === 0}
                className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:bg-gray-400"
              >
                {loading ? 'Running...' : 'Run All Scenarios'}
              </button>
            </div>

            {/* Available Scenarios */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {scenarios.map((scenario) => (
                <div key={scenario.scenario_type} className="border rounded-lg p-4">
                  <h3 className="font-bold text-lg">{scenario.name}</h3>
                  <p className="text-sm text-gray-600 mt-1">{scenario.description}</p>
                  <div className="mt-2 text-sm">
                    <div><strong>Date:</strong> {scenario.date_range}</div>
                    <div><strong>Equity Drop:</strong> {formatPercentage(scenario.market_shock.equity_return)}</div>
                    <div><strong>Vol Spike:</strong> +{(scenario.market_shock.volatility_change * 100).toFixed(0)} pts</div>
                    <div><strong>Probability:</strong> {formatPercentage(scenario.probability)} per year</div>
                  </div>
                </div>
              ))}
            </div>

            {/* Stress Test Results */}
            {stressResults.length > 0 && (
              <div className="space-y-4">
                <h3 className="text-lg font-bold">Stress Test Results</h3>

                {/* Summary Cards */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  {stressResults.map((result) => (
                    <div
                      key={result.scenario_type}
                      className="border rounded-lg p-4 bg-white hover:shadow-lg transition"
                    >
                      <h4 className="font-bold text-sm text-gray-600">{result.scenario_name}</h4>
                      <div className={`text-2xl font-bold mt-2 ${getScenarioColor(result.total_pnl_pct)}`}>
                        {formatPercentage(result.total_pnl_pct)}
                      </div>
                      <div className="text-sm text-gray-600 mt-1">
                        {formatCurrency(result.total_pnl)}
                      </div>
                      <div className="mt-2 text-xs space-y-1">
                        <div>VaR 95%: {formatCurrency(result.var_95)}</div>
                        <div>CVaR 95%: {formatCurrency(result.cvar_95)}</div>
                        <div>Max DD: {formatPercentage(result.max_drawdown)}</div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Detailed Position Breakdown */}
                <div className="border rounded-lg p-4">
                  <h4 className="font-bold mb-3">Position-Level Breakdown</h4>
                  {stressResults.map((result) => (
                    <div key={result.scenario_type} className="mb-6">
                      <h5 className="font-bold text-gray-700 mb-2">{result.scenario_name}</h5>
                      <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-gray-200">
                          <thead className="bg-gray-50">
                            <tr>
                              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">Symbol</th>
                              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">Type</th>
                              <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">Current</th>
                              <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">Stressed</th>
                              <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">P&L</th>
                              <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">Contribution</th>
                            </tr>
                          </thead>
                          <tbody className="bg-white divide-y divide-gray-200">
                            {result.position_results.map((pos, idx) => (
                              <tr key={idx}>
                                <td className="px-4 py-2 text-sm font-medium">{pos.symbol}</td>
                                <td className="px-4 py-2 text-sm">{pos.position_type}</td>
                                <td className="px-4 py-2 text-sm text-right">{formatCurrency(pos.current_value)}</td>
                                <td className="px-4 py-2 text-sm text-right">{formatCurrency(pos.stressed_value)}</td>
                                <td className={`px-4 py-2 text-sm text-right ${getScenarioColor(pos.pnl_pct)}`}>
                                  {formatCurrency(pos.pnl)} ({formatPercentage(pos.pnl_pct)})
                                </td>
                                <td className="px-4 py-2 text-sm text-right">
                                  {pos.contribution_to_total_pnl.toFixed(1)}%
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Monte Carlo Tab */}
        {activeTab === 'montecarlo' && (
          <div className="p-6 space-y-6">
            <div className="flex justify-between items-center">
              <div>
                <h2 className="text-xl font-bold text-gray-900">Monte Carlo Simulation</h2>
                <p className="text-sm text-gray-600">
                  Run 10,000 simulations to understand your portfolio's risk distribution
                </p>
              </div>
              <button
                onClick={handleRunMonteCarlo}
                disabled={mcLoading || portfolio.positions.length === 0}
                className="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-400"
              >
                {mcLoading ? 'Simulating...' : 'Run Monte Carlo (10,000 sims)'}
              </button>
            </div>

            {monteCarloResult && (
              <div className="space-y-6">
                {/* Summary Stats */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="border rounded-lg p-4">
                    <div className="text-sm text-gray-600">Mean P&L</div>
                    <div className="text-2xl font-bold text-gray-900">
                      {formatCurrency(monteCarloResult.mean_pnl)}
                    </div>
                  </div>
                  <div className="border rounded-lg p-4">
                    <div className="text-sm text-gray-600">Median P&L</div>
                    <div className="text-2xl font-bold text-gray-900">
                      {formatCurrency(monteCarloResult.median_pnl)}
                    </div>
                  </div>
                  <div className="border rounded-lg p-4">
                    <div className="text-sm text-gray-600">Std Dev</div>
                    <div className="text-2xl font-bold text-gray-900">
                      {formatCurrency(monteCarloResult.std_pnl)}
                    </div>
                  </div>
                  <div className="border rounded-lg p-4">
                    <div className="text-sm text-gray-600">VaR 95%</div>
                    <div className="text-2xl font-bold text-red-600">
                      {formatCurrency(monteCarloResult.var_95)}
                    </div>
                  </div>
                </div>

                {/* Percentiles */}
                <div className="border rounded-lg p-6">
                  <h3 className="font-bold text-lg mb-4">P&L Distribution</h3>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Worst 5% (5th percentile)</span>
                      <span className="font-mono text-red-600">{formatCurrency(monteCarloResult.pnl_5th)}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">25th percentile</span>
                      <span className="font-mono text-orange-600">{formatCurrency(monteCarloResult.pnl_25th)}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Median (50th percentile)</span>
                      <span className="font-mono text-gray-900">{formatCurrency(monteCarloResult.median_pnl)}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">75th percentile</span>
                      <span className="font-mono text-green-600">{formatCurrency(monteCarloResult.pnl_75th)}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Best 5% (95th percentile)</span>
                      <span className="font-mono text-green-700">{formatCurrency(monteCarloResult.pnl_95th)}</span>
                    </div>
                  </div>
                </div>

                {/* Risk Metrics */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="border rounded-lg p-6">
                    <h3 className="font-bold text-lg mb-4">Risk Metrics</h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Value at Risk (VaR 95%)</span>
                        <span className="font-mono">{formatCurrency(monteCarloResult.var_95)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Conditional VaR (CVaR 95%)</span>
                        <span className="font-mono">{formatCurrency(monteCarloResult.cvar_95)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Mean Max Drawdown</span>
                        <span className="font-mono">{formatPercentage(monteCarloResult.max_drawdown_mean)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Worst 5% Max Drawdown</span>
                        <span className="font-mono">{formatPercentage(monteCarloResult.max_drawdown_95th)}</span>
                      </div>
                    </div>
                  </div>

                  <div className="border rounded-lg p-6">
                    <h3 className="font-bold text-lg mb-4">Outcome Probabilities</h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Probability of &gt;20% Loss</span>
                        <span className="font-mono text-red-600">{formatPercentage(monteCarloResult.prob_loss_20pct)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Probability of &gt;10% Loss</span>
                        <span className="font-mono text-orange-600">{formatPercentage(monteCarloResult.prob_loss_10pct)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Probability of &gt;10% Gain</span>
                        <span className="font-mono text-green-600">{formatPercentage(monteCarloResult.prob_gain_10pct)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Probability of &gt;20% Gain</span>
                        <span className="font-mono text-green-700">{formatPercentage(monteCarloResult.prob_gain_20pct)}</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Simulation Details */}
                <div className="border rounded-lg p-6 bg-gray-50">
                  <h3 className="font-bold text-lg mb-2">Simulation Details</h3>
                  <div className="text-sm text-gray-600 space-y-1">
                    <div>Simulations: {monteCarloResult.num_simulations.toLocaleString()}</div>
                    <div>Time Horizon: {monteCarloResult.time_horizon_days} days</div>
                    <div>Risk Level: {getRiskLevel(monteCarloResult.var_95, calculatePortfolioValue())}</div>
                    <div>Timestamp: {new Date(monteCarloResult.timestamp).toLocaleString()}</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default StressTestingPage;
