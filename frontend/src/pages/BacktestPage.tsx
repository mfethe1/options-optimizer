import React, { useState, useEffect } from 'react';
import {
  runBacktest,
  getAvailableStrategies,
  compareStrategies,
  BacktestConfig,
  BacktestResult,
  AvailableStrategies,
  ComparisonResult
} from '../services/backtestApi';
import toast from 'react-hot-toast';

const BacktestPage: React.FC = () => {
  const [availableStrategies, setAvailableStrategies] = useState<AvailableStrategies | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [comparisonResult, setComparisonResult] = useState<ComparisonResult | null>(null);
  const [mode, setMode] = useState<'single' | 'compare'>('single');

  // Form state
  const [symbol, setSymbol] = useState('SPY');
  const [strategy, setStrategy] = useState('iron_condor');
  const [startDate, setStartDate] = useState('2024-01-01');
  const [endDate, setEndDate] = useState('2024-12-31');
  const [entryDteMin, setEntryDteMin] = useState(30);
  const [entryDteMax, setEntryDteMax] = useState(45);
  const [profitTarget, setProfitTarget] = useState(50);
  const [stopLoss, setStopLoss] = useState(100);
  const [exitDte, setExitDte] = useState(7);
  const [capitalPerTrade, setCapitalPerTrade] = useState(10000);
  const [spreadWidth, setSpreadWidth] = useState(5);

  // Comparison state
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>(['iron_condor', 'bull_call_spread']);

  useEffect(() => {
    loadStrategies();
  }, []);

  const loadStrategies = async () => {
    try {
      const strategies = await getAvailableStrategies();
      setAvailableStrategies(strategies);
    } catch (error: any) {
      toast.error('Failed to load strategies');
      console.error(error);
    }
  };

  const handleRunBacktest = async () => {
    setLoading(true);
    setResult(null);

    try {
      const config: BacktestConfig = {
        symbol,
        strategy_type: strategy,
        start_date: startDate,
        end_date: endDate,
        entry_dte_min: entryDteMin,
        entry_dte_max: entryDteMax,
        profit_target_pct: profitTarget,
        stop_loss_pct: stopLoss,
        exit_dte: exitDte,
        capital_per_trade: capitalPerTrade,
        spread_width: spreadWidth
      };

      const backtest = await runBacktest(config);
      setResult(backtest);
      toast.success(`Backtest complete: ${backtest.metrics.total_trades} trades`);
    } catch (error: any) {
      toast.error(error.message || 'Backtest failed');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleCompare = async () => {
    if (selectedStrategies.length < 2) {
      toast.error('Select at least 2 strategies to compare');
      return;
    }

    setLoading(true);
    setComparisonResult(null);

    try {
      const comparison = await compareStrategies(
        symbol,
        selectedStrategies,
        startDate,
        endDate,
        {
          entry_dte_min: entryDteMin,
          entry_dte_max: entryDteMax,
          profit_target_pct: profitTarget,
          stop_loss_pct: stopLoss,
          exit_dte: exitDte,
          capital_per_trade: capitalPerTrade,
          spread_width: spreadWidth
        }
      );

      setComparisonResult(comparison);
      toast.success(`Compared ${selectedStrategies.length} strategies`);
    } catch (error: any) {
      toast.error(error.message || 'Comparison failed');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const toggleStrategy = (strategyType: string) => {
    setSelectedStrategies(prev =>
      prev.includes(strategyType)
        ? prev.filter(s => s !== strategyType)
        : [...prev, strategyType]
    );
  };

  const getMetricColor = (value: number, type: 'pnl' | 'winrate' | 'sharpe') => {
    if (type === 'pnl') {
      return value > 0 ? 'text-green-700' : value < 0 ? 'text-red-700' : 'text-gray-700';
    }
    if (type === 'winrate') {
      return value >= 60 ? 'text-green-700' : value >= 50 ? 'text-yellow-700' : 'text-red-700';
    }
    if (type === 'sharpe') {
      return value >= 1.5 ? 'text-green-700' : value >= 1 ? 'text-yellow-700' : 'text-red-700';
    }
    return 'text-gray-700';
  };

  const getInsights = (metrics: BacktestResult['metrics']) => {
    const insights: string[] = [];

    if (metrics.win_rate >= 60) {
      insights.push('✅ Strong win rate - strategy has good probability edge');
    } else if (metrics.win_rate < 50) {
      insights.push('⚠️ Low win rate - verify strategy logic and entry criteria');
    }

    if (metrics.profit_factor >= 2) {
      insights.push('✅ Excellent profit factor - winners significantly larger than losers');
    } else if (metrics.profit_factor < 1) {
      insights.push('❌ Negative profit factor - strategy loses money on average');
    }

    if (metrics.sharpe_ratio >= 1.5) {
      insights.push('✅ Strong risk-adjusted returns - consistent performance');
    } else if (metrics.sharpe_ratio < 0.5) {
      insights.push('⚠️ Low Sharpe ratio - returns not worth the risk');
    }

    if (metrics.max_drawdown_pct > 30) {
      insights.push('⚠️ High max drawdown - reduce position size or tighten stops');
    }

    if (metrics.kelly_criterion > 0.1) {
      insights.push(`💡 Kelly Criterion: ${(metrics.kelly_criterion * 100).toFixed(1)}% position size recommended`);
    } else if (metrics.kelly_criterion < 0) {
      insights.push('❌ Negative Kelly Criterion - strategy has negative expectancy');
    }

    if (metrics.consecutive_losses >= 5) {
      insights.push(`⚠️ Max ${metrics.consecutive_losses} consecutive losses - ensure adequate capital for drawdowns`);
    }

    return insights;
  };

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">
          📊 Strategy Backtesting
        </h1>
        <p className="text-gray-600 mt-2">
          Test historical performance with institutional-grade metrics - validate strategies before risking capital
        </p>
      </div>

      {/* Mode Toggle */}
      <div className="bg-white rounded-lg shadow p-4 mb-6">
        <div className="flex gap-4">
          <button
            onClick={() => setMode('single')}
            className={`px-6 py-2 rounded-lg font-medium transition-colors ${
              mode === 'single'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Single Strategy
          </button>
          <button
            onClick={() => setMode('compare')}
            className={`px-6 py-2 rounded-lg font-medium transition-colors ${
              mode === 'compare'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Compare Strategies
          </button>
        </div>
      </div>

      {/* Configuration Form */}
      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">Backtest Configuration</h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Symbol */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Symbol
            </label>
            <input
              type="text"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="SPY"
            />
          </div>

          {/* Start Date */}
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

          {/* End Date */}
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

          {mode === 'single' && (
            <>
              {/* Strategy */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Strategy
                </label>
                <select
                  value={strategy}
                  onChange={(e) => setStrategy(e.target.value)}
                  className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {availableStrategies &&
                    Object.entries(availableStrategies.strategies).map(([key, info]) => (
                      <option key={key} value={key}>
                        {info.name}
                      </option>
                    ))}
                </select>
              </div>

              {/* Show strategy description */}
              {availableStrategies && strategy && (
                <div className="col-span-2">
                  <div className="p-3 bg-blue-50 border border-blue-200 rounded">
                    <div className="text-sm">
                      <span className="font-semibold">
                        {availableStrategies.strategies[strategy]?.name}:
                      </span>{' '}
                      {availableStrategies.strategies[strategy]?.description}
                      <br />
                      <span className="text-gray-600">
                        Best for: {availableStrategies.strategies[strategy]?.best_for}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}

          {/* Entry DTE Min */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Entry DTE Min
            </label>
            <input
              type="number"
              value={entryDteMin}
              onChange={(e) => setEntryDteMin(parseInt(e.target.value))}
              className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Entry DTE Max */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Entry DTE Max
            </label>
            <input
              type="number"
              value={entryDteMax}
              onChange={(e) => setEntryDteMax(parseInt(e.target.value))}
              className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Profit Target % */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Profit Target %
            </label>
            <input
              type="number"
              value={profitTarget}
              onChange={(e) => setProfitTarget(parseInt(e.target.value))}
              className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Stop Loss % */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Stop Loss %
            </label>
            <input
              type="number"
              value={stopLoss}
              onChange={(e) => setStopLoss(parseInt(e.target.value))}
              className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Exit DTE */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Exit DTE
            </label>
            <input
              type="number"
              value={exitDte}
              onChange={(e) => setExitDte(parseInt(e.target.value))}
              className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Capital Per Trade */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Capital Per Trade ($)
            </label>
            <input
              type="number"
              value={capitalPerTrade}
              onChange={(e) => setCapitalPerTrade(parseInt(e.target.value))}
              className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Spread Width */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Spread Width ($)
            </label>
            <input
              type="number"
              value={spreadWidth}
              onChange={(e) => setSpreadWidth(parseInt(e.target.value))}
              className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        {/* Strategy Selection for Comparison */}
        {mode === 'compare' && availableStrategies && (
          <div className="mt-6">
            <label className="block text-sm font-medium text-gray-700 mb-3">
              Select Strategies to Compare (select 2+)
            </label>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {Object.entries(availableStrategies.strategies).map(([key, info]) => (
                <label
                  key={key}
                  className={`p-3 border-2 rounded cursor-pointer transition-colors ${
                    selectedStrategies.includes(key)
                      ? 'border-blue-600 bg-blue-50'
                      : 'border-gray-300 hover:border-blue-400'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={selectedStrategies.includes(key)}
                    onChange={() => toggleStrategy(key)}
                    className="mr-2"
                  />
                  <span className="font-medium">{info.name}</span>
                </label>
              ))}
            </div>
          </div>
        )}

        {/* Run Button */}
        <div className="mt-6">
          <button
            onClick={mode === 'single' ? handleRunBacktest : handleCompare}
            disabled={loading}
            className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 font-semibold text-lg"
          >
            {loading
              ? 'Running...'
              : mode === 'single'
              ? 'Run Backtest'
              : `Compare ${selectedStrategies.length} Strategies`}
          </button>
        </div>
      </div>

      {/* Single Strategy Results */}
      {mode === 'single' && result && (
        <div className="space-y-6">
          {/* Performance Metrics */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              Performance Metrics
            </h2>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              {/* Total P&L */}
              <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg border border-blue-200">
                <div className="text-sm text-gray-600 mb-1">Total P&L</div>
                <div className={`text-2xl font-bold ${getMetricColor(result.metrics.total_pnl, 'pnl')}`}>
                  ${result.metrics.total_pnl.toFixed(2)}
                </div>
                <div className="text-xs text-gray-600 mt-1">
                  {result.metrics.total_pnl_pct.toFixed(2)}%
                </div>
              </div>

              {/* Win Rate */}
              <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 rounded-lg border border-green-200">
                <div className="text-sm text-gray-600 mb-1">Win Rate</div>
                <div className={`text-2xl font-bold ${getMetricColor(result.metrics.win_rate, 'winrate')}`}>
                  {result.metrics.win_rate.toFixed(1)}%
                </div>
                <div className="text-xs text-gray-600 mt-1">
                  {result.metrics.winning_trades}W / {result.metrics.losing_trades}L
                </div>
              </div>

              {/* Sharpe Ratio */}
              <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg border border-purple-200">
                <div className="text-sm text-gray-600 mb-1">Sharpe Ratio</div>
                <div className={`text-2xl font-bold ${getMetricColor(result.metrics.sharpe_ratio, 'sharpe')}`}>
                  {result.metrics.sharpe_ratio.toFixed(2)}
                </div>
                <div className="text-xs text-gray-600 mt-1">Risk-adjusted returns</div>
              </div>

              {/* Max Drawdown */}
              <div className="p-4 bg-gradient-to-br from-red-50 to-red-100 rounded-lg border border-red-200">
                <div className="text-sm text-gray-600 mb-1">Max Drawdown</div>
                <div className="text-2xl font-bold text-red-700">
                  {result.metrics.max_drawdown_pct.toFixed(1)}%
                </div>
                <div className="text-xs text-gray-600 mt-1">
                  ${result.metrics.max_drawdown.toFixed(2)}
                </div>
              </div>
            </div>

            {/* Additional Metrics Grid */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 text-sm">
              <div className="p-3 border rounded">
                <div className="text-gray-600">Profit Factor</div>
                <div className="font-bold text-lg">{result.metrics.profit_factor.toFixed(2)}</div>
              </div>
              <div className="p-3 border rounded">
                <div className="text-gray-600">Avg Win</div>
                <div className="font-bold text-lg text-green-700">${result.metrics.avg_win.toFixed(2)}</div>
              </div>
              <div className="p-3 border rounded">
                <div className="text-gray-600">Avg Loss</div>
                <div className="font-bold text-lg text-red-700">${result.metrics.avg_loss.toFixed(2)}</div>
              </div>
              <div className="p-3 border rounded">
                <div className="text-gray-600">Expectancy</div>
                <div className="font-bold text-lg">${result.metrics.expectancy.toFixed(2)}</div>
              </div>
              <div className="p-3 border rounded">
                <div className="text-gray-600">Kelly %</div>
                <div className="font-bold text-lg">{(result.metrics.kelly_criterion * 100).toFixed(1)}%</div>
              </div>
              <div className="p-3 border rounded">
                <div className="text-gray-600">Sortino</div>
                <div className="font-bold text-lg">{result.metrics.sortino_ratio.toFixed(2)}</div>
              </div>
              <div className="p-3 border rounded">
                <div className="text-gray-600">Avg Days Held</div>
                <div className="font-bold text-lg">{result.metrics.avg_days_held.toFixed(0)}</div>
              </div>
              <div className="p-3 border rounded">
                <div className="text-gray-600">Total Trades</div>
                <div className="font-bold text-lg">{result.metrics.total_trades}</div>
              </div>
            </div>
          </div>

          {/* Actionable Insights */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              💡 Actionable Insights
            </h2>
            <div className="space-y-2">
              {getInsights(result.metrics).map((insight, idx) => (
                <div key={idx} className="p-3 bg-gray-50 rounded border-l-4 border-blue-500">
                  <p className="text-sm">{insight}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Trade History */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              Trade History ({result.trades.length} trades)
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-gray-100 border-b-2 border-gray-300">
                  <tr>
                    <th className="px-4 py-2 text-left">Entry Date</th>
                    <th className="px-4 py-2 text-left">Exit Date</th>
                    <th className="px-4 py-2 text-right">Entry Price</th>
                    <th className="px-4 py-2 text-right">Entry Cost</th>
                    <th className="px-4 py-2 text-right">P&L</th>
                    <th className="px-4 py-2 text-right">P&L %</th>
                    <th className="px-4 py-2 text-center">Days</th>
                    <th className="px-4 py-2 text-left">Exit Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {result.trades.slice(0, 20).map((trade, idx) => (
                    <tr key={idx} className="border-b hover:bg-gray-50">
                      <td className="px-4 py-2">{trade.entry_date}</td>
                      <td className="px-4 py-2">{trade.exit_date || '-'}</td>
                      <td className="px-4 py-2 text-right">${trade.entry_price.toFixed(2)}</td>
                      <td className="px-4 py-2 text-right">${trade.entry_cost.toFixed(2)}</td>
                      <td className={`px-4 py-2 text-right font-bold ${
                        trade.pnl && trade.pnl > 0 ? 'text-green-700' : trade.pnl && trade.pnl < 0 ? 'text-red-700' : ''
                      }`}>
                        {trade.pnl ? `$${trade.pnl.toFixed(2)}` : '-'}
                      </td>
                      <td className={`px-4 py-2 text-right font-bold ${
                        trade.pnl_pct && trade.pnl_pct > 0 ? 'text-green-700' : trade.pnl_pct && trade.pnl_pct < 0 ? 'text-red-700' : ''
                      }`}>
                        {trade.pnl_pct ? `${trade.pnl_pct.toFixed(1)}%` : '-'}
                      </td>
                      <td className="px-4 py-2 text-center">{trade.days_held || '-'}</td>
                      <td className="px-4 py-2 text-xs">{trade.exit_reason || '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {result.trades.length > 20 && (
              <div className="mt-4 text-center text-sm text-gray-600">
                Showing first 20 of {result.trades.length} trades
              </div>
            )}
          </div>
        </div>
      )}

      {/* Comparison Results */}
      {mode === 'compare' && comparisonResult && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Strategy Comparison
          </h2>
          <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded">
            <span className="font-semibold">Best Strategy:</span>{' '}
            {availableStrategies?.strategies[comparisonResult.best_strategy || '']?.name || comparisonResult.best_strategy}
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-100 border-b-2 border-gray-300">
                <tr>
                  <th className="px-4 py-3 text-left">Strategy</th>
                  <th className="px-4 py-3 text-right">Total P&L</th>
                  <th className="px-4 py-3 text-right">Win Rate</th>
                  <th className="px-4 py-3 text-right">Profit Factor</th>
                  <th className="px-4 py-3 text-right">Sharpe</th>
                  <th className="px-4 py-3 text-right">Max DD</th>
                  <th className="px-4 py-3 text-right">Trades</th>
                </tr>
              </thead>
              <tbody>
                {comparisonResult.strategies.map((s, idx) => (
                  <tr key={idx} className={`border-b ${idx === 0 ? 'bg-green-50' : 'hover:bg-gray-50'}`}>
                    <td className="px-4 py-3 font-semibold">
                      {availableStrategies?.strategies[s.strategy_type]?.name || s.strategy_type}
                      {idx === 0 && <span className="ml-2 text-green-600">🏆</span>}
                    </td>
                    <td className={`px-4 py-3 text-right font-bold ${getMetricColor(s.total_pnl, 'pnl')}`}>
                      ${s.total_pnl.toFixed(2)}
                    </td>
                    <td className="px-4 py-3 text-right">{s.win_rate.toFixed(1)}%</td>
                    <td className="px-4 py-3 text-right">{s.profit_factor.toFixed(2)}</td>
                    <td className="px-4 py-3 text-right">{s.sharpe_ratio.toFixed(2)}</td>
                    <td className="px-4 py-3 text-right text-red-700">{s.max_drawdown.toFixed(2)}</td>
                    <td className="px-4 py-3 text-right">{s.total_trades}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default BacktestPage;
