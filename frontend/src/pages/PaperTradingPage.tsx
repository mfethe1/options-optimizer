import React, { useState, useEffect } from 'react';
import {
  getPortfolio,
  getTradeHistory,
  executeTrade,
  getPendingApprovals,
  approveTrade,
  rejectTrade,
  resetPortfolio,
  PortfolioResponse,
  TradeRecommendation,
} from '../services/paperTradingApi';
import toast from 'react-hot-toast';

const PaperTradingPage: React.FC = () => {
  const [portfolio, setPortfolio] = useState<PortfolioResponse | null>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [pendingApprovals, setPendingApprovals] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [showTradeForm, setShowTradeForm] = useState(false);
  const [tradeForm, setTradeForm] = useState<Partial<TradeRecommendation>>({
    symbol: '',
    action: 'buy',
    quantity: 1,
    trade_type: 'stock',
    confidence: 0.8,
  });

  const userId = 'demo_user';

  useEffect(() => {
    loadPortfolio();
    loadHistory();
    loadPendingApprovals();
  }, []);

  const loadPortfolio = async () => {
    try {
      const data = await getPortfolio(userId);
      setPortfolio(data);
    } catch (error: any) {
      console.error('Failed to load portfolio:', error);
    }
  };

  const loadHistory = async () => {
    try {
      const data = await getTradeHistory(userId, 20);
      setHistory(data.trades || []);
    } catch (error: any) {
      console.error('Failed to load history:', error);
    }
  };

  const loadPendingApprovals = async () => {
    try {
      const data = await getPendingApprovals(userId);
      setPendingApprovals(data.pending_approvals || []);
    } catch (error: any) {
      console.error('Failed to load approvals:', error);
    }
  };

  const handleExecuteTrade = async () => {
    if (!tradeForm.symbol || !tradeForm.quantity) {
      toast.error('Please fill in symbol and quantity');
      return;
    }

    setLoading(true);
    try {
      const result = await executeTrade({
        recommendation: tradeForm as TradeRecommendation,
        user_id: userId,
        auto_approve: true, // Auto-approve for demo
        timeout_seconds: 300,
      });

      if (result.status === 'executed') {
        toast.success(`Trade executed: ${tradeForm.action} ${tradeForm.quantity} ${tradeForm.symbol}`);
        setShowTradeForm(false);
        setTradeForm({
          symbol: '',
          action: 'buy',
          quantity: 1,
          trade_type: 'stock',
          confidence: 0.8,
        });
        loadPortfolio();
        loadHistory();
      } else if (result.status === 'rejected') {
        toast.error(`Trade rejected: ${result.reason}`);
      } else {
        toast.info('Trade pending approval');
        loadPendingApprovals();
      }
    } catch (error: any) {
      toast.error(error.message || 'Failed to execute trade');
    } finally {
      setLoading(false);
    }
  };

  const handleApprove = async (tradeId: string) => {
    try {
      await approveTrade(userId, tradeId);
      toast.success('Trade approved');
      loadPendingApprovals();
      loadPortfolio();
      loadHistory();
    } catch (error: any) {
      toast.error(error.message || 'Failed to approve trade');
    }
  };

  const handleReject = async (tradeId: string) => {
    try {
      await rejectTrade(userId, tradeId);
      toast.success('Trade rejected');
      loadPendingApprovals();
    } catch (error: any) {
      toast.error(error.message || 'Failed to reject trade');
    }
  };

  const handleReset = async () => {
    if (!confirm('Are you sure you want to reset your portfolio? This will clear all positions and reset cash to $100,000.')) {
      return;
    }

    try {
      await resetPortfolio(userId);
      toast.success('Portfolio reset');
      loadPortfolio();
      loadHistory();
    } catch (error: any) {
      toast.error(error.message || 'Failed to reset portfolio');
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">
              Paper Trading
            </h1>
            <p className="text-gray-600 mt-2">
              AI-powered autonomous trading with multi-agent consensus
            </p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setShowTradeForm(!showTradeForm)}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              {showTradeForm ? 'Cancel' : 'New Trade'}
            </button>
            <button
              onClick={handleReset}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
            >
              Reset
            </button>
          </div>
        </div>
      </div>

      {showTradeForm && (
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-lg font-semibold mb-4">Execute Trade</h2>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Symbol
              </label>
              <input
                type="text"
                value={tradeForm.symbol}
                onChange={(e) => setTradeForm({ ...tradeForm, symbol: e.target.value.toUpperCase() })}
                placeholder="AAPL"
                className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Action
              </label>
              <select
                value={tradeForm.action}
                onChange={(e) => setTradeForm({ ...tradeForm, action: e.target.value as any })}
                className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="buy">Buy</option>
                <option value="sell">Sell</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Quantity
              </label>
              <input
                type="number"
                value={tradeForm.quantity}
                onChange={(e) => setTradeForm({ ...tradeForm, quantity: parseInt(e.target.value) })}
                min="1"
                className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Price (optional)
              </label>
              <input
                type="number"
                value={tradeForm.price || ''}
                onChange={(e) => setTradeForm({ ...tradeForm, price: parseFloat(e.target.value) || undefined })}
                placeholder="Market price"
                step="0.01"
                className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
          <button
            onClick={handleExecuteTrade}
            disabled={loading}
            className="w-full bg-green-600 text-white py-3 rounded-lg hover:bg-green-700 disabled:opacity-50 font-semibold"
          >
            {loading ? 'Processing...' : 'Execute Trade'}
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Portfolio Summary */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4">Portfolio Overview</h2>
            {portfolio ? (
              <>
                <div className="grid grid-cols-3 gap-4 mb-6">
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="text-sm text-gray-600 mb-1">Cash</div>
                    <div className="text-2xl font-bold text-gray-900">
                      {formatCurrency(portfolio.cash)}
                    </div>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="text-sm text-gray-600 mb-1">Total Value</div>
                    <div className="text-2xl font-bold text-gray-900">
                      {formatCurrency(portfolio.performance.current_value)}
                    </div>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="text-sm text-gray-600 mb-1">Total P&L</div>
                    <div className={`text-2xl font-bold ${portfolio.performance.total_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {formatCurrency(portfolio.performance.total_pnl)}
                    </div>
                    <div className={`text-sm ${portfolio.performance.total_return_pct >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {formatPercent(portfolio.performance.total_return_pct)}
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-4 gap-4 mb-6">
                  <div className="text-center">
                    <div className="text-sm text-gray-600 mb-1">Win Rate</div>
                    <div className="text-xl font-bold text-gray-900">
                      {portfolio.performance.win_rate.toFixed(1)}%
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm text-gray-600 mb-1">Total Trades</div>
                    <div className="text-xl font-bold text-gray-900">
                      {portfolio.performance.total_trades}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm text-gray-600 mb-1">Winners</div>
                    <div className="text-xl font-bold text-green-600">
                      {portfolio.performance.winning_trades}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm text-gray-600 mb-1">Positions</div>
                    <div className="text-xl font-bold text-gray-900">
                      {portfolio.positions_count}
                    </div>
                  </div>
                </div>

                {/* Positions */}
                {portfolio.positions.length > 0 && (
                  <div>
                    <h3 className="font-semibold mb-3">Open Positions</h3>
                    <div className="space-y-2">
                      {portfolio.positions.map((position, idx) => (
                        <div key={idx} className="border border-gray-200 rounded-lg p-4">
                          <div className="flex items-center justify-between mb-2">
                            <span className="font-mono font-bold text-lg">{position.symbol}</span>
                            <span className={`font-bold ${position.pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                              {formatCurrency(position.pnl)}
                            </span>
                          </div>
                          <div className="grid grid-cols-3 gap-4 text-sm">
                            <div>
                              <span className="text-gray-600">Quantity:</span>{' '}
                              <span className="font-medium">{position.quantity}</span>
                            </div>
                            <div>
                              <span className="text-gray-600">Avg Price:</span>{' '}
                              <span className="font-medium">{formatCurrency(position.avg_price)}</span>
                            </div>
                            <div>
                              <span className="text-gray-600">Current:</span>{' '}
                              <span className="font-medium">{formatCurrency(position.current_price)}</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="text-center py-8 text-gray-500">
                Loading portfolio...
              </div>
            )}
          </div>

          {/* Trade History */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4">Trade History</h2>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {history.map((trade, idx) => (
                <div key={idx} className="border border-gray-200 rounded p-3 text-sm">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-mono font-bold">{trade.symbol}</span>
                    <span className={trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'}>
                      {formatCurrency(trade.pnl)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-gray-600">
                    <span className="uppercase">{trade.action} {trade.quantity}</span>
                    <span>{formatCurrency(trade.price)}</span>
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {new Date(trade.timestamp).toLocaleString()}
                  </div>
                </div>
              ))}
              {history.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  No trade history yet
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {/* Pending Approvals */}
          {pendingApprovals.length > 0 && (
            <div className="bg-white rounded-lg shadow p-4">
              <h2 className="font-semibold mb-4">Pending Approvals</h2>
              <div className="space-y-3">
                {pendingApprovals.map((approval, idx) => (
                  <div key={idx} className="border border-orange-200 bg-orange-50 rounded-lg p-3">
                    <div className="font-mono font-bold mb-2">{approval.recommendation.symbol}</div>
                    <div className="text-sm text-gray-700 mb-3">
                      {approval.recommendation.action} {approval.recommendation.quantity}
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={() => handleApprove(approval.trade_id)}
                        className="flex-1 bg-green-600 text-white py-1 rounded text-sm hover:bg-green-700"
                      >
                        Approve
                      </button>
                      <button
                        onClick={() => handleReject(approval.trade_id)}
                        className="flex-1 bg-red-600 text-white py-1 rounded text-sm hover:bg-red-700"
                      >
                        Reject
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Info */}
          <div className="bg-blue-50 rounded-lg p-4">
            <h3 className="font-semibold text-blue-900 mb-2">
              🤖 How it works
            </h3>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• Multi-agent consensus voting</li>
              <li>• Risk manager approval</li>
              <li>• Position size limits (10%)</li>
              <li>• Full audit trail</li>
              <li>• Real-time P&L tracking</li>
            </ul>
          </div>

          <div className="bg-green-50 rounded-lg p-4">
            <h3 className="font-semibold text-green-900 mb-2">
              📊 Stats
            </h3>
            {portfolio && (
              <div className="text-sm text-green-800 space-y-1">
                <div>Realized: {formatCurrency(portfolio.performance.realized_pnl)}</div>
                <div>Unrealized: {formatCurrency(portfolio.performance.unrealized_pnl)}</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PaperTradingPage;
