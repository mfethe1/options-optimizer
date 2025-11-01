/**
 * Schwab Connection Page
 *
 * OAuth authentication and account management for Charles Schwab API.
 * Enables live trading integration for validated strategies.
 */

import React, { useState, useEffect } from 'react';
import { toast } from 'react-hot-toast';
import {
  getSchwabAuthUrl,
  exchangeAuthCode,
  getSchwabAccounts,
  getSchwabPositions,
  formatCurrency,
  formatPercent,
  getPnLColor,
  type SchwabAccount,
  type SchwabPosition
} from '../services/schwabApi';

export default function SchwabConnectionPage() {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [accounts, setAccounts] = useState<SchwabAccount[]>([]);
  const [selectedAccount, setSelectedAccount] = useState<string | null>(null);
  const [positions, setPositions] = useState<SchwabPosition[]>([]);
  const [loadingPositions, setLoadingPositions] = useState(false);

  // Check for OAuth callback
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const code = urlParams.get('code');

    if (code) {
      handleOAuthCallback(code);
      // Clean URL
      window.history.replaceState({}, document.title, window.location.pathname);
    }

    // Check if already connected by trying to fetch accounts
    loadAccounts();
  }, []);

  const handleOAuthCallback = async (code: string) => {
    setIsConnecting(true);
    try {
      const result = await exchangeAuthCode(code);
      if (result.success) {
        toast.success('Successfully connected to Schwab!');
        setIsConnected(true);
        loadAccounts();
      } else {
        toast.error(`Connection failed: ${result.message}`);
      }
    } catch (error) {
      toast.error(`OAuth error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsConnecting(false);
    }
  };

  const handleConnect = async () => {
    try {
      const { auth_url } = await getSchwabAuthUrl();
      window.location.href = auth_url;
    } catch (error) {
      toast.error(`Failed to get auth URL: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  const loadAccounts = async () => {
    try {
      const accountData = await getSchwabAccounts();
      setAccounts(accountData);
      setIsConnected(accountData.length > 0);
      if (accountData.length > 0 && !selectedAccount) {
        setSelectedAccount(accountData[0].account_id);
      }
    } catch (error) {
      // Not connected yet
      setIsConnected(false);
    }
  };

  const loadPositions = async (accountId: string) => {
    setLoadingPositions(true);
    try {
      const positionsData = await getSchwabPositions(accountId);
      setPositions(positionsData);
    } catch (error) {
      toast.error(`Failed to load positions: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setLoadingPositions(false);
    }
  };

  useEffect(() => {
    if (selectedAccount) {
      loadPositions(selectedAccount);
    }
  }, [selectedAccount]);

  if (isConnecting) {
    return (
      <div className="p-8">
        <div className="max-w-md mx-auto text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <h2 className="text-xl font-semibold text-gray-900">Connecting to Schwab...</h2>
          <p className="text-gray-600 mt-2">Please wait while we complete the authentication.</p>
        </div>
      </div>
    );
  }

  if (!isConnected) {
    return (
      <div className="p-8">
        <div className="max-w-2xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-900 mb-6">Connect to Charles Schwab</h1>

          <div className="bg-white rounded-lg shadow-lg p-8">
            <div className="mb-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-3">Live Trading Integration</h2>
              <p className="text-gray-600">
                Connect your Charles Schwab account to enable live trading and real-time market data access.
                This integration allows you to:
              </p>
              <ul className="mt-3 space-y-2 text-gray-600">
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  View real-time account balances and positions
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  Access live options chains with Greeks
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  Execute validated trading strategies automatically
                </li>
                <li className="flex items-start">
                  <span className="text-green-500 mr-2">✓</span>
                  Track P&L and execution quality in real-time
                </li>
              </ul>
            </div>

            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
              <h3 className="text-sm font-semibold text-yellow-800 mb-2">⚠️ Important Setup Instructions</h3>
              <ol className="text-sm text-yellow-700 space-y-1 list-decimal list-inside">
                <li>Create a developer account at developer.schwab.com</li>
                <li>Create a new app and obtain Client ID and Secret</li>
                <li>Set redirect URI to: <code className="bg-yellow-100 px-1 rounded">https://localhost:8000/callback</code></li>
                <li>Add credentials to your .env file</li>
                <li>Restart the backend server</li>
              </ol>
            </div>

            <button
              onClick={handleConnect}
              className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
            >
              Connect to Schwab
            </button>

            <p className="text-xs text-gray-500 mt-4 text-center">
              By connecting, you authorize this application to access your Schwab account data and place trades on your behalf.
              You can revoke access at any time from your Schwab account settings.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const currentAccount = accounts.find(acc => acc.account_id === selectedAccount);

  return (
    <div className="p-8">
      <div className="max-w-7xl mx-auto">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold text-gray-900">Schwab Accounts</h1>
          <button
            onClick={loadAccounts}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
          >
            Refresh Accounts
          </button>
        </div>

        {/* Account Selector */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">Select Account</label>
          <select
            value={selectedAccount || ''}
            onChange={(e) => setSelectedAccount(e.target.value)}
            className="w-full max-w-md px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            {accounts.map(account => (
              <option key={account.account_id} value={account.account_id}>
                {account.account_number} - {account.account_type}
              </option>
            ))}
          </select>
        </div>

        {/* Account Summary */}
        {currentAccount && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div className="bg-white rounded-lg shadow p-6">
              <div className="text-sm text-gray-600 mb-1">Cash Balance</div>
              <div className="text-2xl font-bold text-gray-900">
                {formatCurrency(currentAccount.current_balances.cash_balance)}
              </div>
            </div>
            <div className="bg-white rounded-lg shadow p-6">
              <div className="text-sm text-gray-600 mb-1">Market Value</div>
              <div className="text-2xl font-bold text-gray-900">
                {formatCurrency(currentAccount.current_balances.market_value)}
              </div>
            </div>
            <div className="bg-white rounded-lg shadow p-6">
              <div className="text-sm text-gray-600 mb-1">Buying Power</div>
              <div className="text-2xl font-bold text-gray-900">
                {formatCurrency(currentAccount.current_balances.buying_power)}
              </div>
            </div>
            <div className="bg-white rounded-lg shadow p-6">
              <div className="text-sm text-gray-600 mb-1">Total Value</div>
              <div className="text-2xl font-bold text-gray-900">
                {formatCurrency(currentAccount.current_balances.total_value)}
              </div>
            </div>
          </div>
        )}

        {/* Positions Table */}
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-xl font-semibold text-gray-900">Positions</h2>
          </div>

          {loadingPositions ? (
            <div className="p-8 text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-3"></div>
              <p className="text-gray-600">Loading positions...</p>
            </div>
          ) : positions.length === 0 ? (
            <div className="p-8 text-center text-gray-500">
              No positions found in this account.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Symbol</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Quantity</th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Avg Price</th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Current</th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Market Value</th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">P&L</th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">P&L %</th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Day P&L</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {positions.map((position, idx) => (
                    <tr key={idx} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {position.symbol}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                        {position.instrument_type}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 text-right">
                        {position.quantity.toFixed(0)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 text-right">
                        {formatCurrency(position.average_price)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 text-right">
                        {formatCurrency(position.current_price)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 text-right">
                        {formatCurrency(position.market_value)}
                      </td>
                      <td className={`px-6 py-4 whitespace-nowrap text-sm text-right font-medium ${getPnLColor(position.pnl)}`}>
                        {formatCurrency(position.pnl)}
                      </td>
                      <td className={`px-6 py-4 whitespace-nowrap text-sm text-right font-medium ${getPnLColor(position.pnl_pct)}`}>
                        {formatPercent(position.pnl_pct)}
                      </td>
                      <td className={`px-6 py-4 whitespace-nowrap text-sm text-right font-medium ${getPnLColor(position.day_pnl)}`}>
                        {formatCurrency(position.day_pnl)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Summary Stats */}
        {positions.length > 0 && (
          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white rounded-lg shadow p-6">
              <div className="text-sm text-gray-600 mb-1">Total Positions</div>
              <div className="text-2xl font-bold text-gray-900">{positions.length}</div>
            </div>
            <div className="bg-white rounded-lg shadow p-6">
              <div className="text-sm text-gray-600 mb-1">Total P&L</div>
              <div className={`text-2xl font-bold ${getPnLColor(positions.reduce((sum, p) => sum + p.pnl, 0))}`}>
                {formatCurrency(positions.reduce((sum, p) => sum + p.pnl, 0))}
              </div>
            </div>
            <div className="bg-white rounded-lg shadow p-6">
              <div className="text-sm text-gray-600 mb-1">Day P&L</div>
              <div className={`text-2xl font-bold ${getPnLColor(positions.reduce((sum, p) => sum + p.day_pnl, 0))}`}>
                {formatCurrency(positions.reduce((sum, p) => sum + p.day_pnl, 0))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
