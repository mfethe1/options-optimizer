/**
 * Schwab Trading Page
 *
 * Live order placement and options chain viewing for validated strategies.
 * ⚠️ WARNING: This page places REAL orders with REAL money!
 */

import React, { useState, useEffect } from 'react';
import { toast } from 'react-hot-toast';
import {
  getSchwabAccounts,
  getSchwabQuote,
  getSchwabOptionsChain,
  placeSchwabOrder,
  formatCurrency,
  type SchwabAccount,
  type SchwabQuote,
  type SchwabOptionsChain,
  type SchwabOptionContract,
  type SchwabOrderRequest
} from '../services/schwabApi';

export default function SchwabTradingPage() {
  const [accounts, setAccounts] = useState<SchwabAccount[]>([]);
  const [selectedAccount, setSelectedAccount] = useState<string>('');
  const [symbol, setSymbol] = useState('');
  const [quote, setQuote] = useState<SchwabQuote | null>(null);
  const [optionsChain, setOptionsChain] = useState<SchwabOptionsChain | null>(null);
  const [loadingQuote, setLoadingQuote] = useState(false);
  const [loadingOptions, setLoadingOptions] = useState(false);

  // Order Form State
  const [orderType, setOrderType] = useState<'MARKET' | 'LIMIT'>('LIMIT');
  const [orderAction, setOrderAction] = useState<'BUY' | 'SELL' | 'BUY_TO_OPEN' | 'SELL_TO_OPEN'>('BUY');
  const [quantity, setQuantity] = useState(1);
  const [limitPrice, setLimitPrice] = useState('');
  const [duration, setDuration] = useState<'DAY' | 'GTC'>('DAY');
  const [placingOrder, setPlacingOrder] = useState(false);

  // Options Chain View
  const [selectedExpiration, setSelectedExpiration] = useState<string>('');
  const [optionType, setOptionType] = useState<'calls' | 'puts'>('calls');

  useEffect(() => {
    loadAccounts();
  }, []);

  const loadAccounts = async () => {
    try {
      const accountData = await getSchwabAccounts();
      setAccounts(accountData);
      if (accountData.length > 0 && !selectedAccount) {
        setSelectedAccount(accountData[0].account_id);
      }
    } catch (error) {
      toast.error('Failed to load accounts. Please connect your Schwab account first.');
    }
  };

  const handleGetQuote = async () => {
    if (!symbol) {
      toast.error('Please enter a symbol');
      return;
    }

    setLoadingQuote(true);
    try {
      const quoteData = await getSchwabQuote(symbol.toUpperCase());
      setQuote(quoteData);
      setLimitPrice(quoteData.ask.toFixed(2));
    } catch (error) {
      toast.error(`Failed to get quote: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setLoadingQuote(false);
    }
  };

  const handleGetOptionsChain = async () => {
    if (!symbol) {
      toast.error('Please enter a symbol');
      return;
    }

    setLoadingOptions(true);
    try {
      const chainData = await getSchwabOptionsChain(symbol.toUpperCase());
      setOptionsChain(chainData);
      if (chainData.expirations.length > 0 && !selectedExpiration) {
        setSelectedExpiration(chainData.expirations[0]);
      }
    } catch (error) {
      toast.error(`Failed to get options chain: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setLoadingOptions(false);
    }
  };

  const handlePlaceOrder = async () => {
    if (!selectedAccount) {
      toast.error('Please select an account');
      return;
    }
    if (!symbol) {
      toast.error('Please enter a symbol');
      return;
    }
    if (quantity <= 0) {
      toast.error('Quantity must be greater than 0');
      return;
    }
    if (orderType === 'LIMIT' && (!limitPrice || parseFloat(limitPrice) <= 0)) {
      toast.error('Please enter a valid limit price');
      return;
    }

    // Confirmation dialog
    const orderSummary = `${orderAction} ${quantity} ${symbol} @ ${orderType === 'MARKET' ? 'Market' : `$${limitPrice}`}`;
    if (!confirm(`⚠️ WARNING: You are about to place a REAL order:\n\n${orderSummary}\n\nThis will execute with REAL money. Are you sure?`)) {
      return;
    }

    setPlacingOrder(true);
    try {
      const order: SchwabOrderRequest = {
        account_id: selectedAccount,
        symbol: symbol.toUpperCase(),
        quantity,
        order_type: orderType,
        order_action: orderAction,
        duration,
        price: orderType === 'LIMIT' ? parseFloat(limitPrice) : undefined,
      };

      const result = await placeSchwabOrder(order);
      toast.success(`Order placed successfully! Order ID: ${result.order_id}`);

      // Reset form
      setQuantity(1);
      setLimitPrice('');
    } catch (error) {
      toast.error(`Failed to place order: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setPlacingOrder(false);
    }
  };

  const currentOptions = optionsChain && selectedExpiration
    ? optionsChain[optionType][selectedExpiration] || []
    : [];

  return (
    <div className="p-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Live Trading</h1>
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <p className="text-sm text-red-800">
              ⚠️ <strong>WARNING:</strong> This page places REAL orders with REAL money.
              All orders are immediately sent to Charles Schwab for execution. Always verify your orders before submitting.
            </p>
          </div>
        </div>

        {/* Account Selector */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">Trading Account</label>
          <select
            value={selectedAccount}
            onChange={(e) => setSelectedAccount(e.target.value)}
            className="w-full max-w-md px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            {accounts.map(account => (
              <option key={account.account_id} value={account.account_id}>
                {account.account_number} - {account.account_type} -
                {formatCurrency(account.current_balances.buying_power)} buying power
              </option>
            ))}
          </select>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column: Quote & Order Entry */}
          <div className="space-y-6">
            {/* Symbol Input */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Symbol Lookup</h2>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                  placeholder="Enter symbol (e.g., AAPL)"
                  className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  onKeyPress={(e) => e.key === 'Enter' && handleGetQuote()}
                />
                <button
                  onClick={handleGetQuote}
                  disabled={loadingQuote}
                  className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
                >
                  {loadingQuote ? 'Loading...' : 'Get Quote'}
                </button>
                <button
                  onClick={handleGetOptionsChain}
                  disabled={loadingOptions}
                  className="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50"
                >
                  {loadingOptions ? 'Loading...' : 'Options'}
                </button>
              </div>
            </div>

            {/* Quote Display */}
            {quote && (
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">{quote.symbol} Quote</h2>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-sm text-gray-600">Last</div>
                    <div className="text-2xl font-bold text-gray-900">{formatCurrency(quote.last)}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600">Volume</div>
                    <div className="text-xl font-semibold text-gray-900">{quote.volume.toLocaleString()}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600">Bid</div>
                    <div className="text-lg font-semibold text-gray-900">{formatCurrency(quote.bid)} × {quote.bid_size}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600">Ask</div>
                    <div className="text-lg font-semibold text-gray-900">{formatCurrency(quote.ask)} × {quote.ask_size}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600">Day Range</div>
                    <div className="text-sm text-gray-900">{formatCurrency(quote.low)} - {formatCurrency(quote.high)}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600">Open</div>
                    <div className="text-sm text-gray-900">{formatCurrency(quote.open)}</div>
                  </div>
                </div>
              </div>
            )}

            {/* Order Entry Form */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Place Order</h2>

              <div className="space-y-4">
                {/* Order Type */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Order Type</label>
                  <select
                    value={orderType}
                    onChange={(e) => setOrderType(e.target.value as 'MARKET' | 'LIMIT')}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="MARKET">Market</option>
                    <option value="LIMIT">Limit</option>
                  </select>
                </div>

                {/* Order Action */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Action</label>
                  <select
                    value={orderAction}
                    onChange={(e) => setOrderAction(e.target.value as any)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="BUY">Buy Stock</option>
                    <option value="SELL">Sell Stock</option>
                    <option value="BUY_TO_OPEN">Buy to Open (Options)</option>
                    <option value="SELL_TO_OPEN">Sell to Open (Options)</option>
                    <option value="BUY_TO_CLOSE">Buy to Close (Options)</option>
                    <option value="SELL_TO_CLOSE">Sell to Close (Options)</option>
                  </select>
                </div>

                {/* Quantity */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Quantity</label>
                  <input
                    type="number"
                    value={quantity}
                    onChange={(e) => setQuantity(parseInt(e.target.value) || 0)}
                    min="1"
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                {/* Limit Price (if LIMIT order) */}
                {orderType === 'LIMIT' && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Limit Price</label>
                    <input
                      type="number"
                      step="0.01"
                      value={limitPrice}
                      onChange={(e) => setLimitPrice(e.target.value)}
                      placeholder="0.00"
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                )}

                {/* Duration */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Duration</label>
                  <select
                    value={duration}
                    onChange={(e) => setDuration(e.target.value as 'DAY' | 'GTC')}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="DAY">Day Order</option>
                    <option value="GTC">Good Till Canceled</option>
                  </select>
                </div>

                {/* Place Order Button */}
                <button
                  onClick={handlePlaceOrder}
                  disabled={placingOrder || !selectedAccount}
                  className="w-full bg-red-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-red-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {placingOrder ? 'Placing Order...' : '⚠️ Place Live Order'}
                </button>
              </div>
            </div>
          </div>

          {/* Right Column: Options Chain */}
          <div className="space-y-6">
            {optionsChain && (
              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">
                  Options Chain - {optionsChain.symbol}
                </h2>

                <div className="mb-4">
                  <div className="text-sm text-gray-600">Underlying Price</div>
                  <div className="text-2xl font-bold text-gray-900">
                    {formatCurrency(optionsChain.underlying_price)}
                  </div>
                </div>

                {/* Expiration Selector */}
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">Expiration</label>
                  <select
                    value={selectedExpiration}
                    onChange={(e) => setSelectedExpiration(e.target.value)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  >
                    {optionsChain.expirations.map(exp => (
                      <option key={exp} value={exp}>{exp}</option>
                    ))}
                  </select>
                </div>

                {/* Calls/Puts Toggle */}
                <div className="flex gap-2 mb-4">
                  <button
                    onClick={() => setOptionType('calls')}
                    className={`flex-1 py-2 px-4 rounded-lg font-semibold transition-colors ${
                      optionType === 'calls'
                        ? 'bg-green-600 text-white'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                  >
                    Calls
                  </button>
                  <button
                    onClick={() => setOptionType('puts')}
                    className={`flex-1 py-2 px-4 rounded-lg font-semibold transition-colors ${
                      optionType === 'puts'
                        ? 'bg-red-600 text-white'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                  >
                    Puts
                  </button>
                </div>

                {/* Options Table */}
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Strike</th>
                        <th className="px-3 py-2 text-right text-xs font-medium text-gray-500">Bid</th>
                        <th className="px-3 py-2 text-right text-xs font-medium text-gray-500">Ask</th>
                        <th className="px-3 py-2 text-right text-xs font-medium text-gray-500">Vol</th>
                        <th className="px-3 py-2 text-right text-xs font-medium text-gray-500">OI</th>
                        <th className="px-3 py-2 text-right text-xs font-medium text-gray-500">IV</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {currentOptions.slice(0, 20).map((option, idx) => (
                        <tr
                          key={idx}
                          className={`hover:bg-gray-50 cursor-pointer ${option.in_the_money ? 'bg-yellow-50' : ''}`}
                          onClick={() => {
                            setSymbol(option.symbol);
                            setLimitPrice(option.ask.toFixed(2));
                          }}
                        >
                          <td className="px-3 py-2 text-sm font-medium text-gray-900">
                            {formatCurrency(option.strike)}
                          </td>
                          <td className="px-3 py-2 text-sm text-gray-900 text-right">
                            {option.bid.toFixed(2)}
                          </td>
                          <td className="px-3 py-2 text-sm text-gray-900 text-right">
                            {option.ask.toFixed(2)}
                          </td>
                          <td className="px-3 py-2 text-sm text-gray-600 text-right">
                            {option.volume.toLocaleString()}
                          </td>
                          <td className="px-3 py-2 text-sm text-gray-600 text-right">
                            {option.open_interest.toLocaleString()}
                          </td>
                          <td className="px-3 py-2 text-sm text-gray-600 text-right">
                            {(option.implied_volatility * 100).toFixed(1)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
