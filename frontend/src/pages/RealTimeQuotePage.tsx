/**
 * Real-Time Market Data Page
 *
 * Institutional-grade market data display with <200ms latency.
 * Features:
 * - Real-time streaming quotes
 * - Level 2 order book visualization
 * - Multi-provider aggregation status
 * - Latency monitoring
 */

import React, { useState, useEffect, useCallback } from 'react';
import toast from 'react-hot-toast';
import {
  getRealTimeQuote,
  getOrderBook,
  getLatencyStats,
  getProviderStatus,
  getMarketDataHealth,
  MarketDataStream,
  Quote,
  OrderBook,
  LatencyStats,
  ProviderStatus,
  HealthStatus,
  StreamQuote
} from '../services/marketDataApi';

export default function RealTimeQuotePage() {
  const [symbol, setSymbol] = useState('AAPL');
  const [inputSymbol, setInputSymbol] = useState('AAPL');
  const [quote, setQuote] = useState<Quote | null>(null);
  const [orderBook, setOrderBook] = useState<OrderBook | null>(null);
  const [latencyStats, setLatencyStats] = useState<LatencyStats | null>(null);
  const [providerStatus, setProviderStatus] = useState<ProviderStatus | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [stream, setStream] = useState<MarketDataStream | null>(null);
  const [loading, setLoading] = useState(false);

  // Load initial data
  useEffect(() => {
    loadAllData();
    loadMonitoring();

    // Refresh monitoring data every 10 seconds
    const interval = setInterval(() => {
      loadMonitoring();
    }, 10000);

    return () => clearInterval(interval);
  }, [symbol]);

  const loadAllData = async () => {
    setLoading(true);
    try {
      const [quoteData, orderBookData] = await Promise.all([
        getRealTimeQuote(symbol),
        getOrderBook(symbol)
      ]);

      setQuote(quoteData);
      setOrderBook(orderBookData);
    } catch (error: any) {
      toast.error(error.message || 'Failed to load market data');
    } finally {
      setLoading(false);
    }
  };

  const loadMonitoring = async () => {
    try {
      const [latency, providers, healthData] = await Promise.all([
        getLatencyStats(),
        getProviderStatus(),
        getMarketDataHealth()
      ]);

      setLatencyStats(latency);
      setProviderStatus(providers);
      setHealth(healthData);
    } catch (error: any) {
      console.error('Failed to load monitoring data:', error);
    }
  };

  const handleStreamToggle = () => {
    if (isStreaming) {
      // Stop streaming
      if (stream) {
        stream.disconnect();
        setStream(null);
      }
      setIsStreaming(false);
      toast.success('Streaming stopped');
    } else {
      // Start streaming
      const newStream = new MarketDataStream(
        symbol,
        (streamQuote: StreamQuote) => {
          // Update quote with streamed data
          setQuote(prev => {
            if (!prev) return null;
            return {
              ...prev,
              best_bid: streamQuote.best_bid,
              best_ask: streamQuote.best_ask,
              mid_price: streamQuote.mid_price,
              spread_bps: streamQuote.spread_bps,
              last: streamQuote.last,
              timestamp: streamQuote.timestamp,
              avg_latency_ms: streamQuote.latency_ms
            };
          });
        },
        (error: Error) => {
          toast.error(`Stream error: ${error.message}`);
          setIsStreaming(false);
        }
      );

      newStream.connect();
      setStream(newStream);
      setIsStreaming(true);
      toast.success('Streaming started');
    }
  };

  const handleSymbolChange = () => {
    const newSymbol = inputSymbol.toUpperCase().trim();
    if (newSymbol && newSymbol !== symbol) {
      // Stop existing stream
      if (stream) {
        stream.disconnect();
        setStream(null);
        setIsStreaming(false);
      }

      setSymbol(newSymbol);
      toast.success(`Switched to ${newSymbol}`);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (stream) {
        stream.disconnect();
      }
    };
  }, [stream]);

  const getHealthColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'bg-green-100 text-green-800';
      case 'degraded': return 'bg-yellow-100 text-yellow-800';
      case 'critical': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const formatLatency = (ms: number) => {
    return ms < 200 ? 'text-green-600' : ms < 500 ? 'text-yellow-600' : 'text-red-600';
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Real-Time Market Data</h1>
        <p className="text-gray-600">Institutional-grade quotes with sub-200ms latency</p>
      </div>

      {/* Symbol Input & Controls */}
      <div className="bg-white p-6 rounded-lg shadow-md mb-6">
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Symbol
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={inputSymbol}
                onChange={(e) => setInputSymbol(e.target.value.toUpperCase())}
                onKeyPress={(e) => e.key === 'Enter' && handleSymbolChange()}
                className="flex-1 px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter symbol (e.g., AAPL)"
              />
              <button
                onClick={handleSymbolChange}
                className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
              >
                Load
              </button>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Streaming
            </label>
            <button
              onClick={handleStreamToggle}
              className={`px-6 py-2 rounded-md transition-colors ${
                isStreaming
                  ? 'bg-red-600 text-white hover:bg-red-700'
                  : 'bg-green-600 text-white hover:bg-green-700'
              }`}
            >
              {isStreaming ? 'Stop Stream' : 'Start Stream'}
            </button>
          </div>
        </div>
      </div>

      {/* Health Status */}
      {health && (
        <div className={`p-4 rounded-lg mb-6 ${getHealthColor(health.status)}`}>
          <div className="flex items-center justify-between">
            <div>
              <span className="font-semibold">{health.status.toUpperCase()}</span>
              <span className="ml-2">{health.message}</span>
            </div>
            <div className="text-sm">
              {health.providers_connected}/{health.providers_total} providers
              {health.avg_p95_latency_ms && (
                <span className="ml-2">• {health.avg_p95_latency_ms.toFixed(0)}ms p95</span>
              )}
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Real-Time Quote */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold text-gray-900">{symbol} Quote</h2>
            {isStreaming && (
              <span className="flex items-center text-sm text-green-600">
                <span className="animate-pulse mr-2">●</span>
                Live
              </span>
            )}
          </div>

          {loading && <div className="text-center py-8">Loading...</div>}

          {quote && (
            <div className="space-y-4">
              {/* Bid/Ask */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-green-50 p-4 rounded-lg">
                  <div className="text-sm text-gray-600 mb-1">Best Bid</div>
                  <div className="text-2xl font-bold text-green-700">${quote.best_bid.toFixed(2)}</div>
                  <div className="text-xs text-gray-500 mt-1">
                    Size: {quote.bid_size.toLocaleString()}
                  </div>
                  <div className="text-xs text-gray-500">
                    {quote.best_bid_provider}
                  </div>
                </div>

                <div className="bg-red-50 p-4 rounded-lg">
                  <div className="text-sm text-gray-600 mb-1">Best Ask</div>
                  <div className="text-2xl font-bold text-red-700">${quote.best_ask.toFixed(2)}</div>
                  <div className="text-xs text-gray-500 mt-1">
                    Size: {quote.ask_size.toLocaleString()}
                  </div>
                  <div className="text-xs text-gray-500">
                    {quote.best_ask_provider}
                  </div>
                </div>
              </div>

              {/* Mid Price & Spread */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-sm text-gray-600">Mid Price</div>
                  <div className="text-xl font-semibold">${quote.mid_price.toFixed(2)}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-600">Spread</div>
                  <div className="text-xl font-semibold">
                    {quote.spread_bps.toFixed(1)} bps
                  </div>
                </div>
              </div>

              {/* Last Trade */}
              <div>
                <div className="text-sm text-gray-600">Last Trade</div>
                <div className="text-xl font-semibold">${quote.last.toFixed(2)}</div>
              </div>

              {/* Metadata */}
              <div className="border-t pt-4 space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Providers</span>
                  <span className="font-medium">{quote.num_providers}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Latency</span>
                  <span className={`font-medium ${formatLatency(quote.avg_latency_ms)}`}>
                    {quote.avg_latency_ms.toFixed(0)}ms
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Updated</span>
                  <span className="font-medium">
                    {new Date(quote.timestamp).toLocaleTimeString()}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Level 2 Order Book */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Level 2 Order Book</h2>

          {orderBook && (
            <div>
              {/* Order Book Imbalance */}
              <div className="mb-4 p-3 bg-gray-50 rounded-lg">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Order Book Imbalance</span>
                  <span className={`font-semibold ${
                    orderBook.imbalance > 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {(orderBook.imbalance * 100).toFixed(1)}%
                    {orderBook.imbalance > 0 ? ' Bid' : ' Ask'}
                  </span>
                </div>
              </div>

              {/* Order Book Levels */}
              <div className="space-y-2">
                {/* Asks (reversed to show highest at bottom) */}
                <div className="space-y-1">
                  <div className="text-xs font-semibold text-gray-500 uppercase">Asks</div>
                  {orderBook.asks.slice(0, 5).reverse().map((level, idx) => (
                    <div key={`ask-${idx}`} className="flex justify-between text-sm py-1 px-2 bg-red-50 rounded">
                      <span className="text-red-700 font-medium">${level.price.toFixed(2)}</span>
                      <span className="text-gray-600">{level.size.toLocaleString()}</span>
                    </div>
                  ))}
                </div>

                {/* Spread */}
                <div className="py-2 text-center border-y border-gray-300">
                  <span className="text-xs font-semibold text-gray-500">
                    SPREAD: {orderBook.spread_bps.toFixed(1)} bps
                  </span>
                </div>

                {/* Bids */}
                <div className="space-y-1">
                  <div className="text-xs font-semibold text-gray-500 uppercase">Bids</div>
                  {orderBook.bids.slice(0, 5).map((level, idx) => (
                    <div key={`bid-${idx}`} className="flex justify-between text-sm py-1 px-2 bg-green-50 rounded">
                      <span className="text-green-700 font-medium">${level.price.toFixed(2)}</span>
                      <span className="text-gray-600">{level.size.toLocaleString()}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Provider Status & Latency */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Provider Status */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Data Providers</h2>

          {providerStatus && (
            <div className="space-y-3">
              {Object.entries(providerStatus.providers).map(([provider, isConnected]) => (
                <div key={provider} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center">
                    <span className={`w-3 h-3 rounded-full mr-3 ${
                      isConnected ? 'bg-green-500' : 'bg-red-500'
                    }`} />
                    <span className="font-medium capitalize">{provider.replace('_', ' ')}</span>
                  </div>
                  <span className={`text-sm ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
                    {isConnected ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
              ))}

              <div className="mt-4 pt-4 border-t">
                <div className="text-sm text-gray-600">
                  {providerStatus.connected_count} of {providerStatus.total_count} providers connected
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Latency Statistics */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Latency Statistics</h2>

          {latencyStats && (
            <div className="space-y-4">
              {Object.entries(latencyStats).map(([provider, stats]) => (
                <div key={provider} className="border-b pb-3 last:border-b-0">
                  <div className="font-medium capitalize mb-2">{provider.replace('_', ' ')}</div>
                  <div className="grid grid-cols-3 gap-2 text-sm">
                    <div>
                      <div className="text-gray-600">Avg</div>
                      <div className={`font-semibold ${formatLatency(stats.avg_ms)}`}>
                        {stats.avg_ms.toFixed(0)}ms
                      </div>
                    </div>
                    <div>
                      <div className="text-gray-600">P95</div>
                      <div className={`font-semibold ${formatLatency(stats.p95_ms)}`}>
                        {stats.p95_ms.toFixed(0)}ms
                      </div>
                    </div>
                    <div>
                      <div className="text-gray-600">P99</div>
                      <div className={`font-semibold ${formatLatency(stats.p99_ms)}`}>
                        {stats.p99_ms.toFixed(0)}ms
                      </div>
                    </div>
                  </div>
                </div>
              ))}

              <div className="pt-2 text-xs text-gray-500">
                Target: &lt;200ms latency • Green: Good • Yellow: Fair • Red: Poor
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
