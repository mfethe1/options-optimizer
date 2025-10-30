import React, { useState, useEffect, useRef } from 'react';
import {
  detectAnomalies,
  scanAnomalies,
  Anomaly,
  createAnomalyWebSocket,
  subscribeToSymbols,
} from '../services/anomalyApi';
import toast from 'react-hot-toast';

const AnomalyDetectionPage: React.FC = () => {
  const [symbol, setSymbol] = useState('');
  const [loading, setLoading] = useState(false);
  const [anomalies, setAnomalies] = useState<Record<string, Anomaly[]>>({});
  const [watchlist, setWatchlist] = useState<string[]>(['NVDA', 'AAPL', 'TSLA', 'AMD']);
  const [liveAlerts, setLiveAlerts] = useState<any[]>([]);
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const userId = 'demo_user';

  useEffect(() => {
    // Connect to WebSocket for real-time alerts
    const ws = createAnomalyWebSocket(
      userId,
      (data) => {
        if (data.type === 'anomaly_alert') {
          setLiveAlerts((prev) => [
            { ...data.data, timestamp: new Date() },
            ...prev.slice(0, 49), // Keep last 50 alerts
          ]);
          toast.success(`Anomaly detected: ${data.data.symbol}`, {
            icon: '🚨',
            duration: 5000,
          });
        } else if (data.type === 'connected') {
          setWsConnected(true);
          console.log('WebSocket connected');
        }
      },
      (error) => {
        setWsConnected(false);
        console.error('WebSocket error:', error);
      }
    );

    wsRef.current = ws;

    // Subscribe to watchlist when connected
    ws.addEventListener('open', () => {
      setTimeout(() => {
        subscribeToSymbols(ws, watchlist);
      }, 1000);
    });

    return () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    // Update subscription when watchlist changes
    if (wsRef.current && wsConnected) {
      subscribeToSymbols(wsRef.current, watchlist);
    }
  }, [watchlist, wsConnected]);

  const handleDetectSingle = async () => {
    if (!symbol.trim()) {
      toast.error('Please enter a symbol');
      return;
    }

    setLoading(true);
    try {
      const result = await detectAnomalies({ symbol: symbol.toUpperCase() });
      setAnomalies((prev) => ({
        ...prev,
        [result.symbol]: result.anomalies,
      }));

      if (result.count > 0) {
        toast.success(`Found ${result.count} anomaly(ies) for ${result.symbol}`);
      } else {
        toast.info(`No anomalies detected for ${result.symbol}`);
      }
    } catch (error: any) {
      toast.error(error.message || 'Failed to detect anomalies');
    } finally {
      setLoading(false);
    }
  };

  const handleScanWatchlist = async () => {
    if (watchlist.length === 0) {
      toast.error('Watchlist is empty');
      return;
    }

    setLoading(true);
    try {
      const result = await scanAnomalies(watchlist);
      setAnomalies(result.results || {});
      toast.success(
        `Scanned ${result.symbols_scanned} symbols, found anomalies in ${result.symbols_with_anomalies}`
      );
    } catch (error: any) {
      toast.error(error.message || 'Failed to scan watchlist');
    } finally {
      setLoading(false);
    }
  };

  const addToWatchlist = () => {
    const sym = symbol.toUpperCase().trim();
    if (sym && !watchlist.includes(sym)) {
      setWatchlist([...watchlist, sym]);
      setSymbol('');
    }
  };

  const removeFromWatchlist = (sym: string) => {
    setWatchlist(watchlist.filter((s) => s !== sym));
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-100 border-red-500 text-red-800';
      case 'high':
        return 'bg-orange-100 border-orange-500 text-orange-800';
      case 'medium':
        return 'bg-yellow-100 border-yellow-500 text-yellow-800';
      default:
        return 'bg-blue-100 border-blue-500 text-blue-800';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'volume_spike':
        return '📊';
      case 'price_movement':
        return '💹';
      case 'iv_expansion':
        return '⚡';
      case 'options_flow':
        return '🎯';
      default:
        return '🔔';
    }
  };

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">
              Anomaly Detection
            </h1>
            <p className="text-gray-600 mt-2">
              Real-time statistical anomaly detection for unusual market activity
            </p>
          </div>
          <div className="flex items-center gap-2">
            <div
              className={`w-3 h-3 rounded-full ${
                wsConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
              }`}
            />
            <span className="text-sm text-gray-600">
              {wsConnected ? 'Live' : 'Disconnected'}
            </span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls */}
        <div className="space-y-4">
          <div className="bg-white rounded-lg shadow p-4">
            <h2 className="font-semibold mb-4">Detect Anomalies</h2>
            <div className="space-y-3">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                  onKeyPress={(e) => e.key === 'Enter' && handleDetectSingle()}
                  placeholder="Enter symbol..."
                  className="flex-1 border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <button
                  onClick={handleDetectSingle}
                  disabled={loading}
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
                >
                  Detect
                </button>
              </div>
              <button
                onClick={addToWatchlist}
                disabled={!symbol.trim()}
                className="w-full text-sm text-blue-600 hover:text-blue-700 disabled:opacity-50"
              >
                + Add to watchlist
              </button>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-semibold">Watchlist</h2>
              <button
                onClick={handleScanWatchlist}
                disabled={loading || watchlist.length === 0}
                className="text-sm px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
              >
                Scan All
              </button>
            </div>
            <div className="space-y-2">
              {watchlist.map((sym) => (
                <div
                  key={sym}
                  className="flex items-center justify-between p-2 bg-gray-50 rounded"
                >
                  <span className="font-mono font-medium">{sym}</span>
                  <button
                    onClick={() => removeFromWatchlist(sym)}
                    className="text-red-600 hover:text-red-700 text-sm"
                  >
                    Remove
                  </button>
                </div>
              ))}
              {watchlist.length === 0 && (
                <p className="text-sm text-gray-500 text-center py-4">
                  No symbols in watchlist
                </p>
              )}
            </div>
          </div>

          <div className="bg-blue-50 rounded-lg p-4">
            <h3 className="font-semibold text-blue-900 mb-2">
              Detection Types
            </h3>
            <div className="text-sm text-blue-800 space-y-1">
              <div>📊 <strong>Volume Spikes</strong> (3+ σ)</div>
              <div>💹 <strong>Price Anomalies</strong> (2.5+ σ)</div>
              <div>⚡ <strong>IV Expansion</strong> (2+ σ)</div>
              <div>🎯 <strong>Options Flow</strong> (blocks)</div>
            </div>
          </div>
        </div>

        {/* Detected Anomalies */}
        <div className="lg:col-span-2 space-y-4">
          <div className="bg-white rounded-lg shadow p-4">
            <h2 className="font-semibold mb-4">Detected Anomalies</h2>
            <div className="space-y-3 max-h-[600px] overflow-y-auto">
              {Object.entries(anomalies).flatMap(([sym, anomList]) =>
                anomList.map((anomaly, idx) => (
                  <div
                    key={`${sym}-${idx}`}
                    className={`border-l-4 rounded-lg p-4 ${getSeverityColor(
                      anomaly.severity
                    )}`}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-2xl">{getTypeIcon(anomaly.type)}</span>
                        <div>
                          <div className="font-mono font-bold text-lg">{sym}</div>
                          <div className="text-sm opacity-80">
                            {anomaly.type.replace('_', ' ')}
                          </div>
                        </div>
                      </div>
                      <span className="text-xs font-semibold uppercase">
                        {anomaly.severity}
                      </span>
                    </div>

                    {anomaly.z_score && (
                      <div className="mb-2 text-sm">
                        <strong>Z-Score:</strong> {anomaly.z_score.toFixed(2)}
                      </div>
                    )}

                    {anomaly.multiplier && (
                      <div className="mb-2 text-sm">
                        <strong>Multiplier:</strong> {anomaly.multiplier.toFixed(1)}x
                      </div>
                    )}

                    {anomaly.current_value !== undefined && (
                      <div className="mb-2 text-sm">
                        <strong>Current:</strong> {anomaly.current_value.toLocaleString()}
                        {' vs '}
                        <strong>Avg:</strong> {anomaly.average_value?.toLocaleString()}
                      </div>
                    )}

                    <div className="mt-3 text-sm bg-white bg-opacity-50 rounded p-2">
                      <strong>Implication:</strong> {anomaly.trading_implication}
                    </div>

                    <div className="mt-2 text-xs opacity-70">
                      Detected: {new Date(anomaly.detected_at).toLocaleTimeString()}
                    </div>
                  </div>
                ))
              )}
              {Object.keys(anomalies).length === 0 && (
                <div className="text-center py-12 text-gray-500">
                  <svg
                    className="w-16 h-16 mx-auto mb-4 text-gray-400"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                    />
                  </svg>
                  <p>No anomalies detected yet</p>
                  <p className="text-sm mt-2">
                    Enter a symbol or scan your watchlist
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Live Alerts */}
          {liveAlerts.length > 0 && (
            <div className="bg-white rounded-lg shadow p-4">
              <h2 className="font-semibold mb-4 flex items-center gap-2">
                <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                Live Alerts
              </h2>
              <div className="space-y-2 max-h-[300px] overflow-y-auto">
                {liveAlerts.map((alert, idx) => (
                  <div
                    key={idx}
                    className="border border-gray-200 rounded p-3 text-sm"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-mono font-bold">{alert.symbol}</span>
                      <span className="text-xs text-gray-500">
                        {alert.timestamp.toLocaleTimeString()}
                      </span>
                    </div>
                    <div className="text-gray-700">{alert.anomaly.type}</div>
                    <div className="text-xs text-gray-600 mt-1">
                      {alert.anomaly.trading_implication}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AnomalyDetectionPage;
