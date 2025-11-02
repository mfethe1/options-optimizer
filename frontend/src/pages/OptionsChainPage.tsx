import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  getOptionsChain,
  getExpirations,
  getOptionsSummary,
  createOptionsChainWebSocket,
  OptionsChain,
  OptionStrike,
  OptionsSummary,
} from '../services/optionsChainApi';
import toast from 'react-hot-toast';

const OptionsChainPage: React.FC = () => {
  const [symbol, setSymbol] = useState('AAPL');
  const [inputSymbol, setInputSymbol] = useState('AAPL');
  const [loading, setLoading] = useState(false);
  const [chain, setChain] = useState<OptionsChain | null>(null);
  const [summary, setSummary] = useState<OptionsSummary | null>(null);
  const [expirations, setExpirations] = useState<string[]>([]);
  const [selectedExpiration, setSelectedExpiration] = useState<string>('');
  const [strikes, setStrikes] = useState<OptionStrike[]>([]);
  const [liveUpdates, setLiveUpdates] = useState(true);
  const [focusedCell, setFocusedCell] = useState<{ row: number; col: string } | null>(null);
  const [sortBy, setSortBy] = useState<string>('strike');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');
  const wsRef = useRef<WebSocket | null>(null);

  // Load data for symbol
  const loadSymbol = useCallback(async () => {
    setLoading(true);
    try {
      // Load expirations
      const exps = await getExpirations(symbol);
      setExpirations(exps);

      // Select first expiration by default
      const defaultExp = exps[0];
      setSelectedExpiration(defaultExp);

      // Load chain for first expiration
      const chainData = await getOptionsChain(symbol, defaultExp);
      setChain(chainData);
      setStrikes(chainData.strikes[defaultExp] || []);

      // Load summary
      const summaryData = await getOptionsSummary(symbol);
      setSummary(summaryData);

      toast.success(`Loaded options chain for ${symbol}`);
    } catch (error: any) {
      toast.error(error.message || 'Failed to load options chain');
      console.error('Error loading options chain:', error);
    } finally {
      setLoading(false);
    }
  }, [symbol]);

  // Change expiration
  const handleExpirationChange = useCallback(
    async (exp: string) => {
      setSelectedExpiration(exp);
      if (chain) {
        setStrikes(chain.strikes[exp] || []);
      } else {
        // Reload if not cached
        try {
          const chainData = await getOptionsChain(symbol, exp);
          setChain(chainData);
          setStrikes(chainData.strikes[exp] || []);
        } catch (error) {
          toast.error('Failed to load expiration data');
        }
      }
    },
    [chain, symbol]
  );

  // WebSocket connection
  useEffect(() => {
    if (!liveUpdates || !symbol || !selectedExpiration) return;

    const ws = createOptionsChainWebSocket(
      symbol,
      (data) => {
        if (data.type === 'update' && data.data) {
          setChain(data.data);
          setStrikes(data.data.strikes[selectedExpiration] || []);
        }
      },
      selectedExpiration
    );

    wsRef.current = ws;

    return () => {
      ws.close();
    };
  }, [symbol, selectedExpiration, liveUpdates]);

  // Initial load
  useEffect(() => {
    loadSymbol();
  }, [symbol]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // G - Refresh
      if (e.key === 'g' && !e.ctrlKey && !e.metaKey) {
        loadSymbol();
      }
      // L - Toggle live updates
      if (e.key === 'l' && !e.ctrlKey && !e.metaKey) {
        setLiveUpdates((prev) => !prev);
        toast.success(liveUpdates ? 'Live updates disabled' : 'Live updates enabled');
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [loadSymbol, liveUpdates]);

  // Handle symbol search
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputSymbol.trim()) {
      setSymbol(inputSymbol.trim().toUpperCase());
    }
  };

  // Sort strikes
  const sortedStrikes = React.useMemo(() => {
    if (!strikes) return [];

    const sorted = [...strikes];
    sorted.sort((a, b) => {
      let aVal: any = a.strike;
      let bVal: any = b.strike;

      switch (sortBy) {
        case 'strike':
          aVal = a.strike;
          bVal = b.strike;
          break;
        case 'call_volume':
          aVal = a.call.volume;
          bVal = b.call.volume;
          break;
        case 'put_volume':
          aVal = a.put.volume;
          bVal = b.put.volume;
          break;
        case 'call_oi':
          aVal = a.call.open_interest;
          bVal = b.call.open_interest;
          break;
        case 'put_oi':
          aVal = a.put.open_interest;
          bVal = b.put.open_interest;
          break;
        case 'call_iv':
          aVal = a.call.iv || 0;
          bVal = b.call.iv || 0;
          break;
        case 'put_iv':
          aVal = a.put.iv || 0;
          bVal = b.put.iv || 0;
          break;
      }

      if (sortOrder === 'asc') {
        return aVal - bVal;
      } else {
        return bVal - aVal;
      }
    });

    return sorted;
  }, [strikes, sortBy, sortOrder]);

  // Format currency
  const fmt = (val: number | null) => {
    if (val === null || val === undefined) return '-';
    return val.toFixed(2);
  };

  // Format percentage
  const fmtPct = (val: number | null) => {
    if (val === null || val === undefined) return '-';
    return `${(val * 100).toFixed(1)}%`;
  };

  // Get cell background color
  const getCellBg = (isCall: boolean, inTheMoney: boolean, unusualVolume: boolean) => {
    if (unusualVolume) return 'bg-yellow-100';
    if (inTheMoney) {
      return isCall ? 'bg-green-50' : 'bg-red-50';
    }
    return 'bg-white';
  };

  return (
    <div className="max-w-full mx-auto">
      {/* Header */}
      <div className="mb-4">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Options Chain
        </h1>
        <p className="text-gray-600">
          Bloomberg OMON equivalent • Real-time options data with Greeks
        </p>
      </div>

      {/* Search and Controls */}
      <div className="bg-white rounded-lg shadow p-4 mb-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Symbol Search */}
          <div>
            <form onSubmit={handleSearch} className="flex gap-2">
              <input
                type="text"
                value={inputSymbol}
                onChange={(e) => setInputSymbol(e.target.value.toUpperCase())}
                placeholder="Symbol..."
                className="flex-1 border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono"
              />
              <button
                type="submit"
                disabled={loading}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
              >
                Load
              </button>
            </form>
          </div>

          {/* Expiration Selector */}
          <div>
            <select
              value={selectedExpiration}
              onChange={(e) => handleExpirationChange(e.target.value)}
              className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {expirations.map((exp) => (
                <option key={exp} value={exp}>
                  {new Date(exp).toLocaleDateString('en-US', {
                    month: 'short',
                    day: 'numeric',
                    year: 'numeric',
                  })}
                  {' '}
                  ({Math.ceil((new Date(exp).getTime() - Date.now()) / (1000 * 60 * 60 * 24))}d)
                </option>
              ))}
            </select>
          </div>

          {/* Live Updates Toggle */}
          <div className="flex items-center gap-2">
            <label className="flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={liveUpdates}
                onChange={(e) => setLiveUpdates(e.target.checked)}
                className="mr-2"
              />
              <span className="text-sm">Live Updates</span>
            </label>
            {liveUpdates && (
              <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            )}
          </div>

          {/* Refresh Button */}
          <div>
            <button
              onClick={loadSymbol}
              disabled={loading}
              className="w-full px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 disabled:opacity-50"
            >
              {loading ? 'Loading...' : 'Refresh (G)'}
            </button>
          </div>
        </div>
      </div>

      {/* Summary Card */}
      {summary && (
        <div className="bg-white rounded-lg shadow p-4 mb-4">
          <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
            <div>
              <div className="text-xs text-gray-600">Current Price</div>
              <div className="text-xl font-bold">${summary.current_price.toFixed(2)}</div>
              <div
                className={`text-sm ${
                  summary.price_change >= 0 ? 'text-green-600' : 'text-red-600'
                }`}
              >
                {summary.price_change >= 0 ? '+' : ''}
                {summary.price_change.toFixed(2)} ({summary.price_change_pct.toFixed(2)}%)
              </div>
            </div>

            <div>
              <div className="text-xs text-gray-600">IV Rank</div>
              <div className="text-xl font-bold">
                {summary.iv_rank !== null ? `${summary.iv_rank.toFixed(1)}%` : 'N/A'}
              </div>
              <div className="text-xs text-gray-500">
                {summary.iv_percentile !== null
                  ? `${summary.iv_percentile.toFixed(0)}th percentile`
                  : ''}
              </div>
            </div>

            <div>
              <div className="text-xs text-gray-600">HV (20-day)</div>
              <div className="text-xl font-bold">
                {summary.hv_20 !== null ? `${summary.hv_20.toFixed(1)}%` : 'N/A'}
              </div>
            </div>

            <div>
              <div className="text-xs text-gray-600">HV (30-day)</div>
              <div className="text-xl font-bold">
                {summary.hv_30 !== null ? `${summary.hv_30.toFixed(1)}%` : 'N/A'}
              </div>
            </div>

            <div>
              <div className="text-xs text-gray-600">Max Pain</div>
              <div className="text-xl font-bold">
                {summary.max_pain !== null ? `$${summary.max_pain.toFixed(2)}` : 'N/A'}
              </div>
            </div>

            <div>
              <div className="text-xs text-gray-600">P/C Ratio</div>
              <div className="text-xl font-bold">
                {summary.put_call_ratio_volume !== null
                  ? summary.put_call_ratio_volume.toFixed(3)
                  : 'N/A'}
              </div>
              <div className="text-xs text-gray-500">by volume</div>
            </div>
          </div>
        </div>
      )}

      {/* Options Chain Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-100 border-b-2 border-gray-300">
              <tr>
                {/* Call Headers */}
                <th className="px-2 py-2 text-left text-xs font-semibold text-gray-700 border-r">
                  Bid
                </th>
                <th className="px-2 py-2 text-left text-xs font-semibold text-gray-700 border-r">
                  Ask
                </th>
                <th className="px-2 py-2 text-left text-xs font-semibold text-gray-700 border-r">
                  Last
                </th>
                <th
                  className="px-2 py-2 text-left text-xs font-semibold text-gray-700 border-r cursor-pointer hover:bg-gray-200"
                  onClick={() => {
                    setSortBy('call_volume');
                    setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
                  }}
                >
                  Volume {sortBy === 'call_volume' && (sortOrder === 'asc' ? '↑' : '↓')}
                </th>
                <th
                  className="px-2 py-2 text-left text-xs font-semibold text-gray-700 border-r cursor-pointer hover:bg-gray-200"
                  onClick={() => {
                    setSortBy('call_oi');
                    setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
                  }}
                >
                  OI {sortBy === 'call_oi' && (sortOrder === 'asc' ? '↑' : '↓')}
                </th>
                <th
                  className="px-2 py-2 text-left text-xs font-semibold text-gray-700 border-r cursor-pointer hover:bg-gray-200"
                  onClick={() => {
                    setSortBy('call_iv');
                    setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
                  }}
                >
                  IV {sortBy === 'call_iv' && (sortOrder === 'asc' ? '↑' : '↓')}
                </th>
                <th className="px-2 py-2 text-left text-xs font-semibold text-gray-700 border-r">
                  Δ
                </th>
                <th className="px-2 py-2 text-left text-xs font-semibold text-gray-700 border-r">
                  Γ
                </th>
                <th className="px-2 py-2 text-left text-xs font-semibold text-gray-700 border-r-4 border-gray-400">
                  Θ
                </th>

                {/* Strike */}
                <th
                  className="px-4 py-2 text-center text-xs font-bold text-gray-900 bg-gray-200 cursor-pointer hover:bg-gray-300"
                  onClick={() => {
                    setSortBy('strike');
                    setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
                  }}
                >
                  STRIKE {sortBy === 'strike' && (sortOrder === 'asc' ? '↑' : '↓')}
                </th>

                {/* Put Headers */}
                <th className="px-2 py-2 text-left text-xs font-semibold text-gray-700 border-l-4 border-gray-400">
                  Θ
                </th>
                <th className="px-2 py-2 text-left text-xs font-semibold text-gray-700 border-l">
                  Γ
                </th>
                <th className="px-2 py-2 text-left text-xs font-semibold text-gray-700 border-l">
                  Δ
                </th>
                <th
                  className="px-2 py-2 text-left text-xs font-semibold text-gray-700 border-l cursor-pointer hover:bg-gray-200"
                  onClick={() => {
                    setSortBy('put_iv');
                    setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
                  }}
                >
                  IV {sortBy === 'put_iv' && (sortOrder === 'asc' ? '↑' : '↓')}
                </th>
                <th
                  className="px-2 py-2 text-left text-xs font-semibold text-gray-700 border-l cursor-pointer hover:bg-gray-200"
                  onClick={() => {
                    setSortBy('put_oi');
                    setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
                  }}
                >
                  OI {sortBy === 'put_oi' && (sortOrder === 'asc' ? '↑' : '↓')}
                </th>
                <th
                  className="px-2 py-2 text-left text-xs font-semibold text-gray-700 border-l cursor-pointer hover:bg-gray-200"
                  onClick={() => {
                    setSortBy('put_volume');
                    setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
                  }}
                >
                  Volume {sortBy === 'put_volume' && (sortOrder === 'asc' ? '↑' : '↓')}
                </th>
                <th className="px-2 py-2 text-left text-xs font-semibold text-gray-700 border-l">
                  Last
                </th>
                <th className="px-2 py-2 text-left text-xs font-semibold text-gray-700 border-l">
                  Ask
                </th>
                <th className="px-2 py-2 text-left text-xs font-semibold text-gray-700 border-l">
                  Bid
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedStrikes.map((strike, idx) => (
                <tr
                  key={idx}
                  className="border-b border-gray-200 hover:bg-gray-50 text-xs"
                >
                  {/* Calls */}
                  <td className={`px-2 py-1 ${getCellBg(true, strike.call.in_the_money, strike.call.unusual_volume)}`}>
                    {fmt(strike.call.bid)}
                  </td>
                  <td className={`px-2 py-1 ${getCellBg(true, strike.call.in_the_money, strike.call.unusual_volume)}`}>
                    {fmt(strike.call.ask)}
                  </td>
                  <td className={`px-2 py-1 font-semibold ${getCellBg(true, strike.call.in_the_money, strike.call.unusual_volume)}`}>
                    {fmt(strike.call.last)}
                  </td>
                  <td className={`px-2 py-1 ${getCellBg(true, strike.call.in_the_money, strike.call.unusual_volume)} ${strike.call.unusual_volume ? 'font-bold text-orange-700' : ''}`}>
                    {strike.call.volume.toLocaleString()}
                  </td>
                  <td className={`px-2 py-1 ${getCellBg(true, strike.call.in_the_money, strike.call.unusual_volume)}`}>
                    {strike.call.open_interest.toLocaleString()}
                  </td>
                  <td className={`px-2 py-1 ${getCellBg(true, strike.call.in_the_money, strike.call.unusual_volume)}`}>
                    {fmtPct(strike.call.iv)}
                  </td>
                  <td className={`px-2 py-1 ${getCellBg(true, strike.call.in_the_money, strike.call.unusual_volume)}`}>
                    {fmt(strike.call.delta)}
                  </td>
                  <td className={`px-2 py-1 ${getCellBg(true, strike.call.in_the_money, strike.call.unusual_volume)}`}>
                    {fmt(strike.call.gamma)}
                  </td>
                  <td className={`px-2 py-1 ${getCellBg(true, strike.call.in_the_money, strike.call.unusual_volume)}`}>
                    {fmt(strike.call.theta)}
                  </td>

                  {/* Strike */}
                  <td className="px-4 py-1 text-center font-bold bg-gray-100">
                    ${strike.strike.toFixed(2)}
                  </td>

                  {/* Puts */}
                  <td className={`px-2 py-1 ${getCellBg(false, strike.put.in_the_money, strike.put.unusual_volume)}`}>
                    {fmt(strike.put.theta)}
                  </td>
                  <td className={`px-2 py-1 ${getCellBg(false, strike.put.in_the_money, strike.put.unusual_volume)}`}>
                    {fmt(strike.put.gamma)}
                  </td>
                  <td className={`px-2 py-1 ${getCellBg(false, strike.put.in_the_money, strike.put.unusual_volume)}`}>
                    {fmt(strike.put.delta)}
                  </td>
                  <td className={`px-2 py-1 ${getCellBg(false, strike.put.in_the_money, strike.put.unusual_volume)}`}>
                    {fmtPct(strike.put.iv)}
                  </td>
                  <td className={`px-2 py-1 ${getCellBg(false, strike.put.in_the_money, strike.put.unusual_volume)}`}>
                    {strike.put.open_interest.toLocaleString()}
                  </td>
                  <td className={`px-2 py-1 ${getCellBg(false, strike.put.in_the_money, strike.put.unusual_volume)} ${strike.put.unusual_volume ? 'font-bold text-orange-700' : ''}`}>
                    {strike.put.volume.toLocaleString()}
                  </td>
                  <td className={`px-2 py-1 font-semibold ${getCellBg(false, strike.put.in_the_money, strike.put.unusual_volume)}`}>
                    {fmt(strike.put.last)}
                  </td>
                  <td className={`px-2 py-1 ${getCellBg(false, strike.put.in_the_money, strike.put.unusual_volume)}`}>
                    {fmt(strike.put.ask)}
                  </td>
                  <td className={`px-2 py-1 ${getCellBg(false, strike.put.in_the_money, strike.put.unusual_volume)}`}>
                    {fmt(strike.put.bid)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Keyboard Shortcuts Help */}
      <div className="mt-4 bg-blue-50 rounded-lg p-3 text-sm">
        <div className="font-semibold text-blue-900 mb-2">⌨️ Keyboard Shortcuts</div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-blue-800">
          <div><kbd className="px-2 py-1 bg-white rounded">G</kbd> Refresh data</div>
          <div><kbd className="px-2 py-1 bg-white rounded">L</kbd> Toggle live updates</div>
          <div><kbd className="px-2 py-1 bg-white rounded">/</kbd> Search symbol</div>
          <div><span className="text-yellow-600">●</span> Unusual volume</div>
        </div>
      </div>
    </div>
  );
};

export default OptionsChainPage;
