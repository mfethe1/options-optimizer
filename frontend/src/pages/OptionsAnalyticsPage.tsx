import React, { useState, useEffect } from 'react';
import toast from 'react-hot-toast';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface SkewMetrics {
  expiration: string;
  days_to_expiry: number;
  atm_iv: number;
  put_skew: number;
  call_skew: number;
  skew_slope: number;
  risk_reversal_25delta: number;
  butterfly_25delta: number;
}

interface TermStructure {
  front_month_iv: number;
  back_month_iv: number;
  term_structure_slope: number;
  is_inverted: boolean;
  point_count: number;
}

interface IVSurface {
  min_iv: number;
  max_iv: number;
  atm_iv: number;
  iv_range: number;
  point_count: number;
}

interface CompleteAnalytics {
  symbol: string;
  spot_price: number;
  iv_surface: IVSurface;
  skew_metrics: SkewMetrics[];
  term_structures: {
    atm?: TermStructure;
    otm_put?: TermStructure;
    otm_call?: TermStructure;
  };
  calculation_time: string;
}

const OptionsAnalyticsPage: React.FC = () => {
  const [symbol, setSymbol] = useState('AAPL');
  const [analytics, setAnalytics] = useState<CompleteAnalytics | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (symbol) {
      loadAnalytics();
    }
  }, []);

  const loadAnalytics = async () => {
    if (!symbol) return;

    setLoading(true);
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/options-analytics/${symbol.toUpperCase()}/complete`
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch analytics: ${response.statusText}`);
      }

      const data = await response.json();
      setAnalytics(data);
      toast.success('Analytics loaded successfully');
    } catch (error: any) {
      toast.error(error.message || 'Failed to load analytics');
      console.error('Analytics load error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSymbolSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    loadAnalytics();
  };

  const getSkewInterpretation = (putSkew: number, callSkew: number) => {
    if (Math.abs(putSkew) > Math.abs(callSkew) && putSkew > 0.05) {
      return {
        text: 'Put skew elevated - Fear in market, protective puts expensive',
        color: 'text-red-600',
        recommendation: 'Consider selling put spreads or buying call spreads'
      };
    } else if (Math.abs(callSkew) > Math.abs(putSkew) && callSkew > 0.05) {
      return {
        text: 'Call skew elevated - Upside chase, calls expensive',
        color: 'text-orange-600',
        recommendation: 'Consider selling call spreads or buying put spreads'
      };
    } else {
      return {
        text: 'Balanced skew - Normal market conditions',
        color: 'text-green-600',
        recommendation: 'Neutral strategies like iron condors may work well'
      };
    }
  };

  const getTermStructureInterpretation = (ts: TermStructure) => {
    if (ts.is_inverted) {
      return {
        text: 'Backwardation - Near-term event risk priced in',
        color: 'text-red-600',
        recommendation: 'Potential earnings or catalyst soon. Calendar spreads (sell front, buy back) profitable'
      };
    } else if (ts.term_structure_slope > 0.02) {
      return {
        text: 'Steep Contango - High uncertainty priced in longer term',
        color: 'text-orange-600',
        recommendation: 'Reverse calendar spreads (buy front, sell back) may profit from mean reversion'
      };
    } else {
      return {
        text: 'Normal term structure - Standard IV pricing',
        color: 'text-green-600',
        recommendation: 'Consider directional plays or standard spreads'
      };
    }
  };

  const getRiskReversalInterpretation = (rr: number) => {
    if (rr > 0.03) {
      return {
        text: 'Bullish sentiment - Call IVs > Put IVs',
        color: 'text-green-600',
        recommendation: 'Market expects upside. Consider put credit spreads or covered calls'
      };
    } else if (rr < -0.03) {
      return {
        text: 'Bearish sentiment - Put IVs > Call IVs',
        color: 'text-red-600',
        recommendation: 'Market expects downside. Consider call credit spreads or protective puts'
      };
    } else {
      return {
        text: 'Neutral sentiment - Balanced IV',
        color: 'text-gray-600',
        recommendation: 'Market neutral. Consider iron condors or strangles'
      };
    }
  };

  const formatPercent = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">
          📊 Options Analytics
        </h1>
        <p className="text-gray-600 mt-2">
          Advanced options analytics: IV surface, volatility skew, term structure
        </p>
      </div>

      {/* Symbol Input */}
      <form onSubmit={handleSymbolSubmit} className="mb-6">
        <div className="flex gap-3">
          <input
            type="text"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            placeholder="Enter symbol (e.g., AAPL)"
            className="flex-1 max-w-xs border border-gray-300 rounded px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            type="submit"
            disabled={loading}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? 'Loading...' : 'Analyze'}
          </button>
        </div>
      </form>

      {!analytics ? (
        <div className="bg-white rounded-lg shadow p-12 text-center">
          <div className="text-gray-500">
            {loading ? 'Loading analytics...' : 'Enter a symbol to begin analysis'}
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Overview Card */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-2xl font-bold text-gray-900">{analytics.symbol}</h2>
                <p className="text-gray-600">Spot: ${analytics.spot_price.toFixed(2)}</p>
              </div>
              <div className="text-right">
                <div className="text-sm text-gray-600">IV Range</div>
                <div className="text-2xl font-bold text-blue-600">
                  {formatPercent(analytics.iv_surface.min_iv)} - {formatPercent(analytics.iv_surface.max_iv)}
                </div>
              </div>
            </div>

            {/* IV Surface Summary */}
            <div className="grid grid-cols-4 gap-4">
              <div className="bg-blue-50 rounded p-3">
                <div className="text-sm text-gray-600">ATM IV</div>
                <div className="text-xl font-bold text-blue-600">
                  {formatPercent(analytics.iv_surface.atm_iv)}
                </div>
              </div>
              <div className="bg-purple-50 rounded p-3">
                <div className="text-sm text-gray-600">IV Range</div>
                <div className="text-xl font-bold text-purple-600">
                  {formatPercent(analytics.iv_surface.iv_range)}
                </div>
              </div>
              <div className="bg-green-50 rounded p-3">
                <div className="text-sm text-gray-600">Min IV</div>
                <div className="text-xl font-bold text-green-600">
                  {formatPercent(analytics.iv_surface.min_iv)}
                </div>
              </div>
              <div className="bg-red-50 rounded p-3">
                <div className="text-sm text-gray-600">Max IV</div>
                <div className="text-xl font-bold text-red-600">
                  {formatPercent(analytics.iv_surface.max_iv)}
                </div>
              </div>
            </div>

            {/* Investor Insight */}
            <div className="mt-4 p-4 bg-yellow-50 border-l-4 border-yellow-400">
              <div className="flex items-start gap-2">
                <span className="text-yellow-600 font-bold">💡</span>
                <div>
                  <div className="font-semibold text-yellow-900">Volatility Context:</div>
                  <div className="text-sm text-yellow-800 mt-1">
                    {analytics.iv_surface.iv_range > 0.2 ? (
                      <>Wide IV range ({formatPercent(analytics.iv_surface.iv_range)}) suggests significant mispricing opportunities.
                      Look for overpriced far OTM options to sell, or underpriced ATM options to buy.</>
                    ) : (
                      <>Narrow IV range ({formatPercent(analytics.iv_surface.iv_range)}) suggests efficient pricing.
                      Focus on directional plays rather than volatility arbitrage.</>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Volatility Skew Analysis */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Volatility Skew Analysis</h2>

            {analytics.skew_metrics.map((skew, idx) => {
              const interpretation = getSkewInterpretation(skew.put_skew, skew.call_skew);
              const rrInterpretation = getRiskReversalInterpretation(skew.risk_reversal_25delta);

              return (
                <div key={idx} className="mb-6 last:mb-0">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <h3 className="font-semibold text-gray-900">
                        {new Date(skew.expiration).toLocaleDateString()}
                        <span className="text-sm text-gray-600 ml-2">
                          ({skew.days_to_expiry} days)
                        </span>
                      </h3>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-gray-600">ATM IV</div>
                      <div className="font-bold text-gray-900">{formatPercent(skew.atm_iv)}</div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                    <div className="bg-gray-50 rounded p-3">
                      <div className="text-xs text-gray-600">Put Skew</div>
                      <div className={`text-lg font-bold ${skew.put_skew > 0.05 ? 'text-red-600' : 'text-gray-900'}`}>
                        {formatPercent(skew.put_skew)}
                      </div>
                    </div>
                    <div className="bg-gray-50 rounded p-3">
                      <div className="text-xs text-gray-600">Call Skew</div>
                      <div className={`text-lg font-bold ${skew.call_skew > 0.05 ? 'text-green-600' : 'text-gray-900'}`}>
                        {formatPercent(skew.call_skew)}
                      </div>
                    </div>
                    <div className="bg-gray-50 rounded p-3">
                      <div className="text-xs text-gray-600">Risk Reversal</div>
                      <div className={`text-lg font-bold ${rrInterpretation.color}`}>
                        {formatPercent(skew.risk_reversal_25delta)}
                      </div>
                    </div>
                    <div className="bg-gray-50 rounded p-3">
                      <div className="text-xs text-gray-600">Butterfly</div>
                      <div className="text-lg font-bold text-gray-900">
                        {formatPercent(skew.butterfly_25delta)}
                      </div>
                    </div>
                  </div>

                  {/* Trading Insights */}
                  <div className="space-y-2">
                    <div className={`p-3 bg-gray-50 border-l-4 ${interpretation.color.replace('text-', 'border-')}`}>
                      <div className="flex items-start gap-2">
                        <span>📊</span>
                        <div className="flex-1">
                          <div className={`font-semibold ${interpretation.color}`}>{interpretation.text}</div>
                          <div className="text-sm text-gray-700 mt-1">
                            <strong>Trade Idea:</strong> {interpretation.recommendation}
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className={`p-3 bg-gray-50 border-l-4 ${rrInterpretation.color.replace('text-', 'border-')}`}>
                      <div className="flex items-start gap-2">
                        <span>🎯</span>
                        <div className="flex-1">
                          <div className={`font-semibold ${rrInterpretation.color}`}>{rrInterpretation.text}</div>
                          <div className="text-sm text-gray-700 mt-1">
                            <strong>Strategy:</strong> {rrInterpretation.recommendation}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Term Structure Analysis */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Term Structure Analysis</h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {Object.entries(analytics.term_structures).map(([type, ts]) => {
                if (!ts) return null;

                const interpretation = getTermStructureInterpretation(ts);
                const typeLabel = type === 'atm' ? 'ATM' :
                                 type === 'otm_put' ? '90% Put' : '110% Call';

                return (
                  <div key={type} className="border border-gray-200 rounded-lg p-4">
                    <h3 className="font-semibold text-gray-900 mb-3">{typeLabel}</h3>

                    <div className="space-y-2 mb-3">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Front Month:</span>
                        <span className="font-semibold">{formatPercent(ts.front_month_iv)}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Back Month:</span>
                        <span className="font-semibold">{formatPercent(ts.back_month_iv)}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Slope:</span>
                        <span className={`font-semibold ${ts.is_inverted ? 'text-red-600' : 'text-green-600'}`}>
                          {ts.term_structure_slope.toFixed(4)}
                        </span>
                      </div>
                    </div>

                    <div className={`p-2 rounded text-xs ${interpretation.color.replace('text-', 'bg-').replace('600', '50')} border-l-2 ${interpretation.color.replace('text-', 'border-')}`}>
                      <div className={`font-semibold ${interpretation.color} mb-1`}>{interpretation.text}</div>
                      <div className="text-gray-700">{interpretation.recommendation}</div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Key Takeaways for Investors */}
          <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg shadow p-6 text-white">
            <h2 className="text-xl font-bold mb-4">🎯 Key Trading Opportunities</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h3 className="font-semibold mb-2">Volatility Plays:</h3>
                <ul className="text-sm space-y-1 opacity-90">
                  {analytics.iv_surface.iv_range > 0.2 && (
                    <li>• Wide IV range - Sell overpriced far OTM options</li>
                  )}
                  {analytics.skew_metrics[0]?.put_skew > 0.05 && (
                    <li>• Elevated put skew - Sell put spreads for premium</li>
                  )}
                  {analytics.term_structures.atm?.is_inverted && (
                    <li>• Backwardation detected - Calendar spreads favorable</li>
                  )}
                </ul>
              </div>
              <div>
                <h3 className="font-semibold mb-2">Risk Considerations:</h3>
                <ul className="text-sm space-y-1 opacity-90">
                  <li>• Always size positions appropriately (max 2-5% per trade)</li>
                  <li>• Use defined-risk spreads to limit downside</li>
                  <li>• Monitor for earnings and catalyst events</li>
                  <li>• Combine with technical analysis for entry/exit</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default OptionsAnalyticsPage;
