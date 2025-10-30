import React, { useState } from 'react';
import {
  analyzeSentiment,
  compareSentiment,
  getTrendingSentiment,
  SentimentAnalysisResponse,
} from '../services/sentimentApi';
import toast from 'react-hot-toast';

const SentimentAnalysisPage: React.FC = () => {
  const [symbol, setSymbol] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SentimentAnalysisResponse | null>(null);
  const [compareSymbols, setCompareSymbols] = useState('');
  const [comparisonResult, setComparisonResult] = useState<any>(null);
  const [trending, setTrending] = useState<any>(null);

  const handleAnalyze = async () => {
    if (!symbol.trim()) {
      toast.error('Please enter a symbol');
      return;
    }

    setLoading(true);
    try {
      const response = await analyzeSentiment({
        symbol: symbol.toUpperCase(),
        sources: ['twitter', 'reddit', 'news', 'stocktwits'],
        lookback_hours: 24,
      });
      setResult(response);
      toast.success(`Sentiment analyzed for ${response.symbol}`);
    } catch (error: any) {
      toast.error(error.message || 'Failed to analyze sentiment');
    } finally {
      setLoading(false);
    }
  };

  const handleCompare = async () => {
    const symbols = compareSymbols
      .split(',')
      .map((s) => s.trim().toUpperCase())
      .filter((s) => s);

    if (symbols.length < 2) {
      toast.error('Please enter at least 2 symbols (comma-separated)');
      return;
    }

    setLoading(true);
    try {
      const response = await compareSentiment(symbols);
      setComparisonResult(response);
      toast.success(`Compared sentiment for ${symbols.length} symbols`);
    } catch (error: any) {
      toast.error(error.message || 'Failed to compare sentiment');
    } finally {
      setLoading(false);
    }
  };

  const handleLoadTrending = async () => {
    setLoading(true);
    try {
      const response = await getTrendingSentiment('1h', 20);
      setTrending(response);
      toast.success('Loaded trending sentiment');
    } catch (error: any) {
      toast.error(error.message || 'Failed to load trending sentiment');
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (score: number) => {
    if (score >= 70) return 'text-green-600';
    if (score >= 55) return 'text-blue-600';
    if (score >= 45) return 'text-gray-600';
    if (score >= 30) return 'text-orange-600';
    return 'text-red-600';
  };

  const getSentimentBg = (score: number) => {
    if (score >= 70) return 'bg-green-100';
    if (score >= 55) return 'bg-blue-100';
    if (score >= 45) return 'bg-gray-100';
    if (score >= 30) return 'bg-orange-100';
    return 'bg-red-100';
  };

  const getControversyColor = (score: number) => {
    if (score >= 70) return 'text-red-600';
    if (score >= 50) return 'text-orange-600';
    if (score >= 30) return 'text-yellow-600';
    return 'text-green-600';
  };

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">
          Sentiment Analysis
        </h1>
        <p className="text-gray-600 mt-2">
          Deep sentiment analysis with influencer weighting and controversy detection
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls */}
        <div className="space-y-4">
          <div className="bg-white rounded-lg shadow p-4">
            <h2 className="font-semibold mb-4">Analyze Symbol</h2>
            <div className="space-y-3">
              <input
                type="text"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                onKeyPress={(e) => e.key === 'Enter' && handleAnalyze()}
                placeholder="Enter symbol..."
                className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button
                onClick={handleAnalyze}
                disabled={loading}
                className="w-full px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
              >
                Analyze Sentiment
              </button>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-4">
            <h2 className="font-semibold mb-4">Compare Symbols</h2>
            <div className="space-y-3">
              <input
                type="text"
                value={compareSymbols}
                onChange={(e) => setCompareSymbols(e.target.value.toUpperCase())}
                placeholder="AAPL, MSFT, GOOGL..."
                className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button
                onClick={handleCompare}
                disabled={loading}
                className="w-full px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
              >
                Compare
              </button>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-4">
            <h2 className="font-semibold mb-4">Trending</h2>
            <button
              onClick={handleLoadTrending}
              disabled={loading}
              className="w-full px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50"
            >
              Load Trending Stocks
            </button>
          </div>

          <div className="bg-blue-50 rounded-lg p-4">
            <h3 className="font-semibold text-blue-900 mb-2">
              📊 Metrics
            </h3>
            <div className="text-sm text-blue-800 space-y-1">
              <div><strong>Score:</strong> 0-100 (50 = neutral)</div>
              <div><strong>Controversy:</strong> High = volatility</div>
              <div><strong>Velocity:</strong> Rate of change</div>
              <div><strong>Influencer:</strong> Weighted by followers</div>
            </div>
          </div>
        </div>

        {/* Results */}
        <div className="lg:col-span-2 space-y-4">
          {/* Single Symbol Analysis */}
          {result && (
            <div className="bg-white rounded-lg shadow p-6">
              <div className="mb-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-2xl font-bold font-mono">{result.symbol}</h2>
                  <span className="text-sm text-gray-500">
                    {new Date(result.timestamp).toLocaleString()}
                  </span>
                </div>
              </div>

              {/* Overall Sentiment */}
              <div className="mb-6">
                <h3 className="font-semibold mb-3">Overall Sentiment</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className={`rounded-lg p-4 ${getSentimentBg(result.sentiment.score)}`}>
                    <div className="text-sm text-gray-600 mb-1">Score</div>
                    <div className={`text-3xl font-bold ${getSentimentColor(result.sentiment.score)}`}>
                      {result.sentiment.score.toFixed(1)}
                    </div>
                    <div className="text-sm mt-1 uppercase font-medium">
                      {result.sentiment.bias}
                    </div>
                  </div>
                  <div className="bg-gray-100 rounded-lg p-4">
                    <div className="text-sm text-gray-600 mb-1">Confidence</div>
                    <div className="text-3xl font-bold text-gray-900">
                      {Math.round(result.sentiment.confidence * 100)}%
                    </div>
                    <div className="text-sm mt-1">
                      {result.sentiment.mention_volume.toLocaleString()} mentions
                    </div>
                  </div>
                </div>
              </div>

              {/* By Source */}
              <div className="mb-6">
                <h3 className="font-semibold mb-3">By Source</h3>
                <div className="space-y-2">
                  {Object.entries(result.by_source).map(([source, data]: [string, any]) => (
                    <div key={source} className="border border-gray-200 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium capitalize">{source}</span>
                        <span className={`font-bold ${getSentimentColor(data.score)}`}>
                          {data.score?.toFixed(1)}
                        </span>
                      </div>
                      <div className="text-sm text-gray-600">
                        {data.mention_count?.toLocaleString()} mentions
                        {data.engagement && ` • ${data.engagement.toLocaleString()} engagement`}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Influencer Sentiment */}
              {result.influencer_sentiment && (
                <div className="mb-6">
                  <h3 className="font-semibold mb-3">Influencer Sentiment</h3>
                  <div className="bg-purple-50 rounded-lg p-4">
                    <div className="grid grid-cols-3 gap-4 mb-3">
                      <div>
                        <div className="text-sm text-gray-600">Score</div>
                        <div className={`text-2xl font-bold ${getSentimentColor(result.influencer_sentiment.score)}`}>
                          {result.influencer_sentiment.score.toFixed(1)}
                        </div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-600">Tier 1</div>
                        <div className="text-2xl font-bold text-purple-600">
                          {result.influencer_sentiment.tier_1_count}
                        </div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-600">Tier 2</div>
                        <div className="text-2xl font-bold text-purple-600">
                          {result.influencer_sentiment.tier_2_count}
                        </div>
                      </div>
                    </div>
                    <div className="text-sm">
                      <strong>vs Retail:</strong> {result.influencer_sentiment.bias_vs_retail}
                    </div>
                  </div>
                </div>
              )}

              {/* Metrics */}
              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="border border-gray-200 rounded-lg p-3">
                  <div className="text-sm text-gray-600 mb-1">Controversy</div>
                  <div className={`text-xl font-bold ${getControversyColor(result.controversy_score)}`}>
                    {result.controversy_score.toFixed(1)}
                  </div>
                </div>
                <div className="border border-gray-200 rounded-lg p-3">
                  <div className="text-sm text-gray-600 mb-1">Velocity</div>
                  <div className="text-xl font-bold text-gray-900">
                    {result.sentiment_velocity > 0 ? '+' : ''}
                    {result.sentiment_velocity.toFixed(1)}
                  </div>
                </div>
                <div className="border border-gray-200 rounded-lg p-3">
                  <div className="text-sm text-gray-600 mb-1">Echo Chamber</div>
                  <div className={`text-xl font-bold ${result.echo_chamber_detected ? 'text-red-600' : 'text-green-600'}`}>
                    {result.echo_chamber_detected ? 'Yes' : 'No'}
                  </div>
                </div>
              </div>

              {/* Trading Implication */}
              <div className="bg-blue-50 rounded-lg p-4">
                <h3 className="font-semibold text-blue-900 mb-2">
                  Trading Implication
                </h3>
                <p className="text-blue-800">{result.trading_implication}</p>
              </div>
            </div>
          )}

          {/* Comparison Results */}
          {comparisonResult && (
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-bold mb-4">Sentiment Comparison</h2>
              <div className="space-y-3">
                {Object.entries(comparisonResult.comparisons).map(([sym, data]: [string, any]) => (
                  <div key={sym} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-mono font-bold text-lg">{sym}</span>
                      <span className={`text-xl font-bold ${getSentimentColor(data.sentiment?.score || 50)}`}>
                        {data.sentiment?.score?.toFixed(1) || 'N/A'}
                      </span>
                    </div>
                    {data.trading_implication && (
                      <div className="text-sm text-gray-600">{data.trading_implication}</div>
                    )}
                  </div>
                ))}
              </div>
              <div className="mt-4 pt-4 border-t border-gray-200">
                <h3 className="font-semibold mb-2">Ranking</h3>
                <div className="space-y-1">
                  {Object.entries(comparisonResult.relative_strength).map(([sym, rank]: [string, any]) => (
                    <div key={sym} className="flex items-center justify-between text-sm">
                      <span className="font-mono">{sym}</span>
                      <span className="text-gray-600">{rank}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Trending */}
          {trending && (
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-bold mb-4">Trending Sentiment</h2>
              <div className="text-sm text-gray-600 mb-4">
                Timeframe: {trending.timeframe} • {trending.count} stocks
              </div>
              <div className="space-y-2">
                {trending.trending?.map((item: any, idx: number) => (
                  <div key={idx} className="border border-gray-200 rounded p-3">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-mono font-bold">{item.symbol}</span>
                      <span className={`font-bold ${getSentimentColor(item.sentiment_score)}`}>
                        {item.sentiment_score.toFixed(1)}
                      </span>
                    </div>
                    <div className="text-sm text-gray-600 flex items-center gap-4">
                      <span>{item.mention_volume.toLocaleString()} mentions</span>
                      <span className="text-green-600">
                        +{item.mention_velocity.toFixed(1)}% velocity
                      </span>
                    </div>
                    {item.top_keywords && (
                      <div className="mt-2 flex gap-2">
                        {item.top_keywords.map((keyword: string, kidx: number) => (
                          <span key={kidx} className="text-xs bg-gray-100 px-2 py-1 rounded">
                            {keyword}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {!result && !comparisonResult && !trending && (
            <div className="bg-white rounded-lg shadow p-12 text-center">
              <svg
                className="w-16 h-16 mx-auto text-gray-400 mb-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z"
                />
              </svg>
              <p className="text-gray-600">
                Analyze a symbol to see sentiment results
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SentimentAnalysisPage;
