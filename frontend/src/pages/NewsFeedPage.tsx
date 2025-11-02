import React, { useState, useEffect, useCallback } from 'react';
import { getNews, searchNews, getSymbolNews, NewsArticle, NewsStreamConnection } from '../services/newsApi';
import toast from 'react-hot-toast';

const NewsFeedPage: React.FC = () => {
  const [articles, setArticles] = useState<NewsArticle[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [symbolFilter, setSymbolFilter] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('');
  const [hoursBack, setHoursBack] = useState(24);
  const [liveMode, setLiveMode] = useState(false);
  const [streamConnection, setStreamConnection] = useState<NewsStreamConnection | null>(null);
  const [newsCount, setNewsCount] = useState(0);

  const categories = [
    { value: '', label: 'All Categories' },
    { value: 'earnings', label: 'Earnings' },
    { value: 'merger_acquisition', label: 'M&A' },
    { value: 'ipo', label: 'IPO' },
    { value: 'analyst_rating', label: 'Analyst Ratings' },
    { value: 'fda', label: 'FDA' },
    { value: 'guidance', label: 'Guidance' },
    { value: 'dividend', label: 'Dividends' },
    { value: 'buyback', label: 'Buybacks' },
  ];

  const loadNews = useCallback(async () => {
    setLoading(true);
    try {
      let result;

      if (searchQuery) {
        // Search mode
        const symbols = symbolFilter ? symbolFilter.split(',').map(s => s.trim()) : undefined;
        result = await searchNews(searchQuery, symbols, 100);
        setArticles(result.articles);
        setNewsCount(result.count);
      } else if (symbolFilter && symbolFilter.split(',').length === 1) {
        // Single symbol mode
        result = await getSymbolNews(symbolFilter.trim(), 100, hoursBack);
        setArticles(result.articles);
        setNewsCount(result.count);
      } else {
        // General news feed
        const symbols = symbolFilter ? symbolFilter.split(',').map(s => s.trim()) : undefined;
        const categories = categoryFilter ? [categoryFilter] : undefined;
        result = await getNews(symbols, categories, 100, hoursBack);
        setArticles(result.articles);
        setNewsCount(result.count);
      }

      toast.success(`Loaded ${result.articles?.length || result.count} articles`);
    } catch (error: any) {
      toast.error(error.message || 'Failed to load news');
      console.error('News load error:', error);
    } finally {
      setLoading(false);
    }
  }, [searchQuery, symbolFilter, categoryFilter, hoursBack]);

  useEffect(() => {
    loadNews();
  }, [loadNews]);

  const toggleLiveMode = () => {
    if (!liveMode) {
      // Start live mode
      const connection = new NewsStreamConnection(
        (newArticles) => {
          setArticles(prev => {
            // Add new articles to the top
            const combined = [...newArticles, ...prev];
            // Remove duplicates by ID
            const unique = combined.filter((article, index, self) =>
              index === self.findIndex(a => a.id === article.id)
            );
            // Limit to 200 articles
            return unique.slice(0, 200);
          });
          toast.success(`${newArticles.length} new articles`);
        },
        (error) => {
          console.error('WebSocket error:', error);
          toast.error('News stream connection error');
        },
        () => {
          toast.success('Connected to live news stream');
        }
      );

      const symbols = symbolFilter ? symbolFilter.split(',').map(s => s.trim()) : undefined;
      const categories = categoryFilter ? [categoryFilter] : undefined;

      connection.connect({
        symbols,
        categories,
        interval_seconds: 60,
      });

      setStreamConnection(connection);
      setLiveMode(true);
    } else {
      // Stop live mode
      if (streamConnection) {
        streamConnection.disconnect();
        setStreamConnection(null);
      }
      setLiveMode(false);
      toast.success('Live mode disabled');
    }
  };

  useEffect(() => {
    // Cleanup on unmount
    return () => {
      if (streamConnection) {
        streamConnection.disconnect();
      }
    };
  }, [streamConnection]);

  const formatTimeAgo = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);

    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  };

  const getCategoryColor = (category: string) => {
    const colors: Record<string, string> = {
      earnings: 'bg-green-100 text-green-800',
      merger_acquisition: 'bg-blue-100 text-blue-800',
      ipo: 'bg-purple-100 text-purple-800',
      analyst_rating: 'bg-yellow-100 text-yellow-800',
      fda: 'bg-red-100 text-red-800',
      guidance: 'bg-indigo-100 text-indigo-800',
      dividend: 'bg-teal-100 text-teal-800',
      buyback: 'bg-orange-100 text-orange-800',
    };
    return colors[category] || 'bg-gray-100 text-gray-800';
  };

  const getSentimentColor = (sentiment: string | null) => {
    if (!sentiment) return 'text-gray-500';
    const colors: Record<string, string> = {
      positive: 'text-green-600',
      bullish: 'text-green-700',
      negative: 'text-red-600',
      bearish: 'text-red-700',
      neutral: 'text-gray-600',
    };
    return colors[sentiment] || 'text-gray-500';
  };

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">
              📰 Financial News
            </h1>
            <p className="text-gray-600 mt-2">
              Bloomberg NEWS equivalent - real-time financial news aggregation
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className={`text-sm ${liveMode ? 'text-green-600 font-semibold' : 'text-gray-500'}`}>
              {liveMode ? '● LIVE' : '○ Static'}
            </span>
            <button
              onClick={toggleLiveMode}
              className={`px-4 py-2 rounded-lg font-medium ${
                liveMode
                  ? 'bg-green-600 text-white hover:bg-green-700'
                  : 'bg-blue-600 text-white hover:bg-blue-700'
              }`}
            >
              {liveMode ? 'Stop Live' : 'Go Live'}
            </button>
            <button
              onClick={loadNews}
              disabled={loading}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50"
            >
              {loading ? 'Loading...' : 'Refresh'}
            </button>
          </div>
        </div>

        {/* Filters */}
        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          {/* Search */}
          <input
            type="text"
            placeholder="Search news..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && loadNews()}
            className="border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />

          {/* Symbol Filter */}
          <input
            type="text"
            placeholder="Symbols (e.g., AAPL,MSFT)"
            value={symbolFilter}
            onChange={(e) => setSymbolFilter(e.target.value.toUpperCase())}
            className="border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />

          {/* Category Filter */}
          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
            className="border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {categories.map(cat => (
              <option key={cat.value} value={cat.value}>{cat.label}</option>
            ))}
          </select>

          {/* Time Range */}
          <select
            value={hoursBack}
            onChange={(e) => setHoursBack(parseInt(e.target.value))}
            className="border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value={1}>Last Hour</option>
            <option value={6}>Last 6 Hours</option>
            <option value={24}>Last 24 Hours</option>
            <option value={48}>Last 48 Hours</option>
            <option value={168}>Last Week</option>
          </select>
        </div>

        {/* Stats */}
        <div className="mt-3 text-sm text-gray-600">
          Showing {articles.length} articles
          {symbolFilter && ` for ${symbolFilter}`}
          {categoryFilter && ` in ${categories.find(c => c.value === categoryFilter)?.label}`}
        </div>
      </div>

      {/* News Feed */}
      {loading && articles.length === 0 ? (
        <div className="bg-white rounded-lg shadow p-12 text-center">
          <div className="text-gray-500">Loading news...</div>
        </div>
      ) : articles.length === 0 ? (
        <div className="bg-white rounded-lg shadow p-12 text-center">
          <div className="text-gray-500">No news articles found</div>
        </div>
      ) : (
        <div className="space-y-4">
          {articles.map((article) => (
            <div
              key={article.id}
              className="bg-white rounded-lg shadow hover:shadow-lg transition-shadow p-5"
            >
              <div className="flex items-start gap-4">
                {/* Image */}
                {article.image_url && (
                  <div className="flex-shrink-0 w-32 h-24 overflow-hidden rounded">
                    <img
                      src={article.image_url}
                      alt={article.title}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = 'none';
                      }}
                    />
                  </div>
                )}

                {/* Content */}
                <div className="flex-1 min-w-0">
                  {/* Title */}
                  <a
                    href={article.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-lg font-semibold text-gray-900 hover:text-blue-600 block"
                  >
                    {article.title}
                  </a>

                  {/* Summary */}
                  {article.summary && (
                    <p className="mt-2 text-gray-600 text-sm line-clamp-2">
                      {article.summary}
                    </p>
                  )}

                  {/* Metadata */}
                  <div className="mt-3 flex flex-wrap items-center gap-3 text-sm">
                    {/* Source */}
                    <span className="font-medium text-gray-700">
                      {article.source}
                    </span>

                    {/* Author */}
                    {article.author && (
                      <span className="text-gray-500">
                        by {article.author}
                      </span>
                    )}

                    {/* Time */}
                    <span className="text-gray-500">
                      {formatTimeAgo(article.published_at)}
                    </span>

                    {/* Provider */}
                    <span className="text-xs text-gray-400 uppercase">
                      {article.provider}
                    </span>

                    {/* Sentiment */}
                    {article.sentiment && (
                      <span className={`text-xs font-semibold ${getSentimentColor(article.sentiment)}`}>
                        {article.sentiment.toUpperCase()}
                      </span>
                    )}
                  </div>

                  {/* Tags */}
                  <div className="mt-3 flex flex-wrap gap-2">
                    {/* Symbols */}
                    {article.symbols.map((symbol) => (
                      <span
                        key={symbol}
                        className="px-2 py-1 bg-blue-600 text-white text-xs font-semibold rounded"
                      >
                        ${symbol}
                      </span>
                    ))}

                    {/* Categories */}
                    {article.categories.map((category) => (
                      <span
                        key={category}
                        className={`px-2 py-1 text-xs font-semibold rounded ${getCategoryColor(category)}`}
                      >
                        {category.replace('_', ' ').toUpperCase()}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default NewsFeedPage;
