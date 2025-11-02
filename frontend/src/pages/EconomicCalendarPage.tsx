import React, { useState, useEffect, useCallback } from 'react';
import { getCompleteCalendar, getEarningsHistory, CalendarDay, EarningsHistoryResponse } from '../services/calendarApi';
import toast from 'react-hot-toast';

const EconomicCalendarPage: React.FC = () => {
  const [calendarDays, setCalendarDays] = useState<CalendarDay[]>([]);
  const [loading, setLoading] = useState(false);
  const [daysAhead, setDaysAhead] = useState(14); // Default 2 weeks
  const [symbolFilter, setSymbolFilter] = useState('');
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [earningsHistory, setEarningsHistory] = useState<EarningsHistoryResponse | null>(null);

  const loadCalendar = useCallback(async () => {
    setLoading(true);
    try {
      const result = await getCompleteCalendar(
        undefined,
        undefined,
        symbolFilter || undefined,
        daysAhead
      );

      setCalendarDays(result.days);
      toast.success(`Loaded calendar for ${result.total_days} days`);
    } catch (error: any) {
      toast.error(error.message || 'Failed to load calendar');
      console.error('Calendar load error:', error);
    } finally {
      setLoading(false);
    }
  }, [daysAhead, symbolFilter]);

  useEffect(() => {
    loadCalendar();
  }, [loadCalendar]);

  const loadEarningsHistory = async (symbol: string) => {
    try {
      const history = await getEarningsHistory(symbol, 8);
      setEarningsHistory(history);
      setSelectedSymbol(symbol);
    } catch (error: any) {
      toast.error(`Failed to load earnings history for ${symbol}`);
      console.error('Earnings history error:', error);
    }
  };

  const getImportanceColor = (importance: string) => {
    const colors: Record<string, string> = {
      high: 'bg-red-100 text-red-800 border-red-300',
      medium: 'bg-yellow-100 text-yellow-800 border-yellow-300',
      low: 'bg-gray-100 text-gray-800 border-gray-300',
    };
    return colors[importance] || 'bg-gray-100 text-gray-800 border-gray-300';
  };

  const getEventTypeIcon = (eventType: string) => {
    const icons: Record<string, string> = {
      fed_meeting: '🏦',
      cpi: '📈',
      gdp: '💹',
      jobs: '👔',
      ppi: '🏭',
      pce: '🛒',
      retail_sales: '🛍️',
    };
    return icons[eventType] || '📊';
  };

  const getEarningsTimeLabel = (time: string) => {
    if (time === 'bmo') return 'Before Market';
    if (time === 'amc') return 'After Market';
    return 'During Market';
  };

  const formatCurrency = (value: number | null) => {
    if (value === null) return 'N/A';
    if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
    return `$${value.toFixed(2)}`;
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const today = new Date();
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    if (date.toDateString() === today.toDateString()) {
      return 'Today';
    } else if (date.toDateString() === tomorrow.toDateString()) {
      return 'Tomorrow';
    } else {
      return date.toLocaleDateString('en-US', {
        weekday: 'short',
        month: 'short',
        day: 'numeric'
      });
    }
  };

  const getTradingRecommendation = (day: CalendarDay) => {
    const recommendations: string[] = [];

    if (day.high_importance_count > 0) {
      recommendations.push('High volatility expected - consider reducing position sizes');
    }

    if (day.major_earnings_count >= 3) {
      recommendations.push('Multiple major earnings - good day for selling premium');
    }

    const fedMeeting = day.economic_events.find(e => e.event_type === 'fed_meeting');
    if (fedMeeting) {
      recommendations.push('Fed meeting - expect market-wide volatility spike');
    }

    const cpi = day.economic_events.find(e => e.event_type === 'cpi');
    if (cpi) {
      recommendations.push('CPI release - straddle/strangle opportunities on SPY/QQQ');
    }

    if (recommendations.length === 0) {
      return 'Normal volatility day - focus on directional plays';
    }

    return recommendations.join(' • ');
  };

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">
          📅 Economic Calendar
        </h1>
        <p className="text-gray-600 mt-2">
          Bloomberg EVTS equivalent - Earnings and economic events with volatility implications
        </p>
      </div>

      {/* Filters */}
      <div className="bg-white rounded-lg shadow p-4 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Days Ahead */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Time Range
            </label>
            <select
              value={daysAhead}
              onChange={(e) => setDaysAhead(parseInt(e.target.value))}
              className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value={7}>Next 7 Days</option>
              <option value={14}>Next 14 Days</option>
              <option value={30}>Next 30 Days</option>
              <option value={60}>Next 60 Days</option>
              <option value={90}>Next 90 Days</option>
            </select>
          </div>

          {/* Symbol Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Filter Symbols
            </label>
            <input
              type="text"
              placeholder="e.g., AAPL,MSFT,GOOGL"
              value={symbolFilter}
              onChange={(e) => setSymbolFilter(e.target.value.toUpperCase())}
              className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Refresh Button */}
          <div className="flex items-end">
            <button
              onClick={loadCalendar}
              disabled={loading}
              className="w-full px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
            >
              {loading ? 'Loading...' : 'Refresh Calendar'}
            </button>
          </div>
        </div>
      </div>

      {/* Calendar View */}
      {loading && calendarDays.length === 0 ? (
        <div className="bg-white rounded-lg shadow p-12 text-center">
          <div className="text-gray-500">Loading calendar...</div>
        </div>
      ) : calendarDays.length === 0 ? (
        <div className="bg-white rounded-lg shadow p-12 text-center">
          <div className="text-gray-500">No events found</div>
        </div>
      ) : (
        <div className="space-y-4">
          {calendarDays.map((day) => {
            // Skip days with no events
            if (day.total_events === 0) return null;

            return (
              <div
                key={day.date}
                className={`bg-white rounded-lg shadow hover:shadow-lg transition-shadow ${
                  day.high_importance_count > 0 || day.major_earnings_count > 0
                    ? 'border-l-4 border-red-500'
                    : ''
                }`}
              >
                {/* Day Header */}
                <div className="bg-gray-50 border-b border-gray-200 px-6 py-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h2 className="text-xl font-bold text-gray-900">
                        {formatDate(day.date)}
                      </h2>
                      <p className="text-sm text-gray-600 mt-1">
                        {day.date} • {day.total_events} events
                        {day.high_importance_count > 0 && (
                          <span className="ml-2 px-2 py-1 bg-red-100 text-red-800 text-xs font-semibold rounded">
                            {day.high_importance_count} High Impact
                          </span>
                        )}
                        {day.major_earnings_count > 0 && (
                          <span className="ml-2 px-2 py-1 bg-blue-100 text-blue-800 text-xs font-semibold rounded">
                            {day.major_earnings_count} Major Earnings
                          </span>
                        )}
                      </p>
                    </div>
                  </div>

                  {/* Trading Recommendation */}
                  <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded">
                    <div className="flex items-start gap-2">
                      <span className="text-blue-600 font-semibold text-sm">💡 Strategy:</span>
                      <span className="text-sm text-gray-700">{getTradingRecommendation(day)}</span>
                    </div>
                  </div>
                </div>

                {/* Events */}
                <div className="p-6">
                  {/* Economic Events */}
                  {day.economic_events.length > 0 && (
                    <div className="mb-6">
                      <h3 className="text-lg font-semibold text-gray-900 mb-3">
                        📊 Economic Events
                      </h3>
                      <div className="space-y-2">
                        {day.economic_events.map((event, idx) => (
                          <div
                            key={idx}
                            className={`p-3 border-2 rounded ${getImportanceColor(event.importance)}`}
                          >
                            <div className="flex items-start gap-3">
                              <span className="text-2xl">{getEventTypeIcon(event.event_type)}</span>
                              <div className="flex-1">
                                <div className="flex items-center justify-between">
                                  <div className="font-semibold text-gray-900">{event.name}</div>
                                  <div className="text-sm text-gray-600">{event.time}</div>
                                </div>
                                <div className="mt-1 flex flex-wrap gap-3 text-sm">
                                  {event.estimate && (
                                    <span className="text-gray-600">
                                      Est: <span className="font-medium">{event.estimate}</span>
                                    </span>
                                  )}
                                  {event.actual && (
                                    <span className="text-gray-600">
                                      Actual: <span className="font-medium">{event.actual}</span>
                                    </span>
                                  )}
                                  {event.volatility_expected && (
                                    <span className={`px-2 py-0.5 rounded text-xs font-semibold ${
                                      event.volatility_expected === 'high'
                                        ? 'bg-red-100 text-red-800'
                                        : event.volatility_expected === 'medium'
                                        ? 'bg-yellow-100 text-yellow-800'
                                        : 'bg-gray-100 text-gray-800'
                                    }`}>
                                      {event.volatility_expected.toUpperCase()} VOL
                                    </span>
                                  )}
                                </div>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Earnings Events */}
                  {day.earnings_events.length > 0 && (
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900 mb-3">
                        💼 Earnings Announcements
                      </h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                        {day.earnings_events.map((event, idx) => (
                          <div
                            key={idx}
                            className="p-4 border-2 border-blue-200 bg-blue-50 rounded hover:bg-blue-100 transition-colors cursor-pointer"
                            onClick={() => loadEarningsHistory(event.symbol)}
                          >
                            <div className="flex items-start justify-between mb-2">
                              <div>
                                <div className="font-bold text-lg text-blue-900">
                                  {event.symbol}
                                </div>
                                <div className="text-xs text-gray-600 mt-1">
                                  {event.company_name}
                                </div>
                              </div>
                              <span className="text-xs px-2 py-1 bg-blue-200 text-blue-800 rounded font-semibold">
                                {getEarningsTimeLabel(event.time)}
                              </span>
                            </div>

                            <div className="text-sm space-y-1">
                              <div className="flex justify-between">
                                <span className="text-gray-600">Quarter:</span>
                                <span className="font-medium">{event.fiscal_quarter}</span>
                              </div>

                              {event.eps_estimate && (
                                <div className="flex justify-between">
                                  <span className="text-gray-600">EPS Est:</span>
                                  <span className="font-medium">${event.eps_estimate.toFixed(2)}</span>
                                </div>
                              )}

                              {event.eps_actual && (
                                <div className="flex justify-between">
                                  <span className="text-gray-600">EPS Actual:</span>
                                  <span className={`font-bold ${
                                    event.eps_surprise_pct && event.eps_surprise_pct > 0
                                      ? 'text-green-700'
                                      : event.eps_surprise_pct && event.eps_surprise_pct < 0
                                      ? 'text-red-700'
                                      : 'text-gray-900'
                                  }`}>
                                    ${event.eps_actual.toFixed(2)}
                                    {event.eps_surprise_pct && (
                                      <span className="ml-1 text-xs">
                                        ({event.eps_surprise_pct > 0 ? '+' : ''}{event.eps_surprise_pct.toFixed(1)}%)
                                      </span>
                                    )}
                                  </span>
                                </div>
                              )}

                              {event.implied_move && (
                                <div className="mt-2 pt-2 border-t border-blue-300">
                                  <div className="flex justify-between">
                                    <span className="text-gray-600">Implied Move:</span>
                                    <span className="font-bold text-purple-700">
                                      ±{event.implied_move.toFixed(1)}%
                                    </span>
                                  </div>
                                </div>
                              )}
                            </div>

                            <div className="mt-2 text-xs text-blue-600 font-medium">
                              Click for history →
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Earnings History Modal */}
      {selectedSymbol && earningsHistory && (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            {/* Modal Header */}
            <div className="bg-blue-600 text-white px-6 py-4 flex items-center justify-between sticky top-0">
              <div>
                <h2 className="text-2xl font-bold">{selectedSymbol} Earnings History</h2>
                <p className="text-sm mt-1">
                  {earningsHistory.count} past earnings •
                  Avg Surprise: {earningsHistory.average_eps_surprise_pct?.toFixed(1) || 'N/A'}%
                </p>
              </div>
              <button
                onClick={() => {
                  setSelectedSymbol(null);
                  setEarningsHistory(null);
                }}
                className="text-white hover:text-gray-200 text-2xl"
              >
                ×
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-6">
              <div className="space-y-4">
                {earningsHistory.events.map((event, idx) => (
                  <div
                    key={idx}
                    className="p-4 border-2 border-gray-200 rounded hover:border-blue-300 transition-colors"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <div className="text-sm text-gray-600">{event.date}</div>
                        <div className="font-semibold">
                          {event.fiscal_quarter} {event.fiscal_year}
                        </div>
                      </div>
                      <span className="text-xs px-2 py-1 bg-gray-200 text-gray-800 rounded">
                        {getEarningsTimeLabel(event.time)}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <div className="text-gray-600 mb-1">EPS</div>
                        <div className="flex items-baseline gap-2">
                          <span className="text-gray-500">Est: ${event.eps_estimate?.toFixed(2) || 'N/A'}</span>
                          <span className={`font-bold ${
                            event.eps_surprise_pct && event.eps_surprise_pct > 0
                              ? 'text-green-700'
                              : event.eps_surprise_pct && event.eps_surprise_pct < 0
                              ? 'text-red-700'
                              : 'text-gray-900'
                          }`}>
                            Act: ${event.eps_actual?.toFixed(2) || 'N/A'}
                          </span>
                          {event.eps_surprise_pct && (
                            <span className={`text-xs font-semibold ${
                              event.eps_surprise_pct > 0 ? 'text-green-700' : 'text-red-700'
                            }`}>
                              ({event.eps_surprise_pct > 0 ? '+' : ''}{event.eps_surprise_pct.toFixed(1)}%)
                            </span>
                          )}
                        </div>
                      </div>

                      <div>
                        <div className="text-gray-600 mb-1">Revenue</div>
                        <div className="flex items-baseline gap-2">
                          <span className="text-gray-500">Est: {formatCurrency(event.revenue_estimate)}</span>
                          <span className="font-bold">Act: {formatCurrency(event.revenue_actual)}</span>
                          {event.revenue_surprise_pct && (
                            <span className={`text-xs font-semibold ${
                              event.revenue_surprise_pct > 0 ? 'text-green-700' : 'text-red-700'
                            }`}>
                              ({event.revenue_surprise_pct > 0 ? '+' : ''}{event.revenue_surprise_pct.toFixed(1)}%)
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EconomicCalendarPage;
