/**
 * Economic Calendar API Service
 *
 * Provides earnings calendar, economic events, and volatility implications.
 * Bloomberg EVTS equivalent.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface EarningsEvent {
  symbol: string;
  company_name: string;
  date: string;
  time: string; // 'bmo', 'amc', 'intraday'
  fiscal_quarter: string;
  fiscal_year: number;
  eps_estimate: number | null;
  revenue_estimate: number | null;
  eps_actual: number | null;
  revenue_actual: number | null;
  eps_surprise_pct: number | null;
  revenue_surprise_pct: number | null;
  historical_move_avg: number | null;
  implied_move: number | null;
}

export interface EconomicEvent {
  event_type: string;
  name: string;
  date: string;
  time: string;
  importance: string; // 'high', 'medium', 'low'
  estimate: string | null;
  actual: string | null;
  previous: string | null;
  market_impact: string | null; // 'bullish', 'bearish', 'neutral'
  volatility_expected: string | null;
}

export interface CalendarDay {
  date: string;
  total_events: number;
  high_importance_count: number;
  major_earnings_count: number;
  earnings_events: Array<{
    symbol: string;
    company_name: string;
    time: string;
    fiscal_quarter: string;
    eps_estimate: number | null;
    eps_actual: number | null;
    eps_surprise_pct: number | null;
    implied_move: number | null;
  }>;
  economic_events: Array<{
    event_type: string;
    name: string;
    time: string;
    importance: string;
    estimate: string | null;
    actual: string | null;
    volatility_expected: string | null;
  }>;
}

export interface EarningsCalendarResponse {
  from_date: string;
  to_date: string;
  count: number;
  events: EarningsEvent[];
}

export interface EconomicCalendarResponse {
  from_date: string;
  to_date: string;
  count: number;
  events: EconomicEvent[];
}

export interface CompleteCalendarResponse {
  from_date: string;
  to_date: string;
  total_days: number;
  days: CalendarDay[];
}

export interface EarningsHistoryResponse {
  symbol: string;
  count: number;
  average_eps_surprise_pct: number | null;
  events: Array<{
    date: string;
    time: string;
    fiscal_quarter: string;
    fiscal_year: number;
    eps_estimate: number | null;
    eps_actual: number | null;
    revenue_estimate: number | null;
    revenue_actual: number | null;
    eps_surprise_pct: number | null;
    revenue_surprise_pct: number | null;
  }>;
}

/**
 * Get earnings calendar
 */
export async function getEarningsCalendar(
  fromDate?: string,
  toDate?: string,
  symbols?: string,
  daysAhead: number = 30
): Promise<EarningsCalendarResponse> {
  const params = new URLSearchParams();
  if (fromDate) params.append('from_date', fromDate);
  if (toDate) params.append('to_date', toDate);
  if (symbols) params.append('symbols', symbols);
  params.append('days_ahead', daysAhead.toString());

  const response = await fetch(
    `${API_BASE_URL}/api/calendar/earnings?${params.toString()}`
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch earnings calendar: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get economic events calendar
 */
export async function getEconomicCalendar(
  fromDate?: string,
  toDate?: string,
  eventTypes?: string,
  daysAhead: number = 30
): Promise<EconomicCalendarResponse> {
  const params = new URLSearchParams();
  if (fromDate) params.append('from_date', fromDate);
  if (toDate) params.append('to_date', toDate);
  if (eventTypes) params.append('event_types', eventTypes);
  params.append('days_ahead', daysAhead.toString());

  const response = await fetch(
    `${API_BASE_URL}/api/calendar/economic?${params.toString()}`
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch economic calendar: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get complete calendar by day
 */
export async function getCompleteCalendar(
  fromDate?: string,
  toDate?: string,
  symbols?: string,
  daysAhead: number = 30
): Promise<CompleteCalendarResponse> {
  const params = new URLSearchParams();
  if (fromDate) params.append('from_date', fromDate);
  if (toDate) params.append('to_date', toDate);
  if (symbols) params.append('symbols', symbols);
  params.append('days_ahead', daysAhead.toString());

  const response = await fetch(
    `${API_BASE_URL}/api/calendar/complete?${params.toString()}`
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch complete calendar: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get earnings history for a symbol
 */
export async function getEarningsHistory(
  symbol: string,
  limit: number = 8
): Promise<EarningsHistoryResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/calendar/earnings/${symbol}/history?limit=${limit}`
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch earnings history: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get upcoming week calendar
 */
export async function getUpcomingWeek(): Promise<CompleteCalendarResponse> {
  const response = await fetch(`${API_BASE_URL}/api/calendar/upcoming-week`);

  if (!response.ok) {
    throw new Error(`Failed to fetch upcoming week: ${response.statusText}`);
  }

  return response.json();
}
