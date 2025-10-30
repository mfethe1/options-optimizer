/**
 * TypeScript types for InvestorReport.v1 JSON Schema
 * 
 * Mirrors src/schemas/investor_report_schema.json exactly.
 * Used by frontend components for type safety and autocomplete.
 * 
 * Performance contracts:
 * - API response: <500ms (cached)
 * - UI render: <100ms
 * - WebSocket latency: <50ms
 */

// Confidence scoring
export interface Confidence {
  overall: number;  // 0.0 to 1.0
  drivers: string[];  // Factors contributing to confidence
}

// Authoritative source with provenance
export interface Source {
  title: string;
  url: string;
  provider: string;  // Cboe, SEC, FRED, ExtractAlpha, AlphaSense, LSEG
  as_of: string;  // ISO 8601 timestamp
}

// Risk Panel - 7 institutional metrics
export interface RiskPanel {
  omega: number;  // Probability-weighted gains/losses (>2.0 = Renaissance-level)
  gh1: number;  // Return enhancement + risk reduction vs benchmark
  pain_index: number;  // Drawdown depth × duration (lower is better)
  upside_capture: number;  // % of benchmark gains captured
  downside_capture: number;  // % of benchmark losses captured
  cvar_95: number;  // Expected loss in worst 5% scenarios
  max_drawdown: number;  // Maximum peak-to-trough decline
  explanations?: string[];  // Optional human-readable explanations
}

// Phase 4 Technical Signals (short-horizon 1-5 day)
export interface Phase4Tech {
  options_flow_composite: number | null;  // PCR + IV skew + volume (-1 to +1)
  residual_momentum: number | null;  // Idiosyncratic momentum z-score
  seasonality_score: number | null;  // Calendar patterns (-1 to +1)
  breadth_liquidity: number | null;  // Market internals (-1 to +1)
  explanations?: string[];  // Optional human-readable explanations
  unavailable_with_reason?: string;  // Reason if metrics unavailable
}

// ML Alpha signal
export interface MLAlpha {
  score: number;  // -1 to +1
  explanations?: string[];
}

// Market regime classification
export type Regime = 'Low-Vol' | 'Normal' | 'High-Vol' | 'Crisis';

// Sentiment signal
export interface Sentiment {
  level: number;  // -1 (bearish) to +1 (bullish)
  delta: number;  // Change from previous period
  explanations?: string[];
}

// Smart money positioning
export interface SmartMoney {
  thirteenF: number;  // 13F institutional positioning (-1 to +1)
  insider_bias: number;  // Insider buy/sell bias (-1 to +1)
  options_bias: number;  // Options positioning bias (-1 to +1)
  explanations?: string[];
}

// Alternative data signals
export interface AltData {
  digital_demand: number;  // Web traffic, app downloads, etc. (-1 to +1)
  earnings_surprise_pred: number;  // Predicted earnings surprise (-1 to +1)
  explanations?: string[];
}

// All signals combined
export interface Signals {
  ml_alpha: MLAlpha;
  regime: Regime;
  sentiment: Sentiment;
  smart_money: SmartMoney;
  alt_data: AltData;
  phase4_tech: Phase4Tech;
}

// Executive summary top pick
export interface TopPick {
  ticker: string;
  rationale: string;
  expected_horizon_days: number;
}

// Executive summary
export interface ExecutiveSummary {
  top_picks: TopPick[];
  key_risks: string[];
  thesis: string;
}

// Action item (buy/sell/hold recommendation)
export interface ActionItem {
  ticker: string;
  action: 'buy' | 'hold' | 'sell' | 'watch';
  sizing: string;  // e.g., "2% of portfolio", "half position"
  risk_controls: string;  // e.g., "stop at -5%", "trailing stop"
}

// Metadata (optional)
export interface Metadata {
  schema_version?: string;  // "InvestorReport.v1"
  validated?: boolean;  // Schema validation passed
  fallback?: boolean;  // True if fallback narrative used
  refreshing?: boolean;  // True if async refresh scheduled
  cached?: boolean;  // True if served from cache
  cache_layer?: 'L1' | 'L2';  // Which cache layer served the response
  response_time_ms?: number;  // API response time
  user_id?: string;
  fresh?: boolean;
}

// Complete InvestorReport.v1 schema
export interface InvestorReport {
  as_of: string;  // ISO 8601 timestamp
  universe: string[];  // List of tickers analyzed
  executive_summary: ExecutiveSummary;
  risk_panel: RiskPanel;
  signals: Signals;
  actions: ActionItem[];
  sources: Source[];
  confidence: Confidence;
  metadata?: Metadata;
}

// WebSocket message types
export interface Phase4UpdateMessage {
  type: 'phase4_update';
  timestamp: string;  // ISO 8601
  data: Phase4Tech;
}

export interface HeartbeatMessage {
  type: 'heartbeat';
  timestamp: string;
}

export interface PongMessage {
  type: 'pong';
  timestamp: string;
}

export type WebSocketMessage = Phase4UpdateMessage | HeartbeatMessage | PongMessage;

// API error response
export interface APIError {
  detail: string;
  status_code?: number;
}

// API response wrapper (for 202 Accepted)
export interface ProcessingResponse {
  status: 'processing';
  message?: string;
}

// Type guards
export function isInvestorReport(obj: any): obj is InvestorReport {
  return (
    obj &&
    typeof obj.as_of === 'string' &&
    Array.isArray(obj.universe) &&
    obj.executive_summary &&
    obj.risk_panel &&
    obj.signals &&
    Array.isArray(obj.actions) &&
    Array.isArray(obj.sources) &&
    obj.confidence
  );
}

export function isPhase4UpdateMessage(msg: any): msg is Phase4UpdateMessage {
  return msg && msg.type === 'phase4_update' && msg.data;
}

export function isProcessingResponse(obj: any): obj is ProcessingResponse {
  return obj && obj.status === 'processing';
}

// Color coding helpers for UI
export const RISK_LEVEL_COLORS = {
  Critical: '#ef4444',  // red-500
  High: '#f97316',  // orange-500
  Medium: '#eab308',  // yellow-500
  Low: '#22c55e',  // green-500
} as const;

export const SIGNAL_COLORS = {
  positive: '#22c55e',  // green-500
  negative: '#ef4444',  // red-500
  neutral: '#eab308',  // yellow-500
  warning: '#f97316',  // orange-500
} as const;

// Utility functions
export function getRiskLevel(score: number): keyof typeof RISK_LEVEL_COLORS {
  if (score >= 0.8) return 'Critical';
  if (score >= 0.6) return 'High';
  if (score >= 0.4) return 'Medium';
  return 'Low';
}

export function getSignalColor(score: number): string {
  if (score > 0.3) return SIGNAL_COLORS.positive;
  if (score < -0.3) return SIGNAL_COLORS.negative;
  if (Math.abs(score) < 0.1) return SIGNAL_COLORS.neutral;
  return SIGNAL_COLORS.warning;
}

export function formatPercentage(value: number, decimals: number = 2): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatCurrency(value: number, decimals: number = 2): string {
  return `$${value.toFixed(decimals)}`;
}

export function formatTimestamp(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    timeZoneName: 'short'
  });
}

