/**
 * Swarm Analysis Service
 * Handles all swarm-related API calls including CSV upload and analysis
 *
 * Enhanced to support institutional-grade 17-agent swarm analysis with:
 * - Position-by-position breakdown
 * - Comprehensive stock reports
 * - Replacement recommendations
 * - Swarm health metrics
 * - Agent-to-agent discussion logs
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Enable detailed logging for debugging
const DEBUG_MODE = import.meta.env.DEV || false;

function log(...args: any[]) {
  if (DEBUG_MODE) {
    console.log('[SwarmService]', ...args);
  }
}

function logError(...args: any[]) {
  console.error('[SwarmService]', ...args);
}

/**
 * Enhanced Swarm Analysis Result
 * Supports both legacy format and new institutional-grade format
 */
import type { InvestorReport } from '../types/investor-report';

export interface SwarmAnalysisResult {
  // Core consensus decisions
  consensus_decisions: {
    overall_action: {
      choice: string;
      confidence: number;
      reasoning: string;
    };
    risk_level: {
      choice: string;
      confidence: number;
      reasoning: string;
    };
    market_outlook: {
      choice: string;
      confidence: number;
      reasoning: string;
    };
  };

  // Legacy format (for backward compatibility)
  agent_analyses?: {
    [agentId: string]: {
      analysis: any;
      recommendation: any;
    };
  };

  // NEW: Enhanced institutional-grade fields
  investor_report?: InvestorReport; // Investor-friendly synthesis for UI
  agent_insights?: Array<{
    agent_id: string;
    agent_type: string;
    timestamp: string;
    llm_response_text: string;
    analysis_fields: any;
    recommendation: any;
    error?: string;
  }>;

  position_analysis?: Array<{
    symbol: string;
    asset_type: string;
    option_type?: string;
    strike?: number;
    expiration_date?: string;
    quantity: number;
    current_metrics: any;
    greeks?: any;
    agent_insights_for_position: any[];
    comprehensive_stock_report?: any;
    replacement_recommendations?: any;
    risk_warnings?: any[];
    opportunities?: any[];
  }>;

  swarm_health?: {
    active_agents_count: number;
    contributed_vs_failed: {
      contributed: number;
      failed: number;
      success_rate: number;
    };
    communication_stats: any;
    consensus_strength: any;
  };

  enhanced_consensus?: any;
  discussion_logs?: Array<{
    source_agent: string;
    content: any;
    priority: number;
    confidence: number;
    timestamp: string;
  }>;

  // Standard fields
  portfolio_summary: {
    total_value: number;
    total_unrealized_pnl: number;
    total_unrealized_pnl_pct: number;
    positions_count: number;
  };

  import_stats?: {
    positions_imported: number;
    positions_failed: number;
    chase_conversion?: any;
    errors: string[];
  };

  execution_time: number;
  timestamp: string;
}

export interface SwarmAnalysisRequest {
  portfolio: {
    total_portfolio_value: number;
    total_unrealized_pnl: number;
    total_unrealized_pnl_pct: number;
    positions: any[];
  };
  consensus_method?: string;
}

class SwarmService {
  /**
   * Transform backend response to ensure compatibility
   * Handles both legacy and new institutional-grade formats
   */
  private transformResponse(backendData: any): SwarmAnalysisResult {
    log('Transforming backend response', {
      hasAgentInsights: !!backendData.agent_insights,
      hasAgentAnalyses: !!backendData.agent_analyses,
      hasPositionAnalysis: !!backendData.position_analysis,
      responseKeys: Object.keys(backendData)
    });

    // If backend returns agent_insights (new format), also create agent_analyses (legacy format)
    let agent_analyses = backendData.agent_analyses;

    if (!agent_analyses && backendData.agent_insights) {
      log('Converting agent_insights to agent_analyses for backward compatibility');
      agent_analyses = {};
      backendData.agent_insights.forEach((insight: any) => {
        agent_analyses[insight.agent_id] = {
          analysis: insight.analysis_fields || {},
          recommendation: insight.recommendation || {}
        };
      });
    }

    const result: SwarmAnalysisResult = {
      consensus_decisions: backendData.consensus_decisions,
      investor_report: backendData.investor_report,
      agent_analyses: agent_analyses,
      agent_insights: backendData.agent_insights,
      position_analysis: backendData.position_analysis,
      swarm_health: backendData.swarm_health,
      enhanced_consensus: backendData.enhanced_consensus,
      discussion_logs: backendData.discussion_logs,
      portfolio_summary: backendData.portfolio_summary,
      import_stats: backendData.import_stats,
      execution_time: backendData.execution_time,
      timestamp: backendData.timestamp
    };

    log('Transformation complete', {
      agentAnalysesCount: Object.keys(result.agent_analyses || {}).length,
      agentInsightsCount: result.agent_insights?.length || 0,
      positionAnalysisCount: result.position_analysis?.length || 0
    });

    return result;
  }

  /**
   * Validate response structure
   * Ensures all required fields are present
   */
  private validateResponse(data: any): void {
    const requiredFields = ['consensus_decisions', 'portfolio_summary', 'timestamp'];
    const missingFields = requiredFields.filter(field => !(field in data));

    if (missingFields.length > 0) {
      logError('Response validation failed - missing required fields:', missingFields);
      throw new Error(`Invalid API response: missing fields ${missingFields.join(', ')}`);
    }

    // Validate consensus_decisions structure
    const requiredConsensusFields = ['overall_action', 'risk_level', 'market_outlook'];
    const missingConsensusFields = requiredConsensusFields.filter(
      field => !(field in data.consensus_decisions)
    );

    if (missingConsensusFields.length > 0) {
      logError('Consensus decisions validation failed - missing fields:', missingConsensusFields);
      throw new Error(`Invalid consensus_decisions: missing ${missingConsensusFields.join(', ')}`);
    }

    log('Response validation passed');
  }

  /**
   * Analyze portfolio from CSV file
   * Uploads CSV, imports positions, and runs LLM-powered swarm analysis
   *
   * Enhanced with:
   * - Comprehensive logging
   * - Response validation
   * - Data transformation for compatibility
   * - Error handling
   */
  async analyzeFromCSV(
    file: File,
    chaseFormat: boolean = false,
    consensusMethod: string = 'weighted'
  ): Promise<SwarmAnalysisResult> {
    log('Starting CSV analysis', {
      fileName: file.name,
      fileSize: file.size,
      chaseFormat,
      consensusMethod
    });

    const formData = new FormData();
    formData.append('file', file);

    const url = `${API_BASE_URL}/api/swarm/analyze-csv?chase_format=${chaseFormat}&consensus_method=${consensusMethod}`;
    log('Request URL:', url);

    try {
      const response = await fetch(url, {
        method: 'POST',
        body: formData
      });

      log('Response received', {
        status: response.status,
        statusText: response.statusText,
        headers: Object.fromEntries(response.headers.entries())
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Failed to analyze portfolio' }));
        logError('API error response:', error);
        throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      log('Response data received', {
        dataKeys: Object.keys(data),
        dataSize: JSON.stringify(data).length
      });

      // Validate response
      this.validateResponse(data);

      // Transform response for compatibility
      const transformedData = this.transformResponse(data);

      log('Analysis complete', {
        agentsContributed: transformedData.swarm_health?.contributed_vs_failed?.contributed || 0,
        positionsAnalyzed: transformedData.position_analysis?.length || 0,
        executionTime: transformedData.execution_time
      });

      return transformedData;

    } catch (error: any) {
      logError('Error in analyzeFromCSV:', error);
      throw error;
    }
  }

  /**
   * Analyze existing portfolio
   * Runs swarm analysis on already-imported positions
   */
  async analyzePortfolio(
    request: SwarmAnalysisRequest
  ): Promise<SwarmAnalysisResult> {
    const response = await fetch(`${API_BASE_URL}/api/swarm/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Failed to analyze portfolio' }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  /**
   * Get swarm status
   */
  async getSwarmStatus(): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/api/swarm/status`);

    if (!response.ok) {
      throw new Error('Failed to get swarm status');
    }

    return response.json();
  }

  /**
   * Get agent details
   */
  async getAgentDetails(agentId: string): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/api/swarm/agents/${agentId}`);

    if (!response.ok) {
      throw new Error('Failed to get agent details');
    }

    return response.json();
  }

  /**
   * Get consensus history
   */
  async getConsensusHistory(limit: number = 10): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/api/swarm/consensus/history?limit=${limit}`);

    if (!response.ok) {
      throw new Error('Failed to get consensus history');
    }

    return response.json();
  }
}

export const swarmService = new SwarmService();
export default swarmService;

