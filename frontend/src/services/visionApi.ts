/**
 * API service for Vision-Based Chart Analysis
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface ChartAnalysisResponse {
  analysis: {
    patterns?: Array<{
      type: string;
      bias: string;
      confidence: number;
    }>;
    levels?: {
      support: number[];
      resistance: number[];
    };
    trend?: {
      direction: string;
      strength: string;
    };
    indicators?: Record<string, string>;
    recommendation?: {
      action: string;
      strikes?: number[];
      expiration?: string;
    };
    risks?: string[];
  };
  provider: string;
  timestamp: string;
}

export type AnalysisType = 'comprehensive' | 'pattern' | 'levels' | 'flow';

/**
 * Analyze a chart image
 */
export async function analyzeChart(
  imageFile: File,
  analysisType: AnalysisType = 'comprehensive',
  question?: string
): Promise<ChartAnalysisResponse> {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('analysis_type', analysisType);
  if (question) {
    formData.append('question', question);
  }

  const response = await fetch(`${API_BASE_URL}/api/vision/analyze-chart`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Compare multiple charts
 */
export async function compareCharts(
  chartFiles: File[],
  comparisonType: 'relative_strength' | 'divergence' | 'correlation' = 'relative_strength'
): Promise<any> {
  const formData = new FormData();
  chartFiles.forEach((file) => {
    formData.append('charts', file);
  });
  formData.append('comparison_type', comparisonType);

  const response = await fetch(`${API_BASE_URL}/api/vision/compare-charts`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Get available vision providers
 */
export async function getVisionProviders(): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/api/vision/providers`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Get supported analysis types
 */
export async function getAnalysisTypes(): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/api/vision/analysis-types`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}
