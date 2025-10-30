/**
 * API service for backend communication
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

class ApiService {
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  // Positions
  async getPositions(userId: string, status: string = 'open') {
    return this.request(`/positions?user_id=${userId}&status=${status}`);
  }

  async getPosition(positionId: string) {
    return this.request(`/positions/${positionId}`);
  }

  async createPosition(position: any) {
    return this.request('/positions', {
      method: 'POST',
      body: JSON.stringify(position),
    });
  }

  async updatePosition(positionId: string, updates: any) {
    return this.request(`/positions/${positionId}`, {
      method: 'PUT',
      body: JSON.stringify(updates),
    });
  }

  async deletePosition(positionId: string) {
    return this.request(`/positions/${positionId}`, {
      method: 'DELETE',
    });
  }

  // Analytics
  async calculateGreeks(positionId: string) {
    return this.request(`/analytics/greeks?position_id=${positionId}`, {
      method: 'POST',
    });
  }

  async calculateEV(positionId: string) {
    return this.request(`/analytics/ev?position_id=${positionId}`, {
      method: 'POST',
    });
  }

  // Analysis
  async runAnalysis(userId: string, reportType: string = 'daily') {
    return this.request('/analysis/run', {
      method: 'POST',
      body: JSON.stringify({ user_id: userId, report_type: reportType }),
    });
  }

  // Reports
  async getReports(userId: string, limit: number = 10) {
    return this.request(`/reports?user_id=${userId}&limit=${limit}`);
  }
}

export const api = new ApiService();

