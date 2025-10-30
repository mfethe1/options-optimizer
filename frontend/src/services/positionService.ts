/**
 * Position Service
 * Handles all position-related API calls
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface StockPositionCreate {
  symbol: string;
  quantity: number;
  entry_price: number;
  entry_date?: string;
  target_price?: number;
  stop_loss?: number;
  notes?: string;
}

export interface OptionPositionCreate {
  symbol: string;
  option_type: string;
  strike: number;
  expiration_date: string;
  quantity: number;
  premium_paid: number;
  entry_date?: string;
  target_price?: number;
  target_profit_pct?: number;
  stop_loss_pct?: number;
  notes?: string;
}

export interface CSVImportResult {
  success: number;
  failed: number;
  errors: string[];
  position_ids: string[];
  chase_conversion?: {
    total_rows: number;
    options_converted: number;
    cash_skipped: number;
    conversion_errors: number;
    error_details: string[];
  };
}

class PositionService {
  // Stock Positions
  async getStockPositions(enrich: boolean = false): Promise<any[]> {
    const response = await fetch(`${API_BASE_URL}/api/positions/stocks?enrich=${enrich}`);
    if (!response.ok) throw new Error('Failed to fetch stock positions');
    return response.json();
  }

  async getStockPosition(positionId: string, enrich: boolean = false): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/api/positions/stocks/${positionId}?enrich=${enrich}`);
    if (!response.ok) throw new Error('Failed to fetch stock position');
    return response.json();
  }

  async createStockPosition(position: StockPositionCreate): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/api/positions/stocks`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(position)
    });
    if (!response.ok) throw new Error('Failed to create stock position');
    return response.json();
  }

  async updateStockPosition(positionId: string, updates: Partial<StockPositionCreate>): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/api/positions/stocks/${positionId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates)
    });
    if (!response.ok) throw new Error('Failed to update stock position');
    return response.json();
  }

  async deleteStockPosition(positionId: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/api/positions/stocks/${positionId}`, {
      method: 'DELETE'
    });
    if (!response.ok) throw new Error('Failed to delete stock position');
  }

  // Option Positions
  async getOptionPositions(enrich: boolean = false): Promise<any[]> {
    const response = await fetch(`${API_BASE_URL}/api/positions/options?enrich=${enrich}`);
    if (!response.ok) throw new Error('Failed to fetch option positions');
    return response.json();
  }

  async getOptionPosition(positionId: string, enrich: boolean = false): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/api/positions/options/${positionId}?enrich=${enrich}`);
    if (!response.ok) throw new Error('Failed to fetch option position');
    return response.json();
  }

  async createOptionPosition(position: OptionPositionCreate): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/api/positions/options`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(position)
    });
    if (!response.ok) throw new Error('Failed to create option position');
    return response.json();
  }

  async updateOptionPosition(positionId: string, updates: Partial<OptionPositionCreate>): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/api/positions/options/${positionId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates)
    });
    if (!response.ok) throw new Error('Failed to update option position');
    return response.json();
  }

  async deleteOptionPosition(positionId: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/api/positions/options/${positionId}`, {
      method: 'DELETE'
    });
    if (!response.ok) throw new Error('Failed to delete option position');
  }

  // CSV Templates
  async downloadTemplate(type: 'stocks' | 'options'): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/api/positions/templates/${type}`);
    if (!response.ok) throw new Error('Failed to download template');

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${type}_template_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  }

  // CSV Export
  async exportPositions(type: 'stocks' | 'options'): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/api/positions/export/${type}`);
    if (!response.ok) throw new Error('Failed to export positions');

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${type}_positions_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  }

  // CSV Import
  async importPositions(
    type: 'stocks' | 'options',
    file: File,
    replaceExisting: boolean = false,
    chaseFormat: boolean = false
  ): Promise<CSVImportResult> {
    const formData = new FormData();
    formData.append('file', file);

    const url =
      type === 'options'
        ? `${API_BASE_URL}/api/positions/import/options?replace_existing=${replaceExisting}&chase_format=${chaseFormat}`
        : `${API_BASE_URL}/api/positions/import/stocks?replace_existing=${replaceExisting}`;

    const response = await fetch(url, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) throw new Error('Failed to import positions');
    return response.json();
  }

  // Portfolio Summary
  async getPortfolioSummary(enrich: boolean = true): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/api/positions/summary?enrich=${enrich}`);
    if (!response.ok) throw new Error('Failed to fetch portfolio summary');
    return response.json();
  }
  // Daily Research Plan
  async runDailyResearch(schedule: 'auto' | 'pre_market' | 'market_open' | 'mid_day' | 'end_of_day' = 'auto'): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/api/research/plan/daily?schedule=${schedule}`);
    if (!response.ok) throw new Error('Failed to run daily research plan');
    return response.json();
  }


  // Enrich All Positions
  async enrichAllPositions(): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/api/positions/enrich`, {
      method: 'POST'
    });
    if (!response.ok) throw new Error('Failed to enrich positions');
    return response.json();
  }
}

export const positionService = new PositionService();

