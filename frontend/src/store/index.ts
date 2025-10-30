/**
 * Zustand store for global state management
 */
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

interface Position {
  id: string;
  symbol: string;
  strategy_type: string;
  entry_date: string;
  expiration_date: string;
  status: string;
  total_premium: number;
  market_value?: number;
  pnl?: number;
  pnl_pct?: number;
  legs: any[];
}

interface MarketData {
  [symbol: string]: {
    underlying_price: number;
    iv: number;
    volume: number;
    // ... other fields
  };
}

interface Report {
  id: string;
  report_type: string;
  executive_summary: string;
  risk_score: number;
  generated_at: string;
  // ... other fields
}

interface StoreState {
  // User
  userId: string;
  setUserId: (id: string) => void;

  // Positions
  positions: Position[];
  setPositions: (positions: Position[]) => void;
  addPosition: (position: Position) => void;
  updatePosition: (id: string, updates: Partial<Position>) => void;
  removePosition: (id: string) => void;

  // Market Data
  marketData: MarketData;
  setMarketData: (data: MarketData) => void;
  updateSymbolData: (symbol: string, data: any) => void;

  // Reports
  reports: Report[];
  setReports: (reports: Report[]) => void;
  addReport: (report: Report) => void;

  // UI State
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
  
  selectedPosition: string | null;
  setSelectedPosition: (id: string | null) => void;

  // WebSocket
  wsConnected: boolean;
  setWsConnected: (connected: boolean) => void;
}

export const useStore = create<StoreState>()(
  devtools(
    persist(
      (set) => ({
        // User
        userId: 'demo-user-1',
        setUserId: (id) => set({ userId: id }),

        // Positions
        positions: [],
        setPositions: (positions) => set({ positions }),
        addPosition: (position) =>
          set((state) => ({ positions: [...state.positions, position] })),
        updatePosition: (id, updates) =>
          set((state) => ({
            positions: state.positions.map((p) =>
              p.id === id ? { ...p, ...updates } : p
            ),
          })),
        removePosition: (id) =>
          set((state) => ({
            positions: state.positions.filter((p) => p.id !== id),
          })),

        // Market Data
        marketData: {},
        setMarketData: (data) => set({ marketData: data }),
        updateSymbolData: (symbol, data) =>
          set((state) => ({
            marketData: {
              ...state.marketData,
              [symbol]: { ...state.marketData[symbol], ...data },
            },
          })),

        // Reports
        reports: [],
        setReports: (reports) => set({ reports }),
        addReport: (report) =>
          set((state) => ({ reports: [report, ...state.reports] })),

        // UI State
        isLoading: false,
        setIsLoading: (loading) => set({ isLoading: loading }),
        
        selectedPosition: null,
        setSelectedPosition: (id) => set({ selectedPosition: id }),

        // WebSocket
        wsConnected: false,
        setWsConnected: (connected) => set({ wsConnected: connected }),
      }),
      {
        name: 'options-analysis-store',
        partialize: (state) => ({
          userId: state.userId,
          positions: state.positions,
        }),
      }
    )
  )
);

