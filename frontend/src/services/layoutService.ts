/**
 * Layout Service
 *
 * Multi-monitor layout management with persistence.
 * Enables professional multi-screen trading setups.
 */

export interface LayoutWidget {
  id: string;
  type: 'options-chain' | 'risk-dashboard' | 'news' | 'chart' | 'positions' | 'execution' | 'calendar' | 'backtest' | 'schwab-trading' | 'sentiment' | 'anomalies';
  x: number;
  y: number;
  width: number;
  height: number;
  monitor: number;  // 0 = primary, 1 = secondary, etc.
  config?: Record<string, any>;
}

export interface Layout {
  id: string;
  name: string;
  description: string;
  widgets: LayoutWidget[];
  created: string;
  modified: string;
}

const STORAGE_KEY = 'options-optimizer-layouts';
const ACTIVE_LAYOUT_KEY = 'options-optimizer-active-layout';

// ============================================================================
// Preset Layouts
// ============================================================================

export const PRESET_LAYOUTS: Layout[] = [
  {
    id: 'day-trader',
    name: 'Day Trader',
    description: 'Fast-paced trading with charts and execution',
    created: new Date().toISOString(),
    modified: new Date().toISOString(),
    widgets: [
      // Monitor 1 (Primary)
      { id: '1', type: 'options-chain', x: 0, y: 0, width: 50, height: 50, monitor: 0 },
      { id: '2', type: 'schwab-trading', x: 50, y: 0, width: 50, height: 50, monitor: 0 },
      { id: '3', type: 'execution', x: 0, y: 50, width: 50, height: 50, monitor: 0 },
      { id: '4', type: 'news', x: 50, y: 50, width: 50, height: 50, monitor: 0 },
      // Monitor 2 (Secondary)
      { id: '5', type: 'chart', x: 0, y: 0, width: 100, height: 60, monitor: 1 },
      { id: '6', type: 'positions', x: 0, y: 60, width: 100, height: 40, monitor: 1 },
    ]
  },
  {
    id: 'swing-trader',
    name: 'Swing Trader',
    description: 'Analysis-focused with research and calendar',
    created: new Date().toISOString(),
    modified: new Date().toISOString(),
    widgets: [
      // Monitor 1
      { id: '1', type: 'options-chain', x: 0, y: 0, width: 60, height: 100, monitor: 0 },
      { id: '2', type: 'calendar', x: 60, y: 0, width: 40, height: 50, monitor: 0 },
      { id: '3', type: 'news', x: 60, y: 50, width: 40, height: 50, monitor: 0 },
      // Monitor 2
      { id: '4', type: 'backtest', x: 0, y: 0, width: 50, height: 100, monitor: 1 },
      { id: '5', type: 'risk-dashboard', x: 50, y: 0, width: 50, height: 100, monitor: 1 },
    ]
  },
  {
    id: 'options-specialist',
    name: 'Options Specialist',
    description: 'Greeks, volatility, and risk analysis',
    created: new Date().toISOString(),
    modified: new Date().toISOString(),
    widgets: [
      // Monitor 1
      { id: '1', type: 'options-chain', x: 0, y: 0, width: 50, height: 60, monitor: 0 },
      { id: '2', type: 'risk-dashboard', x: 50, y: 0, width: 50, height: 60, monitor: 0 },
      { id: '3', type: 'execution', x: 0, y: 60, width: 100, height: 40, monitor: 0 },
      // Monitor 2
      { id: '4', type: 'schwab-trading', x: 0, y: 0, width: 50, height: 50, monitor: 1 },
      { id: '5', type: 'backtest', x: 50, y: 0, width: 50, height: 50, monitor: 1 },
      { id: '6', type: 'positions', x: 0, y: 50, width: 100, height: 50, monitor: 1 },
    ]
  },
  {
    id: 'research-mode',
    name: 'Research Mode',
    description: 'Deep analysis with backtesting and sentiment',
    created: new Date().toISOString(),
    modified: new Date().toISOString(),
    widgets: [
      // Monitor 1
      { id: '1', type: 'backtest', x: 0, y: 0, width: 100, height: 100, monitor: 0 },
      // Monitor 2
      { id: '2', type: 'sentiment', x: 0, y: 0, width: 50, height: 50, monitor: 1 },
      { id: '3', type: 'calendar', x: 50, y: 0, width: 50, height: 50, monitor: 1 },
      { id: '4', type: 'news', x: 0, y: 50, width: 50, height: 50, monitor: 1 },
      { id: '5', type: 'anomalies', x: 50, y: 50, width: 50, height: 50, monitor: 1 },
    ]
  },
  {
    id: 'single-monitor',
    name: 'Single Monitor',
    description: 'Efficient layout for one screen',
    created: new Date().toISOString(),
    modified: new Date().toISOString(),
    widgets: [
      { id: '1', type: 'options-chain', x: 0, y: 0, width: 50, height: 50, monitor: 0 },
      { id: '2', type: 'positions', x: 50, y: 0, width: 50, height: 50, monitor: 0 },
      { id: '3', type: 'news', x: 0, y: 50, width: 50, height: 50, monitor: 0 },
      { id: '4', type: 'schwab-trading', x: 50, y: 50, width: 50, height: 50, monitor: 0 },
    ]
  }
];

// ============================================================================
// Layout Management
// ============================================================================

/**
 * Get all saved layouts
 */
export function getSavedLayouts(): Layout[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      return JSON.parse(stored);
    }
  } catch (error) {
    console.error('Failed to load layouts:', error);
  }
  return [];
}

/**
 * Save a layout
 */
export function saveLayout(layout: Layout): void {
  const layouts = getSavedLayouts();
  const existingIndex = layouts.findIndex(l => l.id === layout.id);

  const updatedLayout = {
    ...layout,
    modified: new Date().toISOString()
  };

  if (existingIndex >= 0) {
    layouts[existingIndex] = updatedLayout;
  } else {
    layouts.push(updatedLayout);
  }

  localStorage.setItem(STORAGE_KEY, JSON.stringify(layouts));
}

/**
 * Delete a layout
 */
export function deleteLayout(layoutId: string): void {
  const layouts = getSavedLayouts().filter(l => l.id !== layoutId);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(layouts));
}

/**
 * Get active layout
 */
export function getActiveLayout(): Layout | null {
  try {
    const activeId = localStorage.getItem(ACTIVE_LAYOUT_KEY);
    if (activeId) {
      // Check saved layouts first
      const savedLayouts = getSavedLayouts();
      const saved = savedLayouts.find(l => l.id === activeId);
      if (saved) return saved;

      // Check presets
      const preset = PRESET_LAYOUTS.find(l => l.id === activeId);
      if (preset) return preset;
    }
  } catch (error) {
    console.error('Failed to load active layout:', error);
  }
  return null;
}

/**
 * Set active layout
 */
export function setActiveLayout(layoutId: string): void {
  localStorage.setItem(ACTIVE_LAYOUT_KEY, layoutId);
}

/**
 * Get all layouts (presets + saved)
 */
export function getAllLayouts(): Layout[] {
  return [...PRESET_LAYOUTS, ...getSavedLayouts()];
}

/**
 * Create new empty layout
 */
export function createNewLayout(name: string, description: string): Layout {
  return {
    id: `layout-${Date.now()}`,
    name,
    description,
    widgets: [],
    created: new Date().toISOString(),
    modified: new Date().toISOString()
  };
}

/**
 * Clone a layout
 */
export function cloneLayout(layout: Layout, newName: string): Layout {
  return {
    ...layout,
    id: `layout-${Date.now()}`,
    name: newName,
    created: new Date().toISOString(),
    modified: new Date().toISOString()
  };
}

/**
 * Export layout to JSON
 */
export function exportLayout(layout: Layout): string {
  return JSON.stringify(layout, null, 2);
}

/**
 * Import layout from JSON
 */
export function importLayout(json: string): Layout {
  const layout = JSON.parse(json);
  layout.id = `layout-${Date.now()}`;  // Assign new ID
  layout.created = new Date().toISOString();
  layout.modified = new Date().toISOString();
  return layout;
}

// ============================================================================
// Window Management
// ============================================================================

/**
 * Open widget in new window
 */
export function openWidgetInWindow(widget: LayoutWidget): Window | null {
  const url = getWidgetUrl(widget.type);
  const features = `width=${window.screen.width * widget.width / 100},height=${window.screen.height * widget.height / 100},left=${window.screen.width * widget.x / 100},top=${window.screen.height * widget.y / 100}`;

  return window.open(url, `widget-${widget.id}`, features);
}

/**
 * Get URL for widget type
 */
function getWidgetUrl(type: LayoutWidget['type']): string {
  const baseUrl = window.location.origin;
  const routes: Record<LayoutWidget['type'], string> = {
    'options-chain': '/options-chain',
    'risk-dashboard': '/risk-dashboard',
    'news': '/news',
    'chart': '/chart-analysis',
    'positions': '/positions',
    'execution': '/execution',
    'calendar': '/calendar',
    'backtest': '/backtest',
    'schwab-trading': '/schwab-trading',
    'sentiment': '/sentiment',
    'anomalies': '/anomalies'
  };

  return `${baseUrl}${routes[type]}`;
}

/**
 * Detect number of monitors (approximation)
 */
export function detectMonitorCount(): number {
  // This is an approximation - browsers don't expose exact monitor count
  // We use screen width ratio as a heuristic
  const screenWidth = window.screen.width;
  const availWidth = window.screen.availWidth;

  if (screenWidth > 3000) return 2;  // Likely dual monitor
  if (screenWidth > 5000) return 3;  // Likely triple monitor
  return 1;
}

/**
 * Get monitor dimensions
 */
export function getMonitorDimensions(monitorIndex: number): { x: number; y: number; width: number; height: number } {
  const totalWidth = window.screen.width;
  const height = window.screen.height;
  const monitorCount = detectMonitorCount();

  if (monitorCount === 1) {
    return { x: 0, y: 0, width: totalWidth, height };
  }

  const monitorWidth = totalWidth / monitorCount;
  return {
    x: monitorIndex * monitorWidth,
    y: 0,
    width: monitorWidth,
    height
  };
}
