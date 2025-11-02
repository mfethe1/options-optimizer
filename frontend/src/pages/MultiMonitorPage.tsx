/**
 * Multi-Monitor Layout Manager
 *
 * Professional multi-screen trading setup with preset and custom layouts.
 * Enables Bloomberg Terminal-style workspace management.
 */

import React, { useState, useEffect } from 'react';
import { toast } from 'react-hot-toast';
import {
  getAllLayouts,
  getActiveLayout,
  setActiveLayout,
  saveLayout,
  deleteLayout,
  createNewLayout,
  cloneLayout,
  exportLayout,
  importLayout,
  openWidgetInWindow,
  detectMonitorCount,
  PRESET_LAYOUTS,
  type Layout,
  type LayoutWidget
} from '../services/layoutService';

const WIDGET_TYPES = [
  { id: 'options-chain', name: 'Options Chain', icon: '⚡', color: 'bg-blue-100' },
  { id: 'risk-dashboard', name: 'Risk Dashboard', icon: '🛡️', color: 'bg-red-100' },
  { id: 'news', name: 'News Feed', icon: '📰', color: 'bg-yellow-100' },
  { id: 'chart', name: 'Chart Analysis', icon: '📊', color: 'bg-green-100' },
  { id: 'positions', name: 'Positions', icon: '💼', color: 'bg-purple-100' },
  { id: 'execution', name: 'Execution Quality', icon: '🎯', color: 'bg-pink-100' },
  { id: 'calendar', name: 'Economic Calendar', icon: '📅', color: 'bg-indigo-100' },
  { id: 'backtest', name: 'Backtesting', icon: '📈', color: 'bg-teal-100' },
  { id: 'schwab-trading', name: 'Schwab Trading', icon: '💰', color: 'bg-orange-100' },
  { id: 'sentiment', name: 'Sentiment', icon: '😊', color: 'bg-lime-100' },
  { id: 'anomalies', name: 'Anomalies', icon: '🚨', color: 'bg-red-100' },
] as const;

export default function MultiMonitorPage() {
  const [layouts, setLayouts] = useState<Layout[]>([]);
  const [activeLayout, setActiveLayoutState] = useState<Layout | null>(null);
  const [selectedLayout, setSelectedLayout] = useState<Layout | null>(null);
  const [monitorCount, setMonitorCount] = useState(1);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showEditDialog, setShowEditDialog] = useState(false);
  const [openWindows, setOpenWindows] = useState<Map<string, Window>>(new Map());

  // New layout creation state
  const [newLayoutName, setNewLayoutName] = useState('');
  const [newLayoutDescription, setNewLayoutDescription] = useState('');

  useEffect(() => {
    loadLayouts();
    setMonitorCount(detectMonitorCount());
  }, []);

  const loadLayouts = () => {
    const allLayouts = getAllLayouts();
    setLayouts(allLayouts);

    const active = getActiveLayout();
    setActiveLayoutState(active);
    setSelectedLayout(active);
  };

  const handleActivateLayout = (layout: Layout) => {
    setActiveLayout(layout.id);
    setActiveLayoutState(layout);
    setSelectedLayout(layout);
    toast.success(`Activated layout: ${layout.name}`);
  };

  const handleOpenLayout = (layout: Layout) => {
    // Close existing windows
    openWindows.forEach(window => {
      if (window && !window.closed) {
        window.close();
      }
    });

    // Open new windows for each widget
    const newWindows = new Map<string, Window>();
    layout.widgets.forEach(widget => {
      const win = openWidgetInWindow(widget);
      if (win) {
        newWindows.set(widget.id, win);
      }
    });

    setOpenWindows(newWindows);
    toast.success(`Opened ${layout.widgets.length} windows for ${layout.name}`);
  };

  const handleCreateLayout = () => {
    if (!newLayoutName.trim()) {
      toast.error('Please enter a layout name');
      return;
    }

    const newLayout = createNewLayout(newLayoutName, newLayoutDescription);
    saveLayout(newLayout);
    loadLayouts();
    setShowCreateDialog(false);
    setNewLayoutName('');
    setNewLayoutDescription('');
    toast.success(`Created layout: ${newLayoutName}`);
  };

  const handleCloneLayout = (layout: Layout) => {
    const name = prompt('Enter name for cloned layout:', `${layout.name} (Copy)`);
    if (name) {
      const cloned = cloneLayout(layout, name);
      saveLayout(cloned);
      loadLayouts();
      toast.success(`Cloned layout: ${name}`);
    }
  };

  const handleDeleteLayout = (layout: Layout) => {
    // Don't allow deleting presets
    const isPreset = PRESET_LAYOUTS.some(p => p.id === layout.id);
    if (isPreset) {
      toast.error('Cannot delete preset layouts');
      return;
    }

    if (confirm(`Delete layout "${layout.name}"?`)) {
      deleteLayout(layout.id);
      loadLayouts();
      toast.success(`Deleted layout: ${layout.name}`);
    }
  };

  const handleExportLayout = (layout: Layout) => {
    const json = exportLayout(layout);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${layout.name.toLowerCase().replace(/\s+/g, '-')}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success('Layout exported');
  };

  const handleImportLayout = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          try {
            const json = e.target?.result as string;
            const layout = importLayout(json);
            saveLayout(layout);
            loadLayouts();
            toast.success(`Imported layout: ${layout.name}`);
          } catch (error) {
            toast.error('Failed to import layout');
          }
        };
        reader.readAsText(file);
      }
    };
    input.click();
  };

  const getWidgetInfo = (type: string) => {
    return WIDGET_TYPES.find(w => w.id === type) || { name: type, icon: '📱', color: 'bg-gray-100' };
  };

  return (
    <div className="p-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Multi-Monitor Layout Manager</h1>
          <p className="text-gray-600">
            Professional multi-screen trading workspace with preset and custom layouts.
            Detected monitors: <strong>{monitorCount}</strong>
          </p>
        </div>

        {/* Actions Bar */}
        <div className="flex gap-4 mb-6">
          <button
            onClick={() => setShowCreateDialog(true)}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
          >
            + Create Layout
          </button>
          <button
            onClick={handleImportLayout}
            className="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 transition-colors"
          >
            Import Layout
          </button>
          {activeLayout && (
            <button
              onClick={() => handleOpenLayout(activeLayout)}
              className="bg-purple-600 text-white px-6 py-2 rounded-lg hover:bg-purple-700 transition-colors font-bold"
            >
              🚀 Open Active Layout
            </button>
          )}
        </div>

        {/* Active Layout Info */}
        {activeLayout && (
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg p-6 mb-6">
            <div className="flex justify-between items-start">
              <div>
                <h2 className="text-xl font-semibold text-gray-900 mb-1">
                  Active Layout: {activeLayout.name}
                </h2>
                <p className="text-gray-600 mb-3">{activeLayout.description}</p>
                <div className="flex gap-2 flex-wrap">
                  {activeLayout.widgets.map(widget => {
                    const info = getWidgetInfo(widget.type);
                    return (
                      <span
                        key={widget.id}
                        className={`${info.color} px-3 py-1 rounded text-sm font-medium`}
                      >
                        {info.icon} {info.name} (M{widget.monitor + 1})
                      </span>
                    );
                  })}
                </div>
              </div>
              <button
                onClick={() => handleOpenLayout(activeLayout)}
                className="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 transition-colors whitespace-nowrap"
              >
                Open Now
              </button>
            </div>
          </div>
        )}

        {/* Preset Layouts */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Preset Layouts</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {PRESET_LAYOUTS.map(layout => (
              <div
                key={layout.id}
                className={`bg-white rounded-lg shadow p-6 border-2 transition-all ${
                  activeLayout?.id === layout.id
                    ? 'border-blue-500 shadow-lg'
                    : 'border-transparent hover:border-gray-300'
                }`}
              >
                <div className="flex justify-between items-start mb-3">
                  <h3 className="text-lg font-semibold text-gray-900">{layout.name}</h3>
                  {activeLayout?.id === layout.id && (
                    <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs font-bold">
                      ACTIVE
                    </span>
                  )}
                </div>
                <p className="text-sm text-gray-600 mb-4">{layout.description}</p>

                <div className="mb-4">
                  <div className="text-xs text-gray-500 mb-2">
                    {layout.widgets.length} widgets across {Math.max(...layout.widgets.map(w => w.monitor)) + 1} monitor(s)
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {layout.widgets.slice(0, 6).map(widget => {
                      const info = getWidgetInfo(widget.type);
                      return (
                        <span
                          key={widget.id}
                          className={`${info.color} px-2 py-1 rounded text-xs`}
                          title={`${info.name} on Monitor ${widget.monitor + 1}`}
                        >
                          {info.icon}
                        </span>
                      );
                    })}
                    {layout.widgets.length > 6 && (
                      <span className="bg-gray-100 px-2 py-1 rounded text-xs">
                        +{layout.widgets.length - 6}
                      </span>
                    )}
                  </div>
                </div>

                <div className="flex gap-2">
                  <button
                    onClick={() => handleActivateLayout(layout)}
                    className="flex-1 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-colors text-sm"
                  >
                    Activate
                  </button>
                  <button
                    onClick={() => handleCloneLayout(layout)}
                    className="bg-gray-200 text-gray-700 px-4 py-2 rounded hover:bg-gray-300 transition-colors text-sm"
                  >
                    Clone
                  </button>
                  <button
                    onClick={() => handleExportLayout(layout)}
                    className="bg-gray-200 text-gray-700 px-4 py-2 rounded hover:bg-gray-300 transition-colors text-sm"
                  >
                    Export
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Custom Layouts */}
        {layouts.filter(l => !PRESET_LAYOUTS.some(p => p.id === l.id)).length > 0 && (
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Custom Layouts</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {layouts
                .filter(l => !PRESET_LAYOUTS.some(p => p.id === l.id))
                .map(layout => (
                  <div
                    key={layout.id}
                    className={`bg-white rounded-lg shadow p-6 border-2 transition-all ${
                      activeLayout?.id === layout.id
                        ? 'border-blue-500 shadow-lg'
                        : 'border-transparent hover:border-gray-300'
                    }`}
                  >
                    <div className="flex justify-between items-start mb-3">
                      <h3 className="text-lg font-semibold text-gray-900">{layout.name}</h3>
                      {activeLayout?.id === layout.id && (
                        <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs font-bold">
                          ACTIVE
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-gray-600 mb-4">{layout.description}</p>

                    <div className="mb-4">
                      <div className="text-xs text-gray-500 mb-2">
                        {layout.widgets.length} widgets
                      </div>
                    </div>

                    <div className="flex gap-2">
                      <button
                        onClick={() => handleActivateLayout(layout)}
                        className="flex-1 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-colors text-sm"
                      >
                        Activate
                      </button>
                      <button
                        onClick={() => handleExportLayout(layout)}
                        className="bg-gray-200 text-gray-700 px-4 py-2 rounded hover:bg-gray-300 transition-colors text-sm"
                      >
                        Export
                      </button>
                      <button
                        onClick={() => handleDeleteLayout(layout)}
                        className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700 transition-colors text-sm"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* Create Layout Dialog */}
        {showCreateDialog && (
          <div className="fixed inset-0 z-50 flex items-center justify-center px-4 bg-black bg-opacity-50">
            <div className="bg-white rounded-lg shadow-xl p-6 max-w-md w-full">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">Create New Layout</h2>

              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Layout Name
                </label>
                <input
                  type="text"
                  value={newLayoutName}
                  onChange={(e) => setNewLayoutName(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  placeholder="My Custom Layout"
                />
              </div>

              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Description
                </label>
                <textarea
                  value={newLayoutDescription}
                  onChange={(e) => setNewLayoutDescription(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                  rows={3}
                  placeholder="Description of this layout..."
                />
              </div>

              <div className="flex gap-3">
                <button
                  onClick={handleCreateLayout}
                  className="flex-1 bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Create
                </button>
                <button
                  onClick={() => {
                    setShowCreateDialog(false);
                    setNewLayoutName('');
                    setNewLayoutDescription('');
                  }}
                  className="flex-1 bg-gray-200 text-gray-700 px-6 py-2 rounded-lg hover:bg-gray-300 transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Info Panel */}
        <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-blue-900 mb-3">💡 Multi-Monitor Setup Tips</h3>
          <ul className="space-y-2 text-sm text-blue-800">
            <li className="flex items-start">
              <span className="mr-2">•</span>
              <span>Choose a preset layout that matches your trading style or create a custom one</span>
            </li>
            <li className="flex items-start">
              <span className="mr-2">•</span>
              <span>Click "Open Active Layout" to launch all windows across your monitors</span>
            </li>
            <li className="flex items-start">
              <span className="mr-2">•</span>
              <span>Each widget opens in a separate browser window that you can position on any monitor</span>
            </li>
            <li className="flex items-start">
              <span className="mr-2">•</span>
              <span>Export layouts to share with teammates or backup your custom setups</span>
            </li>
            <li className="flex items-start">
              <span className="mr-2">•</span>
              <span>Clone preset layouts to customize them without losing the original</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
