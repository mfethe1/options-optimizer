/**
 * 3D Volatility Surface Visualization
 *
 * Interactive 3D surface plot for visualizing:
 * - Implied volatility surface (IV by strike/expiry)
 * - Option Greeks surfaces (Delta, Gamma, Vega, Theta)
 * - PINN (Physics-Informed Neural Network) predictions
 *
 * Uses Plotly.js for 3D rendering
 *
 * Features:
 * - Rotate, zoom, pan 3D surface
 * - Color gradient by value
 * - Hover tooltips with exact values
 * - Multiple surface types (IV, Greeks)
 * - Professional color scales
 *
 * Phase 2 - Advanced ML Visualization
 *
 * NOTE: Requires plotly.js-dist-min package
 * Install: npm install plotly.js-dist-min
 * Or: npm install react-plotly.js plotly.js
 */

import React, { useEffect, useRef } from 'react';
import { Box, Typography, Select, MenuItem, FormControl, InputLabel, Chip } from '@mui/material';

export interface Surface3DData {
  x: number[]; // Strike prices or time to expiry
  y: number[]; // Time to expiry or moneyness
  z: number[][]; // Values (IV, Delta, Gamma, etc.) - 2D array [y.length][x.length]
  xlabel?: string;
  ylabel?: string;
  zlabel?: string;
}

export interface VolatilitySurface3DProps {
  data: Surface3DData;
  surfaceType?: 'implied_volatility' | 'delta' | 'gamma' | 'vega' | 'theta' | 'rho';
  title?: string;
  width?: number;
  height?: number;
  theme?: 'dark' | 'light';
  colorscale?: string; // Plotly colorscale name
  className?: string;
}

/**
 * Color scales for different surface types
 */
const SURFACE_COLORSCALES: Record<string, string> = {
  implied_volatility: 'Viridis',
  delta: 'RdYlGn',
  gamma: 'Blues',
  vega: 'Oranges',
  theta: 'Reds',
  rho: 'Purples',
  default: 'Jet',
};

/**
 * Labels and descriptions
 */
const SURFACE_INFO: Record<string, { name: string; description: string; unit: string }> = {
  implied_volatility: {
    name: 'Implied Volatility Surface',
    description: 'Market-implied volatility across strikes and expirations',
    unit: '%',
  },
  delta: {
    name: 'Delta Surface',
    description: 'Rate of change in option price relative to underlying price',
    unit: '',
  },
  gamma: {
    name: 'Gamma Surface',
    description: 'Rate of change in delta relative to underlying price',
    unit: '',
  },
  vega: {
    name: 'Vega Surface',
    description: 'Sensitivity to changes in implied volatility',
    unit: '$',
  },
  theta: {
    name: 'Theta Surface',
    description: 'Time decay - daily loss in option value',
    unit: '$/day',
  },
  rho: {
    name: 'Rho Surface',
    description: 'Sensitivity to changes in interest rates',
    unit: '',
  },
};

/**
 * Volatility Surface 3D Chart
 *
 * Usage:
 * ```tsx
 * <VolatilitySurface3D
 *   data={{
 *     x: [90, 95, 100, 105, 110], // Strikes
 *     y: [7, 14, 30, 60, 90], // Days to expiry
 *     z: [ // IV values
 *       [0.25, 0.22, 0.20, 0.22, 0.25],
 *       [0.28, 0.24, 0.21, 0.24, 0.28],
 *       [0.32, 0.27, 0.23, 0.27, 0.32],
 *       [0.35, 0.30, 0.25, 0.30, 0.35],
 *       [0.38, 0.32, 0.27, 0.32, 0.38],
 *     ],
 *     xlabel: 'Strike Price',
 *     ylabel: 'Days to Expiry',
 *     zlabel: 'Implied Volatility'
 *   }}
 *   surfaceType="implied_volatility"
 *   theme="dark"
 * />
 * ```
 *
 * NOTE: This component requires plotly.js to be installed.
 * If plotly.js is not available, a fallback message is shown.
 */
const VolatilitySurface3D: React.FC<VolatilitySurface3DProps> = ({
  data,
  surfaceType = 'implied_volatility',
  title,
  width = 800,
  height = 600,
  theme = 'dark',
  colorscale,
  className = '',
}) => {
  const plotRef = useRef<HTMLDivElement>(null);
  const [plotlyAvailable, setPlotlyAvailable] = React.useState<boolean>(false);
  const [selectedView, setSelectedView] = React.useState<'3d' | '2d'>('3d');

  /**
   * Check if Plotly is available
   */
  useEffect(() => {
    // Check if Plotly is loaded
    if (typeof window !== 'undefined' && (window as any).Plotly) {
      setPlotlyAvailable(true);
    }
  }, []);

  /**
   * Render 3D surface plot
   */
  useEffect(() => {
    if (!plotlyAvailable || !plotRef.current) return;

    const Plotly = (window as any).Plotly;

    const surfaceInfo = SURFACE_INFO[surfaceType] || SURFACE_INFO.implied_volatility;

    const plotData = [{
      type: 'surface',
      x: data.x,
      y: data.y,
      z: data.z,
      colorscale: colorscale || SURFACE_COLORSCALES[surfaceType] || SURFACE_COLORSCALES.default,
      colorbar: {
        title: surfaceInfo.unit,
        titleside: 'right',
        titlefont: {
          color: theme === 'dark' ? '#d1d4dc' : '#191919',
        },
        tickfont: {
          color: theme === 'dark' ? '#d1d4dc' : '#191919',
        },
      },
      hovertemplate:
        `<b>${data.xlabel || 'X'}</b>: %{x}<br>` +
        `<b>${data.ylabel || 'Y'}</b>: %{y}<br>` +
        `<b>${data.zlabel || 'Value'}</b>: %{z:.4f}${surfaceInfo.unit}<extra></extra>`,
    }];

    const layout = {
      title: {
        text: title || surfaceInfo.name,
        font: {
          color: theme === 'dark' ? '#d1d4dc' : '#191919',
          size: 18,
        },
      },
      paper_bgcolor: theme === 'dark' ? '#131722' : '#ffffff',
      plot_bgcolor: theme === 'dark' ? '#1e222d' : '#f0f3fa',
      scene: {
        xaxis: {
          title: data.xlabel || 'Strike Price',
          titlefont: { color: theme === 'dark' ? '#d1d4dc' : '#191919' },
          tickfont: { color: theme === 'dark' ? '#d1d4dc' : '#191919' },
          gridcolor: theme === 'dark' ? '#2a2e39' : '#e1e3eb',
        },
        yaxis: {
          title: data.ylabel || 'Time to Expiry (days)',
          titlefont: { color: theme === 'dark' ? '#d1d4dc' : '#191919' },
          tickfont: { color: theme === 'dark' ? '#d1d4dc' : '#191919' },
          gridcolor: theme === 'dark' ? '#2a2e39' : '#e1e3eb',
        },
        zaxis: {
          title: data.zlabel || surfaceInfo.name,
          titlefont: { color: theme === 'dark' ? '#d1d4dc' : '#191919' },
          tickfont: { color: theme === 'dark' ? '#d1d4dc' : '#191919' },
          gridcolor: theme === 'dark' ? '#2a2e39' : '#e1e3eb',
        },
        camera: {
          eye: { x: 1.5, y: 1.5, z: 1.3 },
        },
      },
      autosize: false,
      width: width,
      height: height,
      margin: { l: 0, r: 0, t: 40, b: 0 },
    };

    const config = {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['toImage'],
      displaylogo: false,
    };

    Plotly.newPlot(plotRef.current, plotData, layout, config);

    return () => {
      if (plotRef.current) {
        Plotly.purge(plotRef.current);
      }
    };
  }, [data, surfaceType, theme, width, height, title, colorscale, plotlyAvailable]);

  const surfaceInfo = SURFACE_INFO[surfaceType] || SURFACE_INFO.implied_volatility;

  return (
    <div className={`volatility-surface-3d ${className}`}>
      {/* Header */}
      <Box
        sx={{
          padding: '16px',
          backgroundColor: theme === 'dark' ? '#1e222d' : '#f0f3fa',
          color: theme === 'dark' ? '#d1d4dc' : '#191919',
          borderRadius: '8px 8px 0 0',
          marginBottom: '2px',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <div>
            <Typography variant="h6">{surfaceInfo.name}</Typography>
            <Typography variant="caption" style={{ opacity: 0.7 }}>
              {surfaceInfo.description}
            </Typography>
          </div>
          <Chip
            label={surfaceType.toUpperCase().replace('_', ' ')}
            color="primary"
            size="small"
          />
        </div>

        {/* Surface stats */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '12px', fontSize: '12px', marginTop: '12px' }}>
          <div>
            <div style={{ opacity: 0.7 }}>X Points</div>
            <div style={{ fontSize: '18px', fontWeight: 600 }}>{data.x.length}</div>
          </div>
          <div>
            <div style={{ opacity: 0.7 }}>Y Points</div>
            <div style={{ fontSize: '18px', fontWeight: 600 }}>{data.y.length}</div>
          </div>
          <div>
            <div style={{ opacity: 0.7 }}>Total Points</div>
            <div style={{ fontSize: '18px', fontWeight: 600 }}>{data.x.length * data.y.length}</div>
          </div>
          <div>
            <div style={{ opacity: 0.7 }}>Min Value</div>
            <div style={{ fontSize: '18px', fontWeight: 600 }}>
              {Math.min(...data.z.flat()).toFixed(3)}
              {surfaceInfo.unit}
            </div>
          </div>
          <div>
            <div style={{ opacity: 0.7 }}>Max Value</div>
            <div style={{ fontSize: '18px', fontWeight: 600 }}>
              {Math.max(...data.z.flat()).toFixed(3)}
              {surfaceInfo.unit}
            </div>
          </div>
        </div>
      </Box>

      {/* Plot Container */}
      <Box
        sx={{
          backgroundColor: theme === 'dark' ? '#131722' : '#ffffff',
          border: `1px solid ${theme === 'dark' ? '#2a2e39' : '#e1e3eb'}`,
          position: 'relative',
        }}
      >
        {plotlyAvailable ? (
          <div ref={plotRef} />
        ) : (
          <Box
            sx={{
              height: height,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexDirection: 'column',
              gap: 2,
              padding: 4,
            }}
          >
            <Typography variant="h6" color="textSecondary">
              📊 3D Visualization Unavailable
            </Typography>
            <Typography variant="body2" color="textSecondary" style={{ textAlign: 'center', maxWidth: 500 }}>
              This component requires <code>plotly.js</code> to be installed.
              <br />
              <br />
              Install: <code>npm install plotly.js-dist-min</code>
              <br />
              Or: <code>npm install react-plotly.js plotly.js</code>
              <br />
              <br />
              Once installed, reload the page to see the 3D volatility surface.
            </Typography>
            <Chip label="Fallback: 2D Heatmap" color="warning" />
          </Box>
        )}
      </Box>

      {/* Info Footer */}
      <Box
        sx={{
          padding: '16px',
          backgroundColor: theme === 'dark' ? '#1e222d' : '#f0f3fa',
          color: theme === 'dark' ? '#d1d4dc' : '#191919',
          borderRadius: '0 0 8px 8px',
          marginTop: '2px',
        }}
      >
        <Typography variant="subtitle2" gutterBottom style={{ fontWeight: 600, fontSize: '13px' }}>
          📖 How to Use:
        </Typography>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px', fontSize: '12px', marginTop: '8px' }}>
          <div>
            <strong>Rotate:</strong> Click and drag
          </div>
          <div>
            <strong>Zoom:</strong> Scroll or pinch
          </div>
          <div>
            <strong>Pan:</strong> Right-click drag
          </div>
          <div>
            <strong>Hover:</strong> See exact values
          </div>
          <div>
            <strong>Reset:</strong> Double-click
          </div>
          <div>
            <strong>Download:</strong> Camera icon
          </div>
        </div>
      </Box>
    </div>
  );
};

export default VolatilitySurface3D;

/**
 * Helper: Generate sample implied volatility surface
 */
export function generateSampleIVSurface(): Surface3DData {
  // Strikes around ATM (100)
  const strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120];

  // Days to expiry
  const expirations = [7, 14, 30, 60, 90, 180, 365];

  // Generate IV surface (volatility smile + term structure)
  const ivSurface: number[][] = [];

  for (let i = 0; i < expirations.length; i++) {
    const row: number[] = [];
    const termStructure = 0.18 + i * 0.02; // Base vol increases with time

    for (let j = 0; j < strikes.length; j++) {
      const moneyness = strikes[j] / 100; // ATM = 1.0

      // Volatility smile (higher for OTM/ITM)
      const smileEffect = 0.05 * Math.pow(moneyness - 1.0, 2);

      const iv = termStructure + smileEffect;
      row.push(iv);
    }

    ivSurface.push(row);
  }

  return {
    x: strikes,
    y: expirations,
    z: ivSurface,
    xlabel: 'Strike Price ($)',
    ylabel: 'Days to Expiry',
    zlabel: 'Implied Volatility',
  };
}

/**
 * Helper: Generate sample Greeks surface
 */
export function generateSampleGreeksSurface(greekType: 'delta' | 'gamma' | 'vega' | 'theta'): Surface3DData {
  const strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120];
  const expirations = [7, 14, 30, 60, 90, 180, 365];

  const greeksSurface: number[][] = [];

  for (let i = 0; i < expirations.length; i++) {
    const row: number[] = [];
    const dte = expirations[i];

    for (let j = 0; j < strikes.length; j++) {
      const moneyness = strikes[j] / 100;

      let value = 0;
      if (greekType === 'delta') {
        // Delta: 0 (OTM) to 1 (ITM), centered at 0.5 (ATM)
        value = 1 / (1 + Math.exp(-5 * (moneyness - 1)));
      } else if (greekType === 'gamma') {
        // Gamma: Peaks at ATM, higher for shorter expirations
        value = (0.05 / Math.sqrt(dte / 365)) * Math.exp(-50 * Math.pow(moneyness - 1, 2));
      } else if (greekType === 'vega') {
        // Vega: Increases with time to expiry, peaks at ATM
        value = Math.sqrt(dte / 365) * 0.2 * Math.exp(-10 * Math.pow(moneyness - 1, 2));
      } else if (greekType === 'theta') {
        // Theta: Negative (time decay), accelerates near expiry
        value = -(0.1 / Math.sqrt(dte / 365)) * Math.exp(-5 * Math.pow(moneyness - 1, 2));
      }

      row.push(value);
    }

    greeksSurface.push(row);
  }

  return {
    x: strikes,
    y: expirations,
    z: greeksSurface,
    xlabel: 'Strike Price ($)',
    ylabel: 'Days to Expiry',
    zlabel: greekType.charAt(0).toUpperCase() + greekType.slice(1),
  };
}
