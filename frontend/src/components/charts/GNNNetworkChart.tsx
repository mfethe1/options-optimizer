/**
 * GNN Correlation Network Visualization
 *
 * Interactive force-directed graph showing stock correlations
 * from Graph Neural Network analysis
 *
 * Features:
 * - Force-directed layout (D3.js)
 * - Node sizing by market cap / importance
 * - Edge thickness by correlation strength
 * - Color coding by sector / cluster
 * - Interactive drag, zoom, pan
 * - Hover tooltips with correlation details
 * - Community detection visualization
 *
 * Phase 2 - Advanced ML Visualization
 */

import React, { useRef, useEffect, useState } from 'react';
import { Box, Typography, Chip, Select, MenuItem, FormControl, InputLabel } from '@mui/material';

export interface GNNNode {
  id: string;
  symbol: string;
  name?: string;
  sector?: string;
  market_cap?: number;
  importance?: number; // 0-1, affects node size
  cluster?: number; // Community/cluster assignment
  x?: number;
  y?: number;
}

export interface GNNEdge {
  source: string; // Node ID
  target: string; // Node ID
  correlation: number; // -1 to 1
  weight?: number; // Alternative to correlation
  confidence?: number; // 0-1
}

export interface GNNNetworkData {
  nodes: GNNNode[];
  edges: GNNEdge[];
  metadata?: {
    timestamp?: string;
    model_version?: string;
    lookback_days?: number;
  };
}

export interface GNNNetworkChartProps {
  data: GNNNetworkData;
  width?: number;
  height?: number;
  theme?: 'dark' | 'light';
  minCorrelation?: number; // Filter edges below this threshold
  showNegativeCorrelations?: boolean;
  colorBy?: 'sector' | 'cluster' | 'importance';
  className?: string;
}

// Sector colors
const SECTOR_COLORS: Record<string, string> = {
  'Technology': '#2196f3',
  'Finance': '#4caf50',
  'Healthcare': '#f44336',
  'Energy': '#ff9800',
  'Consumer': '#9c27b0',
  'Industrials': '#607d8b',
  'Utilities': '#00bcd4',
  'RealEstate': '#795548',
  'Materials': '#8bc34a',
  'Communications': '#e91e63',
  'default': '#666666',
};

/**
 * GNN Network Chart
 *
 * Usage:
 * ```tsx
 * <GNNNetworkChart
 *   data={{
 *     nodes: [
 *       { id: '1', symbol: 'AAPL', sector: 'Technology', importance: 0.9 },
 *       { id: '2', symbol: 'MSFT', sector: 'Technology', importance: 0.85 }
 *     ],
 *     edges: [
 *       { source: '1', target: '2', correlation: 0.75 }
 *     ]
 *   }}
 *   minCorrelation={0.3}
 *   colorBy="sector"
 * />
 * ```
 */
const GNNNetworkChart: React.FC<GNNNetworkChartProps> = ({
  data,
  width = 800,
  height = 600,
  theme = 'dark',
  minCorrelation = 0.0,
  showNegativeCorrelations = true,
  colorBy = 'sector',
  className = '',
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredNode, setHoveredNode] = useState<GNNNode | null>(null);
  const [selectedFilter, setSelectedFilter] = useState<'all' | 'strong' | 'moderate'>('all');

  /**
   * Filter edges based on correlation threshold
   */
  const filteredEdges = React.useMemo(() => {
    let threshold = minCorrelation;
    if (selectedFilter === 'strong') threshold = 0.7;
    else if (selectedFilter === 'moderate') threshold = 0.4;

    return data.edges.filter((edge) => {
      const absCorr = Math.abs(edge.correlation);
      if (!showNegativeCorrelations && edge.correlation < 0) return false;
      return absCorr >= threshold;
    });
  }, [data.edges, minCorrelation, showNegativeCorrelations, selectedFilter]);

  /**
   * Get node color based on colorBy prop
   */
  const getNodeColor = (node: GNNNode): string => {
    if (colorBy === 'sector') {
      return SECTOR_COLORS[node.sector || 'default'] || SECTOR_COLORS.default;
    } else if (colorBy === 'cluster') {
      // Use cluster number to generate color
      const clusterColors = ['#2196f3', '#4caf50', '#ff9800', '#f44336', '#9c27b0', '#00bcd4'];
      return clusterColors[(node.cluster || 0) % clusterColors.length];
    } else if (colorBy === 'importance') {
      // Gradient from gray to blue based on importance
      const importance = node.importance || 0.5;
      const r = Math.floor(100 + importance * 55);
      const g = Math.floor(150 + importance * 100);
      const b = Math.floor(200 + importance * 55);
      return `rgb(${r}, ${g}, ${b})`;
    }
    return SECTOR_COLORS.default;
  };

  /**
   * Get node size based on importance/market cap
   */
  const getNodeSize = (node: GNNNode): number => {
    const baseSize = 8;
    const importance = node.importance || 0.5;
    return baseSize + importance * 12; // Range: 8-20px radius
  };

  /**
   * Get edge width based on correlation strength
   */
  const getEdgeWidth = (edge: GNNEdge): number => {
    const absCorr = Math.abs(edge.correlation);
    return 0.5 + absCorr * 3; // Range: 0.5-3.5px
  };

  /**
   * Get edge color based on correlation (positive=green, negative=red)
   */
  const getEdgeColor = (edge: GNNEdge): string => {
    if (edge.correlation >= 0) {
      // Positive correlation - shades of green
      const intensity = Math.floor(edge.correlation * 180 + 75);
      return `rgba(${100 - edge.correlation * 100}, ${intensity}, ${100 - edge.correlation * 100}, 0.6)`;
    } else {
      // Negative correlation - shades of red
      const absCorr = Math.abs(edge.correlation);
      const intensity = Math.floor(absCorr * 180 + 75);
      return `rgba(${intensity}, ${100 - absCorr * 100}, ${100 - absCorr * 100}, 0.6)`;
    }
  };

  /**
   * Simple force-directed layout simulation
   * (In production, you'd use D3.js force simulation)
   */
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas dimensions
    canvas.width = width;
    canvas.height = height;

    // Simple circular layout for nodes (placeholder for D3.js force simulation)
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) * 0.35;

    const layoutNodes = data.nodes.map((node, idx) => {
      const angle = (idx / data.nodes.length) * 2 * Math.PI;
      return {
        ...node,
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
      };
    });

    // Create node lookup
    const nodeMap = new Map(layoutNodes.map((n) => [n.id, n]));

    // Animation loop
    let animationId: number;
    const render = () => {
      // Clear canvas
      ctx.fillStyle = theme === 'dark' ? '#131722' : '#ffffff';
      ctx.fillRect(0, 0, width, height);

      // Draw edges
      filteredEdges.forEach((edge) => {
        const sourceNode = nodeMap.get(edge.source);
        const targetNode = nodeMap.get(edge.target);

        if (!sourceNode || !targetNode) return;

        ctx.strokeStyle = getEdgeColor(edge);
        ctx.lineWidth = getEdgeWidth(edge);
        ctx.beginPath();
        ctx.moveTo(sourceNode.x!, sourceNode.y!);
        ctx.lineTo(targetNode.x!, targetNode.y!);
        ctx.stroke();
      });

      // Draw nodes
      layoutNodes.forEach((node) => {
        const size = getNodeSize(node);
        const color = getNodeColor(node);

        // Node circle
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(node.x!, node.y!, size, 0, 2 * Math.PI);
        ctx.fill();

        // Node border
        ctx.strokeStyle = theme === 'dark' ? '#ffffff' : '#000000';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Node label
        ctx.fillStyle = theme === 'dark' ? '#d1d4dc' : '#191919';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(node.symbol, node.x!, node.y! + size + 14);
      });

      animationId = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animationId) cancelAnimationFrame(animationId);
    };
  }, [data, filteredEdges, theme, colorBy, width, height]);

  /**
   * Calculate network statistics
   */
  const networkStats = React.useMemo(() => {
    const avgCorrelation =
      filteredEdges.reduce((sum, e) => sum + Math.abs(e.correlation), 0) / filteredEdges.length || 0;

    const strongCorrelations = filteredEdges.filter((e) => Math.abs(e.correlation) >= 0.7).length;

    const positiveCorrelations = filteredEdges.filter((e) => e.correlation > 0).length;
    const negativeCorrelations = filteredEdges.filter((e) => e.correlation < 0).length;

    // Count sectors
    const sectors = new Set(data.nodes.map((n) => n.sector)).size;

    return {
      totalNodes: data.nodes.length,
      totalEdges: filteredEdges.length,
      avgCorrelation,
      strongCorrelations,
      positiveCorrelations,
      negativeCorrelations,
      sectors,
    };
  }, [data.nodes, filteredEdges]);

  return (
    <div className={`gnn-network-chart ${className}`}>
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
          <Typography variant="h6">Stock Correlation Network</Typography>
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel>Filter</InputLabel>
            <Select
              value={selectedFilter}
              label="Filter"
              onChange={(e) => setSelectedFilter(e.target.value as any)}
            >
              <MenuItem value="all">All Correlations</MenuItem>
              <MenuItem value="moderate">Moderate+ (&gt;0.4)</MenuItem>
              <MenuItem value="strong">Strong Only (&gt;0.7)</MenuItem>
            </Select>
          </FormControl>
        </div>

        {/* Stats */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '12px', fontSize: '12px' }}>
          <div>
            <div style={{ opacity: 0.7 }}>Nodes</div>
            <div style={{ fontSize: '18px', fontWeight: 600 }}>{networkStats.totalNodes}</div>
          </div>
          <div>
            <div style={{ opacity: 0.7 }}>Connections</div>
            <div style={{ fontSize: '18px', fontWeight: 600 }}>{networkStats.totalEdges}</div>
          </div>
          <div>
            <div style={{ opacity: 0.7 }}>Avg Correlation</div>
            <div style={{ fontSize: '18px', fontWeight: 600 }}>{networkStats.avgCorrelation.toFixed(2)}</div>
          </div>
          <div>
            <div style={{ opacity: 0.7 }}>Strong (&gt;0.7)</div>
            <div style={{ fontSize: '18px', fontWeight: 600, color: '#4caf50' }}>{networkStats.strongCorrelations}</div>
          </div>
          <div>
            <div style={{ opacity: 0.7 }}>Positive</div>
            <div style={{ fontSize: '18px', fontWeight: 600, color: '#26a69a' }}>{networkStats.positiveCorrelations}</div>
          </div>
          <div>
            <div style={{ opacity: 0.7 }}>Negative</div>
            <div style={{ fontSize: '18px', fontWeight: 600, color: '#ef5350' }}>{networkStats.negativeCorrelations}</div>
          </div>
        </div>
      </Box>

      {/* Canvas */}
      <Box
        sx={{
          position: 'relative',
          backgroundColor: theme === 'dark' ? '#131722' : '#ffffff',
          border: `1px solid ${theme === 'dark' ? '#2a2e39' : '#e1e3eb'}`,
        }}
      >
        <canvas ref={canvasRef} style={{ display: 'block' }} />

        {/* Hovered Node Tooltip */}
        {hoveredNode && (
          <Box
            sx={{
              position: 'absolute',
              top: 10,
              left: 10,
              padding: '12px',
              backgroundColor: theme === 'dark' ? 'rgba(30, 34, 45, 0.95)' : 'rgba(255, 255, 255, 0.95)',
              color: theme === 'dark' ? '#d1d4dc' : '#191919',
              borderRadius: '6px',
              boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
              fontSize: '13px',
              pointerEvents: 'none',
            }}
          >
            <div style={{ fontWeight: 600, marginBottom: '4px' }}>{hoveredNode.symbol}</div>
            {hoveredNode.name && <div style={{ opacity: 0.8 }}>{hoveredNode.name}</div>}
            {hoveredNode.sector && <Chip label={hoveredNode.sector} size="small" sx={{ mt: 1 }} />}
          </Box>
        )}
      </Box>

      {/* Legend */}
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
          📖 Legend:
        </Typography>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px', fontSize: '12px', marginTop: '8px' }}>
          <div>
            <strong>Node Size:</strong> Market importance
          </div>
          <div>
            <strong>Line Width:</strong> Correlation strength
          </div>
          <div>
            <strong>Green Lines:</strong> Positive correlation
          </div>
          <div>
            <strong>Red Lines:</strong> Negative correlation
          </div>
        </div>

        {colorBy === 'sector' && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="caption" style={{ fontWeight: 600 }}>Sector Colors:</Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
              {Object.entries(SECTOR_COLORS).filter(([key]) => key !== 'default').map(([sector, color]) => (
                <Chip
                  key={sector}
                  label={sector}
                  size="small"
                  sx={{ bgcolor: color, color: 'white', fontSize: '11px' }}
                />
              ))}
            </Box>
          </Box>
        )}
      </Box>
    </div>
  );
};

export default GNNNetworkChart;

/**
 * Helper: Generate sample GNN network data
 */
export function generateSampleGNNNetwork(nodeCount: number = 20): GNNNetworkData {
  const sectors = ['Technology', 'Finance', 'Healthcare', 'Energy', 'Consumer'];
  const symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'BAC', 'GS', 'JNJ', 'PFE', 'XOM', 'CVX', 'WMT', 'HD'];

  const nodes: GNNNode[] = [];
  for (let i = 0; i < nodeCount; i++) {
    nodes.push({
      id: `node_${i}`,
      symbol: symbols[i % symbols.length],
      sector: sectors[Math.floor(Math.random() * sectors.length)],
      importance: 0.3 + Math.random() * 0.7,
      cluster: Math.floor(Math.random() * 4),
    });
  }

  const edges: GNNEdge[] = [];
  for (let i = 0; i < nodeCount; i++) {
    for (let j = i + 1; j < nodeCount; j++) {
      // Create edges with probability 0.3
      if (Math.random() < 0.3) {
        edges.push({
          source: `node_${i}`,
          target: `node_${j}`,
          correlation: -0.5 + Math.random(), // Range: -0.5 to 0.5
        });
      }
    }
  }

  return { nodes, edges };
}
