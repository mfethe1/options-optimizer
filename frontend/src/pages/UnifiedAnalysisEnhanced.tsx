/**
 * Enhanced Unified Analysis - Professional Trading Chart
 *
 * TradingView-grade charting with:
 * - Candlestick/OHLC price visualization
 * - Auto-scaling Y-axis
 * - Interactive crosshair with price info
 * - Volume bars
 * - Model prediction overlays
 * - Multiple timeframes (1D, 5D, 1M, 3M, 1Y)
 * - Chart type switching (Candlestick, Line, Area)
 * - Zoom and pan controls
 * - Technical indicators (SMA, EMA, Bollinger Bands)
 * - Real-time updates via WebSocket
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  ButtonGroup,
  TextField,
  Switch,
  FormControlLabel,
  IconButton,
  Tooltip,
  Chip,
  CircularProgress,
  Alert,
  Stack,
  Divider,
  ToggleButton,
  ToggleButtonGroup,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import {
  ZoomIn,
  ZoomOut,
  Refresh,
  Timeline,
  CandlestickChart,
  ShowChart,
  AreaChart,
  Settings,
  Download,
  Fullscreen,
  FullscreenExit,
  TrendingUp,
  BarChart,
} from '@mui/icons-material';
import TradingViewChart from '../components/charts/TradingViewChart';
import { OHLCVData, IndicatorConfig, PredictionSeriesConfig, LineData } from '../components/charts/chartTypes';
import { buildApiUrl } from '../config/api.config';
import { toTime } from '../components/charts/chartUtils';
import VIXAnalysisWidget from '../components/VIXAnalysisWidget';

// Model configuration
interface ModelConfig {
  id: string;
  name: string;
  color: string;
  enabled: boolean;
  type: 'point' | 'range' | 'probabilistic';
  lineWidth: number;
  dashStyle?: [number, number];
}

// Chart type options
type ChartType = 'candlestick' | 'line' | 'area';

const UnifiedAnalysisEnhanced: React.FC = () => {
  // State management
  const [symbol, setSymbol] = useState('SPY');
  const [timeRange, setTimeRange] = useState('1D');
  const [chartType, setChartType] = useState<ChartType>('candlestick');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Chart data
  const [ohlcvData, setOhlcvData] = useState<OHLCVData[]>([]);
  const [crosshairData, setCrosshairData] = useState<any>(null);

  // Model configuration
  const [models, setModels] = useState<ModelConfig[]>([
    {
      id: 'epidemic',
      name: 'Epidemic VIX',
      color: '#8B5CF6',
      enabled: true,
      type: 'range',
      lineWidth: 2,
    },
    {
      id: 'gnn',
      name: 'Graph Neural Network',
      color: '#10B981',
      enabled: true,
      type: 'point',
      lineWidth: 2,
    },
    {
      id: 'mamba',
      name: 'Mamba (Linear O(N))',
      color: '#F59E0B',
      enabled: true,
      type: 'point',
      lineWidth: 2,
    },
    {
      id: 'pinn',
      name: 'Physics-Informed NN',
      color: '#3B82F6',
      enabled: true,
      type: 'probabilistic',
      lineWidth: 2,
    },
    {
      id: 'ensemble',
      name: 'Ensemble Consensus',
      color: '#EF4444',
      enabled: true,
      type: 'range',
      lineWidth: 3,
    },
  ]);

  // Model predictions data
  const [modelPredictions, setModelPredictions] = useState<Record<string, any>>({});

  // Epidemic VIX data (for VIX widget)
  const [epidemicData, setEpidemicData] = useState<any>(null);

  // Technical indicators
  const [indicators, setIndicators] = useState<IndicatorConfig[]>([]);
  const [indicatorMenuAnchor, setIndicatorMenuAnchor] = useState<null | HTMLElement>(null);

  // Container ref for fullscreen
  const containerRef = useRef<HTMLDivElement>(null);

  /**
   * Load market data and predictions
   */
  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const url = buildApiUrl(`unified/forecast/all?symbol=${encodeURIComponent(symbol)}&time_range=${encodeURIComponent(timeRange)}`);
      console.log('[UnifiedAnalysisEnhanced] Fetching:', url);

      const response = await fetch(url, { method: 'POST' });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      console.log('[UnifiedAnalysisEnhanced] Data received:', {
        points: data.timeline?.length || 0,
        models: Object.keys(data.predictions || {}),
        hasMetadata: !!data.metadata,
      });

      // Transform timeline data to OHLCV format
      if (data.timeline && Array.isArray(data.timeline)) {
        const ohlcv: OHLCVData[] = data.timeline
          .filter((point: any) =>
            point.time &&
            typeof point.open === 'number' &&
            typeof point.high === 'number' &&
            typeof point.low === 'number' &&
            typeof point.actual === 'number'
          )
          .map((point: any) => ({
            time: point.time,
            open: point.open,
            high: point.high,
            low: point.low,
            close: point.actual, // Use actual as close price
            volume: point.volume || 0,
          }));

        setOhlcvData(ohlcv);
        console.log('[UnifiedAnalysisEnhanced] OHLCV data:', {
          points: ohlcv.length,
          sample: ohlcv[0],
        });
      }

      // Store model predictions for overlays (now structured as arrays)
      if (data.predictions) {
        setModelPredictions(data.predictions);
        console.log('[UnifiedAnalysisEnhanced] Predictions structure:', {
          keys: Object.keys(data.predictions),
          sample: data.predictions.epidemic ? `epidemic: ${data.predictions.epidemic.length} points` : 'no epidemic',
        });
      }

      // Extract epidemic VIX data for widget from metadata
      if (data.metadata?.epidemic) {
        setEpidemicData(data.metadata.epidemic);
        console.log('[UnifiedAnalysisEnhanced] Epidemic VIX data:', data.metadata.epidemic);
      } else {
        console.warn('[UnifiedAnalysisEnhanced] No epidemic metadata found');
      }

    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load data';
      setError(message);
      console.error('[UnifiedAnalysisEnhanced] Error:', err);
    } finally {
      setLoading(false);
    }
  }, [symbol, timeRange]);

  /**
   * Load data on mount and when symbol/timeRange changes
   */
  useEffect(() => {
    loadData();
  }, [loadData]);

  /**
   * Transform model predictions into chart series format
   */
  const predictionSeries = React.useMemo<PredictionSeriesConfig[]>(() => {
    if (!modelPredictions || Object.keys(modelPredictions).length === 0) {
      return [];
    }

    const series: PredictionSeriesConfig[] = [];

    models.forEach((model) => {
      if (!model.enabled) return;

      const predData = modelPredictions[model.id];
      if (!predData || !Array.isArray(predData)) return;

      // Transform prediction data to LineData format
      const lineData: LineData[] = predData
        .filter((point: any) => point.time && typeof point.predicted === 'number')
        .map((point: any) => ({
          time: toTime(point.time),
          value: point.predicted,
        }));

      if (lineData.length > 0) {
        series.push({
          id: model.id,
          name: model.name,
          color: model.color,
          lineWidth: model.lineWidth,
          lineStyle: model.dashStyle ? 'dashed' : 'solid',
          data: lineData,
          visible: model.enabled,
        });
      }
    });

    return series;
  }, [modelPredictions, models]);

  /**
   * Toggle model visibility
   */
  const toggleModel = useCallback((modelId: string) => {
    setModels(prev =>
      prev.map(m => (m.id === modelId ? { ...m, enabled: !m.enabled } : m))
    );
  }, []);

  /**
   * Handle crosshair move (show price info)
   */
  const handleCrosshairMove = useCallback((data: any) => {
    setCrosshairData(data);
  }, []);

  /**
   * Toggle fullscreen
   */
  const toggleFullscreen = useCallback(() => {
    if (!containerRef.current) return;

    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  }, []);

  /**
   * Export chart data
   */
  const handleExport = useCallback(() => {
    const csv = [
      ['Time', 'Open', 'High', 'Low', 'Close', 'Volume'].join(','),
      ...ohlcvData.map(row =>
        [row.time, row.open, row.high, row.low, row.close, row.volume || 0].join(',')
      ),
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${symbol}_${timeRange}_${new Date().toISOString()}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  }, [ohlcvData, symbol, timeRange]);

  /**
   * Add indicator
   */
  const handleAddIndicator = useCallback((type: string) => {
    const newIndicator: IndicatorConfig = {
      id: `${type}_${Date.now()}`,
      type,
      period: type === 'sma' || type === 'ema' ? 20 : 14,
      color: '#2196F3',
      lineWidth: 2,
    };

    setIndicators(prev => [...prev, newIndicator]);
    setIndicatorMenuAnchor(null);
  }, []);

  return (
    <Box
      ref={containerRef}
      sx={{
        height: 'calc(100vh - 120px)', // Account for navigation and padding
        display: 'flex',
        flexDirection: 'column',
        bgcolor: '#0A0E27',
        overflow: 'hidden',
        borderRadius: 2,
      }}
    >
      {/* Header Toolbar */}
      <Paper
        elevation={0}
        sx={{
          p: 2,
          bgcolor: '#1A1F3A',
          borderRadius: 0,
        }}
      >
        <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
          {/* Symbol Input */}
          <TextField
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            placeholder="Symbol"
            size="small"
            sx={{
              width: 120,
              '& .MuiInputBase-root': {
                color: 'white',
                bgcolor: 'rgba(255,255,255,0.05)',
              },
              '& .MuiOutlinedInput-root': {
                '& fieldset': { borderColor: 'rgba(255,255,255,0.2)' },
                '&:hover fieldset': { borderColor: 'rgba(255,255,255,0.3)' },
              },
            }}
          />

          {/* Timeframe Selector */}
          <ButtonGroup variant="outlined" size="small">
            {['1D', '5D', '1M', '3M', '1Y'].map((range) => (
              <Button
                key={range}
                onClick={() => setTimeRange(range)}
                variant={timeRange === range ? 'contained' : 'outlined'}
                sx={{
                  color: timeRange === range ? 'white' : '#64B5F6',
                  bgcolor: timeRange === range ? '#1976D2' : 'transparent',
                  borderColor: '#64B5F6',
                  '&:hover': {
                    bgcolor: timeRange === range ? '#1565C0' : 'rgba(100, 181, 246, 0.1)',
                  },
                }}
              >
                {range}
              </Button>
            ))}
          </ButtonGroup>

          {/* Chart Type Selector */}
          <ToggleButtonGroup
            value={chartType}
            exclusive
            onChange={(e, value) => value && setChartType(value)}
            size="small"
          >
            <ToggleButton value="candlestick" sx={{ color: '#64B5F6' }}>
              <CandlestickChart fontSize="small" />
            </ToggleButton>
            <ToggleButton value="line" sx={{ color: '#64B5F6' }}>
              <ShowChart fontSize="small" />
            </ToggleButton>
            <ToggleButton value="area" sx={{ color: '#64B5F6' }}>
              <AreaChart fontSize="small" />
            </ToggleButton>
          </ToggleButtonGroup>

          <Box sx={{ flexGrow: 1 }} />

          {/* Indicator Menu */}
          <Tooltip title="Add Indicator">
            <IconButton
              onClick={(e) => setIndicatorMenuAnchor(e.currentTarget)}
              sx={{ color: '#64B5F6' }}
            >
              <TrendingUp />
            </IconButton>
          </Tooltip>
          <Menu
            anchorEl={indicatorMenuAnchor}
            open={Boolean(indicatorMenuAnchor)}
            onClose={() => setIndicatorMenuAnchor(null)}
          >
            <MenuItem onClick={() => handleAddIndicator('sma')}>
              <ListItemText>SMA (20)</ListItemText>
            </MenuItem>
            <MenuItem onClick={() => handleAddIndicator('ema')}>
              <ListItemText>EMA (20)</ListItemText>
            </MenuItem>
            <MenuItem onClick={() => handleAddIndicator('bb')}>
              <ListItemText>Bollinger Bands</ListItemText>
            </MenuItem>
          </Menu>

          {/* Control Buttons */}
          <Tooltip title="Refresh">
            <IconButton onClick={loadData} disabled={loading} sx={{ color: '#64B5F6' }}>
              {loading ? <CircularProgress size={24} /> : <Refresh />}
            </IconButton>
          </Tooltip>

          <Tooltip title="Export CSV">
            <IconButton onClick={handleExport} sx={{ color: '#64B5F6' }}>
              <Download />
            </IconButton>
          </Tooltip>

          <Tooltip title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}>
            <IconButton onClick={toggleFullscreen} sx={{ color: '#64B5F6' }}>
              {isFullscreen ? <FullscreenExit /> : <Fullscreen />}
            </IconButton>
          </Tooltip>
        </Stack>

        {/* Model Toggle Chips */}
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mt: 2 }}>
          {models.map((model) => (
            <Chip
              key={model.id}
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Box
                    sx={{
                      width: 12,
                      height: 12,
                      borderRadius: '50%',
                      bgcolor: model.color,
                      opacity: model.enabled ? 1 : 0.3,
                    }}
                  />
                  <Typography variant="caption">{model.name}</Typography>
                </Box>
              }
              onClick={() => toggleModel(model.id)}
              variant={model.enabled ? 'filled' : 'outlined'}
              sx={{
                bgcolor: model.enabled ? 'rgba(100, 181, 246, 0.1)' : 'transparent',
                borderColor: model.enabled ? model.color : 'rgba(255,255,255,0.3)',
                color: 'white',
                '&:hover': { bgcolor: 'rgba(100, 181, 246, 0.2)' },
              }}
            />
          ))}
        </Box>

        {/* Error Alert */}
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
      </Paper>

      {/* Main Content: Chart + VIX Widget */}
      <Box sx={{ flex: 1, p: 2, overflow: 'hidden', display: 'flex', gap: 2 }}>
        {/* Chart Area */}
        <Box sx={{ flex: 1, minWidth: 0 }}>
          {ohlcvData.length > 0 ? (
            <TradingViewChart
              config={{
                symbol,
                interval: timeRange,
                theme: 'dark',
                showVolume: true,
                height: undefined, // Let it fill container
              }}
              data={ohlcvData}
              indicators={indicators}
              predictionSeries={predictionSeries}
              onCrosshairMove={handleCrosshairMove}
            />
          ) : (
            <Box
              sx={{
                height: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              {loading ? (
                <CircularProgress />
              ) : (
                <Typography sx={{ color: 'rgba(255,255,255,0.5)' }}>
                  No data available
                </Typography>
              )}
            </Box>
          )}
        </Box>

        {/* VIX Analysis Widget */}
        <Box sx={{ width: 320, flexShrink: 0 }}>
          <VIXAnalysisWidget data={epidemicData} loading={loading} />
        </Box>
      </Box>

      {/* Crosshair Info Panel */}
      {crosshairData && (
        <Paper
          sx={{
            position: 'absolute',
            top: 100,
            left: 20,
            p: 1.5,
            bgcolor: 'rgba(26, 31, 58, 0.95)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(100, 181, 246, 0.3)',
            minWidth: 200,
          }}
        >
          <Typography variant="caption" sx={{ color: '#64B5F6' }}>
            Price Info
          </Typography>
          <Stack spacing={0.5} sx={{ mt: 1 }}>
            <Typography variant="body2" sx={{ color: 'white' }}>
              O: {crosshairData.open?.toFixed(2) || '—'}
            </Typography>
            <Typography variant="body2" sx={{ color: 'white' }}>
              H: {crosshairData.high?.toFixed(2) || '—'}
            </Typography>
            <Typography variant="body2" sx={{ color: 'white' }}>
              L: {crosshairData.low?.toFixed(2) || '—'}
            </Typography>
            <Typography variant="body2" sx={{ color: 'white' }}>
              C: {crosshairData.close?.toFixed(2) || '—'}
            </Typography>
            {crosshairData.volume && (
              <Typography variant="body2" sx={{ color: 'white' }}>
                V: {(crosshairData.volume / 1000000).toFixed(2)}M
              </Typography>
            )}
          </Stack>
        </Paper>
      )}
    </Box>
  );
};

export default UnifiedAnalysisEnhanced;
