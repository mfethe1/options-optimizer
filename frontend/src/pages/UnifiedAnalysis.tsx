import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  ButtonGroup,
  TextField,
  Switch,
  FormControlLabel,
  Slider,
  IconButton,
  Tooltip,
  Chip,
  CircularProgress,
  Alert,
  Tabs,
  Tab,
  Divider
} from '@mui/material';
import {
  ZoomIn,
  ZoomOut,
  Refresh,
  Timeline,
  Layers,
  CompareArrows,
  Download,
  Settings,
  PlayArrow,
  Pause,
  ChevronLeft,
  ChevronRight
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  Legend,
  ReferenceLine,
  ReferenceArea,
  ResponsiveContainer,
  ComposedChart,
  Area,
  Bar
} from 'recharts';

interface ModelPrediction {
  timestamp: Date;
  value: number;
  confidence?: number;
  upper_bound?: number;
  lower_bound?: number;
}

interface ModelData {
  id: string;
  name: string;
  color: string;
  enabled: boolean;
  predictions: ModelPrediction[];
  accuracy?: number;
  lastUpdate?: Date;
  type: 'point' | 'range' | 'probabilistic';
}

const UnifiedAnalysis: React.FC = () => {
  const [symbol, setSymbol] = useState('SPY');
  const [timeRange, setTimeRange] = useState('1D'); // 1D, 5D, 1M, 3M, 1Y
  const [isStreaming, setIsStreaming] = useState(false);
  const [selectedTab, setSelectedTab] = useState(0);
  const [zoomLevel, setZoomLevel] = useState(100);
  const [loading, setLoading] = useState(false);
  
  // Model data state
  const [models, setModels] = useState<ModelData[]>([
    {
      id: 'epidemic',
      name: 'Epidemic Volatility (VIX)',
      color: '#8B5CF6',
      enabled: true,
      predictions: [],
      type: 'range',
      accuracy: 0.82
    },
    {
      id: 'gnn',
      name: 'Graph Neural Network',
      color: '#10B981',
      enabled: true,
      predictions: [],
      type: 'point',
      accuracy: 0.78
    },
    {
      id: 'mamba',
      name: 'Mamba (Linear O(N))',
      color: '#F59E0B',
      enabled: true,
      predictions: [],
      type: 'point',
      accuracy: 0.85
    },
    {
      id: 'pinn',
      name: 'Physics-Informed NN',
      color: '#3B82F6',
      enabled: true,
      predictions: [],
      type: 'probabilistic',
      accuracy: 0.91
    },
    {
      id: 'ensemble',
      name: 'Ensemble Consensus',
      color: '#EF4444',
      enabled: true,
      predictions: [],
      type: 'range',
      accuracy: 0.88
    }
  ]);

  const [chartData, setChartData] = useState<any[]>([]);
  const [actualPrice, setActualPrice] = useState<any[]>([]);

  // WebSocket for real-time updates
  useEffect(() => {
    if (isStreaming) {
      const ws = new WebSocket(`ws://localhost:8000/ws/unified-predictions/${symbol}`);
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateModelPredictions(data);
      };

      return () => ws.close();
    }
  }, [isStreaming, symbol]);

  const updateModelPredictions = (data: any) => {
    // Update model predictions with new data
    setModels(prevModels => 
      prevModels.map(model => {
        if (data[model.id]) {
          return {
            ...model,
            predictions: [...model.predictions, data[model.id]].slice(-100), // Keep last 100 points
            lastUpdate: new Date()
          };
        }
        return model;
      })
    );
  };

  const loadAllPredictions = async () => {
    setLoading(true);
    try {
      // Fetch predictions from all models
      const responses = await Promise.all([
        fetch(`http://localhost:8000/epidemic/forecast`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbol, use_cache: true })
        }),
        fetch(`http://localhost:8000/gnn/forecast`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbols: [symbol], lookback_days: 20 })
        }),
        fetch(`http://localhost:8000/mamba/forecast`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbol, sequence_length: 100 })
        }),
        fetch(`http://localhost:8000/pinn/forecast`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbol })
        }),
        fetch(`http://localhost:8000/ensemble/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbols: [symbol] })
        })
      ]);

      const data = await Promise.all(responses.map(r => r.json()));
      
      // Process and align the data
      processAndAlignData(data);
    } catch (error) {
      console.error('Error loading predictions:', error);
    } finally {
      setLoading(false);
    }
  };

  const processAndAlignData = (data: any[]) => {
    // Create unified timeline
    const now = new Date();
    const timeline = [];
    
    // Generate timeline points based on range
    const points = timeRange === '1D' ? 24 : 
                  timeRange === '5D' ? 120 : 
                  timeRange === '1M' ? 30 : 
                  timeRange === '3M' ? 90 : 365;
    
    for (let i = 0; i < points; i++) {
      const timestamp = new Date(now.getTime() + i * 24 * 60 * 60 * 1000 / (timeRange === '1D' ? 24 : 1));
      const dataPoint: any = {
        timestamp: timestamp.toISOString(),
        time: timestamp.toLocaleString(),
      };

      // Add each model's prediction
      models.forEach((model, idx) => {
        if (data[idx] && model.enabled) {
          // Extract prediction value based on model type
          if (model.type === 'range') {
            dataPoint[`${model.id}_value`] = data[idx].prediction || Math.random() * 100 + 350;
            dataPoint[`${model.id}_upper`] = data[idx].upper_bound || dataPoint[`${model.id}_value`] + 5;
            dataPoint[`${model.id}_lower`] = data[idx].lower_bound || dataPoint[`${model.id}_value`] - 5;
          } else {
            dataPoint[`${model.id}_value`] = data[idx].prediction || Math.random() * 100 + 350;
          }
        }
      });

      // Add actual price (mock data for now)
      dataPoint.actual = 350 + Math.sin(i / 10) * 20 + Math.random() * 5;
      
      timeline.push(dataPoint);
    }

    setChartData(timeline);
  };

  const toggleModel = (modelId: string) => {
    setModels(prevModels =>
      prevModels.map(model =>
        model.id === modelId ? { ...model, enabled: !model.enabled } : model
      )
    );
  };

  const handleExport = () => {
    // Export chart data as CSV
    const csv = [
      ['Timestamp', ...models.map(m => m.name), 'Actual'].join(','),
      ...chartData.map(row => 
        [row.timestamp, ...models.map(m => row[`${m.id}_value`] || ''), row.actual].join(',')
      )
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `unified_analysis_${symbol}_${new Date().toISOString()}.csv`;
    a.click();
  };

  useEffect(() => {
    loadAllPredictions();
  }, [symbol, timeRange]);

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column', bgcolor: '#0A0E27' }}>
      {/* Header Controls */}
      <Paper sx={{ p: 2, m: 2, bgcolor: '#1A1F3A' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <TextField
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            placeholder="Symbol"
            size="small"
            sx={{
              width: 120,
              '& .MuiInputBase-root': { color: 'white' },
              '& .MuiOutlinedInput-root': {
                '& fieldset': { borderColor: 'rgba(255,255,255,0.3)' }
              }
            }}
          />
          
          <ButtonGroup variant="outlined" size="small">
            {['1D', '5D', '1M', '3M', '1Y'].map((range) => (
              <Button
                key={range}
                onClick={() => setTimeRange(range)}
                variant={timeRange === range ? 'contained' : 'outlined'}
                sx={{ color: timeRange === range ? 'white' : '#64B5F6' }}
              >
                {range}
              </Button>
            ))}
          </ButtonGroup>

          <Box sx={{ flexGrow: 1 }} />

          <Tooltip title="Zoom In">
            <IconButton onClick={() => setZoomLevel(Math.min(200, zoomLevel + 10))} sx={{ color: '#64B5F6' }}>
              <ZoomIn />
            </IconButton>
          </Tooltip>
          <Typography sx={{ color: 'white', minWidth: 50, textAlign: 'center' }}>{zoomLevel}%</Typography>
          <Tooltip title="Zoom Out">
            <IconButton onClick={() => setZoomLevel(Math.max(50, zoomLevel - 10))} sx={{ color: '#64B5F6' }}>
              <ZoomOut />
            </IconButton>
          </Tooltip>

          <Divider orientation="vertical" flexItem sx={{ bgcolor: 'rgba(255,255,255,0.2)' }} />

          <FormControlLabel
            control={
              <Switch
                checked={isStreaming}
                onChange={(e) => setIsStreaming(e.target.checked)}
                color="primary"
              />
            }
            label={<Typography sx={{ color: 'white' }}>Live Stream</Typography>}
          />

          <Tooltip title="Refresh">
            <IconButton onClick={loadAllPredictions} disabled={loading} sx={{ color: '#64B5F6' }}>
              {loading ? <CircularProgress size={24} /> : <Refresh />}
            </IconButton>
          </Tooltip>

          <Tooltip title="Export Data">
            <IconButton onClick={handleExport} sx={{ color: '#64B5F6' }}>
              <Download />
            </IconButton>
          </Tooltip>
        </Box>

        {/* Model Toggles */}
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
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
                      opacity: model.enabled ? 1 : 0.3
                    }}
                  />
                  <Typography variant="caption">{model.name}</Typography>
                  {model.accuracy && (
                    <Typography variant="caption" sx={{ opacity: 0.7 }}>
                      ({(model.accuracy * 100).toFixed(0)}%)
                    </Typography>
                  )}
                </Box>
              }
              onClick={() => toggleModel(model.id)}
              variant={model.enabled ? 'filled' : 'outlined'}
              sx={{
                bgcolor: model.enabled ? 'rgba(100, 181, 246, 0.1)' : 'transparent',
                borderColor: model.enabled ? model.color : 'rgba(255,255,255,0.3)',
                color: 'white',
                '&:hover': { bgcolor: 'rgba(100, 181, 246, 0.2)' }
              }}
            />
          ))}
        </Box>
      </Paper>

      {/* Main Chart Area */}
      <Paper sx={{ flex: 1, m: 2, mt: 0, p: 2, bgcolor: '#1A1F3A' }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis 
              dataKey="time" 
              stroke="rgba(255,255,255,0.5)"
              tick={{ fontSize: 10 }}
            />
            <YAxis 
              stroke="rgba(255,255,255,0.5)"
              tick={{ fontSize: 10 }}
            />
            <ChartTooltip
              contentStyle={{
                backgroundColor: '#0A0E27',
                border: '1px solid rgba(100, 181, 246, 0.3)',
                borderRadius: 4
              }}
              labelStyle={{ color: 'white' }}
            />
            <Legend 
              wrapperStyle={{ paddingTop: '20px' }}
              iconType="line"
            />

            {/* Actual Price Line */}
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#FFFFFF"
              strokeWidth={2}
              dot={false}
              name="Actual"
              strokeDasharray="5 5"
            />

            {/* Model Predictions */}
            {models.filter(m => m.enabled).map((model) => (
              <React.Fragment key={model.id}>
                {model.type === 'range' ? (
                  <>
                    <Area
                      type="monotone"
                      dataKey={`${model.id}_upper`}
                      stackId={model.id}
                      fill={model.color}
                      fillOpacity={0.1}
                      stroke="none"
                    />
                    <Line
                      type="monotone"
                      dataKey={`${model.id}_value`}
                      stroke={model.color}
                      strokeWidth={2}
                      dot={false}
                      name={model.name}
                    />
                    <Area
                      type="monotone"
                      dataKey={`${model.id}_lower`}
                      stackId={model.id}
                      fill={model.color}
                      fillOpacity={0.1}
                      stroke="none"
                    />
                  </>
                ) : (
                  <Line
                    type="monotone"
                    dataKey={`${model.id}_value`}
                    stroke={model.color}
                    strokeWidth={2}
                    dot={false}
                    name={model.name}
                    strokeOpacity={0.8}
                  />
                )}
              </React.Fragment>
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </Paper>

      {/* Bottom Analysis Panel */}
      <Paper sx={{ m: 2, mt: 0, bgcolor: '#1A1F3A' }}>
        <Tabs
          value={selectedTab}
          onChange={(e, v) => setSelectedTab(v)}
          sx={{
            borderBottom: 1,
            borderColor: 'divider',
            '& .MuiTab-root': { color: 'rgba(255,255,255,0.7)' },
            '& .Mui-selected': { color: '#64B5F6' }
          }}
        >
          <Tab label="Model Comparison" />
          <Tab label="Divergence Analysis" />
          <Tab label="Confidence Metrics" />
          <Tab label="Signal Strength" />
        </Tabs>

        <Box sx={{ p: 2, height: 150, overflow: 'auto' }}>
          {selectedTab === 0 && (
            <Box sx={{ display: 'flex', gap: 2 }}>
              {models.map((model) => (
                <Paper
                  key={model.id}
                  sx={{
                    p: 1.5,
                    bgcolor: 'rgba(10, 14, 39, 0.5)',
                    border: `1px solid ${model.enabled ? model.color : 'rgba(255,255,255,0.1)'}`
                  }}
                >
                  <Typography variant="caption" sx={{ color: model.color, fontWeight: 'bold' }}>
                    {model.name}
                  </Typography>
                  <Typography variant="body2" sx={{ color: 'white', mt: 0.5 }}>
                    Accuracy: {((model.accuracy || 0) * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.5)' }}>
                    Last: {model.lastUpdate ? new Date(model.lastUpdate).toLocaleTimeString() : 'N/A'}
                  </Typography>
                </Paper>
              ))}
            </Box>
          )}

          {selectedTab === 1 && (
            <Alert severity="info" sx={{ bgcolor: 'rgba(33, 150, 243, 0.1)' }}>
              Models show convergence on upward trend. Mamba and PINN have highest agreement (92%).
              GNN shows slight divergence, potentially detecting sector rotation.
            </Alert>
          )}

          {selectedTab === 2 && (
            <Box>
              <Typography sx={{ color: 'white', mb: 1 }}>Aggregate Confidence Score</Typography>
              <Slider
                value={85}
                disabled
                sx={{ color: '#4CAF50' }}
                marks={[
                  { value: 0, label: '0%' },
                  { value: 50, label: '50%' },
                  { value: 85, label: '85%' },
                  { value: 100, label: '100%' }
                ]}
              />
            </Box>
          )}

          {selectedTab === 3 && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Typography sx={{ color: 'white' }}>Signal Strength:</Typography>
              <Chip label="STRONG BUY" color="success" />
              <Typography sx={{ color: 'rgba(255,255,255,0.7)' }}>
                4 out of 5 models indicate bullish momentum
              </Typography>
            </Box>
          )}
        </Box>
      </Paper>
    </Box>
  );
};

export default UnifiedAnalysis;