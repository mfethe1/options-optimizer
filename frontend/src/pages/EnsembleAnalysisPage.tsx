/**
 * Ensemble Analysis Page
 * Multi-Model Neural Network Comparison Dashboard
 *
 * Features:
 * - Multi-model prediction comparison chart
 * - Ensemble consensus recommendation
 * - Model agreement visualization
 * - Performance metrics tracking
 * - Trading recommendations with position sizing
 * - Intraday and long-term modes
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  Button,
  TextField,
  CircularProgress,
  Alert,
  Card,
  CardContent,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  LinearProgress,
  Tabs,
  Tab,
  Tooltip
} from '@mui/material';
import { Refresh, TrendingUp, ShowChart, Speed, Info } from '@mui/icons-material';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';
import {
  getEnsembleAnalysis,
  getModelPerformance,
  getExplanation,
  EnsembleAnalysis,
  ModelPerformanceMetric
} from '../api/ensembleApi';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const MODEL_COLORS: Record<string, string> = {
  'epidemic_volatility': '#9c27b0',
  'tft_conformal': '#2196f3',
  'gnn': '#4caf50',
  'mamba': '#ff9800',
  'pinn': '#f44336',
  'momentum_baseline': '#607d8b',
  'ensemble': '#000000'
};

const MODEL_NAMES: Record<string, string> = {
  'epidemic_volatility': 'Epidemic Volatility',
  'tft_conformal': 'TFT + Conformal',
  'gnn': 'Graph Neural Network',
  'mamba': 'Mamba (State Space)',
  'pinn': 'PINN (Physics-Informed)',
  'momentum_baseline': 'Momentum Baseline',
  'ensemble': 'Ensemble Consensus'
};

const EnsembleAnalysisPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [symbol, setSymbol] = useState<string>('AAPL');
  const [timeHorizon, setTimeHorizon] = useState<string>('short_term');
  const [analysis, setAnalysis] = useState<EnsembleAnalysis | null>(null);
  const [performance, setPerformance] = useState<any>(null);
  const [explanation, setExplanation] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadAnalysis = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await getEnsembleAnalysis(symbol, timeHorizon, true);
      setAnalysis(data);
    } catch (err: any) {
      console.error('Error loading ensemble analysis:', err);
      setError(err.response?.data?.detail || 'Failed to load analysis');
    } finally {
      setLoading(false);
    }
  };

  const loadPerformance = async () => {
    try {
      const data = await getModelPerformance();
      setPerformance(data);
    } catch (err) {
      console.error('Error loading performance:', err);
    }
  };

  const loadExplanation = async () => {
    try {
      const data = await getExplanation();
      setExplanation(data);
    } catch (err) {
      console.error('Error loading explanation:', err);
    }
  };

  useEffect(() => {
    loadAnalysis();
    loadPerformance();
    loadExplanation();
  }, []);

  const getSignalColor = (signal: string) => {
    if (signal.includes('BUY')) return 'success';
    if (signal.includes('SELL')) return 'error';
    return 'default';
  };

  const getAgreementColor = (agreement: number) => {
    if (agreement >= 0.8) return 'success';
    if (agreement >= 0.5) return 'warning';
    return 'error';
  };

  // Prepare chart data
  const prepareChartData = () => {
    if (!analysis) return [];

    const data = [];

    // Current price point
    const current = {
      name: 'Current',
      price: analysis.current_price,
      ensemble: analysis.current_price
    };

    analysis.model_predictions.forEach(model => {
      current[model.model_name] = analysis.current_price;
    });

    data.push(current);

    // Predicted price point
    const predicted = {
      name: 'Predicted',
      ensemble: analysis.ensemble_prediction
    };

    analysis.model_predictions.forEach(model => {
      predicted[model.model_name] = model.price_prediction;
    });

    data.push(predicted);

    return data;
  };

  // Prepare weights data for bar chart
  const prepareWeightsData = () => {
    if (!analysis) return [];

    return Object.entries(analysis.model_weights).map(([model, weight]) => ({
      model: MODEL_NAMES[model] || model,
      weight: weight * 100,
      color: MODEL_COLORS[model]
    }));
  };

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          🎯 Ensemble Neural Network Analysis
        </Typography>
        <Typography variant="subtitle1" color="textSecondary">
          Multi-Model Predictions & Consensus Recommendations
        </Typography>
      </Box>

      {/* Controls */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={3}>
            <TextField
              fullWidth
              label="Stock Symbol"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              placeholder="AAPL"
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <FormControl fullWidth>
              <InputLabel>Time Horizon</InputLabel>
              <Select
                value={timeHorizon}
                onChange={(e) => setTimeHorizon(e.target.value)}
                label="Time Horizon"
              >
                <MenuItem value="intraday">Intraday (Minutes-Hours)</MenuItem>
                <MenuItem value="short_term">Short Term (1-5 Days)</MenuItem>
                <MenuItem value="medium_term">Medium Term (5-30 Days)</MenuItem>
                <MenuItem value="long_term">Long Term (30+ Days)</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={3}>
            <Button
              fullWidth
              variant="contained"
              startIcon={<Refresh />}
              onClick={loadAnalysis}
              disabled={loading}
              sx={{ height: 56 }}
            >
              {loading ? 'Analyzing...' : 'Analyze'}
            </Button>
          </Grid>
          <Grid item xs={12} md={3}>
            <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'primary.light', color: 'white' }}>
              <Typography variant="caption">Models Active</Typography>
              <Typography variant="h6">
                {analysis ? analysis.model_predictions.length : 0} / 5
              </Typography>
            </Paper>
          </Grid>
        </Grid>
      </Paper>

      {/* Error */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Loading */}
      {loading && !analysis && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {/* Tabs */}
      {analysis && (
        <>
          <Paper sx={{ mb: 2 }}>
            <Tabs value={activeTab} onChange={(e, v) => setActiveTab(v)}>
              <Tab label="Multi-Model Comparison" icon={<ShowChart />} />
              <Tab label="Ensemble Recommendation" icon={<TrendingUp />} />
              <Tab label="Model Performance" icon={<Speed />} />
            </Tabs>
          </Paper>

          {/* Tab 1: Multi-Model Comparison */}
          <TabPanel value={activeTab} index={0}>
            <Grid container spacing={3}>
              {/* Comparison Chart */}
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Multi-Model Price Predictions
                    </Typography>
                    <Box sx={{ height: 400, mt: 2 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={prepareChartData()}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="name" />
                          <YAxis
                            label={{ value: 'Price ($)', angle: -90, position: 'insideLeft' }}
                            domain={['dataMin - 5', 'dataMax + 5']}
                          />
                          <RechartsTooltip />
                          <Legend />

                          {/* Ensemble line (bold) */}
                          <Line
                            type="monotone"
                            dataKey="ensemble"
                            stroke={MODEL_COLORS.ensemble}
                            strokeWidth={4}
                            name="Ensemble Consensus"
                          />

                          {/* Individual model lines */}
                          {analysis.model_predictions.map((model, idx) => (
                            <Line
                              key={idx}
                              type="monotone"
                              dataKey={model.model_name}
                              stroke={MODEL_COLORS[model.model_name]}
                              strokeWidth={2}
                              name={MODEL_NAMES[model.model_name] || model.model_name}
                            />
                          ))}
                        </LineChart>
                      </ResponsiveContainer>
                    </Box>

                    {/* Uncertainty indicator */}
                    <Alert severity="info" sx={{ mt: 2 }}>
                      <Typography variant="body2">
                        <strong>Prediction Std Dev:</strong> ${analysis.prediction_std.toFixed(2)} |{' '}
                        <strong>Model Agreement:</strong> {(analysis.model_agreement * 100).toFixed(0)}%
                        {analysis.model_agreement < 0.5 && ' ⚠️ Low agreement - high uncertainty!'}
                      </Typography>
                    </Alert>
                  </CardContent>
                </Card>
              </Grid>

              {/* Model Predictions Table */}
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Individual Model Predictions
                    </Typography>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell><strong>Model</strong></TableCell>
                          <TableCell align="right"><strong>Prediction</strong></TableCell>
                          <TableCell align="right"><strong>Change</strong></TableCell>
                          <TableCell align="center"><strong>Signal</strong></TableCell>
                          <TableCell align="right"><strong>Confidence</strong></TableCell>
                          <TableCell align="right"><strong>Weight</strong></TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {analysis.model_predictions.map((model, idx) => {
                          const change = ((model.price_prediction - analysis.current_price) / analysis.current_price) * 100;
                          const weight = analysis.model_weights[model.model_name] || 0;

                          return (
                            <TableRow key={idx}>
                              <TableCell>
                                <Chip
                                  label={MODEL_NAMES[model.model_name] || model.model_name}
                                  size="small"
                                  sx={{ bgcolor: MODEL_COLORS[model.model_name], color: 'white' }}
                                />
                              </TableCell>
                              <TableCell align="right">
                                ${model.price_prediction.toFixed(2)}
                              </TableCell>
                              <TableCell align="right" style={{ color: change >= 0 ? 'green' : 'red' }}>
                                {change >= 0 ? '+' : ''}{change.toFixed(2)}%
                              </TableCell>
                              <TableCell align="center">
                                <Chip
                                  label={model.signal}
                                  color={getSignalColor(model.signal) as any}
                                  size="small"
                                />
                              </TableCell>
                              <TableCell align="right">
                                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 1 }}>
                                  <LinearProgress
                                    variant="determinate"
                                    value={model.confidence * 100}
                                    sx={{ width: 60, height: 8, borderRadius: 4 }}
                                  />
                                  <Typography variant="caption">
                                    {(model.confidence * 100).toFixed(0)}%
                                  </Typography>
                                </Box>
                              </TableCell>
                              <TableCell align="right">
                                {(weight * 100).toFixed(1)}%
                              </TableCell>
                            </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  </CardContent>
                </Card>
              </Grid>

              {/* Model Weights Chart */}
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Model Weights (Adaptive)
                    </Typography>
                    <Box sx={{ height: 300, mt: 2 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={prepareWeightsData()}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="model" />
                          <YAxis label={{ value: 'Weight (%)', angle: -90, position: 'insideLeft' }} />
                          <RechartsTooltip />
                          <Bar dataKey="weight" fill="#8884d8">
                            {prepareWeightsData().map((entry, index) => (
                              <Bar key={`bar-${index}`} fill={entry.color} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </Box>
                    <Alert severity="info" sx={{ mt: 2 }}>
                      <Typography variant="body2">
                        Weights are dynamically adjusted based on recent performance, market regime, and time horizon.
                      </Typography>
                    </Alert>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          {/* Tab 2: Ensemble Recommendation */}
          <TabPanel value={activeTab} index={1}>
            <Grid container spacing={3}>
              {/* Main Recommendation Card */}
              <Grid item xs={12} md={8}>
                <Card>
                  <CardContent>
                    <Typography variant="h5" gutterBottom>
                      Ensemble Consensus Recommendation
                    </Typography>

                    <Grid container spacing={3} sx={{ mt: 1 }}>
                      {/* Current Price */}
                      <Grid item xs={6}>
                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                          <Typography variant="caption" color="textSecondary">
                            Current Price
                          </Typography>
                          <Typography variant="h4">
                            ${analysis.current_price.toFixed(2)}
                          </Typography>
                        </Paper>
                      </Grid>

                      {/* Ensemble Prediction */}
                      <Grid item xs={6}>
                        <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'primary.light', color: 'white' }}>
                          <Typography variant="caption">
                            Ensemble Prediction
                          </Typography>
                          <Typography variant="h4">
                            ${analysis.ensemble_prediction.toFixed(2)}
                          </Typography>
                          <Typography variant="caption">
                            ({(analysis.expected_return * 100).toFixed(2)}% expected return)
                          </Typography>
                        </Paper>
                      </Grid>

                      {/* Trading Signal */}
                      <Grid item xs={12}>
                        <Paper sx={{ p: 3, textAlign: 'center' }}>
                          <Typography variant="h6" gutterBottom>
                            Trading Signal
                          </Typography>
                          <Chip
                            label={analysis.ensemble_signal}
                            color={getSignalColor(analysis.ensemble_signal) as any}
                            sx={{ fontSize: '1.5rem', p: 3, height: 'auto' }}
                          />
                          <Box sx={{ mt: 2 }}>
                            <Typography variant="body2" color="textSecondary">
                              Confidence: {(analysis.ensemble_confidence * 100).toFixed(0)}%
                            </Typography>
                            <LinearProgress
                              variant="determinate"
                              value={analysis.ensemble_confidence * 100}
                              sx={{ mt: 1, height: 10, borderRadius: 5 }}
                            />
                          </Box>
                        </Paper>
                      </Grid>

                      {/* Position Sizing */}
                      <Grid item xs={12}>
                        <Paper sx={{ p: 2 }}>
                          <Typography variant="h6" gutterBottom>
                            Position Sizing Recommendation
                          </Typography>
                          <Typography variant="h3" sx={{ mb: 2 }}>
                            {(analysis.position_size * 100).toFixed(0)}%
                          </Typography>
                          <Typography variant="body2" color="textSecondary">
                            of portfolio capital
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={analysis.position_size * 100}
                            sx={{ mt: 2, height: 12, borderRadius: 6 }}
                            color={analysis.position_size > 0.3 ? 'success' : 'warning'}
                          />
                          {analysis.position_size === 0 && (
                            <Alert severity="warning" sx={{ mt: 2 }}>
                              <Typography variant="body2">
                                Position size is 0% - uncertainty is too high. Wait for better setup.
                              </Typography>
                            </Alert>
                          )}
                        </Paper>
                      </Grid>

                      {/* Stop Loss & Take Profit */}
                      {(analysis.stop_loss || analysis.take_profit) && (
                        <Grid item xs={12}>
                          <Paper sx={{ p: 2 }}>
                            <Typography variant="h6" gutterBottom>
                              Risk Management
                            </Typography>
                            <Grid container spacing={2}>
                              {analysis.stop_loss && (
                                <Grid item xs={6}>
                                  <Typography variant="body2" color="textSecondary">
                                    Stop Loss
                                  </Typography>
                                  <Typography variant="h5" color="error.main">
                                    ${analysis.stop_loss.toFixed(2)}
                                  </Typography>
                                </Grid>
                              )}
                              {analysis.take_profit && (
                                <Grid item xs={6}>
                                  <Typography variant="body2" color="textSecondary">
                                    Take Profit
                                  </Typography>
                                  <Typography variant="h5" color="success.main">
                                    ${analysis.take_profit.toFixed(2)}
                                  </Typography>
                                </Grid>
                              )}
                              {analysis.risk_reward_ratio && (
                                <Grid item xs={12}>
                                  <Typography variant="body2" color="textSecondary">
                                    Risk/Reward Ratio: <strong>{analysis.risk_reward_ratio.toFixed(2)}:1</strong>
                                  </Typography>
                                </Grid>
                              )}
                            </Grid>
                          </Paper>
                        </Grid>
                      )}
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>

              {/* Metrics Sidebar */}
              <Grid item xs={12} md={4}>
                <Grid container spacing={2}>
                  {/* Model Agreement */}
                  <Grid item xs={12}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Model Agreement
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                          <Box sx={{ flexGrow: 1 }}>
                            <LinearProgress
                              variant="determinate"
                              value={analysis.model_agreement * 100}
                              sx={{ height: 16, borderRadius: 8 }}
                              color={getAgreementColor(analysis.model_agreement) as any}
                            />
                          </Box>
                          <Typography variant="h5">
                            {(analysis.model_agreement * 100).toFixed(0)}%
                          </Typography>
                        </Box>
                        <Typography variant="body2" color="textSecondary">
                          {analysis.model_agreement >= 0.8
                            ? '✅ Strong consensus - high confidence'
                            : analysis.model_agreement >= 0.5
                            ? '⚠️ Moderate agreement - proceed with caution'
                            : '❌ Low agreement - high uncertainty!'}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>

                  {/* Uncertainty */}
                  <Grid item xs={12}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Prediction Uncertainty
                        </Typography>
                        <Typography variant="h4" sx={{ mb: 1 }}>
                          ±${analysis.prediction_std.toFixed(2)}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Standard deviation across models
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>

                  {/* Time Horizon */}
                  <Grid item xs={12}>
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          Time Horizon
                        </Typography>
                        <Chip
                          label={analysis.time_horizon.replace('_', ' ').toUpperCase()}
                          color="primary"
                          sx={{ fontSize: '1.1rem', p: 2, height: 'auto' }}
                        />
                        <Typography variant="body2" color="textSecondary" sx={{ mt: 2 }}>
                          Model weights are optimized for this horizon
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              </Grid>
            </Grid>
          </TabPanel>

          {/* Tab 3: Model Performance */}
          <TabPanel value={activeTab} index={2}>
            {performance && performance.tracking_enabled ? (
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Model Performance Metrics
                      </Typography>
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell><strong>Model</strong></TableCell>
                            <TableCell align="right"><strong>Accuracy</strong></TableCell>
                            <TableCell align="right"><strong>Sharpe Ratio</strong></TableCell>
                            <TableCell align="right"><strong>Current Weight</strong></TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {performance.model_metrics.map((metric: ModelPerformanceMetric, idx: number) => (
                            <TableRow key={idx}>
                              <TableCell>
                                <Chip
                                  label={MODEL_NAMES[metric.model_name] || metric.model_name}
                                  size="small"
                                  sx={{ bgcolor: MODEL_COLORS[metric.model_name], color: 'white' }}
                                />
                              </TableCell>
                              <TableCell align="right">
                                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 1 }}>
                                  <LinearProgress
                                    variant="determinate"
                                    value={metric.accuracy * 100}
                                    sx={{ width: 80, height: 8, borderRadius: 4 }}
                                  />
                                  <Typography variant="body2">
                                    {(metric.accuracy * 100).toFixed(1)}%
                                  </Typography>
                                </Box>
                              </TableCell>
                              <TableCell align="right">
                                {metric.sharpe_ratio.toFixed(2)}
                              </TableCell>
                              <TableCell align="right">
                                {(metric.current_weight * 100).toFixed(1)}%
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                      <Alert severity="info" sx={{ mt: 2 }}>
                        <Typography variant="body2">
                          Performance is tracked over a rolling window. Weights adapt automatically based on recent accuracy.
                        </Typography>
                      </Alert>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            ) : (
              <Alert severity="info">
                Performance tracking is not yet available. Make predictions over time to build performance history.
              </Alert>
            )}
          </TabPanel>
        </>
      )}
    </Container>
  );
};

export default EnsembleAnalysisPage;
