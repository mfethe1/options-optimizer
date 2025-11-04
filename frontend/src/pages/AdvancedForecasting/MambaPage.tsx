/**
 * Mamba State Space Model Dashboard
 * Priority #3: Linear O(N) complexity for very long sequences
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
  Slider,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Tabs,
  Tab,
  LinearProgress
} from '@mui/material';
import { Refresh, Speed, Timeline, Science } from '@mui/icons-material';
import {
  getMambaForecast,
  analyzeEfficiency,
  getDemoScenarios,
  getExplanation,
  MambaForecast,
  EfficiencyComparison
} from '../../api/mambaApi';

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

const MambaPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [symbol, setSymbol] = useState<string>('AAPL');
  const [sequenceLength, setSequenceLength] = useState<number>(1000);
  const [forecast, setForecast] = useState<MambaForecast | null>(null);
  const [efficiencyData, setEfficiencyData] = useState<any>(null);
  const [demoScenarios, setDemoScenarios] = useState<any>(null);
  const [explanation, setExplanation] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadForecast = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await getMambaForecast(symbol, sequenceLength, true);
      setForecast(data);
    } catch (err: any) {
      console.error('Error loading Mamba forecast:', err);
      setError(err.response?.data?.detail || 'Failed to load forecast');
    } finally {
      setLoading(false);
    }
  };

  const loadEfficiencyAnalysis = async () => {
    try {
      const sequenceLengths = [100, 1000, 10000, 100000, 1000000];
      const data = await analyzeEfficiency(sequenceLengths);
      setEfficiencyData(data);
    } catch (err) {
      console.error('Error loading efficiency analysis:', err);
    }
  };

  const loadDemoScenarios = async () => {
    try {
      const data = await getDemoScenarios();
      setDemoScenarios(data);
    } catch (err) {
      console.error('Error loading demo scenarios:', err);
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
    loadForecast();
    loadEfficiencyAnalysis();
    loadDemoScenarios();
    loadExplanation();
  }, []);

  const getSignalColor = (signal: string) => {
    if (signal.includes('BUY')) return 'success';
    if (signal.includes('SELL')) return 'error';
    return 'default';
  };

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          ⚡ Mamba State Space Model - Linear Complexity
        </Typography>
        <Typography variant="subtitle1" color="textSecondary">
          Priority #3: O(N) complexity • 5x throughput • Handles million-length sequences
        </Typography>
      </Box>

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={activeTab} onChange={(e, v) => setActiveTab(v)}>
          <Tab label="Forecasting" icon={<Timeline />} />
          <Tab label="Efficiency Analysis" icon={<Speed />} />
          <Tab label="Demo Scenarios" icon={<Science />} />
          <Tab label="Explanation" icon={<Science />} />
        </Tabs>
      </Paper>

      {/* Tab 1: Forecasting */}
      <TabPanel value={activeTab} index={0}>
        {/* Input Section */}
        <Paper sx={{ p: 3, mb: 3 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Stock Symbol"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                placeholder="AAPL"
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ px: 2 }}>
                <Typography gutterBottom>
                  Sequence Length: {sequenceLength.toLocaleString()} days
                </Typography>
                <Slider
                  value={sequenceLength}
                  onChange={(e, v) => setSequenceLength(v as number)}
                  min={100}
                  max={5000}
                  step={100}
                  marks={[
                    { value: 100, label: '100' },
                    { value: 1000, label: '1K' },
                    { value: 2500, label: '2.5K' },
                    { value: 5000, label: '5K' }
                  ]}
                />
                <Typography variant="caption" color="textSecondary">
                  Mamba can handle very long sequences efficiently!
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Button
                fullWidth
                variant="contained"
                startIcon={<Refresh />}
                onClick={loadForecast}
                disabled={loading}
                sx={{ height: 56 }}
              >
                {loading ? 'Forecasting...' : 'Get Forecast'}
              </Button>
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
        {loading && !forecast && (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
            <CircularProgress />
          </Box>
        )}

        {/* Results */}
        {forecast && (
          <Grid container spacing={3}>
            {/* Current Price & Signal */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Current Analysis: {forecast.symbol}
                  </Typography>
                  <Box sx={{ mt: 2 }}>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" color="textSecondary">
                        Current Price:
                      </Typography>
                      <Typography variant="h4">
                        ${forecast.current_price.toFixed(2)}
                      </Typography>
                    </Box>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" color="textSecondary">
                        Trading Signal:
                      </Typography>
                      <Chip
                        label={forecast.signal}
                        color={getSignalColor(forecast.signal) as any}
                        size="large"
                        sx={{ fontSize: '1.2rem', mt: 1 }}
                      />
                    </Box>
                    <Box>
                      <Typography variant="body2" color="textSecondary">
                        Confidence:
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <LinearProgress
                          variant="determinate"
                          value={forecast.confidence * 100}
                          sx={{ flexGrow: 1, height: 10, borderRadius: 5 }}
                        />
                        <Typography variant="body2" fontWeight="bold">
                          {(forecast.confidence * 100).toFixed(0)}%
                        </Typography>
                      </Box>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Predictions */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Multi-Horizon Predictions
                  </Typography>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell><strong>Horizon</strong></TableCell>
                        <TableCell align="right"><strong>Price</strong></TableCell>
                        <TableCell align="right"><strong>Change</strong></TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(forecast.predictions).map(([horizon, price]) => {
                        const change = ((price - forecast.current_price) / forecast.current_price) * 100;
                        const changeColor = change >= 0 ? 'green' : 'red';

                        return (
                          <TableRow key={horizon}>
                            <TableCell>{horizon}</TableCell>
                            <TableCell align="right">${price.toFixed(2)}</TableCell>
                            <TableCell align="right" style={{ color: changeColor }}>
                              {change >= 0 ? '+' : ''}{change.toFixed(2)}%
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            </Grid>

            {/* Efficiency Stats */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Efficiency Statistics
                  </Typography>
                  <Grid container spacing={2} sx={{ mt: 1 }}>
                    <Grid item xs={12} sm={6} md={3}>
                      <Paper sx={{ p: 2, bgcolor: 'primary.light', color: 'white' }}>
                        <Typography variant="body2">Sequence Length</Typography>
                        <Typography variant="h5">
                          {forecast.efficiency_stats.sequence_length.toLocaleString()}
                        </Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={12} sm={6} md={3}>
                      <Paper sx={{ p: 2, bgcolor: 'success.light', color: 'white' }}>
                        <Typography variant="body2">Mamba Complexity</Typography>
                        <Typography variant="h5">
                          {forecast.efficiency_stats.mamba_complexity}
                        </Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={12} sm={6} md={3}>
                      <Paper sx={{ p: 2, bgcolor: 'warning.light', color: 'white' }}>
                        <Typography variant="body2">Transformer Complexity</Typography>
                        <Typography variant="h5">
                          {forecast.efficiency_stats.transformer_complexity}
                        </Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={12} sm={6} md={3}>
                      <Paper sx={{ p: 2, bgcolor: 'info.light', color: 'white' }}>
                        <Typography variant="body2">Speedup</Typography>
                        <Typography variant="h5">
                          {forecast.efficiency_stats.theoretical_speedup}
                        </Typography>
                      </Paper>
                    </Grid>
                  </Grid>
                  <Alert severity="success" sx={{ mt: 2 }}>
                    <Typography variant="body2">
                      {forecast.efficiency_stats.can_process_ticks
                        ? '✅ Can process high-frequency tick data (years of millisecond ticks!)'
                        : 'Sequence length suitable for daily/intraday data'}
                    </Typography>
                  </Alert>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        )}
      </TabPanel>

      {/* Tab 2: Efficiency Analysis */}
      <TabPanel value={activeTab} index={1}>
        {efficiencyData && (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Complexity Comparison: Mamba vs Transformers
                  </Typography>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell><strong>Sequence Length</strong></TableCell>
                        <TableCell><strong>Mamba (O(N))</strong></TableCell>
                        <TableCell><strong>Transformer (O(N²))</strong></TableCell>
                        <TableCell><strong>Speedup</strong></TableCell>
                        <TableCell><strong>Tick Data?</strong></TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {efficiencyData.comparisons.map((comp: EfficiencyComparison, idx: number) => (
                        <TableRow key={idx}>
                          <TableCell>{comp.sequence_length.toLocaleString()}</TableCell>
                          <TableCell>{comp.mamba_ops.toLocaleString()} ops</TableCell>
                          <TableCell>{comp.transformer_ops.toLocaleString()} ops</TableCell>
                          <TableCell>
                            <Chip label={comp.theoretical_speedup} color="success" />
                          </TableCell>
                          <TableCell>
                            {comp.can_process_ticks ? '✅ Yes' : '❌ No'}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12}>
              <Alert severity="info">
                <Typography variant="subtitle2" gutterBottom>
                  <strong>Key Insight:</strong>
                </Typography>
                <Typography variant="body2">
                  {efficiencyData.summary.key_insight}
                </Typography>
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    <strong>Use Cases:</strong>
                  </Typography>
                  <ul style={{ margin: 0, paddingLeft: '20px' }}>
                    {efficiencyData.summary.use_cases.map((useCase: string, idx: number) => (
                      <li key={idx}>
                        <Typography variant="body2">{useCase}</Typography>
                      </li>
                    ))}
                  </ul>
                </Box>
              </Alert>
            </Grid>
          </Grid>
        )}
      </TabPanel>

      {/* Tab 3: Demo Scenarios */}
      <TabPanel value={activeTab} index={2}>
        {demoScenarios && (
          <Grid container spacing={3}>
            {demoScenarios.scenarios.map((scenario: any, idx: number) => (
              <Grid item xs={12} md={6} key={idx}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      {scenario.name}
                    </Typography>
                    <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                      {scenario.description}
                    </Typography>
                    <Box sx={{ mb: 1 }}>
                      <Typography variant="body2">
                        <strong>Data Points:</strong> {scenario.data_points.toLocaleString()}
                      </Typography>
                    </Box>
                    <Box sx={{ mb: 1 }}>
                      <Typography variant="body2">
                        <strong>Mamba Time:</strong>{' '}
                        <span style={{ color: 'green' }}>{scenario.mamba_time}</span>
                      </Typography>
                    </Box>
                    <Box sx={{ mb: 1 }}>
                      <Typography variant="body2">
                        <strong>Transformer Time:</strong>{' '}
                        <span style={{ color: 'red' }}>{scenario.transformer_time}</span>
                      </Typography>
                    </Box>
                    <Alert severity="success" sx={{ mt: 2 }}>
                      <strong>Verdict:</strong> {scenario.verdict}
                    </Alert>
                  </CardContent>
                </Card>
              </Grid>
            ))}

            <Grid item xs={12}>
              <Alert severity="warning">
                <Typography variant="subtitle2" gutterBottom>
                  <strong>{demoScenarios.key_insight}</strong>
                </Typography>
              </Alert>
            </Grid>
          </Grid>
        )}
      </TabPanel>

      {/* Tab 4: Explanation */}
      <TabPanel value={activeTab} index={3}>
        {explanation && (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h5" gutterBottom>
                    {explanation.title}
                  </Typography>
                  <Typography variant="body1" sx={{ mb: 2 }}>
                    <strong>Concept:</strong> {explanation.concept}
                  </Typography>
                  <Typography variant="body1" sx={{ mb: 2 }}>
                    <strong>Innovation:</strong> {explanation.innovation}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Architecture
                  </Typography>
                  <Table size="small">
                    <TableBody>
                      {Object.entries(explanation.architecture).map(([key, value]) => (
                        <TableRow key={key}>
                          <TableCell><strong>{key}</strong></TableCell>
                          <TableCell>{value as string}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Performance Metrics
                  </Typography>
                  <Table size="small">
                    <TableBody>
                      {Object.entries(explanation.performance).map(([key, value]) => (
                        <TableRow key={key}>
                          <TableCell><strong>{key}</strong></TableCell>
                          <TableCell>{value as string}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Key Advantages
                  </Typography>
                  <ul>
                    {explanation.key_advantages.map((adv: string, idx: number) => (
                      <li key={idx}>
                        <Typography variant="body1">{adv}</Typography>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12}>
              <Alert severity="info">
                <Typography variant="body2">
                  <strong>Research:</strong> {explanation.research}
                </Typography>
              </Alert>
            </Grid>
          </Grid>
        )}
      </TabPanel>
    </Container>
  );
};

export default MambaPage;
