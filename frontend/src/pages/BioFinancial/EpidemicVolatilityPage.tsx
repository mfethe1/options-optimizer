/**
 * Epidemic Volatility Forecasting Page
 *
 * Bio-Financial Breakthrough: Disease dynamics for market fear prediction
 * Uses SIR/SEIR epidemic models to forecast volatility contagion
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  Button,
  CircularProgress,
  Alert,
  Card,
  CardContent,
  Chip,
  Divider,
  Tab,
  Tabs,
  LinearProgress,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Info,
  Warning,
  CheckCircle,
  Timeline,
  Refresh,
  Science,
  School
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import {
  getEpidemicStatus,
  getEpidemicForecast,
  getCurrentEpidemicState,
  getHistoricalEpisodes,
  trainEpidemicModel,
  evaluateEpidemicModel,
  getEpidemicExplanation,
  EpidemicForecast,
  EpidemicState,
  HistoricalEpisodes,
  ModelExplanation
} from '../../api/epidemicVolatilityApi';
import { VIXForecastChart, generateSampleVIXForecast } from '../../components/charts';

// Regime colors
const REGIME_COLORS: Record<string, string> = {
  'calm': '#4caf50',
  'pre_volatile': '#ff9800',
  'volatile': '#f44336',
  'stabilized': '#2196f3'
};

const REGIME_ICONS: Record<string, React.ReactNode> = {
  'calm': <CheckCircle style={{ color: '#4caf50' }} />,
  'pre_volatile': <Warning style={{ color: '#ff9800' }} />,
  'volatile': <TrendingUp style={{ color: '#f44336' }} />,
  'stabilized': <TrendingDown style={{ color: '#2196f3' }} />
};

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const EpidemicVolatilityPage: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Data states
  const [forecast, setForecast] = useState<EpidemicForecast | null>(null);
  const [currentState, setCurrentState] = useState<EpidemicState | null>(null);
  const [episodes, setEpisodes] = useState<HistoricalEpisodes | null>(null);
  const [explanation, setExplanation] = useState<ModelExplanation | null>(null);

  // Training states
  const [training, setTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);

  // Load initial data
  useEffect(() => {
    loadData();
    loadExplanation();
  }, []);

  const loadData = async () => {
    setLoading(true);
    setError(null);

    try {
      // Load all data in parallel
      const [forecastData, stateData, episodesData] = await Promise.all([
        getEpidemicForecast(30, 'SEIR'),
        getCurrentEpidemicState(),
        getHistoricalEpisodes()
      ]);

      setForecast(forecastData);
      setCurrentState(stateData);
      setEpisodes(episodesData);
    } catch (err: any) {
      console.error('Error loading epidemic data:', err);
      setError(err.response?.data?.detail || 'Failed to load epidemic data');
    } finally {
      setLoading(false);
    }
  };

  const loadExplanation = async () => {
    try {
      const data = await getEpidemicExplanation();
      setExplanation(data);
    } catch (err) {
      console.error('Error loading explanation:', err);
    }
  };

  const handleTrainModel = async () => {
    setTraining(true);
    setError(null);

    try {
      // Simulate progress (actual progress would come from websocket)
      const progressInterval = setInterval(() => {
        setTrainingProgress(prev => Math.min(prev + 5, 90));
      }, 500);

      const result = await trainEpidemicModel('SEIR', 100, 32, 0.1);

      clearInterval(progressInterval);
      setTrainingProgress(100);

      // Reload data after training
      await loadData();

      setTimeout(() => {
        setTraining(false);
        setTrainingProgress(0);
      }, 1000);
    } catch (err: any) {
      console.error('Error training model:', err);
      setError(err.response?.data?.detail || 'Failed to train model');
      setTraining(false);
      setTrainingProgress(0);
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // Render epidemic state pie chart data
  const getEpidemicStateData = () => {
    if (!currentState) return [];

    const data = [
      { name: 'Susceptible (Calm)', value: currentState.susceptible * 100, color: '#4caf50' },
      { name: 'Infected (Volatile)', value: currentState.infected * 100, color: '#f44336' },
      { name: 'Recovered (Stabilized)', value: currentState.recovered * 100, color: '#2196f3' }
    ];

    if (currentState.exposed !== null && currentState.exposed !== undefined) {
      data.splice(1, 0, {
        name: 'Exposed (Pre-Volatile)',
        value: currentState.exposed * 100,
        color: '#ff9800'
      });
    }

    return data;
  };

  // Render trading signal card
  const renderTradingSignal = () => {
    if (!forecast) return null;

    const signal = forecast.trading_signal;
    const actionColor =
      signal.action === 'buy_protection' ? '#f44336' :
      signal.action === 'sell_protection' ? '#4caf50' :
      '#ff9800';

    return (
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Trading Signal
          </Typography>
          <Box sx={{ textAlign: 'center', my: 2 }}>
            <Chip
              label={signal.action.toUpperCase().replace('_', ' ')}
              sx={{
                backgroundColor: actionColor,
                color: 'white',
                fontSize: '1.2rem',
                padding: '20px 10px',
                fontWeight: 'bold'
              }}
            />
          </Box>
          <Typography variant="body2" color="textSecondary" sx={{ mt: 2 }}>
            Confidence: {(signal.confidence * 100).toFixed(1)}%
          </Typography>
          <LinearProgress
            variant="determinate"
            value={signal.confidence * 100}
            sx={{ mt: 1, height: 8, borderRadius: 4 }}
          />
          <Typography variant="body2" sx={{ mt: 2 }}>
            {signal.reasoning}
          </Typography>
        </CardContent>
      </Card>
    );
  };

  // Render forecast interpretation
  const renderForecastInterpretation = () => {
    if (!forecast) return null;

    return (
      <Alert severity="info" icon={<Science />} sx={{ mb: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          <strong>Epidemic Analysis:</strong>
        </Typography>
        <Typography variant="body2">
          {forecast.interpretation}
        </Typography>
      </Alert>
    );
  };

  // Render current state overview
  const renderCurrentState = () => {
    if (!currentState) return null;

    return (
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Market Regime
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, my: 2 }}>
                {REGIME_ICONS[currentState.regime]}
                <Typography variant="h4" sx={{ color: REGIME_COLORS[currentState.regime] }}>
                  {currentState.regime.toUpperCase().replace('_', '-')}
                </Typography>
              </Box>
              <Divider sx={{ my: 2 }} />
              <Grid container spacing={1}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">
                    Current VIX:
                  </Typography>
                  <Typography variant="h6">
                    {currentState.current_vix.toFixed(2)}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="textSecondary">
                    Sentiment:
                  </Typography>
                  <Typography variant="h6">
                    {currentState.current_sentiment >= 0 ? '+' : ''}
                    {currentState.current_sentiment.toFixed(2)}
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Epidemic State Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={getEpidemicStateData()}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={70}
                    label={(entry) => `${entry.value.toFixed(1)}%`}
                  >
                    {getEpidemicStateData().map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <RechartsTooltip />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Epidemic Parameters
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={4}>
                  <Typography variant="body2" color="textSecondary">
                    β (Beta) - Infection Rate:
                  </Typography>
                  <Typography variant="h6">
                    {currentState.beta.toFixed(4)}
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    How fast fear spreads
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={4}>
                  <Typography variant="body2" color="textSecondary">
                    γ (Gamma) - Recovery Rate:
                  </Typography>
                  <Typography variant="h6">
                    {currentState.gamma.toFixed(4)}
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    How fast market stabilizes
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={4}>
                  <Typography variant="body2" color="textSecondary">
                    Susceptible:
                  </Typography>
                  <Typography variant="h6">
                    {(currentState.susceptible * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    Calm market proportion
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  // Render historical episodes
  const renderHistoricalEpisodes = () => {
    if (!episodes || episodes.episodes.length === 0) {
      return (
        <Alert severity="info">
          No historical epidemic episodes detected
        </Alert>
      );
    }

    return (
      <Box>
        <Typography variant="h6" gutterBottom>
          Historical Volatility "Epidemics" ({episodes.total_episodes} episodes)
        </Typography>
        <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
          Periods of volatility contagion resembling disease outbreaks: rapid infection (VIX spike), peak, and recovery.
        </Typography>

        <Grid container spacing={2}>
          {episodes.episodes.slice(0, 6).map((episode, idx) => (
            <Grid item xs={12} sm={6} md={4} key={idx}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Typography variant="subtitle2">
                      {episode.start_date}
                    </Typography>
                    <Chip
                      label={episode.severity}
                      size="small"
                      color={
                        episode.severity === 'high' ? 'error' :
                        episode.severity === 'medium' ? 'warning' :
                        'success'
                      }
                    />
                  </Box>
                  <Divider sx={{ my: 1 }} />
                  <Grid container spacing={1}>
                    <Grid item xs={6}>
                      <Typography variant="caption" color="textSecondary">
                        Duration:
                      </Typography>
                      <Typography variant="body2">
                        {episode.duration_days} days
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="caption" color="textSecondary">
                        Peak VIX:
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 'bold', color: '#f44336' }}>
                        {episode.peak_vix.toFixed(2)}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="caption" color="textSecondary">
                        Start VIX:
                      </Typography>
                      <Typography variant="body2">
                        {episode.start_vix.toFixed(2)}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="caption" color="textSecondary">
                        End VIX:
                      </Typography>
                      <Typography variant="body2">
                        {episode.end_vix.toFixed(2)}
                      </Typography>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  };

  // Render model explanation
  const renderExplanation = () => {
    if (!explanation) return null;

    return (
      <Box>
        <Alert severity="info" icon={<School />} sx={{ mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            {explanation.title}
          </Typography>
          <Typography variant="body2">
            {explanation.concept}
          </Typography>
        </Alert>

        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Epidemic States
                </Typography>
                <Box sx={{ pl: 2 }}>
                  {Object.entries(explanation.model.states).map(([key, value]) => (
                    <Box key={key} sx={{ my: 1 }}>
                      <Typography variant="subtitle2" color="primary">
                        {key}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        {value}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Epidemic Parameters
                </Typography>
                <Box sx={{ pl: 2 }}>
                  {Object.entries(explanation.model.parameters).map(([key, value]) => (
                    <Box key={key} sx={{ my: 1 }}>
                      <Typography variant="subtitle2" color="primary">
                        {key}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        {value}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Why This Is Innovative
                </Typography>
                <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                  {explanation.innovation}
                </Typography>

                <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
                  Advantages
                </Typography>
                <Box component="ul" sx={{ pl: 3 }}>
                  {explanation.advantages.map((advantage, idx) => (
                    <Typography component="li" variant="body2" key={idx} sx={{ mb: 0.5 }}>
                      {advantage}
                    </Typography>
                  ))}
                </Box>

                <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
                  Use Cases
                </Typography>
                <Box component="ul" sx={{ pl: 3 }}>
                  {explanation.use_cases.map((useCase, idx) => (
                    <Typography component="li" variant="body2" key={idx} sx={{ mb: 0.5 }}>
                      {useCase}
                    </Typography>
                  ))}
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
    );
  };

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            🦠 Epidemic Volatility Forecasting
          </Typography>
          <Typography variant="subtitle1" color="textSecondary">
            Bio-Financial Breakthrough: Disease dynamics for market fear prediction
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={loadData}
            disabled={loading}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<Science />}
            onClick={handleTrainModel}
            disabled={training}
          >
            {training ? 'Training...' : 'Train Model'}
          </Button>
        </Box>
      </Box>

      {/* Training Progress */}
      {training && (
        <Box sx={{ mb: 2 }}>
          <Alert severity="info">
            Training epidemic volatility model... {trainingProgress}%
          </Alert>
          <LinearProgress variant="determinate" value={trainingProgress} sx={{ mt: 1 }} />
        </Box>
      )}

      {/* Error Alert */}
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

      {/* Main Content */}
      {!loading && forecast && (
        <Box>
          {/* Forecast Interpretation */}
          {renderForecastInterpretation()}

          {/* Trading Signal & Forecast */}
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} md={4}>
              {renderTradingSignal()}
            </Grid>
            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Volatility Forecast
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="textSecondary">
                        Current VIX:
                      </Typography>
                      <Typography variant="h4">
                        {forecast.current_vix.toFixed(2)}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="textSecondary">
                        Predicted VIX ({forecast.horizon_days}d):
                      </Typography>
                      <Typography
                        variant="h4"
                        sx={{
                          color: forecast.predicted_vix > forecast.current_vix ? '#f44336' : '#4caf50'
                        }}
                      >
                        {forecast.predicted_vix.toFixed(2)}
                        {forecast.predicted_vix > forecast.current_vix ? ' ↑' : ' ↓'}
                      </Typography>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* VIX Forecast Chart - Professional Visualization */}
          <Box sx={{ mb: 3 }}>
            <VIXForecastChart
              currentVIX={forecast.current_vix}
              predictedVIX={forecast.predicted_vix}
              horizonDays={forecast.horizon_days}
              theme="dark"
              showMultiTimeframe={false}
              height={600}
            />
          </Box>

          {/* Tabs */}
          <Paper>
            <Tabs value={tabValue} onChange={handleTabChange}>
              <Tab label="Current State" />
              <Tab label="Historical Episodes" />
              <Tab label="Model Explanation" />
            </Tabs>

            <TabPanel value={tabValue} index={0}>
              {renderCurrentState()}
            </TabPanel>

            <TabPanel value={tabValue} index={1}>
              {renderHistoricalEpisodes()}
            </TabPanel>

            <TabPanel value={tabValue} index={2}>
              {renderExplanation()}
            </TabPanel>
          </Paper>
        </Box>
      )}
    </Container>
  );
};

export default EpidemicVolatilityPage;
