/**
 * Physics-Informed Neural Networks Dashboard
 * Priority #4: Option pricing & portfolio optimization with physics constraints
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
  Tabs,
  Tab,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Slider
} from '@mui/material';
import { Science, TrendingUp, Calculate, AutoGraph } from '@mui/icons-material';
import {
  priceOption,
  optimizePortfolio,
  getExplanation,
  getDemoExamples,
  OptionPriceResponse,
  PortfolioOptimizationResponse
} from '../../api/pinnApi';

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

const PINNPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);

  // Option Pricing State
  const [stockPrice, setStockPrice] = useState<number>(100);
  const [strikePrice, setStrikePrice] = useState<number>(100);
  const [timeToMaturity, setTimeToMaturity] = useState<number>(1.0);
  const [optionType, setOptionType] = useState<'call' | 'put'>('call');
  const [volatility, setVolatility] = useState<number>(0.2);
  const [riskFreeRate, setRiskFreeRate] = useState<number>(0.05);
  const [optionResult, setOptionResult] = useState<OptionPriceResponse | null>(null);
  const [optionLoading, setOptionLoading] = useState(false);

  // Portfolio Optimization State
  const [portfolioSymbols, setPortfolioSymbols] = useState<string>('AAPL, MSFT, GOOGL, AMZN, NVDA');
  const [targetReturn, setTargetReturn] = useState<number>(0.10);
  const [portfolioResult, setPortfolioResult] = useState<PortfolioOptimizationResponse | null>(null);
  const [portfolioLoading, setPortfolioLoading] = useState(false);

  // General State
  const [error, setError] = useState<string | null>(null);
  const [explanation, setExplanation] = useState<any>(null);
  const [demoExamples, setDemoExamples] = useState<any>(null);

  useEffect(() => {
    loadExplanation();
    loadDemoExamples();
  }, []);

  const loadExplanation = async () => {
    try {
      const data = await getExplanation();
      setExplanation(data);
    } catch (err) {
      console.error('Error loading explanation:', err);
    }
  };

  const loadDemoExamples = async () => {
    try {
      const data = await getDemoExamples();
      setDemoExamples(data);
    } catch (err) {
      console.error('Error loading demo examples:', err);
    }
  };

  const handlePriceOption = async () => {
    setOptionLoading(true);
    setError(null);

    try {
      const result = await priceOption({
        stock_price: stockPrice,
        strike_price: strikePrice,
        time_to_maturity: timeToMaturity,
        option_type: optionType,
        risk_free_rate: riskFreeRate,
        volatility: volatility
      });
      setOptionResult(result);
    } catch (err: any) {
      console.error('Error pricing option:', err);
      setError(err.response?.data?.detail || 'Failed to price option');
    } finally {
      setOptionLoading(false);
    }
  };

  const handleOptimizePortfolio = async () => {
    setPortfolioLoading(true);
    setError(null);

    try {
      const symbols = portfolioSymbols.split(',').map(s => s.trim()).filter(s => s.length > 0);

      if (symbols.length < 2) {
        setError('Please enter at least 2 symbols');
        setPortfolioLoading(false);
        return;
      }

      const result = await optimizePortfolio({
        symbols,
        target_return: targetReturn,
        lookback_days: 252
      });
      setPortfolioResult(result);
    } catch (err: any) {
      console.error('Error optimizing portfolio:', err);
      setError(err.response?.data?.detail || 'Failed to optimize portfolio');
    } finally {
      setPortfolioLoading(false);
    }
  };

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          🧬 Physics-Informed Neural Networks (PINN)
        </Typography>
        <Typography variant="subtitle1" color="textSecondary">
          Priority #4: 15-100x data efficiency through physics constraints
        </Typography>
      </Box>

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={activeTab} onChange={(e, v) => setActiveTab(v)}>
          <Tab label="Option Pricing" icon={<Calculate />} />
          <Tab label="Portfolio Optimization" icon={<TrendingUp />} />
          <Tab label="Explanation" icon={<Science />} />
        </Tabs>
      </Paper>

      {/* Error */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Tab 1: Option Pricing */}
      <TabPanel value={activeTab} index={0}>
        <Grid container spacing={3}>
          {/* Input Section */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Option Parameters
                </Typography>
                <Box sx={{ mt: 2 }}>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        type="number"
                        label="Stock Price ($)"
                        value={stockPrice}
                        onChange={(e) => setStockPrice(Number(e.target.value))}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        type="number"
                        label="Strike Price ($)"
                        value={strikePrice}
                        onChange={(e) => setStrikePrice(Number(e.target.value))}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        type="number"
                        label="Time to Maturity (years)"
                        value={timeToMaturity}
                        onChange={(e) => setTimeToMaturity(Number(e.target.value))}
                        inputProps={{ step: 0.1, min: 0.01 }}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <FormControl fullWidth>
                        <InputLabel>Option Type</InputLabel>
                        <Select
                          value={optionType}
                          onChange={(e) => setOptionType(e.target.value as 'call' | 'put')}
                          label="Option Type"
                        >
                          <MenuItem value="call">Call</MenuItem>
                          <MenuItem value="put">Put</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12}>
                      <Typography gutterBottom>
                        Volatility: {(volatility * 100).toFixed(0)}%
                      </Typography>
                      <Slider
                        value={volatility}
                        onChange={(e, v) => setVolatility(v as number)}
                        min={0.05}
                        max={1.0}
                        step={0.05}
                        marks={[
                          { value: 0.1, label: '10%' },
                          { value: 0.3, label: '30%' },
                          { value: 0.5, label: '50%' }
                        ]}
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <Typography gutterBottom>
                        Risk-Free Rate: {(riskFreeRate * 100).toFixed(1)}%
                      </Typography>
                      <Slider
                        value={riskFreeRate}
                        onChange={(e, v) => setRiskFreeRate(v as number)}
                        min={0.0}
                        max={0.10}
                        step={0.005}
                        marks={[
                          { value: 0.0, label: '0%' },
                          { value: 0.05, label: '5%' },
                          { value: 0.10, label: '10%' }
                        ]}
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <Button
                        fullWidth
                        variant="contained"
                        startIcon={<Calculate />}
                        onClick={handlePriceOption}
                        disabled={optionLoading}
                        size="large"
                      >
                        {optionLoading ? 'Pricing...' : 'Price Option (PINN)'}
                      </Button>
                    </Grid>
                  </Grid>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Results Section */}
          <Grid item xs={12} md={6}>
            {optionLoading && (
              <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
                <CircularProgress />
              </Box>
            )}

            {optionResult && !optionLoading && (
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Option Price & Greeks
                  </Typography>
                  <Box sx={{ mt: 2 }}>
                    {/* Price */}
                    <Paper sx={{ p: 2, mb: 2, bgcolor: 'primary.light', color: 'white' }}>
                      <Typography variant="body2">Option Price</Typography>
                      <Typography variant="h3">
                        ${optionResult.price.toFixed(2)}
                      </Typography>
                      <Typography variant="caption">
                        Method: {optionResult.method}
                      </Typography>
                    </Paper>

                    {/* Greeks */}
                    <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                      Greeks (Automatic Differentiation):
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={4}>
                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                          <Typography variant="caption" color="textSecondary">
                            Delta (∂V/∂S)
                          </Typography>
                          <Typography variant="h6">
                            {optionResult.greeks.delta !== null
                              ? optionResult.greeks.delta.toFixed(4)
                              : 'N/A'}
                          </Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={4}>
                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                          <Typography variant="caption" color="textSecondary">
                            Gamma (∂²V/∂S²)
                          </Typography>
                          <Typography variant="h6">
                            {optionResult.greeks.gamma !== null
                              ? optionResult.greeks.gamma.toFixed(4)
                              : 'N/A'}
                          </Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={4}>
                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                          <Typography variant="caption" color="textSecondary">
                            Theta (∂V/∂t)
                          </Typography>
                          <Typography variant="h6">
                            {optionResult.greeks.theta !== null
                              ? optionResult.greeks.theta.toFixed(4)
                              : 'N/A'}
                          </Typography>
                        </Paper>
                      </Grid>
                    </Grid>

                    {/* Details */}
                    <Alert severity="info" sx={{ mt: 2 }}>
                      <Typography variant="body2">
                        <strong>Physics Constraints:</strong> Black-Scholes PDE + Terminal payoff
                        + No-arbitrage conditions
                      </Typography>
                    </Alert>
                  </Box>
                </CardContent>
              </Card>
            )}
          </Grid>
        </Grid>
      </TabPanel>

      {/* Tab 2: Portfolio Optimization */}
      <TabPanel value={activeTab} index={1}>
        <Grid container spacing={3}>
          {/* Input Section */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Portfolio Parameters
                </Typography>
                <Box sx={{ mt: 2 }}>
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        label="Stock Symbols (comma-separated)"
                        value={portfolioSymbols}
                        onChange={(e) => setPortfolioSymbols(e.target.value)}
                        placeholder="AAPL, MSFT, GOOGL, ..."
                        helperText="Enter 2 or more symbols"
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <Typography gutterBottom>
                        Target Annual Return: {(targetReturn * 100).toFixed(0)}%
                      </Typography>
                      <Slider
                        value={targetReturn}
                        onChange={(e, v) => setTargetReturn(v as number)}
                        min={0.05}
                        max={0.30}
                        step={0.01}
                        marks={[
                          { value: 0.05, label: '5%' },
                          { value: 0.15, label: '15%' },
                          { value: 0.30, label: '30%' }
                        ]}
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <Button
                        fullWidth
                        variant="contained"
                        startIcon={<AutoGraph />}
                        onClick={handleOptimizePortfolio}
                        disabled={portfolioLoading}
                        size="large"
                      >
                        {portfolioLoading ? 'Optimizing...' : 'Optimize Portfolio (PINN)'}
                      </Button>
                    </Grid>
                  </Grid>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Results Section */}
          <Grid item xs={12} md={6}>
            {portfolioLoading && (
              <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
                <CircularProgress />
              </Box>
            )}

            {portfolioResult && !portfolioLoading && (
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Optimal Portfolio
                  </Typography>

                  {/* Key Metrics */}
                  <Grid container spacing={2} sx={{ mb: 2 }}>
                    <Grid item xs={4}>
                      <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'success.light', color: 'white' }}>
                        <Typography variant="caption">Expected Return</Typography>
                        <Typography variant="h6">
                          {(portfolioResult.expected_return * 100).toFixed(1)}%
                        </Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={4}>
                      <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'warning.light', color: 'white' }}>
                        <Typography variant="caption">Risk (Volatility)</Typography>
                        <Typography variant="h6">
                          {(portfolioResult.risk * 100).toFixed(1)}%
                        </Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={4}>
                      <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'info.light', color: 'white' }}>
                        <Typography variant="caption">Sharpe Ratio</Typography>
                        <Typography variant="h6">
                          {portfolioResult.sharpe_ratio.toFixed(2)}
                        </Typography>
                      </Paper>
                    </Grid>
                  </Grid>

                  {/* Weights */}
                  <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                    Optimal Weights:
                  </Typography>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell><strong>Symbol</strong></TableCell>
                        <TableCell align="right"><strong>Weight</strong></TableCell>
                        <TableCell align="right"><strong>Allocation</strong></TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {portfolioResult.symbols.map((symbol, idx) => (
                        <TableRow key={symbol}>
                          <TableCell>{symbol}</TableCell>
                          <TableCell align="right">
                            {(portfolioResult.weights[idx] * 100).toFixed(1)}%
                          </TableCell>
                          <TableCell align="right">
                            <Chip
                              label={portfolioResult.weights[idx] > 0.15 ? 'High' : portfolioResult.weights[idx] > 0.05 ? 'Medium' : 'Low'}
                              size="small"
                              color={portfolioResult.weights[idx] > 0.15 ? 'success' : 'default'}
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>

                  <Alert severity="info" sx={{ mt: 2 }}>
                    <Typography variant="body2">
                      <strong>Constraints Satisfied:</strong> Budget (Σw=1) + No short-selling (w≥0) + Target return
                    </Typography>
                  </Alert>
                </CardContent>
              </Card>
            )}
          </Grid>
        </Grid>
      </TabPanel>

      {/* Tab 3: Explanation */}
      <TabPanel value={activeTab} index={2}>
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

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Option Pricing Application
                  </Typography>
                  <Table size="small">
                    <TableBody>
                      {Object.entries(explanation.applications['Option Pricing']).map(([key, value]) => (
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
                    Portfolio Optimization Application
                  </Typography>
                  <Table size="small">
                    <TableBody>
                      {Object.entries(explanation.applications['Portfolio Optimization']).map(([key, value]) => (
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

export default PINNPage;
