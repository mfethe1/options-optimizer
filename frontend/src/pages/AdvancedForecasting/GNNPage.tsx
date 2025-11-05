/**
 * Graph Neural Network Dashboard
 * Priority #2: Universal Consensus Feature
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Paper,
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
  TableRow
} from '@mui/material';
import Grid from '@mui/material/Grid2';
import { Refresh, Timeline, Share } from '@mui/icons-material';
import { getGNNForecast, getStatus, GNNForecast } from '../../api/gnnApi';

const DEFAULT_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM'];

const GNNPage: React.FC = () => {
  const [symbols, setSymbols] = useState<string>(DEFAULT_SYMBOLS.join(', '));
  const [forecast, setForecast] = useState<GNNForecast | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadForecast = async () => {
    setLoading(true);
    setError(null);

    try {
      const symbolList = symbols.split(',').map(s => s.trim()).filter(s => s.length > 0);

      if (symbolList.length < 2) {
        setError('Please enter at least 2 symbols');
        setLoading(false);
        return;
      }

      const data = await getGNNForecast(symbolList, 20);
      setForecast(data);
    } catch (err: any) {
      console.error('Error loading GNN forecast:', err);
      setError(err.response?.data?.detail || 'Failed to load forecast');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadForecast();
  }, []);

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          📊 Graph Neural Network - Stock Correlations
        </Typography>
        <Typography variant="subtitle1" color="textSecondary">
          Priority #2: Universal Consensus - All 3 agents identified as CRITICAL
        </Typography>
      </Box>

      {/* Input Section */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid size={{ xs: 12, md: 8 }}>
            <TextField
              fullWidth
              label="Stock Symbols (comma-separated)"
              value={symbols}
              onChange={(e) => setSymbols(e.target.value)}
              placeholder="AAPL, MSFT, GOOGL, ..."
              helperText="Enter 2 or more symbols to analyze correlations"
            />
          </Grid>
          <Grid size={{ xs: 12, md: 4 }}>
            <Button
              fullWidth
              variant="contained"
              startIcon={<Timeline />}
              onClick={loadForecast}
              disabled={loading}
              sx={{ height: 56 }}
            >
              {loading ? 'Analyzing...' : 'Analyze Graph'}
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
          {/* Graph Stats */}
          <Grid size={{ xs: 12, md: 4 }}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Graph Statistics
                </Typography>
                <Box sx={{ mt: 2 }}>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="textSecondary">
                      Stocks Analyzed:
                    </Typography>
                    <Typography variant="h5">
                      {forecast.graph_stats.num_nodes}
                    </Typography>
                  </Box>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="textSecondary">
                      Correlation Edges:
                    </Typography>
                    <Typography variant="h5">
                      {forecast.graph_stats.num_edges}
                    </Typography>
                  </Box>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="textSecondary">
                      Avg Correlation:
                    </Typography>
                    <Typography variant="h5">
                      {(forecast.graph_stats.avg_correlation * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="body2" color="textSecondary">
                      Max Correlation:
                    </Typography>
                    <Typography variant="h5">
                      {(forecast.graph_stats.max_correlation * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Predictions */}
          <Grid size={{ xs: 12, md: 8 }}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  GNN Predictions (Expected Returns)
                </Typography>
                <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell><strong>Symbol</strong></TableCell>
                        <TableCell align="right"><strong>Prediction</strong></TableCell>
                        <TableCell align="right"><strong>Signal</strong></TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(forecast.predictions).map(([symbol, pred]) => {
                        const signal = pred > 0.02 ? 'BUY' : pred < -0.02 ? 'SELL' : 'HOLD';
                        const color = signal === 'BUY' ? 'success' : signal === 'SELL' ? 'error' : 'default';

                        return (
                          <TableRow key={symbol}>
                            <TableCell>{symbol}</TableCell>
                            <TableCell align="right">
                              {pred >= 0 ? '+' : ''}{(pred * 100).toFixed(2)}%
                            </TableCell>
                            <TableCell align="right">
                              <Chip label={signal} color={color as any} size="small" />
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Top Correlations */}
          <Grid size={{ xs: 12 }}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Strongest Correlations
                </Typography>
                <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                  Stock pairs with correlation &gt; 50%
                </Typography>
                <Grid container spacing={2}>
                  {forecast.top_correlations.map((corr, idx) => (
                    <Grid size={{ xs: 12, sm: 6, md: 4 }} key={idx}>
                      <Paper sx={{ p: 2, bgcolor: 'grey.100' }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                          <Chip label={corr.symbol1} size="small" />
                          <Share fontSize="small" />
                          <Chip label={corr.symbol2} size="small" />
                        </Box>
                        <Typography variant="h6" color="primary">
                          {(corr.correlation * 100).toFixed(1)}%
                        </Typography>
                        <Typography variant="caption" color="textSecondary">
                          Correlation strength
                        </Typography>
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
                {forecast.top_correlations.length === 0 && (
                  <Alert severity="info">
                    No strong correlations (&gt;50%) found. Try adding more related stocks (same sector).
                  </Alert>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Info */}
          <Grid size={{ xs: 12 }}>
            <Alert severity="info">
              <Typography variant="subtitle2" gutterBottom>
                <strong>How it works:</strong>
              </Typography>
              <Typography variant="body2">
                Graph Neural Networks leverage stock correlations for improved predictions.
                Stocks are nodes, correlations are edges. GNN learns from the graph structure
                to capture inter-stock dependencies that univariate models miss. Expected 20-30%
                improvement via correlation exploitation.
              </Typography>
            </Alert>
          </Grid>
        </Grid>
      )}
    </Container>
  );
};

export default GNNPage;
