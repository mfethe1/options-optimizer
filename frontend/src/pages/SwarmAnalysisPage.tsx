import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Alert,
  Chip,
  Paper,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControlLabel,
  Checkbox,
  LinearProgress
} from '@mui/material';
import {
  Upload as UploadIcon,
  Psychology as BrainIcon,
  CheckCircle as CheckIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import { swarmService, SwarmAnalysisResult } from '../services/swarmService';
import PositionAnalysisPanel from '../components/PositionAnalysisPanel';
import SwarmHealthMetrics from '../components/SwarmHealthMetrics';
import Grid from '@mui/material/Grid';

import AgentConversationViewer from '../components/AgentConversationViewer';
import AnalysisProgressTracker from '../components/AnalysisProgressTracker';
import InvestorReportSynopsis from '../components/InvestorReportSynopsis';

const SwarmAnalysisPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isChaseFormat, setIsChaseFormat] = useState(false);
  const [loading, setLoading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<SwarmAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [openUploadDialog, setOpenUploadDialog] = useState(false);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select a CSV file');
      return;
    }

    setLoading(true);
    setError(null);
    setOpenUploadDialog(false);

    try {
      const result = await swarmService.analyzeFromCSV(
        selectedFile,
        isChaseFormat,
        'weighted'
      );
      setAnalysisResult(result);
    } catch (err: any) {
      setError(err.message || 'Failed to analyze portfolio');
    } finally {
      setLoading(false);
    }
  };

  const getActionColor = (action: string) => {
    switch (action.toLowerCase()) {
      case 'buy':
        return 'success';
      case 'sell':
        return 'error';
      case 'hold':
        return 'warning';
      case 'hedge':
        return 'info';
      default:
        return 'default';
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk.toLowerCase()) {
      case 'conservative':
        return 'success';
      case 'moderate':
        return 'warning';
      case 'aggressive':
        return 'error';
      default:
        return 'default';
    }
  };

  const getOutlookColor = (outlook: string) => {
    switch (outlook.toLowerCase()) {
      case 'bullish':
        return 'success';
      case 'bearish':
        return 'error';
      case 'neutral':
        return 'warning';
      default:
        return 'default';
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <BrainIcon fontSize="large" />
          AI-Powered Swarm Analysis
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Upload your portfolio CSV and let our multi-agent AI swarm analyze it using Claude, GPT-4, and LMStudio
        </Typography>
      </Box>

      {/* Upload Section */}
      {!analysisResult && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Upload Portfolio CSV
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Upload a CSV file containing your positions. Supports standard format and Chase.com exports.
            </Typography>

            <Button
              variant="contained"
              startIcon={<UploadIcon />}
              onClick={() => setOpenUploadDialog(true)}
              size="large"
            >
              Select CSV File
            </Button>

            {selectedFile && (
              <Alert severity="info" sx={{ mt: 2 }}>
                Selected: {selectedFile.name}
              </Alert>
            )}
          </CardContent>
        </Card>
      )}

      {/* Loading State with Progress Tracker */}
      {loading && (
        <>
          <AnalysisProgressTracker
            isAnalyzing={loading}
            currentStep="Running 17-agent swarm analysis..."
            estimatedTimeRemaining={180}
          />
          <Card>
            <CardContent>
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <CircularProgress size={60} sx={{ mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  Analyzing Portfolio with AI Swarm...
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Our 17 LLM-powered agents are analyzing your positions
                </Typography>
                <LinearProgress sx={{ mt: 2 }} />
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                  This may take 3-5 minutes for comprehensive analysis
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </>
      )}

      {/* Error State */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Analysis Results */}
      {analysisResult && !loading && (
        <Box>
          {/* Import Stats */}
          {analysisResult.import_stats && (
            <Alert severity="success" sx={{ mb: 3 }}>
              <Typography variant="body2">
                ✅ Imported {analysisResult.import_stats.positions_imported} positions
                {analysisResult.import_stats.chase_conversion && (
                  <> • Converted {analysisResult.import_stats.chase_conversion.options_converted} options from Chase format</>
                )}
              </Typography>
            </Alert>
          )}

          {/* Investor-Friendly Synopsis (blog/chat style) */}
          {analysisResult.investor_report && (
            <Box sx={{ mb: 3 }}>
              <InvestorReportSynopsis report={analysisResult.investor_report} />
            </Box>
          )}

          {/* Technical Details - Collapsible */}
          <details style={{ marginBottom: '24px' }}>
            <summary style={{
              cursor: 'pointer',
              padding: '16px',
              background: '#f5f5f5',
              borderRadius: '8px',
              fontWeight: 600,
              fontSize: '1.1rem'
            }}>
              📊 Technical Analysis Details (Click to expand)
            </summary>
            <Box sx={{ mt: 2 }}>

          {/* Consensus Decisions */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <CheckIcon color="success" />
                AI Consensus Recommendations
              </Typography>
              <Divider sx={{ my: 2 }} />

              <Grid container spacing={3}>
                {/* Overall Action */}
                <Grid size={{ xs: 12, md: 4 }}>
                  <Paper elevation={2} sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="overline" color="text.secondary">
                      Overall Action
                    </Typography>
                    <Box sx={{ my: 2 }}>
                      <Chip
                        label={analysisResult.consensus_decisions.overall_action.choice.toUpperCase()}
                        color={getActionColor(analysisResult.consensus_decisions.overall_action.choice)}
                        size="medium"
                        sx={{ fontSize: '1.2rem', py: 3, px: 2 }}
                      />
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      Confidence: {(analysisResult.consensus_decisions.overall_action.confidence * 100).toFixed(0)}%
                    </Typography>
                    <Typography variant="body2" sx={{ mt: 1 }}>
                      {analysisResult.consensus_decisions.overall_action.reasoning}
                    </Typography>
                  </Paper>
                </Grid>

                {/* Risk Level */}
                <Grid size={{ xs: 12, md: 4 }}>
                  <Paper elevation={2} sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="overline" color="text.secondary">
                      Risk Level
                    </Typography>
                    <Box sx={{ my: 2 }}>
                      <Chip
                        label={analysisResult.consensus_decisions.risk_level.choice.toUpperCase()}
                        color={getRiskColor(analysisResult.consensus_decisions.risk_level.choice)}
                        size="medium"
                        sx={{ fontSize: '1.2rem', py: 3, px: 2 }}
                      />
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      Confidence: {(analysisResult.consensus_decisions.risk_level.confidence * 100).toFixed(0)}%
                    </Typography>
                    <Typography variant="body2" sx={{ mt: 1 }}>
                      {analysisResult.consensus_decisions.risk_level.reasoning}
                    </Typography>
                  </Paper>
                </Grid>

                {/* Market Outlook */}
                <Grid size={{ xs: 12, md: 4 }}>
                  <Paper elevation={2} sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="overline" color="text.secondary">
                      Market Outlook
                    </Typography>
                    <Box sx={{ my: 2 }}>
                      <Chip
                        label={analysisResult.consensus_decisions.market_outlook.choice.toUpperCase()}
                        color={getOutlookColor(analysisResult.consensus_decisions.market_outlook.choice)}
                        size="medium"
                        sx={{ fontSize: '1.2rem', py: 3, px: 2 }}
                      />
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      Confidence: {(analysisResult.consensus_decisions.market_outlook.confidence * 100).toFixed(0)}%
                    </Typography>
                    <Typography variant="body2" sx={{ mt: 1 }}>
                      {analysisResult.consensus_decisions.market_outlook.reasoning}
                    </Typography>
                  </Paper>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {/* Portfolio Summary */}
          {analysisResult.portfolio_summary && (
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Portfolio Summary
                </Typography>
                <Grid container spacing={2}>
                  <Grid size={{ xs: 6, md: 3 }}>
                    <Typography variant="body2" color="text.secondary">Total Value</Typography>
                    <Typography variant="h6">${analysisResult.portfolio_summary.total_value.toLocaleString()}</Typography>
                  </Grid>
                  <Grid size={{ xs: 6, md: 3 }}>
                    <Typography variant="body2" color="text.secondary">Unrealized P&L</Typography>
                    <Typography variant="h6" color={analysisResult.portfolio_summary.total_unrealized_pnl >= 0 ? 'success.main' : 'error.main'}>
                      ${analysisResult.portfolio_summary.total_unrealized_pnl.toLocaleString()}
                    </Typography>
                  </Grid>
                  <Grid size={{ xs: 6, md: 3 }}>
                    <Typography variant="body2" color="text.secondary">P&L %</Typography>
                    <Typography variant="h6" color={analysisResult.portfolio_summary.total_unrealized_pnl_pct >= 0 ? 'success.main' : 'error.main'}>
                      {analysisResult.portfolio_summary.total_unrealized_pnl_pct.toFixed(2)}%
                    </Typography>
                  </Grid>
                  <Grid size={{ xs: 6, md: 3 }}>
                    <Typography variant="body2" color="text.secondary">Positions</Typography>
                    <Typography variant="h6">{analysisResult.portfolio_summary.positions_count}</Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          )}

          {/* NEW: Swarm Health Metrics */}
          {analysisResult.swarm_health && (
            <Box sx={{ mb: 3 }}>
              <SwarmHealthMetrics health={analysisResult.swarm_health} />
            </Box>
          )}

          {/* NEW: Position-by-Position Analysis */}
          {analysisResult.position_analysis && analysisResult.position_analysis.length > 0 && (
            <Box sx={{ mb: 3 }}>
              <PositionAnalysisPanel positions={analysisResult.position_analysis} />
            </Box>
          )}

          {/* NEW: Agent Conversation Logs */}
          {analysisResult.discussion_logs && analysisResult.discussion_logs.length > 0 && (
            <Box sx={{ mb: 3 }}>
              <AgentConversationViewer
                discussionLogs={analysisResult.discussion_logs}
                agentInsights={analysisResult.agent_insights}
              />
            </Box>
          )}

          </Box>
          </details>

          {/* Action Buttons */}
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
              variant="outlined"
              onClick={() => {
                setAnalysisResult(null);
                setSelectedFile(null);
              }}
            >
              Analyze Another Portfolio
            </Button>
          </Box>
        </Box>
      )}

      {/* Upload Dialog */}
      <Dialog open={openUploadDialog} onClose={() => setOpenUploadDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Upload Portfolio CSV</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <Button variant="outlined" component="label" startIcon={<UploadIcon />}>
              Choose CSV File
              <input
                type="file"
                accept=".csv"
                hidden
                onChange={handleFileSelect}
              />
            </Button>

            {selectedFile && (
              <Alert severity="info">
                Selected: {selectedFile.name}
              </Alert>
            )}

            <FormControlLabel
              control={
                <Checkbox
                  checked={isChaseFormat}
                  onChange={(e) => setIsChaseFormat(e.target.checked)}
                />
              }
              label="This is a Chase.com export (auto-convert)"
            />

            <Alert severity="info" icon={<InfoIcon />}>
              <Typography variant="body2">
                The AI swarm will analyze your portfolio using:
              </Typography>
              <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
                <li>Market Analyst (Claude) - Market conditions</li>
                <li>Risk Manager (Claude) - Portfolio risk</li>
                <li>Sentiment Analyst (LMStudio) - Market sentiment</li>
              </ul>
            </Alert>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenUploadDialog(false)}>Cancel</Button>
          <Button onClick={handleAnalyze} variant="contained" disabled={!selectedFile}>
            Analyze with AI
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default SwarmAnalysisPage;

