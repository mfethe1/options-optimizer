/**
 * Swarm Health Metrics Component
 * Displays health and performance metrics for the 17-agent swarm
 */
import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,

  Paper,
  LinearProgress,
  Divider,
  Alert
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Speed as SpeedIcon,
  Forum as ForumIcon,
  TrendingUp as TrendingUpIcon
} from '@mui/icons-material';

import Grid from '@mui/material/Grid';




interface SwarmHealthMetricsProps {

  health: {
    active_agents_count: number;
    contributed_vs_failed: {
      contributed: number;
      failed: number;
      success_rate: number;
    };
    communication_stats: {
      total_messages: number;
      total_state_updates: number;
      average_message_priority: number;
      average_confidence: number;
    };
    consensus_strength: {
      overall_action_confidence: number;
      risk_level_confidence: number;
      market_outlook_confidence: number;
      average_confidence: number;
    };
  };
}

const SwarmHealthMetrics: React.FC<SwarmHealthMetricsProps> = ({ health }) => {
  const getHealthColor = (rate: number) => {
    if (rate >= 90) return 'success';
    if (rate >= 70) return 'warning';
    return 'error';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <SpeedIcon color="primary" />
          Swarm Health & Performance
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Real-time metrics from the 17-agent institutional-grade swarm
        </Typography>
        <Divider sx={{ mb: 3 }} />

        <Grid container spacing={3}>
          {/* Agent Contribution */}
          <Grid size={{ xs: 12, md: 6 }}>
            <Paper elevation={2} sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <CheckCircleIcon color="success" />
                Agent Contribution
              </Typography>

              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">Success Rate</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {health.contributed_vs_failed.success_rate.toFixed(1)}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={health.contributed_vs_failed.success_rate}
                  color={getHealthColor(health.contributed_vs_failed.success_rate)}
                  sx={{ height: 8, borderRadius: 1 }}
                />
              </Box>

              <Grid container spacing={2}>
                <Grid size={4}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" color="primary.main">
                      {health.active_agents_count}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Total Agents
                    </Typography>
                  </Box>
                </Grid>
                <Grid size={4}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" color="success.main">
                      {health.contributed_vs_failed.contributed}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Contributed
                    </Typography>
                  </Box>
                </Grid>
                <Grid size={4}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" color="error.main">
                      {health.contributed_vs_failed.failed}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Failed
                    </Typography>
                  </Box>
                </Grid>
              </Grid>

              {health.contributed_vs_failed.failed > 0 && (
                <Alert severity="warning" sx={{ mt: 2 }}>
                  {health.contributed_vs_failed.failed} agent(s) failed to contribute. Check logs for details.
                </Alert>
              )}
            </Paper>
          </Grid>

          {/* Communication Stats */}
          <Grid size={{ xs: 12, md: 6 }}>
            <Paper elevation={2} sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <ForumIcon color="primary" />
                Communication Stats
              </Typography>

              <Grid container spacing={2}>
                <Grid size={6}>
                  <Box sx={{ textAlign: 'center', mb: 2 }}>
                    <Typography variant="h4" color="primary.main">
                      {health.communication_stats.total_messages}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Total Messages
                    </Typography>
                  </Box>
                </Grid>
                <Grid size={6}>
                  <Box sx={{ textAlign: 'center', mb: 2 }}>
                    <Typography variant="h4" color="primary.main">
                      {health.communication_stats.total_state_updates}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      State Updates
                    </Typography>
                  </Box>
                </Grid>
              </Grid>

              <Divider sx={{ my: 2 }} />

              <Box sx={{ mb: 1 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                  <Typography variant="body2">Avg Message Priority</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {health.communication_stats.average_message_priority.toFixed(1)}
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={(health.communication_stats.average_message_priority / 10) * 100}
                  color="primary"
                  sx={{ height: 6, borderRadius: 1 }}
                />
              </Box>

              <Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                  <Typography variant="body2">Avg Confidence</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {(health.communication_stats.average_confidence * 100).toFixed(0)}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={health.communication_stats.average_confidence * 100}
                  color={getConfidenceColor(health.communication_stats.average_confidence)}
                  sx={{ height: 6, borderRadius: 1 }}
                />
              </Box>
            </Paper>
          </Grid>

          {/* Consensus Strength */}
          <Grid size={12}>
            <Paper elevation={2} sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <TrendingUpIcon color="success" />
                Consensus Strength
              </Typography>

              <Grid container spacing={2}>
                <Grid size={{ xs: 12, md: 3 }}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" color={getConfidenceColor(health.consensus_strength.average_confidence)}>
                      {(health.consensus_strength.average_confidence * 100).toFixed(0)}%
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Overall Consensus
                    </Typography>
                  </Box>
                </Grid>

                <Grid size={{ xs: 12, md: 3 }}>
                  <Box>
                    <Typography variant="body2" gutterBottom>Overall Action</Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <LinearProgress
                        variant="determinate"
                        value={health.consensus_strength.overall_action_confidence * 100}
                        color={getConfidenceColor(health.consensus_strength.overall_action_confidence)}
                        sx={{ flexGrow: 1, height: 8, borderRadius: 1 }}
                      />
                      <Typography variant="caption" fontWeight="bold">
                        {(health.consensus_strength.overall_action_confidence * 100).toFixed(0)}%
                      </Typography>
                    </Box>
                  </Box>
                </Grid>

                <Grid size={{ xs: 12, md: 3 }}>
                  <Box>
                    <Typography variant="body2" gutterBottom>Risk Level</Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <LinearProgress
                        variant="determinate"
                        value={health.consensus_strength.risk_level_confidence * 100}
                        color={getConfidenceColor(health.consensus_strength.risk_level_confidence)}
                        sx={{ flexGrow: 1, height: 8, borderRadius: 1 }}
                      />
                      <Typography variant="caption" fontWeight="bold">
                        {(health.consensus_strength.risk_level_confidence * 100).toFixed(0)}%
                      </Typography>
                    </Box>
                  </Box>
                </Grid>

                <Grid size={{ xs: 12, md: 3 }}>
                  <Box>
                    <Typography variant="body2" gutterBottom>Market Outlook</Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <LinearProgress
                        variant="determinate"
                        value={health.consensus_strength.market_outlook_confidence * 100}
                        color={getConfidenceColor(health.consensus_strength.market_outlook_confidence)}
                        sx={{ flexGrow: 1, height: 8, borderRadius: 1 }}
                      />
                      <Typography variant="caption" fontWeight="bold">
                        {(health.consensus_strength.market_outlook_confidence * 100).toFixed(0)}%
                      </Typography>
                    </Box>
                  </Box>
                </Grid>
              </Grid>

              {health.consensus_strength.average_confidence < 0.6 && (
                <Alert severity="warning" sx={{ mt: 2 }}>
                  Low consensus confidence detected. Agents have divergent opinions - review individual agent insights carefully.
                </Alert>
              )}
            </Paper>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default SwarmHealthMetrics;

