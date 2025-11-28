import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Alert,
  CircularProgress,
  Chip,
  IconButton,
  Tooltip
} from '@mui/material';
import Grid from '@mui/material/Grid';
import { Refresh, Info, Schedule } from '@mui/icons-material';
import {
  ModelAccuracyCard,
  ModelAccuracyCardSkeleton,
  type ModelAccuracy
} from '../components/ModelAccuracyCard';

/**
 * API Response interface for daily accuracy endpoint
 */
interface DailyAccuracyResponse {
  models: ModelAccuracy[];
  updated_at: string;
}

/**
 * TruthDashboard - Model accuracy tracking dashboard
 *
 * Purpose: Display model accuracy tracking - comparing predictions to actuals
 *
 * Features:
 * - Progress bar showing accuracy vs 55% target
 * - Color coding: green (>55%), yellow (50-55%), red (<50%)
 * - Last 5 predictions shown as checkmarks/X marks
 * - Status badge: "Healthy", "Degraded", "Untrained"
 * - Auto-refresh every 5 minutes
 * - Loading skeleton while fetching
 * - Error state with retry button
 * - "Last Updated" timestamp
 */
const TruthDashboard: React.FC = () => {
  const [models, setModels] = useState<ModelAccuracy[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);

  // API base URL
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

  /**
   * Fetch model accuracy data from API
   */
  const fetchAccuracyData = useCallback(async (showLoadingState = true) => {
    if (showLoadingState) {
      setLoading(true);
    } else {
      setIsRefreshing(true);
    }
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/truth/daily-accuracy`);

      if (!response.ok) {
        throw new Error(`Failed to fetch accuracy data (${response.status})`);
      }

      const data = await response.json();

      // Convert models object to array if needed
      let modelsArray: ModelAccuracy[];
      if (Array.isArray(data.models)) {
        modelsArray = data.models;
      } else if (data.models && typeof data.models === 'object') {
        // API returns models as object, convert to array
        modelsArray = Object.entries(data.models).map(([name, stats]: [string, any]) => ({
          model_name: name,
          direction_accuracy_30d: stats.direction_accuracy_30d || 0,
          mape_30d: stats.mape_30d || 0,
          total_predictions: stats.total_predictions || 0,
          correct_predictions: stats.correct_predictions || 0,
          last_5: stats.last_5_results || [],
          status: stats.status || 'untrained',
          last_updated: stats.last_updated || new Date().toISOString(),
        }));
      } else {
        modelsArray = [];
      }

      setModels(modelsArray);
      setLastUpdated(data.timestamp || data.updated_at);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load accuracy data';
      setError(errorMessage);
      console.error('[TruthDashboard] Error fetching accuracy data:', err);
    } finally {
      setLoading(false);
      setIsRefreshing(false);
    }
  }, [API_BASE_URL]);

  // Initial fetch
  useEffect(() => {
    fetchAccuracyData(true);
  }, [fetchAccuracyData]);

  // Auto-refresh every 5 minutes
  useEffect(() => {
    const intervalId = setInterval(() => {
      fetchAccuracyData(false);
    }, 5 * 60 * 1000); // 5 minutes

    return () => clearInterval(intervalId);
  }, [fetchAccuracyData]);

  /**
   * Format the last updated timestamp
   */
  const formatLastUpdated = (timestamp: string | null): string => {
    if (!timestamp) return 'Never';

    try {
      const date = new Date(timestamp);
      return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        hour12: true,
        timeZoneName: 'short'
      });
    } catch {
      return timestamp;
    }
  };

  /**
   * Calculate summary statistics
   */
  const getSummaryStats = () => {
    if (models.length === 0) return null;

    const healthyCount = models.filter(m => m.status === 'healthy').length;
    const degradedCount = models.filter(m => m.status === 'degraded').length;
    const untrainedCount = models.filter(m => m.status === 'untrained').length;
    const avgAccuracy = models.reduce((sum, m) => sum + m.direction_accuracy_30d, 0) / models.length;
    const modelsAboveTarget = models.filter(m => m.direction_accuracy_30d >= 55).length;

    return {
      healthyCount,
      degradedCount,
      untrainedCount,
      avgAccuracy,
      modelsAboveTarget,
      totalModels: models.length
    };
  };

  const stats = getSummaryStats();

  return (
    <Box
      sx={{
        minHeight: '100vh',
        bgcolor: '#0A0E27',
        p: 3
      }}
    >
      {/* Header */}
      <Paper
        sx={{
          p: 3,
          mb: 3,
          bgcolor: '#1A1F3A',
          border: '1px solid rgba(255,255,255,0.1)'
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
          <Box>
            <Typography
              variant="h4"
              sx={{
                color: 'white',
                fontWeight: 700,
                mb: 0.5
              }}
            >
              TRUTH DASHBOARD
            </Typography>
            <Typography
              variant="body2"
              sx={{
                color: 'rgba(255,255,255,0.6)',
                display: 'flex',
                alignItems: 'center',
                gap: 0.5
              }}
            >
              <Schedule sx={{ fontSize: 16 }} />
              Updated Daily 4:30 PM ET
            </Typography>
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            {/* Last updated indicator */}
            <Box sx={{ textAlign: 'right' }}>
              <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.5)', display: 'block' }}>
                Last Updated
              </Typography>
              <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.8)' }}>
                {formatLastUpdated(lastUpdated)}
              </Typography>
            </Box>

            {/* Refresh button */}
            <Tooltip title="Refresh data">
              <IconButton
                onClick={() => fetchAccuracyData(false)}
                disabled={isRefreshing || loading}
                sx={{
                  color: '#64B5F6',
                  '&:hover': { bgcolor: 'rgba(100, 181, 246, 0.1)' }
                }}
              >
                {isRefreshing ? (
                  <CircularProgress size={24} sx={{ color: '#64B5F6' }} />
                ) : (
                  <Refresh />
                )}
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Summary Stats */}
        {stats && (
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <Chip
              label={`${stats.modelsAboveTarget}/${stats.totalModels} models above target`}
              sx={{
                bgcolor: stats.modelsAboveTarget >= stats.totalModels / 2 ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                color: stats.modelsAboveTarget >= stats.totalModels / 2 ? '#10b981' : '#ef4444',
                fontWeight: 600
              }}
            />
            <Chip
              label={`Avg: ${stats.avgAccuracy.toFixed(1)}%`}
              sx={{
                bgcolor: 'rgba(100, 181, 246, 0.2)',
                color: '#64B5F6',
                fontWeight: 600
              }}
            />
            {stats.healthyCount > 0 && (
              <Chip
                label={`${stats.healthyCount} Healthy`}
                size="small"
                sx={{
                  bgcolor: 'rgba(16, 185, 129, 0.1)',
                  color: '#10b981'
                }}
              />
            )}
            {stats.degradedCount > 0 && (
              <Chip
                label={`${stats.degradedCount} Degraded`}
                size="small"
                sx={{
                  bgcolor: 'rgba(245, 158, 11, 0.1)',
                  color: '#f59e0b'
                }}
              />
            )}
            {stats.untrainedCount > 0 && (
              <Chip
                label={`${stats.untrainedCount} Untrained`}
                size="small"
                sx={{
                  bgcolor: 'rgba(107, 114, 128, 0.1)',
                  color: '#6b7280'
                }}
              />
            )}
          </Box>
        )}
      </Paper>

      {/* Info Box */}
      <Paper
        sx={{
          p: 2,
          mb: 3,
          bgcolor: 'rgba(59, 130, 246, 0.1)',
          border: '1px solid rgba(59, 130, 246, 0.3)'
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
          <Info sx={{ color: '#3B82F6', fontSize: 20, mt: 0.3 }} />
          <Box>
            <Typography variant="body2" sx={{ color: '#93C5FD', fontWeight: 600, mb: 0.5 }}>
              Model Accuracy (Last 30 Days)
            </Typography>
            <Typography variant="caption" sx={{ color: 'rgba(147, 197, 253, 0.8)' }}>
              Direction accuracy measures how often the model correctly predicts whether the price will go up or down.
              A target of 55% is considered statistically significant for trading purposes.
              MAPE (Mean Absolute Percentage Error) measures the average magnitude of prediction errors.
            </Typography>
          </Box>
        </Box>
      </Paper>

      {/* Error State */}
      {error && (
        <Alert
          severity="error"
          sx={{
            mb: 3,
            bgcolor: 'rgba(239, 68, 68, 0.1)',
            color: '#ef4444',
            '& .MuiAlert-icon': { color: '#ef4444' }
          }}
          action={
            <Button
              color="inherit"
              size="small"
              onClick={() => fetchAccuracyData(true)}
              sx={{ fontWeight: 600 }}
            >
              RETRY
            </Button>
          }
        >
          {error}
        </Alert>
      )}

      {/* Model Cards Grid */}
      <Typography
        variant="h6"
        sx={{
          color: 'white',
          mb: 2,
          pb: 1,
          borderBottom: '1px solid rgba(255,255,255,0.1)'
        }}
      >
        MODEL ACCURACY (Last 30 Days)
      </Typography>

      <Grid container spacing={3}>
        {loading ? (
          // Loading Skeletons
          <>
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <Grid size={{ xs: 12, sm: 6, lg: 4 }} key={i}>
                <ModelAccuracyCardSkeleton />
              </Grid>
            ))}
          </>
        ) : models.length === 0 ? (
          // Empty State
          <Grid size={{ xs: 12 }}>
            <Paper
              sx={{
                p: 4,
                bgcolor: '#1A1F3A',
                border: '1px solid rgba(255,255,255,0.1)',
                textAlign: 'center'
              }}
            >
              <Typography variant="h6" sx={{ color: 'rgba(255,255,255,0.6)', mb: 1 }}>
                No Model Data Available
              </Typography>
              <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.4)', mb: 2 }}>
                Model accuracy data will appear here once predictions have been tracked.
              </Typography>
              <Button
                variant="outlined"
                onClick={() => fetchAccuracyData(true)}
                sx={{
                  color: '#64B5F6',
                  borderColor: '#64B5F6',
                  '&:hover': {
                    borderColor: '#90CAF9',
                    bgcolor: 'rgba(100, 181, 246, 0.1)'
                  }
                }}
              >
                Refresh
              </Button>
            </Paper>
          </Grid>
        ) : (
          // Model Cards
          models.map((model) => (
            <Grid size={{ xs: 12, sm: 6, lg: 4 }} key={model.model_name}>
              <ModelAccuracyCard model={model} targetAccuracy={55} />
            </Grid>
          ))
        )}
      </Grid>

      {/* Footer Info */}
      <Paper
        sx={{
          p: 2,
          mt: 3,
          bgcolor: '#1A1F3A',
          border: '1px solid rgba(255,255,255,0.1)'
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 2 }}>
          <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.5)' }}>
            Auto-refreshes every 5 minutes. Manual refresh available via button.
          </Typography>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: '#10b981' }} />
              <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.6)' }}>
                {'>'}55% (Target Met)
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: '#f59e0b' }} />
              <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.6)' }}>
                50-55% (Marginal)
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: '#ef4444' }} />
              <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.6)' }}>
                {'<'}50% (Below Target)
              </Typography>
            </Box>
          </Box>
        </Box>
      </Paper>
    </Box>
  );
};

export default TruthDashboard;
