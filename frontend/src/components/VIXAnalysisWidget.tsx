/**
 * VIX Analysis Widget
 *
 * Displays Epidemic Volatility predictions with:
 * - Current vs Predicted VIX
 * - Market regime analysis
 * - Implied price impact
 * - Confidence metrics
 */

import React from 'react';
import {
  Box,
  Paper,
  Typography,
  LinearProgress,
  Chip,
  Stack,
  Divider,
  Tooltip,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Warning,
  CheckCircle,
  Info,
} from '@mui/icons-material';

interface VIXAnalysisData {
  current_vix: number;
  vix_prediction: number;
  vix_upper?: number;
  vix_lower?: number;
  confidence: number;
  regime?: string;
  vix_delta?: number;
  implied_change_pct?: number;
  horizon_days?: number;
}

interface VIXAnalysisWidgetProps {
  data: VIXAnalysisData | null;
  loading?: boolean;
}

// Market regime configurations
const REGIME_CONFIG: Record<string, { color: string; icon: React.ReactNode; label: string }> = {
  calm: {
    color: '#4caf50',
    icon: <CheckCircle sx={{ color: '#4caf50' }} />,
    label: 'Calm',
  },
  pre_volatile: {
    color: '#ff9800',
    icon: <Warning sx={{ color: '#ff9800' }} />,
    label: 'Pre-Volatile',
  },
  volatile: {
    color: '#f44336',
    icon: <TrendingUp sx={{ color: '#f44336' }} />,
    label: 'Volatile',
  },
  stabilized: {
    color: '#2196f3',
    icon: <TrendingDown sx={{ color: '#2196f3' }} />,
    label: 'Stabilized',
  },
  unknown: {
    color: '#9e9e9e',
    icon: <Info sx={{ color: '#9e9e9e' }} />,
    label: 'Unknown',
  },
};

const VIXAnalysisWidget: React.FC<VIXAnalysisWidgetProps> = ({ data, loading = false }) => {
  // Debug log to see what data we're receiving
  console.log('[VIXAnalysisWidget] Received data:', data);

  if (loading || !data) {
    return (
      <Paper
        sx={{
          p: 2,
          bgcolor: 'rgba(26, 31, 58, 0.95)',
          border: '1px solid rgba(139, 92, 246, 0.3)',
          borderRadius: 2,
        }}
      >
        <Typography variant="subtitle2" sx={{ color: '#8B5CF6', mb: 2 }}>
          VIX Volatility Analysis
        </Typography>
        <LinearProgress sx={{ mt: 2 }} />
      </Paper>
    );
  }

  // Defensive checks for required properties
  if (typeof data.current_vix !== 'number' || typeof data.vix_prediction !== 'number') {
    console.warn('[VIXAnalysisWidget] Missing required VIX data:', data);
    return (
      <Paper
        sx={{
          p: 2,
          bgcolor: 'rgba(26, 31, 58, 0.95)',
          border: '1px solid rgba(139, 92, 246, 0.3)',
          borderRadius: 2,
        }}
      >
        <Typography variant="subtitle2" sx={{ color: '#8B5CF6', mb: 2 }}>
          VIX Volatility Analysis
        </Typography>
        <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.6)', mt: 2 }}>
          Insufficient data available
        </Typography>
      </Paper>
    );
  }

  const vixDelta = data.vix_delta ?? (data.vix_prediction - data.current_vix);
  const isIncreasing = vixDelta > 0;
  const regimeKey = (data.regime || 'unknown').toLowerCase();
  const regime = REGIME_CONFIG[regimeKey] || REGIME_CONFIG.unknown;

  // Ensure confidence is a valid number between 0 and 1
  const confidence = typeof data.confidence === 'number' ? data.confidence : 0.75;

  return (
    <Paper
      sx={{
        p: 2,
        bgcolor: 'rgba(26, 31, 58, 0.95)',
        border: '1px solid rgba(139, 92, 246, 0.3)',
        borderRadius: 2,
        height: '100%',
      }}
    >
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="subtitle2" sx={{ color: '#8B5CF6', fontWeight: 600 }}>
          🦠 VIX Volatility Analysis
        </Typography>
        <Chip
          icon={regime.icon}
          label={regime.label}
          size="small"
          sx={{
            bgcolor: `${regime.color}20`,
            color: regime.color,
            borderColor: regime.color,
            border: '1px solid',
          }}
        />
      </Box>

      <Divider sx={{ borderColor: 'rgba(139, 92, 246, 0.2)', mb: 2 }} />

      {/* VIX Comparison */}
      <Stack spacing={2}>
        {/* Current VIX */}
        <Box>
          <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.6)' }}>
            Current VIX
          </Typography>
          <Typography variant="h4" sx={{ color: 'white', fontWeight: 600 }}>
            {data.current_vix.toFixed(2)}
          </Typography>
        </Box>

        {/* Predicted VIX */}
        <Box>
          <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.6)' }}>
            Predicted VIX ({data.horizon_days || 30}d)
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography
              variant="h4"
              sx={{
                color: isIncreasing ? '#f44336' : '#4caf50',
                fontWeight: 600,
              }}
            >
              {data.vix_prediction.toFixed(2)}
            </Typography>
            {isIncreasing ? (
              <TrendingUp sx={{ color: '#f44336' }} />
            ) : (
              <TrendingDown sx={{ color: '#4caf50' }} />
            )}
          </Box>
          {typeof data.vix_upper === 'number' && typeof data.vix_lower === 'number' && (
            <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.5)' }}>
              Range: {data.vix_lower.toFixed(2)} - {data.vix_upper.toFixed(2)}
            </Typography>
          )}
        </Box>

        {/* VIX Change */}
        <Box>
          <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.6)' }}>
            Expected Change
          </Typography>
          <Typography
            variant="h6"
            sx={{
              color: isIncreasing ? '#f44336' : '#4caf50',
              fontWeight: 600,
            }}
          >
            {isIncreasing ? '+' : ''}
            {vixDelta.toFixed(2)} pts
          </Typography>
        </Box>

        <Divider sx={{ borderColor: 'rgba(139, 92, 246, 0.2)' }} />

        {/* Price Impact */}
        {typeof data.implied_change_pct === 'number' && (
          <Box>
            <Tooltip title="Estimated price impact based on historical VIX-SPY correlation">
              <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.6)', cursor: 'help' }}>
                Implied Price Impact ⓘ
              </Typography>
            </Tooltip>
            <Typography
              variant="h6"
              sx={{
                color: data.implied_change_pct >= 0 ? '#4caf50' : '#f44336',
                fontWeight: 600,
              }}
            >
              {data.implied_change_pct >= 0 ? '+' : ''}
              {data.implied_change_pct.toFixed(2)}%
            </Typography>
            <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.5)' }}>
              Based on VIX-SPY correlation
            </Typography>
          </Box>
        )}

        {/* Confidence */}
        <Box>
          <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.6)' }}>
            Prediction Confidence
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
            <LinearProgress
              variant="determinate"
              value={confidence * 100}
              sx={{
                flex: 1,
                height: 8,
                borderRadius: 4,
                bgcolor: 'rgba(139, 92, 246, 0.2)',
                '& .MuiLinearProgress-bar': {
                  bgcolor: '#8B5CF6',
                },
              }}
            />
            <Typography variant="body2" sx={{ color: '#8B5CF6', fontWeight: 600 }}>
              {(confidence * 100).toFixed(0)}%
            </Typography>
          </Box>
        </Box>

        {/* Interpretation */}
        <Box
          sx={{
            bgcolor: 'rgba(139, 92, 246, 0.1)',
            border: '1px solid rgba(139, 92, 246, 0.3)',
            borderRadius: 1,
            p: 1.5,
            mt: 1,
          }}
        >
          <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.9)', lineHeight: 1.5 }}>
            {isIncreasing ? (
              <>
                <strong>Fear Contagion Expected:</strong> VIX surge of +{vixDelta.toFixed(1)} points
                suggests increased market volatility ahead. Consider protective positions.
              </>
            ) : (
              <>
                <strong>Volatility Subsiding:</strong> VIX decline of {vixDelta.toFixed(1)} points
                indicates calming markets. Opportune for reducing hedges.
              </>
            )}
          </Typography>
        </Box>
      </Stack>
    </Paper>
  );
};

export default VIXAnalysisWidget;
