import React from 'react';
import { Box, Paper, Typography, Chip, LinearProgress, Skeleton } from '@mui/material';
import { CheckCircle, Cancel, Warning, HelpOutline } from '@mui/icons-material';

/**
 * Interface for model accuracy data
 */
export interface ModelAccuracy {
  model_name: string;
  direction_accuracy_30d: number;
  mape_30d: number;
  total_predictions: number;
  correct_predictions: number;
  last_5: boolean[];
  status: 'healthy' | 'degraded' | 'untrained';
  last_updated: string;
}

interface ModelAccuracyCardProps {
  model: ModelAccuracy;
  targetAccuracy?: number;
}

/**
 * ModelAccuracyCard - Displays accuracy metrics for a single ML model
 *
 * Features:
 * - Progress bar showing accuracy vs target (default 55%)
 * - Color coding: green (>55%), yellow (50-55%), red (<50%)
 * - Last 5 predictions shown as checkmarks/X marks
 * - Status badge: "Healthy", "Degraded", "Untrained"
 */
export const ModelAccuracyCard: React.FC<ModelAccuracyCardProps> = ({
  model,
  targetAccuracy = 55
}) => {
  // Get color based on accuracy
  const getAccuracyColor = (accuracy: number): string => {
    if (accuracy >= 55) return '#10b981'; // Green
    if (accuracy >= 50) return '#f59e0b'; // Yellow/Orange
    return '#ef4444'; // Red
  };

  // Get status badge properties
  const getStatusBadge = (status: ModelAccuracy['status']) => {
    switch (status) {
      case 'healthy':
        return {
          label: 'Healthy',
          color: 'success' as const,
          icon: <CheckCircle sx={{ fontSize: 14 }} />
        };
      case 'degraded':
        return {
          label: 'Degraded',
          color: 'warning' as const,
          icon: <Warning sx={{ fontSize: 14 }} />
        };
      case 'untrained':
        return {
          label: 'Untrained',
          color: 'default' as const,
          icon: <HelpOutline sx={{ fontSize: 14 }} />
        };
    }
  };

  // Calculate progress bar value (0-100 scale based on 0-100% accuracy)
  const progressValue = Math.min(100, Math.max(0, model.direction_accuracy_30d));
  const accuracyColor = getAccuracyColor(model.direction_accuracy_30d);
  const statusBadge = getStatusBadge(model.status);
  const meetsTarget = model.direction_accuracy_30d >= targetAccuracy;

  return (
    <Paper
      sx={{
        p: 2.5,
        bgcolor: '#1A1F3A',
        border: '1px solid',
        borderColor: model.status === 'degraded' ? '#f59e0b' : 'rgba(255,255,255,0.1)',
        borderRadius: 2,
        transition: 'border-color 0.2s ease',
        '&:hover': {
          borderColor: 'rgba(100, 181, 246, 0.3)'
        }
      }}
    >
      {/* Header: Model Name & Status */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
        <Typography
          variant="subtitle1"
          sx={{
            color: 'white',
            fontWeight: 600,
            fontSize: '1rem'
          }}
        >
          {model.model_name}
        </Typography>
        <Chip
          icon={statusBadge.icon}
          label={statusBadge.label}
          color={statusBadge.color}
          size="small"
          sx={{
            height: 24,
            '& .MuiChip-label': { px: 1, fontSize: '0.7rem' },
            '& .MuiChip-icon': { ml: 0.5 }
          }}
        />
      </Box>

      {/* Direction Accuracy */}
      <Box sx={{ mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
          <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.7)' }}>
            Direction Accuracy:
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography
              variant="body1"
              sx={{
                color: accuracyColor,
                fontWeight: 700,
                fontSize: '1.1rem'
              }}
            >
              {model.direction_accuracy_30d.toFixed(1)}%
            </Typography>
            {meetsTarget ? (
              <CheckCircle sx={{ color: '#10b981', fontSize: 18 }} />
            ) : (
              <Cancel sx={{ color: '#ef4444', fontSize: 18 }} />
            )}
          </Box>
        </Box>

        {/* Progress Bar */}
        <Box sx={{ position: 'relative', mt: 1 }}>
          <LinearProgress
            variant="determinate"
            value={progressValue}
            sx={{
              height: 10,
              borderRadius: 5,
              bgcolor: 'rgba(255,255,255,0.1)',
              '& .MuiLinearProgress-bar': {
                bgcolor: accuracyColor,
                borderRadius: 5
              }
            }}
          />
          {/* Target marker */}
          <Box
            sx={{
              position: 'absolute',
              left: `${targetAccuracy}%`,
              top: -2,
              bottom: -2,
              width: 2,
              bgcolor: 'white',
              opacity: 0.7
            }}
          />
        </Box>

        {/* Target label */}
        <Typography
          variant="caption"
          sx={{
            color: 'rgba(255,255,255,0.5)',
            display: 'block',
            textAlign: 'right',
            mt: 0.5
          }}
        >
          target: {targetAccuracy}%
        </Typography>
      </Box>

      {/* MAPE */}
      <Box sx={{ mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.7)' }}>
            MAPE:
          </Typography>
          <Typography
            variant="body2"
            sx={{
              color: model.mape_30d <= 3 ? '#10b981' : model.mape_30d <= 5 ? '#f59e0b' : '#ef4444',
              fontWeight: 600
            }}
          >
            {model.mape_30d.toFixed(1)}%
          </Typography>
        </Box>
      </Box>

      {/* Last 5 Predictions */}
      <Box>
        <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.7)', mb: 1 }}>
          Last 5:
        </Typography>
        <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center' }}>
          {model.last_5.map((correct, index) => (
            <Box
              key={index}
              sx={{
                width: 24,
                height: 24,
                borderRadius: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                bgcolor: correct ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                border: '1px solid',
                borderColor: correct ? '#10b981' : '#ef4444'
              }}
            >
              {correct ? (
                <CheckCircle sx={{ fontSize: 14, color: '#10b981' }} />
              ) : (
                <Cancel sx={{ fontSize: 14, color: '#ef4444' }} />
              )}
            </Box>
          ))}
          {model.status === 'degraded' && (
            <Chip
              label="DEGRADED"
              size="small"
              sx={{
                ml: 1,
                height: 20,
                bgcolor: 'rgba(245, 158, 11, 0.2)',
                color: '#f59e0b',
                fontSize: '0.65rem',
                fontWeight: 700
              }}
            />
          )}
        </Box>
      </Box>

      {/* Prediction count */}
      <Typography
        variant="caption"
        sx={{
          color: 'rgba(255,255,255,0.4)',
          display: 'block',
          mt: 2
        }}
      >
        {model.correct_predictions}/{model.total_predictions} correct predictions
      </Typography>
    </Paper>
  );
};

/**
 * Loading skeleton for ModelAccuracyCard
 */
export const ModelAccuracyCardSkeleton: React.FC = () => {
  return (
    <Paper
      sx={{
        p: 2.5,
        bgcolor: '#1A1F3A',
        border: '1px solid rgba(255,255,255,0.1)',
        borderRadius: 2
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <Skeleton variant="text" width={150} sx={{ bgcolor: 'rgba(255,255,255,0.1)' }} />
        <Skeleton variant="rectangular" width={70} height={24} sx={{ bgcolor: 'rgba(255,255,255,0.1)', borderRadius: 1 }} />
      </Box>
      <Box sx={{ mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
          <Skeleton variant="text" width={120} sx={{ bgcolor: 'rgba(255,255,255,0.1)' }} />
          <Skeleton variant="text" width={60} sx={{ bgcolor: 'rgba(255,255,255,0.1)' }} />
        </Box>
        <Skeleton variant="rectangular" height={10} sx={{ bgcolor: 'rgba(255,255,255,0.1)', borderRadius: 5 }} />
      </Box>
      <Box sx={{ mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
          <Skeleton variant="text" width={50} sx={{ bgcolor: 'rgba(255,255,255,0.1)' }} />
          <Skeleton variant="text" width={40} sx={{ bgcolor: 'rgba(255,255,255,0.1)' }} />
        </Box>
      </Box>
      <Box>
        <Skeleton variant="text" width={50} sx={{ bgcolor: 'rgba(255,255,255,0.1)', mb: 1 }} />
        <Box sx={{ display: 'flex', gap: 0.5 }}>
          {[1, 2, 3, 4, 5].map((i) => (
            <Skeleton key={i} variant="rectangular" width={24} height={24} sx={{ bgcolor: 'rgba(255,255,255,0.1)', borderRadius: 1 }} />
          ))}
        </Box>
      </Box>
    </Paper>
  );
};

export default ModelAccuracyCard;
