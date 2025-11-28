import React from 'react';
import { Skeleton, Box, Card, CardContent, Grid } from '@mui/material';

/**
 * Reusable skeleton loading components for the Options Probability application
 *
 * These skeletons provide visual feedback during data loading, improving UX by
 * making it clear to users that content is being loaded rather than showing blank areas.
 *
 * Color scheme: Uses #2A3050 for skeleton elements on #1A1F3A backgrounds
 */

/**
 * SignalCardSkeleton - Skeleton for Phase 4 signal cards
 *
 * Mimics the layout of SignalCard component with:
 * - Header with title and info icon
 * - Main value display area
 * - Tags/chips at bottom
 */
export const SignalCardSkeleton: React.FC = () => (
  <Card sx={{ bgcolor: '#1A1F3A', borderRadius: 2 }}>
    <CardContent>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <Skeleton variant="text" width={120} height={24} sx={{ bgcolor: '#2A3050' }} />
        <Skeleton variant="circular" width={24} height={24} sx={{ bgcolor: '#2A3050' }} />
      </Box>
      <Skeleton variant="text" width="80%" height={40} sx={{ bgcolor: '#2A3050' }} />
      <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
        <Skeleton variant="rounded" width={60} height={24} sx={{ bgcolor: '#2A3050' }} />
        <Skeleton variant="rounded" width={80} height={24} sx={{ bgcolor: '#2A3050' }} />
      </Box>
    </CardContent>
  </Card>
);

/**
 * ChartSkeleton - Skeleton for chart areas
 *
 * Displays a placeholder for charts with:
 * - Header area with title and controls
 * - Main chart area with visual placeholder
 *
 * @param height - Optional height for the chart skeleton (default: 400)
 */
export const ChartSkeleton: React.FC<{ height?: number }> = ({ height = 400 }) => (
  <Box sx={{ width: '100%', height, bgcolor: '#1A1F3A', borderRadius: 2, p: 2 }}>
    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
      <Skeleton variant="text" width={200} height={32} sx={{ bgcolor: '#2A3050' }} />
      <Box sx={{ display: 'flex', gap: 1 }}>
        <Skeleton variant="rounded" width={80} height={32} sx={{ bgcolor: '#2A3050' }} />
        <Skeleton variant="rounded" width={80} height={32} sx={{ bgcolor: '#2A3050' }} />
      </Box>
    </Box>
    <Skeleton
      variant="rectangular"
      width="100%"
      height={height - 80}
      sx={{ bgcolor: '#2A3050', borderRadius: 1 }}
    />
  </Box>
);

/**
 * RiskMetricSkeleton - Skeleton for individual risk metric cards
 *
 * Mimics RiskMetricCard layout with:
 * - Title
 * - Value display
 * - Description/trend indicator
 */
export const RiskMetricSkeleton: React.FC = () => (
  <Box sx={{ p: 2, bgcolor: '#1A1F3A', borderRadius: 2 }}>
    <Skeleton variant="text" width={100} height={20} sx={{ bgcolor: '#2A3050' }} />
    <Skeleton variant="text" width={60} height={36} sx={{ bgcolor: '#2A3050', mt: 1 }} />
    <Skeleton variant="text" width={80} height={16} sx={{ bgcolor: '#2A3050', mt: 1 }} />
  </Box>
);

/**
 * Phase4PanelSkeleton - Skeleton for the entire Phase 4 signals panel
 *
 * Displays a 2x2 grid (or 1x4 on mobile) of SignalCardSkeletons
 * to match the Phase4SignalsPanel layout
 */
export const Phase4PanelSkeleton: React.FC = () => (
  <Grid container spacing={2}>
    {[1, 2, 3, 4].map((i) => (
      <Grid key={i} size={{ xs: 12, sm: 6, md: 3 }}>
        <SignalCardSkeleton />
      </Grid>
    ))}
  </Grid>
);

/**
 * Phase4SignalsPanelSkeleton - Full skeleton for Phase4SignalsPanel
 *
 * Matches the complete Phase4SignalsPanel layout including:
 * - Header with title and status indicator
 * - 2x2 grid of signal cards
 */
export const Phase4SignalsPanelSkeleton: React.FC = () => (
  <Box className="bg-[#2a2a2a] border border-[#404040] rounded-lg p-6">
    {/* Header Skeleton */}
    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
      <Box>
        <Skeleton variant="text" width={180} height={32} sx={{ bgcolor: '#404040' }} />
        <Skeleton variant="text" width={150} height={20} sx={{ bgcolor: '#404040', mt: 0.5 }} />
      </Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Skeleton variant="circular" width={8} height={8} sx={{ bgcolor: '#404040' }} />
        <Skeleton variant="text" width={60} height={16} sx={{ bgcolor: '#404040' }} />
      </Box>
    </Box>

    {/* 2x2 Grid of Signal Card Skeletons */}
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: '1fr 1fr' }, gap: 3 }}>
      {[1, 2, 3, 4].map((i) => (
        <Box key={i} sx={{ p: 3, bgcolor: '#1a1a1a', border: '1px solid #404040', borderRadius: 2 }}>
          {/* Card Header */}
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
            <Skeleton variant="text" width={140} height={24} sx={{ bgcolor: '#404040' }} />
            <Skeleton variant="circular" width={20} height={20} sx={{ bgcolor: '#404040' }} />
          </Box>
          {/* Gauge/Value Area */}
          <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
            <Skeleton variant="circular" width={120} height={60} sx={{ bgcolor: '#404040' }} />
          </Box>
          {/* Value Display */}
          <Skeleton variant="text" width={80} height={36} sx={{ bgcolor: '#404040' }} />
          <Skeleton variant="text" width={100} height={20} sx={{ bgcolor: '#404040', mt: 1 }} />
          {/* Progress Bar */}
          <Skeleton variant="rectangular" width="100%" height={8} sx={{ bgcolor: '#404040', borderRadius: 1, mt: 2 }} />
          {/* Tooltip Box */}
          <Skeleton variant="rounded" width="100%" height={40} sx={{ bgcolor: '#404040', mt: 2 }} />
        </Box>
      ))}
    </Box>
  </Box>
);

/**
 * ModelAccuracySkeleton - Skeleton for model accuracy card (Truth Dashboard)
 *
 * Matches ModelAccuracyCard layout with:
 * - Header with model name and status badge
 * - Progress bar
 * - Last 5 predictions indicators
 * - Metrics display
 */
export const ModelAccuracySkeleton: React.FC = () => (
  <Card sx={{ bgcolor: '#1A1F3A', borderRadius: 2 }}>
    <CardContent>
      <Skeleton variant="text" width={150} height={28} sx={{ bgcolor: '#2A3050' }} />
      <Skeleton variant="rectangular" width="100%" height={8} sx={{ bgcolor: '#2A3050', mt: 2, borderRadius: 1 }} />
      <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
        {[1, 2, 3, 4, 5].map((i) => (
          <Skeleton key={i} variant="circular" width={20} height={20} sx={{ bgcolor: '#2A3050' }} />
        ))}
      </Box>
      <Skeleton variant="text" width={80} height={20} sx={{ bgcolor: '#2A3050', mt: 2 }} />
    </CardContent>
  </Card>
);

/**
 * TableRowSkeleton - Skeleton for table rows
 *
 * Creates a row of skeleton cells for table loading states
 *
 * @param columns - Number of columns to render (default: 5)
 */
export const TableRowSkeleton: React.FC<{ columns?: number }> = ({ columns = 5 }) => (
  <Box sx={{ display: 'flex', gap: 2, py: 1 }}>
    {Array.from({ length: columns }).map((_, i) => (
      <Skeleton key={i} variant="text" width={`${100 / columns}%`} height={24} sx={{ bgcolor: '#2A3050' }} />
    ))}
  </Box>
);

/**
 * VIXWidgetSkeleton - Skeleton for VIX Analysis Widget
 *
 * Matches VIXAnalysisWidget layout for consistent loading states
 */
export const VIXWidgetSkeleton: React.FC = () => (
  <Box sx={{ width: '100%', bgcolor: '#1A1F3A', borderRadius: 2, p: 2 }}>
    {/* Header */}
    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
      <Skeleton variant="text" width={120} height={24} sx={{ bgcolor: '#2A3050' }} />
      <Skeleton variant="circular" width={24} height={24} sx={{ bgcolor: '#2A3050' }} />
    </Box>
    {/* Main Value */}
    <Skeleton variant="text" width={100} height={48} sx={{ bgcolor: '#2A3050', mb: 2 }} />
    {/* Metrics Grid */}
    <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2 }}>
      {[1, 2, 3, 4].map((i) => (
        <Box key={i}>
          <Skeleton variant="text" width={60} height={16} sx={{ bgcolor: '#2A3050' }} />
          <Skeleton variant="text" width={80} height={24} sx={{ bgcolor: '#2A3050', mt: 0.5 }} />
        </Box>
      ))}
    </Box>
  </Box>
);

/**
 * DashboardHeaderSkeleton - Skeleton for dashboard headers
 *
 * Provides a consistent loading state for page headers
 */
export const DashboardHeaderSkeleton: React.FC = () => (
  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
    <Box>
      <Skeleton variant="text" width={250} height={40} sx={{ bgcolor: '#2A3050' }} />
      <Skeleton variant="text" width={180} height={20} sx={{ bgcolor: '#2A3050', mt: 0.5 }} />
    </Box>
    <Box sx={{ display: 'flex', gap: 1 }}>
      <Skeleton variant="rounded" width={100} height={36} sx={{ bgcolor: '#2A3050' }} />
      <Skeleton variant="circular" width={36} height={36} sx={{ bgcolor: '#2A3050' }} />
    </Box>
  </Box>
);

/**
 * RiskPanelSkeleton - Skeleton for the full RiskPanelDashboard
 *
 * Displays skeleton loading state for all 7 risk metrics
 */
export const RiskPanelSkeleton: React.FC = () => (
  <Box className="bg-[#2a2a2a] border border-[#404040] rounded-lg p-6">
    {/* Header */}
    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
      <Box>
        <Skeleton variant="text" width={150} height={32} sx={{ bgcolor: '#404040' }} />
        <Skeleton variant="text" width={180} height={20} sx={{ bgcolor: '#404040', mt: 0.5 }} />
      </Box>
      <Skeleton variant="rounded" width={120} height={36} sx={{ bgcolor: '#404040' }} />
    </Box>

    {/* 2x4 Grid of Risk Metric Skeletons */}
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: '1fr 1fr' }, gap: 2 }}>
      {[1, 2, 3, 4, 5, 6, 7].map((i) => (
        <Box key={i} sx={{ p: 2, bgcolor: '#1a1a1a', border: '1px solid #404040', borderRadius: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1.5 }}>
            <Skeleton variant="text" width={100} height={20} sx={{ bgcolor: '#404040' }} />
            <Skeleton variant="circular" width={16} height={16} sx={{ bgcolor: '#404040' }} />
          </Box>
          <Skeleton variant="text" width={70} height={32} sx={{ bgcolor: '#404040' }} />
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 1 }}>
            <Skeleton variant="circular" width={16} height={16} sx={{ bgcolor: '#404040' }} />
            <Skeleton variant="text" width={60} height={16} sx={{ bgcolor: '#404040' }} />
          </Box>
          <Skeleton variant="rectangular" width="100%" height={6} sx={{ bgcolor: '#404040', borderRadius: 1, mt: 1.5 }} />
        </Box>
      ))}
    </Box>
  </Box>
);

export default {
  SignalCardSkeleton,
  ChartSkeleton,
  RiskMetricSkeleton,
  Phase4PanelSkeleton,
  Phase4SignalsPanelSkeleton,
  ModelAccuracySkeleton,
  TableRowSkeleton,
  VIXWidgetSkeleton,
  DashboardHeaderSkeleton,
  RiskPanelSkeleton,
};
