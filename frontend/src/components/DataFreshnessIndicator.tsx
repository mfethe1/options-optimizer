import React, { useState, useEffect } from 'react';
import { Chip, Tooltip } from '@mui/material';
import { AccessTime, Warning, CheckCircle } from '@mui/icons-material';

interface DataFreshnessIndicatorProps {
  lastUpdated: Date | string | null;
  staleThresholdSeconds?: number;  // Yellow warning threshold
  oldThresholdSeconds?: number;    // Red warning threshold
  showTimestamp?: boolean;
  compact?: boolean;
}

export const DataFreshnessIndicator: React.FC<DataFreshnessIndicatorProps> = ({
  lastUpdated,
  staleThresholdSeconds = 60,   // 1 minute = stale
  oldThresholdSeconds = 300,    // 5 minutes = old
  showTimestamp = true,
  compact = false,
}) => {
  const [ageSeconds, setAgeSeconds] = useState<number>(0);

  useEffect(() => {
    // Calculate initial age
    if (lastUpdated) {
      const updated = typeof lastUpdated === 'string'
        ? new Date(lastUpdated)
        : lastUpdated;
      setAgeSeconds(Math.floor((Date.now() - updated.getTime()) / 1000));
    }

    // Update age every second
    const interval = setInterval(() => {
      if (lastUpdated) {
        const updated = typeof lastUpdated === 'string'
          ? new Date(lastUpdated)
          : lastUpdated;
        setAgeSeconds(Math.floor((Date.now() - updated.getTime()) / 1000));
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [lastUpdated]);

  const getStatus = () => {
    if (!lastUpdated) return { color: 'default' as const, label: 'No data', icon: Warning };
    if (ageSeconds > oldThresholdSeconds) return { color: 'error' as const, label: 'Old', icon: Warning };
    if (ageSeconds > staleThresholdSeconds) return { color: 'warning' as const, label: 'Stale', icon: AccessTime };
    return { color: 'success' as const, label: 'Live', icon: CheckCircle };
  };

  const formatAge = (seconds: number): string => {
    if (seconds < 0) return 'just now';
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
  };

  const formatTimestamp = (date: Date | string): string => {
    const d = typeof date === 'string' ? new Date(date) : date;
    return d.toLocaleTimeString();
  };

  const status = getStatus();
  const Icon = status.icon;

  if (compact) {
    return (
      <Tooltip title={lastUpdated ? `Last updated: ${formatAge(ageSeconds)} (${formatTimestamp(lastUpdated)})` : 'No data'}>
        <Icon
          sx={{
            fontSize: 16,
            color: status.color === 'success' ? 'success.main'
                 : status.color === 'warning' ? 'warning.main'
                 : status.color === 'error' ? 'error.main'
                 : 'text.disabled'
          }}
        />
      </Tooltip>
    );
  }

  return (
    <Chip
      icon={<Icon />}
      label={showTimestamp ? formatAge(ageSeconds) : status.label}
      size="small"
      color={status.color}
      variant="outlined"
    />
  );
};

export default DataFreshnessIndicator;
