import React, { Component, ErrorInfo, ReactNode, useState, useCallback } from 'react';
import {
  Alert,
  AlertTitle,
  Button,
  Box,
  Typography,
  Collapse,
  Paper,
  Stack,
  IconButton,
} from '@mui/material';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import RefreshIcon from '@mui/icons-material/Refresh';
import BugReportIcon from '@mui/icons-material/BugReport';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';

// ============================================================================
// Types
// ============================================================================

export interface ErrorBoundaryProps {
  children: ReactNode;
  /** Custom fallback UI to render when an error occurs */
  fallback?: ReactNode;
  /** Callback fired when an error is caught */
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  /** Optional name for this boundary (useful for logging) */
  boundaryName?: string;
  /** Whether to show the retry button (default: true) */
  showRetry?: boolean;
  /** Whether to show error details toggle (default: true in dev, false in prod) */
  showDetails?: boolean;
  /** Custom retry handler */
  onRetry?: () => void;
  /** Compact mode for smaller sections */
  compact?: boolean;
}

export interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

// ============================================================================
// Default Error Fallback Component
// ============================================================================

interface DefaultErrorFallbackProps {
  error: Error | null;
  errorInfo: ErrorInfo | null;
  onRetry: () => void;
  showRetry: boolean;
  showDetails: boolean;
  boundaryName?: string;
  compact?: boolean;
}

function DefaultErrorFallback({
  error,
  errorInfo,
  onRetry,
  showRetry,
  showDetails,
  boundaryName,
  compact = false,
}: DefaultErrorFallbackProps) {
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopyError = useCallback(() => {
    const errorText = `Error: ${error?.message || 'Unknown error'}
Stack: ${error?.stack || 'No stack trace'}
Component Stack: ${errorInfo?.componentStack || 'No component stack'}
Boundary: ${boundaryName || 'Unknown'}
Time: ${new Date().toISOString()}`;

    navigator.clipboard.writeText(errorText).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [error, errorInfo, boundaryName]);

  const handleReportBug = useCallback(() => {
    // In a real app, this would open a bug report dialog or redirect to an issue tracker
    const subject = encodeURIComponent(`Bug Report: ${error?.message || 'Unknown error'}`);
    const body = encodeURIComponent(`
Error occurred in: ${boundaryName || 'Unknown section'}
Error message: ${error?.message || 'Unknown'}
Time: ${new Date().toISOString()}

Steps to reproduce:
1.
2.
3.

Expected behavior:

Actual behavior:
    `);
    window.open(`mailto:support@example.com?subject=${subject}&body=${body}`, '_blank');
  }, [error, boundaryName]);

  if (compact) {
    return (
      <Alert
        severity="error"
        sx={{ m: 1 }}
        action={
          showRetry && (
            <Button color="inherit" size="small" onClick={onRetry} startIcon={<RefreshIcon />}>
              Retry
            </Button>
          )
        }
      >
        <AlertTitle>Error in {boundaryName || 'this section'}</AlertTitle>
        {error?.message || 'An unexpected error occurred'}
      </Alert>
    );
  }

  return (
    <Paper
      elevation={0}
      sx={{
        p: 3,
        m: 2,
        border: '1px solid',
        borderColor: 'error.light',
        borderRadius: 2,
        backgroundColor: 'error.50',
      }}
    >
      <Stack spacing={2}>
        {/* Header */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <WarningAmberIcon color="error" sx={{ fontSize: 32 }} />
          <Box>
            <Typography variant="h6" color="error.dark" fontWeight={600}>
              Something went wrong
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {boundaryName
                ? `An error occurred in ${boundaryName}. The rest of the app is still working.`
                : 'This section encountered an error. The rest of the app is still working.'}
            </Typography>
          </Box>
        </Box>

        {/* Error Message */}
        <Alert severity="error" variant="outlined">
          {error?.message || 'An unexpected error occurred'}
        </Alert>

        {/* Action Buttons */}
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
          {showDetails && (
            <Button
              variant="outlined"
              size="small"
              onClick={() => setDetailsOpen(!detailsOpen)}
              startIcon={detailsOpen ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            >
              {detailsOpen ? 'Hide Details' : 'Show Details'}
            </Button>
          )}
          {showRetry && (
            <Button
              variant="contained"
              size="small"
              onClick={onRetry}
              startIcon={<RefreshIcon />}
              color="primary"
            >
              Retry
            </Button>
          )}
          <Button
            variant="outlined"
            size="small"
            onClick={handleReportBug}
            startIcon={<BugReportIcon />}
            color="secondary"
          >
            Report Bug
          </Button>
        </Stack>

        {/* Error Details */}
        {showDetails && (
          <Collapse in={detailsOpen}>
            <Paper
              variant="outlined"
              sx={{
                p: 2,
                backgroundColor: 'grey.50',
                maxHeight: 300,
                overflow: 'auto',
              }}
            >
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="subtitle2" color="text.secondary">
                  Error Details
                </Typography>
                <IconButton size="small" onClick={handleCopyError} title="Copy error details">
                  <ContentCopyIcon fontSize="small" />
                </IconButton>
              </Box>
              {copied && (
                <Typography variant="caption" color="success.main" sx={{ mb: 1, display: 'block' }}>
                  Copied to clipboard!
                </Typography>
              )}
              <Typography
                component="pre"
                variant="caption"
                sx={{
                  fontFamily: 'monospace',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  m: 0,
                }}
              >
                {`Error: ${error?.message || 'Unknown error'}

Stack Trace:
${error?.stack || 'No stack trace available'}

Component Stack:
${errorInfo?.componentStack || 'No component stack available'}`}
              </Typography>
            </Paper>
          </Collapse>
        )}
      </Stack>
    </Paper>
  );
}

// ============================================================================
// ErrorBoundary Class Component
// ============================================================================

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    // Update state so the next render shows the fallback UI
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Log the error
    console.error(
      `ErrorBoundary${this.props.boundaryName ? ` [${this.props.boundaryName}]` : ''} caught an error:`,
      error,
      errorInfo
    );

    // Update state with error info
    this.setState({ errorInfo });

    // Call optional error callback
    this.props.onError?.(error, errorInfo);
  }

  handleRetry = (): void => {
    // Reset the error state
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });

    // Call optional retry callback
    this.props.onRetry?.();
  };

  render(): ReactNode {
    const { hasError, error, errorInfo } = this.state;
    const {
      children,
      fallback,
      boundaryName,
      showRetry = true,
      showDetails = process.env.NODE_ENV === 'development',
      compact = false,
    } = this.props;

    if (hasError) {
      // If a custom fallback is provided, use it
      if (fallback) {
        return fallback;
      }

      // Otherwise, use the default error fallback
      return (
        <DefaultErrorFallback
          error={error}
          errorInfo={errorInfo}
          onRetry={this.handleRetry}
          showRetry={showRetry}
          showDetails={showDetails}
          boundaryName={boundaryName}
          compact={compact}
        />
      );
    }

    return children;
  }
}

// ============================================================================
// useErrorHandler Hook
// ============================================================================

interface ErrorState {
  error: Error | null;
  hasError: boolean;
}

/**
 * Hook for handling errors in functional components.
 * This can be used to catch and handle errors in event handlers,
 * async operations, or other non-render contexts.
 *
 * For render errors, use the ErrorBoundary component instead.
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { error, handleError, clearError, ErrorDisplay } = useErrorHandler();
 *
 *   const handleClick = async () => {
 *     try {
 *       await someAsyncOperation();
 *     } catch (err) {
 *       handleError(err as Error);
 *     }
 *   };
 *
 *   return (
 *     <div>
 *       <ErrorDisplay />
 *       <button onClick={handleClick}>Do Something</button>
 *     </div>
 *   );
 * }
 * ```
 */
export function useErrorHandler(boundaryName?: string) {
  const [errorState, setErrorState] = useState<ErrorState>({
    error: null,
    hasError: false,
  });

  const handleError = useCallback(
    (error: Error) => {
      console.error(`useErrorHandler${boundaryName ? ` [${boundaryName}]` : ''} caught:`, error);
      setErrorState({ error, hasError: true });
    },
    [boundaryName]
  );

  const clearError = useCallback(() => {
    setErrorState({ error: null, hasError: false });
  }, []);

  const ErrorDisplay = useCallback(
    ({ compact = false }: { compact?: boolean } = {}) => {
      if (!errorState.hasError) return null;

      return (
        <DefaultErrorFallback
          error={errorState.error}
          errorInfo={null}
          onRetry={clearError}
          showRetry={true}
          showDetails={process.env.NODE_ENV === 'development'}
          boundaryName={boundaryName}
          compact={compact}
        />
      );
    },
    [errorState, clearError, boundaryName]
  );

  return {
    error: errorState.error,
    hasError: errorState.hasError,
    handleError,
    clearError,
    ErrorDisplay,
  };
}

// ============================================================================
// withErrorBoundary HOC
// ============================================================================

/**
 * Higher-Order Component that wraps a component with an ErrorBoundary.
 * Useful for wrapping components at the route level or for lazy-loaded components.
 *
 * @example
 * ```tsx
 * const SafeComponent = withErrorBoundary(MyComponent, {
 *   boundaryName: 'MyComponent',
 *   showRetry: true,
 * });
 * ```
 */
export function withErrorBoundary<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  errorBoundaryProps?: Omit<ErrorBoundaryProps, 'children'>
) {
  const displayName = WrappedComponent.displayName || WrappedComponent.name || 'Component';

  const ComponentWithErrorBoundary = (props: P) => (
    <ErrorBoundary {...errorBoundaryProps} boundaryName={errorBoundaryProps?.boundaryName || displayName}>
      <WrappedComponent {...props} />
    </ErrorBoundary>
  );

  ComponentWithErrorBoundary.displayName = `withErrorBoundary(${displayName})`;

  return ComponentWithErrorBoundary;
}

// ============================================================================
// Specialized Error Boundaries
// ============================================================================

/**
 * Error boundary optimized for route-level errors.
 * Shows a full-page error with navigation options.
 */
export function RouteErrorBoundary({ children, routeName }: { children: ReactNode; routeName?: string }) {
  return (
    <ErrorBoundary
      boundaryName={routeName || 'Page'}
      showDetails={process.env.NODE_ENV === 'development'}
      fallback={
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '50vh',
            p: 4,
            textAlign: 'center',
          }}
        >
          <WarningAmberIcon sx={{ fontSize: 64, color: 'error.main', mb: 2 }} />
          <Typography variant="h4" gutterBottom>
            Page Error
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 3, maxWidth: 400 }}>
            {routeName
              ? `The ${routeName} page encountered an error and could not be displayed.`
              : 'This page encountered an error and could not be displayed.'}
          </Typography>
          <Stack direction="row" spacing={2}>
            <Button variant="contained" onClick={() => window.location.reload()} startIcon={<RefreshIcon />}>
              Reload Page
            </Button>
            <Button variant="outlined" onClick={() => (window.location.href = '/')}>
              Go to Home
            </Button>
          </Stack>
        </Box>
      }
    >
      {children}
    </ErrorBoundary>
  );
}

/**
 * Error boundary optimized for widget/card-level errors.
 * Shows a compact inline error that doesn't disrupt the layout.
 */
export function WidgetErrorBoundary({
  children,
  widgetName,
}: {
  children: ReactNode;
  widgetName?: string;
}) {
  return (
    <ErrorBoundary boundaryName={widgetName || 'Widget'} compact showDetails={false}>
      {children}
    </ErrorBoundary>
  );
}

/**
 * Error boundary for WebSocket-dependent components.
 * Provides specific messaging about connection issues.
 */
export function WebSocketErrorBoundary({
  children,
  componentName,
}: {
  children: ReactNode;
  componentName?: string;
}) {
  return (
    <ErrorBoundary
      boundaryName={componentName || 'Real-time Component'}
      fallback={
        <Alert severity="warning" sx={{ m: 1 }}>
          <AlertTitle>Connection Issue</AlertTitle>
          {componentName
            ? `${componentName} requires a live connection to display data.`
            : 'This component requires a live connection to display data.'}
          {' Please check that the backend server is running and try refreshing the page.'}
        </Alert>
      }
    >
      {children}
    </ErrorBoundary>
  );
}

export default ErrorBoundary;
