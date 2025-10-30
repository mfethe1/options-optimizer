/**
 * Analysis Progress Tracker
 * Real-time progress visualization during swarm analysis
 */
import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Paper,
  Stepper,
  Step,
  StepLabel,
  Alert,
  Collapse,
} from '@mui/material';
import {
  CheckCircle as CheckIcon,
  RadioButtonUnchecked as PendingIcon,
  Error as ErrorIcon,
  Sync as SyncIcon,
  Psychology as BrainIcon,
  CloudUpload as UploadIcon,
  Assessment as AssessmentIcon,
  Group as GroupIcon
} from '@mui/icons-material';
import Grid from '@mui/material/Grid';





interface AgentProgress {
  agent_id: string;
  agent_type: string;
  status: 'pending' | 'analyzing' | 'complete' | 'failed';
  progress?: number;
  message?: string;
}

interface AnalysisProgressTrackerProps {
  isAnalyzing: boolean;
  currentStep?: string;
  agentProgress?: AgentProgress[];
  estimatedTimeRemaining?: number;
}

const AnalysisProgressTracker: React.FC<AnalysisProgressTrackerProps> = ({
  isAnalyzing,
  currentStep = 'Initializing...',
  agentProgress = [],
  estimatedTimeRemaining
}) => {
  const [elapsedTime, setElapsedTime] = useState(0);
  const [showDetails, setShowDetails] = useState(true);

  useEffect(() => {
    if (!isAnalyzing) {
      setElapsedTime(0);
      return;
    }

    const interval = setInterval(() => {
      setElapsedTime(prev => prev + 1);
    }, 1000);

    return () => clearInterval(interval);
  }, [isAnalyzing]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getStepIcon = (status: string) => {
    switch (status) {
      case 'complete':
        return <CheckIcon color="success" />;
      case 'in_progress':
        return <SyncIcon color="primary" className="rotating" />;
      case 'error':
        return <ErrorIcon color="error" />;
      default:
        return <PendingIcon color="disabled" />;
    }
  };

  const getAgentStatusColor = (status: string) => {
    switch (status) {
      case 'complete':
        return 'success';
      case 'analyzing':
        return 'primary';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  // Calculate overall progress
  const completedAgents = agentProgress.filter(a => a.status === 'complete').length;
  const totalAgents = agentProgress.length || 17;
  const overallProgress = (completedAgents / totalAgents) * 100;

  // Group agents by tier
  const agentsByTier = agentProgress.reduce((acc, agent) => {
    const tier = agent.agent_type.includes('overseer') ? 'Tier 1: Oversight'
      : agent.agent_type.includes('market') || agent.agent_type.includes('sector') ? 'Tier 2: Market Intelligence'
      : agent.agent_type.includes('fundamental') || agent.agent_type.includes('macro') ? 'Tier 3: Fundamental'
      : agent.agent_type.includes('risk') || agent.agent_type.includes('sentiment') ? 'Tier 4: Risk & Sentiment'
      : agent.agent_type.includes('options') || agent.agent_type.includes('volatility') ? 'Tier 5: Options'
      : agent.agent_type.includes('executor') || agent.agent_type.includes('compliance') ? 'Tier 6: Execution'
      : 'Tier 7: Recommendations';

    if (!acc[tier]) acc[tier] = [];
    acc[tier].push(agent);
    return acc;
  }, {} as Record<string, AgentProgress[]>);

  if (!isAnalyzing) {
    return null;
  }

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <BrainIcon color="primary" />
          Analysis in Progress
        </Typography>

        {/* Overall Progress */}
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="body2" color="text.secondary">
              {currentStep}
            </Typography>
            <Typography variant="body2" fontWeight="bold">
              {completedAgents} / {totalAgents} agents complete
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={overallProgress}
            sx={{ height: 10, borderRadius: 1 }}
          />
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
            <Typography variant="caption" color="text.secondary">
              Elapsed: {formatTime(elapsedTime)}
            </Typography>
            {estimatedTimeRemaining && (
              <Typography variant="caption" color="text.secondary">
                Est. remaining: {formatTime(estimatedTimeRemaining)}
              </Typography>
            )}
          </Box>
        </Box>

        {/* Analysis Stages */}
        <Paper elevation={1} sx={{ p: 2, mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom>Analysis Stages</Typography>
          <Stepper activeStep={1} alternativeLabel>
            <Step completed={true}>
              <StepLabel icon={<UploadIcon />}>Upload CSV</StepLabel>
            </Step>
            <Step completed={false}>
              <StepLabel icon={<GroupIcon />}>Agent Analysis</StepLabel>
            </Step>
            <Step completed={false}>
              <StepLabel icon={<AssessmentIcon />}>Consensus</StepLabel>
            </Step>
          </Stepper>
        </Paper>

        {/* Agent Progress by Tier */}
        <Box>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="subtitle2">Agent Progress</Typography>
            <Chip
              label={showDetails ? 'Hide Details' : 'Show Details'}
              size="small"
              onClick={() => setShowDetails(!showDetails)}
              sx={{ cursor: 'pointer' }}
            />
          </Box>

          <Collapse in={showDetails}>
            {Object.entries(agentsByTier).map(([tier, agents]) => (
              <Paper key={tier} elevation={1} sx={{ p: 2, mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom color="primary">
                  {tier}
                </Typography>
                <List dense>
                  {agents.map((agent) => (
                    <ListItem key={agent.agent_id}>
                      <ListItemIcon sx={{ minWidth: 40 }}>
                        {getStepIcon(agent.status)}
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="body2">
                              {agent.agent_id.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                            </Typography>
                            <Chip
                              label={agent.status}
                              size="small"
                              color={getAgentStatusColor(agent.status) as any}
                            />
                          </Box>
                        }
                        secondary={agent.message}
                      />
                      {agent.progress !== undefined && (
                        <Box sx={{ width: 100, ml: 2 }}>
                          <LinearProgress
                            variant="determinate"
                            value={agent.progress}
                            sx={{ height: 6, borderRadius: 1 }}
                          />
                        </Box>
                      )}
                    </ListItem>
                  ))}
                </List>
              </Paper>
            ))}
          </Collapse>

          {!showDetails && (
            <Grid container spacing={1}>
              {agentProgress.map((agent) => (
                <Grid key={agent.agent_id}>
                  <Chip
                    icon={getStepIcon(agent.status)}
                    label={agent.agent_id.split('_')[0]}
                    size="small"
                    color={getAgentStatusColor(agent.status) as any}
                  />
                </Grid>
              ))}
            </Grid>
          )}
        </Box>

        {/* Helpful Tips */}
        <Alert severity="info" sx={{ mt: 3 }}>
          <Typography variant="body2">
            <strong>What's happening:</strong> The 17-agent swarm is analyzing your portfolio in parallel.
            Each agent brings specialized expertise (fundamental, technical, risk, options, etc.).
            This typically takes 3-5 minutes for comprehensive analysis.
          </Typography>
        </Alert>
      </CardContent>

      <style>{`
        @keyframes rotate {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .rotating {
          animation: rotate 2s linear infinite;
        }
      `}</style>
    </Card>
  );
};

export default AnalysisProgressTracker;

