/**
 * Agent Conversation Viewer
 * Real-time visualization of agent-to-agent communication in the swarm
 */
import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  Paper,
  Chip,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,


  Alert,
  Tab,
  Tabs,
  Badge
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Forum as ForumIcon,

  Psychology as BrainIcon,

  Warning as WarningIcon
} from '@mui/icons-material';
import Grid from '@mui/material/Grid';


interface Message {
  source_agent: string;
  content: any;
  priority: number;
  confidence: number;
  timestamp: string;
}

interface AgentConversationViewerProps {
  discussionLogs: Message[];
  agentInsights?: any[];
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div role="tabpanel" hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 2 }}>{children}</Box>}
    </div>
  );
}

const AgentConversationViewer: React.FC<AgentConversationViewerProps> = ({
  discussionLogs,
  // agentInsights (unused)
}) => {
  const [tabValue, setTabValue] = useState(0);
  const [expandedMessage, setExpandedMessage] = useState<string | false>(false);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleAccordionChange = (messageId: string) => (_event: React.SyntheticEvent, isExpanded: boolean) => {
    setExpandedMessage(isExpanded ? messageId : false);
  };

  const getPriorityColor = (priority: number) => {
    if (priority >= 8) return 'error';
    if (priority >= 6) return 'warning';
    if (priority >= 4) return 'info';
    return 'default';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  const formatAgentName = (agentId: string) => {
    return agentId
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  const getMessageTypeIcon = (content: any) => {
    const type = content?.type || 'unknown';
    if (type.includes('fundamental')) return '📊';
    if (type.includes('technical')) return '📈';
    if (type.includes('sentiment')) return '💭';
    if (type.includes('risk')) return '⚠️';
    if (type.includes('macro')) return '🌍';
    if (type.includes('options')) return '📉';
    if (type.includes('discussion')) return '💬';
    return '📝';
  };

  // Group messages by agent
  const messagesByAgent = discussionLogs.reduce((acc, msg) => {
    if (!acc[msg.source_agent]) {
      acc[msg.source_agent] = [];
    }
    acc[msg.source_agent].push(msg);
    return acc;
  }, {} as Record<string, Message[]>);

  // Calculate statistics
  const stats = {
    totalMessages: discussionLogs.length,
    uniqueAgents: Object.keys(messagesByAgent).length,
    avgPriority: discussionLogs.reduce((sum, msg) => sum + msg.priority, 0) / discussionLogs.length || 0,
    avgConfidence: discussionLogs.reduce((sum, msg) => sum + msg.confidence, 0) / discussionLogs.length || 0,
    highPriorityCount: discussionLogs.filter(msg => msg.priority >= 8).length
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ForumIcon color="primary" />
          Agent Conversation Logs
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Real-time stigmergic communication between the 17 agents
        </Typography>
        <Divider sx={{ mb: 2 }} />

        {/* Statistics */}
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid size={{ xs: 6, md: 3 }}>
            <Paper elevation={1} sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="h4" color="primary.main">{stats.totalMessages}</Typography>
              <Typography variant="caption" color="text.secondary">Total Messages</Typography>
            </Paper>
          </Grid>
          <Grid size={{ xs: 6, md: 3 }}>
            <Paper elevation={1} sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="h4" color="success.main">{stats.uniqueAgents}</Typography>
              <Typography variant="caption" color="text.secondary">Active Agents</Typography>
            </Paper>
          </Grid>
          <Grid size={{ xs: 6, md: 3 }}>
            <Paper elevation={1} sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="h4" color="warning.main">{stats.avgPriority.toFixed(1)}</Typography>
              <Typography variant="caption" color="text.secondary">Avg Priority</Typography>
            </Paper>
          </Grid>
          <Grid size={{ xs: 6, md: 3 }}>
            <Paper elevation={1} sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="h4" color="info.main">{(stats.avgConfidence * 100).toFixed(0)}%</Typography>
              <Typography variant="caption" color="text.secondary">Avg Confidence</Typography>
            </Paper>
          </Grid>
        </Grid>

        {stats.highPriorityCount > 0 && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            {stats.highPriorityCount} high-priority messages detected - critical insights from agents
          </Alert>
        )}

        {/* Tabs */}
        <Tabs value={tabValue} onChange={handleTabChange} sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tab label="Timeline View" />
          <Tab label="By Agent" />
          <Tab label="High Priority" />
        </Tabs>

        {/* Timeline View */}
        <TabPanel value={tabValue} index={0}>
          <List>
            {discussionLogs.map((msg, idx) => (
              <Accordion
                key={`${msg.source_agent}-${idx}`}
                expanded={expandedMessage === `${msg.source_agent}-${idx}`}
                onChange={handleAccordionChange(`${msg.source_agent}-${idx}`)}
                sx={{ mb: 1 }}
              >
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                    <Typography sx={{ fontSize: '1.2rem' }}>{getMessageTypeIcon(msg.content)}</Typography>
                    <Typography variant="body1" sx={{ fontWeight: 'bold', flexGrow: 1 }}>
                      {formatAgentName(msg.source_agent)}
                    </Typography>
                    <Chip
                      label={`P${msg.priority}`}
                      size="small"
                      color={getPriorityColor(msg.priority)}
                    />
                    <Chip
                      label={`${(msg.confidence * 100).toFixed(0)}%`}
                      size="small"
                      color={getConfidenceColor(msg.confidence)}
                    />
                    <Typography variant="caption" color="text.secondary">
                      {new Date(msg.timestamp).toLocaleTimeString()}
                    </Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Paper elevation={0} sx={{ p: 2, bgcolor: 'grey.50' }}>
                    <Typography variant="caption" color="text.secondary" gutterBottom>
                      Message Type: {msg.content?.type || 'unknown'}
                    </Typography>
                    <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.85rem', margin: '8px 0' }}>
                      {JSON.stringify(msg.content, null, 2)}
                    </pre>
                  </Paper>
                </AccordionDetails>
              </Accordion>
            ))}
          </List>
        </TabPanel>

        {/* By Agent View */}
        <TabPanel value={tabValue} index={1}>
          {Object.entries(messagesByAgent).map(([agentId, messages]) => (
            <Accordion key={agentId} sx={{ mb: 1 }}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                  <BrainIcon color="primary" />
                  <Typography variant="body1" sx={{ fontWeight: 'bold', flexGrow: 1 }}>
                    {formatAgentName(agentId)}
                  </Typography>
                  <Badge badgeContent={messages.length} color="primary">
                    <ForumIcon />
                  </Badge>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <List dense>
                  {messages.map((msg, idx) => (
                    <ListItem key={idx} sx={{ flexDirection: 'column', alignItems: 'flex-start' }}>
                      <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                        <Chip label={`P${msg.priority}`} size="small" color={getPriorityColor(msg.priority)} />
                        <Chip label={`${(msg.confidence * 100).toFixed(0)}%`} size="small" color={getConfidenceColor(msg.confidence)} />
                        <Typography variant="caption" color="text.secondary">
                          {new Date(msg.timestamp).toLocaleTimeString()}
                        </Typography>
                      </Box>
                      <Paper elevation={0} sx={{ p: 1, bgcolor: 'grey.50', width: '100%' }}>
                        <Typography variant="caption">{msg.content?.type || 'Message'}</Typography>
                      </Paper>
                    </ListItem>
                  ))}
                </List>
              </AccordionDetails>
            </Accordion>
          ))}
        </TabPanel>

        {/* High Priority View */}
        <TabPanel value={tabValue} index={2}>
          <List>
            {discussionLogs
              .filter(msg => msg.priority >= 7)
              .map((msg, idx) => (
                <ListItem key={idx} sx={{ flexDirection: 'column', alignItems: 'flex-start', mb: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <WarningIcon color="warning" />
                    <Typography variant="body1" fontWeight="bold">
                      {formatAgentName(msg.source_agent)}
                    </Typography>
                    <Chip label={`Priority ${msg.priority}`} size="small" color="error" />
                    <Chip label={`${(msg.confidence * 100).toFixed(0)}% confidence`} size="small" />
                  </Box>
                  <Paper elevation={1} sx={{ p: 2, width: '100%' }}>
                    <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.85rem' }}>
                      {JSON.stringify(msg.content, null, 2)}
                    </pre>
                  </Paper>
                </ListItem>
              ))}
          </List>
          {discussionLogs.filter(msg => msg.priority >= 7).length === 0 && (
            <Alert severity="info">No high-priority messages in this analysis</Alert>
          )}
        </TabPanel>
      </CardContent>
    </Card>
  );
};

export default AgentConversationViewer;

