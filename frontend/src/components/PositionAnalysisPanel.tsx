/**
 * Position Analysis Panel
 * Displays comprehensive position-by-position analysis from the 17-agent swarm
 */
import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Paper,
  Divider,
  Tab,
  Tabs,
  Alert,
  List,
  ListItem,
  ListItemText,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Warning as WarningIcon,
  Lightbulb as LightbulbIcon,
  SwapHoriz as SwapIcon,
  Psychology as BrainIcon
} from '@mui/icons-material';

import Grid from '@mui/material/Grid';



interface PositionAnalysisPanelProps {
  positions: any[];
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`position-tabpanel-${index}`}
      aria-labelledby={`position-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const PositionAnalysisPanel: React.FC<PositionAnalysisPanelProps> = ({ positions }) => {
  const [expandedPosition, setExpandedPosition] = useState<string | false>(false);

  const handleAccordionChange = (symbol: string) => (_event: React.SyntheticEvent, isExpanded: boolean) => {
    setExpandedPosition(isExpanded ? symbol : false);
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <BrainIcon color="primary" />
          Position-by-Position AI Analysis
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Comprehensive analysis from all 17 agents for each position
        </Typography>
        <Divider sx={{ mb: 2 }} />

        {positions.map((position, index) => (
          <Accordion
            key={`${position.symbol}-${index}`}
            expanded={expandedPosition === position.symbol}
            onChange={handleAccordionChange(position.symbol)}
            sx={{ mb: 1 }}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
                <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                  {position.symbol}
                </Typography>
                <Chip
                  label={position.asset_type === 'option' ? `${position.option_type?.toUpperCase()} $${position.strike}` : 'Stock'}
                  size="small"
                  color="primary"
                  variant="outlined"
                />
                {position.current_metrics && (
                  <>
                    <Typography variant="body2" color="text.secondary">
                      Value: {formatCurrency(position.current_metrics.market_value || 0)}
                    </Typography>
                    <Chip
                      label={formatPercent(position.current_metrics.unrealized_pnl_pct || 0)}
                      size="small"
                      color={position.current_metrics.unrealized_pnl >= 0 ? 'success' : 'error'}
                      icon={position.current_metrics.unrealized_pnl >= 0 ? <TrendingUpIcon /> : <TrendingDownIcon />}
                    />
                  </>
                )}
              </Box>
            </AccordionSummary>

            <AccordionDetails>
              <PositionDetailView position={position} />
            </AccordionDetails>
          </Accordion>
        ))}
      </CardContent>
    </Card>
  );
};

const PositionDetailView: React.FC<{ position: any }> = ({ position }) => {
  const [tabValue, setTabValue] = useState(0);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  return (
    <Box>
      <Tabs value={tabValue} onChange={handleTabChange} sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tab label="Overview" />
        <Tab label="Agent Insights" />
        <Tab label="Stock Report" />
        <Tab label="Recommendations" />
        <Tab label="Risks & Opportunities" />
      </Tabs>

      {/* Overview Tab */}
      <TabPanel value={tabValue} index={0}>
        <Grid container spacing={2}>
          {position.current_metrics && (
            <>
              <Grid size={{ xs: 12, md: 6 }}>
                <Paper elevation={1} sx={{ p: 2 }}>
                  <Typography variant="overline" color="text.secondary">Current Metrics</Typography>
                  <Grid container spacing={1} sx={{ mt: 1 }}>
                    <Grid size={6}>
                      <Typography variant="body2">Current Price:</Typography>
                      <Typography variant="h6">${position.current_metrics.current_price?.toFixed(2)}</Typography>
                    </Grid>
                    <Grid size={6}>
                      <Typography variant="body2">Market Value:</Typography>
                      <Typography variant="h6">${position.current_metrics.market_value?.toLocaleString()}</Typography>
                    </Grid>
                    <Grid size={6}>
                      <Typography variant="body2">Unrealized P&L:</Typography>
                      <Typography variant="h6" color={position.current_metrics.unrealized_pnl >= 0 ? 'success.main' : 'error.main'}>
                        ${position.current_metrics.unrealized_pnl?.toLocaleString()}
                      </Typography>
                    </Grid>
                    <Grid size={6}>
                      <Typography variant="body2">P&L %:</Typography>
                      <Typography variant="h6" color={position.current_metrics.unrealized_pnl_pct >= 0 ? 'success.main' : 'error.main'}>
                        {position.current_metrics.unrealized_pnl_pct?.toFixed(2)}%
                      </Typography>
                    </Grid>
                  </Grid>
                </Paper>
              </Grid>

              {position.greeks && (
                <Grid size={{ xs: 12, md: 6 }}>
                  <Paper elevation={1} sx={{ p: 2 }}>
                    <Typography variant="overline" color="text.secondary">Greeks</Typography>
                    <Grid container spacing={1} sx={{ mt: 1 }}>
                      <Grid size={6}>
                        <Typography variant="body2">Delta:</Typography>
                        <Typography variant="h6">{position.greeks.delta?.toFixed(3)}</Typography>
                      </Grid>
                      <Grid size={6}>
                        <Typography variant="body2">Gamma:</Typography>
                        <Typography variant="h6">{position.greeks.gamma?.toFixed(3)}</Typography>
                      </Grid>
                      <Grid size={6}>
                        <Typography variant="body2">Theta:</Typography>
                        <Typography variant="h6">{position.greeks.theta?.toFixed(2)}</Typography>
                      </Grid>
                      <Grid size={6}>
                        <Typography variant="body2">Vega:</Typography>
                        <Typography variant="h6">{position.greeks.vega?.toFixed(2)}</Typography>
                      </Grid>
                    </Grid>
                  </Paper>
                </Grid>
              )}
            </>
          )}
        </Grid>
      </TabPanel>

      {/* Agent Insights Tab */}
      <TabPanel value={tabValue} index={1}>
        {position.agent_insights_for_position && position.agent_insights_for_position.length > 0 ? (
          <List>
            {position.agent_insights_for_position.map((insight: any, idx: number) => (
              <ListItem key={idx} sx={{ flexDirection: 'column', alignItems: 'flex-start', mb: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                  <Chip label={insight.agent_type} size="small" color="primary" />
                  <Chip
                    label={insight.recommendation}
                    size="small"
                    color={insight.recommendation === 'buy' ? 'success' : insight.recommendation === 'sell' ? 'error' : 'warning'}
                  />
                  <Typography variant="caption" color="text.secondary">
                    Confidence: {(insight.confidence * 100).toFixed(0)}%
                  </Typography>
                </Box>
                {insight.stock_specific_analysis && (
                  <Paper elevation={0} sx={{ p: 2, bgcolor: 'grey.50', width: '100%' }}>
                    <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                      {insight.stock_specific_analysis}
                    </Typography>
                  </Paper>
                )}
                {insight.key_insights && insight.key_insights.length > 0 && (
                  <Box sx={{ mt: 1 }}>
                    {insight.key_insights.map((keyInsight: string, kidx: number) => (
                      <Chip key={kidx} label={keyInsight} size="small" sx={{ mr: 0.5, mb: 0.5 }} />
                    ))}
                  </Box>
                )}
              </ListItem>
            ))}
          </List>
        ) : (
          <Alert severity="info">No agent insights available for this position</Alert>
        )}
      </TabPanel>

      {/* Stock Report Tab */}
      <TabPanel value={tabValue} index={2}>
        {position.comprehensive_stock_report ? (
          <Box>
            <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
              {JSON.stringify(position.comprehensive_stock_report, null, 2)}
            </Typography>
          </Box>
        ) : (
          <Alert severity="info">Comprehensive stock report not available</Alert>
        )}
      </TabPanel>

      {/* Recommendations Tab */}
      <TabPanel value={tabValue} index={3}>
        {position.replacement_recommendations ? (
          <Box>
            <Alert severity="info" icon={<SwapIcon />} sx={{ mb: 2 }}>
              AI-powered replacement recommendations based on current market conditions
            </Alert>
            <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
              {JSON.stringify(position.replacement_recommendations, null, 2)}
            </Typography>
          </Box>
        ) : (
          <Alert severity="success">Current position is optimal - no replacements recommended</Alert>
        )}
      </TabPanel>

      {/* Risks & Opportunities Tab */}
      <TabPanel value={tabValue} index={4}>
        <Grid container spacing={2}>
          <Grid size={{ xs: 12, md: 6 }}>
            {position.risk_warnings && position.risk_warnings.length > 0 ? (
              <Paper elevation={1} sx={{ p: 2 }}>
                <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                  <WarningIcon color="warning" />
                  Risk Warnings
                </Typography>
                <List dense>
                  {position.risk_warnings.map((warning: any, idx: number) => (
                    <ListItem key={idx}>
                      <ListItemText primary={warning} />
                    </ListItem>
                  ))}
                </List>
              </Paper>
            ) : (
              <Alert severity="success">No significant risks identified</Alert>
            )}
          </Grid>

          <Grid size={{ xs: 12, md: 6 }}>
            {position.opportunities && position.opportunities.length > 0 ? (
              <Paper elevation={1} sx={{ p: 2 }}>
                <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                  <LightbulbIcon color="success" />
                  Opportunities
                </Typography>
                <List dense>
                  {position.opportunities.map((opp: any, idx: number) => (
                    <ListItem key={idx}>
                      <ListItemText primary={opp} />
                    </ListItem>
                  ))}
                </List>
              </Paper>
            ) : (
              <Alert severity="info">No specific opportunities identified</Alert>
            )}
          </Grid>
        </Grid>
      </TabPanel>
    </Box>
  );
};

export default PositionAnalysisPanel;

