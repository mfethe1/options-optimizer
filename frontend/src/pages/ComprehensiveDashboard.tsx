/**
 * Comprehensive Trading Dashboard
 *
 * A professional, customizable dashboard that provides all critical trading features
 * in a single view. Inspired by Bloomberg Terminal and TradingView.
 *
 * Features:
 * - Unified ML predictions (all 6 models)
 * - Professional chart with overlays
 * - Real-time portfolio & P/L
 * - Risk dashboard (Phase 1-4 signals)
 * - Options chain analysis
 * - AI insights & swarm analysis
 * - News feed
 * - Alerts & notifications
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid2 as Grid,
  Paper,
  Typography,
  IconButton,
  Tooltip,
  AppBar,
  Toolbar,
  TextField,
  Badge,
  Menu,
  MenuItem,
  Chip,
  Tabs,
  Tab,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Switch,
  FormControlLabel,
  Button,
  ButtonGroup,
  CircularProgress,
  Alert
} from '@mui/material';
import {
  Search,
  Notifications,
  Settings,
  Refresh,
  FullscreenIcon,
  Dashboard as DashboardIcon,
  TrendingUp,
  Assessment,
  ShowChart,
  AccountBalance,
  Warning,
  Lightbulb,
  Article,
  CalendarToday,
  DragIndicator,
  Visibility,
  VisibilityOff,
  Save,
  RestartAlt
} from '@mui/icons-material';

// Widget imports (will be created separately)
import { MLPredictionsWidget } from '../components/widgets/MLPredictionsWidget';
import { MainChartWidget } from '../components/widgets/MainChartWidget';
import { PortfolioWidget } from '../components/widgets/PortfolioWidget';
import { RiskDashboardWidget } from '../components/widgets/RiskDashboardWidget';
import { OptionsChainWidget } from '../components/widgets/OptionsChainWidget';
import { AIInsightsWidget } from '../components/widgets/AIInsightsWidget';
import { NewsFeedWidget } from '../components/widgets/NewsFeedWidget';
import { AlertsWidget } from '../components/widgets/AlertsWidget';

interface WidgetConfig {
  id: string;
  title: string;
  component: React.ComponentType<any>;
  visible: boolean;
  gridArea: string;
  icon: React.ReactNode;
}

const ComprehensiveDashboard: React.FC = () => {
  // State
  const [symbol, setSymbol] = useState('SPY');
  const [searchValue, setSearchValue] = useState('SPY');
  const [isLoading, setIsLoading] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [notificationCount, setNotificationCount] = useState(3);
  const [selectedTab, setSelectedTab] = useState(0);
  const [layoutMode, setLayoutMode] = useState<'desktop' | 'compact'>('desktop');

  // Widget visibility state
  const [widgets, setWidgets] = useState<WidgetConfig[]>([
    {
      id: 'ml-predictions',
      title: 'ML Predictions (All Models)',
      component: MLPredictionsWidget,
      visible: true,
      gridArea: 'ml-predictions',
      icon: <ShowChart />
    },
    {
      id: 'main-chart',
      title: 'Price Chart',
      component: MainChartWidget,
      visible: true,
      gridArea: 'main-chart',
      icon: <TrendingUp />
    },
    {
      id: 'portfolio',
      title: 'Portfolio & Positions',
      component: PortfolioWidget,
      visible: true,
      gridArea: 'portfolio',
      icon: <AccountBalance />
    },
    {
      id: 'risk-dashboard',
      title: 'Risk Dashboard',
      component: RiskDashboardWidget,
      visible: true,
      gridArea: 'risk-dashboard',
      icon: <Warning />
    },
    {
      id: 'options-chain',
      title: 'Options Chain',
      component: OptionsChainWidget,
      visible: true,
      gridArea: 'options-chain',
      icon: <Assessment />
    },
    {
      id: 'ai-insights',
      title: 'AI Insights',
      component: AIInsightsWidget,
      visible: true,
      gridArea: 'ai-insights',
      icon: <Lightbulb />
    }
  ]);

  // Bottom panel widgets
  const [bottomPanelTab, setBottomPanelTab] = useState(0);

  // Handle symbol search
  const handleSymbolSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setSymbol(searchValue.toUpperCase());
    setIsLoading(true);
    // Simulate data loading
    setTimeout(() => setIsLoading(false), 500);
  };

  // Toggle widget visibility
  const toggleWidget = (widgetId: string) => {
    setWidgets(prev =>
      prev.map(w =>
        w.id === widgetId ? { ...w, visible: !w.visible } : w
      )
    );
  };

  // Responsive layout detection
  useEffect(() => {
    const handleResize = () => {
      setLayoutMode(window.innerWidth >= 1366 ? 'desktop' : 'compact');
    };
    window.addEventListener('resize', handleResize);
    handleResize();
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh', bgcolor: 'grey.50' }}>
      {/* Top AppBar */}
      <AppBar position="static" elevation={0} sx={{ bgcolor: 'grey.900', borderBottom: 1, borderColor: 'divider' }}>
        <Toolbar sx={{ gap: 2 }}>
          {/* Logo/Title */}
          <DashboardIcon sx={{ color: 'primary.main' }} />
          <Typography variant="h6" sx={{ fontWeight: 600, color: 'white' }}>
            Trading Dashboard
          </Typography>

          {/* Symbol Search */}
          <Box
            component="form"
            onSubmit={handleSymbolSearch}
            sx={{ display: 'flex', alignItems: 'center', bgcolor: 'grey.800', borderRadius: 1, px: 2, py: 0.5, ml: 2 }}
          >
            <Search sx={{ color: 'grey.400', mr: 1 }} />
            <TextField
              value={searchValue}
              onChange={(e) => setSearchValue(e.target.value)}
              placeholder="Search symbol..."
              variant="standard"
              InputProps={{
                disableUnderline: true,
                sx: { color: 'white', fontSize: '0.95rem' }
              }}
              sx={{ width: 150 }}
            />
          </Box>

          {/* Current Symbol Display */}
          <Chip
            label={symbol}
            color="primary"
            sx={{ ml: 1, fontWeight: 600, fontSize: '1rem' }}
          />

          {isLoading && <CircularProgress size={20} sx={{ ml: 1 }} />}

          <Box sx={{ flexGrow: 1 }} />

          {/* Quick Actions */}
          <Tooltip title="Refresh Data">
            <IconButton color="inherit" onClick={() => setIsLoading(true)}>
              <Refresh />
            </IconButton>
          </Tooltip>

          <Tooltip title="Notifications">
            <IconButton color="inherit" onClick={() => setNotificationsOpen(true)}>
              <Badge badgeContent={notificationCount} color="error">
                <Notifications />
              </Badge>
            </IconButton>
          </Tooltip>

          <Tooltip title="Dashboard Settings">
            <IconButton color="inherit" onClick={() => setSettingsOpen(true)}>
              <Settings />
            </IconButton>
          </Tooltip>
        </Toolbar>
      </AppBar>

      {/* Main Dashboard Grid */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
        <Grid
          container
          spacing={2}
          sx={{
            display: 'grid',
            gridTemplateColumns: layoutMode === 'desktop' ? '1fr 2fr 1fr' : '1fr',
            gridTemplateRows: layoutMode === 'desktop' ? 'auto 1fr 1fr' : 'auto',
            gridTemplateAreas: layoutMode === 'desktop' ? `
              "ml-predictions main-chart main-chart"
              "portfolio risk-dashboard ai-insights"
              "options-chain options-chain options-chain"
            ` : `
              "main-chart"
              "ml-predictions"
              "portfolio"
              "risk-dashboard"
              "ai-insights"
              "options-chain"
            `,
            gap: 2,
            minHeight: 0
          }}
        >
          {widgets.filter(w => w.visible).map((widget) => (
            <Box
              key={widget.id}
              sx={{
                gridArea: widget.gridArea,
                minHeight: 300,
                display: 'flex'
              }}
            >
              <Paper
                elevation={2}
                sx={{
                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  overflow: 'hidden',
                  borderRadius: 2,
                  transition: 'box-shadow 0.3s',
                  '&:hover': {
                    boxShadow: 6
                  }
                }}
              >
                {/* Widget Header */}
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    px: 2,
                    py: 1.5,
                    bgcolor: 'grey.100',
                    borderBottom: 1,
                    borderColor: 'divider'
                  }}
                >
                  {widget.icon && (
                    <Box sx={{ mr: 1, color: 'primary.main' }}>
                      {widget.icon}
                    </Box>
                  )}
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, flexGrow: 1 }}>
                    {widget.title}
                  </Typography>

                  <IconButton size="small" onClick={() => toggleWidget(widget.id)}>
                    <VisibilityOff fontSize="small" />
                  </IconButton>
                </Box>

                {/* Widget Content */}
                <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
                  <widget.component symbol={symbol} />
                </Box>
              </Paper>
            </Box>
          ))}
        </Grid>
      </Box>

      {/* Bottom Panel (News, Alerts, Calendar) */}
      <Paper
        elevation={4}
        sx={{
          borderTop: 2,
          borderColor: 'divider',
          height: 250,
          display: 'flex',
          flexDirection: 'column'
        }}
      >
        <Tabs
          value={bottomPanelTab}
          onChange={(_, v) => setBottomPanelTab(v)}
          sx={{ borderBottom: 1, borderColor: 'divider', bgcolor: 'grey.50' }}
        >
          <Tab icon={<Article />} label="News" iconPosition="start" />
          <Tab icon={<Notifications />} label="Alerts" iconPosition="start" />
          <Tab icon={<CalendarToday />} label="Calendar" iconPosition="start" />
        </Tabs>

        <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
          {bottomPanelTab === 0 && <NewsFeedWidget symbol={symbol} />}
          {bottomPanelTab === 1 && <AlertsWidget />}
          {bottomPanelTab === 2 && (
            <Typography variant="body2" color="text.secondary">
              Economic Calendar - Coming soon
            </Typography>
          )}
        </Box>
      </Paper>

      {/* Settings Drawer */}
      <Drawer
        anchor="right"
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
      >
        <Box sx={{ width: 350, p: 3 }}>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
            Dashboard Settings
          </Typography>

          <Divider sx={{ my: 2 }} />

          <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600, mb: 2 }}>
            Visible Widgets
          </Typography>

          <List>
            {widgets.map((widget) => (
              <ListItem key={widget.id} dense>
                <ListItemIcon sx={{ minWidth: 40 }}>
                  {widget.icon}
                </ListItemIcon>
                <ListItemText primary={widget.title} />
                <Switch
                  edge="end"
                  checked={widget.visible}
                  onChange={() => toggleWidget(widget.id)}
                />
              </ListItem>
            ))}
          </List>

          <Divider sx={{ my: 2 }} />

          <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600, mb: 2 }}>
            Layout Presets
          </Typography>

          <ButtonGroup orientation="vertical" fullWidth>
            <Button variant="outlined">Trader View</Button>
            <Button variant="outlined">Analyst View</Button>
            <Button variant="outlined">Risk View</Button>
            <Button variant="outlined">Options View</Button>
            <Button variant="outlined">AI View</Button>
          </ButtonGroup>

          <Box sx={{ mt: 3, display: 'flex', gap: 1 }}>
            <Button
              variant="contained"
              startIcon={<Save />}
              fullWidth
            >
              Save Layout
            </Button>
            <Button
              variant="outlined"
              startIcon={<RestartAlt />}
              fullWidth
            >
              Reset
            </Button>
          </Box>
        </Box>
      </Drawer>

      {/* Notifications Drawer */}
      <Drawer
        anchor="right"
        open={notificationsOpen}
        onClose={() => setNotificationsOpen(false)}
      >
        <Box sx={{ width: 350, p: 3 }}>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
            Notifications
          </Typography>

          <Divider sx={{ my: 2 }} />

          <List>
            <ListItem>
              <Alert severity="info" sx={{ width: '100%' }}>
                PINN model updated - 91% accuracy
              </Alert>
            </ListItem>
            <ListItem>
              <Alert severity="warning" sx={{ width: '100%' }}>
                Risk score exceeded threshold (75)
              </Alert>
            </ListItem>
            <ListItem>
              <Alert severity="success" sx={{ width: '100%' }}>
                Ensemble consensus: STRONG BUY
              </Alert>
            </ListItem>
          </List>

          <Button
            variant="text"
            fullWidth
            onClick={() => {
              setNotificationCount(0);
              setNotificationsOpen(false);
            }}
          >
            Clear All
          </Button>
        </Box>
      </Drawer>
    </Box>
  );
};

export default ComprehensiveDashboard;
