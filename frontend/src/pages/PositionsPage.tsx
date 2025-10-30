import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Tabs,
  Tab,
  Button,
  Paper,

  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  CircularProgress,
  Snackbar
} from '@mui/material';
import {

  Upload as UploadIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
  TrendingUp as TrendingUpIcon
} from '@mui/icons-material';
import { positionService } from '../services/positionService';

import Grid from '@mui/material/Grid';




interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div role="tabpanel" hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const PositionsPage: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [stockPositions, setStockPositions] = useState<any[]>([]);
  const [optionPositions, setOptionPositions] = useState<any[]>([]);
  const [portfolioSummary, setPortfolioSummary] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [openImportDialog, setOpenImportDialog] = useState(false);
  const [isChaseFormat, setIsChaseFormat] = useState(false);
  const [replaceExisting, setReplaceExisting] = useState(false);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' as 'success' | 'error' });

  // Form state for adding positions


  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [stocks, options, summary] = await Promise.all([
        positionService.getStockPositions(true),
        positionService.getOptionPositions(true),
        positionService.getPortfolioSummary()
      ]);
      setStockPositions(stocks);
      setOptionPositions(options);
      setPortfolioSummary(summary);
    } catch (error) {
      showSnackbar('Error loading positions', 'error');
    } finally {
      setLoading(false);
    }
  };

  const showSnackbar = (message: string, severity: 'success' | 'error') => {
    setSnackbar({ open: true, message, severity });
  };



  const handleDownloadTemplate = async (type: 'stocks' | 'options') => {
    try {
      await positionService.downloadTemplate(type);
      showSnackbar('Template downloaded successfully', 'success');
    } catch (error) {
      showSnackbar('Error downloading template', 'error');
    }
  };

  const handleExport = async (type: 'stocks' | 'options') => {
    try {
      await positionService.exportPositions(type);
      showSnackbar('Positions exported successfully', 'success');
    } catch (error) {
      showSnackbar('Error exporting positions', 'error');
    }
  };

  const handleImport = async () => {
    if (!selectedFile) return;

    try {
      const type = tabValue === 0 ? 'stocks' : 'options';
      const result = await positionService.importPositions(
        type,
        selectedFile,
        replaceExisting,
        type === 'options' ? isChaseFormat : false
      );

      if (result.errors.length > 0) {
        showSnackbar(`Imported ${result.success} positions with ${result.failed} errors`, 'error');
      } else {
        const chaseMsg = result.chase_conversion
          ? ` • Converted ${result.chase_conversion.options_converted} options (skipped ${result.chase_conversion.cash_skipped})`
          : '';
        showSnackbar(`Successfully imported ${result.success} positions${chaseMsg}`, 'success');
      }

      setOpenImportDialog(false);
      setSelectedFile(null);
      setIsChaseFormat(false);
      setReplaceExisting(false);
      loadData();
    } catch (error) {
      showSnackbar('Error importing positions', 'error');
    }
  };

  const handleEnrichAll = async () => {
    setLoading(true);
    try {
      await positionService.enrichAllPositions();
      showSnackbar('Positions enriched successfully', 'success');
      loadData();
    } catch (error) {
      showSnackbar('Error enriching positions', 'error');
    } finally {
      setLoading(false);
    }
  };




  const formatCurrency = (value: number | null | undefined) => {
    if (value === null || value === undefined) return 'N/A';
    return `$${value.toFixed(2)}`;
  };

  const formatPercent = (value: number | null | undefined) => {
    if (value === null || value === undefined) return 'N/A';
    return `${value.toFixed(2)}%`;
  };

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      {/* Portfolio Summary */}
      {portfolioSummary && (
        <Paper sx={{ p: 3, mb: 3, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
          <Grid container spacing={3}>
            <Grid size={{ xs: 12, md: 3 }}>
              <Typography variant="h6" color="white">Total Value</Typography>
              <Typography variant="h4" color="white">
                {formatCurrency(portfolioSummary.total_current_value)}
              </Typography>
            </Grid>
            <Grid size={{ xs: 12, md: 3 }}>
              <Typography variant="h6" color="white">Total P&L</Typography>
              <Typography variant="h4" color="white">
                {formatCurrency(portfolioSummary.total_pnl)} ({formatPercent(portfolioSummary.total_pnl_pct)})
              </Typography>
            </Grid>
            <Grid size={{ xs: 12, md: 3 }}>
              <Typography variant="h6" color="white">Positions</Typography>
              <Typography variant="h4" color="white">
                {portfolioSummary.total_stocks} stocks, {portfolioSummary.total_options} options
              </Typography>
            </Grid>
            <Grid size={{ xs: 12, md: 3 }}>
              <Typography variant="h6" color="white">Symbols</Typography>
              <Typography variant="h4" color="white">
                {portfolioSummary.unique_symbols}
              </Typography>
            </Grid>
          </Grid>
        </Paper>
      )}

      {/* Action Buttons */}
      <Box sx={{ mb: 3, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        <Button variant="outlined" startIcon={<UploadIcon />} onClick={() => setOpenImportDialog(true)}>
          Import CSV
        </Button>
        <Button variant="outlined" startIcon={<DownloadIcon />} onClick={() => handleExport(tabValue === 0 ? 'stocks' : 'options')}>
          Export CSV
        </Button>
        <Button variant="outlined" startIcon={<DownloadIcon />} onClick={() => handleDownloadTemplate(tabValue === 0 ? 'stocks' : 'options')}>
          Download Template
        </Button>
        <Button variant="outlined" startIcon={<RefreshIcon />} onClick={handleEnrichAll} disabled={loading}>
          Refresh Data
        </Button>
        <Button variant="contained" color="secondary" startIcon={<TrendingUpIcon />} onClick={async () => {
          try {
            const plan = await positionService.runDailyResearch('auto');
            const totalSyms = plan.symbols?.length || 0;
            const actions = plan.recommendations?.length || 0;
            showSnackbar(`Daily research complete: ${totalSyms} symbols, ${actions} actions`, 'success');
          } catch (e) {
            showSnackbar('Error running daily research plan', 'error');
          }
        }}>
          Run Daily Research Plan
        </Button>
      </Box>
      {/* Import Dialog */}
      <Dialog open={openImportDialog} onClose={() => setOpenImportDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Import {tabValue === 0 ? 'Stock' : 'Option'} Positions</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <Button variant="outlined" component="label" startIcon={<UploadIcon />}>
              Choose CSV File
              <input type="file" accept=".csv" hidden onChange={(e) => {
                const f = e.target.files?.[0] || null;
                setSelectedFile(f || null);
              }} />
            </Button>
            {selectedFile && (
              <Typography variant="body2">Selected: {selectedFile.name}</Typography>
            )}
            {tabValue === 1 && (
              <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                <label>
                  <input type="checkbox" checked={isChaseFormat} onChange={(e) => setIsChaseFormat(e.target.checked)} />
                  &nbsp;This is a Chase.com export (auto-convert)
                </label>
              </Box>
            )}
            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
              <label>
                <input type="checkbox" checked={replaceExisting} onChange={(e) => setReplaceExisting(e.target.checked)} />
                &nbsp;Replace existing {tabValue === 0 ? 'stock' : 'option'} positions
              </label>
            </Box>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenImportDialog(false)}>Cancel</Button>
          <Button onClick={handleImport} variant="contained" disabled={!selectedFile}>Import</Button>
        </DialogActions>
      </Dialog>


      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={tabValue} onChange={(_, v) => setTabValue(v)}>
          <Tab label={`Stock Positions (${stockPositions.length})`} />
          <Tab label={`Option Positions (${optionPositions.length})`} />
        </Tabs>
      </Paper>

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {/* Stock Positions Tab - Content will be added in next file */}
      <TabPanel value={tabValue} index={0}>
        {/* Stock positions grid - to be implemented */}
      </TabPanel>

      {/* Option Positions Tab - Content will be added in next file */}
      <TabPanel value={tabValue} index={1}>
        {/* Option positions grid - to be implemented */}
      </TabPanel>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
      >
        <Alert severity={snackbar.severity} onClose={() => setSnackbar({ ...snackbar, open: false })}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default PositionsPage;

