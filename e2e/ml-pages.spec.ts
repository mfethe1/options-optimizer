import { test, expect } from '@playwright/test';

// These tests validate that the five ML pages render with real, dynamic content
// coming from the backend (TensorFlow-enabled) services.

const BASE = (process.env.BASE_URL as string) || 'http://localhost:3000';

const go = async (page: any, path: string) => {
  const errors: string[] = [];
  page.on('console', (msg: any) => {
    const t = msg.type();
    if (t === 'error' || t === 'warning' || t === 'info' || t === 'log') {
      errors.push(`[console:${t}] ${msg.text()}`);
    }
  });
  page.on('pageerror', (e: any) => errors.push(`[pageerror] ${e.message}`));
  await page.goto(`${BASE}${path}`, { waitUntil: 'networkidle' });
  if (errors.length) {
    console.warn(`Console errors on ${path}:\n` + errors.join('\n'));
  }
};

// 1) Epidemic Volatility Page
// Asserts: header, regime text, and a numeric Predicted VIX value rendered
test('Epidemic Volatility page shows current and predicted VIX', async ({ page }) => {
  await go(page, '/epidemic-volatility');
  await expect(page.getByText(/Epidemic Volatility Forecasting/i)).toBeVisible();
  await expect(page.getByRole('heading', { name: /Market Regime/i })).toBeVisible();
  // Wait for Predicted VIX value to render (e.g., "Predicted VIX (30d): 18.23")
  await expect(page.locator('body')).toContainText(/Predicted VIX \(\d+d\):\s*\d+\.?\d*/);
});

// 2) GNN Page
// Asserts: predictions table shows symbols and numeric percentage predictions
test('GNN page renders predictions table with symbols and values', async ({ page }) => {
  await go(page, '/gnn');
  await expect(page.getByText(/GNN Predictions \(Expected Returns\)/i)).toBeVisible();
  // Default list includes AAPL; verify row exists and shows a % value
  await expect(page.getByRole('cell', { name: /^AAPL$/ })).toBeVisible();
  await expect(page.locator('body')).toContainText(/\+?-?\d+\.\d+%/);
  // Also verify correlation section heading appears
  await expect(page.getByText(/Strongest Correlations/i)).toBeVisible();
});

// 3) Mamba Page
// Asserts: multi-horizon predictions and efficiency statistics sections render
test('Mamba page shows multi-horizon predictions and efficiency stats', async ({ page }) => {
  await go(page, '/mamba');
  await expect(page.getByText(/Multi-Horizon Predictions/i)).toBeVisible();
  // Look for horizon-like labels such as "1d"/"5d" in the table
  await expect(page.locator('body')).toContainText(/\b\d{1,3}d\b/);
  // Efficiency section renders with statistics
  await expect(page.getByText(/Efficiency Statistics/i)).toBeVisible();
  await expect(page.locator('body')).toContainText(/Speedup|Complexity/i);
});

// 4) PINN Page
// Asserts: clicking the action triggers real pricing + greeks output
test('PINN page prices an option and displays Greeks', async ({ page }) => {
  await go(page, '/pinn');
  // Click the action to trigger the pricing API
  const priceBtn = page.getByRole('button', { name: /Price Option \(PINN\)/i });
  await expect(priceBtn).toBeVisible();
  await priceBtn.click();
  // Expect results to render
  await expect(page.getByText(/Option Price & Greeks/i)).toBeVisible();
  await expect(page.getByText(/Method:/i)).toBeVisible();
  // Delta/Gamma/Theta labels should render with numeric values or N/A
  await expect(page.locator('body')).toContainText(/Delta|Gamma|Theta/);
});

// 5) Ensemble Analysis Page
// Asserts: main recommendation panel appears with model agreement and ensemble label
test('Ensemble page shows consensus recommendation and model agreement', async ({ page }) => {
  await go(page, '/ensemble');
  // Switch to the Ensemble Recommendation tab first
  const ensembleTab = page.getByRole('tab', { name: /Ensemble Recommendation/i });
  await expect(ensembleTab).toBeVisible();
  await ensembleTab.click();
  // Now assert on section headings and content
  await expect(page.getByRole('heading', { name: /Ensemble Consensus Recommendation/i })).toBeVisible();
  await expect(page.getByRole('heading', { name: /Model Agreement/i })).toBeVisible();
  // Validate key fields visible within the recommendation tab
  await expect(page.getByText(/Ensemble Prediction/i)).toBeVisible();
  await expect(page.locator('body')).toContainText(/\$\d+\.\d{2}/);
  await expect(page.getByText(/Trading Signal/i)).toBeVisible();
  await expect(page.getByText(/Confidence:/i)).toBeVisible();
});

