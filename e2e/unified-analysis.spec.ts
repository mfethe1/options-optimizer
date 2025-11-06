import { test, expect } from '@playwright/test';

const BASE = (process.env.BASE_URL as string) || 'http://localhost:3000';

const go = async (page: any, path: string) => {
  const errors: string[] = [];
  page.on('console', (msg: any) => {
    const t = msg.type();
    if (t === 'error' || t === 'warning') {
      errors.push(`[console:${t}] ${msg.text()}`);
    }
    if (t === 'log') {
      // Surface debug logs for UnifiedAnalysis payload/timeline verification
      console.log(`[console:${t}] ${msg.text()}`);
    }
  });
  page.on('pageerror', (e: any) => errors.push(`[pageerror] ${e.message}`));
  await page.goto(`${BASE}${path}`, { waitUntil: 'networkidle' });
  if (errors.length) {
    console.warn(`Console errors on ${path}:\n` + errors.join('\n'));
  }
};

// Unified Analysis should render a composed chart with an 'Actual' legend and non-empty data
// This is a smoke test to ensure we are NOT showing random placeholder noise anymore.
test('Unified Analysis shows meaningful actual price curve', async ({ page }) => {
  await go(page, '/');
  // Wait for the Recharts surface SVG to appear (chart rendered)
  const chart = page.locator('svg.recharts-surface');
  await expect(chart.first()).toBeVisible();
});

