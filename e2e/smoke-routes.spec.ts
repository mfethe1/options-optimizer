import { test, expect } from '@playwright/test';

const routes: { path: string; expectText?: string | RegExp }[] = [
  { path: '/', expectText: /Unified|Analysis|Dashboard/i },
  { path: '/charts-demo', expectText: /TradingView Lightweight Charts Demo/i },
  { path: '/ml-predictions', expectText: /ML|Prediction/i },
  { path: '/gnn', expectText: /GNN|Graph Neural/i },
  { path: '/epidemic-volatility', expectText: /Volatility|Epidemic/i },
  { path: '/ensemble', expectText: /Ensemble/i },
  { path: '/positions', expectText: /Positions|Portfolio/i },
];

test.describe('Smoke route rendering', () => {
  for (const r of routes) {
    test(`loads ${r.path} without blank screen`, async ({ page }) => {
      const errors: string[] = [];
      page.on('console', msg => {
        if (msg.type() === 'error') errors.push(`[console:${msg.type()}] ${msg.text()}`);
      });
      page.on('pageerror', (e) => errors.push(`[pageerror] ${e.message}`));

      await page.goto(r.path, { waitUntil: 'domcontentloaded' });

      // If the page threw, surface the errors
      if (errors.length) {
        console.error(`Errors for ${r.path}:\n` + errors.join('\n'));
      }

      // Ensure body has content and not empty
      const bodyText = await page.locator('body').innerText();
      expect(bodyText.trim().length).toBeGreaterThan(0);

      // Optionally match expected text if provided
      if (r.expectText) {
        await expect(page.locator('body')).toContainText(r.expectText);
      }
    });
  }
});

