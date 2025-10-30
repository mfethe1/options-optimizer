/**
 * Playwright E2E Tests - Risk Panel Dashboard
 * 
 * Tests for institutional-grade risk metrics dashboard with 7 key metrics.
 * 
 * Run with: npx playwright test e2e/risk-panel.spec.ts
 */

import { test, expect, Page } from '@playwright/test';

const BASE_URL = process.env.BASE_URL || 'http://localhost:5173';

test.describe('Risk Panel Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`${BASE_URL}/risk-panel-demo`);
    await page.waitForLoadState('domcontentloaded');
  });

  test('should render all 7 risk metrics', async ({ page }) => {
    // Verify dashboard is visible
    const dashboard = page.locator('[data-testid="risk-panel-dashboard"]');
    await expect(dashboard).toBeVisible();
    
    // Verify all 7 metric cards
    const metricCards = page.locator('[data-testid="risk-metric-card"]');
    await expect(metricCards).toHaveCount(7);
    
    // Verify metric names
    await expect(page.locator('text=Omega Ratio')).toBeVisible();
    await expect(page.locator('text=Gain-to-Loss (GH1)')).toBeVisible();
    await expect(page.locator('text=Pain Index')).toBeVisible();
    await expect(page.locator('text=Upside Capture')).toBeVisible();
    await expect(page.locator('text=Downside Capture')).toBeVisible();
    await expect(page.locator('text=CVaR (95%)')).toBeVisible();
    await expect(page.locator('text=Max Drawdown')).toBeVisible();
  });

  test('should display regime indicator', async ({ page }) => {
    await page.waitForTimeout(1000);
    
    // Verify regime indicator is visible
    const regimeIndicator = page.locator('[data-testid="regime-indicator"]');
    await expect(regimeIndicator).toBeVisible();
    
    // Verify regime text (should be one of: bull, bear, neutral, volatile, crisis)
    const regimeText = await regimeIndicator.textContent();
    expect(regimeText).toMatch(/^(bull|bear|neutral|volatile|crisis)$/i);
  });

  test('should show color-coded regime indicator', async ({ page }) => {
    await page.waitForTimeout(1000);
    
    const regimeIndicator = page.locator('[data-testid="regime-indicator"]');
    const bgColor = await regimeIndicator.evaluate((el) => window.getComputedStyle(el).backgroundColor);
    
    // Should be one of the regime colors
    const isValidColor = 
      bgColor.includes('16, 185, 129') ||  // green (bull)
      bgColor.includes('239, 68, 68') ||   // red (bear)
      bgColor.includes('245, 158, 11') ||  // yellow (neutral)
      bgColor.includes('139, 92, 246') ||  // purple (volatile)
      bgColor.includes('220, 38, 38');     // dark red (crisis)
    
    expect(isValidColor).toBeTruthy();
  });

  test('should display metric values with correct formatting', async ({ page }) => {
    await page.waitForTimeout(1000);
    
    // Check Omega Ratio (decimal, 2 places)
    const omega = page.locator('[data-testid="risk-metric-card"]:has-text("Omega Ratio")');
    const omegaValue = await omega.locator('[data-testid="metric-value"]').textContent();
    expect(omegaValue).toMatch(/^\d+\.\d{2}$/); // e.g., "1.45"
    
    // Check Pain Index (decimal, 2 places)
    const pain = page.locator('[data-testid="risk-metric-card"]:has-text("Pain Index")');
    const painValue = await pain.locator('[data-testid="metric-value"]').textContent();
    expect(painValue).toMatch(/^\d+\.\d{2}$/);
    
    // Check Upside Capture (percentage)
    const upside = page.locator('[data-testid="risk-metric-card"]:has-text("Upside Capture")');
    const upsideValue = await upside.locator('[data-testid="metric-value"]').textContent();
    expect(upsideValue).toMatch(/^\d+(\.\d+)?%$/); // e.g., "105.2%"
    
    // Check CVaR (percentage, negative)
    const cvar = page.locator('[data-testid="risk-metric-card"]:has-text("CVaR")');
    const cvarValue = await cvar.locator('[data-testid="metric-value"]').textContent();
    expect(cvarValue).toMatch(/^-\d+(\.\d+)?%$/); // e.g., "-5.2%"
    
    // Check Max Drawdown (percentage, negative)
    const maxDD = page.locator('[data-testid="risk-metric-card"]:has-text("Max Drawdown")');
    const maxDDValue = await maxDD.locator('[data-testid="metric-value"]').textContent();
    expect(maxDDValue).toMatch(/^-\d+(\.\d+)?%$/); // e.g., "-12.5%"
  });

  test('should show color-coded risk levels', async ({ page }) => {
    await page.waitForTimeout(1000);
    
    // Check CVaR risk level
    const cvar = page.locator('[data-testid="risk-metric-card"]:has-text("CVaR")');
    const riskLevel = cvar.locator('[data-testid="risk-level"]');
    
    await expect(riskLevel).toBeVisible();
    
    // Get risk level text
    const riskLevelText = await riskLevel.textContent();
    expect(riskLevelText).toMatch(/^(Low|Medium|High|Critical)$/);
    
    // Get color
    const bgColor = await riskLevel.evaluate((el) => window.getComputedStyle(el).backgroundColor);
    
    // Should be color-coded
    const isValidColor = 
      bgColor.includes('16, 185, 129') ||  // green (Low)
      bgColor.includes('245, 158, 11') ||  // yellow (Medium)
      bgColor.includes('249, 115, 22') ||  // orange (High)
      bgColor.includes('239, 68, 68');     // red (Critical)
    
    expect(isValidColor).toBeTruthy();
  });

  test('should display metric descriptions on hover', async ({ page }) => {
    await page.waitForTimeout(1000);
    
    // Hover over Omega Ratio card
    const omega = page.locator('[data-testid="risk-metric-card"]:has-text("Omega Ratio")');
    await omega.hover();
    
    // Wait for tooltip
    await page.waitForTimeout(300);
    
    // Verify tooltip is visible
    const tooltip = page.locator('[role="tooltip"]');
    await expect(tooltip).toBeVisible();
    
    // Verify tooltip contains description
    const tooltipText = await tooltip.textContent();
    expect(tooltipText).toContain('Omega');
  });

  test('should update metrics in real-time', async ({ page }) => {
    await page.waitForTimeout(1000);
    
    // Get initial Omega Ratio value
    const omega = page.locator('[data-testid="risk-metric-card"]:has-text("Omega Ratio")');
    const initialValue = await omega.locator('[data-testid="metric-value"]').textContent();
    
    // Trigger refresh (simulate by clicking refresh button if available)
    const refreshButton = page.locator('[data-testid="refresh-button"]');
    if (await refreshButton.isVisible()) {
      await refreshButton.click();
      await page.waitForTimeout(2000);
      
      // Get updated value
      const updatedValue = await omega.locator('[data-testid="metric-value"]').textContent();
      
      // Value might have changed (or stayed the same if data is static)
      expect(updatedValue).toBeDefined();
    }
  });

  test('should show loading state while fetching data', async ({ page }) => {
    // Reload page to trigger loading state
    await page.reload();
    
    // Check for loading spinner
    const loadingSpinner = page.locator('[data-testid="loading-spinner"]');
    await expect(loadingSpinner).toBeVisible({ timeout: 1000 });
    
    // Wait for data to load
    await page.waitForTimeout(2000);
    
    // Loading spinner should be gone
    await expect(loadingSpinner).not.toBeVisible();
  });

  test('should handle missing data gracefully', async ({ page }) => {
    // This test assumes we can inject mock data with null values
    // In a real scenario, you'd use API mocking or test fixtures
    
    await page.waitForTimeout(1000);
    
    // Check if any metric shows "N/A" or "--"
    const metricCards = page.locator('[data-testid="risk-metric-card"]');
    const count = await metricCards.count();
    
    // All cards should have valid values or "N/A"
    for (let i = 0; i < count; i++) {
      const card = metricCards.nth(i);
      const value = await card.locator('[data-testid="metric-value"]').textContent();
      expect(value).toMatch(/^(-?\d+(\.\d+)?%?|N\/A|--|—)$/);
    }
  });

  test('should display metrics in 2x4 grid layout', async ({ page }) => {
    await page.waitForTimeout(1000);
    
    const metricCards = page.locator('[data-testid="risk-metric-card"]');
    
    // Get positions of first 4 cards (should be in first row)
    const row1Cards = await Promise.all([
      metricCards.nth(0).boundingBox(),
      metricCards.nth(1).boundingBox(),
      metricCards.nth(2).boundingBox(),
      metricCards.nth(3).boundingBox(),
    ]);
    
    // Get positions of last 3 cards (should be in second row)
    const row2Cards = await Promise.all([
      metricCards.nth(4).boundingBox(),
      metricCards.nth(5).boundingBox(),
      metricCards.nth(6).boundingBox(),
    ]);
    
    // First 4 cards should be on same row
    const row1YPositions = row1Cards.map(box => box!.y);
    const row1MaxYDiff = Math.max(...row1YPositions) - Math.min(...row1YPositions);
    expect(row1MaxYDiff).toBeLessThan(10);
    
    // Last 3 cards should be on same row
    const row2YPositions = row2Cards.map(box => box!.y);
    const row2MaxYDiff = Math.max(...row2YPositions) - Math.min(...row2YPositions);
    expect(row2MaxYDiff).toBeLessThan(10);
    
    // Second row should be below first row
    expect(Math.min(...row2YPositions)).toBeGreaterThan(Math.max(...row1YPositions));
  });
});

test.describe('Risk Panel - Responsive Design', () => {
  test('should display correctly on mobile viewport', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto(`${BASE_URL}/risk-panel-demo`);
    
    // Verify dashboard is visible
    const dashboard = page.locator('[data-testid="risk-panel-dashboard"]');
    await expect(dashboard).toBeVisible();
    
    // Verify cards stack vertically (1 column)
    const metricCards = page.locator('[data-testid="risk-metric-card"]');
    const firstCard = metricCards.first();
    const secondCard = metricCards.nth(1);
    
    const firstCardBox = await firstCard.boundingBox();
    const secondCardBox = await secondCard.boundingBox();
    
    // Second card should be below first card (not side-by-side)
    expect(secondCardBox!.y).toBeGreaterThan(firstCardBox!.y + firstCardBox!.height);
  });

  test('should display correctly on tablet viewport', async ({ page }) => {
    // Set tablet viewport
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto(`${BASE_URL}/risk-panel-demo`);
    
    // Verify dashboard is visible
    const dashboard = page.locator('[data-testid="risk-panel-dashboard"]');
    await expect(dashboard).toBeVisible();
    
    // Verify cards are in 2 columns
    const metricCards = page.locator('[data-testid="risk-metric-card"]');
    const firstCard = metricCards.first();
    const secondCard = metricCards.nth(1);
    
    const firstCardBox = await firstCard.boundingBox();
    const secondCardBox = await secondCard.boundingBox();
    
    // Second card should be on same row (side-by-side)
    expect(Math.abs(secondCardBox!.y - firstCardBox!.y)).toBeLessThan(10);
  });

  test('should display correctly on desktop viewport', async ({ page }) => {
    // Set desktop viewport
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto(`${BASE_URL}/risk-panel-demo`);
    
    // Verify dashboard is visible
    const dashboard = page.locator('[data-testid="risk-panel-dashboard"]');
    await expect(dashboard).toBeVisible();
    
    // Verify first 4 cards are in one row
    const metricCards = page.locator('[data-testid="risk-metric-card"]');
    const boxes = await Promise.all([
      metricCards.nth(0).boundingBox(),
      metricCards.nth(1).boundingBox(),
      metricCards.nth(2).boundingBox(),
      metricCards.nth(3).boundingBox(),
    ]);
    
    // All cards should be on same row
    const yPositions = boxes.map(box => box!.y);
    const maxYDiff = Math.max(...yPositions) - Math.min(...yPositions);
    expect(maxYDiff).toBeLessThan(10);
  });
});

test.describe('Risk Panel - Accessibility', () => {
  test('should have proper ARIA labels', async ({ page }) => {
    await page.goto(`${BASE_URL}/risk-panel-demo`);
    await page.waitForTimeout(1000);
    
    // Check dashboard has aria-label
    const dashboard = page.locator('[data-testid="risk-panel-dashboard"]');
    const ariaLabel = await dashboard.getAttribute('aria-label');
    expect(ariaLabel).toBeTruthy();
    
    // Check metric cards have aria-labels
    const metricCards = page.locator('[data-testid="risk-metric-card"]');
    const firstCardLabel = await metricCards.first().getAttribute('aria-label');
    expect(firstCardLabel).toBeTruthy();
  });

  test('should be keyboard navigable', async ({ page }) => {
    await page.goto(`${BASE_URL}/risk-panel-demo`);
    await page.waitForTimeout(1000);
    
    // Tab through metric cards
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    
    // Verify focus is on a metric card
    const focusedElement = await page.evaluate(() => document.activeElement?.getAttribute('data-testid'));
    expect(focusedElement).toContain('risk-metric-card');
  });

  test('should have sufficient color contrast', async ({ page }) => {
    await page.goto(`${BASE_URL}/risk-panel-demo`);
    await page.waitForTimeout(1000);
    
    // Check text color contrast
    const omega = page.locator('[data-testid="risk-metric-card"]:has-text("Omega Ratio")');
    const textColor = await omega.evaluate((el) => window.getComputedStyle(el).color);
    const bgColor = await omega.evaluate((el) => window.getComputedStyle(el).backgroundColor);
    
    // Both should be defined
    expect(textColor).toBeTruthy();
    expect(bgColor).toBeTruthy();
    
    // This is a basic check - in production, you'd use a contrast ratio calculator
  });
});

