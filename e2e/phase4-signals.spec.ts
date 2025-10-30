/**
 * Playwright E2E Tests - Phase 4 Signals Panel
 * 
 * Tests for Phase 4 technical signals panel with real-time WebSocket updates.
 * 
 * Run with: npx playwright test e2e/phase4-signals.spec.ts
 */

import { test, expect, Page } from '@playwright/test';
import WebSocket from 'ws';

const USER_ID = process.env.USER_ID || 'test-user-123';
const BASE_URL = process.env.BASE_URL || 'http://localhost:5173';
const WS_URL = `ws://localhost:8000/ws/phase4-metrics/${USER_ID}`;

// Helper: Connect to WebSocket
async function connectWebSocket(url: string): Promise<WebSocket> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url);
    ws.on('open', () => resolve(ws));
    ws.on('error', reject);
    setTimeout(() => reject(new Error('WebSocket connection timeout')), 5000);
  });
}

test.describe('Phase 4 Signals Panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`${BASE_URL}/phase4-signals-demo`);
    await page.waitForLoadState('domcontentloaded');
  });

  test('should render all 4 signal cards', async ({ page }) => {
    // Verify panel is visible
    const panel = page.locator('[data-testid="phase4-signals-panel"]');
    await expect(panel).toBeVisible();
    
    // Verify all 4 signal cards
    const signalCards = page.locator('[data-testid="signal-card"]');
    await expect(signalCards).toHaveCount(4);
    
    // Verify signal names
    await expect(page.locator('text=Options Flow Composite')).toBeVisible();
    await expect(page.locator('text=Residual Momentum')).toBeVisible();
    await expect(page.locator('text=Seasonality Score')).toBeVisible();
    await expect(page.locator('text=Breadth & Liquidity')).toBeVisible();
  });

  test('should display signal values with correct formatting', async ({ page }) => {
    // Wait for data to load
    await page.waitForTimeout(1000);
    
    // Check Options Flow Composite (percentage)
    const optionsFlow = page.locator('[data-testid="signal-card"]:has-text("Options Flow")');
    const optionsFlowValue = await optionsFlow.locator('[data-testid="signal-value"]').textContent();
    expect(optionsFlowValue).toMatch(/^-?\d+(\.\d+)?%$/); // e.g., "45.2%" or "-12.3%"
    
    // Check Residual Momentum (decimal)
    const momentum = page.locator('[data-testid="signal-card"]:has-text("Residual Momentum")');
    const momentumValue = await momentum.locator('[data-testid="signal-value"]').textContent();
    expect(momentumValue).toMatch(/^-?\d+\.\d+$/); // e.g., "0.45" or "-0.23"
    
    // Check Seasonality Score (percentage)
    const seasonality = page.locator('[data-testid="signal-card"]:has-text("Seasonality")');
    const seasonalityValue = await seasonality.locator('[data-testid="signal-value"]').textContent();
    expect(seasonalityValue).toMatch(/^-?\d+(\.\d+)?%$/);
    
    // Check Breadth & Liquidity (decimal)
    const breadth = page.locator('[data-testid="signal-card"]:has-text("Breadth")');
    const breadthValue = await breadth.locator('[data-testid="signal-value"]').textContent();
    expect(breadthValue).toMatch(/^-?\d+\.\d+$/);
  });

  test('should show color-coded indicators based on signal strength', async ({ page }) => {
    await page.waitForTimeout(1000);
    
    // Check Options Flow Composite indicator
    const optionsFlow = page.locator('[data-testid="signal-card"]:has-text("Options Flow")');
    const indicator = optionsFlow.locator('[data-testid="signal-indicator"]');
    
    // Get background color
    const bgColor = await indicator.evaluate((el) => window.getComputedStyle(el).backgroundColor);
    
    // Should be green (positive), red (negative), or yellow (neutral)
    const isValidColor = 
      bgColor.includes('16, 185, 129') || // green (#10b981)
      bgColor.includes('239, 68, 68') ||  // red (#ef4444)
      bgColor.includes('245, 158, 11');   // yellow (#f59e0b)
    
    expect(isValidColor).toBeTruthy();
  });

  test('should display tooltips on hover', async ({ page }) => {
    await page.waitForTimeout(1000);
    
    // Hover over Options Flow card
    const optionsFlow = page.locator('[data-testid="signal-card"]:has-text("Options Flow")');
    await optionsFlow.hover();
    
    // Wait for tooltip
    await page.waitForTimeout(300);
    
    // Verify tooltip is visible
    const tooltip = page.locator('[role="tooltip"]');
    await expect(tooltip).toBeVisible();
    
    // Verify tooltip contains description
    const tooltipText = await tooltip.textContent();
    expect(tooltipText).toContain('Options Flow');
  });

  test('should update signals in real-time via WebSocket', async ({ page }) => {
    const ws = await connectWebSocket(WS_URL);
    
    // Get initial value
    const optionsFlow = page.locator('[data-testid="signal-card"]:has-text("Options Flow")');
    const initialValue = await optionsFlow.locator('[data-testid="signal-value"]').textContent();
    
    // Send WebSocket update
    ws.send(JSON.stringify({
      type: 'phase4_update',
      data: {
        options_flow_composite: 0.75,
        residual_momentum: 0.45,
        seasonality_score: 0.60,
        breadth_liquidity: 0.80,
      }
    }));
    
    // Wait for update
    await page.waitForTimeout(500);
    
    // Get updated value
    const updatedValue = await optionsFlow.locator('[data-testid="signal-value"]').textContent();
    
    // Value should have changed
    expect(updatedValue).not.toBe(initialValue);
    expect(updatedValue).toBe('75.0%'); // 0.75 * 100 = 75%
    
    ws.close();
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
    const ws = await connectWebSocket(WS_URL);
    
    // Send update with null values
    ws.send(JSON.stringify({
      type: 'phase4_update',
      data: {
        options_flow_composite: null,
        residual_momentum: null,
        seasonality_score: null,
        breadth_liquidity: null,
      }
    }));
    
    await page.waitForTimeout(500);
    
    // Verify "N/A" or "--" is displayed
    const signalCards = page.locator('[data-testid="signal-card"]');
    const firstCardValue = await signalCards.first().locator('[data-testid="signal-value"]').textContent();
    
    expect(firstCardValue).toMatch(/^(N\/A|--|—)$/);
    
    ws.close();
  });

  test('should display trend arrows for signal changes', async ({ page }) => {
    const ws = await connectWebSocket(WS_URL);
    
    // Send initial update
    ws.send(JSON.stringify({
      type: 'phase4_update',
      data: { options_flow_composite: 0.50 }
    }));
    await page.waitForTimeout(500);
    
    // Send update with higher value
    ws.send(JSON.stringify({
      type: 'phase4_update',
      data: { options_flow_composite: 0.75 }
    }));
    await page.waitForTimeout(500);
    
    // Check for up arrow
    const optionsFlow = page.locator('[data-testid="signal-card"]:has-text("Options Flow")');
    const trendArrow = optionsFlow.locator('[data-testid="trend-arrow"]');
    
    await expect(trendArrow).toBeVisible();
    
    // Verify arrow direction (should be up)
    const arrowClass = await trendArrow.getAttribute('class');
    expect(arrowClass).toContain('up');
    
    ws.close();
  });

  test('should maintain WebSocket connection for 60 seconds', async ({ page }) => {
    const ws = await connectWebSocket(WS_URL);
    
    let messagesReceived = 0;
    ws.on('message', () => {
      messagesReceived++;
    });
    
    // Wait 60 seconds
    await page.waitForTimeout(60000);
    
    // Should have received at least 2 updates (30s interval)
    expect(messagesReceived).toBeGreaterThanOrEqual(2);
    
    ws.close();
  });
});

test.describe('Phase 4 Signals - Responsive Design', () => {
  test('should display correctly on mobile viewport', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto(`${BASE_URL}/phase4-signals-demo`);
    
    // Verify panel is visible
    const panel = page.locator('[data-testid="phase4-signals-panel"]');
    await expect(panel).toBeVisible();
    
    // Verify cards stack vertically (1 column)
    const signalCards = page.locator('[data-testid="signal-card"]');
    const firstCard = signalCards.first();
    const secondCard = signalCards.nth(1);
    
    const firstCardBox = await firstCard.boundingBox();
    const secondCardBox = await secondCard.boundingBox();
    
    // Second card should be below first card (not side-by-side)
    expect(secondCardBox!.y).toBeGreaterThan(firstCardBox!.y + firstCardBox!.height);
  });

  test('should display correctly on tablet viewport', async ({ page }) => {
    // Set tablet viewport
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto(`${BASE_URL}/phase4-signals-demo`);
    
    // Verify panel is visible
    const panel = page.locator('[data-testid="phase4-signals-panel"]');
    await expect(panel).toBeVisible();
    
    // Verify cards are in 2 columns
    const signalCards = page.locator('[data-testid="signal-card"]');
    const firstCard = signalCards.first();
    const secondCard = signalCards.nth(1);
    
    const firstCardBox = await firstCard.boundingBox();
    const secondCardBox = await secondCard.boundingBox();
    
    // Second card should be on same row (side-by-side)
    expect(Math.abs(secondCardBox!.y - firstCardBox!.y)).toBeLessThan(10);
  });

  test('should display correctly on desktop viewport', async ({ page }) => {
    // Set desktop viewport
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto(`${BASE_URL}/phase4-signals-demo`);
    
    // Verify panel is visible
    const panel = page.locator('[data-testid="phase4-signals-panel"]');
    await expect(panel).toBeVisible();
    
    // Verify all 4 cards are in one row
    const signalCards = page.locator('[data-testid="signal-card"]');
    const boxes = await Promise.all([
      signalCards.nth(0).boundingBox(),
      signalCards.nth(1).boundingBox(),
      signalCards.nth(2).boundingBox(),
      signalCards.nth(3).boundingBox(),
    ]);
    
    // All cards should be on same row
    const yPositions = boxes.map(box => box!.y);
    const maxYDiff = Math.max(...yPositions) - Math.min(...yPositions);
    expect(maxYDiff).toBeLessThan(10);
  });
});

