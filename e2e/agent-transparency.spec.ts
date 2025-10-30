/**
 * Playwright E2E Tests - Agent Transparency System
 * 
 * Comprehensive end-to-end tests for institutional-grade options analysis platform.
 * Tests cover: Agent transparency, Phase 4 signals, Risk panel, API integration, Performance.
 * 
 * Run with: npx playwright test e2e/agent-transparency.spec.ts
 */

import { test, expect, Page } from '@playwright/test';
import WebSocket from 'ws';

const USER_ID = process.env.USER_ID || 'test-user-123';
const BASE_URL = process.env.BASE_URL || 'http://localhost:5173';
const API_URL = process.env.API_URL || 'http://localhost:8000';
const WS_URL = `ws://localhost:8000/ws/agent-stream/${USER_ID}`;

// Helper: Wait for WebSocket connection
async function connectWebSocket(url: string): Promise<WebSocket> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url);
    ws.on('open', () => resolve(ws));
    ws.on('error', reject);
    setTimeout(() => reject(new Error('WebSocket connection timeout')), 5000);
  });
}

// Helper: Wait for WebSocket message matching condition
async function waitForMessage(
  ws: WebSocket,
  condition: (msg: any) => boolean,
  timeout = 10000
): Promise<any> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Message timeout')), timeout);
    ws.on('message', (data) => {
      try {
        const msg = JSON.parse(data.toString());
        if (condition(msg)) {
          clearTimeout(timer);
          resolve(msg);
        }
      } catch (e) {
        // Ignore parse errors
      }
    });
  });
}

test.describe('Agent Transparency System', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`${BASE_URL}/agent-transparency`);
    await page.waitForLoadState('domcontentloaded');
  });

  test('should connect to WebSocket and receive heartbeat', async ({ page }) => {
    const ws = await connectWebSocket(WS_URL);
    
    // Wait for heartbeat message
    const heartbeat = await waitForMessage(
      ws,
      (msg) => msg.type === 'heartbeat',
      30000 // 30s timeout (heartbeat every 30s)
    );
    
    expect(heartbeat).toBeDefined();
    expect(heartbeat.type).toBe('heartbeat');
    
    ws.close();
  });

  test('should display agent events in real-time', async ({ page }) => {
    // Connect to WebSocket
    const ws = await connectWebSocket(WS_URL);
    
    // Trigger agent analysis (simulate by sending mock events)
    const mockEvents = [
      { type: 'agent_event', data: { event_type: 'STARTED', content: 'Starting analysis...' } },
      { type: 'agent_event', data: { event_type: 'THINKING', content: 'Computing portfolio metrics...' } },
      { type: 'agent_event', data: { event_type: 'TOOL_CALL', content: 'compute_portfolio_metrics', metadata: { args: {} } } },
      { type: 'agent_event', data: { event_type: 'TOOL_RESULT', content: 'Success', metadata: { result: {} } } },
      { type: 'agent_event', data: { event_type: 'PROGRESS', content: 'Progress update', metadata: { progress_pct: 50 } } },
      { type: 'agent_event', data: { event_type: 'COMPLETED', content: 'Analysis complete' } },
    ];
    
    // Send mock events
    for (const event of mockEvents) {
      ws.send(JSON.stringify(event));
      await page.waitForTimeout(100); // Small delay between events
    }
    
    // Verify events are displayed
    const conversationDisplay = page.locator('[data-testid="conversation-display"]');
    await expect(conversationDisplay).toBeVisible();
    
    // Check for event type labels
    await expect(page.locator('text=STARTED')).toBeVisible();
    await expect(page.locator('text=THINKING')).toBeVisible();
    await expect(page.locator('text=TOOL_CALL')).toBeVisible();
    await expect(page.locator('text=COMPLETED')).toBeVisible();
    
    ws.close();
  });

  test('should update progress bar in real-time', async ({ page }) => {
    const ws = await connectWebSocket(WS_URL);
    
    // Send progress updates
    const progressUpdates = [
      { type: 'agent_progress', data: { progress_pct: 0, current_step: 'Initialization' } },
      { type: 'agent_progress', data: { progress_pct: 25, current_step: 'Computing metrics' } },
      { type: 'agent_progress', data: { progress_pct: 50, current_step: 'Phase 4 analysis' } },
      { type: 'agent_progress', data: { progress_pct: 75, current_step: 'Risk analysis' } },
      { type: 'agent_progress', data: { progress_pct: 100, current_step: 'Complete' } },
    ];
    
    for (const update of progressUpdates) {
      ws.send(JSON.stringify(update));
      await page.waitForTimeout(200);
    }
    
    // Verify progress bar
    const progressBar = page.locator('[data-testid="progress-bar"]');
    await expect(progressBar).toBeVisible();
    
    // Check final progress value (should be 100%)
    const progressValue = await progressBar.getAttribute('aria-valuenow');
    expect(parseInt(progressValue || '0')).toBeGreaterThanOrEqual(75);
    
    ws.close();
  });

  test('should auto-scroll conversation to bottom', async ({ page }) => {
    const ws = await connectWebSocket(WS_URL);
    
    // Send many events to trigger scroll
    for (let i = 0; i < 20; i++) {
      ws.send(JSON.stringify({
        type: 'agent_event',
        data: { event_type: 'THINKING', content: `Event ${i}` }
      }));
      await page.waitForTimeout(50);
    }
    
    // Check if conversation is scrolled to bottom
    const conversationDisplay = page.locator('[data-testid="conversation-display"]');
    const scrollTop = await conversationDisplay.evaluate((el) => el.scrollTop);
    const scrollHeight = await conversationDisplay.evaluate((el) => el.scrollHeight);
    const clientHeight = await conversationDisplay.evaluate((el) => el.clientHeight);
    
    // Should be scrolled to bottom (within 50px tolerance)
    expect(scrollTop + clientHeight).toBeGreaterThanOrEqual(scrollHeight - 50);
    
    ws.close();
  });

  test('should search and filter conversation messages', async ({ page }) => {
    const ws = await connectWebSocket(WS_URL);
    
    // Send events with different content
    const events = [
      { type: 'agent_event', data: { event_type: 'THINKING', content: 'Computing portfolio metrics' } },
      { type: 'agent_event', data: { event_type: 'THINKING', content: 'Analyzing risk factors' } },
      { type: 'agent_event', data: { event_type: 'TOOL_CALL', content: 'compute_phase4_metrics' } },
    ];
    
    for (const event of events) {
      ws.send(JSON.stringify(event));
      await page.waitForTimeout(100);
    }
    
    // Use search input
    const searchInput = page.locator('[data-testid="search-input"]');
    await searchInput.fill('portfolio');
    
    // Verify filtered results
    const messages = page.locator('[data-testid="message"]');
    const count = await messages.count();
    expect(count).toBe(1); // Only 1 message contains "portfolio"
    
    await expect(page.locator('text=Computing portfolio metrics')).toBeVisible();
    await expect(page.locator('text=Analyzing risk factors')).not.toBeVisible();
    
    ws.close();
  });

  test('should export conversation log', async ({ page }) => {
    const ws = await connectWebSocket(WS_URL);
    
    // Send some events
    ws.send(JSON.stringify({
      type: 'agent_event',
      data: { event_type: 'THINKING', content: 'Test event' }
    }));
    await page.waitForTimeout(200);
    
    // Click export button
    const exportButton = page.locator('[data-testid="export-button"]');
    
    // Set up download listener
    const downloadPromise = page.waitForEvent('download');
    await exportButton.click();
    const download = await downloadPromise;
    
    // Verify download
    expect(download.suggestedFilename()).toContain('agent-conversation');
    expect(download.suggestedFilename()).toContain('.txt');
    
    ws.close();
  });

  test('should handle WebSocket disconnection and reconnection', async ({ page }) => {
    const ws = await connectWebSocket(WS_URL);
    
    // Verify connected status
    await expect(page.locator('text=Connected')).toBeVisible();
    
    // Close WebSocket
    ws.close();
    await page.waitForTimeout(1000);
    
    // Verify disconnected status
    await expect(page.locator('text=Disconnected')).toBeVisible();
    
    // Wait for auto-reconnect (should happen within 10s)
    await expect(page.locator('text=Connected')).toBeVisible({ timeout: 15000 });
  });
});

test.describe('Performance Tests', () => {
  test('should load page in under 2 seconds', async ({ page }) => {
    const startTime = Date.now();
    await page.goto(`${BASE_URL}/agent-transparency`);
    await page.waitForLoadState('domcontentloaded');
    const loadTime = Date.now() - startTime;
    
    expect(loadTime).toBeLessThan(2000);
  });

  test('should have smooth 60fps animations', async ({ page }) => {
    await page.goto(`${BASE_URL}/agent-transparency`);
    
    // Measure frame rate during progress bar animation
    const fps = await page.evaluate(() => {
      return new Promise<number>((resolve) => {
        let frames = 0;
        const startTime = performance.now();
        
        function countFrame() {
          frames++;
          if (performance.now() - startTime < 1000) {
            requestAnimationFrame(countFrame);
          } else {
            resolve(frames);
          }
        }
        
        requestAnimationFrame(countFrame);
      });
    });
    
    expect(fps).toBeGreaterThanOrEqual(55); // Allow 5fps tolerance
  });

  test('should have no memory leaks after 100 events', async ({ page }) => {
    await page.goto(`${BASE_URL}/agent-transparency`);
    const ws = await connectWebSocket(WS_URL);
    
    // Get initial memory usage
    const initialMemory = await page.evaluate(() => (performance as any).memory?.usedJSHeapSize || 0);
    
    // Send 100 events
    for (let i = 0; i < 100; i++) {
      ws.send(JSON.stringify({
        type: 'agent_event',
        data: { event_type: 'THINKING', content: `Event ${i}` }
      }));
      await page.waitForTimeout(10);
    }
    
    // Force garbage collection (if available)
    await page.evaluate(() => {
      if ((window as any).gc) {
        (window as any).gc();
      }
    });
    
    // Get final memory usage
    const finalMemory = await page.evaluate(() => (performance as any).memory?.usedJSHeapSize || 0);
    
    // Memory should not increase by more than 10MB
    const memoryIncrease = (finalMemory - initialMemory) / (1024 * 1024);
    expect(memoryIncrease).toBeLessThan(10);
    
    ws.close();
  });
});

