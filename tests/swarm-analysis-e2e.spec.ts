/**
 * End-to-End Playwright Test for Swarm Analysis Page
 * 
 * Tests the complete CSV upload and analysis flow with real 17-agent swarm
 */
import { test, expect } from '@playwright/test';
import path from 'path';

// Configuration
const FRONTEND_URL = 'http://localhost:5173';
const BACKEND_URL = 'http://localhost:8000';
const TEST_CSV_PATH = path.join(__dirname, '../data/examples/positions.csv');
const ANALYSIS_TIMEOUT = 300000; // 5 minutes for LLM calls

test.describe('Swarm Analysis Page - End-to-End', () => {
  
  test.beforeEach(async ({ page }) => {
    // Check if backend is running
    const healthCheck = await page.request.get(`${BACKEND_URL}/health`);
    expect(healthCheck.ok()).toBeTruthy();
    
    // Navigate to swarm analysis page
    await page.goto(FRONTEND_URL);
    await page.waitForLoadState('networkidle');
  });

  test('should display swarm analysis page correctly', async ({ page }) => {
    // Verify page title
    await expect(page.locator('h4:has-text("AI-Powered Swarm Analysis")')).toBeVisible();
    
    // Verify description
    await expect(page.locator('text=Upload your portfolio CSV')).toBeVisible();
    
    // Verify upload button
    await expect(page.locator('button:has-text("Upload CSV")')).toBeVisible();
  });

  test('should upload CSV and run full swarm analysis', async ({ page }) => {
    console.log('Starting CSV upload and analysis test...');
    
    // Click upload button to open dialog
    await page.click('button:has-text("Upload CSV")');
    await expect(page.locator('dialog')).toBeVisible();
    
    // Upload CSV file
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_CSV_PATH);
    
    // Verify file selected
    await expect(page.locator('text=positions.csv')).toBeVisible();
    
    // Check Chase format checkbox if needed
    // await page.check('input[type="checkbox"]');
    
    // Start analysis
    console.log('Starting analysis...');
    await page.click('button:has-text("Analyze with AI")');
    
    // Wait for loading indicator
    await expect(page.locator('text=Analyzing portfolio')).toBeVisible();
    
    // Wait for analysis to complete (this will take 3-5 minutes)
    console.log('Waiting for analysis to complete (this may take 3-5 minutes)...');
    await page.waitForSelector('[data-testid="consensus-decisions"]', {
      timeout: ANALYSIS_TIMEOUT,
      state: 'visible'
    });
    
    console.log('Analysis complete! Verifying results...');
    
    // Verify import stats
    await expect(page.locator('text=Imported')).toBeVisible();
    await expect(page.locator('text=positions')).toBeVisible();
    
    // Verify consensus decisions are displayed
    await expect(page.locator('h5:has-text("AI Consensus Recommendations")')).toBeVisible();
    
    // Verify all three consensus decisions
    await expect(page.locator('text=Overall Action')).toBeVisible();
    await expect(page.locator('text=Risk Level')).toBeVisible();
    await expect(page.locator('text=Market Outlook')).toBeVisible();
    
    // Verify confidence percentages are shown
    const confidenceElements = page.locator('text=/Confidence: \\d+%/');
    await expect(confidenceElements.first()).toBeVisible();
    
    // Verify portfolio summary
    await expect(page.locator('h6:has-text("Portfolio Summary")')).toBeVisible();
    await expect(page.locator('text=Total Value')).toBeVisible();
    await expect(page.locator('text=Unrealized P&L')).toBeVisible();
    
    // Take screenshot of results
    await page.screenshot({
      path: 'test-results/swarm-analysis-results.png',
      fullPage: true
    });
    
    console.log('Test completed successfully!');
  });

  test('should display enhanced swarm health metrics', async ({ page }) => {
    // This test assumes analysis has already been run
    // In a real scenario, you'd run the analysis first or use a saved state
    
    // Upload and analyze (abbreviated)
    await page.click('button:has-text("Upload CSV")');
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_CSV_PATH);
    await page.click('button:has-text("Analyze with AI")');
    
    // Wait for results
    await page.waitForSelector('[data-testid="consensus-decisions"]', {
      timeout: ANALYSIS_TIMEOUT
    });
    
    // Verify swarm health metrics are displayed
    await expect(page.locator('h5:has-text("Swarm Health & Performance")')).toBeVisible();
    
    // Verify agent contribution metrics
    await expect(page.locator('text=Agent Contribution')).toBeVisible();
    await expect(page.locator('text=Success Rate')).toBeVisible();
    await expect(page.locator('text=Total Agents')).toBeVisible();
    await expect(page.locator('text=Contributed')).toBeVisible();
    
    // Verify communication stats
    await expect(page.locator('text=Communication Stats')).toBeVisible();
    await expect(page.locator('text=Total Messages')).toBeVisible();
    
    // Verify consensus strength
    await expect(page.locator('text=Consensus Strength')).toBeVisible();
    await expect(page.locator('text=Overall Consensus')).toBeVisible();
  });

  test('should display position-by-position analysis', async ({ page }) => {
    // Upload and analyze
    await page.click('button:has-text("Upload CSV")');
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_CSV_PATH);
    await page.click('button:has-text("Analyze with AI")');
    
    // Wait for results
    await page.waitForSelector('[data-testid="consensus-decisions"]', {
      timeout: ANALYSIS_TIMEOUT
    });
    
    // Verify position analysis panel
    await expect(page.locator('h5:has-text("Position-by-Position AI Analysis")')).toBeVisible();
    
    // Verify at least one position is shown
    const positionAccordions = page.locator('[role="button"]:has-text("NVDA"), [role="button"]:has-text("TSLA")');
    await expect(positionAccordions.first()).toBeVisible();
    
    // Click to expand first position
    await positionAccordions.first().click();
    
    // Verify tabs are shown
    await expect(page.locator('button:has-text("Overview")')).toBeVisible();
    await expect(page.locator('button:has-text("Agent Insights")')).toBeVisible();
    await expect(page.locator('button:has-text("Stock Report")')).toBeVisible();
    await expect(page.locator('button:has-text("Recommendations")')).toBeVisible();
    await expect(page.locator('button:has-text("Risks & Opportunities")')).toBeVisible();
    
    // Click through tabs
    await page.click('button:has-text("Agent Insights")');
    await expect(page.locator('text=agent_type, text=confidence')).toBeVisible();
    
    await page.click('button:has-text("Stock Report")');
    // Stock report should be visible
    
    await page.click('button:has-text("Recommendations")');
    // Recommendations should be visible
    
    await page.click('button:has-text("Risks & Opportunities")');
    await expect(page.locator('text=Risk Warnings, text=Opportunities')).toBeVisible();
    
    // Take screenshot
    await page.screenshot({
      path: 'test-results/position-analysis-expanded.png',
      fullPage: true
    });
  });

  test('should handle errors gracefully', async ({ page }) => {
    // Try to analyze without selecting a file
    await page.click('button:has-text("Upload CSV")');
    await page.click('button:has-text("Analyze with AI")');
    
    // Should show error
    await expect(page.locator('text=Please select a CSV file')).toBeVisible();
  });

  test('should validate API response structure', async ({ page }) => {
    // Intercept API call
    let apiResponse: any = null;
    
    page.on('response', async (response) => {
      if (response.url().includes('/api/swarm/analyze-csv')) {
        apiResponse = await response.json();
      }
    });
    
    // Upload and analyze
    await page.click('button:has-text("Upload CSV")');
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_CSV_PATH);
    await page.click('button:has-text("Analyze with AI")');
    
    // Wait for results
    await page.waitForSelector('[data-testid="consensus-decisions"]', {
      timeout: ANALYSIS_TIMEOUT
    });
    
    // Validate response structure
    expect(apiResponse).toBeTruthy();
    expect(apiResponse.consensus_decisions).toBeTruthy();
    expect(apiResponse.portfolio_summary).toBeTruthy();
    expect(apiResponse.timestamp).toBeTruthy();
    
    // Validate enhanced fields
    expect(apiResponse.agent_insights).toBeTruthy();
    expect(Array.isArray(apiResponse.agent_insights)).toBeTruthy();
    expect(apiResponse.agent_insights.length).toBeGreaterThan(0);
    
    expect(apiResponse.position_analysis).toBeTruthy();
    expect(Array.isArray(apiResponse.position_analysis)).toBeTruthy();
    
    expect(apiResponse.swarm_health).toBeTruthy();
    expect(apiResponse.swarm_health.active_agents_count).toBe(17);
    
    console.log('API Response Structure Validation:', {
      agentInsightsCount: apiResponse.agent_insights.length,
      positionAnalysisCount: apiResponse.position_analysis.length,
      activeAgents: apiResponse.swarm_health.active_agents_count,
      contributedAgents: apiResponse.swarm_health.contributed_vs_failed.contributed,
      failedAgents: apiResponse.swarm_health.contributed_vs_failed.failed
    });
  });

  test('should log detailed debugging information', async ({ page }) => {
    // Enable console logging
    page.on('console', msg => {
      if (msg.text().includes('[SwarmService]')) {
        console.log('Frontend Log:', msg.text());
      }
    });
    
    // Upload and analyze
    await page.click('button:has-text("Upload CSV")');
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_CSV_PATH);
    await page.click('button:has-text("Analyze with AI")');
    
    // Wait for results
    await page.waitForSelector('[data-testid="consensus-decisions"]', {
      timeout: ANALYSIS_TIMEOUT
    });
    
    // Logs should have been printed to console
    console.log('Test completed - check console logs above for detailed debugging info');
  });
});

test.describe('Swarm Analysis - Performance Tests', () => {
  
  test('should complete analysis within reasonable time', async ({ page }) => {
    await page.goto(FRONTEND_URL);
    
    const startTime = Date.now();
    
    // Upload and analyze
    await page.click('button:has-text("Upload CSV")');
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_CSV_PATH);
    await page.click('button:has-text("Analyze with AI")');
    
    // Wait for results
    await page.waitForSelector('[data-testid="consensus-decisions"]', {
      timeout: ANALYSIS_TIMEOUT
    });
    
    const endTime = Date.now();
    const duration = (endTime - startTime) / 1000; // seconds
    
    console.log(`Analysis completed in ${duration.toFixed(1)} seconds`);
    
    // With sequential execution, expect 3-5 minutes
    // With parallel execution (future), expect 20-30 seconds
    expect(duration).toBeLessThan(360); // 6 minutes max
  });
});

