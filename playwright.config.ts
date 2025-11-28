/**
 * Playwright Configuration
 * 
 * World-class E2E testing configuration for institutional-grade options analysis platform.
 * 
 * Features:
 * - Parallel execution for speed
 * - Multiple browsers (Chromium, Firefox, WebKit)
 * - Mobile and tablet viewports
 * - Video recording on failure
 * - Screenshots on failure
 * - Trace collection for debugging
 * - Retry on failure
 * - Global setup/teardown
 */

import { defineConfig, devices } from '@playwright/test';

/**
 * Read environment variables from file.
 * https://github.com/motdotla/dotenv
 */
// require('dotenv').config();

/**
 * See https://playwright.dev/docs/test-configuration.
 */
export default defineConfig({
  testDir: './e2e',
  
  /* Run tests in files in parallel */
  fullyParallel: true,
  
  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: !!process.env.CI,
  
  /* Retry on CI only */
  retries: process.env.CI ? 2 : 1,
  
  /* Opt out of parallel tests on CI. */
  workers: process.env.CI ? 1 : undefined,
  
  /* Reporter to use. See https://playwright.dev/docs/test-reporters */
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['json', { outputFile: 'playwright-report/results.json' }],
    ['junit', { outputFile: 'playwright-report/results.xml' }],
    ['list'],
  ],
  
  /* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
  use: {
    /* Base URL to use in actions like `await page.goto('/')`. */
    baseURL: process.env.BASE_URL || 'http://localhost:3010',
    
    /* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
    trace: 'on-first-retry',
    
    /* Screenshot on failure */
    screenshot: 'only-on-failure',
    
    /* Video on failure */
    video: 'retain-on-failure',
    
    /* Maximum time each action such as `click()` can take. Defaults to 0 (no limit). */
    actionTimeout: 10000,
    
    /* Maximum time each navigation can take. Defaults to 0 (no limit). */
    navigationTimeout: 30000,
  },

  /* Configure projects for major browsers */
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },

    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },

    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },

    /* Test against mobile viewports. */
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] },
    },
    {
      name: 'Mobile Safari',
      use: { ...devices['iPhone 12'] },
    },

    /* Test against branded browsers. */
    {
      name: 'Microsoft Edge',
      use: { ...devices['Desktop Edge'], channel: 'msedge' },
    },
    {
      name: 'Google Chrome',
      use: { ...devices['Desktop Chrome'], channel: 'chrome' },
    },
    
    /* Test against tablet viewports. */
    {
      name: 'iPad',
      use: { ...devices['iPad Pro'] },
    },
    {
      name: 'iPad Landscape',
      use: { ...devices['iPad Pro landscape'] },
    },
  ],

  /* Run your local dev server before starting the tests */
  webServer: [
    {
      command: 'cd frontend && npm run dev -- --port 3010',
      url: 'http://localhost:3010',
      reuseExistingServer: false,
      timeout: 120000,
      stdout: 'ignore',
      stderr: 'pipe',
      env: {
        VITE_API_URL: 'http://127.0.0.1:8017'
      }
    },
    {
      // Dedicated backend for ML tests; do not reuse to avoid port collisions with other services
      command: 'py -m uvicorn src.api.main:app --host 127.0.0.1 --port 8017',
      url: 'http://127.0.0.1:8017/health',
      reuseExistingServer: true,
      timeout: 120000,
      stdout: 'pipe',
      stderr: 'pipe',
    },
  ],
  
  /* Global timeout for each test */
  timeout: 60000,
  
  /* Expect timeout */
  expect: {
    timeout: 10000,
  },
  
  /* Output folder for test artifacts */
  outputDir: 'test-results/',
  
  /* Folder for test artifacts such as screenshots, videos, traces, etc. */
  snapshotDir: 'e2e/__snapshots__',
});

