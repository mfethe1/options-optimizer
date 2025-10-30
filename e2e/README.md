# Playwright E2E Testing Suite

World-class end-to-end testing for institutional-grade options analysis platform.

## Overview

This E2E testing suite provides comprehensive coverage of:
- **Agent Transparency System**: Real-time LLM agent event streaming
- **Phase 4 Signals Panel**: Technical signals with WebSocket updates
- **Risk Panel Dashboard**: 7 institutional-grade risk metrics
- **API Integration**: Backend endpoints, caching, error handling
- **Performance**: Page load times, WebSocket latency, memory leaks

## Test Files

### 1. `agent-transparency.spec.ts` (10 tests)
Tests for real-time agent transparency system:
- WebSocket connection and heartbeat
- Agent event streaming (STARTED, THINKING, TOOL_CALL, TOOL_RESULT, PROGRESS, COMPLETED)
- Progress bar updates
- Conversation display with auto-scroll
- Search and filter functionality
- Export conversation log
- WebSocket reconnection
- Performance (page load <2s, 60fps animations, no memory leaks)

### 2. `phase4-signals.spec.ts` (12 tests)
Tests for Phase 4 technical signals panel:
- Render all 4 signal cards
- Display signal values with correct formatting
- Color-coded indicators based on signal strength
- Tooltips on hover
- Real-time WebSocket updates
- Loading states
- Missing data handling
- Trend arrows for signal changes
- WebSocket connection maintenance (60s)
- Responsive design (mobile, tablet, desktop)

### 3. `risk-panel.spec.ts` (15 tests)
Tests for risk metrics dashboard:
- Render all 7 risk metrics
- Regime indicator with color coding
- Metric values with correct formatting
- Color-coded risk levels (Low, Medium, High, Critical)
- Metric descriptions on hover
- Real-time updates
- Loading states
- Missing data handling
- 2x4 grid layout
- Responsive design (mobile, tablet, desktop)
- Accessibility (ARIA labels, keyboard navigation, color contrast)

### 4. `api-integration.spec.ts` (18 tests)
Tests for backend API integration:
- GET /api/investor-report endpoint
- InvestorReport.v1 JSON schema validation
- L1/L2 caching behavior
- Cache bypass with fresh=true
- Different symbol combinations
- Missing user_id validation
- Concurrent requests without dog-piling
- Response time <5s for cached requests
- Health check endpoint
- Root endpoint
- WebSocket endpoints
- Error handling (404, 500, malformed requests)
- Performance benchmarks (100 sequential requests, cache hit rate >80%)

## Installation

```bash
# Install Playwright
npm install -D @playwright/test

# Install browsers
npx playwright install
```

## Running Tests

### Run all tests
```bash
npx playwright test
```

### Run specific test file
```bash
npx playwright test e2e/agent-transparency.spec.ts
npx playwright test e2e/phase4-signals.spec.ts
npx playwright test e2e/risk-panel.spec.ts
npx playwright test e2e/api-integration.spec.ts
```

### Run tests in headed mode (see browser)
```bash
npx playwright test --headed
```

### Run tests in debug mode
```bash
npx playwright test --debug
```

### Run tests in specific browser
```bash
npx playwright test --project=chromium
npx playwright test --project=firefox
npx playwright test --project=webkit
```

### Run tests in parallel
```bash
npx playwright test --workers=4
```

### Run tests with UI mode (interactive)
```bash
npx playwright test --ui
```

## Viewing Reports

### HTML Report
```bash
npx playwright show-report
```

### Trace Viewer (for failed tests)
```bash
npx playwright show-trace test-results/path-to-trace.zip
```

## Configuration

See `playwright.config.ts` for configuration options:
- **Parallel execution**: Tests run in parallel for speed
- **Multiple browsers**: Chromium, Firefox, WebKit, Edge, Chrome
- **Mobile/tablet viewports**: Pixel 5, iPhone 12, iPad Pro
- **Video recording**: On failure only
- **Screenshots**: On failure only
- **Trace collection**: On first retry
- **Retry logic**: 2 retries on CI, 1 retry locally
- **Global timeout**: 60s per test
- **Expect timeout**: 10s per assertion

## Environment Variables

Create a `.env` file in the root directory:

```env
BASE_URL=http://localhost:5173
API_URL=http://localhost:8000
USER_ID=test-user-123
```

## CI/CD Integration

### GitHub Actions
```yaml
name: Playwright Tests
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
jobs:
  test:
    timeout-minutes: 60
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-node@v3
      with:
        node-version: 18
    - name: Install dependencies
      run: npm ci
    - name: Install Playwright Browsers
      run: npx playwright install --with-deps
    - name: Run Playwright tests
      run: npx playwright test
    - uses: actions/upload-artifact@v3
      if: always()
      with:
        name: playwright-report
        path: playwright-report/
        retention-days: 30
```

## Best Practices

### 1. Use data-testid attributes
```tsx
<div data-testid="risk-panel-dashboard">
  <div data-testid="risk-metric-card">
    <span data-testid="metric-value">1.45</span>
  </div>
</div>
```

### 2. Wait for elements properly
```typescript
// Good
await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();

// Bad
await page.waitForTimeout(5000); // Avoid arbitrary waits
```

### 3. Use page object model for complex pages
```typescript
class RiskPanelPage {
  constructor(private page: Page) {}
  
  async goto() {
    await this.page.goto('/risk-panel-demo');
  }
  
  async getMetricValue(metricName: string) {
    const card = this.page.locator(`[data-testid="risk-metric-card"]:has-text("${metricName}")`);
    return await card.locator('[data-testid="metric-value"]').textContent();
  }
}
```

### 4. Clean up after tests
```typescript
test.afterEach(async ({ page }) => {
  // Close WebSocket connections
  await page.evaluate(() => {
    (window as any).__ws?.close();
  });
});
```

### 5. Use fixtures for common setup
```typescript
import { test as base } from '@playwright/test';

const test = base.extend({
  authenticatedPage: async ({ page }, use) => {
    await page.goto('/login');
    await page.fill('[name="username"]', 'test-user');
    await page.fill('[name="password"]', 'test-pass');
    await page.click('button[type="submit"]');
    await use(page);
  },
});
```

## Debugging Tips

### 1. Use Playwright Inspector
```bash
npx playwright test --debug
```

### 2. Add console.log in tests
```typescript
test('my test', async ({ page }) => {
  console.log('Current URL:', page.url());
  const value = await page.locator('[data-testid="value"]').textContent();
  console.log('Value:', value);
});
```

### 3. Take screenshots manually
```typescript
await page.screenshot({ path: 'debug-screenshot.png' });
```

### 4. Use trace viewer
```bash
npx playwright show-trace test-results/path-to-trace.zip
```

### 5. Check browser console
```typescript
page.on('console', msg => console.log('Browser console:', msg.text()));
```

## Performance Targets

- **Page load time**: <2s
- **WebSocket latency**: <100ms
- **API response time (cached)**: <500ms
- **API response time (uncached)**: <5s
- **Frame rate**: ≥55fps (60fps target with 5fps tolerance)
- **Memory increase**: <10MB after 100 events
- **Cache hit rate**: >80% after warmup

## Coverage Goals

- **Unit tests**: 100% (Vitest)
- **Integration tests**: 100% (pytest)
- **E2E tests**: 100% critical paths (Playwright)

## Troubleshooting

### Tests fail with "WebSocket connection refused"
- Ensure backend server is running: `uvicorn src.api.main:app --reload`
- Check WebSocket endpoint is accessible: `ws://localhost:8000/ws/agent-stream/test-user`

### Tests fail with "Page not found"
- Ensure frontend dev server is running: `cd frontend && npm run dev`
- Check frontend is accessible: `http://localhost:5173`

### Tests timeout
- Increase timeout in `playwright.config.ts`: `timeout: 120000`
- Check for slow network or CPU-intensive operations

### Screenshots/videos not saved
- Check `playwright.config.ts` settings: `screenshot: 'only-on-failure'`, `video: 'retain-on-failure'`
- Verify `test-results/` directory exists and is writable

## Contributing

1. Write tests for new features
2. Ensure all tests pass before committing
3. Follow naming conventions: `feature-name.spec.ts`
4. Add documentation for complex test scenarios
5. Use descriptive test names: `should display error message when API fails`

## License

MIT

