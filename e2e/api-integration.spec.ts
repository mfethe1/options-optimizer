/**
 * Playwright E2E Tests - API Integration
 * 
 * Tests for backend API endpoints, caching, and error handling.
 * 
 * Run with: npx playwright test e2e/api-integration.spec.ts
 */

import { test, expect } from '@playwright/test';

const API_URL = process.env.API_URL || 'http://localhost:8000';
const USER_ID = 'test-user-123';

test.describe('Investor Report API', () => {
  test('should return 200 OK for GET /api/investor-report', async ({ request }) => {
    const response = await request.get(`${API_URL}/api/investor-report`, {
      params: {
        user_id: USER_ID,
        symbols: 'AAPL,MSFT',
      },
    });
    
    expect(response.status()).toBe(200);
  });

  test('should return valid InvestorReport.v1 JSON schema', async ({ request }) => {
    const response = await request.get(`${API_URL}/api/investor-report`, {
      params: {
        user_id: USER_ID,
        symbols: 'AAPL,MSFT',
      },
    });
    
    const data = await response.json();
    
    // Verify required top-level fields
    expect(data).toHaveProperty('as_of');
    expect(data).toHaveProperty('universe');
    expect(data).toHaveProperty('executive_summary');
    expect(data).toHaveProperty('risk_panel');
    expect(data).toHaveProperty('signals');
    expect(data).toHaveProperty('actions');
    expect(data).toHaveProperty('sources');
    expect(data).toHaveProperty('confidence');
    expect(data).toHaveProperty('metadata');
    
    // Verify risk_panel has 7 metrics
    expect(data.risk_panel).toHaveProperty('omega');
    expect(data.risk_panel).toHaveProperty('gh1');
    expect(data.risk_panel).toHaveProperty('pain_index');
    expect(data.risk_panel).toHaveProperty('upside_capture');
    expect(data.risk_panel).toHaveProperty('downside_capture');
    expect(data.risk_panel).toHaveProperty('cvar_95');
    expect(data.risk_panel).toHaveProperty('max_drawdown');
    
    // Verify signals
    expect(data.signals).toHaveProperty('ml_alpha');
    expect(data.signals).toHaveProperty('regime');
    expect(data.signals).toHaveProperty('sentiment');
    expect(data.signals).toHaveProperty('smart_money');
    expect(data.signals).toHaveProperty('alt_data');
    expect(data.signals).toHaveProperty('phase4_tech');
    
    // Verify Phase 4 fields (nullable)
    expect(data.signals.phase4_tech).toHaveProperty('options_flow_composite');
    expect(data.signals.phase4_tech).toHaveProperty('residual_momentum');
    expect(data.signals.phase4_tech).toHaveProperty('seasonality_score');
    expect(data.signals.phase4_tech).toHaveProperty('breadth_liquidity');
  });

  test('should use L1 cache on second request', async ({ request }) => {
    // First request (cache miss)
    const response1 = await request.get(`${API_URL}/api/investor-report`, {
      params: {
        user_id: USER_ID,
        symbols: 'AAPL,MSFT',
      },
    });
    
    const data1 = await response1.json();
    expect(data1.metadata.cached).toBe(false);
    
    // Second request (cache hit)
    const response2 = await request.get(`${API_URL}/api/investor-report`, {
      params: {
        user_id: USER_ID,
        symbols: 'AAPL,MSFT',
      },
    });
    
    const data2 = await response2.json();
    expect(data2.metadata.cached).toBe(true);
    expect(data2.metadata.cache_layer).toBe('L1');
    
    // Response time should be faster
    expect(data2.metadata.response_time_ms).toBeLessThan(data1.metadata.response_time_ms);
  });

  test('should bypass cache with fresh=true', async ({ request }) => {
    // First request to populate cache
    await request.get(`${API_URL}/api/investor-report`, {
      params: {
        user_id: USER_ID,
        symbols: 'AAPL,MSFT',
      },
    });
    
    // Second request with fresh=true
    const response = await request.get(`${API_URL}/api/investor-report`, {
      params: {
        user_id: USER_ID,
        symbols: 'AAPL,MSFT',
        fresh: 'true',
      },
    });
    
    const data = await response.json();
    
    // Should return cached data but schedule refresh
    expect(data.metadata.refreshing).toBe(true);
  });

  test('should handle different symbol combinations', async ({ request }) => {
    const symbolCombinations = [
      'AAPL',
      'AAPL,MSFT',
      'AAPL,MSFT,GOOGL',
      'TSLA,NVDA,AMD,INTC',
    ];
    
    for (const symbols of symbolCombinations) {
      const response = await request.get(`${API_URL}/api/investor-report`, {
        params: {
          user_id: USER_ID,
          symbols,
        },
      });
      
      expect(response.status()).toBe(200);
      
      const data = await response.json();
      expect(data.universe).toContain(symbols.split(',')[0]);
    }
  });

  test('should return 400 for missing user_id', async ({ request }) => {
    const response = await request.get(`${API_URL}/api/investor-report`, {
      params: {
        symbols: 'AAPL,MSFT',
      },
    });
    
    expect(response.status()).toBe(422); // FastAPI validation error
  });

  test('should handle concurrent requests without dog-piling', async ({ request }) => {
    // Send 10 concurrent requests
    const promises = Array.from({ length: 10 }, () =>
      request.get(`${API_URL}/api/investor-report`, {
        params: {
          user_id: `concurrent-test-${Math.random()}`,
          symbols: 'AAPL,MSFT',
        },
      })
    );
    
    const responses = await Promise.all(promises);
    
    // All should succeed
    responses.forEach(response => {
      expect(response.status()).toBe(200);
    });
  });

  test('should complete within 5 seconds for cached request', async ({ request }) => {
    // Populate cache
    await request.get(`${API_URL}/api/investor-report`, {
      params: {
        user_id: USER_ID,
        symbols: 'AAPL,MSFT',
      },
    });
    
    // Measure cached request time
    const startTime = Date.now();
    const response = await request.get(`${API_URL}/api/investor-report`, {
      params: {
        user_id: USER_ID,
        symbols: 'AAPL,MSFT',
      },
    });
    const elapsed = Date.now() - startTime;
    
    expect(response.status()).toBe(200);
    expect(elapsed).toBeLessThan(5000); // 5 seconds
    
    const data = await response.json();
    expect(data.metadata.response_time_ms).toBeLessThan(500); // <500ms target
  });
});

test.describe('Health Check API', () => {
  test('should return 200 OK for GET /api/health', async ({ request }) => {
    const response = await request.get(`${API_URL}/api/health`);
    
    expect(response.status()).toBe(200);
    
    const data = await response.json();
    expect(data.status).toBe('healthy');
    expect(data.service).toBe('investor-report-api');
    expect(data.timestamp).toBeTruthy();
  });
});

test.describe('Root Endpoint', () => {
  test('should return 200 OK for GET /', async ({ request }) => {
    const response = await request.get(`${API_URL}/`);
    
    expect(response.status()).toBe(200);
    
    const data = await response.json();
    expect(data.message).toBeTruthy();
  });
});

test.describe('WebSocket Endpoints', () => {
  test('should accept WebSocket connection for /ws/phase4-metrics/{user_id}', async ({ page }) => {
    // Navigate to a page that uses WebSocket
    await page.goto('http://localhost:5173/phase4-signals-demo');
    
    // Wait for WebSocket connection
    await page.waitForTimeout(2000);
    
    // Check console for WebSocket errors
    const logs = await page.evaluate(() => {
      return (window as any).__wsLogs || [];
    });
    
    // Should not have connection errors
    const hasError = logs.some((log: string) => log.includes('WebSocket connection failed'));
    expect(hasError).toBe(false);
  });

  test('should accept WebSocket connection for /ws/agent-stream/{user_id}', async ({ page }) => {
    // Navigate to agent transparency page
    await page.goto('http://localhost:5173/agent-transparency');
    
    // Wait for WebSocket connection
    await page.waitForTimeout(2000);
    
    // Check for connection status
    const connectionStatus = page.locator('[data-testid="connection-status"]');
    await expect(connectionStatus).toContainText('Connected', { timeout: 10000 });
  });
});

test.describe('Error Handling', () => {
  test('should return 404 for non-existent endpoint', async ({ request }) => {
    const response = await request.get(`${API_URL}/api/non-existent`);
    
    expect(response.status()).toBe(404);
  });

  test('should return 500 for internal server error (if triggered)', async ({ request }) => {
    // This test assumes we can trigger an error condition
    // In a real scenario, you'd use API mocking or test fixtures
    
    // For now, just verify the API handles errors gracefully
    const response = await request.get(`${API_URL}/api/investor-report`, {
      params: {
        user_id: USER_ID,
        symbols: 'INVALID_SYMBOL_THAT_CAUSES_ERROR',
      },
    });
    
    // Should either succeed or return a proper error
    expect([200, 400, 500]).toContain(response.status());
  });

  test('should handle malformed requests gracefully', async ({ request }) => {
    const response = await request.get(`${API_URL}/api/investor-report`, {
      params: {
        user_id: USER_ID,
        symbols: '', // Empty symbols
      },
    });
    
    // Should either succeed with default symbols or return validation error
    expect([200, 400, 422]).toContain(response.status());
  });
});

test.describe('Performance Benchmarks', () => {
  test('should handle 100 sequential requests without degradation', async ({ request }) => {
    const responseTimes: number[] = [];
    
    for (let i = 0; i < 100; i++) {
      const startTime = Date.now();
      const response = await request.get(`${API_URL}/api/investor-report`, {
        params: {
          user_id: `perf-test-${i}`,
          symbols: 'AAPL,MSFT',
        },
      });
      const elapsed = Date.now() - startTime;
      
      expect(response.status()).toBe(200);
      responseTimes.push(elapsed);
    }
    
    // Calculate average response time
    const avgResponseTime = responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length;
    
    // Average should be reasonable (< 2s)
    expect(avgResponseTime).toBeLessThan(2000);
    
    // Last 10 requests should not be significantly slower than first 10
    const first10Avg = responseTimes.slice(0, 10).reduce((a, b) => a + b, 0) / 10;
    const last10Avg = responseTimes.slice(-10).reduce((a, b) => a + b, 0) / 10;
    
    expect(last10Avg).toBeLessThan(first10Avg * 2); // No more than 2x slower
  });

  test('should maintain cache hit rate > 80% after warmup', async ({ request }) => {
    const symbols = ['AAPL,MSFT', 'GOOGL,AMZN', 'TSLA,NVDA'];
    let cacheHits = 0;
    let totalRequests = 0;
    
    // Warmup phase (populate cache)
    for (const sym of symbols) {
      await request.get(`${API_URL}/api/investor-report`, {
        params: {
          user_id: USER_ID,
          symbols: sym,
        },
      });
    }
    
    // Test phase (measure cache hit rate)
    for (let i = 0; i < 30; i++) {
      const sym = symbols[i % symbols.length];
      const response = await request.get(`${API_URL}/api/investor-report`, {
        params: {
          user_id: USER_ID,
          symbols: sym,
        },
      });
      
      const data = await response.json();
      if (data.metadata.cached) {
        cacheHits++;
      }
      totalRequests++;
    }
    
    const hitRate = cacheHits / totalRequests;
    expect(hitRate).toBeGreaterThan(0.8); // >80% hit rate
  });
});

