// K6 Load Testing Script for PINN Model Performance
// Tests latency improvements from ~1350ms to ~260ms target
// Realistic trading patterns and user behavior simulation

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const pinnLatency = new Trend('pinn_latency_ms');
const pinnErrors = new Rate('pinn_error_rate');
const cacheHits = new Counter('pinn_cache_hits');
const cacheMisses = new Counter('pinn_cache_misses');

// Test configuration
export const options = {
  stages: [
    // Ramp-up: gradually increase load
    { duration: '2m', target: 10 },   // Ramp up to 10 users over 2 minutes
    { duration: '3m', target: 25 },   // Ramp up to 25 users over 3 minutes
    { duration: '5m', target: 50 },   // Ramp up to 50 users over 5 minutes
    { duration: '10m', target: 50 },  // Stay at 50 users for 10 minutes
    { duration: '3m', target: 75 },   // Spike to 75 users
    { duration: '2m', target: 75 },   // Hold spike
    { duration: '3m', target: 25 },   // Ramp down to 25 users
    { duration: '2m', target: 0 },    // Ramp down to 0 users
  ],
  thresholds: {
    // Performance thresholds
    'pinn_latency_ms': ['p(95)<260'],           // 95% of requests < 260ms
    'pinn_error_rate': ['rate<0.05'],           // Error rate < 5%
    'http_req_duration': ['p(90)<300'],         // 90% of requests < 300ms
    'http_req_failed': ['rate<0.02'],           // HTTP failure rate < 2%
    'http_reqs': ['rate>10'],                   // At least 10 requests/second
  },
};

// Base URL from environment or default
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8001';

// Common stock symbols for realistic testing
const SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX'];
const PRICE_RANGES = {
  'AAPL': [140, 180],
  'GOOGL': [2600, 3000],
  'MSFT': [280, 320],
  'TSLA': [180, 250],
  'AMZN': [3000, 3400],
  'NVDA': [400, 500],
  'META': [280, 320],
  'NFLX': [380, 450]
};

// Helper function to get random symbol and price
function getRandomSymbolAndPrice() {
  const symbol = SYMBOLS[Math.floor(Math.random() * SYMBOLS.length)];
  const priceRange = PRICE_RANGES[symbol];
  const price = Math.random() * (priceRange[1] - priceRange[0]) + priceRange[0];
  return { symbol, price: price.toFixed(2) };
}

// Setup function - runs once per VU
export function setup() {
  // Warmup PINN cache
  console.log('Warming up PINN cache...');
  const warmupResponse = http.post(`${BASE_URL}/health/pinn/warmup`);
  check(warmupResponse, {
    'cache warmup successful': (r) => r.status === 200,
  });
  
  // Verify PINN health
  const healthResponse = http.get(`${BASE_URL}/health/pinn`);
  const healthData = JSON.parse(healthResponse.body);
  
  console.log(`PINN Health: ${healthData.overall_status}`);
  console.log(`Cache Size: ${healthData.details.cache_stats.currsize}`);
  
  return {
    baseUrl: BASE_URL,
    pinnHealthy: healthData.overall_status === 'healthy'
  };
}

// Main test function
export default function(data) {
  if (!data.pinnHealthy) {
    console.error('PINN not healthy, skipping test');
    return;
  }
  
  // Simulate realistic user behavior patterns
  const userBehavior = Math.random();
  
  if (userBehavior < 0.6) {
    // 60% - Single stock analysis (most common)
    testSingleStockAnalysis(data.baseUrl);
  } else if (userBehavior < 0.85) {
    // 25% - Multiple stock comparison
    testMultipleStockComparison(data.baseUrl);
  } else {
    // 15% - Rapid-fire analysis (day trader behavior)
    testRapidFireAnalysis(data.baseUrl);
  }
  
  // Random think time between requests (1-5 seconds)
  sleep(Math.random() * 4 + 1);
}

// Test single stock PINN analysis
function testSingleStockAnalysis(baseUrl) {
  const { symbol, price } = getRandomSymbolAndPrice();
  const startTime = Date.now();
  
  const response = http.get(`${baseUrl}/api/pinn/predict?symbol=${symbol}&current_price=${price}`, {
    tags: { test_type: 'single_stock' },
  });
  
  const latency = Date.now() - startTime;
  pinnLatency.add(latency);
  
  const success = check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 260ms': () => latency < 260,
    'has prediction data': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.price !== undefined && data.confidence !== undefined;
      } catch {
        return false;
      }
    },
  });
  
  if (!success) {
    pinnErrors.add(1);
  }
  
  // Check if cache was used (heuristic: very fast response likely cache hit)
  if (latency < 50) {
    cacheHits.add(1);
  } else {
    cacheMisses.add(1);
  }
}

// Test multiple stock comparison
function testMultipleStockComparison(baseUrl) {
  const numStocks = Math.floor(Math.random() * 3) + 2; // 2-4 stocks
  
  for (let i = 0; i < numStocks; i++) {
    const { symbol, price } = getRandomSymbolAndPrice();
    const startTime = Date.now();
    
    const response = http.get(`${baseUrl}/api/pinn/predict?symbol=${symbol}&current_price=${price}`, {
      tags: { test_type: 'multi_stock' },
    });
    
    const latency = Date.now() - startTime;
    pinnLatency.add(latency);
    
    check(response, {
      'multi-stock status is 200': (r) => r.status === 200,
      'multi-stock response time < 260ms': () => latency < 260,
    });
    
    // Brief pause between requests in comparison
    sleep(0.5);
  }
}

// Test rapid-fire analysis (day trader simulation)
function testRapidFireAnalysis(baseUrl) {
  const numRequests = Math.floor(Math.random() * 5) + 3; // 3-7 rapid requests
  
  for (let i = 0; i < numRequests; i++) {
    const { symbol, price } = getRandomSymbolAndPrice();
    const startTime = Date.now();
    
    const response = http.get(`${baseUrl}/api/pinn/predict?symbol=${symbol}&current_price=${price}`, {
      tags: { test_type: 'rapid_fire' },
    });
    
    const latency = Date.now() - startTime;
    pinnLatency.add(latency);
    
    check(response, {
      'rapid-fire status is 200': (r) => r.status === 200,
      'rapid-fire response time < 260ms': () => latency < 260,
    });
    
    // Very short pause for rapid-fire
    sleep(0.1);
  }
}

// Teardown function
export function teardown(data) {
  console.log('Load test completed');
  
  // Get final health check
  const finalHealth = http.get(`${data.baseUrl}/health/pinn`);
  const healthData = JSON.parse(finalHealth.body);
  
  console.log(`Final PINN Health: ${healthData.overall_status}`);
  console.log(`Final Cache Hit Rate: ${healthData.details.cache_stats.hit_rate}`);
}
