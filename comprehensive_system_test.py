"""
Comprehensive System Testing Suite
Tests all components of the recommendation engine and API
"""
import sys
import time
import traceback
import requests
from datetime import datetime
from typing import Dict, List, Any

# Test configuration
TEST_SYMBOLS = ['NVDA', 'AAPL', 'TSLA', 'AMD', 'MSFT', 'GOOGL', 'META', 'AMZN']
API_BASE = 'http://localhost:8000'

class SystemTester:
    def __init__(self):
        self.results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        self.start_time = None
        self.end_time = None
    
    def run_all_tests(self):
        """Run all test suites"""
        print("\n" + "="*80)
        print("COMPREHENSIVE SYSTEM TESTING")
        print("="*80)
        
        self.start_time = time.time()
        
        # Test suites
        self.test_health_check()
        self.test_positions_endpoint()
        self.test_recommendation_engine()
        self.test_multiple_symbols()
        self.test_edge_cases()
        self.test_performance()
        
        self.end_time = time.time()
        
        # Print summary
        self.print_summary()
    
    def test_health_check(self):
        """Test 1: Health Check"""
        print("\n" + "-"*80)
        print("TEST 1: Health Check")
        print("-"*80)
        
        try:
            response = requests.get(f'{API_BASE}/health')
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            data = response.json()
            assert data['status'] == 'healthy', "API not healthy"
            
            print("✓ Health check passed")
            print(f"  Status: {data['status']}")
            print(f"  Timestamp: {data['timestamp']}")
            self.results['passed'] += 1
            
        except Exception as e:
            print(f"✗ Health check failed: {e}")
            self.results['failed'] += 1
            self.results['errors'].append(('Health Check', str(e)))
    
    def test_positions_endpoint(self):
        """Test 2: Positions Endpoint"""
        print("\n" + "-"*80)
        print("TEST 2: Positions Endpoint")
        print("-"*80)
        
        try:
            response = requests.get(f'{API_BASE}/api/positions')
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            data = response.json()
            
            print(f"✓ Positions endpoint passed")
            print(f"  Stock positions: {len(data.get('stocks', []))}")
            print(f"  Option positions: {len(data.get('options', []))}")
            
            # Validate structure
            if data.get('stocks'):
                stock = data['stocks'][0]
                required_fields = ['symbol', 'quantity', 'entry_price', 'current_price']
                for field in required_fields:
                    assert field in stock, f"Missing field: {field}"
            
            self.results['passed'] += 1
            
        except Exception as e:
            print(f"✗ Positions endpoint failed: {e}")
            self.results['failed'] += 1
            self.results['errors'].append(('Positions Endpoint', str(e)))
    
    def test_recommendation_engine(self):
        """Test 3: Recommendation Engine (NVDA)"""
        print("\n" + "-"*80)
        print("TEST 3: Recommendation Engine (NVDA)")
        print("-"*80)
        
        try:
            response = requests.get(f'{API_BASE}/api/recommendations/NVDA')
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            data = response.json()
            
            # Validate structure
            required_fields = [
                'symbol', 'recommendation', 'confidence', 'combined_score',
                'scores', 'actions', 'reasoning', 'risk_factors', 'catalysts'
            ]
            for field in required_fields:
                assert field in data, f"Missing field: {field}"
            
            # Validate scores
            required_scores = ['technical', 'fundamental', 'sentiment', 'risk', 'earnings', 'correlation']
            for score_name in required_scores:
                assert score_name in data['scores'], f"Missing score: {score_name}"
                score_data = data['scores'][score_name]
                assert 'score' in score_data, f"Missing score value for {score_name}"
                assert 0 <= score_data['score'] <= 100, f"Score out of range: {score_data['score']}"
            
            # Validate recommendation
            valid_recommendations = [
                'STRONG_BUY', 'BUY', 'WATCH', 'AVOID',
                'STRONG_HOLD_ADD', 'HOLD', 'HOLD_TRIM', 'REDUCE', 'CLOSE'
            ]
            assert data['recommendation'] in valid_recommendations, f"Invalid recommendation: {data['recommendation']}"
            
            # Validate confidence
            assert 0 <= data['confidence'] <= 100, f"Confidence out of range: {data['confidence']}"
            
            print("✓ Recommendation engine passed")
            print(f"  Symbol: {data['symbol']}")
            print(f"  Recommendation: {data['recommendation']}")
            print(f"  Confidence: {data['confidence']:.1f}%")
            print(f"  Combined Score: {data['combined_score']:.1f}/100")
            print(f"  Actions: {len(data['actions'])}")
            
            self.results['passed'] += 1
            
        except Exception as e:
            print(f"✗ Recommendation engine failed: {e}")
            traceback.print_exc()
            self.results['failed'] += 1
            self.results['errors'].append(('Recommendation Engine', str(e)))
    
    def test_multiple_symbols(self):
        """Test 4: Multiple Symbols"""
        print("\n" + "-"*80)
        print("TEST 4: Multiple Symbols")
        print("-"*80)
        
        passed = 0
        failed = 0
        
        for symbol in TEST_SYMBOLS:
            try:
                response = requests.get(f'{API_BASE}/api/recommendations/{symbol}', timeout=30)
                assert response.status_code == 200, f"Expected 200, got {response.status_code}"
                data = response.json()
                
                # Basic validation
                assert 'recommendation' in data
                assert 'combined_score' in data
                
                print(f"  ✓ {symbol}: {data['recommendation']} ({data['combined_score']:.1f}/100)")
                passed += 1
                
            except Exception as e:
                print(f"  ✗ {symbol}: {e}")
                failed += 1
        
        print(f"\n  Results: {passed} passed, {failed} failed")
        
        if failed == 0:
            self.results['passed'] += 1
        else:
            self.results['failed'] += 1
            self.results['errors'].append(('Multiple Symbols', f'{failed} symbols failed'))
    
    def test_edge_cases(self):
        """Test 5: Edge Cases"""
        print("\n" + "-"*80)
        print("TEST 5: Edge Cases")
        print("-"*80)
        
        test_cases = [
            ('Invalid Symbol', 'INVALID123'),
            ('Lowercase Symbol', 'nvda'),
            ('Symbol with Dot', 'BRK.B'),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, symbol in test_cases:
            try:
                response = requests.get(f'{API_BASE}/api/recommendations/{symbol}', timeout=30)
                
                # Should either succeed or fail gracefully
                if response.status_code == 200:
                    data = response.json()
                    assert 'recommendation' in data or 'detail' in data
                    print(f"  ✓ {test_name} ({symbol}): Handled gracefully")
                    passed += 1
                elif response.status_code in [404, 500]:
                    # Acceptable error responses
                    print(f"  ✓ {test_name} ({symbol}): Returned {response.status_code}")
                    passed += 1
                else:
                    print(f"  ✗ {test_name} ({symbol}): Unexpected status {response.status_code}")
                    failed += 1
                    
            except Exception as e:
                print(f"  ✗ {test_name} ({symbol}): {e}")
                failed += 1
        
        print(f"\n  Results: {passed} passed, {failed} failed")
        
        if failed == 0:
            self.results['passed'] += 1
        else:
            self.results['failed'] += 1
    
    def test_performance(self):
        """Test 6: Performance"""
        print("\n" + "-"*80)
        print("TEST 6: Performance")
        print("-"*80)
        
        try:
            # Test response time for 5 symbols
            test_symbols = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'TSLA']
            times = []
            
            for symbol in test_symbols:
                start = time.time()
                response = requests.get(f'{API_BASE}/api/recommendations/{symbol}', timeout=30)
                end = time.time()
                
                elapsed = end - start
                times.append(elapsed)
                
                status = "✓" if elapsed < 5.0 else "⚠"
                print(f"  {status} {symbol}: {elapsed:.2f}s")
            
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            
            print(f"\n  Average: {avg_time:.2f}s")
            print(f"  Min: {min_time:.2f}s")
            print(f"  Max: {max_time:.2f}s")
            
            # Pass if average < 5 seconds
            if avg_time < 5.0:
                print(f"  ✓ Performance acceptable (avg < 5s)")
                self.results['passed'] += 1
            else:
                print(f"  ⚠ Performance slow (avg >= 5s)")
                self.results['failed'] += 1
                self.results['errors'].append(('Performance', f'Average time {avg_time:.2f}s'))
                
        except Exception as e:
            print(f"✗ Performance test failed: {e}")
            self.results['failed'] += 1
            self.results['errors'].append(('Performance', str(e)))
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        total = self.results['passed'] + self.results['failed']
        elapsed = self.end_time - self.start_time
        
        print(f"\nTotal Tests: {total}")
        print(f"Passed: {self.results['passed']} ✓")
        print(f"Failed: {self.results['failed']} ✗")
        print(f"Success Rate: {(self.results['passed']/total*100):.1f}%")
        print(f"Total Time: {elapsed:.2f}s")
        
        if self.results['errors']:
            print("\n" + "-"*80)
            print("ERRORS")
            print("-"*80)
            for test_name, error in self.results['errors']:
                print(f"\n{test_name}:")
                print(f"  {error}")
        
        print("\n" + "="*80)
        
        if self.results['failed'] == 0:
            print("✓ ALL TESTS PASSED!")
        else:
            print(f"✗ {self.results['failed']} TEST(S) FAILED")
        
        print("="*80 + "\n")

if __name__ == "__main__":
    tester = SystemTester()
    tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if tester.results['failed'] == 0 else 1)

