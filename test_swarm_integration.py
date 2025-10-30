"""
Comprehensive Swarm Integration Testing
Tests the complete flow: Data → Analysis → Recommendations
"""

import requests
import json
import csv
import time
from datetime import datetime
from typing import Dict, List, Any

# Configuration
BACKEND_URL = "http://localhost:8000"
POSITIONS_CSV = "data/examples/positions.csv"

# Test Results
results = {
    'tests': [],
    'performance': {},
    'start_time': datetime.now()
}

def log_result(test_name: str, passed: bool, details: str = "", duration: float = 0):
    """Log test result"""
    result = {
        'name': test_name,
        'passed': passed,
        'details': details,
        'duration': duration,
        'timestamp': datetime.now().isoformat()
    }
    results['tests'].append(result)
    
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {test_name}")
    if details:
        print(f"  {details}")
    if duration > 0:
        print(f"  Duration: {duration:.2f}s")

def load_real_portfolio():
    """Load real portfolio from CSV"""
    positions = []
    with open(POSITIONS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('Ticker') or 'FOOTNOTES' in row.get('Ticker', ''):
                continue
            
            ticker = row.get('Ticker', '').strip()
            try:
                quantity = float(row.get('Quantity', '0').replace(',', ''))
                value = float(row.get('Value', '0').replace(',', '').replace('$', ''))
            except:
                continue
            
            asset_type = 'option' if ('CALL' in ticker or 'PUT' in ticker) else 'stock'
            
            positions.append({
                'symbol': ticker,
                'asset_type': asset_type,
                'quantity': quantity,
                'market_value': value
            })
    
    total_value = sum(p['market_value'] for p in positions)
    return {'positions': positions, 'total_value': total_value}

print("\n" + "="*80)
print("COMPREHENSIVE SWARM INTEGRATION TESTING")
print("="*80)
print(f"\nStart Time: {results['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Backend: {BACKEND_URL}")

# ============================================================================
# TEST 1: Load Real Portfolio Data
# ============================================================================

print("\n" + "-"*80)
print("TEST 1: Load Real Portfolio Data")
print("-"*80)

start = time.time()
portfolio = load_real_portfolio()
duration = time.time() - start

log_result(
    "Load Portfolio from CSV",
    len(portfolio['positions']) > 0,
    f"Loaded {len(portfolio['positions'])} positions, Total: ${portfolio['total_value']:,.2f}",
    duration
)

print(f"\nPortfolio Summary:")
print(f"  Total Positions: {len(portfolio['positions'])}")
print(f"  Total Value: ${portfolio['total_value']:,.2f}")
print(f"  Options: {sum(1 for p in portfolio['positions'] if p['asset_type'] == 'option')}")
print(f"  Stocks: {sum(1 for p in portfolio['positions'] if p['asset_type'] == 'stock')}")

# ============================================================================
# TEST 2: Full Swarm Analysis with Real Data
# ============================================================================

print("\n" + "-"*80)
print("TEST 2: Full Swarm Analysis with Real Portfolio")
print("-"*80)

start = time.time()
try:
    response = requests.post(
        f"{BACKEND_URL}/api/swarm/analyze",
        json={
            'portfolio_data': portfolio,
            'market_data': {},
            'consensus_method': 'weighted'
        },
        timeout=60
    )
    duration = time.time() - start
    
    if response.status_code == 200:
        analysis = response.json()
        
        # Verify response structure
        has_analysis = 'analysis' in analysis
        has_recommendations = 'recommendations' in analysis
        has_metrics = 'metrics' in analysis
        
        all_present = has_analysis and has_recommendations and has_metrics
        
        log_result(
            "Swarm Analysis with Real Data",
            all_present,
            f"Response keys: {list(analysis.keys())}",
            duration
        )
        
        results['performance']['full_analysis'] = duration
        
        # Display analysis summary
        if all_present:
            print(f"\n📊 Analysis Summary:")
            print(f"  Swarm: {analysis.get('swarm_name')}")
            print(f"  Timestamp: {analysis.get('timestamp')}")
            
            # Agent participation
            agent_analyses = analysis['analysis'].get('analyses', {})
            print(f"\n  Agents Participated: {len(agent_analyses)}")
            for agent_id, agent_data in agent_analyses.items():
                print(f"    - {agent_id} ({agent_data.get('agent_type')})")
            
            # Consensus decisions
            consensus = analysis['recommendations'].get('consensus_recommendations', {})
            print(f"\n  Consensus Decisions: {len(consensus)}")
            for decision, data in consensus.items():
                print(f"    - {decision}: {data.get('result')} (confidence: {data.get('confidence', 0)*100:.1f}%)")
            
            # Metrics
            metrics = analysis.get('metrics', {})
            print(f"\n  Swarm Metrics:")
            swarm_metrics = metrics.get('swarm_metrics', {})
            print(f"    Total Decisions: {swarm_metrics.get('total_decisions', 0)}")
            print(f"    Total Errors: {swarm_metrics.get('total_errors', 0)}")
            
            consensus_metrics = metrics.get('consensus_metrics', {})
            print(f"    Consensus Success Rate: {consensus_metrics.get('success_rate', 0)*100:.1f}%")
    else:
        log_result(
            "Swarm Analysis with Real Data",
            False,
            f"HTTP {response.status_code}: {response.text[:200]}",
            duration
        )
except Exception as e:
    duration = time.time() - start
    log_result("Swarm Analysis with Real Data", False, str(e), duration)

# ============================================================================
# TEST 3: Agent-Specific Analysis Validation
# ============================================================================

print("\n" + "-"*80)
print("TEST 3: Agent-Specific Analysis Validation")
print("-"*80)

if response.status_code == 200:
    analysis = response.json()
    agent_analyses = analysis['analysis'].get('analyses', {})
    
    # Test Market Analyst
    market_analyst = next((v for k, v in agent_analyses.items() if 'market_analyst' in k), None)
    if market_analyst:
        ma_analysis = market_analyst.get('analysis', {})
        has_market_data = 'market_data' in ma_analysis
        has_trend = 'trend_analysis' in ma_analysis
        has_sectors = 'sector_analysis' in ma_analysis
        
        log_result(
            "Market Analyst Analysis",
            has_market_data and has_trend and has_sectors,
            f"Market data: {has_market_data}, Trend: {has_trend}, Sectors: {has_sectors}"
        )
        
        if has_market_data:
            print(f"\n  Market Data:")
            for index, data in ma_analysis['market_data'].items():
                price = data.get('current_price', 0) or 0
                change = data.get('change_pct_1m', 0) or 0
                print(f"    {index}: ${price:.2f} ({change:.2f}%)")
    
    # Test Risk Manager
    risk_manager = next((v for k, v in agent_analyses.items() if 'risk_manager' in k), None)
    if risk_manager:
        rm_analysis = risk_manager.get('analysis', {})
        has_greeks = 'greeks_analysis' in rm_analysis
        has_violations = 'limit_violations' in rm_analysis
        
        log_result(
            "Risk Manager Analysis",
            has_greeks and has_violations,
            f"Greeks: {has_greeks}, Violations: {has_violations}"
        )
        
        if has_greeks:
            greeks = rm_analysis['greeks_analysis']
            print(f"\n  Portfolio Greeks:")
            print(f"    Delta: {greeks.get('total_delta', 0):.2f}")
            print(f"    Gamma: {greeks.get('total_gamma', 0):.4f}")
            print(f"    Theta: {greeks.get('total_theta', 0):.2f}")
            print(f"    Vega: {greeks.get('total_vega', 0):.2f}")
        
        if has_violations:
            violations = rm_analysis['limit_violations']
            print(f"\n  Risk Violations: {len(violations)}")
            for violation in violations:
                print(f"    - {violation.get('type')}: {violation.get('symbol')} ({violation.get('severity')})")
    
    # Test Options Strategist
    options_strategist = next((v for k, v in agent_analyses.items() if 'options_strategist' in k), None)
    if options_strategist:
        os_analysis = options_strategist.get('analysis', {})
        has_strategies = 'recommended_strategies' in os_analysis
        
        log_result(
            "Options Strategist Analysis",
            has_strategies,
            f"Strategies: {has_strategies}"
        )
        
        if has_strategies:
            strategies = os_analysis['recommended_strategies']
            print(f"\n  Recommended Strategies: {', '.join(strategies)}")

# ============================================================================
# TEST 4: Consensus Mechanism Validation
# ============================================================================

print("\n" + "-"*80)
print("TEST 4: Consensus Mechanism Validation")
print("-"*80)

if response.status_code == 200:
    analysis = response.json()
    recommendations = analysis.get('recommendations', {})
    consensus = recommendations.get('consensus_recommendations', {})
    individual = recommendations.get('individual_recommendations', {})
    
    # Verify consensus was reached
    has_consensus = len(consensus) > 0
    has_individual = len(individual) > 0
    
    log_result(
        "Consensus Mechanism",
        has_consensus and has_individual,
        f"Consensus decisions: {len(consensus)}, Individual recommendations: {len(individual)}"
    )
    
    # Verify consensus metadata
    for decision_name, decision_data in consensus.items():
        has_result = 'result' in decision_data
        has_confidence = 'confidence' in decision_data
        has_metadata = 'metadata' in decision_data
        has_method = 'method' in decision_data
        
        log_result(
            f"Consensus Decision: {decision_name}",
            has_result and has_confidence and has_metadata and has_method,
            f"Result: {decision_data.get('result')}, Confidence: {decision_data.get('confidence', 0)*100:.1f}%"
        )

# ============================================================================
# TEST 5: Performance Benchmarks
# ============================================================================

print("\n" + "-"*80)
print("TEST 5: Performance Benchmarks")
print("-"*80)

# Test different portfolio sizes
sizes = [1, 5, 10]
for size in sizes:
    test_positions = portfolio['positions'][:size]
    test_portfolio = {
        'positions': test_positions,
        'total_value': sum(p['market_value'] for p in test_positions)
    }
    
    start = time.time()
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/swarm/analyze",
            json={
                'portfolio_data': test_portfolio,
                'market_data': {},
                'consensus_method': 'weighted'
            },
            timeout=60
        )
        duration = time.time() - start
        
        passed = response.status_code == 200
        log_result(
            f"Performance: {size} positions",
            passed,
            f"Response time: {duration:.2f}s",
            duration
        )
        
        results['performance'][f'size_{size}'] = duration
    except Exception as e:
        duration = time.time() - start
        log_result(f"Performance: {size} positions", False, str(e), duration)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

total_tests = len(results['tests'])
passed_tests = sum(1 for t in results['tests'] if t['passed'])
failed_tests = total_tests - passed_tests
pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

end_time = datetime.now()
total_duration = (end_time - results['start_time']).total_seconds()

print(f"\nTotal Tests: {total_tests}")
print(f"Passed: {passed_tests} ({pass_rate:.1f}%)")
print(f"Failed: {failed_tests}")
print(f"Total Duration: {total_duration:.2f}s")

if results['performance']:
    print(f"\n📊 Performance Metrics:")
    for metric, value in results['performance'].items():
        print(f"  {metric}: {value:.2f}s")

if failed_tests > 0:
    print(f"\n❌ Failed Tests:")
    for test in results['tests']:
        if not test['passed']:
            print(f"  - {test['name']}: {test['details']}")

# System Readiness
print(f"\n🎯 System Readiness:")
if pass_rate >= 95:
    print("  ✅ PRODUCTION READY")
elif pass_rate >= 80:
    print("  ⚠️  STAGING READY")
else:
    print("  ❌ NOT READY")

print("\n" + "="*80)

