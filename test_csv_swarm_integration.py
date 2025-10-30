"""
Test CSV Upload + Swarm Analysis Integration

This script tests the complete flow:
1. Upload CSV file
2. Import positions
3. Run LLM-powered swarm analysis
4. Return AI-generated recommendations
"""

import requests
import json
from datetime import datetime
import os

API_BASE = "http://localhost:8000"

print("\n" + "="*80)
print("CSV UPLOAD + SWARM ANALYSIS INTEGRATION TEST")
print("="*80)
print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n")

# ============================================================================
# Test 1: Check if API is running
# ============================================================================
print("-" * 80)
print("TEST 1: API Health Check")
print("-" * 80)

try:
    response = requests.get(f"{API_BASE}/health", timeout=5)
    if response.status_code == 200:
        print("✓ API is running")
    else:
        print(f"✗ API returned status {response.status_code}")
        exit(1)
except Exception as e:
    print(f"✗ API is not running: {e}")
    print("\nPlease start the API server:")
    print("  python -m uvicorn src.api.main:app --reload")
    exit(1)

print()

# ============================================================================
# Test 2: Upload CSV and Run Swarm Analysis
# ============================================================================
print("-" * 80)
print("TEST 2: CSV Upload + Swarm Analysis")
print("-" * 80)

# Check if example CSV exists
csv_file = "data/examples/positions.csv"
if not os.path.exists(csv_file):
    print(f"✗ CSV file not found: {csv_file}")
    print("\nPlease create a CSV file with your positions")
    exit(1)

print(f"✓ Found CSV file: {csv_file}")
print()

# Upload CSV and run swarm analysis
print("Uploading CSV and running swarm analysis...")
print("(This may take 1-3 minutes depending on portfolio size)")
print()

try:
    with open(csv_file, 'rb') as f:
        files = {'file': ('positions.csv', f, 'text/csv')}
        params = {
            'chase_format': 'true',  # This is a Chase.com export
            'consensus_method': 'weighted'
        }
        
        response = requests.post(
            f"{API_BASE}/api/swarm/analyze-csv",
            files=files,
            params=params,
            timeout=300  # 5 minutes timeout
        )
    
    if response.status_code == 200:
        result = response.json()
        
        print("✓ Swarm analysis complete!")
        print()
        
        # Display results
        print("="*80)
        print("ANALYSIS RESULTS")
        print("="*80)
        print()
        
        # Import stats
        if 'import_stats' in result:
            stats = result['import_stats']
            print(f"Positions Imported: {stats.get('positions_imported', 0)}")
            print(f"Positions Failed: {stats.get('positions_failed', 0)}")
            if stats.get('chase_conversion'):
                print(f"Chase Conversion: {stats['chase_conversion']}")
            print()
        
        # Portfolio summary
        if 'portfolio_summary' in result:
            summary = result['portfolio_summary']
            print("Portfolio Summary:")
            print(f"  Total Value: ${summary.get('total_value', 0):,.2f}")
            print(f"  Unrealized P&L: ${summary.get('total_unrealized_pnl', 0):,.2f}")
            print(f"  P&L %: {summary.get('total_unrealized_pnl_pct', 0):.2f}%")
            print(f"  Positions: {summary.get('positions_count', 0)}")
            print()
        
        # Consensus decisions
        if 'consensus_decisions' in result:
            decisions = result['consensus_decisions']
            
            print("AI Consensus Recommendations:")
            print()
            
            # Overall action
            if 'overall_action' in decisions:
                action = decisions['overall_action']
                print(f"  Overall Action: {action.get('choice', 'N/A').upper()}")
                print(f"    Confidence: {action.get('confidence', 0)*100:.0f}%")
                print(f"    Reasoning: {action.get('reasoning', 'N/A')}")
                print()
            
            # Risk level
            if 'risk_level' in decisions:
                risk = decisions['risk_level']
                print(f"  Risk Level: {risk.get('choice', 'N/A').upper()}")
                print(f"    Confidence: {risk.get('confidence', 0)*100:.0f}%")
                print(f"    Reasoning: {risk.get('reasoning', 'N/A')}")
                print()
            
            # Market outlook
            if 'market_outlook' in decisions:
                outlook = decisions['market_outlook']
                print(f"  Market Outlook: {outlook.get('choice', 'N/A').upper()}")
                print(f"    Confidence: {outlook.get('confidence', 0)*100:.0f}%")
                print(f"    Reasoning: {outlook.get('reasoning', 'N/A')}")
                print()
        
        # Agent analyses
        if 'agent_analyses' in result:
            print("Agent Analyses:")
            for agent_id, analysis in result['agent_analyses'].items():
                print(f"  - {agent_id}")
            print()
        
        # Execution time
        if 'execution_time' in result:
            print(f"Execution Time: {result['execution_time']:.2f} seconds")
            print()
        
        # Save full results
        output_file = 'csv_swarm_analysis_results.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✓ Full results saved to: {output_file}")
        print()
        
    else:
        print(f"✗ Swarm analysis failed: {response.status_code}")
        print(f"  Response: {response.text[:500]}")
        exit(1)

except requests.exceptions.Timeout:
    print("✗ Request timed out (analysis took too long)")
    print("  This is normal for large portfolios")
    print("  Try again or increase timeout")
    exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# Summary
# ============================================================================
print("="*80)
print("TEST SUMMARY")
print("="*80)
print()
print("✓ API Health Check: PASSED")
print("✓ CSV Upload + Swarm Analysis: PASSED")
print()
print("The integration is working!")
print()
print("Next Steps:")
print("  1. Open frontend: http://localhost:5173/swarm-analysis")
print("  2. Upload your CSV file")
print("  3. Get AI-powered recommendations!")
print()

