"""
Test metrics integration in swarm analysis.
"""
import requests
import json
import time

def test_metrics_integration():
    """Test that portfolio metrics are calculated and included in investor report"""
    
    print("=" * 80)
    print("TESTING METRICS INTEGRATION")
    print("=" * 80)
    
    # Create test portfolio with multiple positions
    files = {'file': open('data/examples/positions.csv', 'rb')}
    
    print("\n1. Uploading CSV and running swarm analysis...")
    start = time.time()
    
    try:
        response = requests.post(
            'http://localhost:8000/api/swarm/analyze-csv',
            params={
                'chase_format': 'true',
                'consensus_method': 'weighted'
            },
            files=files,
            timeout=600  # 10 minutes max
        )
        
        duration = time.time() - start
        print(f"   ✓ Request completed in {duration:.1f}s")
        print(f"   Status code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   ✗ Error: {response.text[:500]}")
            return False
        
        data = response.json()
        
        # Save full response
        with open('test_output/metrics_test_response.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print("   ✓ Saved full response to test_output/metrics_test_response.json")
        
        # Check for investor_report
        print("\n2. Checking for investor_report...")
        investor_report = data.get('investor_report')
        
        if not investor_report:
            print("   ✗ NO investor_report in response")
            print(f"   Available keys: {list(data.keys())}")
            return False
        
        print("   ✓ investor_report found")
        
        # Save investor report
        with open('test_output/investor_report_with_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(investor_report, f, indent=2)
        print("   ✓ Saved to test_output/investor_report_with_metrics.json")
        
        # Check report structure
        print("\n3. Analyzing investor report structure...")
        print(f"   Report keys: {list(investor_report.keys())}")
        
        # Check for metrics in report sections
        report_text = json.dumps(investor_report, indent=2)
        
        metrics_keywords = [
            'sharpe', 'omega', 'volatility', 'drawdown',
            'concentration', 'diversification', 'win rate',
            'risk-adjusted', 'effective n'
        ]
        
        found_metrics = []
        for keyword in metrics_keywords:
            if keyword.lower() in report_text.lower():
                found_metrics.append(keyword)
        
        print(f"\n4. Metrics found in report:")
        if found_metrics:
            for metric in found_metrics:
                print(f"   ✓ {metric}")
        else:
            print("   ⚠️  No metrics keywords found in report")
        
        # Check executive summary
        exec_summary = investor_report.get('executive_summary', '')
        if exec_summary:
            print(f"\n5. Executive Summary (first 300 chars):")
            print(f"   {exec_summary[:300]}...")
        
        # Check metadata
        metadata = investor_report.get('metadata', {})
        if metadata:
            print(f"\n6. Report Metadata:")
            print(f"   Generated at: {metadata.get('generated_at')}")
            print(f"   Total insights: {metadata.get('total_insights')}")
            print(f"   Categories: {metadata.get('categories')}")
        
        # Success criteria
        print("\n" + "=" * 80)
        print("SUCCESS CRITERIA:")
        print("=" * 80)
        
        criteria = {
            'investor_report exists': investor_report is not None,
            'executive_summary exists': bool(exec_summary),
            'metrics keywords found': len(found_metrics) >= 3,
            'metadata exists': bool(metadata)
        }
        
        for criterion, passed in criteria.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {criterion}")
        
        all_passed = all(criteria.values())
        
        if all_passed:
            print("\n🎉 ALL CRITERIA PASSED!")
            print(f"   Duration: {duration:.1f}s")
            print(f"   Metrics found: {', '.join(found_metrics)}")
        else:
            print("\n⚠️  SOME CRITERIA FAILED")
        
        return all_passed
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import os
    os.makedirs('test_output', exist_ok=True)
    
    success = test_metrics_integration()
    exit(0 if success else 1)

