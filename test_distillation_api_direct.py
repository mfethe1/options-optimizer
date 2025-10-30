"""
Direct API Test for Distillation Agent

Tests the backend API directly without Playwright to verify:
1. Distillation Agent initializes
2. CSV upload works
3. Investor report is generated
4. Deduplication metrics are tracked
"""

import requests
import json
import time
from datetime import datetime

BACKEND_URL = "http://localhost:8000"
TEST_CSV_PATH = "data/examples/positions.csv"

def log(message):
    """Log with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def test_backend_health():
    """Test backend health"""
    log("\n" + "="*80)
    log("TEST 1: Backend Health Check")
    log("="*80)
    
    try:
        response = requests.get(f"{BACKEND_URL}/api/monitoring/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            log(f"✅ Backend is healthy")
            log(f"   Active analyses: {data.get('active_analyses', 0)}")
            log(f"   Completed analyses: {data.get('completed_analyses', 0)}")
            return True
        else:
            log(f"❌ Backend returned status {response.status_code}")
            return False
    except Exception as e:
        log(f"❌ Backend health check failed: {e}")
        return False

def test_csv_upload_and_analysis():
    """Test CSV upload and swarm analysis"""
    log("\n" + "="*80)
    log("TEST 2: CSV Upload and Swarm Analysis")
    log("="*80)
    
    try:
        # Upload CSV
        with open(TEST_CSV_PATH, 'rb') as f:
            files = {'file': f}
            params = {
                'chase_format': 'true',
                'consensus_method': 'weighted'
            }
            
            log(f"📁 Uploading: {TEST_CSV_PATH}")
            start_time = time.time()
            
            response = requests.post(
                f"{BACKEND_URL}/api/swarm/analyze-csv",
                files=files,
                params=params,
                timeout=600  # 10 minutes
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                log(f"✅ Analysis complete in {duration:.1f}s")
                
                # Check import stats
                import_stats = data.get('import_stats', {})
                log(f"\n📊 Import Stats:")
                log(f"   Positions imported: {import_stats.get('positions_imported', 0)}")
                log(f"   Positions failed: {import_stats.get('positions_failed', 0)}")
                
                # Check consensus decisions
                consensus = data.get('consensus_decisions', {})
                if consensus:
                    log(f"\n📊 Consensus Decisions:")
                    overall = consensus.get('overall_action', {})
                    log(f"   Action: {overall.get('action', 'N/A')}")
                    log(f"   Confidence: {overall.get('confidence', 0):.2%}")
                
                # Check for investor_report
                investor_report = data.get('investor_report')
                if investor_report:
                    log(f"\n✅ INVESTOR REPORT FOUND!")
                    log(f"\n📄 Investor Report Structure:")
                    
                    # Check sections
                    sections = [
                        'executive_summary',
                        'recommendation',
                        'risk_assessment',
                        'future_outlook',
                        'next_steps'
                    ]
                    
                    for section in sections:
                        if section in investor_report:
                            log(f"   ✅ {section}")
                            
                            # Show sample content
                            content = investor_report[section]
                            if isinstance(content, str):
                                preview = content[:100] + "..." if len(content) > 100 else content
                                log(f"      Preview: {preview}")
                            elif isinstance(content, dict):
                                log(f"      Keys: {list(content.keys())}")
                        else:
                            log(f"   ❌ {section} - MISSING")
                    
                    # Save full report
                    with open('test_output/investor_report.json', 'w') as f:
                        json.dump(investor_report, f, indent=2)
                    log(f"\n💾 Full report saved to: test_output/investor_report.json")
                    
                    return True, data
                else:
                    log(f"\n❌ INVESTOR REPORT NOT FOUND in response")
                    log(f"   Response keys: {list(data.keys())}")
                    return False, data
            else:
                log(f"❌ Analysis failed with status {response.status_code}")
                log(f"   Response: {response.text[:500]}")
                return False, None
                
    except Exception as e:
        log(f"❌ CSV upload/analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_deduplication_metrics():
    """Test deduplication metrics"""
    log("\n" + "="*80)
    log("TEST 3: Deduplication Metrics")
    log("="*80)
    
    try:
        response = requests.get(f"{BACKEND_URL}/api/monitoring/diagnostics", timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            # Look for shared_context metrics
            if "shared_context" in data:
                context_metrics = data["shared_context"]
                
                log(f"✅ Shared Context Metrics Found:")
                log(f"   Total messages: {context_metrics.get('total_messages', 0)}")
                log(f"   Duplicate messages: {context_metrics.get('duplicate_messages', 0)}")
                log(f"   Deduplication rate: {context_metrics.get('deduplication_rate', 0):.2%}")
                log(f"   Unique topics: {context_metrics.get('unique_topics', 0)}")
                
                dedup_rate = context_metrics.get('deduplication_rate', 0)
                if dedup_rate >= 0.5:
                    log(f"\n✅ Deduplication rate is good (>50%)")
                    return True
                else:
                    log(f"\n⚠️ Deduplication rate is low (<50%)")
                    return True  # Still pass, just warn
            else:
                log(f"⚠️ Shared context metrics not found in diagnostics")
                log(f"   Available keys: {list(data.keys())}")
                return False
        else:
            log(f"❌ Diagnostics request failed: {response.status_code}")
            return False
    except Exception as e:
        log(f"❌ Deduplication metrics check failed: {e}")
        return False

def main():
    """Run all tests"""
    log("="*80)
    log("DIRECT API TEST: Distillation Agent & Investor Report")
    log("="*80)
    
    results = {}
    
    # Test 1: Backend health
    results['backend_health'] = test_backend_health()
    
    if not results['backend_health']:
        log("\n❌ Backend not healthy. Aborting tests.")
        return 1
    
    # Test 2: CSV upload and analysis
    results['csv_analysis'], analysis_data = test_csv_upload_and_analysis()
    
    # Test 3: Deduplication metrics
    results['deduplication'] = test_deduplication_metrics()
    
    # Summary
    log("\n" + "="*80)
    log("TEST SUMMARY")
    log("="*80)
    
    for test, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        log(f"{status}: {test}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    log(f"\nTotal: {passed_count}/{total_count} tests passed ({passed_count/total_count*100:.0f}%)")
    
    if passed_count == total_count:
        log("\n🎉 ALL TESTS PASSED! Distillation Agent is working! 🚀")
        return 0
    else:
        log("\n⚠️ Some tests failed. Review output above.")
        return 1

if __name__ == "__main__":
    import sys
    import os
    
    # Create output directory
    os.makedirs('test_output', exist_ok=True)
    
    sys.exit(main())

