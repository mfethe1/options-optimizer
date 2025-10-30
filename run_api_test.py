"""
Simple API test runner for distillation system
"""
import requests
import json
import time
import os
import sys

def main():
    print("="*80)
    print("API INTEGRATION TEST: Distillation Agent")
    print("="*80)
    
    # Create output directory
    os.makedirs('test_output', exist_ok=True)
    
    # Check backend health
    print("\n[1/3] Checking backend health...")
    try:
        r = requests.get('http://localhost:8000/api/monitoring/health', timeout=5)
        if r.status_code == 200:
            print("✅ Backend is healthy")
        else:
            print(f"❌ Backend returned status {r.status_code}")
            return 1
    except Exception as e:
        print(f"❌ Backend not accessible: {e}")
        return 1
    
    # Upload CSV and analyze
    print("\n[2/3] Uploading CSV and running swarm analysis...")
    print("⏳ This will take 5-10 minutes for 17 agents to analyze...")
    
    try:
        with open('data/examples/positions.csv', 'rb') as f:
            files = {'file': f}
            params = {
                'chase_format': 'true',
                'consensus_method': 'weighted'
            }
            
            start_time = time.time()
            r = requests.post(
                'http://localhost:8000/api/swarm/analyze-csv',
                files=files,
                params=params,
                timeout=1200  # 20 minutes max
            )
            duration = time.time() - start_time
            
            print(f"\n⏱️  Analysis completed in {duration:.1f} seconds ({duration/60:.1f} minutes)")
            
            if r.status_code != 200:
                print(f"❌ Analysis failed with status {r.status_code}")
                print(f"Response: {r.text[:500]}")
                return 1
            
            # Parse response
            data = r.json()
            
            # Save full response
            with open('test_output/analysis_response.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print("💾 Full response saved to: test_output/analysis_response.json")
            
            # Check for investor_report
            investor_report = data.get('investor_report')
            if investor_report:
                print("\n✅ INVESTOR REPORT FOUND!")
                
                # Save investor report
                with open('test_output/investor_report.json', 'w', encoding='utf-8') as f:
                    json.dump(investor_report, f, indent=2)
                print("💾 Investor report saved to: test_output/investor_report.json")
                
                # Show structure
                print("\n📊 Investor Report Structure:")
                sections = [
                    'executive_summary',
                    'recommendation',
                    'risk_assessment',
                    'future_outlook',
                    'next_steps'
                ]
                
                for section in sections:
                    if section in investor_report:
                        print(f"   ✅ {section}")
                    else:
                        print(f"   ❌ {section} - MISSING")
                
                # Show sample content
                if 'recommendation' in investor_report:
                    rec = investor_report['recommendation']
                    print(f"\n📈 Recommendation Preview:")
                    print(f"   Action: {rec.get('action', 'N/A')}")
                    print(f"   Conviction: {rec.get('conviction', 'N/A')}")
                
            else:
                print("\n❌ INVESTOR REPORT NOT FOUND in response")
                print(f"Response keys: {list(data.keys())}")
                return 1
                
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Check deduplication metrics
    print("\n[3/3] Checking deduplication metrics...")
    try:
        r = requests.get('http://localhost:8000/api/monitoring/diagnostics', timeout=10)
        if r.status_code == 200:
            data = r.json()
            
            if 'shared_context' in data:
                ctx = data['shared_context']
                print(f"✅ Shared Context Metrics:")
                print(f"   Total messages: {ctx.get('total_messages', 0)}")
                print(f"   Duplicate messages: {ctx.get('duplicate_messages', 0)}")
                print(f"   Deduplication rate: {ctx.get('deduplication_rate', 0):.2%}")
            else:
                print("⚠️  Shared context metrics not available")
        else:
            print(f"⚠️  Diagnostics returned status {r.status_code}")
    except Exception as e:
        print(f"⚠️  Could not fetch diagnostics: {e}")
    
    print("\n" + "="*80)
    print("✅ API INTEGRATION TEST PASSED!")
    print("="*80)
    return 0

if __name__ == '__main__':
    sys.exit(main())

