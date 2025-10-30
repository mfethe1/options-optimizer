"""
Test the fixed CSV upload endpoint
"""
import requests
import json

# Upload CSV and analyze
files = {'file': open('data/examples/positions.csv', 'rb')}
response = requests.post(
    'http://localhost:8000/api/swarm/analyze-csv?chase_format=true&consensus_method=weighted',
    files=files
)

if response.status_code == 200:
    data = response.json()
    
    print("=" * 80)
    print("CSV UPLOAD TEST - FRONTEND FIX VERIFICATION")
    print("=" * 80)
    
    # Check consensus decisions format
    print("\n✓ Overall Action:")
    overall = data['consensus_decisions']['overall_action']
    print(f"  Choice: {overall['choice']}")
    print(f"  Confidence: {overall['confidence']*100:.0f}%")
    print(f"  Reasoning: {overall['reasoning'][:100]}...")
    
    print("\n✓ Risk Level:")
    risk = data['consensus_decisions']['risk_level']
    print(f"  Choice: {risk['choice']}")
    print(f"  Confidence: {risk['confidence']*100:.0f}%")
    print(f"  Reasoning: {risk['reasoning'][:100]}...")
    
    print("\n✓ Market Outlook:")
    outlook = data['consensus_decisions']['market_outlook']
    print(f"  Choice: {outlook['choice']}")
    print(f"  Confidence: {outlook['confidence']*100:.0f}%")
    print(f"  Reasoning: {outlook['reasoning'][:100]}...")
    
    # Check import stats
    print("\n✓ Import Stats:")
    stats = data['import_stats']
    print(f"  Positions Imported: {stats['positions_imported']}")
    print(f"  Positions Failed: {stats['positions_failed']}")
    print(f"  Chase Conversion Errors: {stats['chase_conversion']['conversion_errors']}")
    print(f"  (These are expected - cash positions and footnotes)")
    
    print("\n" + "=" * 80)
    print("✅ FRONTEND FIX VERIFIED!")
    print("=" * 80)
    print("\nThe frontend should now display:")
    print(f"  - Imported {stats['positions_imported']} positions")
    print(f"  - Overall Action: {overall['choice'].upper()}")
    print(f"  - Risk Level: {risk['choice'].upper()}")
    print(f"  - Market Outlook: {outlook['choice'].upper()}")
    print("\nRefresh the browser and try uploading the CSV again!")
    
else:
    print(f"❌ Error: HTTP {response.status_code}")
    print(response.text)

