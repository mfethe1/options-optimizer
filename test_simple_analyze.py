"""
Simple test to capture actual error
"""

import requests
import json
import sys

payload = {
    'portfolio_data': {
        'positions': [{
            'symbol': 'AAPL',
            'asset_type': 'option',
            'quantity': 10,
            'market_value': 1000
        }],
        'total_value': 1000
    },
    'market_data': {},
    'consensus_method': 'weighted'
}

print("Sending request...")
print(f"Payload: {json.dumps(payload, indent=2)}")

try:
    response = requests.post(
        "http://localhost:8000/api/swarm/analyze",
        json=payload,
        timeout=60
    )
    
    print(f"\nStatus: {response.status_code}")
    print(f"Headers: {response.headers}")
    print(f"Body: {response.text}")
    
    if response.status_code == 200:
        data = response.json()
        print("\n✓ SUCCESS!")
        print(json.dumps(data, indent=2))
    else:
        print("\n✗ FAILED")
        try:
            error = response.json()
            print(f"Error detail: {error}")
        except:
            pass
        
except Exception as e:
    print(f"\n✗ Exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

