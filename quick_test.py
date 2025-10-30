"""
Quick Test - Get recommendation for a symbol
Usage: python quick_test.py NVDA
"""
import sys
import requests
import json

def get_recommendation(symbol):
    """Get and display recommendation for a symbol"""
    print(f"\n{'='*80}")
    print(f"RECOMMENDATION FOR {symbol}")
    print('='*80)
    
    try:
        # Get recommendation
        response = requests.get(f'http://localhost:8000/api/recommendations/{symbol}')
        response.raise_for_status()
        data = response.json()
        
        # Display results
        print(f"\n✓ Recommendation: {data['recommendation']}")
        print(f"✓ Confidence: {data['confidence']:.1f}%")
        print(f"✓ Combined Score: {data['combined_score']:.1f}/100")
        
        print("\n--- Score Breakdown ---")
        for name, score_data in data['scores'].items():
            score = score_data['score']
            reasoning = score_data['reasoning']
            print(f"\n{name.upper()}: {score:.1f}/100")
            print(f"  {reasoning}")
        
        print("\n--- Recommended Actions ---")
        for i, action in enumerate(data['actions'], 1):
            action_type = action.get('action', 'UNKNOWN')
            reason = action.get('reason', 'No reason')
            print(f"{i}. {action_type}: {reason}")
            
            # Show details
            if action.get('quantity'):
                print(f"   Quantity: {action['quantity']}")
            if action.get('stop_price'):
                print(f"   Stop Price: ${action['stop_price']:.2f}")
            if action.get('target_price'):
                print(f"   Target Price: ${action['target_price']:.2f}")
            if action.get('expected_proceeds'):
                print(f"   Expected Proceeds: ${action['expected_proceeds']:,.2f}")
        
        if data.get('risk_factors'):
            print("\n--- Risk Factors ---")
            for risk in data['risk_factors']:
                print(f"⚠ {risk}")
        
        if data.get('catalysts'):
            print("\n--- Catalysts ---")
            for catalyst in data['catalysts'][:5]:
                print(f"+ {catalyst}")
        
        print("\n" + "="*80)
        print("✓ SUCCESS!")
        print("="*80 + "\n")
        
    except requests.exceptions.RequestException as e:
        print(f"\n✗ ERROR: Could not connect to API")
        print(f"  Make sure the server is running: python -m uvicorn src.api.main_simple:app --reload")
        print(f"  Error: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python quick_test.py SYMBOL")
        print("\nExamples:")
        print("  python quick_test.py NVDA")
        print("  python quick_test.py AAPL")
        print("  python quick_test.py TSLA")
        print()
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    get_recommendation(symbol)

