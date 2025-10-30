"""
Test API recommendations for multiple symbols (integration test)
- Parametrized over a small symbol set
- Skips gracefully if backend is not running
"""
import json
import requests
import pytest

SYMBOLS = ['NVDA', 'AAPL', 'TSLA']

@pytest.mark.parametrize("symbol", SYMBOLS)
def test_recommendation(symbol):
    """Test recommendation API for a symbol."""
    print(f"\n{'='*80}")
    print(f"RECOMMENDATION FOR {symbol}")
    print('='*80)

    try:
        response = requests.get(
            f"http://localhost:8000/api/recommendations/{symbol}", timeout=5
        )
    except requests.exceptions.RequestException:
        pytest.skip("Backend not running; skipping API integration test")
        return

    assert response.status_code == 200, f"HTTP {response.status_code}: {response.text[:200]}"
    data = response.json()

    # Basic shape checks
    assert 'recommendation' in data
    assert 'confidence' in data
    assert 'combined_score' in data

    # Log some helpful output
    print(f"\n✓ Recommendation: {data['recommendation']}")
    print(f"✓ Confidence: {float(data['confidence']):.1f}%")
    print(f"✓ Combined Score: {float(data['combined_score']):.1f}/100")

    if 'scores' in data:
        print("\n--- Scores ---")
        for name, score_data in data['scores'].items():
            score = score_data.get('score', 0)
            reasoning = score_data.get('reasoning', '')
            print(f"{name.title()}: {float(score):.1f}/100 - {reasoning}")

    if data.get('actions'):
        print("\n--- Actions ---")
        for action in data['actions']:
            priority = action.get('priority', 0)
            action_type = action.get('action', 'UNKNOWN')
            reason = action.get('reason', 'No reason')
            print(f"{priority}. {action_type}: {reason}")

