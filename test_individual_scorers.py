"""
Test Individual Scorers in Detail
Validates each scorer's logic and output
"""
import sys
from src.analytics.technical_scorer import TechnicalScorer
from src.analytics.fundamental_scorer import FundamentalScorer
from src.analytics.sentiment_scorer import SentimentScorer
from src.analytics.risk_scorer import RiskScorer
from src.analytics.earnings_risk_scorer import EarningsRiskScorer
from src.analytics.correlation_scorer import CorrelationScorer

def test_technical_scorer():
    """Test Technical Scorer"""
    print("\n" + "="*80)
    print("TEST: Technical Scorer")
    print("="*80)
    
    scorer = TechnicalScorer()
    
    # Test with NVDA
    result = scorer.score('NVDA', position=None, market_data=None)
    
    print(f"\nScore: {result.score}/100")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")
    
    print("\nComponents:")
    for name, value in result.components.items():
        print(f"  {name}: {value}")
    
    print("\nSignals:")
    for name, value in result.signals.items():
        print(f"  {name}: {value}")
    
    # Validate
    assert 0 <= result.score <= 100, "Score out of range"
    assert 0 <= result.confidence <= 1.0, "Confidence out of range"
    assert result.reasoning, "No reasoning provided"
    assert result.components, "No components provided"
    assert result.signals, "No signals provided"
    
    print("\n✓ Technical Scorer PASSED")
    return True

def test_fundamental_scorer():
    """Test Fundamental Scorer"""
    print("\n" + "="*80)
    print("TEST: Fundamental Scorer")
    print("="*80)
    
    scorer = FundamentalScorer()
    
    # Test with NVDA
    result = scorer.score('NVDA', position=None, market_data=None)
    
    print(f"\nScore: {result.score}/100")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")
    
    print("\nComponents:")
    for name, value in result.components.items():
        print(f"  {name}: {value}")
    
    print("\nKey Signals:")
    for name, value in list(result.signals.items())[:10]:  # Show first 10
        print(f"  {name}: {value}")
    
    # Validate
    assert 0 <= result.score <= 100, "Score out of range"
    assert 0 <= result.confidence <= 1.0, "Confidence out of range"
    assert result.reasoning, "No reasoning provided"
    
    print("\n✓ Fundamental Scorer PASSED")
    return True

def test_sentiment_scorer():
    """Test Sentiment Scorer"""
    print("\n" + "="*80)
    print("TEST: Sentiment Scorer")
    print("="*80)
    
    scorer = SentimentScorer()
    
    # Test with NVDA
    result = scorer.score('NVDA', position=None, market_data=None)
    
    print(f"\nScore: {result.score}/100")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")
    
    print("\nComponents:")
    for name, value in result.components.items():
        print(f"  {name}: {value}")
    
    print("\nSignals:")
    for name, value in result.signals.items():
        print(f"  {name}: {value}")
    
    # Validate
    assert 0 <= result.score <= 100, "Score out of range"
    assert 0 <= result.confidence <= 1.0, "Confidence out of range"
    
    print("\n✓ Sentiment Scorer PASSED")
    return True

def test_risk_scorer():
    """Test Risk Scorer"""
    print("\n" + "="*80)
    print("TEST: Risk Scorer")
    print("="*80)
    
    scorer = RiskScorer()
    
    # Test with NVDA
    result = scorer.score('NVDA', position=None, market_data=None)
    
    print(f"\nScore: {result.score}/100 (lower is better)")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")
    
    print("\nComponents:")
    for name, value in result.components.items():
        print(f"  {name}: {value}")
    
    print("\nSignals:")
    for name, value in result.signals.items():
        print(f"  {name}: {value}")
    
    # Validate
    assert 0 <= result.score <= 100, "Score out of range"
    assert 0 <= result.confidence <= 1.0, "Confidence out of range"
    
    print("\n✓ Risk Scorer PASSED")
    return True

def test_earnings_risk_scorer():
    """Test Earnings Risk Scorer"""
    print("\n" + "="*80)
    print("TEST: Earnings Risk Scorer")
    print("="*80)
    
    scorer = EarningsRiskScorer()
    
    # Test with NVDA
    result = scorer.score('NVDA', position=None, market_data=None)
    
    print(f"\nScore: {result.score}/100 (lower is better)")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")
    
    print("\nComponents:")
    for name, value in result.components.items():
        print(f"  {name}: {value}")
    
    print("\nSignals:")
    for name, value in result.signals.items():
        print(f"  {name}: {value}")
    
    # Validate
    assert 0 <= result.score <= 100, "Score out of range"
    assert 0 <= result.confidence <= 1.0, "Confidence out of range"
    
    print("\n✓ Earnings Risk Scorer PASSED")
    return True

def test_correlation_scorer():
    """Test Correlation Scorer"""
    print("\n" + "="*80)
    print("TEST: Correlation Scorer")
    print("="*80)
    
    scorer = CorrelationScorer()
    
    # Test with NVDA
    result = scorer.score('NVDA', position=None, market_data=None)
    
    print(f"\nScore: {result.score}/100")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")
    
    print("\nComponents:")
    for name, value in result.components.items():
        print(f"  {name}: {value}")
    
    print("\nSignals:")
    for name, value in result.signals.items():
        print(f"  {name}: {value}")
    
    # Validate
    assert 0 <= result.score <= 100, "Score out of range"
    assert 0 <= result.confidence <= 1.0, "Confidence out of range"
    
    print("\n✓ Correlation Scorer PASSED")
    return True

def main():
    """Run all scorer tests"""
    print("\n" + "="*80)
    print("INDIVIDUAL SCORER TESTING")
    print("="*80)
    
    results = {
        'Technical': False,
        'Fundamental': False,
        'Sentiment': False,
        'Risk': False,
        'Earnings Risk': False,
        'Correlation': False
    }
    
    try:
        results['Technical'] = test_technical_scorer()
    except Exception as e:
        print(f"\n✗ Technical Scorer FAILED: {e}")
    
    try:
        results['Fundamental'] = test_fundamental_scorer()
    except Exception as e:
        print(f"\n✗ Fundamental Scorer FAILED: {e}")
    
    try:
        results['Sentiment'] = test_sentiment_scorer()
    except Exception as e:
        print(f"\n✗ Sentiment Scorer FAILED: {e}")
    
    try:
        results['Risk'] = test_risk_scorer()
    except Exception as e:
        print(f"\n✗ Risk Scorer FAILED: {e}")
    
    try:
        results['Earnings Risk'] = test_earnings_risk_scorer()
    except Exception as e:
        print(f"\n✗ Earnings Risk Scorer FAILED: {e}")
    
    try:
        results['Correlation'] = test_correlation_scorer()
    except Exception as e:
        print(f"\n✗ Correlation Scorer FAILED: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nTotal Scorers: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total*100):.1f}%")
    
    print("\nResults:")
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    if passed == total:
        print("\n✓ ALL SCORERS PASSED!")
        return 0
    else:
        print(f"\n✗ {total - passed} SCORER(S) FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())

