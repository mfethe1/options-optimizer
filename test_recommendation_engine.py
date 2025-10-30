"""
Test script for recommendation engine - helps debug issues
"""
import sys
import traceback
from src.analytics.recommendation_engine import RecommendationEngine
from src.data.position_manager import PositionManager
from src.data.market_data_fetcher import MarketDataFetcher

def test_recommendation_engine():
    """Test the recommendation engine with NVDA"""
    
    print("=" * 80)
    print("Testing Recommendation Engine")
    print("=" * 80)
    
    try:
        # Initialize components
        print("\n1. Initializing components...")
        rec_engine = RecommendationEngine()
        position_manager = PositionManager()
        market_data = MarketDataFetcher()
        print("✓ Components initialized")
        
        # Get position
        print("\n2. Getting NVDA position...")
        position = None
        try:
            stock_positions = position_manager.get_all_stock_positions()
            for pos in stock_positions:
                if pos.symbol == 'NVDA':
                    position = pos.to_dict()
                    print(f"✓ Found position: {position.get('quantity', 0)} shares")
                    break
        except Exception as e:
            print(f"⚠ Could not fetch position: {e}")
        
        # Get market data
        print("\n3. Getting market data...")
        market_data_dict = None
        try:
            market_data_result = market_data.get_stock_price('NVDA')
            if market_data_result:
                # get_stock_price already returns a dict with current_price
                market_data_dict = market_data_result
                current_price = market_data_result.get('current_price', 0)
                print(f"✓ Current price: ${current_price:.2f}")
        except Exception as e:
            print(f"⚠ Could not fetch market data: {e}")
        
        # Generate recommendation
        print("\n4. Generating recommendation...")
        result = rec_engine.analyze(
            symbol='NVDA',
            position=position,
            market_data=market_data_dict
        )
        
        print("\n" + "=" * 80)
        print("RECOMMENDATION RESULT")
        print("=" * 80)
        
        print(f"\nSymbol: {result.symbol}")
        print(f"Recommendation: {result.recommendation}")
        print(f"Confidence: {result.confidence:.1f}%")
        print(f"Combined Score: {result.combined_score:.1f}/100")
        
        print("\n--- Scores ---")
        for name, score in result.scores.items():
            print(f"{name.title()}: {score.score:.1f}/100")
            print(f"  Reasoning: {score.reasoning}")
        
        print("\n--- Actions ---")
        for i, action in enumerate(result.actions, 1):
            print(f"{i}. {action.get('action')}: {action.get('reason')}")
        
        print("\n--- Risk Factors ---")
        for risk in result.risk_factors:
            print(f"- {risk}")
        
        print("\n--- Catalysts ---")
        for catalyst in result.catalysts:
            print(f"+ {catalyst}")
        
        print("\n✓ SUCCESS!")
        return True
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR OCCURRED")
        print("=" * 80)
        print(f"\nError: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_recommendation_engine()
    sys.exit(0 if success else 1)

