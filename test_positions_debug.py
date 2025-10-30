"""
Debug script to test /api/positions endpoint
"""
import sys
sys.path.insert(0, 'e:/Projects/Options_probability')

from src.data.position_manager import PositionManager
from src.data.market_data_fetcher import MarketDataFetcher

def test_positions():
    print("=" * 80)
    print("Testing Position Manager and Market Data Fetcher")
    print("=" * 80)
    
    # Initialize
    print("\n1. Initializing components...")
    position_manager = PositionManager()
    market_data = MarketDataFetcher()
    
    # Get positions
    print("\n2. Getting positions...")
    stock_positions = position_manager.get_all_stock_positions()
    option_positions = position_manager.get_all_option_positions()
    print(f"   Found {len(stock_positions)} stocks, {len(option_positions)} options")
    
    # Test stock enhancement
    print("\n3. Testing stock enhancement...")
    for idx, stock in enumerate(stock_positions):
        print(f"\n   Stock {idx+1}: {stock.symbol}")
        try:
            print(f"   - Fetching market data...")
            market_data_result = market_data.get_stock_price(stock.symbol)
            
            if market_data_result:
                print(f"   - Got price: ${market_data_result.get('current_price')}")
                print(f"   - Calculating metrics...")
                stock.calculate_metrics(market_data_result)
                print(f"   - Success! P&L: ${stock.unrealized_pnl}")
            else:
                print(f"   - WARNING: No market data returned")
                
            print(f"   - Converting to dict...")
            stock_dict = stock.to_dict()
            print(f"   - Success! Dict has {len(stock_dict)} fields")
            
        except Exception as e:
            print(f"   - ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    # Test option enhancement
    print("\n4. Testing option enhancement...")
    for idx, option in enumerate(option_positions):
        print(f"\n   Option {idx+1}: {option.symbol} {option.strike} {option.option_type}")
        try:
            print(f"   - Fetching underlying price...")
            market_data_result = market_data.get_stock_price(option.symbol)
            
            if market_data_result:
                print(f"   - Got underlying: ${market_data_result.get('current_price')}")
            else:
                print(f"   - WARNING: No underlying data")
            
            print(f"   - Fetching option data...")
            option_data = market_data.get_option_price(
                option.symbol,
                option.option_type,
                option.strike,
                option.expiration_date
            )
            
            if option_data:
                print(f"   - Got option price: ${option_data.get('last_price')}")
            else:
                print(f"   - WARNING: No option data")
            
            if market_data_result and option_data:
                print(f"   - Calculating metrics...")
                option.calculate_metrics(market_data_result, option_data)
                print(f"   - Success! P&L: ${option.unrealized_pnl}")
            
            print(f"   - Converting to dict...")
            option_dict = option.to_dict()
            print(f"   - Success! Dict has {len(option_dict)} fields")
            
        except Exception as e:
            print(f"   - ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    # Test portfolio summary
    print("\n5. Testing portfolio summary...")
    try:
        summary = position_manager.get_portfolio_summary()
        print(f"   Success! Summary: {summary}")
    except Exception as e:
        print(f"   ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)

if __name__ == "__main__":
    test_positions()

