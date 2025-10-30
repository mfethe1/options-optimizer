"""
Test Chase CSV Import Functionality
Tests the enhanced import with chase_format parameter
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.position_manager import PositionManager
from src.data.csv_position_service import CSVPositionService
from src.data.position_enrichment_service import PositionEnrichmentService


def test_chase_import():
    """Test importing Chase CSV with chase_format=True"""
    print("="*80)
    print("TESTING CHASE CSV IMPORT")
    print("="*80)
    
    # Initialize services
    pm = PositionManager(storage_file="data/test_chase_positions.json")
    csv_service = CSVPositionService(pm)
    enrichment_service = PositionEnrichmentService(pm)
    
    # Read Chase CSV
    print("\n📂 Reading Chase CSV file...")
    with open('data/examples/positions.csv', 'r', encoding='utf-8') as f:
        chase_csv_content = f.read()
    
    # Import with chase_format=True
    print("\n🔄 Importing with chase_format=True...")
    results = csv_service.import_option_positions(
        chase_csv_content,
        replace_existing=True,
        chase_format=True
    )
    
    # Display results
    print("\n📊 Import Results:")
    print(f"   ✅ Success: {results['success']}")
    print(f"   ❌ Failed: {results['failed']}")
    print(f"   📝 Position IDs: {results['position_ids']}")
    
    if results['errors']:
        print(f"\n⚠️  Errors:")
        for error in results['errors']:
            print(f"   {error}")
    
    # Display Chase conversion stats
    if results['chase_conversion']:
        conv = results['chase_conversion']
        print(f"\n🔄 Chase Conversion Stats:")
        print(f"   Total rows processed: {conv['total_rows']}")
        print(f"   ✅ Options converted: {conv['options_converted']}")
        print(f"   ⏭️  Cash positions skipped: {conv['cash_skipped']}")
        print(f"   ❌ Conversion errors: {conv['conversion_errors']}")
        
        if conv['error_details']:
            print(f"\n   Error details:")
            for error in conv['error_details'][:5]:  # Show first 5
                print(f"   - {error}")
    
    # Enrich positions
    print("\n💰 Enriching positions with real-time data...")
    enrichment_service.enrich_all_positions()
    
    # Display imported positions
    print("\n📋 Imported Positions:")
    for pos_id in results['position_ids']:
        pos = pm.get_option_position(pos_id)
        print(f"\n   {pos.symbol} ${pos.strike} {pos.option_type.upper()} (exp: {pos.expiration_date})")
        print(f"   ├─ Quantity: {pos.quantity} contracts")
        print(f"   ├─ Premium Paid: ${pos.premium_paid:.2f}")
        print(f"   ├─ Entry Date: {pos.entry_date}")
        
        # Show Chase validation fields
        if pos.chase_last_price:
            print(f"   │")
            print(f"   ├─ Chase Data (as of {pos.chase_pricing_date}):")
            print(f"   │  ├─ Last Price: ${pos.chase_last_price:.2f}")
            print(f"   │  ├─ Market Value: ${pos.chase_market_value:.2f}")
            print(f"   │  ├─ Total Cost: ${pos.chase_total_cost:.2f}")
            print(f"   │  ├─ Unrealized P&L: ${pos.chase_unrealized_pnl:.2f} ({pos.chase_unrealized_pnl_pct:.2f}%)")
            print(f"   │  └─ Strategy: {pos.asset_strategy}")
        
        # Show our calculated data
        if pos.current_price:
            print(f"   │")
            print(f"   └─ Our Calculated Data:")
            print(f"      ├─ Current Price: ${pos.current_price:.2f}")
            print(f"      ├─ Unrealized P&L: ${pos.unrealized_pnl:.2f} ({pos.unrealized_pnl_pct:.2f}%)")
            if pos.delta:
                print(f"      ├─ Delta: {pos.delta:.3f}")
                print(f"      ├─ Theta: {pos.theta:.3f}")
            print(f"      └─ Days to Expiry: {pos.days_to_expiry()}")
    
    # Summary
    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)
    print(f"\nImported {results['success']} positions from Chase CSV")
    print(f"All positions include Chase validation data for comparison")
    print(f"\nTest data saved to: data/test_chase_positions.json")


if __name__ == "__main__":
    test_chase_import()

