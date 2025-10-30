"""
Test Position Management System
Comprehensive test of CSV import/export, enrichment, and AI integration
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.position_manager import PositionManager
from src.data.csv_position_service import CSVPositionService
from src.data.position_enrichment_service import PositionEnrichmentService
from src.agents.position_context_service import PositionContextService, ConversationMemory


def test_csv_templates():
    """Test CSV template generation"""
    print("\n" + "="*80)
    print("TEST 1: CSV Template Generation")
    print("="*80)
    
    pm = PositionManager(storage_file="data/test_positions.json")
    csv_service = CSVPositionService(pm)
    
    # Generate templates
    stock_template = csv_service.generate_stock_template()
    option_template = csv_service.generate_option_template()
    
    print("\n✅ Stock Template Generated:")
    print(stock_template[:200] + "...")
    
    print("\n✅ Option Template Generated:")
    print(option_template[:200] + "...")
    
    # Save templates
    with open("stock_template.csv", "w") as f:
        f.write(stock_template)
    with open("option_template.csv", "w") as f:
        f.write(option_template)
    
    print("\n✅ Templates saved to stock_template.csv and option_template.csv")
    return True


def test_csv_import():
    """Test CSV import functionality"""
    print("\n" + "="*80)
    print("TEST 2: CSV Import")
    print("="*80)
    
    pm = PositionManager(storage_file="data/test_positions.json")
    csv_service = CSVPositionService(pm)
    
    # Create sample CSV data
    stock_csv = """symbol,quantity,entry_price,entry_date,target_price,stop_loss,notes
AAPL,100,150.50,2025-01-15,175.00,140.00,Long-term hold
NVDA,50,500.00,2025-02-01,600.00,450.00,AI growth play
MSFT,75,380.00,2025-01-20,420.00,360.00,Cloud leader"""
    
    option_csv = """symbol,option_type,strike,expiration_date,quantity,premium_paid,entry_date,target_price,target_profit_pct,stop_loss_pct,notes
TSLA,call,250.00,2025-12-20,10,15.50,2025-10-01,25.00,50,30,Bullish on EV sector
SPY,put,450.00,2025-11-15,5,8.25,2025-10-10,12.00,40,25,Hedge position
AAPL,call,180.00,2025-11-21,20,5.75,2025-10-05,10.00,60,35,Earnings play"""
    
    # Import stocks
    print("\n📥 Importing stock positions...")
    stock_results = csv_service.import_stock_positions(stock_csv, replace_existing=True)
    print(f"✅ Stocks: {stock_results['success']} imported, {stock_results['failed']} failed")
    if stock_results['errors']:
        print(f"❌ Errors: {stock_results['errors']}")
    
    # Import options
    print("\n📥 Importing option positions...")
    option_results = csv_service.import_option_positions(option_csv, replace_existing=True)
    print(f"✅ Options: {option_results['success']} imported, {option_results['failed']} failed")
    if option_results['errors']:
        print(f"❌ Errors: {option_results['errors']}")
    
    # Verify
    print(f"\n📊 Total positions: {len(pm.stock_positions)} stocks, {len(pm.option_positions)} options")
    
    return stock_results['success'] > 0 and option_results['success'] > 0


def test_position_enrichment():
    """Test position enrichment with real-time data"""
    print("\n" + "="*80)
    print("TEST 3: Position Enrichment")
    print("="*80)
    
    pm = PositionManager(storage_file="data/test_positions.json")
    enrichment_service = PositionEnrichmentService(pm)
    
    print("\n🔄 Enriching all positions with real-time data...")
    results = enrichment_service.enrich_all_positions()
    
    print(f"\n✅ Enriched {results['stocks_enriched']} stocks, {results['options_enriched']} options")
    if results['errors']:
        print(f"⚠️  Errors: {results['errors'][:3]}")  # Show first 3 errors
    
    # Show sample enriched stock
    if pm.stock_positions:
        stock = list(pm.stock_positions.values())[0]
        print(f"\n📈 Sample Stock Position: {stock.symbol}")
        print(f"   Entry: ${stock.entry_price:.2f}")
        print(f"   Current: ${stock.current_price:.2f}" if stock.current_price else "   Current: N/A")
        print(f"   P&L: ${stock.unrealized_pnl:.2f} ({stock.unrealized_pnl_pct:.2f}%)" if stock.unrealized_pnl else "   P&L: N/A")
        print(f"   Status: {stock.get_status()}")
    
    # Show sample enriched option
    if pm.option_positions:
        option = list(pm.option_positions.values())[0]
        print(f"\n📊 Sample Option Position: {option.symbol} ${option.strike} {option.option_type.upper()}")
        print(f"   Premium Paid: ${option.premium_paid:.2f}")
        print(f"   Current: ${option.current_price:.2f}" if option.current_price else "   Current: N/A")
        print(f"   Underlying: ${option.underlying_price:.2f}" if option.underlying_price else "   Underlying: N/A")
        print(f"   P&L: ${option.unrealized_pnl:.2f} ({option.unrealized_pnl_pct:.2f}%)" if option.unrealized_pnl else "   P&L: N/A")
        print(f"   Delta: {option.delta:.3f}" if option.delta else "   Delta: N/A")
        print(f"   Theta: {option.theta:.3f}" if option.theta else "   Theta: N/A")
        print(f"   IV: {option.implied_volatility*100:.1f}%" if option.implied_volatility else "   IV: N/A")
        print(f"   Days to Expiry: {option.days_to_expiry()}")
        print(f"   Status: {option.get_status()}")
        print(f"   Risk Level: {option.get_risk_level()}")
    
    return results['stocks_enriched'] > 0 or results['options_enriched'] > 0


def test_portfolio_summary():
    """Test portfolio summary generation"""
    print("\n" + "="*80)
    print("TEST 4: Portfolio Summary")
    print("="*80)
    
    pm = PositionManager(storage_file="data/test_positions.json")
    enrichment_service = PositionEnrichmentService(pm)
    
    print("\n📊 Generating enriched portfolio summary...")
    summary = enrichment_service.get_enriched_portfolio_summary()
    
    print(f"\n💼 Portfolio Summary:")
    print(f"   Total Stocks: {summary['total_stocks']}")
    print(f"   Total Options: {summary['total_options']}")
    print(f"   Unique Symbols: {summary['unique_symbols']}")
    print(f"   Symbols: {', '.join(summary['symbols'][:10])}")
    print(f"\n💰 Values:")
    print(f"   Stock Value: ${summary.get('total_stock_current_value', 0):,.2f}")
    print(f"   Option Value: ${summary.get('total_option_current_value', 0):,.2f}")
    print(f"   Total Value: ${summary.get('total_current_value', 0):,.2f}")
    print(f"\n📈 P&L:")
    print(f"   Stock P&L: ${summary.get('total_stock_pnl', 0):,.2f}")
    print(f"   Option P&L: ${summary.get('total_option_pnl', 0):,.2f}")
    print(f"   Total P&L: ${summary.get('total_pnl', 0):,.2f} ({summary.get('total_pnl_pct', 0):.2f}%)")
    
    return True


def test_ai_agent_integration():
    """Test AI agent integration with position context"""
    print("\n" + "="*80)
    print("TEST 5: AI Agent Integration")
    print("="*80)
    
    pm = PositionManager(storage_file="data/test_positions.json")
    enrichment_service = PositionEnrichmentService(pm)
    context_service = PositionContextService(pm, enrichment_service)
    
    # Create conversation
    conversation_id = "test_session_001"
    
    print(f"\n💬 Creating conversation: {conversation_id}")
    
    # Add user message
    context_service.conversation_memory.add_message(
        conversation_id,
        "user",
        "What's my portfolio looking like?"
    )
    
    # Generate context for agent
    print("\n🤖 Generating context for AI agent...")
    context = context_service.create_agent_context(
        conversation_id=conversation_id,
        include_summary=True,
        include_positions=True,
        include_expiring=True
    )
    
    print("\n📄 Agent Context Preview:")
    print(context[:500] + "...\n")
    
    # Simulate agent response
    agent_response = "Based on your portfolio, you have 3 stocks and 3 options. Your total P&L is positive..."
    
    # Log interaction
    context_service.log_agent_interaction(
        conversation_id=conversation_id,
        user_query="What's my portfolio looking like?",
        agent_response=agent_response,
        positions_accessed=list(pm.stock_positions.keys())[:2]
    )
    
    print("✅ Agent interaction logged")
    
    # Get conversation history
    history = context_service.conversation_memory.get_conversation(conversation_id)
    print(f"\n📜 Conversation History: {len(history)} messages")
    for msg in history:
        print(f"   [{msg['role']}]: {msg['content'][:50]}...")
    
    return True


def test_csv_export():
    """Test CSV export functionality"""
    print("\n" + "="*80)
    print("TEST 6: CSV Export")
    print("="*80)
    
    pm = PositionManager(storage_file="data/test_positions.json")
    csv_service = CSVPositionService(pm)
    
    # Export stocks
    print("\n📤 Exporting stock positions...")
    stock_csv = csv_service.export_stock_positions()
    with open("exported_stocks.csv", "w") as f:
        f.write(stock_csv)
    print(f"✅ Exported {len(pm.stock_positions)} stocks to exported_stocks.csv")
    
    # Export options
    print("\n📤 Exporting option positions...")
    option_csv = csv_service.export_option_positions()
    with open("exported_options.csv", "w") as f:
        f.write(option_csv)
    print(f"✅ Exported {len(pm.option_positions)} options to exported_options.csv")
    
    print("\n📄 Stock CSV Preview:")
    print(stock_csv[:200] + "...")
    
    print("\n📄 Option CSV Preview:")
    print(option_csv[:200] + "...")
    
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("🚀 POSITION MANAGEMENT SYSTEM - COMPREHENSIVE TEST")
    print("="*80)
    
    tests = [
        ("CSV Template Generation", test_csv_templates),
        ("CSV Import", test_csv_import),
        ("Position Enrichment", test_position_enrichment),
        ("Portfolio Summary", test_portfolio_summary),
        ("AI Agent Integration", test_ai_agent_integration),
        ("CSV Export", test_csv_export)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result, None))
        except Exception as e:
            results.append((name, False, str(e)))
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for name, result, error in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
        if error:
            print(f"   Error: {error}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready to use.")
        print("\n📝 Next Steps:")
        print("   1. Review generated templates: stock_template.csv, option_template.csv")
        print("   2. Check exported positions: exported_stocks.csv, exported_options.csv")
        print("   3. Start the API: python -m uvicorn src.api.main:app --reload")
        print("   4. Open frontend and test UI")
    else:
        print("\n⚠️  Some tests failed. Please review errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

