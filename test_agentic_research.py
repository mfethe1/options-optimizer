"""
Test Agentic Research System with Real-Time Pricing
Demonstrates the complete workflow with the OptionsResearchAgent
"""
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.position_manager import PositionManager
from src.data.csv_position_service import CSVPositionService
from src.data.position_enrichment_service import PositionEnrichmentService
from src.agents.position_context_service import PositionContextService
from src.agents.options_research_agent import OptionsResearchAgent


def test_agentic_research():
    """Test the complete agentic research workflow"""
    print("="*80)
    print("AGENTIC OPTIONS RESEARCH SYSTEM TEST")
    print("="*80)
    
    # Initialize services
    pm = PositionManager(storage_file="data/test_agentic_positions.json")
    csv_service = CSVPositionService(pm)
    enrichment_service = PositionEnrichmentService(pm)
    context_service = PositionContextService(
        position_manager=pm,
        enrichment_service=enrichment_service
    )
    research_agent = OptionsResearchAgent(
        position_manager=pm,
        enrichment_service=enrichment_service,
        context_service=context_service
    )
    
    conversation_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Step 1: Upload Chase CSV
    print("\n" + "="*80)
    print("STEP 1: UPLOAD CHASE CSV")
    print("="*80)
    
    with open('data/examples/positions.csv', 'r', encoding='utf-8') as f:
        chase_csv_content = f.read()
    
    print("\n📂 Uploading Chase CSV...")
    results = csv_service.import_option_positions(
        chase_csv_content,
        replace_existing=True,
        chase_format=True
    )
    
    print(f"✅ Uploaded {results['success']} positions")
    
    # Step 2: Analyze Individual Positions
    print("\n" + "="*80)
    print("STEP 2: ANALYZE INDIVIDUAL POSITIONS")
    print("="*80)
    
    for i, pos_id in enumerate(results['position_ids'], 1):
        print(f"\n{'='*80}")
        print(f"POSITION {i}/{len(results['position_ids'])}")
        print(f"{'='*80}")
        
        # Analyze position with research agent
        analysis = research_agent.analyze_position(pos_id, conversation_id)
        
        # Display analysis
        print(f"\n📊 {analysis['symbol']} ${analysis['strike']} {analysis['option_type'].upper()}")
        print(f"   Expiration: {analysis['expiration_date']} ({analysis['days_to_expiry']} days)")
        print(f"   Quantity: {analysis['quantity']} contracts")
        
        print(f"\n💰 Pricing:")
        print(f"   Current Price: ${analysis['current_price']:.2f}")
        print(f"   Underlying: ${analysis['underlying_price']:.2f}")
        print(f"   Premium Paid: ${analysis['premium_paid']:.2f}")
        
        print(f"\n📈 P&L:")
        print(f"   Unrealized: ${analysis['unrealized_pnl']:.2f} ({analysis['unrealized_pnl_pct']:.2f}%)")
        
        if analysis.get('delta'):
            print(f"\n🔢 Greeks:")
            print(f"   Delta: {analysis['delta']:.3f}")
            print(f"   Gamma: {analysis.get('gamma', 0):.3f}")
            print(f"   Theta: {analysis['theta']:.3f}")
            print(f"   Vega: {analysis.get('vega', 0):.3f}")
        
        print(f"\n⚠️  Risk:")
        print(f"   Risk Level: {analysis['risk_level']}")
        if analysis.get('probability_of_profit'):
            print(f"   Probability of Profit: {analysis['probability_of_profit']:.1f}%")
        if analysis.get('break_even_price'):
            print(f"   Break Even: ${analysis['break_even_price']:.2f}")
        
        # Display recommendation
        rec = analysis['recommendation']
        print(f"\n🎯 RECOMMENDATION:")
        print(f"   Action: {rec['action']}")
        print(f"   Urgency: {rec['urgency']}")
        print(f"   Reasoning: {rec['reasoning']}")
        
        print(f"\n💡 Suggested Adjustments:")
        for adj in rec['suggested_adjustments']:
            print(f"   • {adj}")
        
        # Display market context
        if analysis.get('market_context'):
            mc = analysis['market_context']
            if mc.get('day_change_pct'):
                print(f"\n📊 Market Context:")
                print(f"   Day Change: {mc['day_change_pct']:.2f}%")
                if mc.get('next_earnings'):
                    print(f"   Next Earnings: {mc['next_earnings']}")
    
    # Step 3: Portfolio-Level Analysis
    print("\n" + "="*80)
    print("STEP 3: PORTFOLIO-LEVEL ANALYSIS")
    print("="*80)
    
    portfolio_analysis = research_agent.analyze_portfolio(conversation_id)
    
    print(f"\n📊 Portfolio Overview:")
    print(f"   Total Positions: {portfolio_analysis['total_positions']}")
    print(f"   Total Value: ${portfolio_analysis['total_value']:,.2f}")
    print(f"   Total P&L: ${portfolio_analysis['total_pnl']:,.2f} ({portfolio_analysis['total_pnl_pct']:.2f}%)")
    print(f"   Winning: {portfolio_analysis['winning_positions']}")
    print(f"   Losing: {portfolio_analysis['losing_positions']}")
    
    print(f"\n🎯 Portfolio Recommendations:")
    for rec in portfolio_analysis['recommendations']:
        print(f"   {rec}")
    
    # Step 4: Simulate Agent Conversation
    print("\n" + "="*80)
    print("STEP 4: SIMULATE AGENT CONVERSATION")
    print("="*80)
    
    # Simulate user asking about specific position
    print("\n👤 User: 'What should I do with my AMZN position?'")
    
    # Find AMZN position
    amzn_pos = None
    for pos_id in results['position_ids']:
        pos = pm.get_option_position(pos_id)
        if pos.symbol == 'AMZN':
            amzn_pos = pos
            break
    
    if amzn_pos:
        # Get updated pricing
        print("\n🤖 Agent: 'Let me check the latest pricing for your AMZN position...'")
        updated_pricing = research_agent.get_updated_pricing(amzn_pos.position_id)
        
        print(f"\n📊 Updated Pricing (as of {updated_pricing['updated_at']}):")
        print(f"   Current Price: ${updated_pricing['current_price']:.2f}")
        print(f"   Underlying: ${updated_pricing['underlying_price']:.2f}")
        print(f"   P&L: ${updated_pricing['unrealized_pnl']:.2f} ({updated_pricing['unrealized_pnl_pct']:.2f}%)")
        
        # Get fresh analysis
        analysis = research_agent.analyze_position(amzn_pos.position_id, conversation_id)
        rec = analysis['recommendation']
        
        print(f"\n🤖 Agent: 'Based on the current data:'")
        print(f"   {rec['reasoning']}")
        print(f"\n   My recommendation: {rec['action']}")
        print(f"   Urgency: {rec['urgency']}")
        print(f"\n   Here's what I suggest:")
        for adj in rec['suggested_adjustments']:
            print(f"   • {adj}")
    
    # Step 5: Demonstrate Real-Time Updates
    print("\n" + "="*80)
    print("STEP 5: REAL-TIME PRICING UPDATES")
    print("="*80)
    
    print("\n👤 User: 'Can you refresh all my positions with the latest prices?'")
    print("\n🤖 Agent: 'Refreshing all positions with real-time data...'")
    
    updated_positions = []
    for pos_id in results['position_ids']:
        updated = research_agent.get_updated_pricing(pos_id)
        updated_positions.append(updated)
    
    print(f"\n✅ Updated {len(updated_positions)} positions:")
    for upd in updated_positions:
        print(f"   {upd['symbol']}: ${upd['current_price']:.2f} (P&L: {upd['unrealized_pnl_pct']:.2f}%)")
    
    # Step 6: Review Conversation Memory
    print("\n" + "="*80)
    print("STEP 6: CONVERSATION MEMORY")
    print("="*80)

    # Get conversation history
    history = context_service.conversation_memory.get_conversation(conversation_id)

    print(f"\n📝 Conversation History ({len(history)} interactions):")
    for i, interaction in enumerate(history[-5:], 1):  # Show last 5
        print(f"\n   Interaction {i}:")
        print(f"   Role: {interaction['role']}")
        print(f"   Content: {interaction['content'][:80]}...")
        print(f"   Timestamp: {interaction['timestamp']}")
    
    # Summary
    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Uploaded {results['success']} positions from Chase CSV")
    print(f"   ✅ Analyzed {len(results['position_ids'])} positions individually")
    print(f"   ✅ Generated portfolio-level recommendations")
    print(f"   ✅ Demonstrated agent conversation with real-time updates")
    print(f"   ✅ Logged {len(history)} interactions to conversation memory")
    
    print(f"\n📁 Data saved to:")
    print(f"   - Positions: data/test_agentic_positions.json")
    print(f"   - Conversation: data/conversation_memory.json")
    
    print(f"\n🎯 Key Features Demonstrated:")
    print(f"   ✅ Chase CSV direct upload with auto-conversion")
    print(f"   ✅ Real-time pricing updates via yfinance")
    print(f"   ✅ Intelligent position analysis with Greeks")
    print(f"   ✅ Actionable recommendations with urgency levels")
    print(f"   ✅ Portfolio-level risk assessment")
    print(f"   ✅ Agent conversation with context awareness")
    print(f"   ✅ Multi-session conversation memory")
    
    return {
        'upload_results': results,
        'portfolio_analysis': portfolio_analysis,
        'conversation_id': conversation_id,
        'interactions': len(history)
    }


if __name__ == "__main__":
    results = test_agentic_research()
    
    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    print(json.dumps({
        'positions_uploaded': results['upload_results']['success'],
        'portfolio_value': f"${results['portfolio_analysis']['total_value']:,.2f}",
        'portfolio_pnl': f"${results['portfolio_analysis']['total_pnl']:,.2f}",
        'portfolio_pnl_pct': f"{results['portfolio_analysis']['total_pnl_pct']:.2f}%",
        'conversation_id': results['conversation_id'],
        'total_interactions': results['interactions']
    }, indent=2))

