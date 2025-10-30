"""
Test Chase CSV Upload and Agentic Research System
Tests the complete workflow: Upload → Enrich → Research → Recommendations
"""
import sys
import os
import json
import asyncio
from datetime import datetime
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.position_manager import PositionManager
from src.data.csv_position_service import CSVPositionService
from src.data.position_enrichment_service import PositionEnrichmentService
from src.agents.position_context_service import PositionContextService
from src.agents.options_research_agent import OptionsResearchAgent


@pytest.mark.asyncio
async def test_upload_and_research():
    """Test complete workflow: Upload Chase CSV → Research → Recommendations"""
    print("="*80)
    print("CHASE CSV UPLOAD & AGENTIC RESEARCH TEST")
    print("="*80)
    
    # Initialize services
    pm = PositionManager(storage_file="data/test_research_positions.json")
    csv_service = CSVPositionService(pm)
    enrichment_service = PositionEnrichmentService(pm)
    context_service = PositionContextService(
        position_manager=pm,
        enrichment_service=enrichment_service
    )
    
    # Step 1: Upload Chase CSV
    print("\n" + "="*80)
    print("STEP 1: UPLOAD CHASE CSV")
    print("="*80)
    
    with open('data/examples/positions.csv', 'r', encoding='utf-8') as f:
        chase_csv_content = f.read()
    
    print("\n📂 Uploading Chase CSV with chase_format=True...")
    results = csv_service.import_option_positions(
        chase_csv_content,
        replace_existing=True,
        chase_format=True
    )
    
    print(f"\n✅ Upload Results:")
    print(f"   Success: {results['success']}")
    print(f"   Failed: {results['failed']}")
    print(f"   Position IDs: {len(results['position_ids'])}")
    
    if results['chase_conversion']:
        conv = results['chase_conversion']
        print(f"\n🔄 Chase Conversion:")
        print(f"   Options converted: {conv['options_converted']}")
        print(f"   Cash skipped: {conv['cash_skipped']}")
    
    # Step 2: Enrich Positions
    print("\n" + "="*80)
    print("STEP 2: ENRICH POSITIONS WITH REAL-TIME DATA")
    print("="*80)
    
    print("\n💰 Fetching real-time prices and calculating Greeks...")
    enrichment_results = enrichment_service.enrich_all_positions()
    
    print(f"\n✅ Enrichment Results:")
    print(f"   Stocks enriched: {enrichment_results.get('stocks_enriched', 0)}")
    print(f"   Options enriched: {enrichment_results.get('options_enriched', 0)}")
    
    # Display enriched positions
    print("\n📊 Enriched Positions:")
    for pos_id in results['position_ids']:
        pos = pm.get_option_position(pos_id)
        print(f"\n   {pos.symbol} ${pos.strike} {pos.option_type.upper()} (exp: {pos.expiration_date})")
        print(f"   ├─ Quantity: {pos.quantity} contracts")
        print(f"   ├─ Premium Paid: ${pos.premium_paid:.2f}")
        
        if pos.current_price:
            print(f"   ├─ Current Price: ${pos.current_price:.2f}")
            print(f"   ├─ Unrealized P&L: ${pos.unrealized_pnl:.2f} ({pos.unrealized_pnl_pct:.2f}%)")
            print(f"   ├─ Days to Expiry: {pos.days_to_expiry()}")
            
            if pos.delta:
                print(f"   ├─ Delta: {pos.delta:.3f}")
                print(f"   ├─ Theta: {pos.theta:.3f}")
                print(f"   └─ Risk Level: {pos.get_risk_level()}")
    
    # Step 3: Create Agent Context
    print("\n" + "="*80)
    print("STEP 3: CREATE AGENT CONTEXT FOR RESEARCH")
    print("="*80)
    
    conversation_id = f"test_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n🤖 Creating agent context (conversation_id: {conversation_id})...")
    
    # Get portfolio summary for agents
    summary_context = context_service.get_position_summary_for_agent()
    print("\n📋 Portfolio Summary for Agents:")
    print(summary_context[:500] + "..." if len(summary_context) > 500 else summary_context)
    
    # Step 4: Analyze Each Position
    print("\n" + "="*80)
    print("STEP 4: AGENTIC RESEARCH & RECOMMENDATIONS")
    print("="*80)
    
    recommendations = []
    
    for pos_id in results['position_ids']:
        pos = pm.get_option_position(pos_id)
        
        print(f"\n{'='*80}")
        print(f"ANALYZING: {pos.symbol} ${pos.strike} {pos.option_type.upper()}")
        print(f"{'='*80}")
        
        # Create detailed context for this position
        position_context = context_service.create_agent_context(
            conversation_id=conversation_id,
            include_summary=True,
            include_positions=True,
            symbol=pos.symbol
        )
        
        # Simulate agent analysis
        analysis = analyze_position_with_agents(pos, position_context)
        recommendations.append(analysis)
        
        # Display analysis
        print(f"\n📊 Position Analysis:")
        print(f"   Symbol: {analysis['symbol']}")
        print(f"   Current Status: {analysis['status']}")
        print(f"   P&L: ${analysis['pnl']:.2f} ({analysis['pnl_pct']:.2f}%)")
        print(f"   Days to Expiry: {analysis['days_to_expiry']}")
        
        print(f"\n🎯 Recommendation: {analysis['recommendation']}")
        print(f"   Action: {analysis['action']}")
        print(f"   Reasoning: {analysis['reasoning']}")
        
        if analysis['suggested_adjustments']:
            print(f"\n💡 Suggested Adjustments:")
            for adj in analysis['suggested_adjustments']:
                print(f"   • {adj}")
        
        # Log interaction
        context_service.log_agent_interaction(
            conversation_id=conversation_id,
            user_query=f"Analyze {pos.symbol} position",
            agent_response=json.dumps(analysis, indent=2),
            positions_accessed=[pos_id]
        )
    
    # Step 5: Portfolio-Level Recommendations
    print("\n" + "="*80)
    print("STEP 5: PORTFOLIO-LEVEL RECOMMENDATIONS")
    print("="*80)
    
    portfolio_analysis = analyze_portfolio(recommendations, pm)
    
    print(f"\n📊 Portfolio Overview:")
    print(f"   Total Positions: {portfolio_analysis['total_positions']}")
    print(f"   Total Value: ${portfolio_analysis['total_value']:.2f}")
    print(f"   Total P&L: ${portfolio_analysis['total_pnl']:.2f} ({portfolio_analysis['total_pnl_pct']:.2f}%)")
    print(f"   Winning Positions: {portfolio_analysis['winning_positions']}")
    print(f"   Losing Positions: {portfolio_analysis['losing_positions']}")
    
    print(f"\n🎯 Portfolio Recommendations:")
    for rec in portfolio_analysis['recommendations']:
        print(f"   • {rec}")
    
    print(f"\n⚠️  Risk Alerts:")
    for alert in portfolio_analysis['risk_alerts']:
        print(f"   • {alert}")
    
    # Step 6: Get Updated Pricing After Recommendations
    print("\n" + "="*80)
    print("STEP 6: REFRESH PRICING AFTER RECOMMENDATIONS")
    print("="*80)
    
    print("\n🔄 Refreshing all positions with latest pricing...")
    enrichment_service.enrich_all_positions()
    
    print("\n✅ Updated Pricing:")
    for pos_id in results['position_ids']:
        pos = pm.get_option_position(pos_id)
        print(f"   {pos.symbol} ${pos.strike} {pos.option_type.upper()}: ${pos.current_price:.2f}")
    
    # Summary
    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Uploaded {results['success']} positions from Chase CSV")
    print(f"   ✅ Enriched all positions with real-time data")
    print(f"   ✅ Generated {len(recommendations)} position-level recommendations")
    print(f"   ✅ Generated portfolio-level analysis")
    print(f"   ✅ Refreshed pricing after recommendations")
    
    print(f"\n📁 Data saved to:")
    print(f"   - Positions: data/test_research_positions.json")
    print(f"   - Conversation: data/conversation_memory.json")
    
    return {
        'upload_results': results,
        'enrichment_results': enrichment_results,
        'recommendations': recommendations,
        'portfolio_analysis': portfolio_analysis
    }


def analyze_position_with_agents(position, context):
    """
    Simulate agent analysis of a position
    In production, this would call the actual multi-agent system
    """
    # Calculate key metrics
    pnl = position.unrealized_pnl or 0
    pnl_pct = position.unrealized_pnl_pct or 0
    days_to_expiry = position.days_to_expiry()
    
    # Determine status
    if pnl_pct > 20:
        status = "STRONG_PROFIT"
    elif pnl_pct > 0:
        status = "PROFITABLE"
    elif pnl_pct > -10:
        status = "SLIGHT_LOSS"
    else:
        status = "SIGNIFICANT_LOSS"
    
    # Generate recommendation based on metrics
    if status == "STRONG_PROFIT" and days_to_expiry < 30:
        recommendation = "TAKE_PROFIT"
        action = "Consider closing position to lock in gains"
        reasoning = f"Position up {pnl_pct:.1f}% with only {days_to_expiry} days left. Time decay accelerating."
        adjustments = [
            "Close 50% of position now",
            "Set trailing stop at 15% below current price",
            "Let remaining 50% ride until 7 days before expiry"
        ]
    elif status == "SIGNIFICANT_LOSS" and days_to_expiry < 60:
        recommendation = "CUT_LOSS"
        action = "Close position to prevent further losses"
        reasoning = f"Position down {abs(pnl_pct):.1f}% with {days_to_expiry} days left. Limited time for recovery."
        adjustments = [
            "Close entire position",
            "Consider tax loss harvesting",
            "Reassess thesis before re-entering"
        ]
    elif days_to_expiry > 180:
        recommendation = "HOLD"
        action = "Monitor position, plenty of time left"
        reasoning = f"{days_to_expiry} days to expiry provides time for thesis to play out."
        adjustments = [
            "Set alert for 90 days before expiry",
            "Review quarterly earnings",
            "Monitor delta and adjust if needed"
        ]
    else:
        recommendation = "MONITOR"
        action = "Watch closely, decision point approaching"
        reasoning = f"{days_to_expiry} days left. Monitor for entry/exit signals."
        adjustments = [
            "Check daily for significant moves",
            "Consider rolling if still bullish",
            "Set stop loss at -25%"
        ]
    
    return {
        'symbol': position.symbol,
        'option_type': position.option_type,
        'strike': position.strike,
        'expiration': position.expiration_date,
        'status': status,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'days_to_expiry': days_to_expiry,
        'current_price': position.current_price or 0,
        'delta': position.delta,
        'theta': position.theta,
        'recommendation': recommendation,
        'action': action,
        'reasoning': reasoning,
        'suggested_adjustments': adjustments
    }


def analyze_portfolio(recommendations, position_manager):
    """Generate portfolio-level analysis and recommendations"""
    total_positions = len(recommendations)
    total_pnl = sum(r['pnl'] for r in recommendations)
    
    # Calculate total value
    total_value = 0
    for rec in recommendations:
        pos_id = f"OPT_{rec['symbol']}_{rec['option_type'].upper()}_{rec['strike']}_{rec['expiration'].replace('-', '')}"
        pos = position_manager.get_option_position(pos_id)
        if pos and pos.current_price:
            total_value += pos.current_price * pos.quantity * 100
    
    total_pnl_pct = (total_pnl / total_value * 100) if total_value > 0 else 0
    
    winning_positions = sum(1 for r in recommendations if r['pnl'] > 0)
    losing_positions = sum(1 for r in recommendations if r['pnl'] < 0)
    
    # Generate recommendations
    portfolio_recommendations = []
    risk_alerts = []
    
    # Check for concentration risk
    symbols = [r['symbol'] for r in recommendations]
    if len(set(symbols)) < len(symbols):
        risk_alerts.append("⚠️  Concentration risk: Multiple positions in same underlying")
    
    # Check for expiration clustering
    expirations = [r['expiration'] for r in recommendations]
    if len(set(expirations)) < 3:
        risk_alerts.append("⚠️  Expiration clustering: Consider spreading out expiration dates")
    
    # Check for overall P&L
    if total_pnl_pct < -15:
        portfolio_recommendations.append("Consider reducing overall options exposure")
        risk_alerts.append(f"⚠️  Portfolio down {abs(total_pnl_pct):.1f}% - review risk management")
    elif total_pnl_pct > 25:
        portfolio_recommendations.append("Consider taking profits on winning positions")
    
    # Check for near-term expirations
    near_term = sum(1 for r in recommendations if r['days_to_expiry'] < 30)
    if near_term > 0:
        portfolio_recommendations.append(f"{near_term} position(s) expiring within 30 days - review urgently")
    
    # Diversification recommendations
    if len(set(symbols)) < 3:
        portfolio_recommendations.append("Consider diversifying across more underlyings")
    
    return {
        'total_positions': total_positions,
        'total_value': total_value,
        'total_pnl': total_pnl,
        'total_pnl_pct': total_pnl_pct,
        'winning_positions': winning_positions,
        'losing_positions': losing_positions,
        'recommendations': portfolio_recommendations,
        'risk_alerts': risk_alerts
    }


if __name__ == "__main__":
    # Run async test
    results = asyncio.run(test_upload_and_research())
    
    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    print(json.dumps({
        'upload_success': results['upload_results']['success'],
        'recommendations_generated': len(results['recommendations']),
        'portfolio_total_pnl': f"${results['portfolio_analysis']['total_pnl']:.2f}",
        'portfolio_pnl_pct': f"{results['portfolio_analysis']['total_pnl_pct']:.2f}%"
    }, indent=2))

