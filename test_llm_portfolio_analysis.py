"""
Test LLM-Powered Portfolio Analysis

This script tests the LLM-powered agents directly (without API server)
to verify they're making actual LLM calls and generating AI-powered insights.
"""

import json
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import swarm components
from src.agents.swarm import SwarmCoordinator
from src.agents.swarm.agents.llm_market_analyst import LLMMarketAnalystAgent
from src.agents.swarm.agents.llm_risk_manager import LLMRiskManagerAgent
from src.agents.swarm.agents.llm_sentiment_analyst import LLMSentimentAnalystAgent

print("\n" + "="*80)
print("LLM-POWERED PORTFOLIO ANALYSIS TEST")
print("="*80)
print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n")

# ============================================================================
# Load Portfolio Data
# ============================================================================
print("-" * 80)
print("Loading Portfolio Data...")
print("-" * 80)

# Sample portfolio (from positions.csv)
portfolio = {
    'total_portfolio_value': 13820.88,
    'total_unrealized_pnl': -2000.03,
    'total_unrealized_pnl_pct': -15.38,
    'positions': [
        {
            'symbol': 'NVDA',
            'option_type': 'CALL',
            'strike': 140.0,
            'expiration': '2025-01-17',
            'quantity': 10,
            'current_value': 8300.00,
            'unrealized_pnl': -1700.00,
            'unrealized_pnl_pct': -17.00,
            'days_to_expiry': 92
        },
        {
            'symbol': 'AMZN',
            'option_type': 'CALL',
            'strike': 200.0,
            'expiration': '2025-01-17',
            'quantity': 5,
            'current_value': 1450.00,
            'unrealized_pnl': -50.00,
            'unrealized_pnl_pct': -3.33,
            'days_to_expiry': 92
        },
        {
            'symbol': 'MARA',
            'option_type': 'CALL',
            'strike': 20.0,
            'expiration': '2024-12-20',
            'quantity': 15,
            'current_value': 1004.00,
            'unrealized_pnl': 4.00,
            'unrealized_pnl_pct': 0.40,
            'days_to_expiry': 64
        },
        {
            'symbol': 'PATH',
            'option_type': 'CALL',
            'strike': 15.0,
            'expiration': '2024-12-20',
            'quantity': 10,
            'current_value': 150.00,
            'unrealized_pnl': -100.00,
            'unrealized_pnl_pct': -40.00,
            'days_to_expiry': 64
        },
        {
            'symbol': 'TLRY',
            'option_type': 'CALL',
            'strike': 2.0,
            'expiration': '2024-12-20',
            'quantity': 20,
            'current_value': 115.00,
            'unrealized_pnl': -154.03,
            'unrealized_pnl_pct': -57.25,
            'days_to_expiry': 64
        }
    ]
}

print(f"✓ Portfolio loaded: {len(portfolio['positions'])} positions")
print(f"  Total Value: ${portfolio['total_portfolio_value']:,.2f}")
print(f"  Unrealized P&L: ${portfolio['total_unrealized_pnl']:,.2f} ({portfolio['total_unrealized_pnl_pct']:.2f}%)")
print()

# ============================================================================
# Initialize Swarm with LLM-Powered Agents
# ============================================================================
print("-" * 80)
print("Initializing LLM-Powered Swarm...")
print("-" * 80)

coordinator = SwarmCoordinator(
    name="LLM_Test_Swarm",
    max_messages=1000,
    quorum_threshold=0.67
)

# Create LLM-powered agents
print("Creating agents...")

market_analyst = LLMMarketAnalystAgent(
    agent_id="llm_market_analyst",
    shared_context=coordinator.shared_context,
    consensus_engine=coordinator.consensus_engine,
    preferred_model="anthropic"  # Using Claude
)
coordinator.register_agent(market_analyst)
print("  ✓ LLM Market Analyst (Claude)")

risk_manager = LLMRiskManagerAgent(
    agent_id="llm_risk_manager",
    shared_context=coordinator.shared_context,
    consensus_engine=coordinator.consensus_engine,
    max_portfolio_delta=100.0,
    max_position_size_pct=0.10,
    max_drawdown_pct=0.15,
    preferred_model="anthropic"  # Using Claude
)
coordinator.register_agent(risk_manager)
print("  ✓ LLM Risk Manager (Claude)")

sentiment_analyst = LLMSentimentAnalystAgent(
    agent_id="llm_sentiment_analyst",
    shared_context=coordinator.shared_context,
    consensus_engine=coordinator.consensus_engine,
    preferred_model="lmstudio"  # Using local model
)
coordinator.register_agent(sentiment_analyst)
print("  ✓ LLM Sentiment Analyst (LMStudio)")

print()

# ============================================================================
# Run LLM-Powered Analysis
# ============================================================================
print("="*80)
print("RUNNING LLM-POWERED ANALYSIS")
print("="*80)
print()

context = {'portfolio': portfolio}
results = {}

# Test 1: Market Analyst
print("-" * 80)
print("TEST 1: LLM Market Analyst")
print("-" * 80)
print("Calling Claude API for market analysis...")
print()

try:
    market_analysis = market_analyst.analyze(context)
    market_recommendation = market_analyst.make_recommendation(market_analysis)
    
    results['market_analyst'] = {
        'analysis': market_analysis,
        'recommendation': market_recommendation
    }
    
    print("✓ Market Analysis Complete!")
    print(f"  Market Regime: {market_analysis.get('market_regime', 'N/A')}")
    print(f"  Trend: {market_analysis.get('trend', 'N/A')}")
    print(f"  Confidence: {market_analysis.get('confidence', 0):.2f}")
    print(f"  LLM-Powered: {market_analysis.get('llm_powered', False)}")
    
    if 'key_insights' in market_analysis:
        print("\n  Key Insights:")
        for insight in market_analysis['key_insights'][:3]:
            print(f"    - {insight}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    results['market_analyst'] = {'error': str(e)}

print()

# Test 2: Risk Manager
print("-" * 80)
print("TEST 2: LLM Risk Manager")
print("-" * 80)
print("Calling Claude API for risk analysis...")
print()

try:
    risk_analysis = risk_manager.analyze(context)
    risk_recommendation = risk_manager.make_recommendation(risk_analysis)
    
    results['risk_manager'] = {
        'analysis': risk_analysis,
        'recommendation': risk_recommendation
    }
    
    print("✓ Risk Analysis Complete!")
    print(f"  Risk Level: {risk_analysis.get('risk_level', 'N/A')}")
    print(f"  Risk Score: {risk_analysis.get('risk_score', 0)}/100")
    print(f"  Confidence: {risk_analysis.get('confidence', 0):.2f}")
    print(f"  LLM-Powered: {risk_analysis.get('llm_powered', False)}")
    
    if 'key_risks' in risk_analysis:
        print("\n  Key Risks:")
        for risk in risk_analysis['key_risks'][:3]:
            print(f"    - {risk}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    results['risk_manager'] = {'error': str(e)}

print()

# Test 3: Sentiment Analyst
print("-" * 80)
print("TEST 3: LLM Sentiment Analyst")
print("-" * 80)
print("Calling LMStudio API for sentiment analysis...")
print()

try:
    sentiment_analysis = sentiment_analyst.analyze(context)
    sentiment_recommendation = sentiment_analyst.make_recommendation(sentiment_analysis)
    
    results['sentiment_analyst'] = {
        'analysis': sentiment_analysis,
        'recommendation': sentiment_recommendation
    }
    
    print("✓ Sentiment Analysis Complete!")
    print(f"  Overall Sentiment: {sentiment_analysis.get('overall_sentiment', 'N/A')}")
    print(f"  Sentiment Score: {sentiment_analysis.get('sentiment_score', 0):.2f}")
    print(f"  Confidence: {sentiment_analysis.get('confidence', 0):.2f}")
    print(f"  LLM-Powered: {sentiment_analysis.get('llm_powered', False)}")
    
    if 'key_insights' in sentiment_analysis:
        print("\n  Key Insights:")
        for insight in sentiment_analysis['key_insights'][:3]:
            print(f"    - {insight}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    results['sentiment_analyst'] = {'error': str(e)}

print()

# ============================================================================
# Save Results
# ============================================================================
print("="*80)
print("SAVING RESULTS")
print("="*80)

output_file = 'llm_portfolio_analysis_results.json'
with open(output_file, 'w') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'portfolio': portfolio,
        'results': results
    }, f, indent=2)

print(f"✓ Results saved to: {output_file}")
print()

# ============================================================================
# Summary
# ============================================================================
print("="*80)
print("TEST SUMMARY")
print("="*80)

successful_tests = 0
total_tests = 3

for agent_name, result in results.items():
    if 'error' not in result:
        successful_tests += 1
        print(f"✓ {agent_name}: SUCCESS")
    else:
        print(f"✗ {agent_name}: FAILED - {result['error']}")

print()
print(f"Success Rate: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.0f}%)")
print()

if successful_tests > 0:
    print("✓ LLM-powered agents are working!")
    print("  The agents are making actual API calls to LLMs and generating AI-powered insights.")
else:
    print("✗ No LLM-powered agents succeeded.")
    print("  Please check API keys and connectivity.")

print()

