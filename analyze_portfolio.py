"""
Analyze Portfolio Using Multi-Agent Swarm System

This script analyzes the example portfolio holdings and generates
comprehensive recommendations using the multi-agent swarm system.
"""

import requests
import json
from datetime import datetime

# API Configuration
API_BASE = "http://localhost:8000"
USERNAME = "trader"
PASSWORD = "trader123"

print("\n" + "="*80)
print("PORTFOLIO ANALYSIS USING MULTI-AGENT SWARM SYSTEM")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n")

# ============================================================================
# STEP 1: Authenticate
# ============================================================================
print("-" * 80)
print("STEP 1: Authenticating...")
print("-" * 80)

auth_response = requests.post(
    f"{API_BASE}/api/auth/login",
    data={
        "username": USERNAME,
        "password": PASSWORD
    }
)

if auth_response.status_code == 200:
    token = auth_response.json()["access_token"]
    print(f"✓ Authentication successful")
    print(f"  Token: {token[:20]}...")
else:
    print(f"✗ Authentication failed: {auth_response.status_code}")
    print(f"  Response: {auth_response.text}")
    exit(1)

print()

# ============================================================================
# STEP 2: Prepare Portfolio Data
# ============================================================================
print("-" * 80)
print("STEP 2: Preparing Portfolio Data...")
print("-" * 80)

# Portfolio summary from positions.csv
portfolio_data = {
    "positions": [
        {
            "ticker": "NVDA CALL 01/15/27 $175.00",
            "asset_class": "Equity Options",
            "quantity": 2,
            "current_value": 8260,
            "cost_basis": 9601.34,
            "unrealized_pnl": -1341.34,
            "unrealized_pnl_pct": -13.97,
            "expiration": "01/15/2027",
            "strike": 175.00,
            "option_type": "CALL",
            "current_price": 41.30,
            "price_change_pct": -1.43
        },
        {
            "ticker": "AMZN CALL 12/19/25 $225.00",
            "asset_class": "Equity Options",
            "quantity": 1,
            "current_value": 1046,
            "cost_basis": 1319.67,
            "unrealized_pnl": -273.67,
            "unrealized_pnl_pct": -20.74,
            "expiration": "12/19/2025",
            "strike": 225.00,
            "option_type": "CALL",
            "current_price": 10.46,
            "price_change_pct": -2.70
        },
        {
            "ticker": "MARA CALL 09/18/26 $18.00",
            "asset_class": "Equity Options",
            "quantity": 1,
            "current_value": 966,
            "cost_basis": 985.67,
            "unrealized_pnl": -19.67,
            "unrealized_pnl_pct": -2.00,
            "expiration": "09/18/2026",
            "strike": 18.00,
            "option_type": "CALL",
            "current_price": 9.66,
            "price_change_pct": 1.42
        },
        {
            "ticker": "PATH CALL 12/19/25 $19.00",
            "asset_class": "Equity Options",
            "quantity": 3,
            "current_value": 522,
            "cost_basis": 869.01,
            "unrealized_pnl": -347.01,
            "unrealized_pnl_pct": -39.93,
            "expiration": "12/19/2025",
            "strike": 19.00,
            "option_type": "CALL",
            "current_price": 1.74,
            "price_change_pct": 2.96
        },
        {
            "ticker": "TLRY CALL 12/19/25 $1.50",
            "asset_class": "Equity Options",
            "quantity": 5,
            "current_value": 225,
            "cost_basis": 243.34,
            "unrealized_pnl": -18.34,
            "unrealized_pnl_pct": -7.54,
            "expiration": "12/19/2025",
            "strike": 1.50,
            "option_type": "CALL",
            "current_price": 0.45,
            "price_change_pct": 18.42
        }
    ],
    "cash": {
        "usd": 2684.65,
        "chase_sweep": 116.23
    },
    "total_portfolio_value": 13820.88,
    "total_options_value": 11019.00,
    "total_cash": 2800.88,
    "total_unrealized_pnl": -2000.03,
    "total_unrealized_pnl_pct": -15.38
}

print(f"✓ Portfolio data prepared")
print(f"  Total Positions: {len(portfolio_data['positions'])}")
print(f"  Total Value: ${portfolio_data['total_portfolio_value']:,.2f}")
print(f"  Options Value: ${portfolio_data['total_options_value']:,.2f}")
print(f"  Cash: ${portfolio_data['total_cash']:,.2f}")
print(f"  Unrealized P&L: ${portfolio_data['total_unrealized_pnl']:,.2f} ({portfolio_data['total_unrealized_pnl_pct']:.2f}%)")
print()

# ============================================================================
# STEP 3: Run Swarm Analysis
# ============================================================================
print("-" * 80)
print("STEP 3: Running Multi-Agent Swarm Analysis...")
print("-" * 80)
print("This may take 30-60 seconds as 8 specialized agents analyze the portfolio...")
print()

analysis_request = {
    "portfolio_data": portfolio_data,
    "analysis_type": "comprehensive",
    "include_recommendations": True,
    "consensus_method": "weighted",
    "context": """
    Analyze this options portfolio with the following considerations:
    
    1. RISK ASSESSMENT:
       - All positions are currently showing unrealized losses
       - Total portfolio down -15.38%
       - PATH position down -39.93% (most concerning)
       - AMZN position down -20.74%
       
    2. TIME DECAY CONCERNS:
       - 3 positions expire in 2 months (12/19/2025): AMZN, PATH, TLRY
       - These positions are losing value rapidly due to theta decay
       
    3. POSITION ANALYSIS:
       - NVDA: Long-dated (Jan 2027), down -13.97%, still has time
       - AMZN: Expires soon, down -20.74%, needs decision
       - MARA: Mid-term (Sep 2026), down -2%, relatively stable
       - PATH: Expires soon, down -39.93%, CRITICAL
       - TLRY: Expires soon, down -7.54%, small position
       
    4. RECOMMENDATIONS NEEDED:
       - Should I hold, roll, or close each position?
       - What hedging strategies should I consider?
       - Are there better opportunities to redeploy capital?
       - What's my overall risk exposure?
       - How can I improve this portfolio?
    
    Please provide specific, actionable recommendations with reasoning.
    """
}

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

try:
    swarm_response = requests.post(
        f"{API_BASE}/api/swarm/analyze",
        headers=headers,
        json=analysis_request,
        timeout=120  # 2 minute timeout
    )
    
    if swarm_response.status_code == 200:
        result = swarm_response.json()
        print(f"✓ Swarm analysis complete!")
        print()
        
        # ================================================================
        # STEP 4: Display Results
        # ================================================================
        print("=" * 80)
        print("SWARM ANALYSIS RESULTS")
        print("=" * 80)
        print()
        
        # Consensus recommendation
        if "consensus_recommendation" in result:
            print("-" * 80)
            print("CONSENSUS RECOMMENDATION")
            print("-" * 80)
            print(result["consensus_recommendation"])
            print()
        
        # Individual agent insights
        if "agent_insights" in result:
            print("-" * 80)
            print("INDIVIDUAL AGENT INSIGHTS")
            print("-" * 80)
            for agent_name, insight in result["agent_insights"].items():
                print(f"\n{agent_name}:")
                print(f"  {insight}")
            print()
        
        # Risk assessment
        if "risk_assessment" in result:
            print("-" * 80)
            print("RISK ASSESSMENT")
            print("-" * 80)
            print(json.dumps(result["risk_assessment"], indent=2))
            print()
        
        # Recommendations
        if "recommendations" in result:
            print("-" * 80)
            print("SPECIFIC RECOMMENDATIONS")
            print("-" * 80)
            for i, rec in enumerate(result["recommendations"], 1):
                print(f"\n{i}. {rec}")
            print()
        
        # Confidence scores
        if "confidence_scores" in result:
            print("-" * 80)
            print("CONFIDENCE SCORES")
            print("-" * 80)
            print(json.dumps(result["confidence_scores"], indent=2))
            print()
        
        # Save full results to file
        with open("portfolio_analysis_results.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print("-" * 80)
        print("✓ Full results saved to: portfolio_analysis_results.json")
        print("-" * 80)
        
    elif swarm_response.status_code == 429:
        print(f"✗ Rate limit exceeded. Please wait and try again.")
        print(f"  Response: {swarm_response.text}")
    else:
        print(f"✗ Swarm analysis failed: {swarm_response.status_code}")
        print(f"  Response: {swarm_response.text}")
        
except requests.exceptions.Timeout:
    print("✗ Request timed out. The swarm analysis is taking longer than expected.")
    print("  This is normal for complex portfolios. Try increasing the timeout.")
except Exception as e:
    print(f"✗ Error: {str(e)}")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

