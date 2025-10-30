"""
Test the enhanced position-by-position analysis with comprehensive stock reports
and intelligent replacement recommendations.

This demonstrates the new features:
1. Stock-specific text extraction from LLM responses
2. Structured metrics extraction
3. Comprehensive stock reports assembled from all agents
4. Replacement recommendations from RecommendationAgent
"""

import json
from pathlib import Path
from datetime import datetime

# Create mock enhanced response with comprehensive stock reports
mock_enhanced_response = {
    "consensus_decisions": {
        "overall_action": {"choice": "buy", "confidence": 0.625},
        "risk_level": {"choice": "moderate", "confidence": 0.75},
        "market_outlook": {"choice": "bullish", "confidence": 0.8125}
    },
    
    "position_analysis": [
        {
            "symbol": "NVDA",
            "asset_type": "option",
            "option_type": "call",
            "strike": 175.0,
            "expiration_date": "2027-01-15",
            "quantity": 2,
            
            "current_metrics": {
                "current_price": 43.68,
                "underlying_price": 175.50,
                "market_value": 8736.0,
                "unrealized_pnl": -865.34,
                "unrealized_pnl_pct": -9.01,
                "days_to_expiry": 455,
                "iv": 0.42
            },
            
            "greeks": {
                "delta": 0.65,
                "gamma": 0.012,
                "theta": -5.20,
                "vega": 12.50
            },
            
            "agent_insights_for_position": [
                {
                    "agent_id": "fundamental_analyst_1",
                    "agent_type": "LLMFundamentalAnalystAgent",
                    "key_insights": [
                        "Exceptional fundamentals with $29B cash",
                        "Dominant 85% market share in AI accelerators",
                        "Revenue growth of 40% YoY driven by AI demand"
                    ],
                    "recommendation": "buy",
                    "confidence": 0.85,
                    "stock_specific_analysis": """**NVIDIA Corporation (NVDA)**
- Market Cap: $1.2T | P/E: 45x | Revenue Growth: 40% YoY
- **Financial Health**: Exceptional. $29B cash, minimal debt, FCF of $15B annually
- **Earnings Quality**: High. Revenue growth driven by AI chip demand, not accounting gimmicks
- **Competitive Moat**: Dominant 85% market share in AI accelerators. CUDA ecosystem creates switching costs
- **Valuation**: Premium but justified. DCF suggests fair value $160-180. Currently at $175
- **Risks**: China exposure (25% revenue), TSMC dependency, AMD competition emerging
- **Rating**: STRONG BUY - Best-in-class fundamentals with secular AI tailwind""",
                    "stock_metrics": {
                        "pe_ratio": 45.0,
                        "revenue_growth": 40.0,
                        "market_cap": "$1.2T",
                        "rating": "STRONG BUY",
                        "valuation": "fair"
                    }
                },
                {
                    "agent_id": "market_analyst_claude",
                    "agent_type": "LLMMarketAnalystAgent",
                    "key_insights": [
                        "Strong bullish momentum with RSI at 62",
                        "Support at $165, resistance at $185",
                        "Above-average volume on up days"
                    ],
                    "recommendation": "buy",
                    "confidence": 0.75,
                    "stock_specific_analysis": """**NVDA Technical Analysis**
- Current Price: $175.50
- Support: $165 (strong), Resistance: $185 (moderate)
- RSI: 62 (healthy bullish range, not overbought)
- MACD: Positive crossover, confirming uptrend
- Volume: Above-average on up days, suggesting institutional accumulation
- Momentum: Strong bullish - outperforming Nasdaq by 8% this quarter""",
                    "stock_metrics": {
                        "momentum": "strong_bullish",
                        "support_level": 165.0,
                        "resistance_level": 185.0,
                        "rsi": 62.0
                    }
                },
                {
                    "agent_id": "risk_manager_claude",
                    "agent_type": "LLMRiskManagerAgent",
                    "key_insights": [
                        "Moderate risk with high vega exposure",
                        "Long duration provides recovery time",
                        "Position underwater but manageable"
                    ],
                    "recommendation": "hold",
                    "confidence": 0.60,
                    "stock_specific_analysis": """**NVDA CALL 01/15/27 $175 Risk Analysis**
- Delta: 0.65 | Theta: -$5.20/day | Vega: $12.50
- Time to Expiry: 455 days (GOOD - low time decay pressure)
- Moneyness: ATM (underlying $175.50)
- Risk: MODERATE - High vega exposure to IV changes
- P&L: -$865 (-9%) - underwater but time to recover
- **Action**: HOLD - Long duration provides recovery time""",
                    "stock_metrics": {
                        "risk_level": "moderate",
                        "volatility": 42.0
                    }
                }
            ],
            
            "comprehensive_stock_report": {
                "symbol": "NVDA",
                "analysis_by_category": {
                    "fundamental": {
                        "agent_count": 2,
                        "summary": "Bullish (2/2 agents recommend BUY)",
                        "agents": [
                            {
                                "agent_id": "fundamental_analyst_1",
                                "recommendation": "buy",
                                "confidence": 0.85,
                                "stock_metrics": {
                                    "pe_ratio": 45.0,
                                    "revenue_growth": 40.0,
                                    "market_cap": "$1.2T",
                                    "rating": "STRONG BUY"
                                }
                            }
                        ]
                    },
                    "market_technical": {
                        "agent_count": 3,
                        "summary": "Bullish (3/3 agents recommend BUY)",
                        "agents": [
                            {
                                "agent_id": "market_analyst_claude",
                                "recommendation": "buy",
                                "confidence": 0.75,
                                "stock_metrics": {
                                    "momentum": "strong_bullish",
                                    "rsi": 62.0
                                }
                            }
                        ]
                    },
                    "risk": {
                        "agent_count": 2,
                        "summary": "Neutral (1/2 agents recommend HOLD)",
                        "agents": [
                            {
                                "agent_id": "risk_manager_claude",
                                "recommendation": "hold",
                                "confidence": 0.60,
                                "stock_metrics": {
                                    "risk_level": "moderate",
                                    "volatility": 42.0
                                }
                            }
                        ]
                    }
                },
                "consensus_view": {
                    "total_agents": 13,
                    "buy_percentage": 76.9,
                    "hold_percentage": 15.4,
                    "sell_percentage": 7.7,
                    "consensus_recommendation": "buy",
                    "confidence": 76.9
                },
                "key_metrics": {
                    "pe_ratio": 45.0,
                    "revenue_growth": 40.0,
                    "market_cap": "$1.2T",
                    "rating": "STRONG BUY",
                    "valuation": "fair",
                    "momentum": "strong_bullish",
                    "support_level": 165.0,
                    "resistance_level": 185.0,
                    "rsi": 62.0,
                    "risk_level": "moderate",
                    "volatility": 42.0
                },
                "full_analysis_text": """# COMPREHENSIVE ANALYSIS: NVDA

## FUNDAMENTAL ANALYSIS

**LLMFundamentalAnalystAgent:**
**NVIDIA Corporation (NVDA)**
- Market Cap: $1.2T | P/E: 45x | Revenue Growth: 40% YoY
- **Financial Health**: Exceptional. $29B cash, minimal debt, FCF of $15B annually
- **Rating**: STRONG BUY - Best-in-class fundamentals with secular AI tailwind

## MARKET_TECHNICAL ANALYSIS

**LLMMarketAnalystAgent:**
**NVDA Technical Analysis**
- Current Price: $175.50
- Support: $165 (strong), Resistance: $185 (moderate)
- RSI: 62 (healthy bullish range, not overbought)
- Momentum: Strong bullish - outperforming Nasdaq by 8% this quarter

## RISK ANALYSIS

**LLMRiskManagerAgent:**
**NVDA CALL 01/15/27 $175 Risk Analysis**
- Delta: 0.65 | Theta: -$5.20/day | Vega: $12.50
- Risk: MODERATE - High vega exposure to IV changes
- **Action**: HOLD - Long duration provides recovery time
"""
            },
            
            "risk_warnings": [
                "📉 Position underwater: -9.0% loss",
                "⏰ Daily time decay: $5.20/day",
                "📊 High vega exposure: $12.50 per 1% IV change"
            ],
            
            "opportunities": [
                "⏳ Long time horizon: 455 days - low time decay pressure",
                "✅ At-the-money (delta: 0.65) - good probability of profit",
                "🎯 Strong fundamentals support recovery"
            ],
            
            "replacement_recommendations": {
                "assessment": "NVDA position is solid with long duration and strong fundamentals. However, consider taking partial profits and reallocating to diversify tech exposure.",
                "action": "HOLD",
                "stock_alternative": {
                    "symbol": "MSFT",
                    "current_price": "$415.00",
                    "quantity": "21",
                    "total_cost": "$8,715.00",
                    "probability_of_high_return": "68%",
                    "risk_level": "Moderate",
                    "key_catalyst": "Azure AI growth and enterprise adoption",
                    "why_better": "Lower volatility, similar AI exposure, better diversification"
                },
                "option_alternative": {
                    "symbol": "MSFT",
                    "type": "CALL",
                    "strike": "$420",
                    "expiration": "2026-06-19",
                    "contracts": "2",
                    "total_cost": "$8,600.00",
                    "delta": "0.62",
                    "probability_of_profit": "65%",
                    "risk_level": "Moderate",
                    "key_catalyst": "Azure AI growth and enterprise adoption",
                    "why_better": "Similar delta, lower IV, better risk/reward in current market"
                },
                "agent_id": "recommendation_agent_1"
            }
        }
    ],
    
    "swarm_health": {
        "active_agents_count": 17,
        "contributed_vs_failed": {
            "contributed": 14,
            "failed": 3,
            "success_rate": 82.35
        }
    }
}

# Save mock response
output_dir = Path("enhanced_swarm_test_output")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "enhanced_position_analysis_demo.json"
with open(output_file, 'w') as f:
    json.dump(mock_enhanced_response, f, indent=2)

print("=" * 80)
print("ENHANCED POSITION-BY-POSITION ANALYSIS - DEMONSTRATION")
print("=" * 80)
print()
print(f"✓ Mock response saved: {output_file}")
print()
print("=" * 80)
print("COMPREHENSIVE STOCK REPORT - NVDA")
print("=" * 80)
print()

nvda_position = mock_enhanced_response['position_analysis'][0]
stock_report = nvda_position['comprehensive_stock_report']

print(f"Symbol: {stock_report['symbol']}")
print()
print("CONSENSUS VIEW:")
print(f"  Total Agents: {stock_report['consensus_view']['total_agents']}")
print(f"  BUY: {stock_report['consensus_view']['buy_percentage']:.1f}%")
print(f"  HOLD: {stock_report['consensus_view']['hold_percentage']:.1f}%")
print(f"  SELL: {stock_report['consensus_view']['sell_percentage']:.1f}%")
print(f"  Consensus: {stock_report['consensus_view']['consensus_recommendation'].upper()} ({stock_report['consensus_view']['confidence']:.1f}% confidence)")
print()
print("KEY METRICS:")
for key, value in stock_report['key_metrics'].items():
    print(f"  {key}: {value}")
print()
print("ANALYSIS BY CATEGORY:")
for category, analysis in stock_report['analysis_by_category'].items():
    print(f"  {category.upper()}: {analysis['summary']}")
print()
print("=" * 80)
print("REPLACEMENT RECOMMENDATIONS")
print("=" * 80)
print()

rec = nvda_position['replacement_recommendations']
print(f"Assessment: {rec['assessment']}")
print(f"Action: {rec['action']}")
print()
if rec['stock_alternative']:
    print("STOCK ALTERNATIVE:")
    for key, value in rec['stock_alternative'].items():
        print(f"  {key}: {value}")
print()
if rec['option_alternative']:
    print("OPTION ALTERNATIVE:")
    for key, value in rec['option_alternative'].items():
        print(f"  {key}: {value}")
print()
print("=" * 80)
print("✅ DEMONSTRATION COMPLETE")
print("=" * 80)
print()
print("📁 Output file:", output_file)
print("📖 Documentation: ENHANCED_POSITION_ANALYSIS_GUIDE.md")

