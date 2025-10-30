"""
Create demonstration visualization with mock enhanced swarm analysis data.
This shows what the enhanced output looks like while we optimize the parallel execution.
"""

import json
from datetime import datetime
from pathlib import Path

# Import HTML generator
import sys
sys.path.insert(0, '.')
from test_enhanced_swarm_playwright import generate_visualization_html

OUTPUT_DIR = "enhanced_swarm_test_output"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Create comprehensive mock data that demonstrates all enhanced features
mock_data = {
    "consensus_decisions": {
        "overall_action": {
            "choice": "buy",
            "confidence": 0.625,
            "reasoning": "Strong bullish signals from 10/16 agents. NVDA and AMZN show solid fundamentals with AI tailwinds. Portfolio is moderately positioned with manageable risk. Recommend selective additions to high-conviction positions."
        },
        "risk_level": {
            "choice": "moderate",
            "confidence": 0.75
        },
        "market_outlook": {
            "choice": "bullish",
            "confidence": 0.8125
        }
    },
    
    "agent_insights": [
        {
            "agent_id": "market_analyst_claude",
            "agent_type": "LLMMarketAnalystAgent",
            "timestamp": "2025-10-17T22:45:00",
            "llm_response_text": """**Market Analysis - Technology Sector Focus**

**Current Market Regime**: Risk-On with Tech Leadership
The market is in a clear risk-on phase with technology stocks leading the advance. The Nasdaq 100 has outperformed the S&P 500 by 8% over the past quarter, driven by AI enthusiasm and strong earnings from mega-cap tech.

**Sector Rotation Analysis**:
- Technology: STRONG BUY - AI infrastructure buildout continues
- Semiconductors: BUY - NVDA dominance in AI chips, supply constraints easing
- Cloud Computing: BUY - AMZN AWS growth accelerating, enterprise migration ongoing
- Cannabis: HOLD - TLRY facing regulatory headwinds, sector consolidation

**Key Technical Levels**:
- NVDA: Support at $165, resistance at $185. Currently trading at $175 with bullish momentum
- AMZN: Support at $200, resistance at $220. Consolidating after earnings beat
- Volatility: VIX at 14, suggesting complacency. Watch for spikes above 18

**Momentum Indicators**:
- RSI: NVDA (62), AMZN (58) - both in healthy bullish range
- MACD: Positive crossover on both, confirming uptrend
- Volume: Above-average on up days, suggesting institutional accumulation

**Risk Factors**:
- Fed policy uncertainty - next meeting in 2 weeks
- Earnings season approaching - tech valuations stretched
- Geopolitical tensions - China export restrictions on chips

**Recommendation**: The market environment favors selective tech exposure. NVDA and AMZN are well-positioned for continued strength. Consider taking profits on speculative positions (TLRY, PATH) and reallocating to core holdings.""",
            "analysis_fields": {
                "outlook": "bullish",
                "market_regime": "risk_on",
                "key_insights": [
                    "Tech sector leading with AI tailwinds",
                    "NVDA and AMZN showing strong momentum",
                    "VIX at 14 suggests low volatility environment",
                    "Fed policy uncertainty is key risk factor"
                ],
                "symbols_analyzed": ["NVDA", "AMZN", "TLRY", "PATH", "MARA"]
            },
            "recommendation": {
                "overall_action": {"choice": "buy", "confidence": 0.75},
                "risk_level": {"choice": "moderate", "confidence": 0.70},
                "market_outlook": {"choice": "bullish", "confidence": 0.80}
            }
        },
        {
            "agent_id": "fundamental_analyst_1",
            "agent_type": "LLMFundamentalAnalystAgent",
            "timestamp": "2025-10-17T22:45:15",
            "llm_response_text": """**Deep Fundamental Analysis**

**NVIDIA Corporation (NVDA)**
- Market Cap: $1.2T | P/E: 45x | Revenue Growth: 40% YoY
- **Financial Health**: Exceptional. $29B cash, minimal debt, FCF of $15B annually
- **Earnings Quality**: High. Revenue growth driven by AI chip demand, not accounting gimmicks
- **Competitive Moat**: Dominant 85% market share in AI accelerators. CUDA ecosystem creates switching costs
- **Valuation**: Premium but justified. DCF suggests fair value $160-180. Currently at $175
- **Risks**: China exposure (25% revenue), TSMC dependency, AMD competition emerging
- **Rating**: STRONG BUY - Best-in-class fundamentals with secular AI tailwind

**Amazon.com (AMZN)**
- Market Cap: $1.8T | P/E: 38x | Revenue Growth: 12% YoY
- **Financial Health**: Strong. $88B cash, manageable debt, improving FCF margins
- **Earnings Quality**: Good. AWS margins expanding (32% → 35%), retail improving
- **Competitive Moat**: Unmatched logistics network, AWS market leadership (32% share)
- **Valuation**: Fair. Trading at 2.5x sales vs 5-year avg of 3.2x. Room for multiple expansion
- **Risks**: Regulatory scrutiny, labor costs, retail margin pressure
- **Rating**: BUY - Diversified business model with AWS growth driver

**Marathon Digital (MARA)**
- Market Cap: $4.2B | P/E: N/A (unprofitable) | Revenue Growth: -15% YoY
- **Financial Health**: Weak. Burning cash, high debt, Bitcoin price dependent
- **Earnings Quality**: Poor. Revenue tied to volatile Bitcoin mining economics
- **Competitive Moat**: None. Commoditized business with high energy costs
- **Valuation**: Speculative. Trades on Bitcoin sentiment, not fundamentals
- **Risks**: Bitcoin volatility, energy costs, regulatory uncertainty, dilution risk
- **Rating**: SELL - Fundamentals do not support current valuation

**UiPath (PATH)**
- Market Cap: $8.5B | P/E: N/A (unprofitable) | Revenue Growth: 8% YoY
- **Financial Health**: Adequate. $1.2B cash, no debt, but burning $200M/year
- **Earnings Quality**: Mixed. Revenue growth slowing, customer acquisition costs high
- **Competitive Moat**: Moderate. RPA market leader but facing Microsoft competition
- **Valuation**: Expensive. Trading at 6x sales with slowing growth
- **Risks**: Competition from Microsoft Power Automate, AI disruption of RPA
- **Rating**: HOLD - Wait for better entry point or growth reacceleration

**Tilray Brands (TLRY)**
- Market Cap: $1.1B | P/E: N/A (unprofitable) | Revenue Growth: -8% YoY
- **Financial Health**: Poor. High debt, negative FCF, dilution risk
- **Earnings Quality**: Very Poor. Revenue declining, margins compressing
- **Competitive Moat**: None. Commoditized cannabis market with oversupply
- **Valuation**: Speculative. Trades on legalization hopes, not fundamentals
- **Risks**: Regulatory uncertainty, oversupply, debt burden, dilution
- **Rating**: SELL - Avoid until fundamentals improve""",
            "analysis_fields": {
                "outlook": "mixed",
                "valuation_level": "fair",
                "quality_score": 0.65,
                "key_insights": [
                    "NVDA: Exceptional fundamentals, dominant AI position",
                    "AMZN: Strong diversified model, AWS growth driver",
                    "MARA: Weak fundamentals, Bitcoin dependent",
                    "PATH: Slowing growth, expensive valuation",
                    "TLRY: Poor fundamentals, avoid"
                ],
                "symbols_analyzed": ["NVDA", "AMZN", "MARA", "PATH", "TLRY"]
            },
            "recommendation": {
                "overall_action": {"choice": "buy", "confidence": 0.70},
                "risk_level": {"choice": "moderate", "confidence": 0.75}
            }
        },
        {
            "agent_id": "risk_manager_claude",
            "agent_type": "LLMRiskManagerAgent",
            "timestamp": "2025-10-17T22:45:30",
            "llm_response_text": """**Portfolio Risk Assessment**

**Position-Level Risk Analysis**:

1. **NVDA CALL 01/15/27 $175** (2 contracts)
   - Delta: 0.65 | Theta: -$5.20/day | Vega: $12.50
   - Time to Expiry: 455 days (GOOD - low time decay pressure)
   - Moneyness: ATM (underlying $175.50)
   - Risk: MODERATE - High vega exposure to IV changes
   - P&L: -$865 (-9%) - underwater but time to recover
   - **Action**: HOLD - Long duration provides recovery time

2. **AMZN CALL 05/15/26 $210** (1 contract)
   - Delta: 0.58 | Theta: -$3.80/day | Vega: $8.20
   - Time to Expiry: 210 days (MODERATE - watch time decay)
   - Moneyness: Slightly OTM (underlying $207.40)
   - Risk: MODERATE - Needs 1.3% move to breakeven
   - P&L: -$61 (-2%) - minor loss
   - **Action**: HOLD - Still time for recovery

3. **MARA CALL 09/18/26 $18** (1 contract)
   - Delta: 0.42 | Theta: -$2.10/day | Vega: $4.50
   - Time to Expiry: 336 days
   - Moneyness: Deep OTM (underlying $15.30, needs 18% move)
   - Risk: HIGH - Low probability of profit
   - P&L: -$221 (-22%) - significant loss
   - **Action**: CONSIDER CLOSING - Cut losses on speculative position

4. **PATH CALL 12/19/25 $19** (3 contracts)
   - Delta: 0.18 | Theta: -$1.50/day | Vega: $2.80
   - Time to Expiry: 63 days (HIGH RISK - rapid time decay)
   - Moneyness: Far OTM (underlying $15.96, needs 19% move)
   - Risk: VERY HIGH - Low delta, high theta, short time
   - P&L: -$581 (-67%) - major loss
   - **Action**: CLOSE POSITION - Salvage remaining value

5. **TLRY CALL 12/19/25 $1.50** (5 contracts)
   - Delta: 0.35 | Theta: -$0.70/day | Vega: $1.20
   - Time to Expiry: 63 days (HIGH RISK - rapid time decay)
   - Moneyness: Deep OTM (underlying $1.05, needs 43% move)
   - Risk: VERY HIGH - Extremely low probability
   - P&L: -$68 (-28%) - significant loss
   - **Action**: CLOSE POSITION - Avoid further decay

**Portfolio-Level Risk Metrics**:
- **Total Delta**: +3.85 (moderately bullish exposure)
- **Total Theta**: -$13.30/day ($399/month time decay)
- **Total Vega**: +$29.20 (high IV sensitivity)
- **Concentration**: 63% in NVDA (HIGH - consider diversification)
- **Correlation**: NVDA/AMZN 0.75 (tech sector concentration)

**Risk Warnings**:
1. ⚠️ High time decay: Losing $399/month to theta
2. ⚠️ Concentration risk: 63% in single position (NVDA)
3. ⚠️ Speculative positions: PATH and TLRY have <20% probability of profit
4. ⚠️ IV risk: Portfolio vega of +$29 means $290 loss per 1% IV drop

**Recommendations**:
1. CLOSE PATH and TLRY positions immediately - salvage $463 before further decay
2. HOLD NVDA and AMZN - long duration provides recovery time
3. CONSIDER CLOSING MARA - speculative position with poor risk/reward
4. REDUCE concentration - NVDA is 63% of portfolio
5. HEDGE with protective puts if maintaining long exposure""",
            "analysis_fields": {
                "portfolio_delta": 3.85,
                "portfolio_theta": -13.30,
                "portfolio_vega": 29.20,
                "concentration_risk": "high",
                "key_insights": [
                    "High time decay: $399/month",
                    "Concentration: 63% in NVDA",
                    "PATH and TLRY have <20% profit probability",
                    "High vega exposure to IV changes"
                ],
                "symbols_analyzed": ["NVDA", "AMZN", "MARA", "PATH", "TLRY"]
            },
            "recommendation": {
                "overall_action": {"choice": "hold", "confidence": 0.60},
                "risk_level": {"choice": "moderate", "confidence": 0.85}
            }
        }
    ],
    
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
                    "agent_id": "market_analyst_claude",
                    "agent_type": "LLMMarketAnalystAgent",
                    "key_insights": ["Tech sector leading with AI tailwinds", "NVDA showing strong momentum"],
                    "recommendation": "buy",
                    "confidence": 0.75
                },
                {
                    "agent_id": "fundamental_analyst_1",
                    "agent_type": "LLMFundamentalAnalystAgent",
                    "key_insights": ["Exceptional fundamentals", "Dominant AI position"],
                    "recommendation": "buy",
                    "confidence": 0.85
                }
            ],
            "risk_warnings": [
                "📉 Position underwater: -9.0% loss",
                "⏰ Daily time decay: $5.20/day",
                "📊 High vega exposure: $12.50 per 1% IV change"
            ],
            "opportunities": [
                "⏳ Long time horizon: 455 days - low time decay pressure",
                "✅ At-the-money (delta: 0.65) - good probability of profit",
                "🎯 Strong fundamentals support recovery"
            ]
        },
        {
            "symbol": "PATH",
            "asset_type": "option",
            "option_type": "call",
            "strike": 19.0,
            "expiration_date": "2025-12-19",
            "quantity": 3,
            "current_metrics": {
                "current_price": 0.96,
                "underlying_price": 15.96,
                "market_value": 288.0,
                "unrealized_pnl": -581.01,
                "unrealized_pnl_pct": -66.86,
                "days_to_expiry": 63,
                "iv": 0.58
            },
            "greeks": {
                "delta": 0.18,
                "gamma": 0.008,
                "theta": -1.50,
                "vega": 2.80
            },
            "agent_insights_for_position": [],
            "risk_warnings": [
                "⚠️ High time decay risk - only 63 days to expiration",
                "📉 Position underwater: -66.9% loss",
                "⏰ High daily time decay: $1.50/day",
                "🎯 Far out-of-the-money (delta: 0.18) - needs 19% move"
            ],
            "opportunities": []
        }
    ],
    
    "swarm_health": {
        "active_agents_count": 16,
        "contributed_vs_failed": {
            "contributed": 13,
            "failed": 3,
            "success_rate": 81.25
        },
        "communication_stats": {
            "total_messages": 47,
            "total_state_updates": 28,
            "average_message_priority": 7.2,
            "average_confidence": 0.73
        },
        "consensus_strength": {
            "overall_action_confidence": 0.625,
            "risk_level_confidence": 0.75,
            "market_outlook_confidence": 0.8125,
            "average_confidence": 0.73
        }
    },
    
    "enhanced_consensus": {
        "vote_breakdown_by_agent": {
            "overall_action": {
                "market_analyst_claude": {"choice": "buy", "confidence": 0.75},
                "fundamental_analyst_1": {"choice": "buy", "confidence": 0.70},
                "risk_manager_claude": {"choice": "hold", "confidence": 0.60}
            }
        },
        "dissenting_opinions": [
            {
                "agent_id": "risk_manager_claude",
                "dissenting_choice": "hold",
                "consensus_choice": "buy",
                "confidence": 0.60,
                "reasoning": "High time decay and concentration risk warrant caution"
            }
        ],
        "top_contributors": [
            {"agent_id": "fundamental_analyst_1", "confidence": 0.85},
            {"agent_id": "market_analyst_claude", "confidence": 0.75}
        ]
    },
    
    "discussion_logs": [
        {
            "source_agent": "market_analyst_claude",
            "content": {
                "type": "market_analysis",
                "key_insights": ["Tech sector leading", "AI tailwinds strong"],
                "outlook": "bullish"
            },
            "priority": 8,
            "confidence": 0.75,
            "timestamp": "2025-10-17T22:45:00"
        },
        {
            "source_agent": "fundamental_analyst_1",
            "content": {
                "type": "fundamental_analysis",
                "key_insights": ["NVDA exceptional fundamentals", "AMZN strong AWS growth"],
                "valuation": "fair"
            },
            "priority": 9,
            "confidence": 0.80,
            "timestamp": "2025-10-17T22:45:15"
        }
    ],
    
    "portfolio_summary": {
        "total_value": 13851.20,
        "total_unrealized_pnl": -1796.02,
        "total_unrealized_pnl_pct": -11.48,
        "positions_count": 6
    },
    
    "import_stats": {
        "positions_imported": 6,
        "positions_failed": 0,
        "chase_conversion": True
    }
}

# Save mock data
mock_file = f"{OUTPUT_DIR}/mock_api_response.json"
with open(mock_file, 'w') as f:
    json.dump(mock_data, f, indent=2)

print(f"✓ Mock data saved: {mock_file}")

# Generate visualization
html_content = generate_visualization_html(mock_data)
viz_file = f"{OUTPUT_DIR}/demo_swarm_analysis_visualization.html"
with open(viz_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✓ Visualization created: {viz_file}")
print(f"\n🎨 Open the visualization in your browser:")
print(f"   file:///{Path(viz_file).absolute()}")

