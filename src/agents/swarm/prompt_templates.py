"""
Role-Specific Prompt Templates for Swarm Agents

Provides differentiated perspectives and focus areas for each agent type
to prevent redundant analysis and ensure diverse viewpoints.
"""

from typing import Dict, Any, Optional, List


# Role-specific perspectives for all 17 agents
ROLE_PERSPECTIVES = {
    # Tier 1: Oversight & Coordination
    "SwarmOverseer": {
        "perspective": "You are a strategic coordinator overseeing a team of 17 specialized analysts.",
        "focus": "Synthesize high-level insights, identify conflicts, ensure comprehensive coverage, and coordinate agent priorities.",
        "avoid": "Do not perform detailed technical, fundamental, or market analysis. Focus on coordination and strategic oversight."
    },
    
    # Tier 2: Market Intelligence
    "MarketAnalyst": {
        "perspective": "You are a market microstructure specialist focused on price action, volume, and liquidity.",
        "focus": "Analyze order flow, bid-ask spreads, market depth, volume patterns, and intraday price movements.",
        "avoid": "Do not duplicate fundamental analysis, sentiment analysis, or options-specific analysis."
    },
    "LLMMarketAnalyst": {
        "perspective": "You are an AI-powered market intelligence analyst with access to real-time data.",
        "focus": "Identify market trends, momentum shifts, support/resistance levels, and trading patterns using advanced analytics.",
        "avoid": "Do not duplicate fundamental valuation, risk assessment, or compliance analysis."
    },
    
    # Tier 3: Fundamental & Macro
    "FundamentalAnalyst": {
        "perspective": "You are a value investor focused on intrinsic worth and financial health.",
        "focus": "Analyze earnings quality, cash flow generation, balance sheet strength, valuation multiples, and competitive positioning.",
        "avoid": "Do not duplicate technical analysis, sentiment analysis, or short-term trading signals."
    },
    "LLMFundamentalAnalyst": {
        "perspective": "You are an AI-powered fundamental analyst with deep financial modeling expertise.",
        "focus": "Evaluate business models, revenue quality, margin sustainability, capital allocation, and long-term value drivers.",
        "avoid": "Do not duplicate market microstructure, options analysis, or compliance checks."
    },
    "MacroEconomist": {
        "perspective": "You are a macroeconomic strategist analyzing top-down market drivers.",
        "focus": "Assess interest rate trends, inflation dynamics, GDP growth, sector rotation, and policy impacts on asset prices.",
        "avoid": "Do not duplicate company-specific fundamental analysis or technical trading signals."
    },
    "LLMMacroEconomist": {
        "perspective": "You are an AI-powered macro strategist with global economic expertise.",
        "focus": "Identify macro regime shifts, central bank policy impacts, currency effects, and cross-asset correlations.",
        "avoid": "Do not duplicate individual stock analysis, options strategies, or compliance reviews."
    },
    
    # Tier 4: Risk & Sentiment
    "RiskManager": {
        "perspective": "You are a risk officer focused on downside protection and tail risk.",
        "focus": "Identify potential losses, correlation risks, stress scenarios, liquidity risks, and portfolio concentration.",
        "avoid": "Do not duplicate performance analysis, fundamental valuation, or market timing signals."
    },
    "LLMRiskManager": {
        "perspective": "You are an AI-powered risk analyst with advanced scenario modeling capabilities.",
        "focus": "Quantify Value-at-Risk, stress test portfolios, assess tail risks, and identify hidden correlations.",
        "avoid": "Do not duplicate sentiment analysis, options pricing, or execution strategies."
    },
    "SentimentAnalyst": {
        "perspective": "You are a behavioral finance specialist analyzing market psychology and investor sentiment.",
        "focus": "Gauge fear/greed indicators, social media sentiment, news sentiment, and contrarian signals.",
        "avoid": "Do not duplicate fundamental analysis, technical analysis, or risk quantification."
    },
    "LLMSentimentAnalyst": {
        "perspective": "You are an AI-powered sentiment analyst with natural language processing expertise.",
        "focus": "Extract sentiment from news, social media, analyst reports, and earnings calls using advanced NLP.",
        "avoid": "Do not duplicate market microstructure, fundamental valuation, or options analysis."
    },
    
    # Tier 5: Options & Volatility
    "OptionsStrategist": {
        "perspective": "You are an options specialist focused on derivatives strategies and volatility trading.",
        "focus": "Analyze implied volatility, skew, term structure, Greeks, and optimal options strategies.",
        "avoid": "Do not duplicate equity fundamental analysis, macro analysis, or compliance checks."
    },
    "VolatilitySpecialist": {
        "perspective": "You are a volatility trader analyzing realized and implied volatility dynamics.",
        "focus": "Assess volatility regimes, vol-of-vol, correlation breakdowns, and volatility arbitrage opportunities.",
        "avoid": "Do not duplicate fundamental analysis, sentiment analysis, or execution logistics."
    },
    "LLMVolatilitySpecialist": {
        "perspective": "You are an AI-powered volatility analyst with advanced statistical modeling capabilities.",
        "focus": "Model volatility surfaces, identify vol anomalies, forecast volatility changes, and assess tail risk.",
        "avoid": "Do not duplicate market microstructure, fundamental analysis, or compliance reviews."
    },
    
    # Tier 6: Execution & Compliance
    "TradeExecutor": {
        "perspective": "You are an execution specialist focused on optimal trade implementation.",
        "focus": "Determine optimal execution strategies, timing, order types, and slippage minimization.",
        "avoid": "Do not duplicate investment analysis, risk assessment, or compliance reviews. Focus on execution mechanics."
    },
    "ComplianceOfficer": {
        "perspective": "You are a compliance officer ensuring regulatory adherence and risk controls.",
        "focus": "Verify position limits, concentration limits, regulatory restrictions, and suitability requirements.",
        "avoid": "Do not duplicate investment analysis, market analysis, or execution strategies. Focus on compliance."
    },
    
    # Tier 7: Recommendation Engine
    "LLMRecommendationAgent": {
        "perspective": "You are a portfolio manager synthesizing insights from multiple analysts to make final investment decisions.",
        "focus": "Integrate fundamental, technical, risk, and sentiment inputs to generate actionable Buy/Sell/Hold recommendations.",
        "avoid": "Do not perform original analysis. Synthesize existing insights from other agents into clear recommendations."
    }
}


def get_agent_prompt(agent_type: str, base_context: str, swarm_context: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate role-specific prompt with unique perspective.
    
    Args:
        agent_type: Type of agent (e.g., "MarketAnalyst", "RiskManager")
        base_context: Base context/data to analyze
        swarm_context: Optional context from other agents (topics covered, key insights)
    
    Returns:
        Formatted prompt with role-specific instructions
    """
    role_config = ROLE_PERSPECTIVES.get(agent_type, {
        "perspective": "You are a financial analyst.",
        "focus": "Analyze the provided data.",
        "avoid": "Avoid redundant analysis."
    })
    
    prompt = f"""
{role_config.get('perspective', '')}

FOCUS AREAS:
{role_config.get('focus', '')}

AVOID DUPLICATION:
{role_config.get('avoid', '')}
"""
    
    # Add swarm context if available
    if swarm_context:
        topics_covered = swarm_context.get('topics_covered', [])
        uncovered_topics = swarm_context.get('uncovered_topics', [])
        
        if topics_covered:
            prompt += f"""
TOPICS ALREADY ANALYZED BY OTHER AGENTS:
{', '.join(topics_covered)}
"""
        
        if uncovered_topics:
            prompt += f"""
PRIORITY TOPICS TO FOCUS ON:
{', '.join(uncovered_topics)}
"""
    
    prompt += f"""
CONTEXT TO ANALYZE:
{base_context}

Provide your unique perspective based on your specialized role. Be specific and actionable.
"""
    
    return prompt


def get_distillation_prompt(
    insights: Dict[str, List[Dict]],
    position_data: Dict[str, Any]
) -> str:
    """
    Generate prompt for Distillation Agent to synthesize insights into investor narrative.

    Args:
        insights: Categorized insights from all agents
        position_data: Position data being analyzed

    Returns:
        Synthesis prompt for narrative generation
    """
    import json

    # Extract metrics if available
    metrics = position_data.get('metrics', {})
    metrics_section = ""

    if metrics:
        metrics_section = f"""
INSTITUTIONAL-GRADE PORTFOLIO METRICS:
Performance Metrics:
- Total Return: {metrics.get('total_return', 0):.2%}
- Average Position Return: {metrics.get('avg_position_return', 0):.2%}
- Win Rate: {metrics.get('win_rate', 0):.1%}

Risk Metrics:
- Volatility: {metrics.get('volatility', 0):.2%}
- Max Drawdown: {metrics.get('max_drawdown', 0):.2%}
- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f} (>1.0 good, >2.0 excellent)
- Omega Ratio: {metrics.get('omega_ratio', 0):.2f} (>1.0 good, >2.0 excellent)

Diversification:
- Position Count: {metrics.get('position_count', 0)}
- Concentration Risk (HHI): {metrics.get('concentration_risk', 0):.3f} (0=diversified, 1=concentrated)
- Effective N: {metrics.get('effective_n', 0):.1f} (equivalent equal-weighted positions)

INTERPRETATION GUIDE:
- Sharpe Ratio: Measures risk-adjusted returns. >1.0 is good, >2.0 is excellent (Renaissance Technologies targets 2+)
- Omega Ratio: Probability-weighted gains/losses. >1.0 means gains outweigh losses, >2.0 is exceptional
- Concentration Risk: Lower is better. <0.2 is well-diversified, >0.5 is concentrated
- Effective N: Higher is better. Shows true diversification (e.g., 10 positions with HHI=0.1 → Effective N=10)
"""

    return f"""
You are an institutional investment analyst creating a client-facing report.

POSITION DATA:
{json.dumps(position_data.get('portfolio', {}), indent=2)}

{metrics_section}

AGENT INSIGHTS SUMMARY:
- Bullish Signals: {len(insights.get('bullish_signals', []))} insights
- Bearish Signals: {len(insights.get('bearish_signals', []))} insights
- Risk Factors: {len(insights.get('risk_factors', []))} insights
- Opportunities: {len(insights.get('opportunities', []))} insights
- Technical Levels: {len(insights.get('technical_levels', []))} insights
- Fundamental Metrics: {len(insights.get('fundamental_metrics', []))} insights

DETAILED INSIGHTS:
{json.dumps(insights, indent=2)}

Create an investor-friendly report with these sections:

1. EXECUTIVE SUMMARY (2-3 paragraphs)
   - High-level investment thesis
   - Key portfolio metrics interpretation (Sharpe, Omega, diversification)
   - Overall recommendation

2. INVESTMENT RECOMMENDATION
   - Clear Buy/Sell/Hold rating
   - Conviction level (High/Medium/Low)
   - Price target (if applicable)
   - Rationale (3-5 bullet points)
   - Reference key metrics (Sharpe, Omega, Win Rate)

3. RISK ASSESSMENT
   - Portfolio risk metrics (Volatility, Max Drawdown, Concentration)
   - Primary risks (ranked by severity)
   - Probability and impact
   - Mitigation strategies

4. FUTURE OUTLOOK
   - 3-month, 6-month, 12-month projections
   - Key catalysts to watch
   - Scenarios (bull/base/bear case)
   - Expected impact on portfolio metrics

5. ACTIONABLE NEXT STEPS
   - Specific actions for investor
   - Monitoring triggers (metric thresholds)
   - Rebalancing recommendations to improve Sharpe/Omega/diversification

GUIDELINES:
- Use clear, professional language
- Interpret metrics for retail investors (explain what Sharpe 1.5 means in plain English)
- Avoid jargon or define technical terms
- Be specific and actionable
- Focus on what matters to investors
- Maintain objectivity while being decisive
- Compare metrics to benchmarks (S&P 500 Sharpe ~0.5-1.0, Renaissance ~2.0+)
- Target reading level: Professional investor (Flesch-Kincaid 10-12)

Generate the report now:
"""

