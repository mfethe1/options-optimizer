#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM-powered Recommendation Agent for suggesting alternative positions.

This agent analyzes underperforming positions and recommends replacements
with similar pricing but better risk/reward profiles.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..base_swarm_agent import BaseSwarmAgent
from ..llm_agent_base import LLMAgentBase
from ..shared_context import SharedContext
from ..consensus_engine import ConsensusEngine

logger = logging.getLogger(__name__)


class LLMRecommendationAgent(BaseSwarmAgent, LLMAgentBase):
    """
    LLM-powered agent that recommends alternative positions.
    
    Analyzes underperforming positions and suggests replacements with:
    - Similar pricing (within 20% of current position value)
    - Higher probability of return
    - Lower risk profile
    - Better risk/reward ratio
    
    Recommends both stock and option alternatives.
    """
    
    def __init__(
        self,
        agent_id: str,
        shared_context: SharedContext,
        consensus_engine: ConsensusEngine,
        preferred_model: str = "anthropic"
    ):
        BaseSwarmAgent.__init__(
            self,
            agent_id=agent_id,
            agent_type="RecommendationAgent",
            shared_context=shared_context,
            consensus_engine=consensus_engine,
            priority=7,
            confidence_threshold=0.65
        )

        LLMAgentBase.__init__(
            self,
            agent_id=agent_id,
            agent_type="RecommendationAgent",
            preferred_model=preferred_model
        )

        logger.info(f"{agent_id}: LLM Recommendation Agent initialized with {preferred_model}")
    
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze positions and recommend alternatives.
        
        Args:
            context: Contains portfolio data, market data, and position details
        
        Returns:
            Analysis with recommendations for each position
        """
        try:
            portfolio = context.get('portfolio', {})
            market = context.get('market', {})
            
            positions = portfolio.get('positions', [])
            cash_available = portfolio.get('cash_available', 0)
            
            # Build comprehensive prompt
            prompt = self._build_recommendation_prompt(positions, cash_available, market)
            
            # Call LLM
            response = self.call_llm(
                prompt=prompt,
                max_tokens=3000,
                temperature=0.3  # Lower temperature for more focused recommendations
            )
            
            # Parse response
            parsed = self._parse_llm_response(response, positions)
            
            # Record success
            self.record_success(context)
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error in {self.agent_id} analysis: {e}")
            self.record_error(e, context)
            return {'error': str(e)}

    def make_recommendation(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make recommendations based on the analysis.

        This method is required by BaseSwarmAgent abstract class.
        For the RecommendationAgent, the analysis already contains the recommendations,
        so we extract and format them for the consensus engine.

        Args:
            analysis: Analysis results from analyze() method

        Returns:
            Recommendations with confidence levels
        """
        try:
            recommendations = analysis.get('recommendations', [])
            replacements_recommended = analysis.get('replacements_recommended', 0)
            total_analyzed = analysis.get('total_positions_analyzed', 0)

            # Calculate overall confidence based on replacement rate
            if total_analyzed > 0:
                replacement_rate = replacements_recommended / total_analyzed
                # Higher replacement rate = more opportunities found = higher confidence
                confidence = 0.60 + (replacement_rate * 0.30)
            else:
                confidence = 0.50

            # Determine overall action
            if replacements_recommended > 0:
                action = 'optimize'  # Suggest portfolio optimization
                reasoning = f"Found {replacements_recommended} positions with better alternatives out of {total_analyzed} analyzed"
            else:
                action = 'hold'
                reasoning = f"Current positions are optimal. No better alternatives found for {total_analyzed} positions."

            recommendation = {
                'overall_action': {
                    'choice': action,
                    'confidence': min(confidence, 0.95),
                    'reasoning': reasoning
                },
                'positions_analyzed': total_analyzed,
                'replacements_recommended': replacements_recommended,
                'replacement_rate': replacements_recommended / total_analyzed if total_analyzed > 0 else 0,
                'detailed_recommendations': recommendations,
                'timestamp': datetime.utcnow().isoformat()
            }

            return recommendation

        except Exception as e:
            logger.error(f"{self.agent_id}: Error making recommendation: {e}")
            return {'error': str(e)}

    def _build_recommendation_prompt(
        self,
        positions: List[Dict[str, Any]],
        cash_available: float,
        market: Dict[str, Any]
    ) -> str:
        """Build comprehensive prompt for LLM"""
        
        # Format positions
        positions_text = ""
        for i, pos in enumerate(positions, 1):
            symbol = pos.get('symbol', 'UNKNOWN')
            asset_type = pos.get('asset_type', 'unknown')
            current_value = pos.get('market_value', 0)
            pnl = pos.get('unrealized_pnl', 0)
            pnl_pct = pos.get('unrealized_pnl_pct', 0)
            
            if asset_type == 'option':
                option_type = pos.get('option_type', 'call').upper()
                strike = pos.get('strike', 0)
                expiry = pos.get('expiration_date', 'N/A')
                delta = pos.get('delta', 0)
                theta = pos.get('theta', 0)
                days_to_expiry = pos.get('days_to_expiry', 0)
                
                positions_text += f"""
{i}. {symbol} {option_type} ${strike} exp {expiry}
   - Current Value: ${current_value:,.2f}
   - P&L: ${pnl:,.2f} ({pnl_pct:+.1f}%)
   - Delta: {delta:.2f}, Theta: ${theta:.2f}/day
   - Days to Expiry: {days_to_expiry}
"""
            else:
                quantity = pos.get('quantity', 0)
                positions_text += f"""
{i}. {symbol} Stock ({quantity} shares)
   - Current Value: ${current_value:,.2f}
   - P&L: ${pnl:,.2f} ({pnl_pct:+.1f}%)
"""
        
        # Get market sentiment
        market_sentiment = market.get('sentiment', 'neutral')
        vix = market.get('vix', 15)
        
        prompt = f"""You are an expert portfolio advisor specializing in options and stock recommendations.

**PORTFOLIO ANALYSIS REQUEST**

**Current Positions:**
{positions_text}

**Available Cash:** ${cash_available:,.2f}

**Market Conditions:**
- Overall Sentiment: {market_sentiment}
- VIX (Volatility): {vix}

**YOUR TASK:**

For EACH position above, provide:

1. **Position Assessment** (1-2 sentences)
   - Is this position worth keeping or should it be replaced?
   - Why or why not?

2. **If Replacement Recommended:**
   
   **STOCK ALTERNATIVE:**
   - Symbol: [TICKER]
   - Current Price: $[PRICE]
   - Quantity: [SHARES] (to match ~same dollar value)
   - Total Cost: $[COST]
   - Probability of High Return: [XX]% (12-month horizon)
   - Risk Level: [Low/Moderate/High]
   - Key Catalyst: [1 sentence]
   - Why Better: [1-2 sentences]
   
   **OPTION ALTERNATIVE:**
   - Symbol: [TICKER]
   - Type: [CALL/PUT]
   - Strike: $[STRIKE]
   - Expiration: [DATE] ([XX] days out)
   - Contracts: [N] (to match ~same dollar value)
   - Total Cost: $[COST]
   - Delta: [X.XX]
   - Probability of Profit: [XX]%
   - Risk Level: [Low/Moderate/High]
   - Key Catalyst: [1 sentence]
   - Why Better: [1-2 sentences]

**IMPORTANT GUIDELINES:**

1. **Price Matching:** Recommendations should cost within ±20% of current position value
2. **Risk/Reward:** Prioritize high probability of return with minimized risk
3. **Diversification:** Don't recommend the same ticker multiple times
4. **Market Conditions:** Consider current VIX and sentiment
5. **Time Horizon:** 
   - Stocks: 12-month outlook
   - Options: 60-180 days to expiration (avoid short-term decay)
6. **Probability Estimates:** Be realistic based on:
   - Historical volatility
   - Sector trends
   - Technical levels
   - Fundamental catalysts

**FORMAT YOUR RESPONSE AS:**

Position 1: [SYMBOL]
Assessment: [Your assessment]
Action: [KEEP / REPLACE]

[If REPLACE:]
STOCK ALTERNATIVE:
Symbol: [TICKER]
Current Price: $[PRICE]
...

OPTION ALTERNATIVE:
Symbol: [TICKER]
Type: [CALL/PUT]
...

---

Position 2: [SYMBOL]
...

**BEGIN YOUR ANALYSIS:**
"""
        
        return prompt
    
    def _parse_llm_response(
        self,
        response: str,
        positions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse LLM response into structured recommendations"""
        
        recommendations = []
        symbols_analyzed = []
        
        # Split by position
        position_blocks = response.split('Position ')
        
        for block in position_blocks[1:]:  # Skip first empty split
            try:
                lines = block.strip().split('\n')
                
                # Extract position info
                first_line = lines[0] if lines else ""
                symbol = first_line.split(':')[1].strip() if ':' in first_line else "UNKNOWN"
                symbols_analyzed.append(symbol)
                
                # Extract assessment and action
                assessment = ""
                action = "KEEP"
                stock_alt = {}
                option_alt = {}
                
                current_section = None
                
                for line in lines:
                    line_clean = line.strip()
                    
                    if line_clean.startswith('Assessment:'):
                        assessment = line_clean.split('Assessment:')[1].strip()
                    elif line_clean.startswith('Action:'):
                        action = line_clean.split('Action:')[1].strip().upper()
                    elif 'STOCK ALTERNATIVE:' in line_clean:
                        current_section = 'stock'
                    elif 'OPTION ALTERNATIVE:' in line_clean:
                        current_section = 'option'
                    elif current_section and ':' in line_clean:
                        key, value = line_clean.split(':', 1)
                        key = key.strip().lower().replace(' ', '_')
                        value = value.strip()
                        
                        if current_section == 'stock':
                            stock_alt[key] = value
                        elif current_section == 'option':
                            option_alt[key] = value
                
                recommendations.append({
                    'symbol': symbol,
                    'assessment': assessment,
                    'action': action,
                    'stock_alternative': stock_alt if stock_alt else None,
                    'option_alternative': option_alt if option_alt else None
                })
                
            except Exception as e:
                logger.warning(f"Error parsing position block: {e}")
                continue
        
        return {
            'recommendations': recommendations,
            'symbols_analyzed': symbols_analyzed,
            'total_positions_analyzed': len(recommendations),
            'replacements_recommended': sum(1 for r in recommendations if r['action'] == 'REPLACE'),
            'llm_response': response
        }

