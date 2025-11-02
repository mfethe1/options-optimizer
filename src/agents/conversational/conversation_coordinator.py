"""
Conversation Coordinator Agent

Handles multi-turn dialogue for natural language trading interactions.
Maintains context across conversations and routes to specialized agents.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import json

from anthropic import Anthropic
from src.agents.swarm.shared_context import SharedContext
from src.agents.conversational.semantic_router import SemanticIntentRouter

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Single turn in a conversation"""
    user_message: str
    agent_response: str
    intent: str
    confidence: float
    timestamp: datetime
    context: Dict[str, Any]


class ConversationCoordinatorAgent:
    """
    COMPETITIVE ADVANTAGE: Natural language trading interface

    Enables traders to interact with the platform conversationally:
    - Multi-turn dialogue (15+ turns with context)
    - Intent-aware routing (trade, analyze, research, portfolio)
    - Contextual understanding (remembers previous questions)
    - Educational mode (explains complex concepts)

    Example conversations:
    - "What's the risk/reward on selling NVDA 950 puts expiring next Friday?"
    - "Show me high IV stocks in the tech sector"
    - "What happens if AAPL drops 10% before my calls expire?"
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize conversation coordinator"""
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.context = SharedContext()
        self.router = SemanticIntentRouter()

        # Conversation history per user
        self.conversations: Dict[str, List[ConversationTurn]] = {}

        # System prompt for natural trading dialogue
        self.system_prompt = """You are an expert options trading advisor with deep knowledge of:
- Options strategies (covered calls, cash-secured puts, spreads, straddles, strangles)
- Risk management (position sizing, stop losses, hedging)
- Technical analysis (support/resistance, momentum, volatility)
- Options Greeks (Delta, Gamma, Theta, Vega, Rho)
- Market dynamics (IV, skew, term structure, open interest)

Your goal is to help traders make informed decisions through clear, conversational explanations.

Guidelines:
1. Ask clarifying questions if the request is ambiguous
2. Provide specific numbers and probabilities when analyzing trades
3. Always mention risks alongside potential rewards
4. Use plain language, then explain technical terms if needed
5. Offer actionable recommendations with reasoning
6. Remember context from previous messages in the conversation

Format responses as JSON with keys:
- "response": Your conversational response to the user
- "data": Any numerical data, calculations, or structured information
- "recommendations": Specific actionable steps (if applicable)
- "risks": Key risks to be aware of (if applicable)
- "next_questions": Suggested follow-up questions the user might ask
"""

    async def process_message(
        self,
        user_message: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a conversational trading request

        Args:
            user_message: User's natural language message
            user_id: Unique user identifier
            context: Additional context (portfolio, positions, etc.)

        Returns:
            Dictionary with response, intent, confidence, and recommendations
        """
        logger.info(f"Processing message for user {user_id}: {user_message[:100]}")

        try:
            # Classify intent
            intent_result = await self.router.classify_intent(user_message)
            intent = intent_result['intent']
            confidence = intent_result['confidence']

            logger.info(f"Classified intent: {intent} (confidence: {confidence:.2f})")

            # Get conversation history
            history = self.conversations.get(user_id, [])

            # Build context-aware prompt
            prompt = self._build_prompt(user_message, history, context)

            # Route to specialized workflow based on intent
            if intent == "trade_execution":
                response = await self._trade_execution_workflow(prompt, context)
            elif intent == "risk_analysis":
                response = await self._risk_analysis_workflow(prompt, context)
            elif intent == "research":
                response = await self._research_workflow(prompt, context)
            elif intent == "portfolio_review":
                response = await self._portfolio_review_workflow(prompt, context)
            elif intent == "education":
                response = await self._education_workflow(prompt, context)
            else:
                # General conversation
                response = await self._general_workflow(prompt, context)

            # Store conversation turn
            turn = ConversationTurn(
                user_message=user_message,
                agent_response=response['response'],
                intent=intent,
                confidence=confidence,
                timestamp=datetime.now(),
                context=context or {}
            )

            if user_id not in self.conversations:
                self.conversations[user_id] = []
            self.conversations[user_id].append(turn)

            # Limit conversation history to last 15 turns
            if len(self.conversations[user_id]) > 15:
                self.conversations[user_id] = self.conversations[user_id][-15:]

            return {
                "response": response['response'],
                "intent": intent,
                "confidence": confidence,
                "data": response.get('data'),
                "recommendations": response.get('recommendations', []),
                "risks": response.get('risks', []),
                "next_questions": response.get('next_questions', []),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return {
                "response": f"I encountered an error processing your request: {str(e)}. Please try rephrasing or contact support.",
                "intent": "error",
                "confidence": 0.0,
                "error": str(e)
            }

    def _build_prompt(
        self,
        user_message: str,
        history: List[ConversationTurn],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build context-aware prompt from conversation history"""

        prompt_parts = []

        # Add conversation history (last 5 turns for context)
        if history:
            prompt_parts.append("Conversation history:")
            for turn in history[-5:]:
                prompt_parts.append(f"User: {turn.user_message}")
                prompt_parts.append(f"Assistant: {turn.agent_response}\n")

        # Add current context (portfolio, positions, market data)
        if context:
            prompt_parts.append("\nCurrent context:")
            if 'portfolio' in context:
                prompt_parts.append(f"Portfolio value: ${context['portfolio'].get('total_value', 0):,.2f}")
                prompt_parts.append(f"Positions: {len(context['portfolio'].get('positions', []))}")
            if 'market_data' in context:
                prompt_parts.append(f"Market data available: {', '.join(context['market_data'].keys())}")

        # Add current user message
        prompt_parts.append(f"\nCurrent request: {user_message}")
        prompt_parts.append("\nProvide a helpful, conversational response in JSON format.")

        return "\n".join(prompt_parts)

    async def _trade_execution_workflow(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Handle trade execution requests"""

        enhanced_prompt = f"""{prompt}

For this trade request, provide:
1. Trade details (symbol, strike, expiry, quantity)
2. Risk/reward analysis (max profit, max loss, break-even)
3. Win probability based on current IV
4. Position sizing recommendation
5. Entry and exit criteria

Remember to ask for missing information (e.g., which symbol, strike, expiry)."""

        return await self._call_claude(enhanced_prompt)

    async def _risk_analysis_workflow(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Handle risk analysis requests"""

        enhanced_prompt = f"""{prompt}

For this risk analysis, provide:
1. Current risk metrics (Delta, Theta, Vega exposure)
2. Scenario analysis (what if stock moves ±5%, ±10%)
3. Risk level assessment (Low/Medium/High/Critical)
4. Mitigation strategies (hedging, position sizing)
5. Risk/reward ratio"""

        return await self._call_claude(enhanced_prompt)

    async def _research_workflow(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Handle research requests"""

        enhanced_prompt = f"""{prompt}

For this research request, provide:
1. Key findings (earnings, news, catalyst events)
2. Technical analysis (support/resistance, trend)
3. Options flow analysis (unusual activity)
4. Sentiment summary (bullish/bearish/neutral)
5. Actionable insights (trade ideas based on research)"""

        return await self._call_claude(enhanced_prompt)

    async def _portfolio_review_workflow(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Handle portfolio review requests"""

        enhanced_prompt = f"""{prompt}

For this portfolio review, provide:
1. Performance summary (P&L, returns, Sharpe ratio)
2. Risk assessment (concentration, correlation, Greeks)
3. Position recommendations (trim, hold, add)
4. Rebalancing suggestions
5. Upcoming risks (earnings, expirations)"""

        return await self._call_claude(enhanced_prompt)

    async def _education_workflow(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Handle educational requests (explain concepts)"""

        enhanced_prompt = f"""{prompt}

For this educational request, provide:
1. Simple explanation (ELI5 - Explain Like I'm 5)
2. Technical details (for deeper understanding)
3. Practical example (real-world scenario)
4. Common misconceptions (what people get wrong)
5. Further reading (related topics to explore)"""

        return await self._call_claude(enhanced_prompt)

    async def _general_workflow(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Handle general conversation"""

        return await self._call_claude(prompt)

    async def _call_claude(self, prompt: str) -> Dict[str, Any]:
        """Call Claude API with structured output"""

        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.7,
                system=self.system_prompt,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Parse JSON response
            content = response.content[0].text

            # Try to parse as JSON, fallback to plain text
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                result = {
                    "response": content,
                    "data": None,
                    "recommendations": [],
                    "risks": [],
                    "next_questions": []
                }

            return result

        except Exception as e:
            logger.error(f"Error calling Claude: {e}")
            return {
                "response": f"I encountered an error: {str(e)}. Please try again.",
                "data": None,
                "recommendations": [],
                "risks": [],
                "next_questions": []
            }

    def get_conversation_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a user"""

        history = self.conversations.get(user_id, [])
        return [
            {
                "user_message": turn.user_message,
                "agent_response": turn.agent_response,
                "intent": turn.intent,
                "confidence": turn.confidence,
                "timestamp": turn.timestamp.isoformat()
            }
            for turn in history
        ]

    def clear_conversation_history(self, user_id: str):
        """Clear conversation history for a user"""

        if user_id in self.conversations:
            del self.conversations[user_id]
            logger.info(f"Cleared conversation history for user {user_id}")
