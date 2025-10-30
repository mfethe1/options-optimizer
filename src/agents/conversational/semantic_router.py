"""
Semantic Intent Router

Classifies user messages into intents for proper agent routing.
Uses pattern matching and keyword analysis for fast, accurate classification.
"""

import logging
from typing import Dict, List
import re

logger = logging.getLogger(__name__)


class SemanticIntentRouter:
    """
    COMPETITIVE ADVANTAGE: Intent-aware routing

    Classifies user messages into intents to route to specialized agents:
    - trade_execution: Buy, sell, open, close positions
    - risk_analysis: Risk assessment, scenario analysis
    - research: News, earnings, fundamentals
    - portfolio_review: Performance, positions, P&L
    - education: Explain concepts, definitions
    - market_data: Quotes, prices, volume
    """

    def __init__(self):
        """Initialize semantic router with intent patterns"""

        # Intent patterns (keywords and phrases)
        self.intent_patterns = {
            "trade_execution": [
                r"\b(buy|sell|execute|place|open|close|enter|exit)\s+(order|position|trade)\b",
                r"\b(long|short|buy|sell)\s+\d+\s+(shares|contracts)\b",
                r"\bexecute\s+(this|that|the)\s+trade\b",
                r"\bplace\s+an?\s+order\b",
                r"\b(sell|buy)\s+(call|put)\s+option",
                r"\bwhat.*if.*execute",
                r"\bshould\s+i\s+(buy|sell)",
            ],
            "risk_analysis": [
                r"\b(risk|downside|upside|max\s+loss|max\s+profit|break[-\s]?even)\b",
                r"\bwhat.*if.*drop|fall|rise|increase|decrease",
                r"\b(worst|best)\s+case",
                r"\bhow\s+much.*lose|gain",
                r"\b(greeks|delta|gamma|theta|vega|rho)\b",
                r"\bscenario\s+analysis\b",
                r"\b(var|value\s+at\s+risk|drawdown)\b",
            ],
            "research": [
                r"\b(news|earnings|catalyst|fundamental|analyst)\b",
                r"\bwhat.*happening\s+with\b",
                r"\b(why|what).*stock\s+(up|down|moving)\b",
                r"\bshow.*news\s+(for|about)\b",
                r"\b(quarterly|annual)\s+(report|results)\b",
                r"\b(revenue|earnings|profit)\s+(growth|decline)\b",
                r"\b(p/e|pe\s+ratio|market\s+cap)\b",
            ],
            "portfolio_review": [
                r"\b(portfolio|positions|holdings)\b",
                r"\bhow.*doing|performing",
                r"\bshow.*my\s+(positions|portfolio|p&l|pnl)\b",
                r"\b(total|overall)\s+(return|profit|loss|pnl|p&l)\b",
                r"\b(sharpe|sortino|calmar)\s+ratio\b",
                r"\bperformance\s+(summary|review)\b",
            ],
            "education": [
                r"\bwhat\s+(is|are|does|do)\b",
                r"\bexplain\b",
                r"\b(definition|meaning)\s+of\b",
                r"\bhow\s+(does|do|to)\b",
                r"\bteach\s+me\b",
                r"\b(eli5|explain\s+like\s+i'?m\s+5)\b",
            ],
            "market_data": [
                r"\b(price|quote|current|latest)\s+(of|for)\b",
                r"\bwhat.*trading\s+at\b",
                r"\bshow.*price",
                r"\b(volume|open\s+interest|iv\s+rank)\b",
                r"\b(bid|ask|last)\s+price\b",
            ],
        }

    async def classify_intent(self, message: str) -> Dict[str, any]:
        """
        Classify user message into intent

        Args:
            message: User's natural language message

        Returns:
            Dictionary with intent and confidence
        """

        message_lower = message.lower()

        # Score each intent
        scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    score += 1
            scores[intent] = score

        # Get best intent
        if not scores or max(scores.values()) == 0:
            # No clear intent, default to general conversation
            return {
                "intent": "general",
                "confidence": 0.5,
                "scores": scores
            }

        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        # Calculate confidence (normalize by number of patterns)
        max_possible = len(self.intent_patterns[best_intent])
        confidence = min(1.0, best_score / max(1, max_possible * 0.5))  # Scale confidence

        return {
            "intent": best_intent,
            "confidence": confidence,
            "scores": scores
        }

    def add_intent_pattern(self, intent: str, pattern: str):
        """Add a new pattern to an existing intent"""

        if intent not in self.intent_patterns:
            self.intent_patterns[intent] = []

        self.intent_patterns[intent].append(pattern)
        logger.info(f"Added pattern to {intent}: {pattern}")
