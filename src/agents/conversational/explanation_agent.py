"""
Explanation Agent

Provides educational content and explains complex trading concepts.
Supports ELI5 (Explain Like I'm 5) mode for beginners.
"""

import os
import logging
from typing import Dict, Any, Optional
from anthropic import Anthropic

logger = logging.getLogger(__name__)


class ExplanationAgent:
    """
    COMPETITIVE ADVANTAGE: Educational content generation

    Transforms complex options concepts into clear, understandable explanations:
    - ELI5 mode for beginners
    - Technical mode for advanced traders
    - Practical examples with real scenarios
    - Common misconceptions and pitfalls

    Example topics:
    - "What is implied volatility?"
    - "How does theta decay work?"
    - "What's the difference between Delta and Gamma?"
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize explanation agent"""
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

        self.system_prompt = """You are an expert trading educator who excels at explaining complex concepts clearly.

Your teaching style:
1. Start with the simplest possible explanation (ELI5 level)
2. Use analogies and real-world examples
3. Build up to technical details progressively
4. Highlight common misconceptions
5. Provide practical applications

Structure your explanations as JSON with:
- "simple": ELI5 explanation (1-2 sentences)
- "detailed": Technical explanation (2-3 paragraphs)
- "example": Real-world scenario
- "misconceptions": Common mistakes (list)
- "related_topics": Related concepts to explore (list)
"""

    async def explain(
        self,
        topic: str,
        complexity: str = "medium"
    ) -> Dict[str, Any]:
        """
        Explain a trading concept

        Args:
            topic: Concept to explain (e.g., "implied volatility", "iron condor")
            complexity: Level of detail ("simple", "medium", "advanced")

        Returns:
            Dictionary with explanations at different levels
        """

        complexity_prompts = {
            "simple": "Explain this as if I'm 10 years old and never traded before.",
            "medium": "Explain this for someone who understands basic stock trading but is new to options.",
            "advanced": "Provide a detailed technical explanation for an experienced options trader."
        }

        prompt = f"""Explain: {topic}

{complexity_prompts.get(complexity, complexity_prompts["medium"])}

Provide a comprehensive explanation in JSON format."""

        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1500,
                temperature=0.3,  # Low temperature for consistent educational content
                system=self.system_prompt,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            content = response.content[0].text

            # Parse JSON
            import json
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Fallback if not valid JSON
                result = {
                    "simple": content[:200],
                    "detailed": content,
                    "example": "See detailed explanation",
                    "misconceptions": [],
                    "related_topics": []
                }

            return result

        except Exception as e:
            logger.error(f"Error explaining topic: {e}")
            return {
                "simple": f"Error explaining {topic}: {str(e)}",
                "detailed": "Please try again or contact support.",
                "example": "",
                "misconceptions": [],
                "related_topics": []
            }

    async def explain_strategy(
        self,
        strategy_name: str,
        market_condition: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Explain an options strategy

        Args:
            strategy_name: Strategy to explain (e.g., "covered call", "iron condor")
            market_condition: Market condition context (e.g., "bullish", "high IV")

        Returns:
            Strategy explanation with setup, risks, and best use cases
        """

        market_context = f" in a {market_condition} market" if market_condition else ""

        prompt = f"""Explain the {strategy_name} strategy{market_context}.

Provide:
1. Strategy overview (what it is, how it works)
2. Setup (legs, strikes, expirations)
3. Max profit and max loss
4. Break-even points
5. Best market conditions to use it
6. Risks and management
7. Real example with numbers

Format as JSON."""

        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.3,
                system=self.system_prompt,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            content = response.content[0].text

            import json
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                result = {"explanation": content}

            return result

        except Exception as e:
            logger.error(f"Error explaining strategy: {e}")
            return {"error": str(e)}
