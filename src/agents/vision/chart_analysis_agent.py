"""
Chart Analysis Agent

Uses GPT-4 Vision to analyze charts, screenshots, and technical patterns.
Enables users to upload images for AI-powered analysis.
"""

import os
import logging
import base64
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

from openai import OpenAI
from anthropic import Anthropic

logger = logging.getLogger(__name__)


class ChartAnalysisAgent:
    """
    COMPETITIVE ADVANTAGE: Vision-based chart analysis

    First options platform to offer AI-powered chart image analysis:
    - Upload screenshots from TradingView, ToS, Webull, etc.
    - Pattern recognition (head & shoulders, triangles, etc.)
    - Support/resistance level identification
    - Options flow visualization analysis
    - Social media chart verification (FinTwit screenshots)

    Use Cases:
    1. Analyze chart screenshots from Twitter/Discord
    2. Verify claims from influencers
    3. Multi-chart comparison analysis
    4. Extract insights from YouTube thumbnails
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        preferred_provider: str = "openai"
    ):
        """
        Initialize chart analysis agent

        Args:
            openai_api_key: OpenAI API key (for GPT-4 Vision)
            anthropic_api_key: Anthropic API key (for Claude 3.5 Sonnet Vision)
            preferred_provider: Which provider to use ("openai" or "anthropic")
        """
        self.preferred_provider = preferred_provider

        # Initialize clients
        if openai_api_key or os.getenv("OPENAI_API_KEY"):
            self.openai_client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        else:
            self.openai_client = None

        if anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"):
            self.anthropic_client = Anthropic(api_key=anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"))
        else:
            self.anthropic_client = None

        logger.info(f"ChartAnalysisAgent initialized with provider: {preferred_provider}")

    async def analyze_chart(
        self,
        image_path: str,
        question: Optional[str] = None,
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Analyze a chart image

        Args:
            image_path: Path to chart image
            question: Specific question about the chart
            analysis_type: Type of analysis ("comprehensive", "pattern", "levels", "flow")

        Returns:
            Dictionary with analysis results
        """

        logger.info(f"Analyzing chart: {image_path}")

        # Read and encode image
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error reading image: {e}")
            return {"error": f"Failed to read image: {str(e)}"}

        # Determine file type
        file_ext = Path(image_path).suffix.lower()
        media_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }.get(file_ext, "image/png")

        # Build prompt based on analysis type
        prompt = self._build_analysis_prompt(analysis_type, question)

        # Use preferred provider
        if self.preferred_provider == "openai" and self.openai_client:
            return await self._analyze_with_openai(image_data, media_type, prompt)
        elif self.preferred_provider == "anthropic" and self.anthropic_client:
            return await self._analyze_with_anthropic(image_data, media_type, prompt)
        else:
            # Fallback
            if self.openai_client:
                return await self._analyze_with_openai(image_data, media_type, prompt)
            elif self.anthropic_client:
                return await self._analyze_with_anthropic(image_data, media_type, prompt)
            else:
                return {"error": "No vision API client available"}

    def _build_analysis_prompt(self, analysis_type: str, question: Optional[str]) -> str:
        """Build analysis prompt based on type"""

        base_prompts = {
            "comprehensive": """Analyze this chart comprehensively and provide:

1. **Chart Pattern Recognition**: Identify any technical patterns (head & shoulders, double top/bottom, triangles, flags, channels)
2. **Support and Resistance Levels**: Key price levels where the stock tends to bounce or reverse
3. **Trend Analysis**: Overall trend (uptrend, downtrend, sideways), trend strength
4. **Technical Indicators**: Interpret any visible indicators (RSI, MACD, Moving Averages, Volume)
5. **Options Flow** (if visible): Unusual options activity, put/call ratio, high volume strikes
6. **Trading Recommendation**: Based on the chart, suggest potential trades (calls, puts, spreads)
7. **Risk Assessment**: Key risks and important levels to watch

Provide your analysis in JSON format with keys: patterns, levels, trend, indicators, options_flow, recommendation, risks""",

            "pattern": """Focus on identifying chart patterns in this image:

1. What patterns do you see? (head & shoulders, double top/bottom, triangles, wedges, flags, pennants)
2. Are these patterns bullish or bearish?
3. What are the key price levels for these patterns?
4. What's the expected move if the pattern plays out?
5. What would invalidate this pattern?

Format as JSON with keys: patterns, bias, key_levels, expected_move, invalidation""",

            "levels": """Identify support and resistance levels in this chart:

1. **Major Support Levels**: Strong price floors (list with exact prices)
2. **Major Resistance Levels**: Strong price ceilings (list with exact prices)
3. **Current Price Position**: Is price at support, resistance, or between?
4. **Next Key Levels**: Where is the next support/resistance if price moves?
5. **Volume Profile** (if visible): High volume price levels

Format as JSON with keys: support_levels, resistance_levels, current_position, next_levels, volume_profile""",

            "flow": """Analyze options flow and unusual activity in this chart (if visible):

1. **Unusual Volume**: Any strikes with abnormally high volume?
2. **Large Trades**: Any block trades or large orders?
3. **Put/Call Ratio**: Are calls or puts dominant?
4. **Expiration Concentration**: Which expiration dates have the most activity?
5. **Smart Money Indicators**: Evidence of institutional/smart money positioning?
6. **Trading Implication**: What does this options activity suggest about future price movement?

Format as JSON with keys: unusual_volume, large_trades, put_call_ratio, expirations, smart_money, trading_implication"""
        }

        prompt = base_prompts.get(analysis_type, base_prompts["comprehensive"])

        if question:
            prompt += f"\n\nAdditional specific question: {question}"

        return prompt

    async def _analyze_with_openai(
        self,
        image_data: str,
        media_type: str,
        prompt: str
    ) -> Dict[str, Any]:
        """Analyze chart using GPT-4 Vision"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.3
            )

            content = response.choices[0].message.content

            # Try to parse as JSON
            try:
                result = json.loads(content)
                result['provider'] = 'openai'
            except json.JSONDecodeError:
                result = {
                    "analysis": content,
                    "provider": "openai"
                }

            return result

        except Exception as e:
            logger.error(f"Error with OpenAI vision: {e}")
            return {"error": str(e), "provider": "openai"}

    async def _analyze_with_anthropic(
        self,
        image_data: str,
        media_type: str,
        prompt: str
    ) -> Dict[str, Any]:
        """Analyze chart using Claude 3.5 Sonnet Vision"""

        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            content = response.content[0].text

            # Try to parse as JSON
            try:
                result = json.loads(content)
                result['provider'] = 'anthropic'
            except json.JSONDecodeError:
                result = {
                    "analysis": content,
                    "provider": "anthropic"
                }

            return result

        except Exception as e:
            logger.error(f"Error with Anthropic vision: {e}")
            return {"error": str(e), "provider": "anthropic"}

    async def compare_charts(
        self,
        image_paths: List[str],
        comparison_type: str = "relative_strength"
    ) -> Dict[str, Any]:
        """
        Compare multiple charts

        Args:
            image_paths: List of chart image paths
            comparison_type: Type of comparison ("relative_strength", "divergence", "correlation")

        Returns:
            Comparison analysis
        """

        if len(image_paths) > 4:
            return {"error": "Maximum 4 charts for comparison"}

        # Encode all images
        images = []
        for path in image_paths:
            try:
                with open(path, "rb") as f:
                    data = base64.b64encode(f.read()).decode('utf-8')
                    ext = Path(path).suffix.lower()
                    media_type = {"jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(ext, "image/png")
                    images.append({"data": data, "media_type": media_type})
            except Exception as e:
                logger.error(f"Error reading {path}: {e}")
                continue

        if not images:
            return {"error": "No valid images to compare"}

        prompt = f"""Compare these {len(images)} charts and provide a {comparison_type} analysis:

1. **Individual Assessment**: Briefly describe each chart
2. **Comparison**: How do these charts compare to each other?
3. **Divergences**: Any notable divergences or correlations?
4. **Trading Insight**: What does this comparison tell us about potential trades?

Format as JSON."""

        # Use first available client
        if self.anthropic_client:
            # Claude can handle multiple images
            content_blocks = []
            for img in images:
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img['media_type'],
                        "data": img['data']
                    }
                })
            content_blocks.append({"type": "text", "text": prompt})

            try:
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": content_blocks}]
                )
                content = response.content[0].text
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"analysis": content}
            except Exception as e:
                return {"error": str(e)}

        return {"error": "No vision client available for multi-chart comparison"}
