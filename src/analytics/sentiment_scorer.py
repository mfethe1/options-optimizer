"""
Sentiment Scorer - Calculate sentiment score with correlation awareness (0-100)
"""
import logging
from typing import Dict, Any, Optional
import requests
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SentimentScorer:
    """
    Calculate sentiment score from multiple sources + correlated stocks

    Components:
    1. Direct sentiment (60%)
       - News sentiment
       - Social sentiment
       - Analyst sentiment
    2. Correlated stock sentiment (25%)
    3. Emerging trends (15%)
    """

    def __init__(self):
        self.weights = {
            'direct': 0.60,
            'correlated': 0.25,
            'emerging': 0.15
        }

        # API keys
        self.firecrawl_key = os.getenv('FIRECRAWL_API_KEY')
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')

    def calculate_score(self, symbol: str) -> 'ScoreResult':
        """
        Calculate sentiment score for symbol

        Returns:
            ScoreResult with score, components, signals, reasoning
        """
        logger.info(f"Calculating sentiment score for {symbol}")

        try:
            # Get direct sentiment
            direct_score, direct_signals = self._get_direct_sentiment(symbol)

            # Get correlated stocks sentiment
            correlated_score, corr_signals = self._get_correlated_sentiment(symbol)

            # Detect emerging trends
            emerging_score, emerging_signals = self._detect_emerging_trends(symbol)

            # Weighted combination
            total_score = (
                direct_score * self.weights['direct'] +
                correlated_score * self.weights['correlated'] +
                emerging_score * self.weights['emerging']
            )

            # Combine all signals
            all_signals = {
                **direct_signals,
                **corr_signals,
                **emerging_signals
            }

            # Generate reasoning
            reasoning = self._generate_reasoning(
                total_score, direct_score, correlated_score, emerging_score
            )

            # Calculate confidence
            confidence = self._calculate_confidence(all_signals)

            from .recommendation_engine import ScoreResult
            return ScoreResult(
                score=total_score,
                components={
                    'direct': direct_score,
                    'correlated': correlated_score,
                    'emerging': emerging_score
                },
                signals=all_signals,
                reasoning=reasoning,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Error calculating sentiment score for {symbol}: {e}")
            return self._create_default_result(f"Error: {str(e)}")

    def score(self, symbol: str, position: Optional[Dict[str, Any]] = None, market_data: Optional[Dict[str, Any]] = None) -> 'ScoreResult':
        """Compatibility wrapper for tests: delegate to calculate_score."""
        return self.calculate_score(symbol)

    def _get_direct_sentiment(self, symbol: str) -> tuple:
        """
        Get direct sentiment from news, social, analysts

        For now, uses research service data
        Later: Integrate Firecrawl, Reddit, etc.
        """
        score = 50  # Start neutral
        signals = {}

        # Try to get research data from our service (optional)
        try:
            from ..services.research_service import ResearchService
            research_service = ResearchService()
            research_data = research_service.research_symbol(symbol)

            if research_data and isinstance(research_data, dict):
                # News sentiment
                news_sentiment = research_data.get('news', {}).get('sentiment', 'neutral')
                if news_sentiment == 'positive':
                    score += 15
                    signals['news'] = 'positive'
                elif news_sentiment == 'negative':
                    score -= 15
                    signals['news'] = 'negative'
                else:
                    signals['news'] = 'neutral'

                # Social sentiment
                social_sentiment = research_data.get('social', {}).get('overall_sentiment', 'neutral')
                if social_sentiment == 'positive':
                    score += 10
                    signals['social'] = 'positive'
                elif social_sentiment == 'negative':
                    score -= 10
                    signals['social'] = 'negative'
                else:
                    signals['social'] = 'neutral'
            else:
                logger.debug(f"No research data available for {symbol}")
                signals['news'] = 'unavailable'
                signals['social'] = 'unavailable'

        except Exception as e:
            logger.debug(f"Research service unavailable (this is OK): {e}")
            signals['news'] = 'unavailable'
            signals['social'] = 'unavailable'

        # Analyst sentiment (from yfinance)
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info

            recommendation = info.get('recommendationKey')
            if recommendation in ['strong_buy', 'buy']:
                score += 25
                signals['analyst'] = 'bullish'
            elif recommendation == 'hold':
                signals['analyst'] = 'neutral'
            elif recommendation in ['sell', 'strong_sell']:
                score -= 25
                signals['analyst'] = 'bearish'

        except Exception as e:
            logger.warning(f"Could not fetch analyst data: {e}")
            signals['analyst'] = 'unavailable'

        return max(0, min(100, score)), signals

    def _get_correlated_sentiment(self, symbol: str) -> tuple:
        """
        Get sentiment from correlated stocks

        Process:
        1. Get correlated stocks (sector peers)
        2. Check their sentiment
        3. Aggregate impact on target symbol
        """
        score = 50  # Start neutral
        signals = {}

        # Get correlated stocks
        correlated = self._get_correlated_stocks(symbol)

        if not correlated:
            signals['correlated'] = 'unavailable'
            return score, signals

        # Check sentiment for each correlated stock
        bullish_count = 0
        bearish_count = 0

        for peer in correlated[:5]:  # Top 5 correlated
            try:
                import yfinance as yf
                ticker = yf.Ticker(peer)
                info = ticker.info

                recommendation = info.get('recommendationKey')
                if recommendation in ['strong_buy', 'buy']:
                    bullish_count += 1
                elif recommendation in ['sell', 'strong_sell']:
                    bearish_count += 1

            except Exception as e:
                logger.warning(f"Could not fetch data for {peer}: {e}")
                continue

        # Calculate sector sentiment
        if bullish_count > bearish_count:
            score += 20
            signals['sector_sentiment'] = 'bullish'
        elif bearish_count > bullish_count:
            score -= 20
            signals['sector_sentiment'] = 'bearish'
        else:
            signals['sector_sentiment'] = 'neutral'

        return max(0, min(100, score)), signals

    def _detect_emerging_trends(self, symbol: str) -> tuple:
        """
        Detect emerging trends from correlated stocks

        For now, placeholder
        Later: Implement headline analysis with LLM
        """
        score = 50  # Start neutral
        signals = {}

        signals['emerging_trends'] = 'not_implemented'

        return score, signals

    def _get_correlated_stocks(self, symbol: str) -> list:
        """
        Get correlated stocks (sector peers)

        For now, hardcoded mappings
        Later: Calculate from price correlation
        """
        # Hardcoded sector peers
        correlations = {
            'NVDA': ['AMD', 'INTC', 'TSM', 'AVGO', 'QCOM', 'MU'],
            'AMD': ['NVDA', 'INTC', 'TSM', 'AVGO', 'QCOM'],
            'AAPL': ['MSFT', 'GOOGL', 'META', 'AMZN'],
            'MSFT': ['AAPL', 'GOOGL', 'META', 'AMZN'],
            'TSLA': ['RIVN', 'LCID', 'NIO', 'F', 'GM'],
        }

        return correlations.get(symbol, [])

    def _generate_reasoning(
        self,
        total: float,
        direct: float,
        correlated: float,
        emerging: float
    ) -> str:
        """Generate human-readable reasoning"""

        if total >= 70:
            sentiment = "Strong positive"
        elif total >= 55:
            sentiment = "Positive"
        elif total >= 45:
            sentiment = "Neutral"
        elif total >= 30:
            sentiment = "Negative"
        else:
            sentiment = "Strong negative"

        return f"{sentiment} sentiment (Direct: {direct:.0f}, Sector: {correlated:.0f}, Trends: {emerging:.0f})"

    def _calculate_confidence(self, signals: Dict) -> float:
        """Calculate confidence based on data availability"""

        # Count available signals
        available = sum(1 for v in signals.values() if v != 'unavailable' and v != 'not_implemented')
        total = len(signals)

        return available / total if total > 0 else 0.0

    def _create_default_result(self, reason: str) -> 'ScoreResult':
        """Create default result when data is unavailable"""
        from .recommendation_engine import ScoreResult
        return ScoreResult(
            score=50.0,
            components={'direct': 50, 'correlated': 50, 'emerging': 50},
            signals={'status': 'unavailable'},
            reasoning=reason,
            confidence=0.0
        )

