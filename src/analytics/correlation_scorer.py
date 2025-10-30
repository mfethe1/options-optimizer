"""
Correlation Scorer - Analyze correlated stocks for sector trends (0-100)
"""
import logging
from typing import Dict, Any, List
import yfinance as yf
import numpy as np

logger = logging.getLogger(__name__)


class CorrelationScorer:
    """
    Calculate correlation score based on sector trends

    Process:
    1. Identify correlated stocks (sector peers)
    2. Analyze their performance and sentiment
    3. Detect sector-wide trends
    4. Calculate impact on target symbol
    """

    def __init__(self):
        # Hardcoded sector mappings (later: calculate dynamically)
        self.sector_peers = {
            'NVDA': ['AMD', 'INTC', 'TSM', 'AVGO', 'QCOM', 'MU'],
            'AMD': ['NVDA', 'INTC', 'TSM', 'AVGO', 'QCOM'],
            'AAPL': ['MSFT', 'GOOGL', 'META', 'AMZN'],
            'MSFT': ['AAPL', 'GOOGL', 'META', 'AMZN'],
            'TSLA': ['RIVN', 'LCID', 'NIO', 'F', 'GM'],
            'GOOGL': ['MSFT', 'AAPL', 'META', 'AMZN'],
            'META': ['GOOGL', 'MSFT', 'AAPL', 'SNAP', 'PINS'],
            'AMZN': ['MSFT', 'GOOGL', 'AAPL', 'WMT', 'TGT'],
        }

    def calculate_score(self, symbol: str) -> 'ScoreResult':
        """
        Calculate correlation score for symbol

        Returns:
            ScoreResult with score based on sector trends
        """
        logger.info(f"Calculating correlation score for {symbol}")

        try:
            # Get correlated stocks
            peers = self.sector_peers.get(symbol, [])

            if not peers:
                logger.warning(f"No correlated stocks found for {symbol}")
                return self._create_default_result("No sector peers defined")

            # Analyze sector performance
            sector_score, sector_signals = self._analyze_sector_performance(symbol, peers)

            # Analyze sector sentiment
            sentiment_score, sentiment_signals = self._analyze_sector_sentiment(peers)

            # Detect divergence
            divergence_score, div_signals = self._detect_divergence(symbol, peers)

            # Combined score
            total_score = (
                sector_score * 0.50 +
                sentiment_score * 0.30 +
                divergence_score * 0.20
            )

            # Combine all signals
            all_signals = {
                **sector_signals,
                **sentiment_signals,
                **div_signals
            }

            # Generate reasoning
            reasoning = self._generate_reasoning(
                total_score, sector_score, sentiment_score, peers
            )

            # Calculate confidence
            confidence = self._calculate_confidence(all_signals)

            from .recommendation_engine import ScoreResult
            return ScoreResult(
                score=total_score,
                components={
                    'sector_performance': sector_score,
                    'sector_sentiment': sentiment_score,
                    'divergence': divergence_score
                },
                signals=all_signals,
                reasoning=reasoning,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Error calculating correlation score for {symbol}: {e}")
            return self._create_default_result(f"Error: {str(e)}")

    def score(self, symbol: str, position: Dict[str, Any] | None = None, market_data: Dict[str, Any] | None = None) -> 'ScoreResult':
        """Compatibility wrapper for tests: delegate to calculate_score."""
        return self.calculate_score(symbol)

    def _analyze_sector_performance(self, symbol: str, peers: List[str]) -> tuple:
        """Analyze performance of sector peers"""
        score = 50  # Start neutral
        signals = {}

        try:
            # Get recent performance for peers
            peer_returns = []

            for peer in peers[:5]:  # Top 5 peers
                try:
                    ticker = yf.Ticker(peer)
                    hist = ticker.history(period='1mo')

                    if len(hist) > 1:
                        # Calculate 1-month return
                        ret = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
                        peer_returns.append(ret)

                except Exception as e:
                    logger.warning(f"Could not fetch data for {peer}: {e}")
                    continue

            if peer_returns:
                # Calculate average sector return
                avg_return = np.mean(peer_returns)

                if avg_return > 10:
                    score = 80
                    signals['sector_trend'] = 'strong_bullish'
                elif avg_return > 5:
                    score = 65
                    signals['sector_trend'] = 'bullish'
                elif avg_return > 0:
                    score = 55
                    signals['sector_trend'] = 'slightly_bullish'
                elif avg_return > -5:
                    score = 45
                    signals['sector_trend'] = 'slightly_bearish'
                elif avg_return > -10:
                    score = 35
                    signals['sector_trend'] = 'bearish'
                else:
                    score = 20
                    signals['sector_trend'] = 'strong_bearish'

                signals['sector_avg_return'] = f"{avg_return:.1f}%"
            else:
                signals['sector_trend'] = 'unavailable'

        except Exception as e:
            logger.warning(f"Error analyzing sector performance: {e}")
            signals['sector_trend'] = 'error'

        return score, signals

    def _analyze_sector_sentiment(self, peers: List[str]) -> tuple:
        """Analyze sentiment of sector peers"""
        score = 50  # Start neutral
        signals = {}

        try:
            bullish_count = 0
            bearish_count = 0
            total_count = 0

            for peer in peers[:5]:
                try:
                    ticker = yf.Ticker(peer)
                    info = ticker.info

                    recommendation = info.get('recommendationKey')
                    if recommendation:
                        total_count += 1
                        if recommendation in ['strong_buy', 'buy']:
                            bullish_count += 1
                        elif recommendation in ['sell', 'strong_sell']:
                            bearish_count += 1

                except Exception as e:
                    logger.warning(f"Could not fetch sentiment for {peer}: {e}")
                    continue

            if total_count > 0:
                bullish_pct = bullish_count / total_count

                if bullish_pct > 0.7:
                    score = 75
                    signals['sector_sentiment'] = 'strong_bullish'
                elif bullish_pct > 0.5:
                    score = 60
                    signals['sector_sentiment'] = 'bullish'
                elif bullish_pct > 0.3:
                    score = 50
                    signals['sector_sentiment'] = 'neutral'
                else:
                    score = 35
                    signals['sector_sentiment'] = 'bearish'
            else:
                signals['sector_sentiment'] = 'unavailable'

        except Exception as e:
            logger.warning(f"Error analyzing sector sentiment: {e}")
            signals['sector_sentiment'] = 'error'

        return score, signals

    def _detect_divergence(self, symbol: str, peers: List[str]) -> tuple:
        """Detect if symbol is diverging from sector"""
        score = 50  # Start neutral
        signals = {}

        try:
            # Get symbol performance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1mo')

            if len(hist) > 1:
                symbol_return = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100

                # Get sector average (from previous calculation)
                # For now, just mark as no divergence
                signals['divergence'] = 'none'
                score = 50
            else:
                signals['divergence'] = 'unavailable'

        except Exception as e:
            logger.warning(f"Error detecting divergence: {e}")
            signals['divergence'] = 'error'

        return score, signals

    def _generate_reasoning(
        self,
        total: float,
        sector: float,
        sentiment: float,
        peers: List[str]
    ) -> str:
        """Generate human-readable reasoning"""

        if total >= 70:
            trend = "Strong sector tailwind"
        elif total >= 55:
            trend = "Positive sector trend"
        elif total >= 45:
            trend = "Neutral sector trend"
        elif total >= 30:
            trend = "Negative sector trend"
        else:
            trend = "Strong sector headwind"

        peer_list = ', '.join(peers[:3])
        return f"{trend} (Peers: {peer_list})"

    def _calculate_confidence(self, signals: Dict) -> float:
        """Calculate confidence based on data availability"""
        available = sum(1 for v in signals.values() if v not in ['unknown', 'unavailable', 'error'])
        total = len(signals)
        return available / total if total > 0 else 0.0

    def _create_default_result(self, reason: str) -> 'ScoreResult':
        """Create default result when data is unavailable"""
        from .recommendation_engine import ScoreResult
        return ScoreResult(
            score=50.0,
            components={'sector_performance': 50, 'sector_sentiment': 50, 'divergence': 50},
            signals={'status': 'unavailable'},
            reasoning=reason,
            confidence=0.0
        )

