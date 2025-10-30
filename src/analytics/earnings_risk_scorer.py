"""
Earnings Risk Scorer - Calculate earnings-related risk (0-100, lower is better)
"""
import logging
from typing import Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EarningsRiskScorer:
    """
    Calculate earnings risk score (0-100, lower is better)

    Components:
    1. Time to earnings (25%)
    2. Historical volatility (30%)
    3. Implied vs historical (25%)
    4. Estimate spread (15%)
    5. Guidance history (5%)
    """

    def __init__(self):
        self.weights = {
            'time_risk': 0.25,
            'volatility_risk': 0.30,
            'implied_risk': 0.25,
            'estimate_risk': 0.15,
            'guidance_risk': 0.05
        }

    def calculate_score(self, symbol: str) -> 'ScoreResult':
        """
        Calculate earnings risk score for symbol

        Returns:
            ScoreResult with score (0-100, lower is better)
        """
        logger.info(f"Calculating earnings risk score for {symbol}")

        try:
            # Get earnings data
            from ..services.earnings_service import EarningsService
            earnings_service = EarningsService()

            # Get next earnings date
            try:
                next_earnings = earnings_service.get_next_earnings(symbol)
                earnings_date = next_earnings.get('date') if next_earnings else None
            except:
                earnings_date = None

            # Calculate component scores
            time_score, time_signals = self._time_risk(earnings_date)
            vol_score, vol_signals = self._volatility_risk(symbol)
            implied_score, implied_signals = self._implied_risk(symbol, earnings_date)
            estimate_score, estimate_signals = self._estimate_risk(symbol)
            guidance_score, guidance_signals = self._guidance_risk(symbol)

            # Weighted combination
            total_score = (
                time_score * self.weights['time_risk'] +
                vol_score * self.weights['volatility_risk'] +
                implied_score * self.weights['implied_risk'] +
                estimate_score * self.weights['estimate_risk'] +
                guidance_score * self.weights['guidance_risk']
            )

            # Combine all signals
            all_signals = {
                **time_signals,
                **vol_signals,
                **implied_signals,
                **estimate_signals,
                **guidance_signals
            }

            # Generate reasoning
            reasoning = self._generate_reasoning(
                total_score, time_score, earnings_date
            )

            # Calculate confidence
            confidence = self._calculate_confidence(all_signals)

            from .recommendation_engine import ScoreResult
            return ScoreResult(
                score=total_score,
                components={
                    'time_risk': time_score,
                    'volatility_risk': vol_score,
                    'implied_risk': implied_score,
                    'estimate_risk': estimate_score,
                    'guidance_risk': guidance_score
                },
                signals=all_signals,
                reasoning=reasoning,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Error calculating earnings risk for {symbol}: {e}")
            return self._create_default_result(f"Error: {str(e)}")

    def score(self, symbol: str, position: Dict[str, Any] | None = None, market_data: Dict[str, Any] | None = None) -> 'ScoreResult':
        """Compatibility wrapper for tests: delegate to calculate_score."""
        return self.calculate_score(symbol)

    def _time_risk(self, earnings_date: str) -> tuple:
        """Calculate risk based on time to earnings"""
        score = 0  # Start with no risk
        signals = {}

        if not earnings_date:
            signals['time_to_earnings'] = 'unknown'
            return score, signals

        try:
            # Parse earnings date
            if isinstance(earnings_date, str):
                earnings_dt = datetime.fromisoformat(earnings_date.replace('Z', '+00:00'))
            else:
                earnings_dt = earnings_date

            # Calculate days to earnings
            days_to_earnings = (earnings_dt - datetime.now()).days

            if days_to_earnings < 0:
                score = 0
                signals['time_to_earnings'] = 'past'
            elif days_to_earnings < 3:
                score = 90
                signals['time_to_earnings'] = 'imminent'
            elif days_to_earnings < 7:
                score = 70
                signals['time_to_earnings'] = 'very_soon'
            elif days_to_earnings < 14:
                score = 50
                signals['time_to_earnings'] = 'soon'
            elif days_to_earnings < 30:
                score = 30
                signals['time_to_earnings'] = 'approaching'
            else:
                score = 10
                signals['time_to_earnings'] = 'distant'

        except Exception as e:
            logger.warning(f"Error parsing earnings date: {e}")
            signals['time_to_earnings'] = 'error'

        return score, signals

    def _volatility_risk(self, symbol: str) -> tuple:
        """Calculate risk based on historical earnings volatility"""
        score = 50  # Start neutral
        signals = {}

        # Placeholder - would analyze historical earnings moves
        signals['historical_volatility'] = 'unknown'

        return score, signals

    def _implied_risk(self, symbol: str, earnings_date: str) -> tuple:
        """Calculate risk based on implied move vs historical"""
        score = 50  # Start neutral
        signals = {}

        # Placeholder - would calculate implied move from options
        signals['implied_move'] = 'unknown'

        return score, signals

    def _estimate_risk(self, symbol: str) -> tuple:
        """Calculate risk based on estimate spread"""
        score = 50  # Start neutral
        signals = {}

        # Placeholder - would analyze analyst estimate spread
        signals['estimate_spread'] = 'unknown'

        return score, signals

    def _guidance_risk(self, symbol: str) -> tuple:
        """Calculate risk based on guidance history"""
        score = 50  # Start neutral
        signals = {}

        # Placeholder - would analyze guidance history
        signals['guidance_history'] = 'unknown'

        return score, signals

    def _generate_reasoning(
        self,
        total: float,
        time_score: float,
        earnings_date: str
    ) -> str:
        """Generate human-readable reasoning"""

        if total < 30:
            risk_level = "Low earnings risk"
        elif total < 50:
            risk_level = "Moderate earnings risk"
        elif total < 70:
            risk_level = "High earnings risk"
        else:
            risk_level = "Very high earnings risk"

        if earnings_date:
            return f"{risk_level} (Next earnings: {earnings_date})"
        else:
            return f"{risk_level} (No upcoming earnings)"

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
            components={
                'time_risk': 50,
                'volatility_risk': 50,
                'implied_risk': 50,
                'estimate_risk': 50,
                'guidance_risk': 50
            },
            signals={'status': 'unavailable'},
            reasoning=reason,
            confidence=0.0
        )

