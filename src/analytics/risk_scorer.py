"""
Risk Scorer - Calculate risk score (0-100, lower is better)
"""
import logging
from typing import Dict, Any, Optional
import yfinance as yf
import numpy as np

logger = logging.getLogger(__name__)


class RiskScorer:
    """
    Calculate risk score (0-100, lower is better)

    Components:
    1. Volatility risk (30%)
    2. Beta risk (20%)
    3. Liquidity risk (15%)
    4. Concentration risk (20%)
    5. Correlation risk (15%)
    """

    def __init__(self):
        self.weights = {
            'volatility': 0.30,
            'beta': 0.20,
            'liquidity': 0.15,
            'concentration': 0.20,
            'correlation': 0.15
        }

    def calculate_score(
        self,
        symbol: str,
        position: Optional[Dict[str, Any]] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> 'ScoreResult':
        """
        Calculate risk score for symbol

        Returns:
            ScoreResult with score (0-100, lower is better)
        """
        logger.info(f"Calculating risk score for {symbol}")

        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period='6mo')

            # Calculate component scores
            vol_score, vol_signals = self._volatility_risk(hist, info)
            beta_score, beta_signals = self._beta_risk(info)
            liq_score, liq_signals = self._liquidity_risk(info, hist)
            conc_score, conc_signals = self._concentration_risk(position)
            corr_score, corr_signals = self._correlation_risk(symbol)

            # Weighted combination
            total_score = (
                vol_score * self.weights['volatility'] +
                beta_score * self.weights['beta'] +
                liq_score * self.weights['liquidity'] +
                conc_score * self.weights['concentration'] +
                corr_score * self.weights['correlation']
            )

            # Combine all signals
            all_signals = {
                **vol_signals,
                **beta_signals,
                **liq_signals,
                **conc_signals,
                **corr_signals
            }

            # Generate reasoning
            reasoning = self._generate_reasoning(
                total_score, vol_score, beta_score, liq_score, conc_score
            )

            # Calculate confidence
            confidence = self._calculate_confidence(info, all_signals)

            from .recommendation_engine import ScoreResult
            return ScoreResult(
                score=total_score,
                components={
                    'volatility': vol_score,
                    'beta': beta_score,
                    'liquidity': liq_score,
                    'concentration': conc_score,
                    'correlation': corr_score
                },
                signals=all_signals,
                reasoning=reasoning,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Error calculating risk score for {symbol}: {e}")
            return self._create_default_result(f"Error: {str(e)}")

    def score(self, symbol: str, position: Optional[Dict[str, Any]] = None, market_data: Optional[Dict[str, Any]] = None) -> 'ScoreResult':
        """Compatibility wrapper for tests: delegate to calculate_score with position and market_data."""
        return self.calculate_score(symbol, position=position, market_data=market_data)

    def _volatility_risk(self, hist, info) -> tuple:
        """Calculate volatility risk (higher volatility = higher risk)"""
        score = 50  # Start neutral
        signals = {}

        if len(hist) > 20:
            # Calculate historical volatility (annualized)
            returns = hist['Close'].pct_change().dropna()
            hist_vol = returns.std() * np.sqrt(252)  # Annualized

            if hist_vol < 0.20:  # <20%
                score = 20
                signals['volatility'] = 'low'
            elif hist_vol < 0.35:  # <35%
                score = 40
                signals['volatility'] = 'moderate'
            elif hist_vol < 0.50:  # <50%
                score = 60
                signals['volatility'] = 'high'
            else:  # >50%
                score = 80
                signals['volatility'] = 'very_high'

        return score, signals

    def _beta_risk(self, info) -> tuple:
        """Calculate beta risk (higher beta = higher risk)"""
        score = 50  # Start neutral
        signals = {}

        beta = info.get('beta')
        if beta is not None:
            if beta < 0.8:
                score = 20
                signals['beta'] = 'low'
            elif beta < 1.2:
                score = 40
                signals['beta'] = 'moderate'
            elif beta < 1.5:
                score = 60
                signals['beta'] = 'high'
            else:
                score = 80
                signals['beta'] = 'very_high'

        return score, signals

    def _liquidity_risk(self, info, hist) -> tuple:
        """Calculate liquidity risk (lower volume = higher risk)"""
        score = 50  # Start neutral
        signals = {}

        # Average volume
        avg_volume = info.get('averageVolume')
        if avg_volume:
            if avg_volume > 10_000_000:  # 10M+
                score = 10
                signals['liquidity'] = 'excellent'
            elif avg_volume > 1_000_000:  # 1M+
                score = 30
                signals['liquidity'] = 'good'
            elif avg_volume > 100_000:  # 100K+
                score = 60
                signals['liquidity'] = 'moderate'
            else:
                score = 90
                signals['liquidity'] = 'poor'

        return score, signals

    def _concentration_risk(self, position: Optional[Dict]) -> tuple:
        """Calculate concentration risk (larger position = higher risk)"""
        score = 50  # Start neutral
        signals = {}

        if not position:
            signals['concentration'] = 'no_position'
            return 0, signals

        # For now, assume we don't have portfolio value
        # In production, calculate position_value / portfolio_value
        signals['concentration'] = 'unknown'

        return score, signals

    def _correlation_risk(self, symbol: str) -> tuple:
        """Calculate correlation risk (high correlation = higher risk)"""
        score = 50  # Start neutral
        signals = {}

        # Placeholder - would calculate correlation with portfolio
        signals['correlation'] = 'unknown'

        return score, signals

    def _generate_reasoning(
        self,
        total: float,
        vol: float,
        beta: float,
        liq: float,
        conc: float
    ) -> str:
        """Generate human-readable reasoning"""

        if total < 30:
            risk_level = "Low risk"
        elif total < 50:
            risk_level = "Moderate risk"
        elif total < 70:
            risk_level = "High risk"
        else:
            risk_level = "Very high risk"

        return f"{risk_level} (Vol: {vol:.0f}, Beta: {beta:.0f}, Liq: {liq:.0f})"

    def _calculate_confidence(self, info: Dict, signals: Dict) -> float:
        """Calculate confidence based on data availability"""
        available = sum(1 for v in signals.values() if v not in ['unknown', 'unavailable'])
        total = len(signals)
        return available / total if total > 0 else 0.0

    def _create_default_result(self, reason: str) -> 'ScoreResult':
        """Create default result when data is unavailable"""
        from .recommendation_engine import ScoreResult
        return ScoreResult(
            score=50.0,
            components={'volatility': 50, 'beta': 50, 'liquidity': 50, 'concentration': 50, 'correlation': 50},
            signals={'status': 'unavailable'},
            reasoning=reason,
            confidence=0.0
        )

