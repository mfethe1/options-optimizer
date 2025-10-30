"""
Fundamental Scorer - Calculate fundamental analysis score (0-100)
"""
import logging
from typing import Dict, Any, Optional
import yfinance as yf

logger = logging.getLogger(__name__)


class FundamentalScorer:
    """
    Calculate fundamental score from financial metrics

    Components:
    1. Valuation (25%)
    2. Growth (30%)
    3. Profitability (25%)
    4. Financial Health (15%)
    5. Competitive Position (5%)
    """

    def __init__(self):
        self.weights = {
            'valuation': 0.25,
            'growth': 0.30,
            'profitability': 0.25,
            'financial_health': 0.15,
            'competitive': 0.05
        }

    def calculate_score(
        self,
        symbol: str,
        market_data: Optional[Dict[str, Any]] = None
    ) -> 'ScoreResult':
        """
        Calculate fundamental score for symbol

        Returns:
            ScoreResult with score, components, signals, reasoning
        """
        logger.info(f"Calculating fundamental score for {symbol}")

        try:
            # Fetch fundamental data
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info or len(info) < 5:
                logger.warning(f"Insufficient fundamental data for {symbol}")
                return self._create_default_result("Insufficient data")

            # Calculate component scores
            valuation_score, val_signals = self._valuation_score(info)
            growth_score, growth_signals = self._growth_score(info)
            profit_score, profit_signals = self._profitability_score(info)
            health_score, health_signals = self._financial_health_score(info)
            competitive_score, comp_signals = self._competitive_score(info)

            # Weighted combination
            total_score = (
                valuation_score * self.weights['valuation'] +
                growth_score * self.weights['growth'] +
                profit_score * self.weights['profitability'] +
                health_score * self.weights['financial_health'] +
                competitive_score * self.weights['competitive']
            )

            # Combine all signals
            all_signals = {
                **val_signals,
                **growth_signals,
                **profit_signals,
                **health_signals,
                **comp_signals
            }

            # Generate reasoning
            reasoning = self._generate_reasoning(
                total_score, valuation_score, growth_score,
                profit_score, health_score
            )

            # Calculate confidence
            confidence = self._calculate_confidence(info, all_signals)

            from .recommendation_engine import ScoreResult
            return ScoreResult(
                score=total_score,
                components={
                    'valuation': valuation_score,
                    'growth': growth_score,
                    'profitability': profit_score,
                    'financial_health': health_score,
                    'competitive': competitive_score
                },
                signals=all_signals,
                reasoning=reasoning,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Error calculating fundamental score for {symbol}: {e}")
            return self._create_default_result(f"Error: {str(e)}")

    def score(self, symbol: str, position: Optional[Dict[str, Any]] = None, market_data: Optional[Dict[str, Any]] = None) -> 'ScoreResult':
        """Compatibility wrapper for tests: delegate to calculate_score, passing market_data if provided."""
        return self.calculate_score(symbol, market_data=market_data)

    def _valuation_score(self, info: Dict) -> tuple:
        """
        Calculate valuation score (0-100)

        Metrics:
        - P/E ratio
        - PEG ratio
        - P/S ratio
        - P/B ratio
        """
        score = 50  # Start neutral
        signals = {}

        # P/E ratio
        pe = info.get('trailingPE') or info.get('forwardPE')
        if pe:
            if pe < 15:
                score += 15
                signals['pe'] = 'undervalued'
            elif pe < 25:
                score += 5
                signals['pe'] = 'fair'
            elif pe < 40:
                score -= 5
                signals['pe'] = 'expensive'
            else:
                score -= 15
                signals['pe'] = 'overvalued'

        # PEG ratio
        peg = info.get('pegRatio')
        if peg:
            if peg < 1.0:
                score += 15
                signals['peg'] = 'undervalued'
            elif peg < 1.5:
                score += 5
                signals['peg'] = 'fair'
            elif peg < 2.0:
                score -= 5
                signals['peg'] = 'expensive'
            else:
                score -= 15
                signals['peg'] = 'overvalued'

        # P/S ratio
        ps = info.get('priceToSalesTrailing12Months')
        if ps:
            if ps < 2:
                score += 10
                signals['ps'] = 'undervalued'
            elif ps < 5:
                signals['ps'] = 'fair'
            else:
                score -= 10
                signals['ps'] = 'expensive'

        # P/B ratio
        pb = info.get('priceToBook')
        if pb:
            if pb < 2:
                score += 10
                signals['pb'] = 'undervalued'
            elif pb < 5:
                signals['pb'] = 'fair'
            else:
                score -= 10
                signals['pb'] = 'expensive'

        return max(0, min(100, score)), signals

    def _growth_score(self, info: Dict) -> tuple:
        """
        Calculate growth score (0-100)

        Metrics:
        - Revenue growth
        - Earnings growth
        - Analyst estimates
        """
        score = 50  # Start neutral
        signals = {}

        # Revenue growth
        revenue_growth = info.get('revenueGrowth')
        if revenue_growth:
            if revenue_growth > 0.30:  # 30%+
                score += 20
                signals['revenue_growth'] = 'strong'
            elif revenue_growth > 0.15:  # 15%+
                score += 10
                signals['revenue_growth'] = 'good'
            elif revenue_growth > 0:
                score += 5
                signals['revenue_growth'] = 'positive'
            else:
                score -= 15
                signals['revenue_growth'] = 'negative'

        # Earnings growth
        earnings_growth = info.get('earningsGrowth')
        if earnings_growth:
            if earnings_growth > 0.30:
                score += 20
                signals['earnings_growth'] = 'strong'
            elif earnings_growth > 0.15:
                score += 10
                signals['earnings_growth'] = 'good'
            elif earnings_growth > 0:
                score += 5
                signals['earnings_growth'] = 'positive'
            else:
                score -= 15
                signals['earnings_growth'] = 'negative'

        # Analyst target price
        current_price = info.get('currentPrice')
        target_price = info.get('targetMeanPrice')

        if current_price and target_price:
            upside = ((target_price - current_price) / current_price) * 100
            if upside > 20:
                score += 10
                signals['analyst_target'] = 'bullish'
            elif upside > 10:
                score += 5
                signals['analyst_target'] = 'positive'
            elif upside < -10:
                score -= 10
                signals['analyst_target'] = 'bearish'

        return max(0, min(100, score)), signals

    def _profitability_score(self, info: Dict) -> tuple:
        """
        Calculate profitability score (0-100)

        Metrics:
        - ROE
        - ROA
        - Profit margins
        """
        score = 50  # Start neutral
        signals = {}

        # ROE
        roe = info.get('returnOnEquity')
        if roe:
            if roe > 0.20:  # 20%+
                score += 15
                signals['roe'] = 'excellent'
            elif roe > 0.15:
                score += 10
                signals['roe'] = 'good'
            elif roe > 0.10:
                score += 5
                signals['roe'] = 'fair'
            else:
                score -= 10
                signals['roe'] = 'poor'

        # ROA
        roa = info.get('returnOnAssets')
        if roa:
            if roa > 0.10:
                score += 10
                signals['roa'] = 'excellent'
            elif roa > 0.05:
                score += 5
                signals['roa'] = 'good'
            elif roa < 0:
                score -= 10
                signals['roa'] = 'poor'

        # Profit margins
        profit_margin = info.get('profitMargins')
        if profit_margin:
            if profit_margin > 0.20:
                score += 15
                signals['profit_margin'] = 'excellent'
            elif profit_margin > 0.10:
                score += 10
                signals['profit_margin'] = 'good'
            elif profit_margin > 0.05:
                score += 5
                signals['profit_margin'] = 'fair'
            else:
                score -= 10
                signals['profit_margin'] = 'poor'

        # Operating margin
        operating_margin = info.get('operatingMargins')
        if operating_margin:
            if operating_margin > 0.20:
                score += 10
                signals['operating_margin'] = 'excellent'
            elif operating_margin > 0.10:
                score += 5
                signals['operating_margin'] = 'good'

        return max(0, min(100, score)), signals

    def _financial_health_score(self, info: Dict) -> tuple:
        """
        Calculate financial health score (0-100)

        Metrics:
        - Debt to equity
        - Current ratio
        - Quick ratio
        - Free cash flow
        """
        score = 50  # Start neutral
        signals = {}

        # Debt to equity
        debt_to_equity = info.get('debtToEquity')
        if debt_to_equity is not None:
            if debt_to_equity < 0.5:
                score += 15
                signals['debt'] = 'low'
            elif debt_to_equity < 1.0:
                score += 5
                signals['debt'] = 'moderate'
            elif debt_to_equity < 2.0:
                score -= 5
                signals['debt'] = 'high'
            else:
                score -= 15
                signals['debt'] = 'very_high'

        # Current ratio
        current_ratio = info.get('currentRatio')
        if current_ratio:
            if current_ratio > 2.0:
                score += 10
                signals['liquidity'] = 'strong'
            elif current_ratio > 1.5:
                score += 5
                signals['liquidity'] = 'good'
            elif current_ratio < 1.0:
                score -= 10
                signals['liquidity'] = 'weak'

        # Quick ratio
        quick_ratio = info.get('quickRatio')
        if quick_ratio:
            if quick_ratio > 1.5:
                score += 10
                signals['quick_ratio'] = 'strong'
            elif quick_ratio > 1.0:
                score += 5
                signals['quick_ratio'] = 'good'
            elif quick_ratio < 0.5:
                score -= 10
                signals['quick_ratio'] = 'weak'

        # Free cash flow
        free_cash_flow = info.get('freeCashflow')
        if free_cash_flow:
            if free_cash_flow > 0:
                score += 15
                signals['cash_flow'] = 'positive'
            else:
                score -= 15
                signals['cash_flow'] = 'negative'

        return max(0, min(100, score)), signals

    def _competitive_score(self, info: Dict) -> tuple:
        """
        Calculate competitive position score (0-100)

        Metrics:
        - Market cap (size/stability)
        - Analyst recommendations
        """
        score = 50  # Start neutral
        signals = {}

        # Market cap
        market_cap = info.get('marketCap')
        if market_cap:
            if market_cap > 200e9:  # $200B+ (mega cap)
                score += 20
                signals['market_cap'] = 'mega_cap'
            elif market_cap > 10e9:  # $10B+ (large cap)
                score += 10
                signals['market_cap'] = 'large_cap'
            elif market_cap > 2e9:  # $2B+ (mid cap)
                signals['market_cap'] = 'mid_cap'
            else:
                score -= 10
                signals['market_cap'] = 'small_cap'

        # Analyst recommendations
        recommendation = info.get('recommendationKey')
        if recommendation:
            if recommendation in ['strong_buy', 'buy']:
                score += 30
                signals['analyst_rec'] = 'buy'
            elif recommendation == 'hold':
                signals['analyst_rec'] = 'hold'
            else:
                score -= 20
                signals['analyst_rec'] = 'sell'

        return max(0, min(100, score)), signals

    def _generate_reasoning(
        self,
        total: float,
        valuation: float,
        growth: float,
        profitability: float,
        health: float
    ) -> str:
        """Generate human-readable reasoning"""

        if total >= 70:
            assessment = "Strong fundamentals"
        elif total >= 55:
            assessment = "Good fundamentals"
        elif total >= 45:
            assessment = "Mixed fundamentals"
        elif total >= 30:
            assessment = "Weak fundamentals"
        else:
            assessment = "Poor fundamentals"

        return f"{assessment} (Val: {valuation:.0f}, Growth: {growth:.0f}, Profit: {profitability:.0f}, Health: {health:.0f})"

    def _calculate_confidence(self, info: Dict, signals: Dict) -> float:
        """Calculate confidence based on data availability"""

        # Count available metrics
        key_metrics = [
            'trailingPE', 'forwardPE', 'pegRatio', 'revenueGrowth',
            'earningsGrowth', 'returnOnEquity', 'profitMargins',
            'debtToEquity', 'currentRatio', 'marketCap'
        ]

        available = sum(1 for metric in key_metrics if info.get(metric) is not None)
        data_quality = available / len(key_metrics)

        return data_quality

    def _create_default_result(self, reason: str) -> 'ScoreResult':
        """Create default result when data is unavailable"""
        from .recommendation_engine import ScoreResult
        return ScoreResult(
            score=50.0,
            components={
                'valuation': 50,
                'growth': 50,
                'profitability': 50,
                'financial_health': 50,
                'competitive': 50
            },
            signals={'status': 'unavailable'},
            reasoning=reason,
            confidence=0.0
        )

