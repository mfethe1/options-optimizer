"""
Unit Tests for Phase 4 Technical & Cross-Asset Metrics

Tests all Phase 4 functions with synthetic data to ensure:
- Correct calculations
- Graceful degradation on missing data
- Performance <200ms per asset
- Proper interpretation generation
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.analytics.technical_cross_asset import (
    options_flow_composite,
    compute_options_flow_metrics,
    residual_momentum,
    seasonality_score,
    breadth_liquidity,
    compute_phase4_metrics
)


class TestOptionsFlowComposite:
    """Test options flow composite score calculation"""
    
    def test_bullish_flow(self):
        """Test bullish options flow (low PCR, negative skew, high volume)"""
        score = options_flow_composite(
            pcr=0.5,  # Low PCR = bullish
            iv_skew=-0.05,  # Negative skew = bullish
            volume_ratio=2.0  # High volume = conviction
        )
        assert score > 0.5, "Should be strongly bullish"
        assert score <= 1.0, "Should be capped at 1.0"
    
    def test_bearish_flow(self):
        """Test bearish options flow (high PCR, positive skew, high volume)"""
        score = options_flow_composite(
            pcr=1.2,  # High PCR = bearish
            iv_skew=0.15,  # Positive skew = bearish
            volume_ratio=2.0  # High volume = conviction
        )
        assert score < -0.5, "Should be strongly bearish"
        assert score >= -1.0, "Should be capped at -1.0"
    
    def test_neutral_flow(self):
        """Test neutral options flow"""
        score = options_flow_composite(
            pcr=0.7,  # Neutral PCR
            iv_skew=0.05,  # Neutral skew
            volume_ratio=1.0  # Average volume
        )
        assert -0.3 < score < 0.3, "Should be near neutral"
    
    def test_low_volume_dampening(self):
        """Test that low volume dampens the signal"""
        high_vol_score = options_flow_composite(pcr=0.5, iv_skew=-0.05, volume_ratio=2.0)
        low_vol_score = options_flow_composite(pcr=0.5, iv_skew=-0.05, volume_ratio=0.5)
        
        assert abs(low_vol_score) < abs(high_vol_score), "Low volume should dampen signal"
    
    def test_metrics_object(self):
        """Test full metrics object with interpretation"""
        metrics = compute_options_flow_metrics(pcr=0.5, iv_skew=-0.05, volume_ratio=2.0)
        
        assert metrics.pcr == 0.5
        assert metrics.iv_skew == -0.05
        assert metrics.volume_ratio == 2.0
        assert metrics.composite_score > 0.5
        assert "bullish" in metrics.interpretation.lower()


class TestResidualMomentum:
    """Test residual momentum calculation"""
    
    def test_positive_residual(self):
        """Test positive idiosyncratic momentum"""
        # Asset outperforms market
        asset_returns = np.array([0.02, 0.03, 0.01, 0.04, 0.02] * 4)  # 20 days
        market_returns = np.array([0.01, 0.01, 0.00, 0.01, 0.01] * 4)
        
        z_score = residual_momentum(asset_returns, market_returns)
        assert z_score > 0, "Should have positive residual momentum"
    
    def test_negative_residual(self):
        """Test negative idiosyncratic momentum"""
        # Asset underperforms market
        asset_returns = np.array([0.00, -0.01, -0.02, 0.00, -0.01] * 4)
        market_returns = np.array([0.01, 0.01, 0.00, 0.01, 0.01] * 4)
        
        z_score = residual_momentum(asset_returns, market_returns)
        assert z_score < 0, "Should have negative residual momentum"
    
    def test_with_sector(self):
        """Test residual momentum with sector adjustment"""
        asset_returns = np.array([0.02, 0.03, 0.01, 0.04, 0.02] * 4)
        market_returns = np.array([0.01, 0.01, 0.00, 0.01, 0.01] * 4)
        sector_returns = np.array([0.015, 0.015, 0.005, 0.015, 0.015] * 4)
        
        z_score = residual_momentum(asset_returns, market_returns, sector_returns)
        assert isinstance(z_score, float)
    
    def test_insufficient_data(self):
        """Test graceful handling of insufficient data"""
        asset_returns = np.array([0.01, 0.02])  # Only 2 days
        market_returns = np.array([0.01, 0.01])
        
        z_score = residual_momentum(asset_returns, market_returns, lookback=20)
        assert z_score == 0.0, "Should return 0 for insufficient data"


class TestSeasonalityScore:
    """Test seasonality score calculation"""
    
    def test_turn_of_month_effect(self):
        """Test turn-of-month seasonality detection"""
        # Create 90 days of returns with turn-of-month bias
        dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
        returns = []
        
        for date in dates:
            if date.day >= 28 or date.day <= 2:
                returns.append(0.01)  # Positive at turn of month
            else:
                returns.append(0.0)  # Neutral otherwise
        
        returns_series = pd.Series(returns, index=dates)
        score = seasonality_score(returns_series, calendar_effects=['turn_of_month'])
        
        assert score > 0, "Should detect positive turn-of-month effect"
    
    def test_day_of_week_effect(self):
        """Test day-of-week seasonality"""
        dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
        returns = []
        
        for date in dates:
            if date.dayofweek == 0:  # Monday
                returns.append(0.01)
            else:
                returns.append(0.0)
        
        returns_series = pd.Series(returns, index=dates)
        score = seasonality_score(returns_series, calendar_effects=['day_of_week'])
        
        # Score depends on whether last day was Monday
        assert isinstance(score, float)
        assert -1 <= score <= 1
    
    def test_insufficient_data(self):
        """Test graceful handling of insufficient data"""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        returns_series = pd.Series(np.random.randn(30) * 0.01, index=dates)
        
        score = seasonality_score(returns_series)
        assert score == 0.0, "Should return 0 for insufficient data"
    
    def test_non_datetime_index(self):
        """Test graceful handling of non-datetime index"""
        returns_series = pd.Series([0.01, 0.02, -0.01])
        score = seasonality_score(returns_series)
        assert score == 0.0, "Should return 0 for non-datetime index"


class TestBreadthLiquidity:
    """Test market breadth & liquidity composite"""
    
    def test_strong_breadth(self):
        """Test strong positive breadth"""
        score = breadth_liquidity(
            advancing=800,
            declining=200,
            volume_ratio=1.5
        )
        assert score > 0.5, "Should show strong positive breadth"
    
    def test_weak_breadth(self):
        """Test weak negative breadth"""
        score = breadth_liquidity(
            advancing=200,
            declining=800,
            volume_ratio=1.5
        )
        assert score < -0.5, "Should show weak negative breadth"
    
    def test_with_liquidity(self):
        """Test breadth with liquidity component"""
        score_tight = breadth_liquidity(
            advancing=600,
            declining=400,
            volume_ratio=1.5,
            spread_bps=5  # Tight spread = good liquidity
        )
        score_wide = breadth_liquidity(
            advancing=600,
            declining=400,
            volume_ratio=1.5,
            spread_bps=25  # Wide spread = poor liquidity
        )
        
        assert score_tight > score_wide, "Tighter spreads should improve score"
    
    def test_zero_stocks(self):
        """Test graceful handling of zero stocks"""
        score = breadth_liquidity(advancing=0, declining=0, volume_ratio=1.0)
        assert score == 0.0, "Should return 0 for zero stocks"


class TestComputePhase4Metrics:
    """Test full Phase 4 metrics computation"""
    
    def test_all_metrics_available(self):
        """Test when all data is available"""
        asset_returns = np.random.randn(60) * 0.01
        market_returns = np.random.randn(60) * 0.01
        dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
        returns_series = pd.Series(asset_returns, index=dates)
        
        metrics = compute_phase4_metrics(
            pcr=0.7,
            iv_skew=0.05,
            volume_ratio=1.2,
            asset_returns=asset_returns,
            market_returns=market_returns,
            returns_series=returns_series,
            advancing=600,
            declining=400,
            market_volume_ratio=1.3
        )
        
        assert metrics['options_flow_composite'] is not None
        assert metrics['residual_momentum'] is not None
        assert metrics['seasonality_score'] is not None
        assert metrics['breadth_liquidity'] is not None
    
    def test_partial_data(self):
        """Test graceful degradation with partial data"""
        metrics = compute_phase4_metrics(
            pcr=0.7,
            iv_skew=0.05,
            volume_ratio=1.2
            # Missing other data
        )
        
        assert metrics['options_flow_composite'] is not None
        assert metrics['residual_momentum'] is None
        assert metrics['seasonality_score'] is None
        assert metrics['breadth_liquidity'] is None
    
    def test_no_data(self):
        """Test with no data"""
        metrics = compute_phase4_metrics()
        
        assert all(v is None for v in metrics.values())


class TestPerformance:
    """Test performance requirements"""
    
    def test_options_flow_performance(self):
        """Test options flow computation is fast"""
        import time
        
        start = time.time()
        for _ in range(100):
            options_flow_composite(pcr=0.7, iv_skew=0.05, volume_ratio=1.2)
        elapsed = time.time() - start
        
        avg_time_ms = (elapsed / 100) * 1000
        assert avg_time_ms < 10, f"Should be <10ms per call, got {avg_time_ms:.2f}ms"
    
    def test_full_phase4_performance(self):
        """Test full Phase 4 computation is <200ms"""
        import time
        
        asset_returns = np.random.randn(60) * 0.01
        market_returns = np.random.randn(60) * 0.01
        dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
        returns_series = pd.Series(asset_returns, index=dates)
        
        start = time.time()
        compute_phase4_metrics(
            pcr=0.7,
            iv_skew=0.05,
            volume_ratio=1.2,
            asset_returns=asset_returns,
            market_returns=market_returns,
            returns_series=returns_series,
            advancing=600,
            declining=400,
            market_volume_ratio=1.3
        )
        elapsed_ms = (time.time() - start) * 1000
        
        assert elapsed_ms < 200, f"Should be <200ms, got {elapsed_ms:.2f}ms"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

