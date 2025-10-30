"""
Integration Test: Portfolio Metrics + Phase 4

Verifies that Phase 4 metrics are properly integrated into PortfolioMetrics
and that calculate_all_metrics() computes them with graceful degradation.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.analytics.portfolio_metrics import PortfolioAnalytics, PortfolioMetrics


class TestPortfolioPhase4Integration:
    """Test Phase 4 integration with portfolio metrics"""
    
    def test_phase4_fields_exist(self):
        """Test that PortfolioMetrics has Phase 4 fields"""
        # Create synthetic data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        portfolio_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
        benchmark_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
        position_weights = pd.Series({'AAPL': 0.5, 'MSFT': 0.5})
        position_returns = pd.DataFrame({
            'AAPL': np.random.randn(100) * 0.01,
            'MSFT': np.random.randn(100) * 0.01
        }, index=dates)
        
        # Calculate metrics
        analytics = PortfolioAnalytics()
        metrics = analytics.calculate_all_metrics(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            position_weights=position_weights,
            position_returns=position_returns
        )
        
        # Verify Phase 4 fields exist
        assert hasattr(metrics, 'options_flow_composite')
        assert hasattr(metrics, 'residual_momentum')
        assert hasattr(metrics, 'seasonality_score')
        assert hasattr(metrics, 'breadth_liquidity')
        assert hasattr(metrics, 'data_sources')
        assert hasattr(metrics, 'as_of')
    
    def test_residual_momentum_computed(self):
        """Test that residual momentum is computed when data is sufficient"""
        # Create synthetic data with 100 days (>= 20 required)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        portfolio_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
        benchmark_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
        position_weights = pd.Series({'AAPL': 0.5, 'MSFT': 0.5})
        position_returns = pd.DataFrame({
            'AAPL': np.random.randn(100) * 0.01,
            'MSFT': np.random.randn(100) * 0.01
        }, index=dates)
        
        # Calculate metrics
        analytics = PortfolioAnalytics()
        metrics = analytics.calculate_all_metrics(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            position_weights=position_weights,
            position_returns=position_returns
        )
        
        # Residual momentum should be computed (not None)
        assert metrics.residual_momentum is not None
        assert isinstance(metrics.residual_momentum, float)
        assert 'computed' in metrics.data_sources
    
    def test_seasonality_computed(self):
        """Test that seasonality is computed when data is sufficient"""
        # Create synthetic data with 100 days (>= 60 required)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        portfolio_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
        benchmark_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
        position_weights = pd.Series({'AAPL': 0.5, 'MSFT': 0.5})
        position_returns = pd.DataFrame({
            'AAPL': np.random.randn(100) * 0.01,
            'MSFT': np.random.randn(100) * 0.01
        }, index=dates)
        
        # Calculate metrics
        analytics = PortfolioAnalytics()
        metrics = analytics.calculate_all_metrics(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            position_weights=position_weights,
            position_returns=position_returns
        )
        
        # Seasonality should be computed (not None)
        assert metrics.seasonality_score is not None
        assert isinstance(metrics.seasonality_score, float)
        assert metrics.seasonality_score >= -1.0
        assert metrics.seasonality_score <= 1.0
    
    def test_graceful_degradation_insufficient_data(self):
        """Test graceful degradation when data is insufficient"""
        # Create synthetic data with only 10 days (< 20 required for residual momentum)
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        portfolio_returns = pd.Series(np.random.randn(10) * 0.01, index=dates)
        benchmark_returns = pd.Series(np.random.randn(10) * 0.01, index=dates)
        position_weights = pd.Series({'AAPL': 0.5, 'MSFT': 0.5})
        position_returns = pd.DataFrame({
            'AAPL': np.random.randn(10) * 0.01,
            'MSFT': np.random.randn(10) * 0.01
        }, index=dates)
        
        # Calculate metrics
        analytics = PortfolioAnalytics()
        metrics = analytics.calculate_all_metrics(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            position_weights=position_weights,
            position_returns=position_returns
        )
        
        # Phase 4 metrics should be None (insufficient data)
        assert metrics.residual_momentum is None
        assert metrics.seasonality_score is None
        assert metrics.options_flow_composite is None
        assert metrics.breadth_liquidity is None
        
        # Data sources should still be populated
        assert len(metrics.data_sources) > 0
    
    def test_options_flow_and_breadth_null(self):
        """Test that options flow and breadth/liquidity are None (require external data)"""
        # Create synthetic data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        portfolio_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
        benchmark_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
        position_weights = pd.Series({'AAPL': 0.5, 'MSFT': 0.5})
        position_returns = pd.DataFrame({
            'AAPL': np.random.randn(100) * 0.01,
            'MSFT': np.random.randn(100) * 0.01
        }, index=dates)
        
        # Calculate metrics
        analytics = PortfolioAnalytics()
        metrics = analytics.calculate_all_metrics(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            position_weights=position_weights,
            position_returns=position_returns
        )
        
        # Options flow and breadth/liquidity should be None (require external data)
        assert metrics.options_flow_composite is None
        assert metrics.breadth_liquidity is None
    
    def test_as_of_timestamp(self):
        """Test that as_of timestamp is populated"""
        # Create synthetic data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        portfolio_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
        benchmark_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
        position_weights = pd.Series({'AAPL': 0.5, 'MSFT': 0.5})
        position_returns = pd.DataFrame({
            'AAPL': np.random.randn(100) * 0.01,
            'MSFT': np.random.randn(100) * 0.01
        }, index=dates)
        
        # Calculate metrics
        analytics = PortfolioAnalytics()
        metrics = analytics.calculate_all_metrics(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            position_weights=position_weights,
            position_returns=position_returns
        )
        
        # as_of should be a valid ISO 8601 timestamp
        assert metrics.as_of is not None
        assert isinstance(metrics.as_of, str)
        
        # Should be parseable as datetime
        parsed_dt = datetime.fromisoformat(metrics.as_of.replace('Z', '+00:00'))
        assert isinstance(parsed_dt, datetime)
    
    def test_performance_target(self):
        """Test that Phase 4 computation meets <200ms per asset target"""
        import time
        
        # Create synthetic data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        portfolio_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
        benchmark_returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
        position_weights = pd.Series({'AAPL': 0.5, 'MSFT': 0.5})
        position_returns = pd.DataFrame({
            'AAPL': np.random.randn(100) * 0.01,
            'MSFT': np.random.randn(100) * 0.01
        }, index=dates)
        
        # Measure time
        analytics = PortfolioAnalytics()
        start_time = time.time()
        metrics = analytics.calculate_all_metrics(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            position_weights=position_weights,
            position_returns=position_returns
        )
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Should complete in <200ms per asset (2 assets = 400ms budget)
        assert elapsed_ms < 400, f"Computation took {elapsed_ms:.2f}ms (budget: 400ms for 2 assets)"
        
        # Verify metrics were computed
        assert metrics.residual_momentum is not None or metrics.seasonality_score is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

