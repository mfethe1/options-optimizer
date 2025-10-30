"""
Institutional-Grade Portfolio Analytics

Calculates advanced metrics used by hedge funds and institutional investors:
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Alpha, Beta, R-Squared
- Maximum Drawdown, Value at Risk (VaR)
- Correlation Matrix
- Information Ratio, Treynor Ratio
- Portfolio diversification metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

# Import Phase 4 metrics
from .technical_cross_asset import compute_phase4_metrics

logger = logging.getLogger(__name__)


@dataclass
class PortfolioMetrics:
    """Container for all portfolio metrics"""
    
    # Return Metrics
    total_return: float
    annualized_return: float
    cagr: float
    
    # Risk Metrics
    volatility: float
    downside_deviation: float
    max_drawdown: float
    var_95: float  # Value at Risk (95% confidence)
    cvar_95: float  # Conditional VaR
    
    # Risk-Adjusted Returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    treynor_ratio: float
    information_ratio: float
    
    # Market Sensitivity
    alpha: float
    beta: float
    r_squared: float
    
    # Diversification
    correlation_matrix: pd.DataFrame
    concentration_risk: float  # Herfindahl index
    effective_n: float  # Effective number of positions
    
    # Additional Metrics
    win_rate: float
    profit_factor: float
    recovery_factor: float

    # ==================== Advanced Renaissance-Level Metrics ====================

    # Enhanced Risk Metrics
    omega_ratio: float  # Tail risk measure (>1.0 good, >2.0 excellent)
    upside_capture: float  # % of benchmark gains captured
    downside_capture: float  # % of benchmark losses captured
    pain_index: float  # Ulcer index - drawdown depth & duration
    gh1_ratio: float  # Return enhancement + risk reduction vs benchmark

    # ==================== Phase 4: Technical & Cross-Asset Metrics ====================
    # Short-horizon (1-5 day) edge from options flow, momentum, seasonality, breadth
    # Sources: ExtractAlpha CAM, Cboe, academic research on options-equity lead-lag

    options_flow_composite: Optional[float]  # PCR + IV skew + volume (-1 to +1)
    residual_momentum: Optional[float]  # Idiosyncratic momentum z-score
    seasonality_score: Optional[float]  # Calendar patterns (-1 to +1)
    breadth_liquidity: Optional[float]  # Market internals (-1 to +1)

    # Metadata
    data_sources: List[str]  # Authoritative sources used
    as_of: str  # ISO 8601 timestamp


class PortfolioAnalytics:
    """
    Institutional-grade portfolio analytics engine
    
    Calculates comprehensive risk and performance metrics used by
    hedge funds, asset managers, and institutional investors.
    """
    
    def __init__(self, risk_free_rate: float = 0.04):
        """
        Initialize analytics engine
        
        Args:
            risk_free_rate: Annual risk-free rate (default 4% for 2025)
        """
        self.risk_free_rate = risk_free_rate
        logger.info(f"📊 Portfolio Analytics initialized (risk-free rate: {risk_free_rate:.2%})")
    
    def calculate_all_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        position_weights: pd.Series,
        position_returns: pd.DataFrame
    ) -> PortfolioMetrics:
        """
        Calculate all portfolio metrics
        
        Args:
            portfolio_returns: Time series of portfolio returns
            benchmark_returns: Time series of benchmark returns (e.g., S&P 500)
            position_weights: Current position weights
            position_returns: DataFrame of individual position returns
            
        Returns:
            PortfolioMetrics object with all calculated metrics
        """
        logger.info("📈 Calculating institutional-grade portfolio metrics...")
        
        # Return Metrics
        total_return = self._calculate_total_return(portfolio_returns)
        annualized_return = self._calculate_annualized_return(portfolio_returns)
        cagr = self._calculate_cagr(portfolio_returns)
        
        # Risk Metrics
        volatility = self._calculate_volatility(portfolio_returns)
        downside_deviation = self._calculate_downside_deviation(portfolio_returns)
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        var_95 = self._calculate_var(portfolio_returns, confidence=0.95)
        cvar_95 = self._calculate_cvar(portfolio_returns, confidence=0.95)
        
        # Risk-Adjusted Returns
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns, volatility)
        sortino_ratio = self._calculate_sortino_ratio(portfolio_returns, downside_deviation)
        calmar_ratio = self._calculate_calmar_ratio(annualized_return, max_drawdown)
        
        # Market Sensitivity
        alpha, beta, r_squared = self._calculate_alpha_beta(portfolio_returns, benchmark_returns)
        treynor_ratio = self._calculate_treynor_ratio(annualized_return, beta)
        information_ratio = self._calculate_information_ratio(portfolio_returns, benchmark_returns)
        
        # Diversification
        correlation_matrix = self._calculate_correlation_matrix(position_returns)
        concentration_risk = self._calculate_concentration_risk(position_weights)
        effective_n = self._calculate_effective_n(position_weights)
        
        # Additional Metrics
        win_rate = self._calculate_win_rate(portfolio_returns)
        profit_factor = self._calculate_profit_factor(portfolio_returns)
        recovery_factor = self._calculate_recovery_factor(total_return, max_drawdown)

        # Advanced Renaissance-level metrics
        omega_ratio = self._calculate_omega_ratio(portfolio_returns)
        upside_capture, downside_capture = self._calculate_upside_downside_capture(
            portfolio_returns, benchmark_returns
        )
        pain_index = self._calculate_pain_index(portfolio_returns)
        gh1_ratio = self._calculate_gh1_ratio(portfolio_returns, benchmark_returns)

        # Phase 4: Technical & Cross-Asset Metrics (short-horizon edge)
        phase4_metrics, phase4_sources = self._compute_phase4_metrics(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            position_returns=position_returns
        )

        metrics = PortfolioMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            cagr=cagr,
            volatility=volatility,
            downside_deviation=downside_deviation,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            treynor_ratio=treynor_ratio,
            information_ratio=information_ratio,
            alpha=alpha,
            beta=beta,
            r_squared=r_squared,
            correlation_matrix=correlation_matrix,
            concentration_risk=concentration_risk,
            effective_n=effective_n,
            win_rate=win_rate,
            profit_factor=profit_factor,
            recovery_factor=recovery_factor,
            omega_ratio=omega_ratio,
            upside_capture=upside_capture,
            downside_capture=downside_capture,
            pain_index=pain_index,
            gh1_ratio=gh1_ratio,
            # Phase 4 metrics
            options_flow_composite=phase4_metrics.get('options_flow_composite'),
            residual_momentum=phase4_metrics.get('residual_momentum'),
            seasonality_score=phase4_metrics.get('seasonality_score'),
            breadth_liquidity=phase4_metrics.get('breadth_liquidity'),
            data_sources=phase4_sources,
            as_of=datetime.now().isoformat()
        )
        
        logger.info("✅ Portfolio metrics calculated successfully")
        return metrics
    
    # ==================== Return Metrics ====================
    
    def _calculate_total_return(self, returns: pd.Series) -> float:
        """Calculate cumulative total return"""
        return (1 + returns).prod() - 1
    
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return"""
        total_return = self._calculate_total_return(returns)
        n_periods = len(returns)
        periods_per_year = 252  # Trading days
        return (1 + total_return) ** (periods_per_year / n_periods) - 1
    
    def _calculate_cagr(self, returns: pd.Series) -> float:
        """Calculate Compound Annual Growth Rate"""
        return self._calculate_annualized_return(returns)
    
    # ==================== Risk Metrics ====================
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility (standard deviation)"""
        return returns.std() * np.sqrt(252)
    
    def _calculate_downside_deviation(self, returns: pd.Series, mar: float = 0.0) -> float:
        """
        Calculate downside deviation (semi-deviation)
        Only considers returns below Minimum Acceptable Return (MAR)
        """
        downside_returns = returns[returns < mar]
        if len(downside_returns) == 0:
            return 0.0
        return downside_returns.std() * np.sqrt(252)
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def _calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk (VaR) at given confidence level"""
        return abs(np.percentile(returns, (1 - confidence) * 100))
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional VaR (Expected Shortfall)"""
        var = self._calculate_var(returns, confidence)
        return abs(returns[returns <= -var].mean())
    
    # ==================== Risk-Adjusted Returns ====================
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, volatility: float) -> float:
        """Calculate Sharpe Ratio"""
        excess_return = returns.mean() * 252 - self.risk_free_rate
        if volatility == 0:
            return 0.0
        return excess_return / volatility
    
    def _calculate_sortino_ratio(self, returns: pd.Series, downside_deviation: float) -> float:
        """Calculate Sortino Ratio"""
        excess_return = returns.mean() * 252 - self.risk_free_rate
        if downside_deviation == 0:
            return 0.0
        return excess_return / downside_deviation
    
    def _calculate_calmar_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """Calculate Calmar Ratio"""
        if max_drawdown == 0:
            return 0.0
        return annualized_return / max_drawdown
    
    def _calculate_treynor_ratio(self, annualized_return: float, beta: float) -> float:
        """Calculate Treynor Ratio"""
        if beta == 0:
            return 0.0
        excess_return = annualized_return - self.risk_free_rate
        return excess_return / beta
    
    def _calculate_information_ratio(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Information Ratio"""
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(252)
        if tracking_error == 0:
            return 0.0
        return (active_returns.mean() * 252) / tracking_error
    
    # ==================== Market Sensitivity ====================
    
    def _calculate_alpha_beta(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Tuple[float, float, float]:
        """
        Calculate Alpha, Beta, and R-Squared using linear regression
        
        Returns:
            Tuple of (alpha, beta, r_squared)
        """
        # Align series
        aligned = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned) < 2:
            return 0.0, 1.0, 0.0
        
        # Calculate beta (covariance / variance)
        covariance = aligned['portfolio'].cov(aligned['benchmark'])
        variance = aligned['benchmark'].var()
        
        if variance == 0:
            beta = 1.0
        else:
            beta = covariance / variance
        
        # Calculate alpha (excess return)
        portfolio_mean = aligned['portfolio'].mean() * 252
        benchmark_mean = aligned['benchmark'].mean() * 252
        alpha = portfolio_mean - (self.risk_free_rate + beta * (benchmark_mean - self.risk_free_rate))
        
        # Calculate R-squared (correlation^2)
        correlation = aligned['portfolio'].corr(aligned['benchmark'])
        r_squared = correlation ** 2
        
        return alpha, beta, r_squared
    
    # ==================== Diversification ====================
    
    def _calculate_correlation_matrix(self, position_returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix of positions"""
        return position_returns.corr()
    
    def _calculate_concentration_risk(self, weights: pd.Series) -> float:
        """
        Calculate Herfindahl-Hirschman Index (HHI)
        Measures portfolio concentration (0 = perfectly diversified, 1 = single position)
        """
        return (weights ** 2).sum()
    
    def _calculate_effective_n(self, weights: pd.Series) -> float:
        """
        Calculate effective number of positions
        Inverse of HHI - shows how many equal-weighted positions would give same diversification
        """
        hhi = self._calculate_concentration_risk(weights)
        if hhi == 0:
            return 0.0
        return 1.0 / hhi
    
    # ==================== Additional Metrics ====================
    
    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate percentage of positive return periods"""
        if len(returns) == 0:
            return 0.0
        return (returns > 0).sum() / len(returns)
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate ratio of gross profits to gross losses"""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        if losses == 0:
            return float('inf') if gains > 0 else 0.0
        return gains / losses
    
    def _calculate_recovery_factor(self, total_return: float, max_drawdown: float) -> float:
        """Calculate recovery factor (total return / max drawdown)"""
        if max_drawdown == 0:
            return 0.0
        return total_return / max_drawdown

    # ==================== Advanced Risk Metrics (Renaissance-Level) ====================

    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """
        Calculate Omega Ratio - captures tail risk beyond Sharpe

        Omega = (Probability-weighted gains above threshold) / (Probability-weighted losses below threshold)
        Higher Omega = better tail risk profile

        Args:
            returns: Return series
            threshold: Return threshold (default 0%)

        Returns:
            Omega ratio (>1.0 is good, >2.0 is excellent)
        """
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]

        if losses.sum() == 0:
            return float('inf') if gains.sum() > 0 else 0.0

        return gains.sum() / losses.sum()

    def _calculate_upside_downside_capture(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Tuple[float, float]:
        """
        Calculate Upside and Downside Capture Ratios

        Upside Capture: Portfolio return in up markets / Benchmark return in up markets
        Downside Capture: Portfolio return in down markets / Benchmark return in down markets

        Ideal: >100% upside capture, <100% downside capture (asymmetric performance)

        Returns:
            Tuple of (upside_capture, downside_capture) as percentages
        """
        aligned = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()

        # Up markets
        up_markets = aligned[aligned['benchmark'] > 0]
        if len(up_markets) > 0 and up_markets['benchmark'].sum() != 0:
            upside_capture = (up_markets['portfolio'].sum() / up_markets['benchmark'].sum()) * 100
        else:
            upside_capture = 0.0

        # Down markets
        down_markets = aligned[aligned['benchmark'] < 0]
        if len(down_markets) > 0 and down_markets['benchmark'].sum() != 0:
            downside_capture = (down_markets['portfolio'].sum() / down_markets['benchmark'].sum()) * 100
        else:
            downside_capture = 0.0

        return upside_capture, downside_capture

    def _calculate_pain_index(self, returns: pd.Series) -> float:
        """
        Calculate Pain Index (Ulcer Index) - measures depth and duration of drawdowns

        Lower is better - shows how much "pain" investors endure from drawdowns
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        # Square the drawdowns and take average, then sqrt
        pain_index = np.sqrt((drawdown ** 2).mean())
        return abs(pain_index)

    def _calculate_gh1_ratio(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate GH1 Ratio (Hühn & Scholz measure)

        Evaluates if strategy yields higher return for same risk vs benchmark
        Combines return enhancement and risk reduction into one metric

        GH1 > 0: Strategy adds value
        GH1 < 0: Strategy destroys value
        """
        # Calculate excess returns
        excess_returns = portfolio_returns - benchmark_returns

        # Calculate tracking error
        tracking_error = excess_returns.std()

        if tracking_error == 0:
            return 0.0

        # GH1 = (Excess Return / Tracking Error) - (Benchmark Volatility / Portfolio Volatility)
        portfolio_vol = portfolio_returns.std()
        benchmark_vol = benchmark_returns.std()

        if portfolio_vol == 0:
            return 0.0

        information_ratio = excess_returns.mean() / tracking_error
        vol_ratio = benchmark_vol / portfolio_vol

        gh1 = information_ratio - (1 - vol_ratio)
        return gh1

    # ==================== Phase 4: Technical & Cross-Asset Metrics ====================

    def _compute_phase4_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        position_returns: pd.DataFrame
    ) -> Tuple[Dict[str, Optional[float]], List[str]]:
        """
        Compute Phase 4 technical & cross-asset metrics with graceful degradation.

        Phase 4 provides short-horizon (1-5 day) edge from:
        - Options flow (PCR, IV skew, volume)
        - Residual momentum (idiosyncratic returns)
        - Seasonality (calendar patterns)
        - Breadth & liquidity (market internals)

        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            position_returns: Individual position returns

        Returns:
            Tuple of (metrics_dict, data_sources_list)

        Sources:
            - ExtractAlpha Cross-Asset Model (CAM)
            - Cboe options market statistics
            - Academic research on options-equity lead-lag
        """
        metrics: Dict[str, Optional[float]] = {
            'options_flow_composite': None,
            'residual_momentum': None,
            'seasonality_score': None,
            'breadth_liquidity': None
        }
        sources: List[str] = []

        try:
            # Residual momentum (always computable from returns)
            if len(portfolio_returns) >= 20 and len(benchmark_returns) >= 20:
                asset_returns_np = np.array(portfolio_returns.values, dtype=float)
                market_returns_np = np.array(benchmark_returns.values, dtype=float)

                # Use compute_phase4_metrics for residual momentum
                phase4_result = compute_phase4_metrics(
                    asset_returns=asset_returns_np,
                    market_returns=market_returns_np
                )

                residual_mom = phase4_result.get('residual_momentum')
                if residual_mom is not None:
                    metrics['residual_momentum'] = float(residual_mom)
                    sources.append('computed')
                    logger.debug(f"✅ Residual momentum: {metrics['residual_momentum']:.2f}")

            # Seasonality (requires DatetimeIndex)
            if isinstance(portfolio_returns.index, pd.DatetimeIndex) and len(portfolio_returns) >= 60:
                phase4_result = compute_phase4_metrics(
                    returns_series=portfolio_returns
                )

                seasonality = phase4_result.get('seasonality_score')
                if seasonality is not None:
                    metrics['seasonality_score'] = float(seasonality)
                    sources.append('computed')
                    logger.debug(f"✅ Seasonality score: {metrics['seasonality_score']:.2f}")

            # Options flow & breadth/liquidity require external data (set to None)
            # These would be populated by data providers (Cboe, LSEG, etc.)
            logger.debug("⚠️ Options flow & breadth/liquidity require external data (not computed)")

        except Exception as e:
            logger.error(f"❌ Error computing Phase 4 metrics: {e}")

        # Add data sources
        if not sources:
            sources = ['local']  # Fallback

        return metrics, sources

