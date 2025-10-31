"""
API Routes for Advanced Portfolio Risk Dashboard

Bloomberg PORT equivalent - institutional-grade risk management
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from datetime import datetime, timedelta
import logging

from ..analytics.portfolio_risk_service import (
    PortfolioRiskService,
    RiskDashboard,
    PortfolioGreeks,
    VaRMetrics,
    StressTestResults,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/risk-dashboard", tags=["Risk Dashboard"])

# Initialize service
risk_service = PortfolioRiskService()


@router.get("/{user_id}")
async def get_risk_dashboard(
    user_id: str,
    lookback_days: int = Query(default=252, ge=30, le=1000, description="Days of historical data for VaR calculation"),
) -> dict:
    """
    Get complete risk dashboard for a user's portfolio.

    Bloomberg PORT equivalent - provides:
    - Portfolio Greeks (Delta, Gamma, Theta, Vega, Rho)
    - Value at Risk (VaR) metrics at 95% and 99% confidence
    - Stress test results for 6 scenarios
    - Concentration risk analysis
    - Performance attribution (alpha, beta, Greeks P&L)
    - Risk metrics (Sharpe, Sortino, Max Drawdown)

    Args:
        user_id: User identifier
        lookback_days: Number of days of historical data for VaR (default 252 = 1 year)

    Returns:
        Complete risk dashboard with all metrics
    """
    try:
        logger.info(f"Calculating risk dashboard for user: {user_id}")

        # Calculate complete dashboard
        dashboard = await risk_service.calculate_risk_dashboard(
            user_id=user_id,
            lookback_days=lookback_days
        )

        # Convert to dict for JSON serialization
        result = {
            "user_id": user_id,
            "timestamp": dashboard.timestamp.isoformat(),
            "portfolio_value": dashboard.portfolio_value,
            "cash": dashboard.cash,
            "positions_value": dashboard.positions_value,
            "margin_used": dashboard.margin_used,
            "margin_available": dashboard.margin_available,
            "buying_power": dashboard.buying_power,

            # Greeks
            "greeks": {
                "total_delta": dashboard.greeks.total_delta,
                "total_gamma": dashboard.greeks.total_gamma,
                "total_theta": dashboard.greeks.total_theta,
                "total_vega": dashboard.greeks.total_vega,
                "total_rho": dashboard.greeks.total_rho,
                "delta_by_symbol": dashboard.greeks.delta_by_symbol,
                "net_delta_exposure": dashboard.greeks.net_delta_exposure,
            },

            # VaR
            "var_metrics": {
                "var_1day_95": dashboard.var_metrics.var_1day_95,
                "var_1day_99": dashboard.var_metrics.var_1day_99,
                "var_10day_95": dashboard.var_metrics.var_10day_95,
                "var_10day_99": dashboard.var_metrics.var_10day_99,
                "cvar_1day_95": dashboard.var_metrics.cvar_1day_95,
                "cvar_1day_99": dashboard.var_metrics.cvar_1day_99,
                "var_as_pct_of_portfolio": dashboard.var_metrics.var_as_pct_of_portfolio,
                "method": dashboard.var_metrics.method,
                "confidence_level": dashboard.var_metrics.confidence_level,
            },

            # Stress Tests
            "stress_tests": [
                {
                    "scenario": st.scenario,
                    "portfolio_change": st.portfolio_change,
                    "portfolio_change_pct": st.portfolio_change_pct,
                    "new_portfolio_value": st.new_portfolio_value,
                    "breaches_margin": st.breaches_margin,
                    "delta_impact": st.delta_impact,
                    "gamma_impact": st.gamma_impact,
                    "vega_impact": st.vega_impact,
                }
                for st in dashboard.stress_tests
            ],

            # Concentration
            "concentration": {
                "largest_position_pct": dashboard.concentration.largest_position_pct,
                "top_5_concentration_pct": dashboard.concentration.top_5_concentration_pct,
                "position_count": dashboard.concentration.position_count,
                "herfindahl_index": dashboard.concentration.herfindahl_index,
                "effective_positions": dashboard.concentration.effective_positions,
                "sector_exposure": dashboard.concentration.sector_exposure,
            },

            # Attribution
            "attribution": {
                "total_pnl": dashboard.attribution.total_pnl,
                "alpha_pnl": dashboard.attribution.alpha_pnl,
                "beta_pnl": dashboard.attribution.beta_pnl,
                "theta_pnl": dashboard.attribution.theta_pnl,
                "vega_pnl": dashboard.attribution.vega_pnl,
                "gamma_pnl": dashboard.attribution.gamma_pnl,
                "realized_pnl": dashboard.attribution.realized_pnl,
                "unrealized_pnl": dashboard.attribution.unrealized_pnl,
            },

            # Performance Metrics
            "sharpe_ratio": dashboard.sharpe_ratio,
            "sortino_ratio": dashboard.sortino_ratio,
            "max_drawdown": dashboard.max_drawdown,
            "max_drawdown_pct": dashboard.max_drawdown_pct,

            # Metadata
            "calculation_time_ms": dashboard.calculation_time_ms,
            "data_source": dashboard.data_source,
        }

        logger.info(f"Risk dashboard calculated in {dashboard.calculation_time_ms}ms")
        return result

    except Exception as e:
        logger.error(f"Failed to calculate risk dashboard for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate risk dashboard: {str(e)}")


@router.get("/{user_id}/greeks")
async def get_portfolio_greeks(user_id: str) -> dict:
    """
    Get aggregated portfolio Greeks only.

    Faster endpoint when you only need Greeks without full risk calculations.

    Args:
        user_id: User identifier

    Returns:
        Portfolio Greeks (Delta, Gamma, Theta, Vega, Rho)
    """
    try:
        logger.info(f"Calculating portfolio Greeks for user: {user_id}")

        # Get dashboard but we'll only return Greeks
        dashboard = await risk_service.calculate_risk_dashboard(user_id=user_id, lookback_days=30)

        return {
            "user_id": user_id,
            "timestamp": dashboard.timestamp.isoformat(),
            "portfolio_value": dashboard.portfolio_value,
            "greeks": {
                "total_delta": dashboard.greeks.total_delta,
                "total_gamma": dashboard.greeks.total_gamma,
                "total_theta": dashboard.greeks.total_theta,
                "total_vega": dashboard.greeks.total_vega,
                "total_rho": dashboard.greeks.total_rho,
                "delta_by_symbol": dashboard.greeks.delta_by_symbol,
                "net_delta_exposure": dashboard.greeks.net_delta_exposure,
            },
        }

    except Exception as e:
        logger.error(f"Failed to calculate Greeks for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate Greeks: {str(e)}")


@router.get("/{user_id}/var")
async def get_var_metrics(
    user_id: str,
    lookback_days: int = Query(default=252, ge=30, le=1000),
) -> dict:
    """
    Get Value at Risk (VaR) metrics only.

    Args:
        user_id: User identifier
        lookback_days: Days of historical data (default 252 = 1 year)

    Returns:
        VaR metrics at 95% and 99% confidence for 1-day and 10-day horizons
    """
    try:
        logger.info(f"Calculating VaR for user: {user_id}")

        dashboard = await risk_service.calculate_risk_dashboard(
            user_id=user_id,
            lookback_days=lookback_days
        )

        return {
            "user_id": user_id,
            "timestamp": dashboard.timestamp.isoformat(),
            "portfolio_value": dashboard.portfolio_value,
            "var_metrics": {
                "var_1day_95": dashboard.var_metrics.var_1day_95,
                "var_1day_99": dashboard.var_metrics.var_1day_99,
                "var_10day_95": dashboard.var_metrics.var_10day_95,
                "var_10day_99": dashboard.var_metrics.var_10day_99,
                "cvar_1day_95": dashboard.var_metrics.cvar_1day_95,
                "cvar_1day_99": dashboard.var_metrics.cvar_1day_99,
                "var_as_pct_of_portfolio": dashboard.var_metrics.var_as_pct_of_portfolio,
                "method": dashboard.var_metrics.method,
                "confidence_level": dashboard.var_metrics.confidence_level,
            },
            "interpretation": {
                "var_1day_95_meaning": f"95% confident we won't lose more than ${dashboard.var_metrics.var_1day_95:,.2f} in 1 day",
                "var_1day_99_meaning": f"99% confident we won't lose more than ${dashboard.var_metrics.var_1day_99:,.2f} in 1 day",
                "var_10day_95_meaning": f"95% confident we won't lose more than ${dashboard.var_metrics.var_10day_95:,.2f} in 10 days",
                "cvar_explanation": "Expected Shortfall (CVaR) is the average loss when losses exceed VaR",
            },
        }

    except Exception as e:
        logger.error(f"Failed to calculate VaR for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate VaR: {str(e)}")


@router.get("/{user_id}/stress-tests")
async def get_stress_tests(user_id: str) -> dict:
    """
    Get stress test results for various market scenarios.

    Tests 6 scenarios:
    - SPY down 5%
    - SPY down 10% (market crash)
    - VIX up 10 points
    - VIX up 20 points (volatility spike)
    - Interest rates up 0.5%
    - Interest rates down 0.5%

    Args:
        user_id: User identifier

    Returns:
        Stress test results showing portfolio impact for each scenario
    """
    try:
        logger.info(f"Running stress tests for user: {user_id}")

        dashboard = await risk_service.calculate_risk_dashboard(user_id=user_id, lookback_days=30)

        return {
            "user_id": user_id,
            "timestamp": dashboard.timestamp.isoformat(),
            "portfolio_value": dashboard.portfolio_value,
            "margin_available": dashboard.margin_available,
            "stress_tests": [
                {
                    "scenario": st.scenario,
                    "portfolio_change": st.portfolio_change,
                    "portfolio_change_pct": st.portfolio_change_pct,
                    "new_portfolio_value": st.new_portfolio_value,
                    "breaches_margin": st.breaches_margin,
                    "delta_impact": st.delta_impact,
                    "gamma_impact": st.gamma_impact,
                    "vega_impact": st.vega_impact,
                    "severity": "CRITICAL" if st.breaches_margin else (
                        "HIGH" if abs(st.portfolio_change_pct) > 10 else (
                            "MEDIUM" if abs(st.portfolio_change_pct) > 5 else "LOW"
                        )
                    ),
                }
                for st in dashboard.stress_tests
            ],
            "worst_case": max(dashboard.stress_tests, key=lambda x: abs(x.portfolio_change)),
        }

    except Exception as e:
        logger.error(f"Failed to run stress tests for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run stress tests: {str(e)}")


@router.get("/{user_id}/concentration")
async def get_concentration_risk(user_id: str) -> dict:
    """
    Get portfolio concentration risk metrics.

    Analyzes:
    - Largest position size
    - Top 5 concentration
    - Herfindahl Index (portfolio diversification)
    - Effective number of positions
    - Sector exposure

    Args:
        user_id: User identifier

    Returns:
        Concentration risk metrics and warnings
    """
    try:
        logger.info(f"Calculating concentration risk for user: {user_id}")

        dashboard = await risk_service.calculate_risk_dashboard(user_id=user_id, lookback_days=30)

        # Generate warnings
        warnings = []
        if dashboard.concentration.largest_position_pct > 20:
            warnings.append(f"⚠️ Largest position is {dashboard.concentration.largest_position_pct:.1f}% of portfolio (>20% is risky)")
        if dashboard.concentration.top_5_concentration_pct > 60:
            warnings.append(f"⚠️ Top 5 positions are {dashboard.concentration.top_5_concentration_pct:.1f}% of portfolio (>60% is concentrated)")
        if dashboard.concentration.effective_positions < 5:
            warnings.append(f"⚠️ Effective positions: {dashboard.concentration.effective_positions:.1f} (less than 5 indicates poor diversification)")

        return {
            "user_id": user_id,
            "timestamp": dashboard.timestamp.isoformat(),
            "portfolio_value": dashboard.portfolio_value,
            "concentration": {
                "largest_position_pct": dashboard.concentration.largest_position_pct,
                "top_5_concentration_pct": dashboard.concentration.top_5_concentration_pct,
                "position_count": dashboard.concentration.position_count,
                "herfindahl_index": dashboard.concentration.herfindahl_index,
                "effective_positions": dashboard.concentration.effective_positions,
                "sector_exposure": dashboard.concentration.sector_exposure,
            },
            "warnings": warnings,
            "risk_level": "HIGH" if len(warnings) >= 2 else ("MEDIUM" if len(warnings) == 1 else "LOW"),
        }

    except Exception as e:
        logger.error(f"Failed to calculate concentration risk for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate concentration risk: {str(e)}")


@router.get("/{user_id}/attribution")
async def get_performance_attribution(user_id: str) -> dict:
    """
    Get performance attribution breakdown.

    Breaks down P&L into:
    - Alpha P&L (stock selection, timing)
    - Beta P&L (market exposure)
    - Theta P&L (time decay)
    - Vega P&L (volatility changes)
    - Gamma P&L (convexity)
    - Realized vs Unrealized

    Args:
        user_id: User identifier

    Returns:
        P&L attribution by source
    """
    try:
        logger.info(f"Calculating performance attribution for user: {user_id}")

        dashboard = await risk_service.calculate_risk_dashboard(user_id=user_id, lookback_days=252)

        total = dashboard.attribution.total_pnl or 0.01  # Avoid division by zero

        return {
            "user_id": user_id,
            "timestamp": dashboard.timestamp.isoformat(),
            "portfolio_value": dashboard.portfolio_value,
            "attribution": {
                "total_pnl": dashboard.attribution.total_pnl,
                "alpha_pnl": dashboard.attribution.alpha_pnl,
                "beta_pnl": dashboard.attribution.beta_pnl,
                "theta_pnl": dashboard.attribution.theta_pnl,
                "vega_pnl": dashboard.attribution.vega_pnl,
                "gamma_pnl": dashboard.attribution.gamma_pnl,
                "realized_pnl": dashboard.attribution.realized_pnl,
                "unrealized_pnl": dashboard.attribution.unrealized_pnl,
            },
            "attribution_pct": {
                "alpha_pct": (dashboard.attribution.alpha_pnl / total * 100) if total != 0 else 0,
                "beta_pct": (dashboard.attribution.beta_pnl / total * 100) if total != 0 else 0,
                "theta_pct": (dashboard.attribution.theta_pnl / total * 100) if total != 0 else 0,
                "vega_pct": (dashboard.attribution.vega_pnl / total * 100) if total != 0 else 0,
                "gamma_pct": (dashboard.attribution.gamma_pnl / total * 100) if total != 0 else 0,
            },
            "realized_vs_unrealized": {
                "realized_pct": (dashboard.attribution.realized_pnl / total * 100) if total != 0 else 0,
                "unrealized_pct": (dashboard.attribution.unrealized_pnl / total * 100) if total != 0 else 0,
            },
        }

    except Exception as e:
        logger.error(f"Failed to calculate attribution for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate attribution: {str(e)}")


@router.get("/{user_id}/performance-metrics")
async def get_performance_metrics(
    user_id: str,
    lookback_days: int = Query(default=252, ge=30, le=1000),
) -> dict:
    """
    Get risk-adjusted performance metrics.

    Calculates:
    - Sharpe Ratio (risk-adjusted return)
    - Sortino Ratio (downside risk-adjusted return)
    - Maximum Drawdown (worst peak-to-trough decline)

    Args:
        user_id: User identifier
        lookback_days: Days of historical data (default 252 = 1 year)

    Returns:
        Performance metrics with interpretations
    """
    try:
        logger.info(f"Calculating performance metrics for user: {user_id}")

        dashboard = await risk_service.calculate_risk_dashboard(
            user_id=user_id,
            lookback_days=lookback_days
        )

        # Interpret Sharpe ratio
        sharpe_interpretation = "Excellent" if dashboard.sharpe_ratio > 2 else (
            "Very Good" if dashboard.sharpe_ratio > 1 else (
                "Good" if dashboard.sharpe_ratio > 0.5 else (
                    "Fair" if dashboard.sharpe_ratio > 0 else "Poor"
                )
            )
        )

        return {
            "user_id": user_id,
            "timestamp": dashboard.timestamp.isoformat(),
            "portfolio_value": dashboard.portfolio_value,
            "metrics": {
                "sharpe_ratio": dashboard.sharpe_ratio,
                "sortino_ratio": dashboard.sortino_ratio,
                "max_drawdown": dashboard.max_drawdown,
                "max_drawdown_pct": dashboard.max_drawdown_pct,
            },
            "interpretation": {
                "sharpe_rating": sharpe_interpretation,
                "sharpe_explanation": "Measures excess return per unit of risk. >1 is good, >2 is excellent.",
                "sortino_explanation": "Like Sharpe but only penalizes downside volatility. Higher is better.",
                "max_drawdown_explanation": f"Worst decline from peak: ${dashboard.max_drawdown:,.2f} ({dashboard.max_drawdown_pct:.1f}%)",
            },
            "lookback_days": lookback_days,
        }

    except Exception as e:
        logger.error(f"Failed to calculate performance metrics for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate performance metrics: {str(e)}")


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint for risk dashboard service."""
    return {
        "status": "ok",
        "service": "risk-dashboard",
        "timestamp": datetime.now().isoformat(),
        "capabilities": [
            "Portfolio Greeks",
            "Value at Risk (VaR)",
            "Stress Testing",
            "Concentration Analysis",
            "Performance Attribution",
            "Risk Metrics (Sharpe, Sortino, Max Drawdown)",
        ],
    }
