"""
Stress Testing & Scenario Analysis API Routes

REST API endpoints for portfolio stress testing and Monte Carlo simulation.
Enables risk analysis through historical scenarios and probabilistic simulations.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/stress-testing", tags=["stress-testing"])

# Global stress testing engine (initialized on startup)
stress_engine = None


# ============================================================================
# Request/Response Models
# ============================================================================

class MarketShockRequest(BaseModel):
    """Custom market shock parameters"""
    equity_return: float = -0.10
    volatility_change: float = 0.20
    interest_rate_change: float = 0.0
    time_horizon_days: int = 30
    correlation_shock: float = 0.0


class PortfolioPosition(BaseModel):
    """Portfolio position for stress testing"""
    symbol: str
    type: str  # "stock", "call", "put"
    quantity: float
    current_price: float
    strike: Optional[float] = None
    expiration: Optional[str] = None
    time_value: Optional[float] = None


class PortfolioRequest(BaseModel):
    """Portfolio for stress testing"""
    positions: List[PortfolioPosition]


class ScenarioRunRequest(BaseModel):
    """Request to run specific scenario"""
    scenario_type: str
    portfolio: PortfolioRequest
    custom_shock: Optional[MarketShockRequest] = None


class MonteCarloRequest(BaseModel):
    """Monte Carlo simulation request"""
    portfolio: PortfolioRequest
    num_simulations: int = 10000
    time_horizon_days: int = 30
    confidence_level: float = 0.95


class PositionStressResponse(BaseModel):
    """Position-level stress test result"""
    symbol: str
    position_type: str
    current_value: float
    stressed_value: float
    pnl: float
    pnl_pct: float
    contribution_to_total_pnl: float


class MarketShockResponse(BaseModel):
    """Market shock details"""
    equity_return: float
    volatility_change: float
    interest_rate_change: float
    time_horizon_days: int
    correlation_shock: float


class PortfolioStressResponse(BaseModel):
    """Portfolio stress test result"""
    scenario_name: str
    scenario_type: str
    timestamp: str
    current_portfolio_value: float
    stressed_portfolio_value: float
    total_pnl: float
    total_pnl_pct: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    position_results: List[PositionStressResponse]
    market_shock: MarketShockResponse


class MonteCarloResponse(BaseModel):
    """Monte Carlo simulation result"""
    timestamp: str
    num_simulations: int
    time_horizon_days: int
    mean_pnl: float
    median_pnl: float
    std_pnl: float
    pnl_5th: float
    pnl_25th: float
    pnl_75th: float
    pnl_95th: float
    var_95: float
    cvar_95: float
    max_drawdown_mean: float
    max_drawdown_95th: float
    prob_loss_10pct: float
    prob_loss_20pct: float
    prob_gain_10pct: float
    prob_gain_20pct: float
    pnl_distribution: Optional[List[float]] = None  # Optional to reduce payload


class ScenarioInfoResponse(BaseModel):
    """Historical scenario information"""
    scenario_type: str
    name: str
    description: str
    date_range: str
    probability: float
    market_shock: MarketShockResponse


# ============================================================================
# Startup/Shutdown
# ============================================================================

async def initialize_stress_engine():
    """Initialize stress testing engine"""
    global stress_engine
    try:
        from ..risk.stress_testing_engine import StressTestingEngine
        stress_engine = StressTestingEngine()
        logger.info("Stress testing engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize stress testing engine: {e}")
        raise


# ============================================================================
# Stress Testing Endpoints
# ============================================================================

@router.post("/scenario/run", response_model=PortfolioStressResponse)
async def run_scenario(request: ScenarioRunRequest):
    """
    Run stress test scenario on portfolio.

    Applies historical or custom market shock to portfolio positions.

    Available scenarios:
    - 2008_financial_crisis: -30% equity, +40 vol points
    - covid_crash_2020: -34% equity, +55 vol points
    - flash_crash_2010: -9% equity, +15 vol points
    - volmageddon_2018: -4% equity, +25 vol points
    - custom: User-defined shock

    Example:
    ```json
    {
      "scenario_type": "covid_crash_2020",
      "portfolio": {
        "positions": [
          {"symbol": "AAPL", "type": "stock", "quantity": 100, "current_price": 180.50},
          {"symbol": "SPY", "type": "call", "quantity": 10, "current_price": 5.50, "strike": 450}
        ]
      }
    }
    ```

    Returns:
    - Portfolio P&L under stress
    - Position-level breakdown
    - Risk metrics (VaR, CVaR, max drawdown)
    """
    if not stress_engine:
        raise HTTPException(status_code=503, detail="Stress testing engine not initialized")

    try:
        # Convert portfolio to dict
        portfolio_dict = {
            'positions': [pos.dict() for pos in request.portfolio.positions]
        }

        # Convert custom shock if provided
        custom_shock = None
        if request.custom_shock:
            from ..risk.stress_testing_engine import MarketShock
            custom_shock = MarketShock(
                equity_return=request.custom_shock.equity_return,
                volatility_change=request.custom_shock.volatility_change,
                interest_rate_change=request.custom_shock.interest_rate_change,
                time_horizon_days=request.custom_shock.time_horizon_days,
                correlation_shock=request.custom_shock.correlation_shock
            )

        # Run scenario
        from ..risk.stress_testing_engine import ScenarioType
        scenario_type = ScenarioType(request.scenario_type)

        result = stress_engine.run_scenario(portfolio_dict, scenario_type, custom_shock)

        # Convert to response
        return PortfolioStressResponse(
            scenario_name=result.scenario_name,
            scenario_type=result.scenario_type.value,
            timestamp=result.timestamp.isoformat(),
            current_portfolio_value=result.current_portfolio_value,
            stressed_portfolio_value=result.stressed_portfolio_value,
            total_pnl=result.total_pnl,
            total_pnl_pct=result.total_pnl_pct,
            max_drawdown=result.max_drawdown,
            var_95=result.var_95,
            cvar_95=result.cvar_95,
            position_results=[
                PositionStressResponse(
                    symbol=pos.symbol,
                    position_type=pos.position_type,
                    current_value=pos.current_value,
                    stressed_value=pos.stressed_value,
                    pnl=pos.pnl,
                    pnl_pct=pos.pnl_pct,
                    contribution_to_total_pnl=pos.contribution_to_total_pnl
                ) for pos in result.position_results
            ],
            market_shock=MarketShockResponse(
                equity_return=result.market_shock.equity_return,
                volatility_change=result.market_shock.volatility_change,
                interest_rate_change=result.market_shock.interest_rate_change,
                time_horizon_days=result.market_shock.time_horizon_days,
                correlation_shock=result.market_shock.correlation_shock
            )
        )

    except Exception as e:
        logger.error(f"Error running scenario: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scenario/run-all")
async def run_all_scenarios(portfolio: PortfolioRequest):
    """
    Run all historical scenarios on portfolio.

    Runs 2008 Financial Crisis, COVID Crash, Flash Crash, and Volmageddon scenarios.

    Returns:
    - Array of stress test results, one per scenario
    """
    if not stress_engine:
        raise HTTPException(status_code=503, detail="Stress testing engine not initialized")

    try:
        portfolio_dict = {
            'positions': [pos.dict() for pos in portfolio.positions]
        }

        results = stress_engine.run_all_scenarios(portfolio_dict)

        return {
            "scenarios": [
                PortfolioStressResponse(
                    scenario_name=result.scenario_name,
                    scenario_type=result.scenario_type.value,
                    timestamp=result.timestamp.isoformat(),
                    current_portfolio_value=result.current_portfolio_value,
                    stressed_portfolio_value=result.stressed_portfolio_value,
                    total_pnl=result.total_pnl,
                    total_pnl_pct=result.total_pnl_pct,
                    max_drawdown=result.max_drawdown,
                    var_95=result.var_95,
                    cvar_95=result.cvar_95,
                    position_results=[
                        PositionStressResponse(
                            symbol=pos.symbol,
                            position_type=pos.position_type,
                            current_value=pos.current_value,
                            stressed_value=pos.stressed_value,
                            pnl=pos.pnl,
                            pnl_pct=pos.pnl_pct,
                            contribution_to_total_pnl=pos.contribution_to_total_pnl
                        ) for pos in result.position_results
                    ],
                    market_shock=MarketShockResponse(
                        equity_return=result.market_shock.equity_return,
                        volatility_change=result.market_shock.volatility_change,
                        interest_rate_change=result.market_shock.interest_rate_change,
                        time_horizon_days=result.market_shock.time_horizon_days,
                        correlation_shock=result.market_shock.correlation_shock
                    )
                ) for result in results
            ],
            "count": len(results)
        }

    except Exception as e:
        logger.error(f"Error running all scenarios: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monte-carlo", response_model=MonteCarloResponse)
async def run_monte_carlo(request: MonteCarloRequest):
    """
    Run Monte Carlo simulation on portfolio.

    Simulates 10,000 market scenarios using realistic price movements and correlations.

    Args:
        portfolio: Portfolio positions
        num_simulations: Number of simulations (default: 10000)
        time_horizon_days: Time horizon in days (default: 30)
        confidence_level: Confidence level for VaR/CVaR (default: 0.95)

    Returns:
        - P&L distribution statistics
        - Risk metrics (VaR, CVaR, max drawdown)
        - Probabilities of different outcomes

    Example:
    ```json
    {
      "portfolio": {
        "positions": [
          {"symbol": "AAPL", "type": "stock", "quantity": 100, "current_price": 180.50}
        ]
      },
      "num_simulations": 10000,
      "time_horizon_days": 30
    }
    ```
    """
    if not stress_engine:
        raise HTTPException(status_code=503, detail="Stress testing engine not initialized")

    try:
        portfolio_dict = {
            'positions': [pos.dict() for pos in request.portfolio.positions]
        }

        result = stress_engine.run_monte_carlo(
            portfolio=portfolio_dict,
            num_simulations=request.num_simulations,
            time_horizon_days=request.time_horizon_days,
            confidence_level=request.confidence_level
        )

        # Return without full distribution to reduce payload (can add as separate endpoint)
        return MonteCarloResponse(
            timestamp=result.timestamp.isoformat(),
            num_simulations=result.num_simulations,
            time_horizon_days=result.time_horizon_days,
            mean_pnl=result.mean_pnl,
            median_pnl=result.median_pnl,
            std_pnl=result.std_pnl,
            pnl_5th=result.pnl_5th,
            pnl_25th=result.pnl_25th,
            pnl_75th=result.pnl_75th,
            pnl_95th=result.pnl_95th,
            var_95=result.var_95,
            cvar_95=result.cvar_95,
            max_drawdown_mean=result.max_drawdown_mean,
            max_drawdown_95th=result.max_drawdown_95th,
            prob_loss_10pct=result.prob_loss_10pct,
            prob_loss_20pct=result.prob_loss_20pct,
            prob_gain_10pct=result.prob_gain_10pct,
            prob_gain_20pct=result.prob_gain_20pct,
            pnl_distribution=None  # Omit to reduce response size
        )

    except Exception as e:
        logger.error(f"Error running Monte Carlo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Scenario Information Endpoints
# ============================================================================

@router.get("/scenarios", response_model=List[ScenarioInfoResponse])
async def get_scenarios():
    """
    Get information about all available historical scenarios.

    Returns scenario definitions including:
    - Historical context and dates
    - Market shock parameters
    - Estimated probability
    """
    if not stress_engine:
        raise HTTPException(status_code=503, detail="Stress testing engine not initialized")

    try:
        scenarios = stress_engine.get_all_scenarios_info()

        return [
            ScenarioInfoResponse(
                scenario_type=scenario.scenario_type.value,
                name=scenario.name,
                description=scenario.description,
                date_range=scenario.date_range,
                probability=scenario.probability,
                market_shock=MarketShockResponse(
                    equity_return=scenario.market_shock.equity_return,
                    volatility_change=scenario.market_shock.volatility_change,
                    interest_rate_change=scenario.market_shock.interest_rate_change,
                    time_horizon_days=scenario.market_shock.time_horizon_days,
                    correlation_shock=scenario.market_shock.correlation_shock
                )
            ) for scenario in scenarios
        ]

    except Exception as e:
        logger.error(f"Error getting scenarios: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scenarios/{scenario_type}", response_model=ScenarioInfoResponse)
async def get_scenario_info(scenario_type: str):
    """
    Get detailed information about a specific scenario.

    Available scenarios:
    - 2008_financial_crisis
    - covid_crash_2020
    - flash_crash_2010
    - volmageddon_2018
    """
    if not stress_engine:
        raise HTTPException(status_code=503, detail="Stress testing engine not initialized")

    try:
        from ..risk.stress_testing_engine import ScenarioType
        scenario = stress_engine.get_scenario_info(ScenarioType(scenario_type))

        if not scenario:
            raise HTTPException(status_code=404, detail=f"Scenario not found: {scenario_type}")

        return ScenarioInfoResponse(
            scenario_type=scenario.scenario_type.value,
            name=scenario.name,
            description=scenario.description,
            date_range=scenario.date_range,
            probability=scenario.probability,
            market_shock=MarketShockResponse(
                equity_return=scenario.market_shock.equity_return,
                volatility_change=scenario.market_shock.volatility_change,
                interest_rate_change=scenario.market_shock.interest_rate_change,
                time_horizon_days=scenario.market_shock.time_horizon_days,
                correlation_shock=scenario.market_shock.correlation_shock
            )
        )

    except Exception as e:
        logger.error(f"Error getting scenario info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check for stress testing service"""
    if not stress_engine:
        return {
            "status": "unavailable",
            "message": "Stress testing engine not initialized"
        }

    return {
        "status": "healthy",
        "message": "Stress testing engine operational",
        "available_scenarios": 4,
        "monte_carlo_enabled": True,
        "timestamp": datetime.now().isoformat()
    }
