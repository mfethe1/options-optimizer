"""
Paper Trading API Routes

AI-powered autonomous trading with approval workflows.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from ..trading.paper_trading_engine import PaperTradingEngine, PaperTrade

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/paper-trading",
    tags=["Paper Trading"]
)

# Initialize engine (one per user in production, using dependency injection)
# For now, use a simple in-memory store
_engines: Dict[str, PaperTradingEngine] = {}


def get_engine(user_id: str) -> PaperTradingEngine:
    """Get or create paper trading engine for user"""
    if user_id not in _engines:
        _engines[user_id] = PaperTradingEngine()
    return _engines[user_id]


# Request/Response models
class TradeRecommendation(BaseModel):
    """Trade recommendation from agents"""
    symbol: str = Field(..., description="Stock symbol")
    action: str = Field(..., description="Action: buy, sell, open, close")
    quantity: int = Field(..., description="Number of shares/contracts")
    price: Optional[float] = Field(None, description="Limit price (if None, uses market price)")
    trade_type: str = Field("stock", description="Trade type: stock, option")
    option_details: Optional[Dict[str, Any]] = Field(None, description="Option details if trade_type=option")
    confidence: float = Field(..., description="Agent confidence (0-1)")
    reasoning: Optional[str] = Field(None, description="Agent reasoning")


class ExecuteTradeRequest(BaseModel):
    """Request to execute a trade recommendation"""
    recommendation: TradeRecommendation
    user_id: str = Field(..., description="User ID")
    auto_approve: bool = Field(False, description="Auto-approve after timeout")
    timeout_seconds: int = Field(300, description="Approval timeout in seconds (default 5 min)")


class ExecuteTradeResponse(BaseModel):
    """Response from trade execution"""
    status: str = Field(..., description="Status: executed, rejected, pending")
    trade: Optional[Dict[str, Any]] = Field(None, description="Trade details if executed")
    consensus: Optional[Dict[str, Any]] = Field(None, description="Multi-agent consensus result")
    risk_check: Optional[Dict[str, Any]] = Field(None, description="Risk check result")
    reason: Optional[str] = Field(None, description="Rejection reason if rejected")
    portfolio: Optional[Dict[str, Any]] = Field(None, description="Updated portfolio summary")
    timestamp: str


class PortfolioResponse(BaseModel):
    """Paper trading portfolio summary"""
    cash: float
    positions_count: int
    positions: List[Dict[str, Any]]
    performance: Dict[str, Any]
    timestamp: str


class TradeHistoryResponse(BaseModel):
    """Trade history"""
    trades: List[Dict[str, Any]]
    count: int
    timestamp: str


# Routes

@router.post("/execute", response_model=ExecuteTradeResponse)
async def execute_trade(request: ExecuteTradeRequest):
    """
    Execute AI-recommended trade with safety guardrails

    **COMPETITIVE ADVANTAGE**: First options platform with AI approval workflows

    Workflow:
    1. **Multi-agent consensus**: Agents vote on trade (70%+ agreement required)
    2. **Risk manager approval**: Checks position limits, cash, portfolio Greeks
    3. **User notification**: SMS/email with 1-click approval (or auto after timeout)
    4. **Execute**: Paper trade executed, portfolio updated, performance tracked

    Safety features:
    - Position size limits (max 10% per trade)
    - Portfolio risk limits (max delta, theta, VaR)
    - Multi-agent consensus (weighted voting)
    - User override capability
    - Full audit trail

    Args:
        request: Trade recommendation and execution parameters

    Returns:
        Execution result with consensus, risk check, and portfolio update
    """
    try:
        logger.info(f"Executing trade for user {request.user_id}: {request.recommendation.action} {request.recommendation.quantity} {request.recommendation.symbol}")

        # Get user's paper trading engine
        engine = get_engine(request.user_id)

        # Convert recommendation to dict
        recommendation = request.recommendation.dict()

        # Execute through approval workflow
        result = await engine.execute_agent_recommendation(
            recommendation=recommendation,
            user_id=request.user_id,
            auto_approve=request.auto_approve,
            timeout_seconds=request.timeout_seconds
        )

        return ExecuteTradeResponse(
            status=result['status'],
            trade=result.get('trade'),
            consensus=result.get('consensus'),
            risk_check=result.get('risk_check'),
            reason=result.get('reason'),
            portfolio=result.get('portfolio'),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error executing trade: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to execute trade: {str(e)}")


@router.get("/portfolio/{user_id}", response_model=PortfolioResponse)
async def get_portfolio(user_id: str):
    """
    Get paper trading portfolio for user

    Returns:
        Current portfolio with cash, positions, and performance metrics
    """
    try:
        logger.info(f"Fetching portfolio for user {user_id}")

        # Get user's engine
        engine = get_engine(user_id)

        # Get portfolio summary
        portfolio = engine._get_portfolio_summary()

        return PortfolioResponse(
            cash=portfolio['cash'],
            positions_count=portfolio['positions_count'],
            positions=portfolio['positions'],
            performance=portfolio['performance'],
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error fetching portfolio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch portfolio: {str(e)}")


@router.get("/history/{user_id}", response_model=TradeHistoryResponse)
async def get_trade_history(user_id: str, limit: int = 50):
    """
    Get paper trading history for user

    Args:
        user_id: User identifier
        limit: Max trades to return (default 50)

    Returns:
        Recent trade history with P&L
    """
    try:
        logger.info(f"Fetching trade history for user {user_id}")

        # Get user's engine
        engine = get_engine(user_id)

        # Get trade history
        trades = engine.get_trade_history(limit=limit)

        return TradeHistoryResponse(
            trades=trades,
            count=len(trades),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error fetching trade history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch trade history: {str(e)}")


@router.post("/portfolio/{user_id}/reset")
async def reset_portfolio(user_id: str):
    """
    Reset paper trading portfolio to starting state

    Clears all positions and resets cash to $100,000.

    Args:
        user_id: User identifier

    Returns:
        Success message
    """
    try:
        logger.info(f"Resetting portfolio for user {user_id}")

        # Get user's engine
        engine = get_engine(user_id)

        # Reset portfolio
        engine.reset_portfolio()

        return {
            "message": "Portfolio reset successfully",
            "starting_cash": 100000.0,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error resetting portfolio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset portfolio: {str(e)}")


@router.get("/risk-limits/{user_id}")
async def get_risk_limits(user_id: str):
    """
    Get risk limits for user's paper trading

    Returns:
        Current risk limit configuration
    """
    try:
        # Get user's engine
        engine = get_engine(user_id)

        return {
            "risk_limits": engine.risk_limits,
            "description": {
                "max_position_size_pct": "Maximum % of portfolio per position",
                "max_portfolio_delta": "Maximum net delta exposure",
                "max_portfolio_theta": "Maximum daily theta decay ($)",
                "max_drawdown_pct": "Maximum % drawdown from peak",
                "max_var_95": "Maximum Value at Risk (95% confidence)"
            }
        }

    except Exception as e:
        logger.error(f"Error fetching risk limits: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch risk limits: {str(e)}")


@router.put("/risk-limits/{user_id}")
async def update_risk_limits(user_id: str, risk_limits: Dict[str, float]):
    """
    Update risk limits for user's paper trading

    **Note**: Only available in paper trading. Real money trading limits require compliance approval.

    Args:
        user_id: User identifier
        risk_limits: Dictionary of risk limits to update

    Returns:
        Updated risk limits
    """
    try:
        logger.info(f"Updating risk limits for user {user_id}")

        # Get user's engine
        engine = get_engine(user_id)

        # Validate limits
        valid_keys = set(engine.risk_limits.keys())
        provided_keys = set(risk_limits.keys())

        if not provided_keys.issubset(valid_keys):
            invalid_keys = provided_keys - valid_keys
            raise HTTPException(
                status_code=400,
                detail=f"Invalid risk limit keys: {invalid_keys}. Valid keys: {valid_keys}"
            )

        # Update limits
        engine.risk_limits.update(risk_limits)

        return {
            "message": "Risk limits updated successfully",
            "risk_limits": engine.risk_limits,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating risk limits: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update risk limits: {str(e)}")


@router.get("/approvals/{user_id}")
async def get_pending_approvals(user_id: str):
    """
    Get pending trade approvals for user

    Returns:
        List of trades awaiting user approval
    """
    try:
        logger.info(f"Fetching pending approvals for user {user_id}")

        # Get user's engine
        engine = get_engine(user_id)

        # Filter approvals for this user
        pending = [
            {
                "trade_id": trade_id,
                "recommendation": approval['recommendation'],
                "consensus": approval['consensus'],
                "requested_at": approval['requested_at'].isoformat(),
                "expires_at": approval['expires_at'].isoformat(),
                "status": approval['status']
            }
            for trade_id, approval in engine.pending_approvals.items()
            if approval['user_id'] == user_id and approval['status'] == 'pending'
        ]

        return {
            "user_id": user_id,
            "pending_approvals": pending,
            "count": len(pending),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching pending approvals: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch pending approvals: {str(e)}")


@router.post("/approvals/{user_id}/{trade_id}/approve")
async def approve_trade(user_id: str, trade_id: str):
    """
    Manually approve a pending trade

    Args:
        user_id: User identifier
        trade_id: Trade identifier

    Returns:
        Approval confirmation
    """
    try:
        logger.info(f"Approving trade {trade_id} for user {user_id}")

        # Get user's engine
        engine = get_engine(user_id)

        # Check if trade exists and belongs to user
        if trade_id not in engine.pending_approvals:
            raise HTTPException(status_code=404, detail="Trade not found")

        approval = engine.pending_approvals[trade_id]
        if approval['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Trade does not belong to user")

        # Approve trade
        approval['status'] = 'approved'

        return {
            "message": "Trade approved successfully",
            "trade_id": trade_id,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving trade: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to approve trade: {str(e)}")


@router.post("/approvals/{user_id}/{trade_id}/reject")
async def reject_trade(user_id: str, trade_id: str):
    """
    Manually reject a pending trade

    Args:
        user_id: User identifier
        trade_id: Trade identifier

    Returns:
        Rejection confirmation
    """
    try:
        logger.info(f"Rejecting trade {trade_id} for user {user_id}")

        # Get user's engine
        engine = get_engine(user_id)

        # Check if trade exists and belongs to user
        if trade_id not in engine.pending_approvals:
            raise HTTPException(status_code=404, detail="Trade not found")

        approval = engine.pending_approvals[trade_id]
        if approval['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Trade does not belong to user")

        # Reject trade
        approval['status'] = 'rejected'

        return {
            "message": "Trade rejected successfully",
            "trade_id": trade_id,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rejecting trade: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reject trade: {str(e)}")
