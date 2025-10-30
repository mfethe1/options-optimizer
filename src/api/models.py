"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import date, datetime
from decimal import Decimal


class PositionLeg(BaseModel):
    """Individual option leg in a position."""
    option_type: str = Field(..., description="call or put")
    strike: float
    quantity: int
    is_short: bool = False
    entry_price: float
    current_price: Optional[float] = None
    multiplier: int = 100


class PositionCreate(BaseModel):
    """Create position request."""
    user_id: str
    symbol: str
    strategy_type: str
    entry_date: date
    expiration_date: date
    total_premium: float
    legs: List[PositionLeg]
    notes: Optional[str] = None


class PositionUpdate(BaseModel):
    """Update position request."""
    market_value: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    status: Optional[str] = None
    notes: Optional[str] = None


class Position(BaseModel):
    """Position response."""
    id: str
    user_id: str
    symbol: str
    strategy_type: str
    entry_date: date
    expiration_date: date
    status: str
    total_premium: float
    market_value: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    legs: List[PositionLeg]
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class GreeksResponse(BaseModel):
    """Greeks calculation response."""
    position_id: str
    greeks: Dict[str, float]
    timestamp: str


class EVResponse(BaseModel):
    """Expected Value calculation response."""
    position_id: str
    expected_value: float
    expected_return_pct: float
    probability_profit: float
    confidence_interval: tuple
    method_breakdown: Dict[str, float]
    timestamp: str


class AnalysisRequest(BaseModel):
    """Analysis request."""
    user_id: str
    report_type: str = "daily"


class AnalysisResponse(BaseModel):
    """Analysis response."""
    status: str
    report: Dict[str, Any]
    risk_score: float
    timestamp: str


class ReportResponse(BaseModel):
    """Report response."""
    id: str
    user_id: str
    report_type: str
    executive_summary: str
    market_overview: str
    portfolio_analysis: str
    risk_assessment: str
    recommendations: List[Dict[str, Any]]
    action_items: List[Dict[str, Any]]
    risk_score: float
    generated_at: datetime


class UserPreferences(BaseModel):
    """User preferences."""
    user_id: str
    risk_tolerance: str = "moderate"
    notification_preferences: Dict[str, Any] = {}
    report_frequency: str = "daily"
    preferred_strategies: List[str] = []


class MarketData(BaseModel):
    """Market data."""
    symbol: str
    underlying_price: float
    iv: float
    historical_iv: Optional[float] = None
    iv_rank: Optional[float] = None
    volume: Optional[int] = None
    avg_volume: Optional[int] = None
    put_call_ratio: Optional[float] = None
    days_to_earnings: Optional[int] = None
    sector: Optional[str] = None
    timestamp: datetime


class RiskAlert(BaseModel):
    """Risk alert."""
    id: str
    user_id: str
    position_id: Optional[str] = None
    alert_type: str
    severity: str
    message: str
    is_acknowledged: bool = False
    created_at: datetime
    acknowledged_at: Optional[datetime] = None

