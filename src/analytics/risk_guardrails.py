"""
Institutional Risk Management Guardrails

Comprehensive risk management system with institutional-grade controls.
Enforces position limits, portfolio constraints, and risk budgets.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Risk Limits Configuration
# ============================================================================

class RiskLevel(Enum):
    """Risk tolerance levels"""
    ULTRA_CONSERVATIVE = "ultra_conservative"
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ULTRA_AGGRESSIVE = "ultra_aggressive"


@dataclass
class RiskLimits:
    """Risk limits configuration"""
    # Position Limits
    max_position_size_pct: float  # Max % of portfolio in single position
    max_sector_exposure_pct: float  # Max % in single sector
    max_correlation_exposure: float  # Max combined weight of correlated positions

    # Loss Limits
    max_daily_loss_pct: float  # Max daily portfolio loss %
    max_weekly_loss_pct: float  # Max weekly portfolio loss %
    max_monthly_loss_pct: float  # Max monthly portfolio loss %
    max_drawdown_pct: float  # Max portfolio drawdown %
    max_loss_per_trade_pct: float  # Max loss per trade (% of portfolio)

    # Leverage Limits
    max_leverage: float  # Max portfolio leverage
    max_options_notional_pct: float  # Max notional options exposure

    # Concentration Limits
    max_positions: int  # Max number of open positions
    min_positions: int  # Min positions for diversification
    max_same_expiration_pct: float  # Max % in same expiration

    # Volatility Limits
    max_portfolio_volatility: float  # Max annualized portfolio vol
    max_position_beta: float  # Max beta for single position

    # Liquidity Requirements
    min_daily_volume: int  # Min average daily volume
    min_open_interest: int  # Min open interest for options
    max_bid_ask_spread_pct: float  # Max bid-ask spread %

    # Capital Requirements
    min_cash_reserve_pct: float  # Min % portfolio in cash
    max_capital_deployment_pct: float  # Max % deployed


# Preset risk configurations for different investor types
RISK_PRESETS = {
    RiskLevel.ULTRA_CONSERVATIVE: RiskLimits(
        max_position_size_pct=5.0,
        max_sector_exposure_pct=20.0,
        max_correlation_exposure=0.5,
        max_daily_loss_pct=0.5,
        max_weekly_loss_pct=2.0,
        max_monthly_loss_pct=5.0,
        max_drawdown_pct=10.0,
        max_loss_per_trade_pct=1.0,
        max_leverage=1.0,
        max_options_notional_pct=20.0,
        max_positions=20,
        min_positions=10,
        max_same_expiration_pct=15.0,
        max_portfolio_volatility=15.0,
        max_position_beta=1.2,
        min_daily_volume=500000,
        min_open_interest=1000,
        max_bid_ask_spread_pct=2.0,
        min_cash_reserve_pct=25.0,
        max_capital_deployment_pct=75.0
    ),
    RiskLevel.CONSERVATIVE: RiskLimits(
        max_position_size_pct=8.0,
        max_sector_exposure_pct=30.0,
        max_correlation_exposure=0.6,
        max_daily_loss_pct=1.0,
        max_weekly_loss_pct=3.0,
        max_monthly_loss_pct=8.0,
        max_drawdown_pct=15.0,
        max_loss_per_trade_pct=1.5,
        max_leverage=1.5,
        max_options_notional_pct=30.0,
        max_positions=15,
        min_positions=8,
        max_same_expiration_pct=20.0,
        max_portfolio_volatility=20.0,
        max_position_beta=1.5,
        min_daily_volume=250000,
        min_open_interest=500,
        max_bid_ask_spread_pct=3.0,
        min_cash_reserve_pct=20.0,
        max_capital_deployment_pct=80.0
    ),
    RiskLevel.MODERATE: RiskLimits(
        max_position_size_pct=10.0,
        max_sector_exposure_pct=40.0,
        max_correlation_exposure=0.7,
        max_daily_loss_pct=2.0,
        max_weekly_loss_pct=5.0,
        max_monthly_loss_pct=12.0,
        max_drawdown_pct=20.0,
        max_loss_per_trade_pct=2.0,
        max_leverage=2.0,
        max_options_notional_pct=50.0,
        max_positions=12,
        min_positions=5,
        max_same_expiration_pct=25.0,
        max_portfolio_volatility=25.0,
        max_position_beta=2.0,
        min_daily_volume=100000,
        min_open_interest=250,
        max_bid_ask_spread_pct=5.0,
        min_cash_reserve_pct=15.0,
        max_capital_deployment_pct=85.0
    ),
    RiskLevel.AGGRESSIVE: RiskLimits(
        max_position_size_pct=15.0,
        max_sector_exposure_pct=50.0,
        max_correlation_exposure=0.8,
        max_daily_loss_pct=3.0,
        max_weekly_loss_pct=8.0,
        max_monthly_loss_pct=18.0,
        max_drawdown_pct=30.0,
        max_loss_per_trade_pct=3.0,
        max_leverage=3.0,
        max_options_notional_pct=75.0,
        max_positions=10,
        min_positions=3,
        max_same_expiration_pct=35.0,
        max_portfolio_volatility=35.0,
        max_position_beta=2.5,
        min_daily_volume=50000,
        min_open_interest=100,
        max_bid_ask_spread_pct=8.0,
        min_cash_reserve_pct=10.0,
        max_capital_deployment_pct=90.0
    ),
    RiskLevel.ULTRA_AGGRESSIVE: RiskLimits(
        max_position_size_pct=20.0,
        max_sector_exposure_pct=60.0,
        max_correlation_exposure=0.9,
        max_daily_loss_pct=5.0,
        max_weekly_loss_pct=12.0,
        max_monthly_loss_pct=25.0,
        max_drawdown_pct=40.0,
        max_loss_per_trade_pct=5.0,
        max_leverage=4.0,
        max_options_notional_pct=100.0,
        max_positions=8,
        min_positions=2,
        max_same_expiration_pct=50.0,
        max_portfolio_volatility=50.0,
        max_position_beta=3.0,
        min_daily_volume=25000,
        min_open_interest=50,
        max_bid_ask_spread_pct=12.0,
        min_cash_reserve_pct=5.0,
        max_capital_deployment_pct=95.0
    )
}


# ============================================================================
# Risk Check Results
# ============================================================================

@dataclass
class RiskViolation:
    """A single risk limit violation"""
    severity: str  # CRITICAL, WARNING, INFO
    rule_name: str
    current_value: float
    limit_value: float
    message: str
    blocking: bool  # If true, prevents trade execution


@dataclass
class RiskCheckResult:
    """Result of risk checks"""
    approved: bool
    violations: List[RiskViolation]
    warnings: List[RiskViolation]
    max_position_size: float  # Max size allowed for this trade
    suggested_position_size: float  # AI-suggested optimal size
    risk_score: float  # 0-100, higher = riskier
    risk_level: str
    detailed_report: str


# ============================================================================
# Portfolio State
# ============================================================================

@dataclass
class Position:
    """Current portfolio position"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    position_type: str  # STOCK, CALL, PUT
    expiration: Optional[str]
    sector: str
    beta: float


@dataclass
class PortfolioState:
    """Current portfolio state"""
    total_value: float
    cash: float
    positions: List[Position]
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float
    max_drawdown: float
    current_leverage: float


# ============================================================================
# Risk Guardrails Service
# ============================================================================

class RiskGuardrailsService:
    """Institutional-grade risk management guardrails"""

    def __init__(self, risk_level: RiskLevel = RiskLevel.MODERATE):
        self.risk_level = risk_level
        self.limits = RISK_PRESETS[risk_level]
        self.loss_tracking: Dict[str, List[Tuple[datetime, float]]] = {
            "daily": [],
            "weekly": [],
            "monthly": []
        }

    def check_new_position(
        self,
        symbol: str,
        proposed_size: float,  # Dollar amount
        position_type: str,
        portfolio: PortfolioState,
        market_data: Dict[str, Any]
    ) -> RiskCheckResult:
        """
        Comprehensive risk check for new position.

        Returns approval status and any violations/warnings.
        """
        violations = []
        warnings = []
        risk_score = 0

        # 1. Position Size Check
        position_pct = (proposed_size / portfolio.total_value) * 100
        if position_pct > self.limits.max_position_size_pct:
            violations.append(RiskViolation(
                severity="CRITICAL",
                rule_name="Max Position Size",
                current_value=position_pct,
                limit_value=self.limits.max_position_size_pct,
                message=f"Position size {position_pct:.1f}% exceeds limit of {self.limits.max_position_size_pct:.1f}%",
                blocking=True
            ))
            risk_score += 30
        elif position_pct > self.limits.max_position_size_pct * 0.8:
            warnings.append(RiskViolation(
                severity="WARNING",
                rule_name="Position Size",
                current_value=position_pct,
                limit_value=self.limits.max_position_size_pct,
                message=f"Position size {position_pct:.1f}% approaching limit",
                blocking=False
            ))
            risk_score += 15

        # 2. Portfolio Concentration Check
        if len(portfolio.positions) >= self.limits.max_positions:
            violations.append(RiskViolation(
                severity="CRITICAL",
                rule_name="Max Positions",
                current_value=len(portfolio.positions),
                limit_value=self.limits.max_positions,
                message=f"Portfolio at max {self.limits.max_positions} positions",
                blocking=True
            ))
            risk_score += 25

        # 3. Cash Reserve Check
        cash_pct = (portfolio.cash / portfolio.total_value) * 100
        new_cash_pct = ((portfolio.cash - proposed_size) / portfolio.total_value) * 100
        if new_cash_pct < self.limits.min_cash_reserve_pct:
            violations.append(RiskViolation(
                severity="CRITICAL",
                rule_name="Minimum Cash Reserve",
                current_value=new_cash_pct,
                limit_value=self.limits.min_cash_reserve_pct,
                message=f"Cash reserve {new_cash_pct:.1f}% below minimum {self.limits.min_cash_reserve_pct:.1f}%",
                blocking=True
            ))
            risk_score += 35

        # 4. Daily Loss Limit Check
        daily_loss_pct = abs(portfolio.daily_pnl / portfolio.total_value) * 100
        if portfolio.daily_pnl < 0 and daily_loss_pct > self.limits.max_daily_loss_pct:
            violations.append(RiskViolation(
                severity="CRITICAL",
                rule_name="Daily Loss Limit",
                current_value=daily_loss_pct,
                limit_value=self.limits.max_daily_loss_pct,
                message=f"Daily loss {daily_loss_pct:.1f}% exceeds limit of {self.limits.max_daily_loss_pct:.1f}%",
                blocking=True
            ))
            risk_score += 40

        # 5. Drawdown Check
        if portfolio.max_drawdown > self.limits.max_drawdown_pct:
            violations.append(RiskViolation(
                severity="CRITICAL",
                rule_name="Maximum Drawdown",
                current_value=portfolio.max_drawdown,
                limit_value=self.limits.max_drawdown_pct,
                message=f"Drawdown {portfolio.max_drawdown:.1f}% exceeds limit of {self.limits.max_drawdown_pct:.1f}%",
                blocking=True
            ))
            risk_score += 45

        # 6. Leverage Check
        new_leverage = (portfolio.total_value + proposed_size) / (portfolio.total_value - portfolio.cash)
        if new_leverage > self.limits.max_leverage:
            violations.append(RiskViolation(
                severity="CRITICAL",
                rule_name="Maximum Leverage",
                current_value=new_leverage,
                limit_value=self.limits.max_leverage,
                message=f"Leverage {new_leverage:.2f}x exceeds limit of {self.limits.max_leverage:.2f}x",
                blocking=True
            ))
            risk_score += 35

        # 7. Liquidity Check (if market data available)
        if "avg_volume" in market_data:
            if market_data["avg_volume"] < self.limits.min_daily_volume:
                warnings.append(RiskViolation(
                    severity="WARNING",
                    rule_name="Minimum Volume",
                    current_value=market_data["avg_volume"],
                    limit_value=self.limits.min_daily_volume,
                    message=f"Average volume {market_data['avg_volume']:,} below minimum",
                    blocking=False
                ))
                risk_score += 10

        # 8. Bid-Ask Spread Check
        if "bid_ask_spread_pct" in market_data:
            if market_data["bid_ask_spread_pct"] > self.limits.max_bid_ask_spread_pct:
                warnings.append(RiskViolation(
                    severity="WARNING",
                    rule_name="Bid-Ask Spread",
                    current_value=market_data["bid_ask_spread_pct"],
                    limit_value=self.limits.max_bid_ask_spread_pct,
                    message=f"Bid-ask spread {market_data['bid_ask_spread_pct']:.2f}% too wide",
                    blocking=False
                ))
                risk_score += 8

        # Calculate max and suggested position sizes
        max_size_by_limits = portfolio.total_value * (self.limits.max_position_size_pct / 100)
        max_size_by_cash = portfolio.cash - (portfolio.total_value * (self.limits.min_cash_reserve_pct / 100))
        max_position_size = min(max_size_by_limits, max_size_by_cash)

        # Suggested size is conservative (60% of max)
        suggested_position_size = max_position_size * 0.6

        # Determine approval
        blocking_violations = [v for v in violations if v.blocking]
        approved = len(blocking_violations) == 0 and risk_score < 70

        # Generate detailed report
        report = self._generate_risk_report(
            violations, warnings, risk_score, portfolio, proposed_size
        )

        # Determine risk level
        if risk_score >= 80:
            risk_level_str = "EXTREME"
        elif risk_score >= 60:
            risk_level_str = "HIGH"
        elif risk_score >= 40:
            risk_level_str = "MODERATE"
        elif risk_score >= 20:
            risk_level_str = "LOW"
        else:
            risk_level_str = "MINIMAL"

        return RiskCheckResult(
            approved=approved,
            violations=violations,
            warnings=warnings,
            max_position_size=max_position_size,
            suggested_position_size=suggested_position_size,
            risk_score=risk_score,
            risk_level=risk_level_str,
            detailed_report=report
        )

    def _generate_risk_report(
        self,
        violations: List[RiskViolation],
        warnings: List[RiskViolation],
        risk_score: float,
        portfolio: PortfolioState,
        proposed_size: float
    ) -> str:
        """Generate detailed risk report"""
        report = "=" * 80 + "\n"
        report += "INSTITUTIONAL RISK MANAGEMENT REPORT\n"
        report += "=" * 80 + "\n\n"

        report += f"Risk Configuration: {self.risk_level.value.upper()}\n"
        report += f"Overall Risk Score: {risk_score:.1f}/100\n\n"

        report += f"Portfolio Status:\n"
        report += f"  Total Value: ${portfolio.total_value:,.2f}\n"
        report += f"  Cash: ${portfolio.cash:,.2f} ({portfolio.cash/portfolio.total_value*100:.1f}%)\n"
        report += f"  Positions: {len(portfolio.positions)}/{self.limits.max_positions}\n"
        report += f"  Leverage: {portfolio.current_leverage:.2f}x\n"
        report += f"  Daily P&L: ${portfolio.daily_pnl:,.2f}\n"
        report += f"  Max Drawdown: {portfolio.max_drawdown:.1f}%\n\n"

        if violations:
            report += f"CRITICAL VIOLATIONS ({len(violations)}):\n"
            for v in violations:
                report += f"  ❌ {v.rule_name}: {v.message}\n"
            report += "\n"

        if warnings:
            report += f"WARNINGS ({len(warnings)}):\n"
            for w in warnings:
                report += f"  ⚠️  {w.rule_name}: {w.message}\n"
            report += "\n"

        blocking = [v for v in violations if v.blocking]
        if blocking:
            report += "❌ TRADE REJECTED - Critical risk violations must be resolved.\n"
        else:
            report += "✅ TRADE APPROVED - All risk checks passed.\n"

        return report

    def calculate_portfolio_var(
        self,
        portfolio: PortfolioState,
        confidence_level: float = 0.95,
        time_horizon_days: int = 1
    ) -> float:
        """
        Calculate portfolio Value at Risk (VaR).

        Returns maximum expected loss at given confidence level.
        """
        # Simplified VaR calculation using historical method
        # In production, would use actual returns data
        portfolio_volatility = 0.20  # Assume 20% annual vol
        daily_volatility = portfolio_volatility / np.sqrt(252)

        # Z-score for 95% confidence
        z_score = 1.645 if confidence_level == 0.95 else 1.96

        var = portfolio.total_value * daily_volatility * z_score * np.sqrt(time_horizon_days)

        return var

    def get_position_sizing_recommendation(
        self,
        kelly_fraction: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        portfolio_value: float
    ) -> Dict[str, float]:
        """
        Get position sizing recommendation using Kelly Criterion and risk limits.

        Returns dict with full_kelly, half_kelly, and recommended sizes.
        """
        # Full Kelly
        full_kelly_pct = kelly_fraction * 100

        # Half Kelly (more conservative)
        half_kelly_pct = full_kelly_pct / 2

        # Apply risk limits
        max_allowed_pct = self.limits.max_position_size_pct
        recommended_pct = min(half_kelly_pct, max_allowed_pct)

        # Never exceed 2% risk per trade
        max_risk_pct = min(recommended_pct, self.limits.max_loss_per_trade_pct)

        return {
            "full_kelly_pct": full_kelly_pct,
            "full_kelly_dollars": portfolio_value * (full_kelly_pct / 100),
            "half_kelly_pct": half_kelly_pct,
            "half_kelly_dollars": portfolio_value * (half_kelly_pct / 100),
            "recommended_pct": recommended_pct,
            "recommended_dollars": portfolio_value * (recommended_pct / 100),
            "max_risk_pct": max_risk_pct,
            "max_risk_dollars": portfolio_value * (max_risk_pct / 100)
        }
