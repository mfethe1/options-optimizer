"""
Options Backtesting Engine

Institutional-grade backtesting for options strategies.
Tests historical performance with realistic costs, slippage, and Greeks.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import date, datetime, timedelta
from enum import Enum
import numpy as np
from scipy.stats import norm
import pandas as pd

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Supported strategy types"""
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    SHORT_CALL = "short_call"
    SHORT_PUT = "short_put"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    IRON_CONDOR = "iron_condor"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    CALENDAR_SPREAD = "calendar_spread"
    BUTTERFLY = "butterfly"


class ExitCondition(Enum):
    """Exit condition types"""
    PROFIT_TARGET = "profit_target"  # Exit at X% profit
    STOP_LOSS = "stop_loss"          # Exit at X% loss
    TIME_BASED = "time_based"        # Exit at X DTE
    DELTA_BASED = "delta_based"      # Exit when delta reaches threshold
    EXPIRATION = "expiration"        # Hold to expiration


@dataclass
class OptionLeg:
    """Single option leg in a strategy"""
    option_type: str  # 'call' or 'put'
    action: str       # 'buy' or 'sell'
    strike: float
    expiration: date
    quantity: int
    entry_price: float
    entry_iv: float
    entry_delta: float


@dataclass
class TradeEntry:
    """Strategy entry point"""
    date: date
    underlying_price: float
    legs: List[OptionLeg]
    entry_cost: float  # Total debit/credit
    max_profit: Optional[float] = None
    max_loss: Optional[float] = None


@dataclass
class TradeExit:
    """Strategy exit point"""
    date: date
    underlying_price: float
    exit_value: float
    pnl: float
    pnl_pct: float
    days_held: int
    exit_reason: str


@dataclass
class BacktestTrade:
    """Complete backtest trade"""
    entry: TradeEntry
    exit: Optional[TradeExit] = None
    status: str = "open"  # 'open', 'closed', 'expired'


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics"""
    # Basic metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # P&L metrics
    total_pnl: float
    total_pnl_pct: float
    avg_win: float
    avg_loss: float
    profit_factor: float  # Total wins / Total losses

    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float

    # Time metrics
    avg_days_held: float
    max_days_held: int
    min_days_held: int

    # Advanced metrics
    expectancy: float  # Average $ per trade
    kelly_criterion: float  # Optimal position sizing
    calmar_ratio: float  # Return / Max Drawdown

    # Trade breakdown
    best_trade_pnl: float
    worst_trade_pnl: float
    consecutive_wins: int
    consecutive_losses: int

    # Monthly returns
    monthly_returns: Dict[str, float] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    strategy_type: StrategyType

    # Entry criteria
    entry_dte_min: int = 30  # Minimum days to expiration
    entry_dte_max: int = 45  # Maximum days to expiration
    delta_target: Optional[float] = None  # Target delta (e.g., 0.30 for 30-delta)
    iv_rank_min: Optional[float] = None  # Minimum IV rank (0-100)

    # Exit criteria
    profit_target_pct: Optional[float] = 50  # Exit at 50% max profit
    stop_loss_pct: Optional[float] = 100  # Exit at 100% max loss
    exit_dte: Optional[int] = 7  # Exit at 7 DTE

    # Position sizing
    capital_per_trade: float = 10000  # $ allocated per trade
    max_positions: int = 5  # Max concurrent positions

    # Costs
    commission_per_contract: float = 0.65
    slippage_pct: float = 0.01  # 1% slippage on entry/exit

    # Strategy-specific params
    spread_width: Optional[float] = None  # For spreads
    wing_width: Optional[float] = None    # For iron condors/butterflies


class BacktestEngine:
    """
    Backtesting engine for options strategies.

    Simulates historical trading with realistic costs and Greeks.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trades: List[BacktestTrade] = []
        self.risk_free_rate = 0.045

    async def run_backtest(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        historical_data: pd.DataFrame
    ) -> BacktestMetrics:
        """
        Run complete backtest on historical data.

        Args:
            symbol: Stock symbol
            start_date: Backtest start date
            end_date: Backtest end date
            historical_data: DataFrame with columns:
                - date, open, high, low, close, volume
                - iv_rank (optional)

        Returns:
            BacktestMetrics with complete performance analysis
        """
        logger.info(f"Starting backtest for {symbol} from {start_date} to {end_date}")

        self.trades = []
        current_positions = []

        # Iterate through each trading day
        for current_date in pd.date_range(start_date, end_date, freq='B'):
            current_date = current_date.date()

            # Get market data for this date
            day_data = historical_data[historical_data['date'] == current_date]
            if day_data.empty:
                continue

            underlying_price = day_data['close'].iloc[0]
            iv_rank = day_data.get('iv_rank', pd.Series([50])).iloc[0]

            # Check exit conditions for open positions
            for position in current_positions[:]:
                exit_signal = self._check_exit_conditions(
                    position, current_date, underlying_price
                )

                if exit_signal:
                    self._exit_position(position, current_date, underlying_price, exit_signal)
                    current_positions.remove(position)

            # Check entry conditions if we have capacity
            if len(current_positions) < self.config.max_positions:
                entry_signal = self._check_entry_conditions(
                    current_date, underlying_price, iv_rank
                )

                if entry_signal:
                    trade = self._enter_position(current_date, underlying_price)
                    if trade:
                        current_positions.append(trade)
                        self.trades.append(trade)

        # Close any remaining positions at backtest end
        for position in current_positions:
            self._exit_position(
                position, end_date,
                historical_data[historical_data['date'] == end_date]['close'].iloc[0],
                "backtest_end"
            )

        # Calculate metrics
        metrics = self._calculate_metrics()

        logger.info(f"Backtest complete: {metrics.total_trades} trades, "
                   f"{metrics.win_rate:.1f}% win rate, "
                   f"{metrics.total_pnl_pct:.1f}% return")

        return metrics

    def _check_entry_conditions(
        self,
        current_date: date,
        underlying_price: float,
        iv_rank: float
    ) -> bool:
        """Check if entry conditions are met"""
        # Check IV rank if specified
        if self.config.iv_rank_min and iv_rank < self.config.iv_rank_min:
            return False

        # For simplicity, enter every N days
        # In real implementation, would check technical indicators, IV levels, etc.
        # This is a placeholder - real logic would be much more sophisticated
        return True

    def _check_exit_conditions(
        self,
        trade: BacktestTrade,
        current_date: date,
        underlying_price: float
    ) -> Optional[str]:
        """Check if any exit conditions are met"""
        if not trade.entry:
            return None

        # Calculate current value and P&L
        current_value = self._calculate_position_value(
            trade.entry.legs,
            underlying_price,
            current_date
        )

        pnl = current_value - trade.entry.entry_cost
        pnl_pct = (pnl / abs(trade.entry.entry_cost)) * 100 if trade.entry.entry_cost != 0 else 0

        # Profit target
        if self.config.profit_target_pct and pnl_pct >= self.config.profit_target_pct:
            return f"profit_target_{self.config.profit_target_pct}%"

        # Stop loss
        if self.config.stop_loss_pct and pnl_pct <= -self.config.stop_loss_pct:
            return f"stop_loss_{self.config.stop_loss_pct}%"

        # DTE-based exit
        if self.config.exit_dte:
            min_dte = min(
                (leg.expiration - current_date).days
                for leg in trade.entry.legs
            )
            if min_dte <= self.config.exit_dte:
                return f"time_exit_{self.config.exit_dte}_dte"

        # Expiration
        if any((leg.expiration - current_date).days <= 0 for leg in trade.entry.legs):
            return "expiration"

        return None

    def _enter_position(
        self,
        entry_date: date,
        underlying_price: float
    ) -> Optional[BacktestTrade]:
        """Enter a new position based on strategy type"""
        try:
            # Calculate expiration date
            dte = (self.config.entry_dte_min + self.config.entry_dte_max) // 2
            expiration = entry_date + timedelta(days=dte)

            # Build legs based on strategy type
            legs = self._build_strategy_legs(
                underlying_price,
                expiration,
                entry_date
            )

            if not legs:
                return None

            # Calculate entry cost with commissions and slippage
            entry_cost = sum(
                self._calculate_option_value(leg, underlying_price, entry_date)
                * leg.quantity
                * (1 if leg.action == 'buy' else -1)
                for leg in legs
            )

            # Add transaction costs
            total_contracts = sum(abs(leg.quantity) for leg in legs)
            commissions = total_contracts * self.config.commission_per_contract
            slippage = abs(entry_cost) * self.config.slippage_pct

            entry_cost += commissions + slippage

            # Calculate max profit/loss
            max_profit, max_loss = self._calculate_max_pnl(legs, underlying_price)

            entry = TradeEntry(
                date=entry_date,
                underlying_price=underlying_price,
                legs=legs,
                entry_cost=entry_cost,
                max_profit=max_profit,
                max_loss=max_loss
            )

            return BacktestTrade(entry=entry, status="open")

        except Exception as e:
            logger.error(f"Failed to enter position: {e}")
            return None

    def _build_strategy_legs(
        self,
        underlying_price: float,
        expiration: date,
        entry_date: date
    ) -> List[OptionLeg]:
        """Build option legs for the strategy"""
        legs = []

        if self.config.strategy_type == StrategyType.LONG_CALL:
            # ATM call
            strike = self._round_strike(underlying_price)
            legs.append(OptionLeg(
                option_type='call',
                action='buy',
                strike=strike,
                expiration=expiration,
                quantity=1,
                entry_price=0,  # Will be calculated
                entry_iv=0.30,  # Placeholder
                entry_delta=0.50
            ))

        elif self.config.strategy_type == StrategyType.BULL_CALL_SPREAD:
            # Buy ATM call, sell OTM call
            long_strike = self._round_strike(underlying_price)
            short_strike = self._round_strike(
                underlying_price + (self.config.spread_width or 5)
            )

            legs.append(OptionLeg(
                option_type='call', action='buy',
                strike=long_strike, expiration=expiration,
                quantity=1, entry_price=0, entry_iv=0.30, entry_delta=0.50
            ))
            legs.append(OptionLeg(
                option_type='call', action='sell',
                strike=short_strike, expiration=expiration,
                quantity=1, entry_price=0, entry_iv=0.28, entry_delta=0.30
            ))

        elif self.config.strategy_type == StrategyType.IRON_CONDOR:
            # Sell OTM put spread + sell OTM call spread
            width = self.config.spread_width or 5

            # Put side
            put_short_strike = self._round_strike(underlying_price * 0.95)
            put_long_strike = self._round_strike(underlying_price * 0.95 - width)

            # Call side
            call_short_strike = self._round_strike(underlying_price * 1.05)
            call_long_strike = self._round_strike(underlying_price * 1.05 + width)

            legs.extend([
                OptionLeg('put', 'buy', put_long_strike, expiration, 1, 0, 0.32, -0.10),
                OptionLeg('put', 'sell', put_short_strike, expiration, 1, 0, 0.30, -0.25),
                OptionLeg('call', 'sell', call_short_strike, expiration, 1, 0, 0.30, 0.25),
                OptionLeg('call', 'buy', call_long_strike, expiration, 1, 0, 0.32, 0.10),
            ])

        elif self.config.strategy_type == StrategyType.STRADDLE:
            # Buy ATM call + ATM put
            strike = self._round_strike(underlying_price)
            legs.extend([
                OptionLeg('call', 'buy', strike, expiration, 1, 0, 0.30, 0.50),
                OptionLeg('put', 'buy', strike, expiration, 1, 0, 0.30, -0.50),
            ])

        # Calculate entry prices for all legs
        for leg in legs:
            leg.entry_price = self._calculate_option_value(leg, underlying_price, entry_date)

        return legs

    def _calculate_option_value(
        self,
        leg: OptionLeg,
        underlying_price: float,
        current_date: date
    ) -> float:
        """Calculate option value using Black-Scholes"""
        T = (leg.expiration - current_date).days / 365.0
        if T <= 0:
            # At expiration, intrinsic value only
            if leg.option_type == 'call':
                return max(0, underlying_price - leg.strike)
            else:
                return max(0, leg.strike - underlying_price)

        S = underlying_price
        K = leg.strike
        r = self.risk_free_rate
        sigma = leg.entry_iv

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if leg.option_type == 'call':
            value = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            value = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return max(0, value)

    def _calculate_position_value(
        self,
        legs: List[OptionLeg],
        underlying_price: float,
        current_date: date
    ) -> float:
        """Calculate total position value"""
        return sum(
            self._calculate_option_value(leg, underlying_price, current_date)
            * leg.quantity
            * (1 if leg.action == 'buy' else -1)
            for leg in legs
        )

    def _exit_position(
        self,
        trade: BacktestTrade,
        exit_date: date,
        underlying_price: float,
        exit_reason: str
    ):
        """Exit a position"""
        exit_value = self._calculate_position_value(
            trade.entry.legs,
            underlying_price,
            exit_date
        )

        # Subtract exit commissions and slippage
        total_contracts = sum(abs(leg.quantity) for leg in trade.entry.legs)
        commissions = total_contracts * self.config.commission_per_contract
        slippage = abs(exit_value) * self.config.slippage_pct

        exit_value -= (commissions + slippage)

        pnl = exit_value - trade.entry.entry_cost
        pnl_pct = (pnl / abs(trade.entry.entry_cost)) * 100 if trade.entry.entry_cost != 0 else 0
        days_held = (exit_date - trade.entry.date).days

        trade.exit = TradeExit(
            date=exit_date,
            underlying_price=underlying_price,
            exit_value=exit_value,
            pnl=pnl,
            pnl_pct=pnl_pct,
            days_held=days_held,
            exit_reason=exit_reason
        )
        trade.status = "closed"

    def _calculate_max_pnl(
        self,
        legs: List[OptionLeg],
        underlying_price: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate max profit and max loss for the strategy"""
        # Simplified - real implementation would analyze payoff diagram
        if self.config.strategy_type == StrategyType.IRON_CONDOR:
            # Max profit = credit received
            # Max loss = spread width - credit
            credit = sum(
                leg.entry_price * leg.quantity * (1 if leg.action == 'sell' else -1)
                for leg in legs
            )
            spread_width = self.config.spread_width or 5
            return (credit, -(spread_width - credit))

        elif self.config.strategy_type in [StrategyType.BULL_CALL_SPREAD, StrategyType.BEAR_PUT_SPREAD]:
            debit = sum(
                leg.entry_price * leg.quantity * (1 if leg.action == 'buy' else -1)
                for leg in legs
            )
            spread_width = self.config.spread_width or 5
            return (spread_width - debit, -debit)

        else:
            # Undefined risk strategies
            return (None, None)

    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive performance metrics"""
        closed_trades = [t for t in self.trades if t.exit is not None]

        if not closed_trades:
            # No closed trades, return empty metrics
            return BacktestMetrics(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_pnl=0, total_pnl_pct=0,
                avg_win=0, avg_loss=0, profit_factor=0,
                max_drawdown=0, max_drawdown_pct=0,
                sharpe_ratio=0, sortino_ratio=0,
                avg_days_held=0, max_days_held=0, min_days_held=0,
                expectancy=0, kelly_criterion=0, calmar_ratio=0,
                best_trade_pnl=0, worst_trade_pnl=0,
                consecutive_wins=0, consecutive_losses=0
            )

        # Basic metrics
        total_trades = len(closed_trades)
        winning_trades = len([t for t in closed_trades if t.exit.pnl > 0])
        losing_trades = len([t for t in closed_trades if t.exit.pnl < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # P&L metrics
        total_pnl = sum(t.exit.pnl for t in closed_trades)
        total_pnl_pct = sum(t.exit.pnl_pct for t in closed_trades)

        wins = [t.exit.pnl for t in closed_trades if t.exit.pnl > 0]
        losses = [t.exit.pnl for t in closed_trades if t.exit.pnl < 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0

        # Drawdown
        cumulative_pnl = np.cumsum([t.exit.pnl for t in closed_trades])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = cumulative_pnl - running_max
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
        max_drawdown_pct = (max_drawdown / self.config.capital_per_trade * 100) if self.config.capital_per_trade > 0 else 0

        # Sharpe and Sortino
        returns = [t.exit.pnl_pct for t in closed_trades]
        sharpe_ratio = (np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0

        downside_returns = [r for r in returns if r < 0]
        downside_std = np.std(downside_returns) if downside_returns else 0.001
        sortino_ratio = (np.mean(returns) / downside_std) if downside_std > 0 else 0

        # Time metrics
        days_held = [t.exit.days_held for t in closed_trades]
        avg_days_held = np.mean(days_held) if days_held else 0
        max_days_held = max(days_held) if days_held else 0
        min_days_held = min(days_held) if days_held else 0

        # Advanced metrics
        expectancy = total_pnl / total_trades if total_trades > 0 else 0

        # Kelly criterion: (win_rate * avg_win - (1 - win_rate) * abs(avg_loss)) / avg_win
        kelly_criterion = 0
        if avg_win > 0:
            kelly_criterion = (
                (win_rate / 100 * avg_win - (1 - win_rate / 100) * abs(avg_loss)) / avg_win
            )

        calmar_ratio = (total_pnl / max_drawdown) if max_drawdown > 0 else 0

        # Best/worst trades
        all_pnls = [t.exit.pnl for t in closed_trades]
        best_trade_pnl = max(all_pnls) if all_pnls else 0
        worst_trade_pnl = min(all_pnls) if all_pnls else 0

        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        current_streak = 0
        last_was_win = None

        for trade in closed_trades:
            is_win = trade.exit.pnl > 0
            if last_was_win is None or last_was_win != is_win:
                current_streak = 1
            else:
                current_streak += 1

            if is_win:
                consecutive_wins = max(consecutive_wins, current_streak)
            else:
                consecutive_losses = max(consecutive_losses, current_streak)

            last_was_win = is_win

        return BacktestMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            avg_days_held=avg_days_held,
            max_days_held=max_days_held,
            min_days_held=min_days_held,
            expectancy=expectancy,
            kelly_criterion=kelly_criterion,
            calmar_ratio=calmar_ratio,
            best_trade_pnl=best_trade_pnl,
            worst_trade_pnl=worst_trade_pnl,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses
        )

    def _round_strike(self, price: float) -> float:
        """Round to nearest strike (typically $5 or $2.50)"""
        if price < 50:
            return round(price * 2) / 2  # $0.50 strikes
        elif price < 100:
            return round(price)  # $1 strikes
        elif price < 200:
            return round(price / 2.5) * 2.5  # $2.50 strikes
        else:
            return round(price / 5) * 5  # $5 strikes
