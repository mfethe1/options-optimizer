"""
Execution Quality Tracking Service

Tracks trade execution metrics to optimize fill quality and minimize slippage.
Critical for maximizing returns - even 1% slippage = 12% annual performance drag.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, date, time
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order type"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class ExecutionRecord:
    """Single execution record"""
    order_id: str
    symbol: str
    order_type: OrderType
    order_side: OrderSide
    quantity: int

    # Timing
    order_time: datetime
    fill_time: Optional[datetime] = None
    time_to_fill_ms: Optional[float] = None

    # Pricing
    expected_price: float = 0.0  # Quote at order time
    limit_price: Optional[float] = None
    fill_price: Optional[float] = None

    # Execution quality
    slippage_dollars: Optional[float] = None
    slippage_bps: Optional[float] = None  # Basis points (1 bp = 0.01%)
    price_improvement: Optional[float] = None

    # Fill details
    filled_quantity: int = 0
    status: OrderStatus = OrderStatus.PENDING

    # Context
    broker: str = "unknown"
    venue: Optional[str] = None  # Execution venue (ARCA, NASDAQ, etc.)
    market_conditions: Optional[str] = None  # "normal", "high_vol", "low_liquidity"
    bid_ask_spread: Optional[float] = None

    # Post-execution tracking
    price_1min_after: Optional[float] = None
    price_5min_after: Optional[float] = None
    adverse_selection: Optional[float] = None  # Price moved against you


@dataclass
class ExecutionQualityMetrics:
    """Aggregate execution quality metrics"""
    # Overall metrics
    total_executions: int
    total_volume: float
    avg_slippage_bps: float
    median_slippage_bps: float

    # Fill metrics
    fill_rate: float  # % of orders filled
    avg_time_to_fill_ms: float
    partial_fill_rate: float

    # Quality metrics
    price_improvement_rate: float  # % of orders with price improvement
    avg_price_improvement_bps: float
    adverse_selection_rate: float  # % with adverse price move
    avg_adverse_selection_bps: float

    # Slippage distribution
    slippage_25th_percentile: float
    slippage_75th_percentile: float
    slippage_95th_percentile: float
    worst_slippage_bps: float
    best_slippage_bps: float

    # Cost analysis
    total_slippage_cost: float
    estimated_annual_drag: float  # Annualized performance drag


@dataclass
class ExecutionAnalysis:
    """Detailed execution analysis with breakdowns"""
    overall_metrics: ExecutionQualityMetrics

    # Breakdowns
    by_broker: Dict[str, ExecutionQualityMetrics]
    by_time_of_day: Dict[str, ExecutionQualityMetrics]  # "morning", "midday", "close"
    by_order_type: Dict[str, ExecutionQualityMetrics]
    by_symbol: Dict[str, ExecutionQualityMetrics]

    # Recommendations
    recommendations: List[str] = field(default_factory=list)


class ExecutionQualityService:
    """Service for tracking and analyzing execution quality"""

    def __init__(self):
        self.executions: List[ExecutionRecord] = []

    async def record_order(
        self,
        order_id: str,
        symbol: str,
        order_type: OrderType,
        order_side: OrderSide,
        quantity: int,
        expected_price: float,
        broker: str,
        limit_price: Optional[float] = None,
        bid_ask_spread: Optional[float] = None
    ) -> ExecutionRecord:
        """
        Record a new order when placed.

        Returns:
            ExecutionRecord with initial data
        """
        record = ExecutionRecord(
            order_id=order_id,
            symbol=symbol,
            order_type=order_type,
            order_side=order_side,
            quantity=quantity,
            order_time=datetime.now(),
            expected_price=expected_price,
            limit_price=limit_price,
            broker=broker,
            bid_ask_spread=bid_ask_spread,
            status=OrderStatus.PENDING
        )

        self.executions.append(record)
        logger.info(f"Recorded order {order_id} for {symbol}")

        return record

    async def record_fill(
        self,
        order_id: str,
        fill_price: float,
        filled_quantity: int,
        venue: Optional[str] = None,
        partial: bool = False
    ) -> Optional[ExecutionRecord]:
        """
        Record order fill.

        Updates execution record with fill details and calculates metrics.
        """
        # Find execution record
        record = next((e for e in self.executions if e.order_id == order_id), None)

        if not record:
            logger.error(f"Order {order_id} not found")
            return None

        # Update fill details
        record.fill_time = datetime.now()
        record.fill_price = fill_price
        record.filled_quantity = filled_quantity
        record.venue = venue
        record.status = OrderStatus.PARTIALLY_FILLED if partial else OrderStatus.FILLED

        # Calculate time to fill
        if record.order_time and record.fill_time:
            delta = record.fill_time - record.order_time
            record.time_to_fill_ms = delta.total_seconds() * 1000

        # Calculate slippage
        self._calculate_slippage(record)

        logger.info(f"Recorded fill for {order_id}: {filled_quantity} @ ${fill_price:.2f}, "
                   f"slippage: {record.slippage_bps:.1f} bps")

        return record

    async def record_price_after_fill(
        self,
        order_id: str,
        price_1min: float,
        price_5min: float
    ):
        """Record prices after fill to calculate adverse selection"""
        record = next((e for e in self.executions if e.order_id == order_id), None)

        if not record or not record.fill_price:
            return

        record.price_1min_after = price_1min
        record.price_5min_after = price_5min

        # Calculate adverse selection (5min price)
        if record.order_side == OrderSide.BUY:
            # For buys, adverse = price went up after fill (paid more than needed)
            record.adverse_selection = (price_5min - record.fill_price) / record.fill_price * 10000
        else:
            # For sells, adverse = price went down after fill (got less than could have)
            record.adverse_selection = (record.fill_price - price_5min) / record.fill_price * 10000

    def _calculate_slippage(self, record: ExecutionRecord):
        """Calculate slippage metrics for an execution"""
        if not record.fill_price or not record.expected_price:
            return

        # Slippage in dollars
        if record.order_side == OrderSide.BUY:
            # Positive slippage = paid more than expected (bad)
            record.slippage_dollars = (record.fill_price - record.expected_price) * record.filled_quantity
        else:
            # Positive slippage = received less than expected (bad)
            record.slippage_dollars = (record.expected_price - record.fill_price) * record.filled_quantity

        # Slippage in basis points (1 bp = 0.01%)
        record.slippage_bps = (record.slippage_dollars / (record.expected_price * record.filled_quantity)) * 10000

        # Price improvement (negative slippage is good)
        if record.slippage_bps < 0:
            record.price_improvement = -record.slippage_bps

    async def get_execution_analysis(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        symbol: Optional[str] = None,
        broker: Optional[str] = None
    ) -> ExecutionAnalysis:
        """
        Get comprehensive execution quality analysis.

        Returns:
            ExecutionAnalysis with metrics and recommendations
        """
        # Filter executions
        filtered = self._filter_executions(start_date, end_date, symbol, broker)

        if not filtered:
            # Return empty metrics
            empty_metrics = ExecutionQualityMetrics(
                total_executions=0, total_volume=0, avg_slippage_bps=0,
                median_slippage_bps=0, fill_rate=0, avg_time_to_fill_ms=0,
                partial_fill_rate=0, price_improvement_rate=0,
                avg_price_improvement_bps=0, adverse_selection_rate=0,
                avg_adverse_selection_bps=0, slippage_25th_percentile=0,
                slippage_75th_percentile=0, slippage_95th_percentile=0,
                worst_slippage_bps=0, best_slippage_bps=0,
                total_slippage_cost=0, estimated_annual_drag=0
            )

            return ExecutionAnalysis(
                overall_metrics=empty_metrics,
                by_broker={},
                by_time_of_day={},
                by_order_type={},
                by_symbol={},
                recommendations=["No execution data available"]
            )

        # Calculate overall metrics
        overall_metrics = self._calculate_metrics(filtered)

        # Calculate breakdowns
        by_broker = self._breakdown_by_broker(filtered)
        by_time_of_day = self._breakdown_by_time_of_day(filtered)
        by_order_type = self._breakdown_by_order_type(filtered)
        by_symbol = self._breakdown_by_symbol(filtered)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_metrics, by_broker, by_time_of_day, by_order_type
        )

        return ExecutionAnalysis(
            overall_metrics=overall_metrics,
            by_broker=by_broker,
            by_time_of_day=by_time_of_day,
            by_order_type=by_order_type,
            by_symbol=by_symbol,
            recommendations=recommendations
        )

    def _filter_executions(
        self,
        start_date: Optional[date],
        end_date: Optional[date],
        symbol: Optional[str],
        broker: Optional[str]
    ) -> List[ExecutionRecord]:
        """Filter executions by criteria"""
        filtered = [e for e in self.executions if e.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]]

        if start_date:
            filtered = [e for e in filtered if e.order_time.date() >= start_date]
        if end_date:
            filtered = [e for e in filtered if e.order_time.date() <= end_date]
        if symbol:
            filtered = [e for e in filtered if e.symbol == symbol]
        if broker:
            filtered = [e for e in filtered if e.broker == broker]

        return filtered

    def _calculate_metrics(self, executions: List[ExecutionRecord]) -> ExecutionQualityMetrics:
        """Calculate aggregate metrics for executions"""
        if not executions:
            return ExecutionQualityMetrics(
                total_executions=0, total_volume=0, avg_slippage_bps=0,
                median_slippage_bps=0, fill_rate=0, avg_time_to_fill_ms=0,
                partial_fill_rate=0, price_improvement_rate=0,
                avg_price_improvement_bps=0, adverse_selection_rate=0,
                avg_adverse_selection_bps=0, slippage_25th_percentile=0,
                slippage_75th_percentile=0, slippage_95th_percentile=0,
                worst_slippage_bps=0, best_slippage_bps=0,
                total_slippage_cost=0, estimated_annual_drag=0
            )

        # Extract slippages
        slippages = [e.slippage_bps for e in executions if e.slippage_bps is not None]
        slippage_dollars = [e.slippage_dollars for e in executions if e.slippage_dollars is not None]
        times_to_fill = [e.time_to_fill_ms for e in executions if e.time_to_fill_ms is not None]
        adverse_selections = [e.adverse_selection for e in executions if e.adverse_selection is not None]

        # Calculate metrics
        total_executions = len(executions)
        total_volume = sum(e.filled_quantity * (e.fill_price or 0) for e in executions)

        avg_slippage_bps = np.mean(slippages) if slippages else 0
        median_slippage_bps = np.median(slippages) if slippages else 0

        filled_count = len([e for e in executions if e.status == OrderStatus.FILLED])
        fill_rate = (filled_count / total_executions * 100) if total_executions > 0 else 0

        avg_time_to_fill_ms = np.mean(times_to_fill) if times_to_fill else 0

        partial_fills = len([e for e in executions if e.status == OrderStatus.PARTIALLY_FILLED])
        partial_fill_rate = (partial_fills / total_executions * 100) if total_executions > 0 else 0

        price_improvements = [e for e in executions if e.price_improvement and e.price_improvement > 0]
        price_improvement_rate = (len(price_improvements) / total_executions * 100) if total_executions > 0 else 0
        avg_price_improvement_bps = np.mean([e.price_improvement for e in price_improvements]) if price_improvements else 0

        adverse_count = len([a for a in adverse_selections if a > 0])
        adverse_selection_rate = (adverse_count / len(adverse_selections) * 100) if adverse_selections else 0
        avg_adverse_selection_bps = np.mean([a for a in adverse_selections if a > 0]) if adverse_selections else 0

        # Slippage distribution
        slippage_25th = np.percentile(slippages, 25) if slippages else 0
        slippage_75th = np.percentile(slippages, 75) if slippages else 0
        slippage_95th = np.percentile(slippages, 95) if slippages else 0
        worst_slippage = max(slippages) if slippages else 0
        best_slippage = min(slippages) if slippages else 0

        # Cost analysis
        total_slippage_cost = sum(slippage_dollars) if slippage_dollars else 0

        # Estimate annual drag (assuming 250 trading days, current rate)
        daily_avg_slippage = total_slippage_cost / max(1, len(set(e.order_time.date() for e in executions)))
        estimated_annual_drag = daily_avg_slippage * 250

        return ExecutionQualityMetrics(
            total_executions=total_executions,
            total_volume=total_volume,
            avg_slippage_bps=avg_slippage_bps,
            median_slippage_bps=median_slippage_bps,
            fill_rate=fill_rate,
            avg_time_to_fill_ms=avg_time_to_fill_ms,
            partial_fill_rate=partial_fill_rate,
            price_improvement_rate=price_improvement_rate,
            avg_price_improvement_bps=avg_price_improvement_bps,
            adverse_selection_rate=adverse_selection_rate,
            avg_adverse_selection_bps=avg_adverse_selection_bps,
            slippage_25th_percentile=slippage_25th,
            slippage_75th_percentile=slippage_75th,
            slippage_95th_percentile=slippage_95th,
            worst_slippage_bps=worst_slippage,
            best_slippage_bps=best_slippage,
            total_slippage_cost=total_slippage_cost,
            estimated_annual_drag=estimated_annual_drag
        )

    def _breakdown_by_broker(self, executions: List[ExecutionRecord]) -> Dict[str, ExecutionQualityMetrics]:
        """Break down metrics by broker"""
        brokers = set(e.broker for e in executions)
        return {
            broker: self._calculate_metrics([e for e in executions if e.broker == broker])
            for broker in brokers
        }

    def _breakdown_by_time_of_day(self, executions: List[ExecutionRecord]) -> Dict[str, ExecutionQualityMetrics]:
        """Break down metrics by time of day"""
        def get_time_period(exec_time: datetime) -> str:
            hour = exec_time.hour
            if 9 <= hour < 11:
                return "morning_open"
            elif 11 <= hour < 14:
                return "midday"
            elif 14 <= hour < 16:
                return "afternoon_close"
            else:
                return "extended_hours"

        periods = {}
        for period in ["morning_open", "midday", "afternoon_close", "extended_hours"]:
            period_execs = [e for e in executions if get_time_period(e.order_time) == period]
            if period_execs:
                periods[period] = self._calculate_metrics(period_execs)

        return periods

    def _breakdown_by_order_type(self, executions: List[ExecutionRecord]) -> Dict[str, ExecutionQualityMetrics]:
        """Break down metrics by order type"""
        order_types = set(e.order_type.value for e in executions)
        return {
            order_type: self._calculate_metrics([e for e in executions if e.order_type.value == order_type])
            for order_type in order_types
        }

    def _breakdown_by_symbol(self, executions: List[ExecutionRecord]) -> Dict[str, ExecutionQualityMetrics]:
        """Break down metrics by symbol"""
        symbols = set(e.symbol for e in executions)
        # Only include symbols with 5+ executions
        return {
            symbol: self._calculate_metrics([e for e in executions if e.symbol == symbol])
            for symbol in symbols
            if len([e for e in executions if e.symbol == symbol]) >= 5
        }

    def _generate_recommendations(
        self,
        overall: ExecutionQualityMetrics,
        by_broker: Dict[str, ExecutionQualityMetrics],
        by_time: Dict[str, ExecutionQualityMetrics],
        by_order_type: Dict[str, ExecutionQualityMetrics]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Overall slippage assessment
        if overall.avg_slippage_bps > 10:
            recommendations.append(
                f"⚠️ High average slippage ({overall.avg_slippage_bps:.1f} bps) - "
                f"costing ${overall.total_slippage_cost:.2f} total. "
                f"Estimated annual drag: ${overall.estimated_annual_drag:.2f}"
            )
        elif overall.avg_slippage_bps < 3:
            recommendations.append(
                f"✅ Excellent execution quality ({overall.avg_slippage_bps:.1f} bps average slippage)"
            )

        # Broker comparison
        if len(by_broker) > 1:
            best_broker = min(by_broker.items(), key=lambda x: x[1].avg_slippage_bps)
            worst_broker = max(by_broker.items(), key=lambda x: x[1].avg_slippage_bps)

            if best_broker[1].avg_slippage_bps < worst_broker[1].avg_slippage_bps - 3:
                recommendations.append(
                    f"💡 {best_broker[0]} has significantly better execution "
                    f"({best_broker[1].avg_slippage_bps:.1f} bps vs {worst_broker[1].avg_slippage_bps:.1f} bps). "
                    f"Consider routing more orders to {best_broker[0]}"
                )

        # Time of day optimization
        if by_time:
            best_time = min(by_time.items(), key=lambda x: x[1].avg_slippage_bps)
            worst_time = max(by_time.items(), key=lambda x: x[1].avg_slippage_bps)

            if best_time[1].avg_slippage_bps < worst_time[1].avg_slippage_bps - 5:
                recommendations.append(
                    f"📅 Best execution during {best_time[0]} ({best_time[1].avg_slippage_bps:.1f} bps). "
                    f"Avoid {worst_time[0]} ({worst_time[1].avg_slippage_bps:.1f} bps) when possible"
                )

        # Order type optimization
        if len(by_order_type) > 1:
            market_orders = by_order_type.get('market')
            limit_orders = by_order_type.get('limit')

            if market_orders and limit_orders:
                if market_orders.avg_slippage_bps > limit_orders.avg_slippage_bps + 5:
                    recommendations.append(
                        f"💡 Market orders have {market_orders.avg_slippage_bps - limit_orders.avg_slippage_bps:.1f} bps "
                        f"more slippage than limit orders. Use limit orders when not time-sensitive"
                    )

        # Fill rate
        if overall.fill_rate < 90:
            recommendations.append(
                f"⚠️ Low fill rate ({overall.fill_rate:.1f}%). "
                f"Consider widening limit prices or using marketable limit orders"
            )

        # Adverse selection
        if overall.adverse_selection_rate > 50:
            recommendations.append(
                f"⚠️ High adverse selection rate ({overall.adverse_selection_rate:.1f}%). "
                f"Price moves against you after fills - may indicate poor timing or predictable patterns"
            )

        # Price improvement
        if overall.price_improvement_rate > 20:
            recommendations.append(
                f"✅ Getting price improvement on {overall.price_improvement_rate:.1f}% of orders "
                f"(avg {overall.avg_price_improvement_bps:.1f} bps saved)"
            )

        return recommendations
