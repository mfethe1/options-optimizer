"""
API Routes for Execution Quality Tracking

Track and analyze trade execution quality to minimize slippage.
"""
from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional, Dict, Any
from datetime import date, datetime
import logging

from ..analytics.execution_quality_service import (
    ExecutionQualityService,
    OrderType,
    OrderSide,
    ExecutionRecord
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/execution", tags=["Execution Quality"])

# Initialize service
execution_service = ExecutionQualityService()


@router.post("/record-order")
async def record_order(
    order_id: str = Body(...),
    symbol: str = Body(...),
    order_type: str = Body(...),
    order_side: str = Body(...),
    quantity: int = Body(...),
    expected_price: float = Body(...),
    broker: str = Body(...),
    limit_price: Optional[float] = Body(None),
    bid_ask_spread: Optional[float] = Body(None)
) -> Dict[str, Any]:
    """
    Record a new order when placed.

    Captures initial order details for later execution quality analysis.

    Returns:
        Order record confirmation
    """
    try:
        # Parse enums
        order_type_enum = OrderType(order_type)
        order_side_enum = OrderSide(order_side)

        record = await execution_service.record_order(
            order_id=order_id,
            symbol=symbol,
            order_type=order_type_enum,
            order_side=order_side_enum,
            quantity=quantity,
            expected_price=expected_price,
            broker=broker,
            limit_price=limit_price,
            bid_ask_spread=bid_ask_spread
        )

        return {
            'status': 'recorded',
            'order_id': order_id,
            'symbol': symbol,
            'order_time': record.order_time.isoformat()
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid order type or side: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to record order: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record order: {str(e)}")


@router.post("/record-fill")
async def record_fill(
    order_id: str = Body(...),
    fill_price: float = Body(...),
    filled_quantity: int = Body(...),
    venue: Optional[str] = Body(None),
    partial: bool = Body(False)
) -> Dict[str, Any]:
    """
    Record order fill.

    Updates execution record and calculates slippage metrics.

    Returns:
        Fill record with slippage calculation
    """
    try:
        record = await execution_service.record_fill(
            order_id=order_id,
            fill_price=fill_price,
            filled_quantity=filled_quantity,
            venue=venue,
            partial=partial
        )

        if not record:
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found")

        return {
            'status': 'filled',
            'order_id': order_id,
            'fill_price': fill_price,
            'filled_quantity': filled_quantity,
            'slippage_bps': record.slippage_bps,
            'slippage_dollars': record.slippage_dollars,
            'time_to_fill_ms': record.time_to_fill_ms,
            'price_improvement': record.price_improvement
        }

    except Exception as e:
        logger.error(f"Failed to record fill: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record fill: {str(e)}")


@router.post("/record-price-after-fill")
async def record_price_after_fill(
    order_id: str = Body(...),
    price_1min: float = Body(...),
    price_5min: float = Body(...)
) -> Dict[str, str]:
    """
    Record prices after fill for adverse selection tracking.

    Tracks if price moved against you after execution.
    """
    try:
        await execution_service.record_price_after_fill(
            order_id=order_id,
            price_1min=price_1min,
            price_5min=price_5min
        )

        return {'status': 'recorded'}

    except Exception as e:
        logger.error(f"Failed to record post-fill prices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis")
async def get_execution_analysis(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
    broker: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """
    Get comprehensive execution quality analysis.

    Analyzes slippage, fill quality, and provides actionable recommendations.

    Returns:
        Complete execution quality metrics with breakdowns
    """
    try:
        # Parse dates
        start = datetime.strptime(start_date, '%Y-%m-%d').date() if start_date else None
        end = datetime.strptime(end_date, '%Y-%m-%d').date() if end_date else None

        analysis = await execution_service.get_execution_analysis(
            start_date=start,
            end_date=end,
            symbol=symbol,
            broker=broker
        )

        # Convert to dict
        return {
            'overall_metrics': {
                'total_executions': analysis.overall_metrics.total_executions,
                'total_volume': analysis.overall_metrics.total_volume,
                'avg_slippage_bps': round(analysis.overall_metrics.avg_slippage_bps, 2),
                'median_slippage_bps': round(analysis.overall_metrics.median_slippage_bps, 2),
                'fill_rate': round(analysis.overall_metrics.fill_rate, 1),
                'avg_time_to_fill_ms': round(analysis.overall_metrics.avg_time_to_fill_ms, 0),
                'partial_fill_rate': round(analysis.overall_metrics.partial_fill_rate, 1),
                'price_improvement_rate': round(analysis.overall_metrics.price_improvement_rate, 1),
                'avg_price_improvement_bps': round(analysis.overall_metrics.avg_price_improvement_bps, 2),
                'adverse_selection_rate': round(analysis.overall_metrics.adverse_selection_rate, 1),
                'avg_adverse_selection_bps': round(analysis.overall_metrics.avg_adverse_selection_bps, 2),
                'slippage_25th_percentile': round(analysis.overall_metrics.slippage_25th_percentile, 2),
                'slippage_75th_percentile': round(analysis.overall_metrics.slippage_75th_percentile, 2),
                'slippage_95th_percentile': round(analysis.overall_metrics.slippage_95th_percentile, 2),
                'worst_slippage_bps': round(analysis.overall_metrics.worst_slippage_bps, 2),
                'best_slippage_bps': round(analysis.overall_metrics.best_slippage_bps, 2),
                'total_slippage_cost': round(analysis.overall_metrics.total_slippage_cost, 2),
                'estimated_annual_drag': round(analysis.overall_metrics.estimated_annual_drag, 2)
            },
            'by_broker': {
                broker: {
                    'avg_slippage_bps': round(metrics.avg_slippage_bps, 2),
                    'total_executions': metrics.total_executions,
                    'fill_rate': round(metrics.fill_rate, 1)
                }
                for broker, metrics in analysis.by_broker.items()
            },
            'by_time_of_day': {
                period: {
                    'avg_slippage_bps': round(metrics.avg_slippage_bps, 2),
                    'total_executions': metrics.total_executions
                }
                for period, metrics in analysis.by_time_of_day.items()
            },
            'by_order_type': {
                order_type: {
                    'avg_slippage_bps': round(metrics.avg_slippage_bps, 2),
                    'total_executions': metrics.total_executions
                }
                for order_type, metrics in analysis.by_order_type.items()
            },
            'by_symbol': {
                symbol: {
                    'avg_slippage_bps': round(metrics.avg_slippage_bps, 2),
                    'total_executions': metrics.total_executions
                }
                for symbol, metrics in analysis.by_symbol.items()
            },
            'recommendations': analysis.recommendations
        }

    except Exception as e:
        logger.error(f"Failed to generate execution analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/generate-mock-data")
async def generate_mock_execution_data(
    num_orders: int = Body(50, description="Number of mock orders to generate")
) -> Dict[str, str]:
    """
    Generate mock execution data for testing/demo.

    Creates realistic execution records with varying slippage.
    """
    try:
        import random
        from datetime import timedelta

        symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'NVDA']
        brokers = ['Schwab', 'TD Ameritrade', 'Interactive Brokers', 'Robinhood']
        order_types = [OrderType.MARKET, OrderType.LIMIT]
        order_sides = [OrderSide.BUY, OrderSide.SELL]

        base_time = datetime.now() - timedelta(days=30)

        for i in range(num_orders):
            symbol = random.choice(symbols)
            broker = random.choice(brokers)
            order_type = random.choice(order_types)
            order_side = random.choice(order_sides)

            # Random price and quantity
            expected_price = random.uniform(100, 500)
            quantity = random.choice([1, 5, 10, 20, 50, 100])

            # Varying slippage by broker (realistic scenario)
            broker_slippage_factor = {
                'Schwab': 0.8,
                'TD Ameritrade': 1.0,
                'Interactive Brokers': 0.6,
                'Robinhood': 1.5
            }

            base_slippage_bps = random.gauss(5, 3) * broker_slippage_factor[broker]

            # Convert bps to price
            slippage_pct = base_slippage_bps / 10000
            if order_side == OrderSide.BUY:
                fill_price = expected_price * (1 + slippage_pct)
            else:
                fill_price = expected_price * (1 - slippage_pct)

            # Record order
            order_id = f"ORD-{i:06d}"
            order_time = base_time + timedelta(days=random.randint(0, 30), hours=random.randint(9, 15))

            record = ExecutionRecord(
                order_id=order_id,
                symbol=symbol,
                order_type=order_type,
                order_side=order_side,
                quantity=quantity,
                order_time=order_time,
                expected_price=expected_price,
                broker=broker
            )

            # Simulate fill
            record.fill_time = order_time + timedelta(milliseconds=random.randint(50, 2000))
            record.fill_price = fill_price
            record.filled_quantity = quantity
            record.status = "filled"

            # Calculate metrics
            execution_service._calculate_slippage(record)

            execution_service.executions.append(record)

        return {
            'status': 'generated',
            'num_orders': num_orders,
            'message': f'Generated {num_orders} mock execution records'
        }

    except Exception as e:
        logger.error(f"Failed to generate mock data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint for execution quality service."""
    return {
        "status": "ok",
        "service": "execution_quality",
        "timestamp": datetime.now().isoformat(),
        "capabilities": [
            "Order Tracking",
            "Fill Quality Analysis",
            "Slippage Measurement",
            "Broker Comparison",
            "Time-of-Day Analysis",
            "Adverse Selection Detection",
            "Performance Optimization Recommendations"
        ]
    }
