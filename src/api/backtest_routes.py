"""
API Routes for Backtesting

Historical strategy testing with institutional-grade metrics.
"""
from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional, List, Dict, Any
from datetime import date, datetime, timedelta
import logging
import pandas as pd
import numpy as np

from ..analytics.backtest_engine import (
    BacktestEngine,
    BacktestConfig,
    StrategyType,
    BacktestMetrics
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtest", tags=["Backtesting"])


@router.post("/run")
async def run_backtest(
    symbol: str = Body(..., description="Stock symbol"),
    strategy_type: str = Body(..., description="Strategy type"),
    start_date: str = Body(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Body(..., description="End date (YYYY-MM-DD)"),
    entry_dte_min: int = Body(30, description="Minimum DTE at entry"),
    entry_dte_max: int = Body(45, description="Maximum DTE at entry"),
    profit_target_pct: Optional[float] = Body(50, description="Profit target %"),
    stop_loss_pct: Optional[float] = Body(100, description="Stop loss %"),
    exit_dte: Optional[int] = Body(7, description="Exit DTE"),
    capital_per_trade: float = Body(10000, description="Capital per trade"),
    max_positions: int = Body(5, description="Max concurrent positions"),
    commission_per_contract: float = Body(0.65, description="Commission per contract"),
    slippage_pct: float = Body(0.01, description="Slippage percentage"),
    spread_width: Optional[float] = Body(None, description="Spread width for strategies"),
    iv_rank_min: Optional[float] = Body(None, description="Minimum IV rank filter")
) -> Dict[str, Any]:
    """
    Run a backtest for an options strategy.

    Tests historical performance with realistic costs.

    Returns:
        Complete backtest results with metrics and trade history
    """
    try:
        # Parse dates
        start = datetime.strptime(start_date, '%Y-%m-%d').date()
        end = datetime.strptime(end_date, '%Y-%m-%d').date()

        # Parse strategy type
        try:
            strategy_enum = StrategyType(strategy_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy type: {strategy_type}. "
                       f"Valid types: {[s.value for s in StrategyType]}"
            )

        # Create config
        config = BacktestConfig(
            strategy_type=strategy_enum,
            entry_dte_min=entry_dte_min,
            entry_dte_max=entry_dte_max,
            profit_target_pct=profit_target_pct,
            stop_loss_pct=stop_loss_pct,
            exit_dte=exit_dte,
            capital_per_trade=capital_per_trade,
            max_positions=max_positions,
            commission_per_contract=commission_per_contract,
            slippage_pct=slippage_pct,
            spread_width=spread_width,
            iv_rank_min=iv_rank_min
        )

        # Get historical data (mock for now - would fetch from data provider)
        historical_data = _get_mock_historical_data(symbol, start, end)

        # Run backtest
        engine = BacktestEngine(config)
        metrics = await engine.run_backtest(symbol, start, end, historical_data)

        # Get trade history
        trade_history = [
            {
                'entry_date': trade.entry.date.isoformat(),
                'entry_price': trade.entry.underlying_price,
                'entry_cost': trade.entry.entry_cost,
                'exit_date': trade.exit.date.isoformat() if trade.exit else None,
                'exit_price': trade.exit.underlying_price if trade.exit else None,
                'pnl': trade.exit.pnl if trade.exit else None,
                'pnl_pct': trade.exit.pnl_pct if trade.exit else None,
                'days_held': trade.exit.days_held if trade.exit else None,
                'exit_reason': trade.exit.exit_reason if trade.exit else None,
                'status': trade.status,
                'legs': [
                    {
                        'option_type': leg.option_type,
                        'action': leg.action,
                        'strike': leg.strike,
                        'quantity': leg.quantity,
                        'entry_price': leg.entry_price
                    }
                    for leg in trade.entry.legs
                ]
            }
            for trade in engine.trades
        ]

        return {
            'symbol': symbol,
            'strategy_type': strategy_type,
            'start_date': start_date,
            'end_date': end_date,
            'config': {
                'entry_dte_range': f"{entry_dte_min}-{entry_dte_max}",
                'profit_target_pct': profit_target_pct,
                'stop_loss_pct': stop_loss_pct,
                'exit_dte': exit_dte,
                'capital_per_trade': capital_per_trade,
                'max_positions': max_positions
            },
            'metrics': {
                'total_trades': metrics.total_trades,
                'winning_trades': metrics.winning_trades,
                'losing_trades': metrics.losing_trades,
                'win_rate': round(metrics.win_rate, 2),
                'total_pnl': round(metrics.total_pnl, 2),
                'total_pnl_pct': round(metrics.total_pnl_pct, 2),
                'avg_win': round(metrics.avg_win, 2),
                'avg_loss': round(metrics.avg_loss, 2),
                'profit_factor': round(metrics.profit_factor, 2),
                'max_drawdown': round(metrics.max_drawdown, 2),
                'max_drawdown_pct': round(metrics.max_drawdown_pct, 2),
                'sharpe_ratio': round(metrics.sharpe_ratio, 2),
                'sortino_ratio': round(metrics.sortino_ratio, 2),
                'avg_days_held': round(metrics.avg_days_held, 1),
                'expectancy': round(metrics.expectancy, 2),
                'kelly_criterion': round(metrics.kelly_criterion, 3),
                'calmar_ratio': round(metrics.calmar_ratio, 2),
                'best_trade_pnl': round(metrics.best_trade_pnl, 2),
                'worst_trade_pnl': round(metrics.worst_trade_pnl, 2),
                'consecutive_wins': metrics.consecutive_wins,
                'consecutive_losses': metrics.consecutive_losses
            },
            'trades': trade_history
        }

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


@router.get("/strategies")
async def get_available_strategies() -> Dict[str, Any]:
    """
    Get list of available strategy types for backtesting.

    Returns:
        Available strategies with descriptions
    """
    strategies = {
        'long_call': {
            'name': 'Long Call',
            'description': 'Buy call option',
            'risk': 'Limited (premium paid)',
            'reward': 'Unlimited',
            'best_for': 'Bullish directional plays'
        },
        'long_put': {
            'name': 'Long Put',
            'description': 'Buy put option',
            'risk': 'Limited (premium paid)',
            'reward': 'High (stock to $0)',
            'best_for': 'Bearish directional plays'
        },
        'bull_call_spread': {
            'name': 'Bull Call Spread',
            'description': 'Buy call, sell higher strike call',
            'risk': 'Limited (debit paid)',
            'reward': 'Limited (spread width)',
            'best_for': 'Moderate bullish view, lower cost'
        },
        'bear_put_spread': {
            'name': 'Bear Put Spread',
            'description': 'Buy put, sell lower strike put',
            'risk': 'Limited (debit paid)',
            'reward': 'Limited (spread width)',
            'best_for': 'Moderate bearish view, lower cost'
        },
        'iron_condor': {
            'name': 'Iron Condor',
            'description': 'Sell OTM put spread + OTM call spread',
            'risk': 'Limited (spread width - credit)',
            'reward': 'Limited (credit received)',
            'best_for': 'Range-bound markets, high IV'
        },
        'straddle': {
            'name': 'Long Straddle',
            'description': 'Buy ATM call + ATM put',
            'risk': 'Limited (premium paid)',
            'reward': 'Unlimited',
            'best_for': 'Expected volatility, direction unknown'
        },
        'strangle': {
            'name': 'Long Strangle',
            'description': 'Buy OTM call + OTM put',
            'risk': 'Limited (premium paid)',
            'reward': 'Unlimited',
            'best_for': 'Expected volatility, lower cost than straddle'
        },
        'calendar_spread': {
            'name': 'Calendar Spread',
            'description': 'Sell near-term, buy far-term same strike',
            'risk': 'Limited (debit paid)',
            'reward': 'Limited',
            'best_for': 'Theta decay, stable price'
        },
        'butterfly': {
            'name': 'Butterfly Spread',
            'description': 'Buy 1 ITM, sell 2 ATM, buy 1 OTM',
            'risk': 'Limited (debit paid)',
            'reward': 'Limited',
            'best_for': 'Narrow range, low volatility'
        }
    }

    return {
        'strategies': strategies,
        'count': len(strategies)
    }


@router.post("/compare")
async def compare_strategies(
    symbol: str = Body(...),
    strategies: List[str] = Body(...),
    start_date: str = Body(...),
    end_date: str = Body(...),
    base_config: Dict[str, Any] = Body({})
) -> Dict[str, Any]:
    """
    Compare multiple strategies side-by-side.

    Run backtests for multiple strategies with same parameters.

    Returns:
        Comparative metrics for all strategies
    """
    try:
        results = []

        for strategy_type in strategies:
            # Run backtest for this strategy
            result = await run_backtest(
                symbol=symbol,
                strategy_type=strategy_type,
                start_date=start_date,
                end_date=end_date,
                **base_config
            )
            results.append(result)

        # Sort by total P&L
        results.sort(key=lambda x: x['metrics']['total_pnl'], reverse=True)

        # Create comparison table
        comparison = {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'strategies': [
                {
                    'strategy_type': r['strategy_type'],
                    'total_pnl': r['metrics']['total_pnl'],
                    'win_rate': r['metrics']['win_rate'],
                    'profit_factor': r['metrics']['profit_factor'],
                    'sharpe_ratio': r['metrics']['sharpe_ratio'],
                    'max_drawdown': r['metrics']['max_drawdown'],
                    'total_trades': r['metrics']['total_trades']
                }
                for r in results
            ],
            'best_strategy': results[0]['strategy_type'] if results else None,
            'detailed_results': results
        }

        return comparison

    except Exception as e:
        logger.error(f"Strategy comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint for backtest service."""
    return {
        "status": "ok",
        "service": "backtesting",
        "timestamp": datetime.now().isoformat(),
        "capabilities": [
            "Historical Strategy Testing",
            "Multi-Strategy Comparison",
            "Realistic Transaction Costs",
            "Greeks Tracking",
            "Advanced Performance Metrics",
            "Risk/Reward Analysis"
        ]
    }


def _get_mock_historical_data(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Get mock historical data for backtesting.

    In production, this would fetch from data provider.
    """
    # Generate random price data
    dates = pd.date_range(start_date, end_date, freq='B')
    n = len(dates)

    # Simulate stock price with drift and volatility
    np.random.seed(42)  # For reproducibility
    initial_price = 150
    returns = np.random.normal(0.0005, 0.015, n)  # Slight upward drift
    prices = initial_price * np.exp(np.cumsum(returns))

    # Create DataFrame
    df = pd.DataFrame({
        'date': dates.date,
        'open': prices * (1 + np.random.normal(0, 0.005, n)),
        'high': prices * (1 + np.random.uniform(0, 0.02, n)),
        'low': prices * (1 - np.random.uniform(0, 0.02, n)),
        'close': prices,
        'volume': np.random.randint(10000000, 50000000, n),
        'iv_rank': np.random.uniform(20, 80, n)  # IV rank 20-80
    })

    return df
