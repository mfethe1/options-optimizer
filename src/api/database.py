"""
Database connection and operations
"""
import asyncpg
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import os
import json

logger = logging.getLogger(__name__)


class Database:
    """Database connection and operations."""
    
    def __init__(self, connection_pool: asyncpg.Pool):
        """Initialize database with connection pool."""
        self.pool = connection_pool
    
    async def create_position(self, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new position."""
        async with self.pool.acquire() as conn:
            # Insert position
            position_id = await conn.fetchval(
                """
                INSERT INTO positions (
                    user_id, symbol, strategy_type, entry_date,
                    expiration_date, total_premium, status
                )
                VALUES ($1, $2, $3, $4, $5, $6, 'open')
                RETURNING id
                """,
                position_data['user_id'],
                position_data['symbol'],
                position_data['strategy_type'],
                position_data['entry_date'],
                position_data['expiration_date'],
                position_data['total_premium']
            )
            
            # Insert legs
            for leg in position_data['legs']:
                await conn.execute(
                    """
                    INSERT INTO position_legs (
                        position_id, option_type, strike, quantity,
                        is_short, entry_price, multiplier
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    position_id,
                    leg['option_type'],
                    leg['strike'],
                    leg['quantity'],
                    leg.get('is_short', False),
                    leg['entry_price'],
                    leg.get('multiplier', 100)
                )
            
            # Fetch and return created position
            return await self.get_position(position_id)
    
    async def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Get a position by ID."""
        async with self.pool.acquire() as conn:
            # Get position
            position = await conn.fetchrow(
                "SELECT * FROM positions WHERE id = $1",
                position_id
            )
            
            if not position:
                return None
            
            # Get legs
            legs = await conn.fetch(
                "SELECT * FROM position_legs WHERE position_id = $1",
                position_id
            )
            
            return {
                **dict(position),
                'legs': [dict(leg) for leg in legs]
            }
    
    async def get_positions(
        self,
        user_id: str,
        status: str = "open"
    ) -> List[Dict[str, Any]]:
        """Get all positions for a user."""
        async with self.pool.acquire() as conn:
            positions = await conn.fetch(
                "SELECT * FROM positions WHERE user_id = $1 AND status = $2",
                user_id, status
            )
            
            result = []
            for position in positions:
                legs = await conn.fetch(
                    "SELECT * FROM position_legs WHERE position_id = $1",
                    position['id']
                )
                
                result.append({
                    **dict(position),
                    'legs': [dict(leg) for leg in legs]
                })
            
            return result
    
    async def update_position(
        self,
        position_id: str,
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update a position."""
        async with self.pool.acquire() as conn:
            # Build update query
            fields = []
            values = []
            idx = 1
            
            for key, value in update_data.items():
                if value is not None:
                    fields.append(f"{key} = ${idx}")
                    values.append(value)
                    idx += 1
            
            if fields:
                query = f"""
                    UPDATE positions
                    SET {', '.join(fields)}
                    WHERE id = ${idx}
                """
                values.append(position_id)
                
                await conn.execute(query, *values)
            
            return await self.get_position(position_id)
    
    async def delete_position(self, position_id: str):
        """Delete a position."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM positions WHERE id = $1",
                position_id
            )
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get latest market data for a symbol."""
        async with self.pool.acquire() as conn:
            data = await conn.fetchrow(
                """
                SELECT * FROM market_data
                WHERE symbol = $1
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                symbol
            )
            
            if data:
                return dict(data)
            
            # Return default data if not found
            return {
                'symbol': symbol,
                'underlying_price': 100.0,
                'iv': 0.30,
                'historical_iv': 0.28,
                'iv_rank': 50.0,
                'volume': 1000000,
                'avg_volume': 1000000,
                'put_call_ratio': 1.0,
                'days_to_earnings': 999,
                'sector': 'Unknown',
                'risk_free_rate': 0.05,
                'time_to_expiry': 0.25
            }
    
    async def save_greeks(
        self,
        position_id: str,
        greeks: Dict[str, float]
    ):
        """Save Greeks calculation."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO greeks_history (
                    position_id, delta, gamma, theta, vega, rho
                )
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                position_id,
                greeks.get('delta', 0),
                greeks.get('gamma', 0),
                greeks.get('theta', 0),
                greeks.get('vega', 0),
                greeks.get('rho', 0)
            )
    
    async def save_ev(self, position_id: str, ev_result):
        """Save EV calculation."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ev_calculations (
                    position_id, expected_value, expected_return_pct,
                    probability_profit, confidence_interval_lower,
                    confidence_interval_upper, method_breakdown
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                position_id,
                ev_result.expected_value,
                ev_result.expected_return_pct,
                ev_result.probability_profit,
                ev_result.confidence_interval[0],
                ev_result.confidence_interval[1],
                json.dumps(ev_result.method_breakdown)
            )
    
    async def save_report(self, user_id: str, report: Dict[str, Any]):
        """Save analysis report."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO analysis_reports (
                    user_id, report_type, executive_summary,
                    market_overview, portfolio_analysis, risk_assessment,
                    recommendations, action_items, risk_score
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                user_id,
                report.get('report_type', 'daily'),
                report.get('executive_summary', ''),
                report.get('market_overview', ''),
                report.get('portfolio_analysis', ''),
                report.get('risk_assessment', ''),
                json.dumps(report.get('recommendations', [])),
                json.dumps(report.get('action_items', [])),
                report.get('risk_score', 0)
            )
    
    async def get_reports(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent reports for a user."""
        async with self.pool.acquire() as conn:
            reports = await conn.fetch(
                """
                SELECT * FROM analysis_reports
                WHERE user_id = $1
                ORDER BY generated_at DESC
                LIMIT $2
                """,
                user_id, limit
            )
            
            return [dict(report) for report in reports]
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences."""
        async with self.pool.acquire() as conn:
            prefs = await conn.fetchrow(
                "SELECT * FROM user_preferences WHERE user_id = $1",
                user_id
            )
            
            if prefs:
                return dict(prefs)
            
            # Return defaults
            return {
                'risk_tolerance': 'moderate',
                'notification_preferences': {},
                'report_frequency': 'daily',
                'preferred_strategies': []
            }
    
    async def get_active_users(self) -> List[Dict[str, Any]]:
        """Get all active users."""
        async with self.pool.acquire() as conn:
            users = await conn.fetch(
                "SELECT * FROM users WHERE is_active = TRUE"
            )
            
            return [dict(user) for user in users]


async def get_db_pool():
    """Create database connection pool."""
    database_url = os.getenv(
        'DATABASE_URL',
        'postgresql://postgres:postgres@localhost:5432/options_analysis'
    )
    
    pool = await asyncpg.create_pool(
        database_url,
        min_size=5,
        max_size=20
    )
    
    return pool


async def get_db():
    """Dependency for getting database connection."""
    pool = await get_db_pool()
    db = Database(pool)
    try:
        yield db
    finally:
        await pool.close()

