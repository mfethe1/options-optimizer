"""
InvestorReport API Routes

Implements GET /api/investor-report endpoint per frontend integration plan.
Returns schema-validated InvestorReport.v1 JSON with institutional-grade analytics.

Performance targets:
- Response time: <500ms
- Schema validation: <50ms
- Phase 4 computation: <200ms per asset
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
import numpy as np
import pandas as pd
import json
import hashlib
import os
import asyncio
from cachetools import TTLCache

from ..analytics.portfolio_metrics import PortfolioAnalytics
from ..agents.swarm.agents.distillation_agent import DistillationAgent
from ..agents.swarm.shared_context import SharedContext
from ..agents.swarm.agent_instrumentation import instrument_distillation_agent
from ..data.position_manager import PositionManager
from .ml_integration_helpers import fetch_historical_prices

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["investor-report"])

# L1 in-memory cache (15 min TTL, max 1024 entries)
L1_CACHE = TTLCache(maxsize=1024, ttl=900)
_cache_locks: Dict[str, asyncio.Lock] = {}

# Redis configuration (optional - graceful degradation if unavailable)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
_redis_client = None


def _cache_key(user_id: str, symbols: Optional[List[str]]) -> str:
    """Generate cache key from user_id and sorted symbols"""
    syms = ",".join(sorted(symbols)) if symbols else ""
    return f"ir_v1:{user_id}:{syms}"


def _etag(doc: Dict[str, Any]) -> str:
    """Generate ETag hash for cache validation"""
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]


async def _get_redis():
    """Get Redis client with lazy initialization and graceful degradation"""
    global _redis_client
    if _redis_client is None:
        try:
            import aioredis
            _redis_client = await aioredis.from_url(
                REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=1.0
            )
            logger.info("✅ Redis L2 cache connected")
        except Exception as e:
            logger.warning(f"⚠️ Redis unavailable, using L1 only: {e}")
            _redis_client = False  # Sentinel to avoid retrying
    return _redis_client if _redis_client is not False else None


async def _compute_and_cache_report(user_id: str, symbol_list: List[str]) -> Dict[str, Any]:
    """
    Compute full InvestorReport and cache in L1/L2.
    This is the expensive path that calls DistillationAgent V2.
    """
    start_time = datetime.now(timezone.utc)

    # Fetch real portfolio positions
    from ..data.position_manager import PositionManager
    
    position_manager = PositionManager()
    all_positions = position_manager.get_all_stock_positions()
    
    # Filter by requested symbols if provided
    if symbol_list:
        target_positions = [p for p in all_positions if p.symbol in symbol_list]
        # If symbols provided but not in portfolio, treat as watchlist/simulation
        if not target_positions:
            logger.info(f"Symbols {symbol_list} not in portfolio, treating as watchlist")
            target_positions = [{'symbol': s, 'quantity': 1, 'entry_price': 0} for s in symbol_list]
    else:
        target_positions = all_positions
        
    # If no positions at all, default to SPY/QQQ for demo
    if not target_positions:
        logger.info("No positions found, using default watchlist")
        target_positions = [
            {'symbol': 'SPY', 'quantity': 1, 'entry_price': 0},
            {'symbol': 'QQQ', 'quantity': 1, 'entry_price': 0}
        ]
        
    positions = []
    total_value = 0
    
    # Fetch real returns for each position
    from ..agents.swarm.mcp_tools import JarvisMCPTools
    
    for pos in target_positions:
        symbol = pos.get('symbol') if isinstance(pos, dict) else pos.symbol
        quantity = pos.get('quantity', 1) if isinstance(pos, dict) else pos.quantity
        
        # Get price history (real data)
        history = JarvisMCPTools.get_price_history(symbol, days=252)
        
        if history.get('success') and history.get('returns'):
            returns = history['returns']
            current_price = history.get('current_price', 0)
            market_value = current_price * quantity
            total_value += market_value
            
            positions.append({
                'symbol': symbol,
                'returns': returns,
                'weight': 0,  # Will calculate after total known
                'market_value': market_value
            })
        else:
            logger.warning(f"Failed to fetch data for {symbol}, skipping")
            
    # Normalize weights
    if total_value > 0:
        for p in positions:
            p['weight'] = p['market_value'] / total_value
    elif positions:
        # Equal weight fallback
        weight = 1.0 / len(positions)
        for p in positions:
            p['weight'] = weight

    # Compute market benchmark returns (SPY) for portfolio metrics and Phase 4
    from ..agents.swarm.mcp_tools import JarvisMCPTools

    market_history = JarvisMCPTools.get_price_history('SPY', days=252)
    market_returns = market_history.get('returns') if market_history.get('success') else None

    # Compute portfolio metrics using MCP tools with real benchmark when available
    metrics_result = JarvisMCPTools.compute_portfolio_metrics(
        positions=positions,
        benchmark_returns=market_returns
    )

    if 'error' in metrics_result:
        raise ValueError(f"Metrics computation failed: {metrics_result['error']}")

    # Compute Phase 4 metrics
    # Fetch options flow metrics
    primary_symbol = positions[0]['symbol'] if positions else 'SPY'
    flow_metrics = JarvisMCPTools.get_options_flow_metrics(primary_symbol)

    pcr = flow_metrics.get('pcr') if flow_metrics.get('success') else None
    iv_skew = flow_metrics.get('iv_skew') if flow_metrics.get('success') else None
    volume_ratio = flow_metrics.get('volume_ratio') if flow_metrics.get('success') else None

    phase4_result = JarvisMCPTools.compute_phase4_metrics(
        asset_returns=positions[0]['returns'] if positions else None,
        market_returns=market_returns,
        pcr=pcr,
        iv_skew=iv_skew,
        volume_ratio=volume_ratio
    )

    # Build position data for DistillationAgent
    position_data = {
        'symbol': symbol_list[0] if symbol_list else 'PORTFOLIO',
        'metrics': metrics_result,
        'phase4': phase4_result
    }

    # Generate narrative using DistillationAgent V2 with instrumentation
    from ..agents.swarm.consensus_engine import ConsensusEngine

    shared_context = SharedContext()
    consensus_engine = ConsensusEngine(shared_context)
    agent = DistillationAgent(
        shared_context=shared_context,
        consensus_engine=consensus_engine
    )

    # Generate report with schema validation and real-time event streaming
    report = await instrument_distillation_agent(
        user_id=user_id,
        agent_instance=agent,
        portfolio_data=position_data
    )

    # Add metadata
    elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

    if 'metadata' not in report:
        report['metadata'] = {}

    report['metadata'].update({
        'schema_version': 'InvestorReport.v1',
        'validated': True,
        'fallback': report.get('metadata', {}).get('fallback', False),
        'response_time_ms': round(elapsed_ms, 2),
        'cached': False
    })

    # Cache in L1
    cache_key = _cache_key(user_id, symbol_list)
    L1_CACHE[cache_key] = report

    # Cache in L2 (Redis) if available
    redis = await _get_redis()
    if redis:
        try:
            await redis.set(cache_key, json.dumps(report), ex=900)  # 15 min TTL
            logger.debug(f"✓ Cached report in L2 (Redis) for {user_id}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to cache in Redis: {e}")

    logger.info(f"✓ InvestorReport computed in {elapsed_ms:.2f}ms for {user_id}")
    return report


class InvestorReportRequest(BaseModel):
    """Request model for investor report generation"""
    user_id: str
    symbols: Optional[List[str]] = None
    fresh: bool = False  # Force fresh computation (bypass cache)


@router.get("/investor-report")
async def get_investor_report(
    user_id: str = Query(..., description="User ID for portfolio lookup"),
    symbols: Optional[str] = Query(None, description="Comma-separated symbols (optional, defaults to user's portfolio)"),
    fresh: bool = Query(False, description="Force fresh computation (bypass cache)"),
    background_tasks: BackgroundTasks = None
):
    """
    Get InvestorReport.v1 JSON for a user's portfolio.

    Caching strategy (stale-while-revalidate):
    - L1 (in-memory): 15min TTL, <10ms lookup
    - L2 (Redis): 15min TTL, <50ms lookup
    - fresh=true: Return cached + schedule async refresh
    - No cache: Return 202 Accepted on first call

    Returns schema-validated JSON with:
    - Executive summary (top picks, key risks, thesis)
    - Risk panel (7 institutional metrics: omega, gh1, pain_index, etc.)
    - Signals (ml_alpha, regime, sentiment, smart_money, alt_data, phase4_tech)
    - Actions (buy/sell/hold recommendations with sizing & risk controls)
    - Sources (authoritative provenance with URLs)
    - Confidence (overall score + drivers)

    Performance: <500ms response time (cached), <50ms schema validation overhead
    """
    try:
        start_time = datetime.now(timezone.utc)

        # Parse symbols
        symbol_list = symbols.split(',') if symbols else ['AAPL', 'MSFT']
        cache_key = _cache_key(user_id, symbol_list)

        # Try L1 cache first (fastest)
        if cache_key in L1_CACHE and not fresh:
            report = L1_CACHE[cache_key]
            report['metadata']['cached'] = True
            report['metadata']['cache_layer'] = 'L1'
            elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            report['metadata']['response_time_ms'] = round(elapsed_ms, 2)
            logger.info(f"✓ L1 cache hit for {user_id} ({elapsed_ms:.2f}ms)")
            return report

        # Try L2 cache (Redis)
        redis = await _get_redis()
        if redis and not fresh:
            try:
                cached_json = await redis.get(cache_key)
                if cached_json:
                    report = json.loads(cached_json)
                    # Populate L1 for next request
                    L1_CACHE[cache_key] = report
                    report['metadata']['cached'] = True
                    report['metadata']['cache_layer'] = 'L2'
                    elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    report['metadata']['response_time_ms'] = round(elapsed_ms, 2)
                    logger.info(f"✓ L2 cache hit for {user_id} ({elapsed_ms:.2f}ms)")
                    return report
            except Exception as e:
                logger.warning(f"⚠️ L2 cache lookup failed: {e}")

        # fresh=true: Return cached if available and schedule async refresh
        if fresh:
            # Check if we have any cached version
            if cache_key in L1_CACHE:
                report = L1_CACHE[cache_key].copy()
                # Fix: Mark as cached and fresh in metadata
                report['metadata']['cached'] = True
                report['metadata']['cache_layer'] = 'L1'
                report['metadata']['fresh'] = True
                report['metadata']['refreshing'] = True
                if background_tasks:
                    background_tasks.add_task(_compute_and_cache_report, user_id, symbol_list)
                logger.info(f"✓ Returning L1 cached report for {user_id} (fresh=true), refresh scheduled")
                return report

            if redis:
                try:
                    cached_json = await redis.get(cache_key)
                    if cached_json:
                        report = json.loads(cached_json)
                        # Fix: Mark as cached and fresh in metadata
                        report['metadata']['cached'] = True
                        report['metadata']['cache_layer'] = 'L2'
                        report['metadata']['fresh'] = True
                        report['metadata']['refreshing'] = True
                        if background_tasks:
                            background_tasks.add_task(_compute_and_cache_report, user_id, symbol_list)
                        logger.info(f"✓ Returning L2 cached report for {user_id} (fresh=true), refresh scheduled")
                        return report
                except Exception as e:
                    logger.warning(f"⚠️ L2 cache lookup failed: {e}")

            # No cache available with fresh=true - compute now
            logger.info(f"⚠️ No cache for {user_id} with fresh=true, computing now")

        # Dog-pile protection: Use lock to prevent concurrent computation
        if cache_key not in _cache_locks:
            _cache_locks[cache_key] = asyncio.Lock()

        lock = _cache_locks[cache_key]

        async with lock:
            # Double-check cache after acquiring lock
            if cache_key in L1_CACHE and not fresh:
                report = L1_CACHE[cache_key]
                report['metadata']['cached'] = True
                logger.info(f"✓ L1 cache hit after lock for {user_id}")
                return report

            # Compute and cache
            report = await _compute_and_cache_report(user_id, symbol_list)

            # Add request metadata
            elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            report['metadata']['response_time_ms'] = round(elapsed_ms, 2)
            report['metadata']['user_id'] = user_id
            report['metadata']['fresh'] = fresh

            return report

    except Exception as e:
        logger.error(f"❌ Error generating investor report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "investor-report-api",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

