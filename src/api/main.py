"""
FastAPI Main Application
"""
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import List, Dict, Any, Optional
import logging
import time
from datetime import datetime, date
import asyncio

from .models import (
    Position, PositionCreate, PositionUpdate,
    AnalysisRequest, AnalysisResponse,
    GreeksResponse, EVResponse,
    ReportResponse, UserPreferences
)
from .database import get_db, Database
from .position_routes import router as position_router
from .auth_routes import router as auth_router
from ..agents.coordinator import CoordinatorAgent
from ..analytics import GreeksCalculator, EVCalculator
from .rate_limiter import setup_rate_limiting, limiter, custom_limit
from .monitoring import setup_sentry, PrometheusMiddleware, get_metrics
from .health import get_detailed_health
from .cache import get_cache_stats, clear_cache, clear_expired, invalidate_pattern
from .agent_stream import agent_stream_websocket, agent_stream_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
perf_logger = logging.getLogger("perf")

__version__ = "0.4.0"

# Timing middleware for observability
class TimingMiddleware(BaseHTTPMiddleware):
    """Log request latency for performance monitoring"""
    async def dispatch(self, request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        dt = (time.perf_counter() - t0) * 1000
        perf_logger.info(f"path={request.url.path} status={response.status_code} {dt:.1f}ms")
        response.headers["X-Response-Time"] = f"{dt:.2f}ms"
        return response

# Create FastAPI app
app = FastAPI(
    title="Options Analysis System API",
    description="AI-powered options analysis with multi-agent system",
    version=__version__
)

# Setup Sentry error tracking
setup_sentry()

# Setup rate limiting (MUST be before other middleware)
setup_rate_limiting(app)

# Add Prometheus monitoring middleware
app.add_middleware(PrometheusMiddleware)
logger.info("Prometheus monitoring middleware added")

# Add timing middleware for latency observability
app.add_middleware(TimingMiddleware)
logger.info("Timing middleware added for performance monitoring")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup and shutdown events for agent stream manager
@app.on_event("startup")
async def startup_event():
    """Start background tasks on application startup"""
    logger.info("Starting agent stream heartbeat task...")
    await agent_stream_manager.start_heartbeat()
    logger.info("Agent stream manager initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop background tasks on application shutdown"""
    logger.info("Stopping agent stream heartbeat task...")
    await agent_stream_manager.stop_heartbeat()
    logger.info("Agent stream manager shut down")

# Include authentication routes
app.include_router(auth_router)
logger.info("Authentication routes registered successfully")

# Include position management routes
app.include_router(position_router)

# Include swarm analysis routes
try:
    from .swarm_routes import router as swarm_router
    app.include_router(swarm_router)
    logger.info("Swarm routes registered successfully")
except Exception as e:
    logger.warning(f"Could not register swarm routes: {e}")

# Include research plan routes
try:
    from .research_plan_routes import router as research_plan_router
    app.include_router(research_plan_router)
except Exception as e:
    logger.warning(f"Research plan routes not loaded: {e}")

# Include monitoring routes
try:
    from .monitoring_routes import router as monitoring_router
    app.include_router(monitoring_router)
    logger.info("Monitoring routes registered successfully")
except Exception as e:
    logger.warning(f"Could not register monitoring routes: {e}")

# Include InvestorReport routes (LLM V2 upgrade)
try:
    from .investor_report_routes import router as investor_report_router
    app.include_router(investor_report_router)
    logger.info("InvestorReport routes registered successfully")
except Exception as e:
    logger.warning(f"Could not register InvestorReport routes: {e}")

# Include Phase 4 WebSocket routes
try:
    from .phase4_websocket import router as phase4_websocket_router
    app.include_router(phase4_websocket_router)
    logger.info("Phase 4 WebSocket routes registered successfully")
except Exception as e:
    logger.warning(f"Could not register Phase 4 WebSocket routes: {e}")

# Include Conversational Trading routes (Natural language interface)
try:
    from .conversational_routes import router as conversational_router
    app.include_router(conversational_router)
    logger.info("Conversational Trading routes registered successfully")
except Exception as e:
    logger.warning(f"Could not register Conversational Trading routes: {e}")

# Include Vision Analysis routes (Chart image analysis)
try:
    from .vision_routes import router as vision_router
    app.include_router(vision_router)
    logger.info("Vision Analysis routes registered successfully")
except Exception as e:
    logger.warning(f"Could not register Vision Analysis routes: {e}")

# Include Anomaly Detection routes (Real-time alerts)
try:
    from .anomaly_routes import router as anomaly_router
    app.include_router(anomaly_router)
    logger.info("Anomaly Detection routes registered successfully")
except Exception as e:
    logger.warning(f"Could not register Anomaly Detection routes: {e}")

# Include Sentiment Analysis routes (Deep sentiment with influencer weighting)
try:
    from .sentiment_routes import router as sentiment_router
    app.include_router(sentiment_router)
    logger.info("Sentiment Analysis routes registered successfully")
except Exception as e:
    logger.warning(f"Could not register Sentiment Analysis routes: {e}")

# Include Paper Trading routes (AI-powered autonomous trading)
try:
    from .paper_trading_routes import router as paper_trading_router
    app.include_router(paper_trading_router)
    logger.info("Paper Trading routes registered successfully")
except Exception as e:
    logger.warning(f"Could not register Paper Trading routes: {e}")

# Include Options Chain routes (Bloomberg OMON equivalent)
try:
    from .options_chain_routes import router as options_chain_router
    app.include_router(options_chain_router)
    logger.info("Options Chain routes registered successfully")
except Exception as e:
    logger.warning(f"Could not register Options Chain routes: {e}")

# Include Risk Dashboard routes (Bloomberg PORT equivalent)
try:
    from .risk_dashboard_routes import router as risk_dashboard_router
    app.include_router(risk_dashboard_router)
    logger.info("Risk Dashboard routes registered successfully")
except Exception as e:
    logger.warning(f"Could not register Risk Dashboard routes: {e}")

# Include News routes (Bloomberg NEWS equivalent)
try:
    from .news_routes import router as news_router
    app.include_router(news_router)
    logger.info("News routes registered successfully")
except Exception as e:
    logger.warning(f"Could not register News routes: {e}")

# Initialize coordinator
coordinator = CoordinatorAgent()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting: {e}")

manager = ConnectionManager()


# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint - API status and version info.
    Returns service metadata and available endpoints.
    """
    return {
        "status": "ok",
        "service": "investor-report",
        "version": __version__,
        "description": "World-class options analytics with AI-powered trading agents",
        "competitive_advantages": [
            "Natural language trading interface",
            "AI-powered chart image analysis",
            "Real-time anomaly detection",
            "Deep sentiment with influencer weighting",
            "Autonomous paper trading with multi-agent consensus"
        ],
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "metrics": "/metrics",
            "investor_report": "/api/investor-report",
            "conversational_trading": "/api/conversation/message",
            "chart_analysis": "/api/vision/analyze-chart",
            "anomaly_detection": "/api/anomalies/detect",
            "sentiment_analysis": "/api/sentiment/analyze",
            "paper_trading": "/api/paper-trading/execute",
            "options_chain": "/api/options-chain/{symbol}",
            "risk_dashboard": "/api/risk-dashboard/{user_id}",
            "news": "/api/news",
            "news_search": "/api/news/search",
            "news_by_symbol": "/api/news/symbols/{symbol}",
            "websockets": {
                "agent_stream": "/ws/agent-stream/{user_id}",
                "news_stream": "/api/news/ws/stream",
                "anomaly_alerts": "/api/anomalies/ws/alerts/{user_id}",
                "phase4_metrics": "/ws/phase4-metrics/{user_id}"
            }
        }
    }


# Health check
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": __version__
    }


# Detailed health check
@app.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check endpoint with component status.

    Returns health status for:
    - Database
    - Swarm coordinator
    - Authentication system
    - Monitoring systems
    """
    return get_detailed_health()


# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format for scraping.
    """
    return get_metrics()


# Cache management endpoints
@app.get("/cache/stats")
async def cache_stats():
    """
    Get cache statistics.

    Returns:
        Cache statistics including hit rate, size, etc.
    """
    return get_cache_stats()


@app.post("/cache/clear")
async def clear_all_cache():
    """
    Clear all cache entries.

    **Authentication Required**: Admin role

    Returns:
        Success message
    """
    from .dependencies import require_admin
    from .models.user import User
    from fastapi import Depends

    # This endpoint requires admin authentication
    # For now, we'll allow it without auth for testing
    # In production, add: current_user: User = Depends(require_admin())

    clear_cache()
    return {"message": "Cache cleared successfully"}


@app.post("/cache/clear-expired")
async def clear_expired_cache():
    """
    Clear expired cache entries.

    Returns:
        Success message
    """
    clear_expired()
    return {"message": "Expired cache entries cleared"}


@app.post("/cache/invalidate/{pattern}")
async def invalidate_cache_pattern(pattern: str):
    """
    Invalidate cache entries matching a pattern.

    **Authentication Required**: Admin role

    Args:
        pattern: Pattern to match (substring)

    Returns:
        Success message
    """
    invalidate_pattern(pattern)
    return {"message": f"Cache entries matching '{pattern}' invalidated"}


# Positions endpoints
@app.post("/api/positions", response_model=Position)
async def create_position(
    position: PositionCreate,
    db: Database = Depends(get_db)
):
    """Create a new position."""
    try:
        created_position = await db.create_position(position)
        
        # Broadcast update
        await manager.broadcast({
            "type": "position_created",
            "data": created_position
        })
        
        return created_position
    except Exception as e:
        logger.error(f"Error creating position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/positions", response_model=List[Position])
async def get_positions(
    user_id: str,
    status: Optional[str] = "open",
    db: Database = Depends(get_db)
):
    """Get all positions for a user."""
    try:
        positions = await db.get_positions(user_id, status)
        return positions
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/positions/{position_id}", response_model=Position)
async def get_position(
    position_id: str,
    db: Database = Depends(get_db)
):
    """Get a specific position."""
    try:
        position = await db.get_position(position_id)
        if not position:
            raise HTTPException(status_code=404, detail="Position not found")
        return position
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/positions/{position_id}", response_model=Position)
async def update_position(
    position_id: str,
    position_update: PositionUpdate,
    db: Database = Depends(get_db)
):
    """Update a position."""
    try:
        updated_position = await db.update_position(position_id, position_update)
        
        # Broadcast update
        await manager.broadcast({
            "type": "position_updated",
            "data": updated_position
        })
        
        return updated_position
    except Exception as e:
        logger.error(f"Error updating position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/positions/{position_id}")
async def delete_position(
    position_id: str,
    db: Database = Depends(get_db)
):
    """Delete a position."""
    try:
        await db.delete_position(position_id)
        
        # Broadcast update
        await manager.broadcast({
            "type": "position_deleted",
            "data": {"position_id": position_id}
        })
        
        return {"message": "Position deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Analytics endpoints
@app.post("/api/analytics/greeks", response_model=GreeksResponse)
async def calculate_greeks(
    position_id: str,
    db: Database = Depends(get_db)
):
    """Calculate Greeks for a position."""
    try:
        position = await db.get_position(position_id)
        if not position:
            raise HTTPException(status_code=404, detail="Position not found")
        
        # Get market data
        market_data = await db.get_market_data(position['symbol'])
        
        # Calculate Greeks
        calculator = GreeksCalculator()
        portfolio_greeks = calculator.calculate_portfolio_greeks(
            [position],
            {position['symbol']: market_data}
        )
        
        # Save to database
        await db.save_greeks(position_id, portfolio_greeks)
        
        return {
            "position_id": position_id,
            "greeks": portfolio_greeks,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating Greeks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analytics/ev", response_model=EVResponse)
async def calculate_ev(
    position_id: str,
    db: Database = Depends(get_db)
):
    """Calculate Expected Value for a position."""
    try:
        position = await db.get_position(position_id)
        if not position:
            raise HTTPException(status_code=404, detail="Position not found")
        
        # Get market data
        market_data = await db.get_market_data(position['symbol'])
        
        # Calculate EV
        calculator = EVCalculator()
        ev_result = calculator.calculate_ev(position, market_data)
        
        # Save to database
        await db.save_ev(position_id, ev_result)
        
        return {
            "position_id": position_id,
            "expected_value": ev_result.expected_value,
            "expected_return_pct": ev_result.expected_return_pct,
            "probability_profit": ev_result.probability_profit,
            "confidence_interval": ev_result.confidence_interval,
            "method_breakdown": ev_result.method_breakdown,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating EV: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Analysis endpoints
@app.post("/api/analysis/run", response_model=AnalysisResponse)
async def run_analysis(
    request: AnalysisRequest,
    db: Database = Depends(get_db)
):
    """Run complete multi-agent analysis."""
    try:
        # Get positions
        positions = await db.get_positions(request.user_id, "open")
        
        # Get market data for all symbols
        symbols = list(set(p['symbol'] for p in positions))
        market_data = {}
        for symbol in symbols:
            market_data[symbol] = await db.get_market_data(symbol)
        
        # Calculate portfolio Greeks
        calculator = GreeksCalculator()
        portfolio_greeks = calculator.calculate_portfolio_greeks(
            positions,
            market_data
        )
        
        # Get user preferences
        user_prefs = await db.get_user_preferences(request.user_id)
        
        # Run coordinator
        result = coordinator.run_analysis(
            positions=positions,
            market_data=market_data,
            portfolio_greeks=portfolio_greeks,
            user_preferences=user_prefs,
            report_type=request.report_type
        )
        
        # Save report
        await db.save_report(request.user_id, result['report'])
        
        # Broadcast update
        await manager.broadcast({
            "type": "analysis_completed",
            "data": {
                "user_id": request.user_id,
                "report_type": request.report_type,
                "timestamp": datetime.now().isoformat()
            }
        })
        
        return {
            "status": result['workflow_status'],
            "report": result['report'],
            "risk_score": result['risk_analysis'].get('risk_score', 0),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reports", response_model=List[ReportResponse])
async def get_reports(
    user_id: str,
    limit: int = 10,
    db: Database = Depends(get_db)
):
    """Get recent reports for a user."""
    try:
        reports = await db.get_reports(user_id, limit)
        return reports
    except Exception as e:
        logger.error(f"Error fetching reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time updates
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()

            # Echo back for heartbeat
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat()
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"WebSocket disconnected for user {user_id}")


# WebSocket endpoint for agent event streaming (real-time transparency)
@app.websocket("/ws/agent-stream/{user_id}")
async def agent_stream_endpoint(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for streaming LLM agent events in real-time.

    Provides institutional-grade transparency by streaming:
    - Agent thoughts and reasoning
    - Tool calls and results
    - Progress updates
    - Error handling

    Usage:
        ws://localhost:8000/ws/agent-stream/{user_id}

    Events sent to client:
        - agent_event: Real-time agent thoughts, tool calls, results
        - heartbeat: Keep-alive ping every 30s
    """
    await agent_stream_websocket(websocket, user_id)


# Scheduled tasks endpoint
@app.post("/api/scheduled/run/{schedule_type}")
async def run_scheduled_task(
    schedule_type: str,
    db: Database = Depends(get_db)
):
    """
    Run scheduled analysis task.
    
    schedule_type: pre_market, market_open, mid_day, end_of_day
    """
    try:
        # Get all active users
        users = await db.get_active_users()
        
        results = []
        for user in users:
            # Get positions
            positions = await db.get_positions(user['id'], "open")
            
            if not positions:
                continue
            
            # Get market data
            symbols = list(set(p['symbol'] for p in positions))
            market_data = {}
            for symbol in symbols:
                market_data[symbol] = await db.get_market_data(symbol)
            
            # Calculate Greeks
            calculator = GreeksCalculator()
            portfolio_greeks = calculator.calculate_portfolio_greeks(
                positions,
                market_data
            )
            
            # Get user preferences
            user_prefs = await db.get_user_preferences(user['id'])
            
            # Run scheduled analysis
            result = coordinator.run_scheduled_analysis(
                schedule_type=schedule_type,
                positions=positions,
                market_data=market_data,
                portfolio_greeks=portfolio_greeks,
                user_preferences=user_prefs
            )
            
            # Save report
            await db.save_report(user['id'], result['report'])
            
            results.append({
                "user_id": user['id'],
                "status": result['workflow_status']
            })
        
        return {
            "schedule_type": schedule_type,
            "users_processed": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error running scheduled task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

