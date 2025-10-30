"""
Enhanced FastAPI Application with Position Management and Real-time Data
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from ..agents.coordinator import CoordinatorAgent
from ..agents.sentiment_research_agent import SentimentResearchAgent
from ..agents.multi_model_discussion import MultiModelDiscussion
from ..analytics import GreeksCalculator, EVCalculator
from ..analytics.recommendation_engine import RecommendationEngine
from ..data.position_manager import PositionManager
from ..data.market_data_fetcher import MarketDataFetcher
from ..services.research_service import ResearchService
from ..services.earnings_service import EarningsService
from ..services.scheduler import DataScheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Options Analysis System API",
    description="AI-powered options analysis with real-time data and sentiment research",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
coordinator = CoordinatorAgent()
sentiment_agent = SentimentResearchAgent()
multi_model_discussion = MultiModelDiscussion()
position_manager = PositionManager()
market_data = MarketDataFetcher()
research_service = ResearchService()
earnings_service = EarningsService()
scheduler = DataScheduler(research_service, earnings_service, position_manager)

# In-memory storage for testing
reports_store: List[Dict[str, Any]] = []


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting Options Analysis API")
    logger.info(f"Loaded {len(position_manager.get_all_stock_positions())} stocks, "
                f"{len(position_manager.get_all_option_positions())} options")

    # Start background scheduler
    try:
        scheduler.start()
        logger.info("Background scheduler started")
    except Exception as e:
        logger.error(f"Error starting scheduler: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Options Analysis API")
    try:
        scheduler.stop()
        logger.info("Background scheduler stopped")
    except Exception as e:
        logger.error(f"Error stopping scheduler: {e}")


# Pydantic models for requests
class StockPositionCreate(BaseModel):
    symbol: str
    quantity: int
    entry_price: float
    entry_date: Optional[str] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    notes: Optional[str] = None


class OptionPositionCreate(BaseModel):
    symbol: str
    option_type: str  # 'call' or 'put'
    strike: float
    expiration_date: str
    quantity: int
    premium_paid: float
    entry_date: Optional[str] = None
    target_price: Optional[float] = None
    target_profit_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    notes: Optional[str] = None


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    summary = position_manager.get_portfolio_summary()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "portfolio": summary
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Options Analysis System API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "features": [
            "Position Management (Stocks & Options)",
            "Real-time Market Data",
            "Sentiment Analysis & Research",
            "Multi-Agent AI Analysis",
            "Expected Value Calculations",
            "Greeks & Risk Metrics"
        ]
    }


# ============================================================================
# POSITION MANAGEMENT ENDPOINTS
# ============================================================================

@app.post("/api/positions/stock")
async def add_stock_position(position: StockPositionCreate):
    """Add a new stock position"""
    try:
        position_id = position_manager.add_stock_position(
            symbol=position.symbol,
            quantity=position.quantity,
            entry_price=position.entry_price,
            entry_date=position.entry_date,
            target_price=position.target_price,
            stop_loss=position.stop_loss,
            notes=position.notes
        )

        return {
            "status": "success",
            "position_id": position_id,
            "message": f"Added stock position for {position.symbol}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error adding stock position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/positions/option")
async def add_option_position(position: OptionPositionCreate):
    """Add a new option position"""
    try:
        position_id = position_manager.add_option_position(
            symbol=position.symbol,
            option_type=position.option_type,
            strike=position.strike,
            expiration_date=position.expiration_date,
            quantity=position.quantity,
            premium_paid=position.premium_paid,
            entry_date=position.entry_date,
            target_price=position.target_price,
            target_profit_pct=position.target_profit_pct,
            stop_loss_pct=position.stop_loss_pct,
            notes=position.notes
        )

        return {
            "status": "success",
            "position_id": position_id,
            "message": f"Added option position for {position.symbol}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error adding option position: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/positions/test")
async def test_positions():
    """Test endpoint to debug positions"""
    try:
        stock_positions = position_manager.get_all_stock_positions()
        option_positions = position_manager.get_all_option_positions()

        return {
            "stock_count": len(stock_positions),
            "option_count": len(option_positions),
            "stocks": [{"symbol": s.symbol, "quantity": s.quantity} for s in stock_positions],
            "options": [{"symbol": o.symbol, "strike": o.strike} for o in option_positions]
        }
    except Exception as e:
        logger.exception(f"Test error: {e}")
        return {"error": str(e), "type": type(e).__name__}



@app.get("/api/positions/simple")
async def get_positions_simple():
    """Get all positions without real-time market data (for testing)"""
    try:
        stock_positions = position_manager.get_all_stock_positions()
        option_positions = position_manager.get_all_option_positions()

        return {
            "stocks": [stock.to_dict() for stock in stock_positions],
            "options": [option.to_dict() for option in option_positions],
            "summary": position_manager.get_portfolio_summary(),
            "timestamp": datetime.now().isoformat(),
            "note": "This endpoint returns positions without real-time market data"
        }
    except Exception as e:
        logger.exception(f"Error getting simple positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/positions")
async def get_all_positions():
    """Get all positions with real-time metrics"""
    try:
        # Get all positions
        stock_positions = position_manager.get_all_stock_positions()
        option_positions = position_manager.get_all_option_positions()

        # Enhance stock positions with real-time data
        enhanced_stocks = []
        for stock in stock_positions:
            try:
                # Get market data
                market_data_result = market_data.get_stock_price(stock.symbol)
                if market_data_result:
                    # Calculate metrics
                    stock.calculate_metrics(market_data_result)
                enhanced_stocks.append(stock.to_dict())
            except Exception as e:
                logger.exception(f"Error enhancing stock {stock.symbol}: {e}")
                # Still append the position even if enhancement fails
                enhanced_stocks.append(stock.to_dict())

        # Enhance option positions with real-time data
        enhanced_options = []
        for option in option_positions:
            try:
                # Get market data for underlying
                market_data_result = market_data.get_stock_price(option.symbol)

                # Get option data
                option_data = market_data.get_option_price(
                    option.symbol,
                    option.option_type,
                    option.strike,
                    option.expiration_date
                )

                if market_data_result and option_data:
                    # Calculate metrics
                    option.calculate_metrics(market_data_result, option_data)

                enhanced_options.append(option.to_dict())
            except Exception as e:
                logger.exception(f"Error enhancing option {option.symbol}: {e}")
                # Still append the position even if enhancement fails
                enhanced_options.append(option.to_dict())

        return {
            "stocks": enhanced_stocks,
            "options": enhanced_options,
            "summary": position_manager.get_portfolio_summary(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.exception(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/positions/enhanced")
async def get_enhanced_positions():
    """Get all positions with real-time metrics AND sentiment data"""
    try:
        # Get all positions with metrics
        stock_positions = position_manager.get_all_stock_positions()
        option_positions = position_manager.get_all_option_positions()

        # Get unique symbols
        symbols = set()
        for stock in stock_positions:
            symbols.add(stock.symbol)
        for option in option_positions:
            symbols.add(option.symbol)

        # Get sentiment for all symbols
        sentiment_data = {}
        for symbol in symbols:
            try:
                # Create state for sentiment agent
                state = {
                    'positions': [{'symbol': symbol}],
                    'sentiment_research': {},
                    'errors': []
                }
                # Process sentiment
                result = sentiment_agent.process(state)
                if 'sentiment_research' in result and symbol in result['sentiment_research']:
                    sentiment_data[symbol] = result['sentiment_research'][symbol]
            except Exception as e:
                logger.warning(f"Error getting sentiment for {symbol}: {e}")
                sentiment_data[symbol] = {
                    'sentiment_score': 0,
                    'sentiment_label': 'neutral',
                    'news_summary': 'No data available'
                }

        # Enhance stock positions
        enhanced_stocks = []
        for stock in stock_positions:
            try:
                market_data_result = market_data.get_stock_price(stock.symbol)
                if market_data_result:
                    stock.calculate_metrics(market_data_result)

                stock_dict = stock.to_dict()
                stock_dict['sentiment'] = sentiment_data.get(stock.symbol, {})
                enhanced_stocks.append(stock_dict)
            except Exception as e:
                logger.warning(f"Error enhancing stock {stock.symbol}: {e}")
                stock_dict = stock.to_dict()
                stock_dict['sentiment'] = {}
                enhanced_stocks.append(stock_dict)

        # Enhance option positions
        enhanced_options = []
        for option in option_positions:
            try:
                market_data_result = market_data.get_stock_price(option.symbol)
                option_data = market_data.get_option_price(
                    option.symbol,
                    option.option_type,
                    option.strike,
                    option.expiration_date
                )

                if market_data_result and option_data:
                    option.calculate_metrics(market_data_result, option_data)

                option_dict = option.to_dict()
                option_dict['sentiment'] = sentiment_data.get(option.symbol, {})
                enhanced_options.append(option_dict)
            except Exception as e:
                logger.warning(f"Error enhancing option {option.symbol}: {e}")
                option_dict = option.to_dict()
                option_dict['sentiment'] = {}
                enhanced_options.append(option_dict)

        return {
            "stocks": enhanced_stocks,
            "options": enhanced_options,
            "summary": position_manager.get_portfolio_summary(),
            "sentiment_summary": sentiment_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting enhanced positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/positions/{symbol}")
async def get_positions_by_symbol(symbol: str):
    """Get all positions for a specific symbol"""
    try:
        positions = position_manager.get_positions_by_symbol(symbol)

        return {
            "symbol": symbol,
            "stocks": [p.to_dict() for p in positions['stocks']],
            "options": [p.to_dict() for p in positions['options']],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting positions for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/positions/stock/{position_id}")
async def remove_stock_position(position_id: str):
    """Remove a stock position"""
    try:
        success = position_manager.remove_stock_position(position_id)
        if success:
            return {
                "status": "success",
                "message": f"Removed stock position {position_id}",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Position not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing stock position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/positions/option/{position_id}")
async def remove_option_position(position_id: str):
    """Remove an option position"""
    try:
        success = position_manager.remove_option_position(position_id)
        if success:
            return {
                "status": "success",
                "message": f"Removed option position {position_id}",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Position not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing option position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MARKET DATA ENDPOINTS
# ============================================================================

@app.get("/api/market/stock/{symbol}")
async def get_stock_data(symbol: str):
    """Get real-time stock data"""
    try:
        data = market_data.get_stock_price(symbol)
        if data:
            return data
        else:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stock data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/option/{symbol}")
async def get_option_data(
    symbol: str,
    option_type: str,
    strike: float,
    expiration_date: str
):
    """Get real-time option data"""
    try:
        data = market_data.get_option_price(symbol, option_type, strike, expiration_date)
        if data:
            return data
        else:
            raise HTTPException(status_code=404, detail="Option not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting option data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/chain/{symbol}")
async def get_option_chain(symbol: str, expiration_date: Optional[str] = None):
    """Get option chain for a symbol"""
    try:
        data = market_data.get_option_chain(symbol, expiration_date)
        if data:
            return data
        else:
            raise HTTPException(status_code=404, detail=f"No option chain found for {symbol}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting option chain: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/volatility/{symbol}")
async def get_volatility(symbol: str):
    """Get historical and implied volatility"""
    try:
        hv = market_data.get_historical_volatility(symbol)
        iv = market_data.get_implied_volatility(symbol)

        return {
            "symbol": symbol,
            "historical_volatility": hv,
            "implied_volatility": iv,
            "iv_hv_ratio": iv / hv if (hv and iv) else None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting volatility: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SENTIMENT & RESEARCH ENDPOINTS
# ============================================================================

@app.get("/api/sentiment/{symbol}")
async def get_sentiment(symbol: str):
    """Get sentiment analysis for a symbol"""
    try:
        # Create a minimal state for the agent
        state = {
            'positions': [{'symbol': symbol}],
            'errors': []
        }

        # Run sentiment research
        result = sentiment_agent.process(state)

        if symbol in result.get('sentiment_research', {}):
            return result['sentiment_research'][symbol]
        else:
            raise HTTPException(status_code=404, detail=f"No sentiment data for {symbol}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analysis/full")
async def run_full_analysis():
    """Run complete analysis on all positions with real-time data and sentiment"""
    try:
        # Get all positions
        stock_positions = position_manager.get_all_stock_positions()
        option_positions = position_manager.get_all_option_positions()

        if not stock_positions and not option_positions:
            raise HTTPException(status_code=400, detail="No positions found. Add positions first.")

        # Convert to analysis format
        positions = []

        for stock in stock_positions:
            positions.append({
                'symbol': stock.symbol,
                'type': 'stock',
                'quantity': stock.quantity,
                'entry_price': stock.entry_price,
                'target_price': stock.target_price,
                'position_id': stock.position_id
            })

        for option in option_positions:
            positions.append({
                'symbol': option.symbol,
                'type': 'option',
                'option_type': option.option_type,
                'strike': option.strike,
                'expiration_date': option.expiration_date,
                'quantity': option.quantity,
                'premium_paid': option.premium_paid,
                'target_price': option.target_price,
                'position_id': option.position_id,
                'days_to_expiry': option.days_to_expiry(),
                'time_to_expiry': option.time_to_expiry()
            })

        # Get real-time market data for all symbols
        symbols = position_manager.get_unique_symbols()
        market_data_dict = {}

        for symbol in symbols:
            data = market_data.get_market_data_for_position(symbol)
            if data:
                market_data_dict[symbol] = data

        # Calculate portfolio Greeks
        calculator = GreeksCalculator()
        portfolio_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

        # Run multi-agent analysis
        result = coordinator.run_analysis(
            positions=positions,
            market_data=market_data_dict,
            portfolio_greeks=portfolio_greeks,
            report_type='full'
        )

        # Store report
        reports_store.append(result['report'])

        return {
            "status": result['workflow_status'],
            "report": result['report'],
            "risk_score": result['risk_analysis'].get('risk_score', 0),
            "positions_analyzed": len(positions),
            "symbols": symbols,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running full analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analysis/demo")
async def run_demo_analysis():
    """Run a demo analysis with sample data."""
    try:
        # Sample positions
        positions = [
            {
                'symbol': 'AAPL',
                'strategy_type': 'long_call',
                'market_value': 1200.0,
                'pnl': 200.0,
                'pnl_pct': 20.0,
                'days_to_expiry': 45,
                'sector': 'Technology',
                'legs': [
                    {
                        'option_type': 'call',
                        'strike': 150.0,
                        'quantity': 1,
                        'is_short': False,
                        'entry_price': 10.0,
                        'time_to_expiry': 0.12,
                        'multiplier': 100
                    }
                ]
            }
        ]

        market_data = {
            'AAPL': {
                'underlying_price': 155.0,
                'iv': 0.30,
                'historical_iv': 0.28,
                'iv_rank': 65.0,
                'volume': 50000000,
                'avg_volume': 40000000,
                'put_call_ratio': 0.9,
                'days_to_earnings': 45,
                'sector': 'Technology',
                'time_to_expiry': 0.12,
                'risk_free_rate': 0.05
            }
        }

        # Calculate portfolio Greeks
        calculator = GreeksCalculator()
        portfolio_greeks = calculator.calculate_portfolio_greeks(positions, market_data)

        # Run analysis
        result = coordinator.run_analysis(
            positions=positions,
            market_data=market_data,
            portfolio_greeks=portfolio_greeks,
            report_type='demo'
        )

        # Store report
        reports_store.append(result['report'])

        return {
            "status": result['workflow_status'],
            "report": result['report'],
            "risk_score": result['risk_analysis'].get('risk_score', 0),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error running demo analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analytics/ev/demo")
async def calculate_demo_ev():
    """Calculate EV for a demo position."""
    try:
        position = {
            'symbol': 'AAPL',
            'strategy_type': 'long_call',
            'total_premium': 500.0,
            'legs': [
                {
                    'option_type': 'call',
                    'strike': 150.0,
                    'quantity': 1,
                    'is_short': False,
                    'entry_price': 5.0,
                    'multiplier': 100
                }
            ]
        }

        market_data = {
            'underlying_price': 155.0,
            'iv': 0.30,
            'time_to_expiry': 0.25,
            'risk_free_rate': 0.05
        }

        calculator = EVCalculator()
        result = calculator.calculate_ev(position, market_data)

        return {
            "expected_value": result.expected_value,
            "expected_return_pct": result.expected_return_pct,
            "probability_profit": result.probability_profit,
            "confidence_interval": result.confidence_interval,
            "method_breakdown": result.method_breakdown,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error calculating EV: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analytics/greeks/demo")
async def calculate_demo_greeks():
    """Calculate Greeks for a demo option."""
    try:
        calculator = GreeksCalculator()

        greeks = calculator.calculate_greeks(
            option_type='call',
            underlying_price=155.0,
            strike=150.0,
            time_to_expiry=0.25,
            iv=0.30
        )

        return {
            "delta": greeks.delta,
            "gamma": greeks.gamma,
            "theta": greeks.theta,
            "vega": greeks.vega,
            "rho": greeks.rho,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error calculating Greeks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reports/demo")
async def get_demo_reports():
    """Get demo reports."""
    return {
        "reports": reports_store[-5:] if reports_store else [],
        "count": len(reports_store),
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# MULTI-MODEL DISCUSSION ENDPOINTS
# ============================================================================

@app.post("/api/discussion/analyze/{symbol}")
async def run_multi_model_discussion(symbol: str):
    """
    Run a multi-model discussion for a symbol.

    Uses GPT-4, Claude Sonnet 4.5, and LM Studio in a 5-round discussion.
    Integrates Firecrawl for additional research data.
    """
    try:
        logger.info(f"Starting multi-model discussion for {symbol}")

        # Get position data
        positions = position_manager.get_positions_by_symbol(symbol)
        if not positions['stocks'] and not positions['options']:
            raise HTTPException(status_code=404, detail=f"No positions found for {symbol}")

        # Get market data
        market_data_result = market_data.get_stock_price(symbol)

        # Get sentiment data
        sentiment_state = {
            'positions': [{'symbol': symbol}],
            'sentiment_research': {},
            'errors': []
        }
        sentiment_result = sentiment_agent.process(sentiment_state)
        sentiment_data = sentiment_result.get('sentiment_research', {}).get(symbol, {})

        # TODO: Get Firecrawl data (implement when Firecrawl MCP is available)
        firecrawl_data = {
            'news': 'Firecrawl integration pending',
            'social_media': 'Firecrawl integration pending',
            'youtube': 'Firecrawl integration pending'
        }

        # Prepare position data
        position_data = {
            'stocks': [p.to_dict() for p in positions['stocks']],
            'options': [p.to_dict() for p in positions['options']]
        }

        # Run multi-model discussion
        discussion_result = multi_model_discussion.start_discussion(
            symbol=symbol,
            position_data=position_data,
            market_data=market_data_result,
            sentiment_data=sentiment_data,
            firecrawl_data=firecrawl_data
        )

        return discussion_result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in multi-model discussion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/discussion/status")
async def get_discussion_status():
    """Get status of multi-model discussion system."""
    from ..agents.multi_model_config import validate_configurations, AGENT_MODEL_ASSIGNMENTS

    config_status = validate_configurations()

    return {
        "models": config_status,
        "agent_assignments": AGENT_MODEL_ASSIGNMENTS,
        "discussion_rounds": 5,
        "status": "ready" if all(config_status.values()) else "partial",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "services": {
            "position_manager": True,
            "market_data": True,
            "research": research_service is not None,
            "earnings": earnings_service is not None,
            "scheduler": scheduler.is_running if scheduler else False
        }
    }


@app.get("/api/scheduler/status")
async def get_scheduler_status():
    """Get scheduler status and job information"""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not available")

    return scheduler.get_status()


# ============================================================================
# RESEARCH ENDPOINTS
# ============================================================================

@app.get("/api/research/{symbol}")
async def get_research(symbol: str, max_age_hours: int = 24):
    """
    Get comprehensive research for a symbol.
    Includes news, social media, and YouTube sentiment.
    """
    try:
        research = research_service.research_symbol(symbol.upper(), max_age_hours=max_age_hours)
        return research
    except Exception as e:
        logger.error(f"Error getting research for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/research/{symbol}/news")
async def get_news(symbol: str):
    """Get news for a symbol"""
    try:
        news = research_service.get_news(symbol.upper())
        return news
    except Exception as e:
        logger.error(f"Error getting news for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/research/{symbol}/social")
async def get_social(symbol: str):
    """Get social media sentiment for a symbol"""
    try:
        social = research_service.get_social_sentiment(symbol.upper())
        return social
    except Exception as e:
        logger.error(f"Error getting social sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/research/{symbol}/youtube")
async def get_youtube(symbol: str):
    """Get YouTube sentiment for a symbol"""
    try:
        youtube = research_service.get_youtube_sentiment(symbol.upper())
        return youtube
    except Exception as e:
        logger.error(f"Error getting YouTube sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/research/github/search")
async def search_github(query: str, max_results: int = 10):
    """Search GitHub repositories"""
    try:
        repos = research_service.search_github_repos(query, max_results=max_results)
        return {"query": query, "results": repos}
    except Exception as e:
        logger.error(f"Error searching GitHub: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# EARNINGS ENDPOINTS
# ============================================================================

@app.get("/api/earnings/calendar")
async def get_earnings_calendar(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    symbol: Optional[str] = None
):
    """
    Get earnings calendar.
    Defaults to next 30 days if no dates specified.
    """
    try:
        earnings = earnings_service.get_earnings_calendar(
            start_date=start_date,
            end_date=end_date,
            symbol=symbol.upper() if symbol else None
        )
        from datetime import timedelta
        return {
            "start_date": start_date or datetime.now().strftime('%Y-%m-%d'),
            "end_date": end_date or (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            "count": len(earnings),
            "earnings": earnings
        }
    except Exception as e:
        logger.error(f"Error getting earnings calendar: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/earnings/{symbol}/next")
async def get_next_earnings(symbol: str):
    """Get next earnings date for a symbol"""
    try:
        next_earnings = earnings_service.get_next_earnings(symbol.upper())
        if not next_earnings:
            raise HTTPException(status_code=404, detail=f"No upcoming earnings found for {symbol}")
        return next_earnings
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting next earnings for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/earnings/{symbol}/risk")
async def analyze_earnings_risk(
    symbol: str,
    position_type: str = 'stock',
    strike: Optional[float] = None,
    expiration: Optional[str] = None
):
    """Analyze earnings risk for a position"""
    try:
        risk = earnings_service.analyze_earnings_risk(
            symbol.upper(),
            position_type=position_type,
            strike=strike,
            expiration=expiration
        )
        return risk
    except Exception as e:
        logger.error(f"Error analyzing earnings risk for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/earnings/{symbol}/implied-move")
async def calculate_implied_move(
    symbol: str,
    earnings_date: str,
    current_price: float,
    atm_straddle_price: Optional[float] = None
):
    """Calculate implied earnings move"""
    try:
        move = earnings_service.calculate_implied_move(
            symbol.upper(),
            earnings_date=earnings_date,
            current_price=current_price,
            atm_straddle_price=atm_straddle_price
        )
        return move
    except Exception as e:
        logger.error(f"Error calculating implied move for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# RECOMMENDATION ENGINE ENDPOINTS
# ============================================================================

@app.get("/api/recommendations/{symbol}")
async def get_recommendation(symbol: str):
    """
    Get comprehensive recommendation for a symbol

    Returns:
    - Multi-factor score (0-100)
    - Recommendation (BUY/SELL/HOLD)
    - Specific actions to take
    - Risk factors and catalysts
    - Confidence level
    """
    try:
        logger.info(f"Getting recommendation for {symbol}")

        # Initialize recommendation engine
        rec_engine = RecommendationEngine()

        # Get current position (if any)
        position = None
        try:
            stock_positions = position_manager.get_all_stock_positions()
            for pos in stock_positions:
                if pos.symbol == symbol.upper():
                    position = pos.to_dict()
                    break
        except Exception as e:
            logger.warning(f"Could not fetch position for {symbol}: {e}")

        # Get market data
        market_data_dict = None
        try:
            market_data_result = market_data.get_stock_price(symbol.upper())
            if market_data_result:
                # get_stock_price already returns a dict with current_price
                market_data_dict = market_data_result
        except Exception as e:
            logger.warning(f"Could not fetch market data for {symbol}: {e}")

        # Generate recommendation
        result = rec_engine.analyze(
            symbol=symbol.upper(),
            position=position,
            market_data=market_data_dict
        )

        # Convert to dict for JSON response
        return {
            'symbol': result.symbol,
            'recommendation': result.recommendation,
            'confidence': result.confidence,
            'combined_score': result.combined_score,
            'scores': {
                name: {
                    'score': score.score,
                    'components': score.components,
                    'signals': score.signals,
                    'reasoning': score.reasoning,
                    'confidence': score.confidence
                }
                for name, score in result.scores.items()
            },
            'actions': result.actions,
            'reasoning': result.reasoning,
            'risk_factors': result.risk_factors,
            'catalysts': result.catalysts,
            'expected_outcome': result.expected_outcome,
            'timestamp': result.timestamp
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendation for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
