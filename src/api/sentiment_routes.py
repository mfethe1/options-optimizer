"""
Deep Sentiment Analysis API Routes

Advanced sentiment with influencer weighting and controversy detection.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from ..agents.swarm.agents.sentiment_deep_dive_agent import SentimentDeepDiveAgent
from ..agents.swarm.shared_context import SharedContext

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/sentiment",
    tags=["Sentiment Analysis"]
)

# Initialize agent
shared_context = SharedContext()
sentiment_agent = SentimentDeepDiveAgent(agent_id="sentiment_analyzer", shared_context=shared_context)


# Request/Response models
class SentimentAnalysisRequest(BaseModel):
    """Request for sentiment analysis"""
    symbol: str = Field(..., description="Stock symbol to analyze")
    sources: Optional[List[str]] = Field(
        None,
        description="Data sources: twitter, reddit, news, stocktwits. If None, uses all available."
    )
    lookback_hours: Optional[int] = Field(24, description="Hours of historical data to analyze")


class SentimentAnalysisResponse(BaseModel):
    """Response with sentiment analysis"""
    symbol: str
    sentiment: Dict[str, Any] = Field(..., description="Overall sentiment metrics")
    by_source: Dict[str, Any] = Field(..., description="Sentiment breakdown by source")
    influencer_sentiment: Optional[Dict[str, Any]] = Field(None, description="Sentiment from influencers")
    controversy_score: float = Field(..., description="Controversy score (0-100)")
    sentiment_velocity: float = Field(..., description="Rate of sentiment change")
    echo_chamber_detected: bool = Field(..., description="Whether echo chamber/bot activity detected")
    trading_implication: str = Field(..., description="What sentiment suggests for trading")
    timestamp: str


class SentimentComparisonRequest(BaseModel):
    """Request for multi-symbol sentiment comparison"""
    symbols: List[str] = Field(..., description="List of symbols to compare (2-10)")
    sources: Optional[List[str]] = Field(None, description="Data sources to use")


class SentimentComparisonResponse(BaseModel):
    """Response with comparative sentiment analysis"""
    symbols: List[str]
    comparisons: Dict[str, Dict[str, Any]] = Field(..., description="Sentiment for each symbol")
    relative_strength: Dict[str, str] = Field(..., description="Relative sentiment ranking")
    timestamp: str


# Routes

@router.post("/analyze", response_model=SentimentAnalysisResponse)
async def analyze_sentiment(request: SentimentAnalysisRequest):
    """
    Deep sentiment analysis with influencer weighting

    **COMPETITIVE ADVANTAGE**: Advanced sentiment backed by research

    LSEG research shows sentiment replicates multifactor performance with 0.73 correlation.

    Features:
    - **Influencer weighting**: Posts from accounts with 10K+ followers weighted more heavily
    - **Sentiment velocity**: Detects rapid sentiment shifts (momentum indicator)
    - **Controversy detection**: High disagreement = volatility signal
    - **Echo chamber filtering**: Identifies bot campaigns and coordinated manipulation

    Data sources:
    - Twitter/X (real-time, high signal)
    - Reddit (retail sentiment, r/wallstreetbets, r/options)
    - Financial news (institutional sentiment)
    - StockTwits (active trader sentiment)

    Args:
        request: Symbol and analysis parameters

    Returns:
        Comprehensive sentiment analysis with trading implications
    """
    try:
        logger.info(f"Analyzing sentiment for {request.symbol}")

        # Run deep sentiment analysis through agent
        result = await sentiment_agent.analyze(
            task={
                'symbol': request.symbol,
                'sources': request.sources or ['twitter', 'reddit', 'news', 'stocktwits'],
                'lookback_hours': request.lookback_hours
            },
            portfolio_context={}
        )

        # Extract key metrics from analysis
        analysis = result.get('analysis', {})

        return SentimentAnalysisResponse(
            symbol=request.symbol,
            sentiment=analysis.get('overall_sentiment', {}),
            by_source=analysis.get('by_source', {}),
            influencer_sentiment=analysis.get('influencer_sentiment'),
            controversy_score=analysis.get('controversy_score', 0.0),
            sentiment_velocity=analysis.get('sentiment_velocity', 0.0),
            echo_chamber_detected=analysis.get('echo_chamber_detected', False),
            trading_implication=analysis.get('trading_implication', 'Neutral'),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to analyze sentiment: {str(e)}")


@router.post("/compare", response_model=SentimentComparisonResponse)
async def compare_sentiment(request: SentimentComparisonRequest):
    """
    Compare sentiment across multiple symbols

    Useful for:
    - Sector sentiment analysis (compare AAPL, MSFT, GOOGL, etc.)
    - Identifying sentiment leaders and laggards
    - Finding sentiment divergences (bullish sentiment but price declining)

    Args:
        request: List of symbols to compare (2-10)

    Returns:
        Comparative sentiment analysis with relative rankings
    """
    try:
        if len(request.symbols) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 symbols for comparison")
        if len(request.symbols) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 symbols for comparison")

        logger.info(f"Comparing sentiment for {len(request.symbols)} symbols")

        comparisons = {}
        sentiment_scores = {}

        # Analyze each symbol
        for symbol in request.symbols:
            try:
                analysis_request = SentimentAnalysisRequest(
                    symbol=symbol,
                    sources=request.sources
                )
                result = await analyze_sentiment(analysis_request)

                comparisons[symbol] = {
                    'sentiment': result.sentiment,
                    'controversy_score': result.controversy_score,
                    'sentiment_velocity': result.sentiment_velocity,
                    'trading_implication': result.trading_implication
                }

                # Store overall sentiment score for ranking
                sentiment_scores[symbol] = result.sentiment.get('score', 50.0)

            except Exception as e:
                logger.warning(f"Error analyzing {symbol}: {e}")
                comparisons[symbol] = {'error': str(e)}
                sentiment_scores[symbol] = 50.0  # Neutral

        # Rank by sentiment score
        ranked_symbols = sorted(sentiment_scores.items(), key=lambda x: x[1], reverse=True)
        relative_strength = {
            symbol: f"#{idx + 1} - {score:.1f}/100"
            for idx, (symbol, score) in enumerate(ranked_symbols)
        }

        return SentimentComparisonResponse(
            symbols=request.symbols,
            comparisons=comparisons,
            relative_strength=relative_strength,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing sentiment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to compare sentiment: {str(e)}")


@router.get("/trending")
async def get_trending_sentiment(
    timeframe: str = "1h",
    limit: int = 20
):
    """
    Get stocks with trending sentiment

    Identifies stocks with:
    - Rapidly increasing mention volume
    - Strong sentiment shifts
    - High controversy scores (disagreement = volatility opportunity)

    Args:
        timeframe: Time window (1h, 4h, 24h)
        limit: Number of results to return

    Returns:
        List of trending stocks with sentiment metrics
    """
    try:
        logger.info(f"Fetching trending sentiment (timeframe: {timeframe}, limit: {limit})")

        # In production, this would query a trending sentiment database
        # For now, return a sample structure

        # Validate timeframe
        if timeframe not in ['1h', '4h', '24h']:
            raise HTTPException(status_code=400, detail="Invalid timeframe. Use: 1h, 4h, 24h")

        # This would be populated from real-time sentiment tracking
        trending = [
            {
                "symbol": "NVDA",
                "mention_volume": 15234,
                "mention_velocity": 12.3,  # % increase
                "sentiment_score": 78.5,
                "sentiment_shift": 15.2,  # Points change
                "controversy_score": 45.2,
                "top_keywords": ["earnings", "data center", "AI"],
                "trading_implication": "Bullish momentum"
            },
            # More would be populated in production
        ]

        return {
            "timeframe": timeframe,
            "trending": trending[:limit],
            "count": len(trending),
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching trending sentiment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch trending sentiment: {str(e)}")


@router.get("/influencers/{symbol}")
async def get_influencer_sentiment(symbol: str, limit: int = 10):
    """
    Get sentiment from top influencers for a symbol

    Tracks what major FinTwit accounts are saying about a stock.

    Influencer tiers:
    - Tier 1: 100K+ followers (institutional/celebrity traders)
    - Tier 2: 50K-100K followers (experienced traders)
    - Tier 3: 10K-50K followers (active community members)

    Args:
        symbol: Stock symbol
        limit: Number of influencer posts to return

    Returns:
        Recent posts from top influencers with sentiment analysis
    """
    try:
        logger.info(f"Fetching influencer sentiment for {symbol}")

        # In production, this would query influencer tracking database
        influencer_posts = [
            {
                "username": "@example_trader",
                "followers": 125000,
                "tier": 1,
                "post": "Bullish setup on $NVDA - breaking out of consolidation",
                "sentiment": "bullish",
                "sentiment_score": 85.0,
                "engagement": 1250,  # likes + retweets
                "timestamp": "2025-10-30T10:15:00Z"
            },
            # More would be populated in production
        ]

        return {
            "symbol": symbol,
            "influencer_posts": influencer_posts[:limit],
            "count": len(influencer_posts),
            "overall_influencer_sentiment": "bullish",  # Aggregated
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching influencer sentiment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch influencer sentiment: {str(e)}")


@router.get("/metrics")
async def get_sentiment_metrics():
    """
    Get sentiment analysis methodology and metrics

    Returns:
        Explanation of sentiment scoring and weighting
    """
    return {
        "methodology": {
            "weighting": {
                "influencer_weight": "log(followers) / 10 (to avoid extreme outliers)",
                "retail_weight": "1.0 (baseline)",
                "news_weight": "2.0 (institutional signal)"
            },
            "sentiment_score": {
                "scale": "0-100 (50 = neutral)",
                "calculation": "Weighted average of individual post sentiments",
                "components": ["positive_mentions", "negative_mentions", "engagement_weight"]
            },
            "controversy_score": {
                "scale": "0-100 (0 = consensus, 100 = extreme disagreement)",
                "calculation": "Standard deviation of sentiment scores * 10",
                "interpretation": "High controversy = volatility opportunity"
            },
            "sentiment_velocity": {
                "scale": "-100 to +100 (points per hour)",
                "calculation": "Rate of change in sentiment score",
                "interpretation": "Momentum indicator for sentiment shifts"
            }
        },
        "data_sources": {
            "twitter": {
                "update_frequency": "Real-time",
                "signal_quality": "High (active traders)",
                "coverage": "Broad"
            },
            "reddit": {
                "update_frequency": "Real-time",
                "signal_quality": "Medium (retail sentiment)",
                "coverage": "Focused (WSB, r/options)"
            },
            "news": {
                "update_frequency": "Hourly",
                "signal_quality": "High (institutional)",
                "coverage": "Selective (major events)"
            },
            "stocktwits": {
                "update_frequency": "Real-time",
                "signal_quality": "Medium (trader community)",
                "coverage": "Moderate"
            }
        },
        "echo_chamber_detection": {
            "signals": [
                "Identical or near-identical posts from multiple accounts",
                "Coordinated timing of posts",
                "Low-follower accounts with high engagement",
                "Copy-paste patterns"
            ],
            "action": "Filter out or downweight suspected bot activity"
        }
    }
