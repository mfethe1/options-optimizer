"""
API Routes for News Integration

Bloomberg NEWS equivalent - real-time financial news feeds
"""
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query
from typing import List, Optional
from datetime import datetime, timedelta
import logging
import asyncio

from ..news import (
    NewsAggregator,
    get_news_aggregator,
    NewsArticle,
    NewsCategory,
    NewsSentiment,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/news", tags=["News"])

# Initialize aggregator
news_aggregator = get_news_aggregator()


@router.get("/")
async def get_news(
    symbols: Optional[str] = Query(None, description="Comma-separated list of symbols"),
    categories: Optional[str] = Query(None, description="Comma-separated list of categories"),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum number of articles"),
    hours: int = Query(default=24, ge=1, le=168, description="Look back hours (max 7 days)"),
) -> dict:
    """
    Get latest financial news.

    Args:
        symbols: Filter by stock symbols (e.g., "AAPL,MSFT,TSLA")
        categories: Filter by categories (e.g., "earnings,merger_acquisition")
        limit: Maximum number of articles (1-200)
        hours: Look back hours (1-168, i.e., up to 7 days)

    Returns:
        News feed with articles
    """
    try:
        # Parse symbols
        symbol_list = None
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]

        # Parse categories
        category_list = None
        if categories:
            category_list = []
            for cat in categories.split(','):
                try:
                    category_list.append(NewsCategory(cat.strip()))
                except ValueError:
                    logger.warning(f"Invalid category: {cat}")

        # Calculate since date
        since = datetime.now() - timedelta(hours=hours)

        # Fetch news
        articles = await news_aggregator.get_news(
            symbols=symbol_list,
            categories=category_list,
            limit=limit,
            since=since,
        )

        # Serialize articles
        return {
            'count': len(articles),
            'articles': [
                {
                    'id': article.id,
                    'title': article.title,
                    'summary': article.summary,
                    'url': article.url,
                    'source': article.source,
                    'author': article.author,
                    'published_at': article.published_at.isoformat(),
                    'symbols': article.symbols,
                    'categories': [cat.value for cat in article.categories],
                    'sentiment': article.sentiment.value if article.sentiment else None,
                    'sentiment_score': article.sentiment_score,
                    'image_url': article.image_url,
                    'provider': article.provider,
                }
                for article in articles
            ],
            'filters': {
                'symbols': symbol_list,
                'categories': [cat.value for cat in category_list] if category_list else None,
                'hours': hours,
            },
        }

    except Exception as e:
        logger.error(f"Failed to fetch news: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch news: {str(e)}")


@router.get("/search")
async def search_news(
    q: str = Query(..., description="Search query"),
    symbols: Optional[str] = Query(None, description="Comma-separated list of symbols"),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict:
    """
    Search financial news.

    Args:
        q: Search query string
        symbols: Filter by stock symbols
        limit: Maximum number of articles

    Returns:
        Search results
    """
    try:
        # Parse symbols
        symbol_list = None
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]

        # Search news
        articles = await news_aggregator.search_news(
            query=q,
            symbols=symbol_list,
            limit=limit,
        )

        return {
            'query': q,
            'count': len(articles),
            'articles': [
                {
                    'id': article.id,
                    'title': article.title,
                    'summary': article.summary,
                    'url': article.url,
                    'source': article.source,
                    'author': article.author,
                    'published_at': article.published_at.isoformat(),
                    'symbols': article.symbols,
                    'categories': [cat.value for cat in article.categories],
                    'sentiment': article.sentiment.value if article.sentiment else None,
                    'sentiment_score': article.sentiment_score,
                    'image_url': article.image_url,
                    'provider': article.provider,
                }
                for article in articles
            ],
        }

    except Exception as e:
        logger.error(f"Failed to search news: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search news: {str(e)}")


@router.get("/symbols/{symbol}")
async def get_symbol_news(
    symbol: str,
    limit: int = Query(default=50, ge=1, le=200),
    hours: int = Query(default=24, ge=1, le=168),
) -> dict:
    """
    Get news for a specific symbol.

    Args:
        symbol: Stock symbol (e.g., "AAPL")
        limit: Maximum number of articles
        hours: Look back hours

    Returns:
        News feed for the symbol
    """
    try:
        since = datetime.now() - timedelta(hours=hours)

        articles = await news_aggregator.get_news(
            symbols=[symbol.upper()],
            limit=limit,
            since=since,
        )

        return {
            'symbol': symbol.upper(),
            'count': len(articles),
            'articles': [
                {
                    'id': article.id,
                    'title': article.title,
                    'summary': article.summary,
                    'url': article.url,
                    'source': article.source,
                    'author': article.author,
                    'published_at': article.published_at.isoformat(),
                    'symbols': article.symbols,
                    'categories': [cat.value for cat in article.categories],
                    'sentiment': article.sentiment.value if article.sentiment else None,
                    'sentiment_score': article.sentiment_score,
                    'image_url': article.image_url,
                    'provider': article.provider,
                }
                for article in articles
            ],
        }

    except Exception as e:
        logger.error(f"Failed to fetch news for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch news: {str(e)}")


@router.get("/categories")
async def get_categories() -> dict:
    """
    Get available news categories.

    Returns:
        List of available categories
    """
    return {
        'categories': [
            {
                'value': cat.value,
                'name': cat.value.replace('_', ' ').title(),
            }
            for cat in NewsCategory
        ]
    }


@router.get("/providers/status")
async def get_provider_status() -> dict:
    """
    Get status of news providers.

    Returns:
        Provider status information
    """
    try:
        status = news_aggregator.get_provider_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get provider status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get provider status: {str(e)}")


@router.websocket("/ws/stream")
async def news_stream_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time news streaming.

    Sends new articles as they become available (polling-based).

    Usage:
        ws://localhost:8000/api/news/ws/stream

    Client can send config:
        {
            "symbols": ["AAPL", "MSFT"],
            "categories": ["earnings", "analyst_rating"],
            "interval_seconds": 60
        }
    """
    await websocket.accept()
    logger.info("News stream WebSocket connected")

    # Default config
    config = {
        'symbols': None,
        'categories': None,
        'interval_seconds': 60,  # Check for new news every 60 seconds
    }

    last_check = datetime.now() - timedelta(minutes=5)  # Start with last 5 minutes

    try:
        # Send welcome message
        await websocket.send_json({
            'type': 'connected',
            'message': 'Connected to news stream',
            'timestamp': datetime.now().isoformat(),
        })

        # Main loop
        while True:
            # Check for client messages (non-blocking)
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=0.1
                )

                # Update config
                if 'symbols' in data:
                    config['symbols'] = data['symbols']
                if 'categories' in data:
                    config['categories'] = [NewsCategory(cat) for cat in data['categories']]
                if 'interval_seconds' in data:
                    config['interval_seconds'] = max(10, min(600, data['interval_seconds']))

                await websocket.send_json({
                    'type': 'config_updated',
                    'config': {
                        'symbols': config['symbols'],
                        'categories': [cat.value for cat in config['categories']] if config['categories'] else None,
                        'interval_seconds': config['interval_seconds'],
                    },
                    'timestamp': datetime.now().isoformat(),
                })

            except asyncio.TimeoutError:
                pass  # No message from client, continue

            # Fetch new news
            now = datetime.now()
            time_since_last_check = (now - last_check).total_seconds()

            if time_since_last_check >= config['interval_seconds']:
                try:
                    # Fetch news since last check
                    articles = await news_aggregator.get_news(
                        symbols=config['symbols'],
                        categories=config['categories'],
                        limit=20,
                        since=last_check,
                    )

                    if articles:
                        # Send articles
                        await websocket.send_json({
                            'type': 'news',
                            'count': len(articles),
                            'articles': [
                                {
                                    'id': article.id,
                                    'title': article.title,
                                    'summary': article.summary,
                                    'url': article.url,
                                    'source': article.source,
                                    'author': article.author,
                                    'published_at': article.published_at.isoformat(),
                                    'symbols': article.symbols,
                                    'categories': [cat.value for cat in article.categories],
                                    'sentiment': article.sentiment.value if article.sentiment else None,
                                    'image_url': article.image_url,
                                    'provider': article.provider,
                                }
                                for article in articles
                            ],
                            'timestamp': now.isoformat(),
                        })

                        logger.info(f"Sent {len(articles)} new articles via WebSocket")

                    last_check = now

                except Exception as e:
                    logger.error(f"Error fetching news for WebSocket: {e}")
                    await websocket.send_json({
                        'type': 'error',
                        'message': f"Failed to fetch news: {str(e)}",
                        'timestamp': now.isoformat(),
                    })

            # Heartbeat
            await websocket.send_json({
                'type': 'heartbeat',
                'timestamp': now.isoformat(),
            })

            # Sleep before next iteration
            await asyncio.sleep(min(5, config['interval_seconds']))

    except WebSocketDisconnect:
        logger.info("News stream WebSocket disconnected")
    except Exception as e:
        logger.error(f"News stream WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint for news service."""
    return {
        "status": "ok",
        "service": "news",
        "timestamp": datetime.now().isoformat(),
        "providers": news_aggregator.get_provider_status(),
    }
