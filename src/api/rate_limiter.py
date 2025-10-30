"""
Rate Limiting Configuration for FastAPI
Uses slowapi for in-memory rate limiting
"""

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi import Request, Response
from typing import Callable
import logging
import os

logger = logging.getLogger(__name__)

# Rate limit configurations
RATE_LIMITS = {
    # Analysis endpoints (expensive operations)
    'analysis': "10/minute",

    # Swarm endpoints (very expensive operations)
    'swarm': "5/minute",

    # Read-only endpoints (cheap operations)
    'read': "100/minute",

    # Write endpoints (moderate operations)
    'write': "30/minute",

    # Health check (unlimited)
    'health': "1000/minute",
}

# Custom key function that can use user ID if authenticated
def get_rate_limit_key(request: Request) -> str:
    """
    Get rate limit key from request.
    Uses user_id from query params if available, otherwise IP address.
    """
    # Try to get user_id from query params
    user_id = request.query_params.get('user_id')
    if user_id:
        return f"user:{user_id}"

    # Fall back to IP address
    return get_remote_address(request)


# Temporarily rename .env to avoid Unicode error
env_file = ".env"
env_backup = ".env.backup"
env_exists = os.path.exists(env_file)

if env_exists:
    try:
        os.rename(env_file, env_backup)
    except:
        pass

# Create limiter instance
try:
    limiter = Limiter(
        key_func=get_rate_limit_key,
        default_limits=["100/minute"],  # Default limit for all endpoints
        storage_uri="memory://",  # In-memory storage (no Redis needed)
        strategy="fixed-window",  # Simple fixed window strategy
        headers_enabled=True,  # Add rate limit headers to responses
    )
finally:
    # Restore .env file
    if env_exists:
        try:
            os.rename(env_backup, env_file)
        except:
            pass


def get_endpoint_limit(request: Request) -> str:
    """
    Determine rate limit based on endpoint path.
    Returns the appropriate rate limit string.
    """
    path = request.url.path
    
    # Health check - very high limit
    if path == "/health":
        return RATE_LIMITS['health']
    
    # Swarm analysis - very restrictive
    if "/api/swarm/analyze" in path:
        return RATE_LIMITS['swarm']
    
    # Other swarm endpoints - restrictive
    if "/api/swarm" in path:
        return RATE_LIMITS['analysis']
    
    # Analysis endpoints - restrictive
    if "/api/analysis" in path or "/api/analytics" in path:
        return RATE_LIMITS['analysis']
    
    # Write operations - moderate
    if request.method in ["POST", "PUT", "DELETE", "PATCH"]:
        return RATE_LIMITS['write']
    
    # Read operations - permissive
    return RATE_LIMITS['read']


class CustomRateLimitMiddleware:
    """
    Custom middleware to apply dynamic rate limits based on endpoint.
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Create request object
        request = Request(scope, receive)
        
        # Get appropriate rate limit for this endpoint
        rate_limit = get_endpoint_limit(request)
        
        # Add rate limit info to request state
        request.state.rate_limit = rate_limit
        
        # Continue with normal processing
        await self.app(scope, receive, send)


def setup_rate_limiting(app):
    """
    Setup rate limiting for FastAPI app.
    
    Args:
        app: FastAPI application instance
    """
    # Add limiter to app state
    app.state.limiter = limiter
    
    # Add exception handler for rate limit exceeded
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # Add SlowAPI middleware
    app.add_middleware(SlowAPIMiddleware)
    
    logger.info("Rate limiting configured successfully")
    logger.info(f"Rate limits: {RATE_LIMITS}")
    
    return limiter


# Decorator for custom rate limits on specific endpoints
def custom_limit(limit: str):
    """
    Decorator to apply custom rate limit to specific endpoint.
    
    Usage:
        @app.get("/api/special")
        @custom_limit("5/minute")
        async def special_endpoint():
            return {"message": "Limited to 5 per minute"}
    """
    return limiter.limit(limit)


# Decorator to exempt endpoint from rate limiting
def exempt_from_rate_limit():
    """
    Decorator to exempt endpoint from rate limiting.
    
    Usage:
        @app.get("/api/unlimited")
        @exempt_from_rate_limit()
        async def unlimited_endpoint():
            return {"message": "No rate limit"}
    """
    return limiter.exempt

