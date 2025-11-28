"""
Rate Limiting Middleware for FastAPI

Provides token bucket rate limiting with:
- Per-client rate limits (by user_id or IP)
- Endpoint-specific rate limit rules
- Rate limit headers (X-RateLimit-*)
- Thread-safe in-memory storage
- Graceful handling of WebSocket connections

Security Features:
- Protects expensive endpoints (swarm analysis, ML training)
- Per-user tracking via user_id query param
- IP-based fallback for anonymous requests
- Proper 429 responses with Retry-After header

For production: Consider using Redis for distributed rate limiting.
"""
import time
import logging
import threading
from typing import Dict, Optional, Callable, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


@dataclass
class RateLimitRule:
    """Rate limit rule configuration."""
    max_requests: int
    window_seconds: int
    description: str = ""


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    tokens: float
    last_update: float
    max_tokens: int
    refill_rate: float  # tokens per second

    def consume(self, current_time: float) -> Tuple[bool, int, float]:
        """
        Try to consume a token.

        Returns:
            Tuple of (allowed, remaining_tokens, reset_time)
        """
        # Refill tokens based on elapsed time
        elapsed = current_time - self.last_update
        self.tokens = min(self.max_tokens, self.tokens + (elapsed * self.refill_rate))
        self.last_update = current_time

        if self.tokens >= 1:
            self.tokens -= 1
            remaining = int(self.tokens)
            reset_time = current_time + ((self.max_tokens - self.tokens) / self.refill_rate)
            return True, remaining, reset_time
        else:
            # Calculate when next token available
            retry_after = (1 - self.tokens) / self.refill_rate
            return False, 0, current_time + retry_after


class RateLimiter:
    """
    Thread-safe token bucket rate limiter with endpoint-specific rules.

    Rate limit rules (requests per window):
    - /api/swarm/*: 5 per minute (expensive multi-agent analysis)
    - /api/gnn/train: 2 per 5 minutes (very expensive training)
    - /api/pinn/train: 2 per 5 minutes (very expensive training)
    - /api/mamba/train: 2 per 5 minutes (very expensive training)
    - /api/forecast/*: 30 per minute (ML inference)
    - /api/unified/*: 30 per minute (unified endpoints)
    - /api/investor-report: 10 per minute (LLM-heavy)
    - /api/*: 100 per minute (default API)
    - /cache/*: 20 per minute (cache management)
    """

    # Rate limit rules: path_prefix -> RateLimitRule
    # Order matters - more specific rules should come first
    RULES: Dict[str, RateLimitRule] = {
        # Very expensive operations - strict limits
        '/api/swarm/analyze': RateLimitRule(5, 60, "Swarm analysis - multi-agent LLM calls"),
        '/api/gnn/train': RateLimitRule(2, 300, "GNN training - GPU intensive"),
        '/api/pinn/train': RateLimitRule(2, 300, "PINN training - GPU intensive"),
        '/api/mamba/train': RateLimitRule(2, 300, "Mamba training - GPU intensive"),
        '/api/ml/train': RateLimitRule(2, 300, "ML training - GPU intensive"),

        # Expensive operations - moderate limits
        '/api/swarm/': RateLimitRule(10, 60, "Swarm endpoints"),
        '/api/investor-report': RateLimitRule(10, 60, "Investor report - LLM heavy"),
        '/api/ai/': RateLimitRule(20, 60, "AI services"),

        # ML inference - reasonable limits
        '/api/forecast/': RateLimitRule(30, 60, "ML forecasting"),
        '/api/unified/': RateLimitRule(30, 60, "Unified analysis"),
        '/api/gnn/': RateLimitRule(30, 60, "GNN inference"),
        '/api/pinn/': RateLimitRule(30, 60, "PINN inference"),
        '/api/mamba/': RateLimitRule(30, 60, "Mamba inference"),
        '/api/epidemic/': RateLimitRule(30, 60, "Epidemic model"),

        # Cache management - prevent abuse
        '/cache/clear': RateLimitRule(5, 60, "Cache clear - admin only"),
        '/cache/': RateLimitRule(20, 60, "Cache operations"),

        # Default API limit
        '/api/': RateLimitRule(100, 60, "Default API limit"),
    }

    # Paths exempt from rate limiting
    EXEMPT_PATHS = frozenset([
        '/health',
        '/health/detailed',
        '/metrics',
        '/',
        '/docs',
        '/redoc',
        '/openapi.json',
        '/debug/tf',
    ])

    # Path prefixes to exempt (WebSocket, static files)
    EXEMPT_PREFIXES = (
        '/ws/',
        '/static/',
    )

    def __init__(self):
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.RLock()
        self._cleanup_counter = 0
        self._cleanup_interval = 100  # Cleanup every N requests

    def _get_rule(self, path: str) -> RateLimitRule:
        """Get the rate limit rule for a path (most specific match wins)."""
        for prefix, rule in self.RULES.items():
            if path.startswith(prefix):
                return rule
        # Default: 100 requests per minute
        return RateLimitRule(100, 60, "Default")

    def _get_client_id(self, request: Request) -> str:
        """
        Get client identifier for rate limiting.

        Priority:
        1. user_id from query params (authenticated users)
        2. X-Forwarded-For header (behind proxy)
        3. Client IP address
        """
        # Check for user_id in query params
        user_id = request.query_params.get('user_id')
        if user_id:
            # Sanitize user_id to prevent injection
            sanitized = ''.join(c for c in user_id[:64] if c.isalnum() or c in '-_')
            if sanitized:
                return f"user:{sanitized}"

        # Check X-Forwarded-For header
        forwarded = request.headers.get('X-Forwarded-For')
        if forwarded:
            # Take first IP (client IP, rest are proxies)
            client_ip = forwarded.split(',')[0].strip()
            return f"ip:{client_ip}"

        # Fall back to direct client IP
        client = request.client
        if client:
            return f"ip:{client.host}"

        return "ip:unknown"

    def _get_bucket_key(self, client_id: str, rule: RateLimitRule) -> str:
        """Generate bucket key for client+rule combination."""
        # Include max_requests and window in key so different rules have separate buckets
        return f"{client_id}:{rule.max_requests}:{rule.window_seconds}"

    def _cleanup_old_buckets(self, current_time: float):
        """Remove stale buckets to prevent memory growth."""
        # Only cleanup periodically
        self._cleanup_counter += 1
        if self._cleanup_counter < self._cleanup_interval:
            return
        self._cleanup_counter = 0

        # Remove buckets that haven't been used for 10 minutes
        stale_threshold = current_time - 600
        stale_keys = [
            key for key, bucket in self._buckets.items()
            if bucket.last_update < stale_threshold
        ]
        for key in stale_keys:
            del self._buckets[key]

        if stale_keys:
            logger.debug(f"Rate limiter cleanup: removed {len(stale_keys)} stale buckets")

    def check_rate_limit(self, request: Request) -> Optional[JSONResponse]:
        """
        Check if request should be rate limited.

        Returns:
            None if allowed, JSONResponse(429) if rate limited
        """
        path = request.url.path

        # Check exemptions
        if path in self.EXEMPT_PATHS:
            return None

        for prefix in self.EXEMPT_PREFIXES:
            if path.startswith(prefix):
                return None

        # Get rate limit rule and client ID
        rule = self._get_rule(path)
        client_id = self._get_client_id(request)
        bucket_key = self._get_bucket_key(client_id, rule)
        current_time = time.time()

        with self._lock:
            # Cleanup stale buckets periodically
            self._cleanup_old_buckets(current_time)

            # Get or create bucket
            if bucket_key not in self._buckets:
                # Create new bucket with full tokens
                refill_rate = rule.max_requests / rule.window_seconds
                self._buckets[bucket_key] = TokenBucket(
                    tokens=rule.max_requests,
                    last_update=current_time,
                    max_tokens=rule.max_requests,
                    refill_rate=refill_rate
                )

            bucket = self._buckets[bucket_key]
            allowed, remaining, reset_time = bucket.consume(current_time)

        if not allowed:
            retry_after = int(reset_time - current_time) + 1
            logger.warning(
                f"Rate limit exceeded: client={client_id}, path={path}, "
                f"limit={rule.max_requests}/{rule.window_seconds}s, retry_after={retry_after}s"
            )

            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {rule.max_requests} requests per {rule.window_seconds} seconds for this endpoint",
                    "retry_after": retry_after,
                    "limit_description": rule.description
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(rule.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(reset_time))
                }
            )

        # Store rate limit info for response headers
        request.state.rate_limit_info = {
            "limit": rule.max_requests,
            "remaining": remaining,
            "reset": int(reset_time)
        }

        return None

    def get_rate_limit_headers(self, request: Request) -> Dict[str, str]:
        """Get rate limit headers to add to response."""
        if hasattr(request.state, 'rate_limit_info'):
            info = request.state.rate_limit_info
            return {
                "X-RateLimit-Limit": str(info["limit"]),
                "X-RateLimit-Remaining": str(info["remaining"]),
                "X-RateLimit-Reset": str(info["reset"])
            }
        return {}


# Global rate limiter instance
_rate_limiter = RateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.

    Adds rate limit headers to all responses and blocks requests
    that exceed the rate limit with 429 Too Many Requests.
    """

    async def dispatch(self, request: Request, call_next: Callable):
        # Check rate limit
        rate_limit_response = _rate_limiter.check_rate_limit(request)
        if rate_limit_response:
            return rate_limit_response

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        headers = _rate_limiter.get_rate_limit_headers(request)
        for key, value in headers.items():
            response.headers[key] = value

        return response


def setup_rate_limiting(app):
    """
    Setup rate limiting middleware for FastAPI app.

    IMPORTANT: Call this BEFORE adding other middleware to ensure
    rate limiting is applied first.

    Args:
        app: FastAPI application instance

    Example:
        app = FastAPI()
        setup_rate_limiting(app)  # Add rate limiting first
        app.add_middleware(CORSMiddleware, ...)  # Then other middleware
    """
    app.add_middleware(RateLimitMiddleware)

    logger.info("Rate limiting middleware enabled")
    logger.info(f"Rate limit rules: {len(RateLimiter.RULES)} endpoint-specific rules configured")
    for path, rule in RateLimiter.RULES.items():
        logger.debug(f"  {path}: {rule.max_requests}/{rule.window_seconds}s - {rule.description}")

    return _rate_limiter


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    return _rate_limiter


# Compatibility exports for existing code that imports from rate_limiter.py
# These provide a simplified interface for the old slowapi-style decorators

class CompatibilityLimiter:
    """
    Compatibility shim for code expecting slowapi-style limiter.

    Provides a no-op decorator since rate limiting is now handled
    by middleware. The actual limits are enforced by RateLimitMiddleware.
    """

    def limit(self, limit_string: str):
        """
        No-op decorator for compatibility.

        Rate limiting is now handled by middleware, not decorators.
        This exists only to prevent import errors in existing code.
        """
        def decorator(func):
            return func
        return decorator

    def exempt(self, func):
        """No-op exempt decorator for compatibility."""
        return func


# Export compatibility limiter for existing code
limiter = CompatibilityLimiter()


def custom_limit(limit: str):
    """
    Compatibility decorator (no-op).

    Rate limiting is now handled by middleware based on path patterns.
    Keeping this for backward compatibility with existing code.
    """
    return limiter.limit(limit)
