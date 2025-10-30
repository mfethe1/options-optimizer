"""
Monitoring and Observability

Provides Sentry error tracking and Prometheus metrics for the application.
"""

import os
import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

logger = logging.getLogger(__name__)

# ============================================================================
# SENTRY CONFIGURATION
# ============================================================================

def setup_sentry():
    """
    Initialize Sentry error tracking.
    
    Set SENTRY_DSN environment variable to enable Sentry.
    Set SENTRY_ENVIRONMENT to specify environment (e.g., production, staging).
    Set SENTRY_TRACES_SAMPLE_RATE to control performance monitoring (0.0 to 1.0).
    """
    sentry_dsn = os.getenv("SENTRY_DSN")
    
    if not sentry_dsn:
        logger.info("Sentry DSN not configured. Error tracking disabled.")
        return
    
    environment = os.getenv("SENTRY_ENVIRONMENT", "development")
    traces_sample_rate = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1"))
    
    sentry_sdk.init(
        dsn=sentry_dsn,
        environment=environment,
        traces_sample_rate=traces_sample_rate,
        integrations=[
            StarletteIntegration(
                transaction_style="endpoint",
                failed_request_status_codes={403, *range(500, 599)},
            ),
            FastApiIntegration(
                transaction_style="endpoint",
                failed_request_status_codes={403, *range(500, 599)},
            ),
        ],
        # Send default PII (user info, request data)
        send_default_pii=True,
        # Enable performance profiling
        profiles_sample_rate=traces_sample_rate,
    )
    
    logger.info(f"Sentry initialized (environment: {environment}, traces_sample_rate: {traces_sample_rate})")


# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

# HTTP Request metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"]
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

http_request_size_bytes = Histogram(
    "http_request_size_bytes",
    "HTTP request size in bytes",
    ["method", "endpoint"],
    buckets=(100, 1000, 10000, 100000, 1000000, 10000000)
)

http_response_size_bytes = Histogram(
    "http_response_size_bytes",
    "HTTP response size in bytes",
    ["method", "endpoint"],
    buckets=(100, 1000, 10000, 100000, 1000000, 10000000)
)

http_requests_in_progress = Gauge(
    "http_requests_in_progress",
    "Number of HTTP requests currently being processed",
    ["method", "endpoint"]
)

# Swarm-specific metrics
swarm_analysis_duration_seconds = Histogram(
    "swarm_analysis_duration_seconds",
    "Time spent on swarm analysis in seconds",
    ["consensus_method"],
    buckets=(1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0)
)

swarm_agent_performance_seconds = Histogram(
    "swarm_agent_performance_seconds",
    "Time spent by individual agents in seconds",
    ["agent_type"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
)

swarm_consensus_time_seconds = Histogram(
    "swarm_consensus_time_seconds",
    "Time spent on consensus calculation in seconds",
    ["consensus_method"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
)

swarm_analysis_total = Counter(
    "swarm_analysis_total",
    "Total number of swarm analyses",
    ["consensus_method", "status"]
)

swarm_agent_errors_total = Counter(
    "swarm_agent_errors_total",
    "Total number of agent errors",
    ["agent_type"]
)

# Authentication metrics
auth_requests_total = Counter(
    "auth_requests_total",
    "Total number of authentication requests",
    ["endpoint", "status"]
)

auth_token_validations_total = Counter(
    "auth_token_validations_total",
    "Total number of token validations",
    ["status"]
)

# Cache metrics (for future use)
cache_hits_total = Counter(
    "cache_hits_total",
    "Total number of cache hits",
    ["cache_type"]
)

cache_misses_total = Counter(
    "cache_misses_total",
    "Total number of cache misses",
    ["cache_type"]
)


# ============================================================================
# PROMETHEUS MIDDLEWARE
# ============================================================================

class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Middleware to collect Prometheus metrics for all HTTP requests.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get endpoint path
        endpoint = request.url.path
        method = request.method
        
        # Track requests in progress
        http_requests_in_progress.labels(method=method, endpoint=endpoint).inc()
        
        # Track request size
        content_length = request.headers.get("content-length")
        if content_length:
            http_request_size_bytes.labels(method=method, endpoint=endpoint).observe(int(content_length))
        
        # Start timer
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Track metrics
            status = f"{response.status_code // 100}xx"
            http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
            http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
            
            # Track response size
            if hasattr(response, "body"):
                http_response_size_bytes.labels(method=method, endpoint=endpoint).observe(len(response.body))
            
            return response
        
        except Exception as e:
            # Track error
            http_requests_total.labels(method=method, endpoint=endpoint, status="5xx").inc()
            
            # Re-raise exception (Sentry will catch it)
            raise
        
        finally:
            # Decrement in-progress counter
            http_requests_in_progress.labels(method=method, endpoint=endpoint).dec()


# ============================================================================
# METRICS ENDPOINT
# ============================================================================

def get_metrics() -> Response:
    """
    Generate Prometheus metrics in text format.
    
    Returns:
        Response with metrics in Prometheus format
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def track_swarm_analysis(consensus_method: str, duration: float, status: str = "success"):
    """
    Track swarm analysis metrics.
    
    Args:
        consensus_method: Consensus method used
        duration: Analysis duration in seconds
        status: Analysis status (success, error)
    """
    swarm_analysis_duration_seconds.labels(consensus_method=consensus_method).observe(duration)
    swarm_analysis_total.labels(consensus_method=consensus_method, status=status).inc()


def track_agent_performance(agent_type: str, duration: float):
    """
    Track individual agent performance.
    
    Args:
        agent_type: Type of agent
        duration: Execution duration in seconds
    """
    swarm_agent_performance_seconds.labels(agent_type=agent_type).observe(duration)


def track_consensus_time(consensus_method: str, duration: float):
    """
    Track consensus calculation time.
    
    Args:
        consensus_method: Consensus method used
        duration: Consensus duration in seconds
    """
    swarm_consensus_time_seconds.labels(consensus_method=consensus_method).observe(duration)


def track_agent_error(agent_type: str):
    """
    Track agent errors.
    
    Args:
        agent_type: Type of agent that errored
    """
    swarm_agent_errors_total.labels(agent_type=agent_type).inc()


def track_auth_request(endpoint: str, status: str):
    """
    Track authentication requests.
    
    Args:
        endpoint: Authentication endpoint
        status: Request status (success, failed)
    """
    auth_requests_total.labels(endpoint=endpoint, status=status).inc()


def track_token_validation(status: str):
    """
    Track token validations.
    
    Args:
        status: Validation status (valid, invalid, expired)
    """
    auth_token_validations_total.labels(status=status).inc()


def track_cache_hit(cache_type: str):
    """
    Track cache hits.
    
    Args:
        cache_type: Type of cache
    """
    cache_hits_total.labels(cache_type=cache_type).inc()


def track_cache_miss(cache_type: str):
    """
    Track cache misses.
    
    Args:
        cache_type: Type of cache
    """
    cache_misses_total.labels(cache_type=cache_type).inc()

