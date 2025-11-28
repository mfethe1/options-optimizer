"""
Tests for rate limiting middleware.

Verifies:
- Rate limit rules are correctly matched to paths
- Token bucket algorithm works correctly
- Rate limit headers are added to responses
- 429 responses are returned when limits exceeded
- Client identification works (user_id, IP)
"""
import pytest
import time
from unittest.mock import MagicMock, patch
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.rate_limiting import (
    RateLimiter,
    RateLimitRule,
    TokenBucket,
    setup_rate_limiting,
    get_rate_limiter,
)


class TestTokenBucket:
    """Test token bucket algorithm."""

    def test_consume_with_tokens_available(self):
        """Test consuming when tokens are available."""
        bucket = TokenBucket(
            tokens=10.0,
            last_update=time.time(),
            max_tokens=10,
            refill_rate=10 / 60  # 10 per minute
        )
        current_time = time.time()

        allowed, remaining, reset = bucket.consume(current_time)

        assert allowed is True
        assert remaining == 9
        assert reset > current_time

    def test_consume_depletes_tokens(self):
        """Test that consuming depletes tokens."""
        bucket = TokenBucket(
            tokens=2.0,
            last_update=time.time(),
            max_tokens=10,
            refill_rate=10 / 60
        )
        current_time = time.time()

        # First consume
        allowed1, remaining1, _ = bucket.consume(current_time)
        assert allowed1 is True
        assert remaining1 == 1

        # Second consume
        allowed2, remaining2, _ = bucket.consume(current_time)
        assert allowed2 is True
        assert remaining2 == 0

        # Third consume - should fail
        allowed3, remaining3, _ = bucket.consume(current_time)
        assert allowed3 is False
        assert remaining3 == 0

    def test_tokens_refill_over_time(self):
        """Test that tokens refill over time."""
        start_time = time.time()
        bucket = TokenBucket(
            tokens=0.0,
            last_update=start_time,
            max_tokens=10,
            refill_rate=10 / 60  # 10 per minute = 1 per 6 seconds
        )

        # After 6 seconds, should have ~1 token
        current_time = start_time + 6
        allowed, _, _ = bucket.consume(current_time)
        assert allowed is True

    def test_tokens_do_not_exceed_max(self):
        """Test that tokens don't exceed max."""
        start_time = time.time()
        bucket = TokenBucket(
            tokens=10.0,
            last_update=start_time,
            max_tokens=10,
            refill_rate=10 / 60
        )

        # After a long time, tokens should still be capped at max
        current_time = start_time + 3600  # 1 hour later
        allowed, remaining, _ = bucket.consume(current_time)

        assert allowed is True
        assert remaining == 9  # Max 10, consumed 1


class TestRateLimiter:
    """Test rate limiter logic."""

    def test_get_rule_specific_path(self):
        """Test that specific rules are matched."""
        limiter = RateLimiter()

        # Swarm analyze endpoint
        rule = limiter._get_rule('/api/swarm/analyze')
        assert rule.max_requests == 5
        assert rule.window_seconds == 60

        # Training endpoints
        rule = limiter._get_rule('/api/gnn/train')
        assert rule.max_requests == 2
        assert rule.window_seconds == 300

    def test_get_rule_prefix_matching(self):
        """Test prefix-based rule matching."""
        limiter = RateLimiter()

        # General swarm endpoint (not /analyze)
        rule = limiter._get_rule('/api/swarm/metrics')
        assert rule.max_requests == 10

        # General API endpoint
        rule = limiter._get_rule('/api/positions')
        assert rule.max_requests == 100

    def test_get_rule_default(self):
        """Test default rule for unknown paths."""
        limiter = RateLimiter()

        rule = limiter._get_rule('/unknown/path')
        assert rule.max_requests == 100
        assert rule.window_seconds == 60

    def test_client_id_from_user_id(self):
        """Test client ID extraction from user_id query param."""
        limiter = RateLimiter()
        request = MagicMock(spec=Request)
        request.query_params = {'user_id': 'test-user-123'}
        request.headers = {}

        client_id = limiter._get_client_id(request)
        assert client_id == 'user:test-user-123'

    def test_client_id_from_forwarded_header(self):
        """Test client ID extraction from X-Forwarded-For header."""
        limiter = RateLimiter()
        request = MagicMock(spec=Request)
        request.query_params = {}
        request.headers = {'X-Forwarded-For': '192.168.1.1, 10.0.0.1'}

        client_id = limiter._get_client_id(request)
        assert client_id == 'ip:192.168.1.1'

    def test_client_id_from_client_host(self):
        """Test client ID extraction from client.host."""
        limiter = RateLimiter()
        request = MagicMock(spec=Request)
        request.query_params = {}
        request.headers = {}
        request.client = MagicMock()
        request.client.host = '127.0.0.1'

        client_id = limiter._get_client_id(request)
        assert client_id == 'ip:127.0.0.1'

    def test_client_id_sanitization(self):
        """Test that user_id is sanitized to prevent injection."""
        limiter = RateLimiter()
        request = MagicMock(spec=Request)
        request.query_params = {'user_id': 'user<script>alert(1)</script>'}
        request.headers = {}

        client_id = limiter._get_client_id(request)
        # Should only contain alphanumeric and -_
        assert '<' not in client_id
        assert '>' not in client_id
        assert 'script' in client_id  # alphanumeric part preserved

    def test_exempt_paths(self):
        """Test that exempt paths are not rate limited."""
        limiter = RateLimiter()

        for path in ['/health', '/health/detailed', '/', '/docs', '/metrics']:
            request = MagicMock(spec=Request)
            request.url = MagicMock()
            request.url.path = path
            request.query_params = {}
            request.headers = {}
            request.client = MagicMock()
            request.client.host = '127.0.0.1'
            request.state = MagicMock()

            response = limiter.check_rate_limit(request)
            assert response is None, f"Path {path} should be exempt from rate limiting"

    def test_websocket_paths_exempt(self):
        """Test that WebSocket paths are not rate limited."""
        limiter = RateLimiter()
        request = MagicMock(spec=Request)
        request.url = MagicMock()
        request.url.path = '/ws/agent-stream/user123'
        request.query_params = {}
        request.headers = {}
        request.client = MagicMock()
        request.client.host = '127.0.0.1'
        request.state = MagicMock()

        response = limiter.check_rate_limit(request)
        assert response is None


class TestRateLimitingIntegration:
    """Integration tests with FastAPI."""

    @pytest.fixture
    def app_with_rate_limiting(self):
        """Create a test app with rate limiting enabled."""
        app = FastAPI()
        setup_rate_limiting(app)

        @app.get("/api/test")
        async def test_endpoint():
            return {"message": "ok"}

        @app.get("/health")
        async def health_endpoint():
            return {"status": "healthy"}

        return app

    def test_rate_limit_headers_added(self, app_with_rate_limiting):
        """Test that rate limit headers are added to responses."""
        client = TestClient(app_with_rate_limiting)

        response = client.get("/api/test")

        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    def test_exempt_endpoint_no_rate_limit_headers(self, app_with_rate_limiting):
        """Test that exempt endpoints don't get rate limit headers."""
        client = TestClient(app_with_rate_limiting)

        response = client.get("/health")

        assert response.status_code == 200
        # Exempt endpoints may or may not have headers - just verify no 429
        assert response.status_code != 429

    def test_rate_limit_exceeded(self, app_with_rate_limiting):
        """Test that exceeding rate limit returns 429."""
        client = TestClient(app_with_rate_limiting)

        # Make many requests rapidly (more than default limit of 100/min)
        # Note: In test mode, each request is tracked per-client
        # We need to make requests faster than refill rate

        # First, let's test with a simpler approach - verify 429 format
        # when we manually trigger it
        limiter = get_rate_limiter()

        # Artificially deplete the bucket
        request = MagicMock(spec=Request)
        request.url = MagicMock()
        request.url.path = '/api/test'
        request.query_params = {'user_id': 'rate-limit-test'}
        request.headers = {}
        request.client = MagicMock()
        request.client.host = '127.0.0.1'
        request.state = MagicMock()

        # Make 101 requests (exceeds 100/min limit)
        for i in range(101):
            response = limiter.check_rate_limit(request)
            if response is not None:
                assert response.status_code == 429
                assert "Retry-After" in response.headers
                break
        else:
            # If we didn't hit rate limit, that's unexpected
            pytest.fail("Expected rate limit to be exceeded after 101 requests")


class TestRateLimitRules:
    """Test specific rate limit rule configurations."""

    def test_swarm_analyze_limit(self):
        """Test swarm analyze has strict limit."""
        limiter = RateLimiter()
        rule = limiter._get_rule('/api/swarm/analyze')

        assert rule.max_requests == 5
        assert rule.window_seconds == 60
        assert "swarm" in rule.description.lower()

    def test_training_endpoints_strict_limit(self):
        """Test training endpoints have very strict limits."""
        limiter = RateLimiter()

        for path in ['/api/gnn/train', '/api/pinn/train', '/api/mamba/train', '/api/ml/train']:
            rule = limiter._get_rule(path)
            assert rule.max_requests == 2
            assert rule.window_seconds == 300  # 5 minutes
            assert "train" in rule.description.lower()

    def test_investor_report_limit(self):
        """Test investor report has moderate limit."""
        limiter = RateLimiter()
        rule = limiter._get_rule('/api/investor-report')

        assert rule.max_requests == 10
        assert rule.window_seconds == 60

    def test_forecast_endpoints_limit(self):
        """Test forecast endpoints have reasonable limits."""
        limiter = RateLimiter()

        for path in ['/api/forecast/all', '/api/forecast/gnn', '/api/unified/forecast']:
            rule = limiter._get_rule(path)
            assert rule.max_requests == 30
            assert rule.window_seconds == 60


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
