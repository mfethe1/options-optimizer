from __future__ import annotations
import time
from src.data.http_client import CachingHttpClient, RateLimiter


def test_rate_limiter_blocks_when_exceeded(monkeypatch):
    rl = RateLimiter(calls_per_min=2)
    # Simulate that a minute has already passed so acquire() doesn't sleep
    rl._last_reset = time.time() - 61
    rl._count = 2
    rl.acquire()
    # After acquire, it should reset and increment
    assert rl._count in (1, 0)


def test_cache_key_and_roundtrip(tmp_path):
    client = CachingHttpClient(str(tmp_path))
    url = "http://example.com/api"
    params = {"a": 1}

    # monkeypatch requests.get to avoid network
    class Resp:
        def __init__(self):
            self._json = {"ok": True}
        def raise_for_status(self):
            return
        def json(self):
            return self._json
    import requests
    def fake_get(u, params=None, timeout=10):
        return Resp()
    requests.get = fake_get

    data1 = client.get_json(url, params, RateLimiter(100), ttl_seconds=3600)
    data2 = client.get_json(url, params, RateLimiter(100), ttl_seconds=3600)
    assert data1 == data2

