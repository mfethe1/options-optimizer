from __future__ import annotations
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class RateLimiter:
    calls_per_min: int
    _last_reset: float = time.time()
    _count: int = 0

    def acquire(self):
        now = time.time()
        if now - self._last_reset >= 60.0:
            self._last_reset = now
            self._count = 0
        if self._count >= self.calls_per_min:
            sleep_for = 60.0 - (now - self._last_reset)
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_reset = time.time()
            self._count = 0
        self._count += 1


class CachingHttpClient:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, url: str, params: Optional[Dict[str, Any]]) -> str:
        key = url + "?" + json.dumps(params or {}, sort_keys=True)
        digest = hashlib.sha256(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{digest}.json")

    def get_json(self, url: str, params: Optional[Dict[str, Any]], rate_limiter: RateLimiter, ttl_seconds: int = 300) -> Dict[str, Any]:
        path = self._cache_path(url, params)
        # read cache if fresh
        if os.path.exists(path):
            try:
                mtime = os.path.getmtime(path)
                if time.time() - mtime < ttl_seconds:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
            except Exception:
                pass
        # rate limit and fetch
        rate_limiter.acquire()
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass
        return data

