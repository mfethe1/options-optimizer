from __future__ import annotations
import os
import json
import time
from typing import Any, Optional

_DEFAULT_BASE = os.path.join("data", "cache", "analytics")


def _ns_dir(base: str, namespace: str) -> str:
    path = os.path.join(base, namespace)
    os.makedirs(path, exist_ok=True)
    return path


def _key_path(base: str, namespace: str, key: str) -> str:
    safe_key = key.replace(os.sep, "_")
    return os.path.join(_ns_dir(base, namespace), f"{safe_key}.json")


def load_json_cache(namespace: str, key: str, ttl_seconds: int, base_dir: str = _DEFAULT_BASE) -> Optional[Any]:
    """
    Load a cached JSON object if present and not expired.
    Returns None if missing or stale.
    """
    path = _key_path(base_dir, namespace, key)
    if not os.path.exists(path):
        return None
    try:
        mtime = os.path.getmtime(path)
        if (time.time() - mtime) > ttl_seconds:
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_json_cache(namespace: str, key: str, obj: Any, base_dir: str = _DEFAULT_BASE) -> None:
    """
    Persist a JSON object to cache.
    """
    path = _key_path(base_dir, namespace, key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)

