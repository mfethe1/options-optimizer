"""
Market Data Caching

Provides in-memory caching for market data with TTL (Time-To-Live).
"""

import logging
from typing import Any, Dict, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps
import json
import hashlib

logger = logging.getLogger(__name__)

# ============================================================================
# CACHE STORAGE
# ============================================================================

class CacheEntry:
    """
    A single cache entry with TTL.
    """
    def __init__(self, value: Any, ttl_seconds: int):
        self.value = value
        self.created_at = datetime.utcnow()
        self.expires_at = self.created_at + timedelta(seconds=ttl_seconds)
    
    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        return datetime.utcnow() >= self.expires_at
    
    def age_seconds(self) -> float:
        """Get the age of this cache entry in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()


class InMemoryCache:
    """
    Simple in-memory cache with TTL support.
    
    Thread-safe for basic operations.
    """
    def __init__(self):
        self._cache: Dict[str, CacheEntry] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        entry = self._cache.get(key)
        
        if entry is None:
            self._stats["misses"] += 1
            return None
        
        if entry.is_expired():
            # Remove expired entry
            del self._cache[key]
            self._stats["misses"] += 1
            self._stats["evictions"] += 1
            return None
        
        self._stats["hits"] += 1
        return entry.value
    
    def set(self, key: str, value: Any, ttl_seconds: int = 300):
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds (default: 5 minutes)
        """
        self._cache[key] = CacheEntry(value, ttl_seconds)
        self._stats["sets"] += 1
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was deleted, False if not found
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self):
        """Clear all cache entries."""
        count = len(self._cache)
        self._cache.clear()
        self._stats["evictions"] += count
        logger.info(f"Cache cleared: {count} entries removed")
    
    def clear_expired(self):
        """Remove all expired entries."""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            self._stats["evictions"] += len(expired_keys)
            logger.info(f"Cleared {len(expired_keys)} expired cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = (
            self._stats["hits"] / total_requests
            if total_requests > 0
            else 0.0
        )
        
        return {
            "size": len(self._cache),
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "sets": self._stats["sets"],
            "evictions": self._stats["evictions"],
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }
    
    def get_keys(self) -> list:
        """Get all cache keys."""
        return list(self._cache.keys())


# Global cache instance
_cache = InMemoryCache()


# ============================================================================
# CACHE DECORATORS
# ============================================================================

def cache_key(*args, **kwargs) -> str:
    """
    Generate a cache key from function arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Cache key string
    """
    # Create a stable representation of arguments
    key_data = {
        "args": args,
        "kwargs": sorted(kwargs.items())
    }
    
    # Hash the JSON representation
    key_json = json.dumps(key_data, sort_keys=True, default=str)
    key_hash = hashlib.md5(key_json.encode()).hexdigest()
    
    return key_hash


def cached(ttl_seconds: int = 300, key_prefix: str = ""):
    """
    Decorator to cache function results.
    
    Args:
        ttl_seconds: Time-to-live in seconds (default: 5 minutes)
        key_prefix: Prefix for cache keys
        
    Example:
        @cached(ttl_seconds=60, key_prefix="market_data")
        def get_market_data(symbol: str):
            # ... expensive operation ...
            return data
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            func_name = f"{func.__module__}.{func.__name__}"
            arg_key = cache_key(*args, **kwargs)
            full_key = f"{key_prefix}:{func_name}:{arg_key}" if key_prefix else f"{func_name}:{arg_key}"
            
            # Try to get from cache
            cached_value = _cache.get(full_key)
            if cached_value is not None:
                logger.debug(f"Cache hit: {full_key}")
                return cached_value
            
            # Call function and cache result
            logger.debug(f"Cache miss: {full_key}")
            result = func(*args, **kwargs)
            _cache.set(full_key, result, ttl_seconds)
            
            return result
        
        # Add cache control methods
        wrapper.cache_clear = lambda: _cache.clear()
        wrapper.cache_stats = lambda: _cache.get_stats()
        
        return wrapper
    
    return decorator


# ============================================================================
# CACHE MANAGEMENT FUNCTIONS
# ============================================================================

def get_cache() -> InMemoryCache:
    """Get the global cache instance."""
    return _cache


def clear_cache():
    """Clear all cache entries."""
    _cache.clear()


def clear_expired():
    """Remove all expired cache entries."""
    _cache.clear_expired()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return _cache.get_stats()


def invalidate_pattern(pattern: str):
    """
    Invalidate all cache keys matching a pattern.
    
    Args:
        pattern: Pattern to match (simple substring match)
    """
    keys_to_delete = [
        key for key in _cache.get_keys()
        if pattern in key
    ]
    
    for key in keys_to_delete:
        _cache.delete(key)
    
    logger.info(f"Invalidated {len(keys_to_delete)} cache entries matching pattern: {pattern}")


# ============================================================================
# MARKET DATA CACHE HELPERS
# ============================================================================

def cache_market_data(symbol: str, data: Any, ttl_seconds: int = 300):
    """
    Cache market data for a symbol.
    
    Args:
        symbol: Stock symbol
        data: Market data to cache
        ttl_seconds: Time-to-live in seconds (default: 5 minutes)
    """
    key = f"market_data:{symbol}"
    _cache.set(key, data, ttl_seconds)


def get_cached_market_data(symbol: str) -> Optional[Any]:
    """
    Get cached market data for a symbol.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Cached market data or None if not found
    """
    key = f"market_data:{symbol}"
    return _cache.get(key)


def invalidate_market_data(symbol: str):
    """
    Invalidate cached market data for a symbol.
    
    Args:
        symbol: Stock symbol
    """
    key = f"market_data:{symbol}"
    _cache.delete(key)
    logger.info(f"Invalidated market data cache for {symbol}")

