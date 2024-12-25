# services/cache.py
from typing import Dict, Any, Optional, List, Union, Callable
import logging
from datetime import datetime, timedelta
import threading
from functools import wraps
import json
import hashlib
from collections import OrderedDict
import time

class CacheEntry:
    """Container for cached data with metadata."""
    def __init__(self, key: str, value: Any, ttl: int = 3600):
        self.key = key
        self.value = value
        self.created_at = datetime.now()
        self.expires_at = self.created_at + timedelta(seconds=ttl)
        self.access_count = 0
        self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() > self.expires_at

    def access(self):
        """Record cache entry access."""
        self.access_count += 1
        self.last_accessed = datetime.now()

class LRUCache:
    """LRU Cache implementation with size limit."""
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self.cache:
                return None
            
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value

    def put(self, key: str, value: Any):
        """Put value in cache."""
        with self._lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used item
                self.cache.popitem(last=False)
            self.cache[key] = value

class CacheManager:
    """Manages different types of caches and caching strategies."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize different cache types
        self.memory_cache = {}  # Simple in-memory cache
        self.lru_cache = LRUCache(
            max_size=config.get('max_cache_size', 1000)
        )
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
        # Cache locks for thread safety
        self._cache_locks = {}
        self._stats_lock = threading.Lock()
        
        # Start maintenance thread
        self._start_maintenance()

    def get(self, key: str, cache_type: str = 'memory') -> Optional[Any]:
        """Get value from specified cache."""
        if cache_type == 'lru':
            return self._get_lru(key)
        else:
            return self._get_memory(key)

    def put(self, key: str, value: Any, ttl: int = 3600,
            cache_type: str = 'memory'):
        """Put value in specified cache."""
        if cache_type == 'lru':
            self._put_lru(key, value)
        else:
            self._put_memory(key, value, ttl)

    def invalidate(self, key: str, cache_type: str = 'memory'):
        """Invalidate cache entry."""
        if cache_type == 'lru':
            with self._get_lock(key):
                if key in self.lru_cache.cache:
                    del self.lru_cache.cache[key]
        else:
            with self._get_lock(key):
                if key in self.memory_cache:
                    del self.memory_cache[key]

    def clear(self, cache_type: str = 'memory'):
        """Clear entire cache."""
        if cache_type == 'lru':
            self.lru_cache.cache.clear()
        else:
            self.memory_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._stats_lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (
                self.stats['hits'] / total_requests 
                if total_requests > 0 else 0
            )
            
            return {
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'hit_rate': hit_rate,
                'memory_cache_size': len(self.memory_cache),
                'lru_cache_size': len(self.lru_cache.cache)
            }

    def _get_memory(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        with self._get_lock(key):
            entry = self.memory_cache.get(key)
            if not entry:
                with self._stats_lock:
                    self.stats['misses'] += 1
                return None
                
            if entry.is_expired():
                del self.memory_cache[key]
                with self._stats_lock:
                    self.stats['evictions'] += 1
                    self.stats['misses'] += 1
                return None
                
            entry.access()
            with self._stats_lock:
                self.stats['hits'] += 1
            return entry.value

    def _put_memory(self, key: str, value: Any, ttl: int):
        """Put value in memory cache."""
        with self._get_lock(key):
            self.memory_cache[key] = CacheEntry(key, value, ttl)

    def _get_lru(self, key: str) -> Optional[Any]:
        """Get value from LRU cache."""
        value = self.lru_cache.get(key)
        with self._stats_lock:
            if value is None:
                self.stats['misses'] += 1
            else:
                self.stats['hits'] += 1
        return value

    def _put_lru(self, key: str, value: Any):
        """Put value in LRU cache."""
        self.lru_cache.put(key, value)

    def _get_lock(self, key: str) -> threading.Lock:
        """Get or create lock for cache key."""
        if key not in self._cache_locks:
            self._cache_locks[key] = threading.Lock()
        return self._cache_locks[key]

    def _start_maintenance(self):
        """Start cache maintenance thread."""
        def maintenance_task():
            while True:
                try:
                    self._cleanup_expired()
                    time.sleep(60)  # Run every minute
                except Exception as e:
                    self.logger.error(f"Cache maintenance error: {str(e)}")

        thread = threading.Thread(
            target=maintenance_task,
            daemon=True
        )
        thread.start()

    def _cleanup_expired(self):
        """Clean up expired cache entries."""
        with self._stats_lock:
            # Clean memory cache
            expired_keys = [
                key for key, entry in self.memory_cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                with self._get_lock(key):
                    if key in self.memory_cache:
                        del self.memory_cache[key]
                        self.stats['evictions'] += 1

def cache_result(ttl: int = 3600, cache_type: str = 'memory'):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = _generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            result = cache_manager.get(key, cache_type)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.put(key, result, ttl, cache_type)
            return result
        return wrapper
    return decorator

class QueryCache:
    """Cache for database query results."""
    
    def __init__(self, max_size: int = 1000):
        self.cache = LRUCache(max_size)
        self.stats = {
            'hits': 0,
            'misses': 0
        }
        self._lock = threading.Lock()

    def get_query_result(self, query: str, 
                        params: Dict[str, Any]) -> Optional[Any]:
        """Get cached query result."""
        key = self._generate_query_key(query, params)
        with self._lock:
            result = self.cache.get(key)
            if result is not None:
                self.stats['hits'] += 1
            else:
                self.stats['misses'] += 1
            return result

    def cache_query_result(self, query: str, params: Dict[str, Any],
                          result: Any):
        """Cache query result."""
        key = self._generate_query_key(query, params)
        self.cache.put(key, result)

    def _generate_query_key(self, query: str, params: Dict[str, Any]) -> str:
        """Generate cache key for query."""
        query_data = {
            'query': query,
            'params': params
        }
        return hashlib.md5(
            json.dumps(query_data, sort_keys=True).encode()
        ).hexdigest()

def _generate_cache_key(func_name: str, args: tuple,
                       kwargs: Dict[str, Any]) -> str:
    """Generate cache key from function call."""
    key_data = {
        'function': func_name,
        'args': args,
        'kwargs': kwargs
    }
    return hashlib.md5(
        json.dumps(key_data, sort_keys=True).encode()
    ).hexdigest()

# Initialize global cache manager
cache_manager = CacheManager({
    'max_cache_size': 1000,
    'default_ttl': 3600
})