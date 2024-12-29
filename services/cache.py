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
from dataclasses import dataclass
from pathlib import Path
import pickle

@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    size_bytes: Optional[int] = None

    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at

    def access(self):
        self.access_count += 1
        self.last_accessed = datetime.now()

class LRUCache:
    """Memory-based LRU cache implementation."""
    
    def __init__(self, max_size: int, ttl: int = 300, max_memory_mb: int = 512):
        self.max_size = max_size
        self.ttl = ttl
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_bytes = 0
        self.cache: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with LRU tracking."""
        with self._lock:
            if key not in self.cache:
                self._stats['misses'] += 1
                return None

            entry = self.cache[key]
            if entry.is_expired():
                self._remove_entry(key)
                self._stats['misses'] += 1
                return None

            # Update LRU info
            entry.access()
            self._move_to_front(key)
            self._stats['hits'] += 1
            return entry.value

    def put(self, key: str, value: Any):
        """Add value to cache with memory management."""
        with self._lock:
            # Calculate entry size
            entry_size = self._estimate_size(value)

            # Check if we need to make space
            while (self.current_memory_bytes + entry_size > self.max_memory_bytes or 
                   len(self.cache) >= self.max_size):
                if not self._evict_one():
                    break

            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=self.ttl),
                size_bytes=entry_size
            )

            # Update cache
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_memory_bytes -= old_entry.size_bytes or 0
            
            self.cache[key] = entry
            self.current_memory_bytes += entry_size

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of a value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default estimation if unable to pickle

    def _move_to_front(self, key: str):
        """Move entry to front of LRU order."""
        entry = self.cache.pop(key)
        self.cache[key] = entry

    def _evict_one(self) -> bool:
        """Evict least recently used entry."""
        if not self.cache:
            return False

        # Find LRU entry
        lru_key = min(
            self.cache.keys(),
            key=lambda k: (
                self.cache[k].last_accessed or self.cache[k].created_at
            )
        )

        self._remove_entry(lru_key)
        self._stats['evictions'] += 1
        return True

    def _remove_entry(self, key: str):
        """Remove entry and update memory usage."""
        entry = self.cache.pop(key)
        self.current_memory_bytes -= entry.size_bytes or 0

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.current_memory_bytes = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            return {
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'hit_rate': self._stats['hits'] / total_requests if total_requests > 0 else 0,
                'size': len(self.cache),
                'memory_usage_mb': self.current_memory_bytes / (1024 * 1024)
            }

class DiskCache:
    """Persistent disk-based cache."""
    
    def __init__(self, cache_dir: Path, max_size_gb: float = 1.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self._lock = threading.Lock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }

    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        file_path = self._get_file_path(key)
        if not file_path.exists():
            self._stats['misses'] += 1
            return None

        try:
            with file_path.open('rb') as f:
                entry = pickle.load(f)
                if entry.is_expired():
                    self._remove_file(file_path)
                    self._stats['misses'] += 1
                    return None

                entry.access()
                self._save_entry(entry)
                self._stats['hits'] += 1
                return entry.value

        except Exception as e:
            logging.error(f"Error reading from disk cache: {str(e)}")
            self._stats['misses'] += 1
            return None

    def put(self, key: str, value: Any, ttl: int = 3600):
        """Store value in disk cache."""
        with self._lock:
            try:
                # Ensure we have space
                self._ensure_space_available()

                # Create entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(seconds=ttl)
                )

                self._save_entry(entry)

            except Exception as e:
                logging.error(f"Error writing to disk cache: {str(e)}")

    def _ensure_space_available(self):
        """Ensure cache directory is within size limits."""
        current_size = sum(f.stat().st_size for f in self.cache_dir.glob('*.cache'))
        
        if current_size > self.max_size_bytes:
            # Remove oldest files until we're under limit
            files = sorted(
                self.cache_dir.glob('*.cache'),
                key=lambda p: p.stat().st_mtime
            )
            
            for file in files:
                if current_size <= self.max_size_bytes:
                    break
                size = file.stat().st_size
                self._remove_file(file)
                current_size -= size
                self._stats['evictions'] += 1

    def _save_entry(self, entry: CacheEntry):
        """Save cache entry to disk."""
        file_path = self._get_file_path(entry.key)
        with file_path.open('wb') as f:
            pickle.dump(entry, f)

    def _remove_file(self, file_path: Path):
        """Remove cache file."""
        try:
            file_path.unlink()
        except Exception as e:
            logging.error(f"Error removing cache file: {str(e)}")

    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.cache"

    def clear(self):
        """Clear all cache files."""
        with self._lock:
            for file in self.cache_dir.glob('*.cache'):
                self._remove_file(file)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            current_size = sum(f.stat().st_size for f in self.cache_dir.glob('*.cache'))
            return {
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'hit_rate': self._stats['hits'] / total_requests if total_requests > 0 else 0,
                'size': len(list(self.cache_dir.glob('*.cache'))),
                'disk_usage_gb': current_size / (1024 * 1024 * 1024)
            }

class CacheManager:
    """Manages multi-level cache system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.l1_cache = LRUCache(
            max_size=config.get('l1_cache_size', 1000),
            ttl=config.get('l1_cache_ttl', 300),
            max_memory_mb=config.get('l1_cache_memory_mb', 512)
        )
        
        self.l2_cache = DiskCache(
            cache_dir=Path(config.get('cache_dir', 'cache')),
            max_size_gb=config.get('l2_cache_size_gb', 1.0)
        )
        
        self._stats = {
            'l1_hits': 0,
            'l2_hits': 0,
            'misses': 0
        }

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache hierarchy."""
        # Try L1 cache
        value = self.l1_cache.get(key)
        if value is not None:
            self._stats['l1_hits'] += 1
            return value

        # Try L2 cache
        value = self.l2_cache.get(key)
        if value is not None:
            self._stats['l2_hits'] += 1
            # Promote to L1
            self.l1_cache.put(key, value)
            return value

        self._stats['misses'] += 1
        return None

    def put(self, key: str, value: Any, ttl: Optional[int] = None):
        """Store value in cache hierarchy."""
        # Store in L1
        self.l1_cache.put(key, value)
        
        # Store in L2 with longer TTL
        l2_ttl = ttl or 3600  # Default 1 hour for L2
        self.l2_cache.put(key, value, l2_ttl)

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

    def clear(self):
        """Clear all cache levels."""
        self.l1_cache.clear()
        self.l2_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()
        
        total_requests = sum([
            self._stats['l1_hits'],
            self._stats['l2_hits'],
            self._stats['misses']
        ])
        
        return {
            'l1_cache': l1_stats,
            'l2_cache': l2_stats,
            'total_hits': self._stats['l1_hits'] + self._stats['l2_hits'],
            'total_misses': self._stats['misses'],
            'hit_rate': (self._stats['l1_hits'] + self._stats['l2_hits']) / total_requests if total_requests > 0 else 0,
            'l1_hit_rate': self._stats['l1_hits'] / total_requests if total_requests > 0 else 0,
            'l2_hit_rate': self._stats['l2_hits'] / total_requests if total_requests > 0 else 0
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
    'l1_cache_size': 1000,
    'l1_cache_ttl': 300,
    'l1_cache_memory_mb': 512,
    'cache_dir': 'cache',
    'l2_cache_size_gb': 1.0
})