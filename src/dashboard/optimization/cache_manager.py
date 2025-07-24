"""
Cache Manager for Dashboard Performance Optimization

This module implements a comprehensive caching strategy for the dashboard,
including:
- In-memory caching with TTL (Time To Live)
- Lazy loading for large datasets
- Progressive data loading
- WebGL optimization support
- Pattern discovery algorithm caching
"""

import time
import hashlib
import json
import pickle
import threading
from typing import Any, Dict, Optional, Callable, List, Tuple
from collections import OrderedDict
from functools import wraps
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheEntry:
    """Represents a single cache entry with metadata"""
    
    def __init__(self, key: str, value: Any, ttl: int = 3600):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0
        self.size_bytes = self._calculate_size(value)
    
    def _calculate_size(self, obj: Any) -> int:
        """Calculate approximate size of object in bytes"""
        try:
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.memory_usage(deep=True).sum()
            elif isinstance(obj, np.ndarray):
                return obj.nbytes
            elif isinstance(obj, (dict, list)):
                return len(pickle.dumps(obj))
            else:
                return len(str(obj).encode('utf-8'))
        except:
            return 0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() - self.created_at > self.ttl
    
    def access(self):
        """Update access metadata"""
        self.last_accessed = time.time()
        self.access_count += 1


class LRUCache:
    """Least Recently Used (LRU) cache implementation"""
    
    def __init__(self, max_size: int = 100, max_memory_mb: int = 500):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.total_memory_bytes = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if not entry.is_expired():
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    entry.access()
                    return entry.value
                else:
                    # Remove expired entry
                    self._remove(key)
            return None
    
    def put(self, key: str, value: Any, ttl: int = 3600):
        """Put value in cache"""
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                self._remove(key)
            
            # Create new entry
            entry = CacheEntry(key, value, ttl)
            
            # Check memory limit
            while (self.total_memory_bytes + entry.size_bytes > self.max_memory_bytes 
                   and len(self.cache) > 0):
                # Remove least recently used
                self._remove(next(iter(self.cache)))
            
            # Check size limit
            while len(self.cache) >= self.max_size:
                # Remove least recently used
                self._remove(next(iter(self.cache)))
            
            # Add new entry
            self.cache[key] = entry
            self.total_memory_bytes += entry.size_bytes
    
    def _remove(self, key: str):
        """Remove entry from cache"""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.total_memory_bytes -= entry.size_bytes
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.total_memory_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                'size': len(self.cache),
                'memory_mb': self.total_memory_bytes / (1024 * 1024),
                'hit_rate': self._calculate_hit_rate(),
                'entries': [
                    {
                        'key': k,
                        'size_mb': v.size_bytes / (1024 * 1024),
                        'age_seconds': time.time() - v.created_at,
                        'access_count': v.access_count
                    }
                    for k, v in list(self.cache.items())[:10]  # Top 10
                ]
            }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_accesses = sum(e.access_count for e in self.cache.values())
        return min(1.0, total_accesses / max(1, len(self.cache)))


class CacheManager:
    """Main cache manager for dashboard optimization"""
    
    def __init__(self):
        # Different cache levels
        self.memory_cache = LRUCache(max_size=100, max_memory_mb=500)
        self.pattern_cache = LRUCache(max_size=50, max_memory_mb=200)
        self.visualization_cache = LRUCache(max_size=30, max_memory_mb=100)
        
        # Lazy loading configuration
        self.lazy_load_threshold = 10000  # rows
        self.chunk_size = 1000
        
        # WebGL optimization flags
        self.webgl_enabled = True
        self.webgl_threshold = 5000  # points
        
        # Background tasks
        self._cleanup_task = None
        self._preload_task = None
    
    def cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def cached(self, ttl: int = 3600, cache_type: str = 'memory'):
        """Decorator for caching function results"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                key = self.cache_key(func.__name__, *args, **kwargs)
                
                # Select cache
                cache = getattr(self, f'{cache_type}_cache', self.memory_cache)
                
                # Check cache
                result = cache.get(key)
                if result is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
                
                # Execute function
                logger.debug(f"Cache miss for {func.__name__}")
                result = func(*args, **kwargs)
                
                # Store in cache
                cache.put(key, result, ttl)
                
                return result
            
            return wrapper
        return decorator
    
    def lazy_load_dataframe(self, 
                           data_source: Callable,
                           total_rows: int,
                           columns: List[str] = None) -> 'LazyDataFrame':
        """Create a lazy-loading DataFrame wrapper"""
        return LazyDataFrame(
            data_source=data_source,
            total_rows=total_rows,
            columns=columns,
            chunk_size=self.chunk_size,
            cache_manager=self
        )
    
    def optimize_for_webgl(self, data: np.ndarray, max_points: int = None) -> Dict[str, Any]:
        """Optimize data for WebGL rendering"""
        if not self.webgl_enabled:
            return {'data': data, 'optimized': False}
        
        max_points = max_points or self.webgl_threshold
        
        if len(data) <= max_points:
            return {'data': data, 'optimized': False}
        
        # Downsample for WebGL
        step = len(data) // max_points
        optimized_data = data[::step]
        
        # Calculate LOD (Level of Detail) levels
        lod_levels = []
        current_data = data
        level = 0
        
        while len(current_data) > max_points and level < 5:
            step = 2 ** (level + 1)
            lod_data = data[::step]
            lod_levels.append({
                'level': level,
                'data': lod_data,
                'points': len(lod_data)
            })
            current_data = lod_data
            level += 1
        
        return {
            'data': optimized_data,
            'optimized': True,
            'original_size': len(data),
            'optimized_size': len(optimized_data),
            'lod_levels': lod_levels
        }
    
    def cache_pattern_discovery(self, 
                               data: pd.DataFrame,
                               pattern_type: str,
                               parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Cache pattern discovery results"""
        # Generate cache key
        key = self.cache_key('pattern_discovery', pattern_type, parameters)
        
        # Check cache
        result = self.pattern_cache.get(key)
        if result is not None:
            logger.debug(f"Cache hit for pattern discovery: {pattern_type}")
            return result
        
        # This would call the actual pattern discovery algorithm
        # For now, return placeholder
        logger.info(f"Caching pattern discovery: {pattern_type}")
        result = []
        
        # Store in cache
        self.pattern_cache.put(key, result, 7200)
        
        return result
    
    def progressive_load(self, 
                        data_source: Callable,
                        callback: Callable,
                        chunk_size: int = None):
        """Load data progressively with callbacks"""
        chunk_size = chunk_size or self.chunk_size
        
        async def load_chunks():
            offset = 0
            while True:
                # Load chunk
                chunk = await asyncio.to_thread(
                    data_source, 
                    offset=offset, 
                    limit=chunk_size
                )
                
                if chunk is None or len(chunk) == 0:
                    break
                
                # Process chunk
                await callback(chunk, offset)
                
                offset += len(chunk)
                
                # Small delay to prevent blocking
                await asyncio.sleep(0.01)
        
        # Run in background
        asyncio.create_task(load_chunks())
    
    def preload_common_data(self, data_configs: List[Dict[str, Any]]):
        """Preload commonly used data in background"""
        async def preload():
            for config in data_configs:
                try:
                    # Generate cache key
                    key = self.cache_key(**config)
                    
                    # Check if already cached
                    if self.memory_cache.get(key) is not None:
                        continue
                    
                    # Load data
                    data_func = config.get('function')
                    if data_func:
                        result = await asyncio.to_thread(
                            data_func,
                            *config.get('args', []),
                            **config.get('kwargs', {})
                        )
                        
                        # Cache result
                        self.memory_cache.put(
                            key, 
                            result, 
                            config.get('ttl', 3600)
                        )
                        
                        logger.info(f"Preloaded data for {config.get('name', 'unknown')}")
                
                except Exception as e:
                    logger.error(f"Error preloading data: {e}")
                
                # Delay between preloads
                await asyncio.sleep(1)
        
        # Run in background
        self._preload_task = asyncio.create_task(preload())
    
    def start_cleanup_task(self, interval: int = 300):
        """Start background cleanup task"""
        async def cleanup():
            while True:
                try:
                    # Clean expired entries
                    for cache in [self.memory_cache, self.pattern_cache, self.visualization_cache]:
                        expired_keys = []
                        with cache.lock:
                            for key, entry in cache.cache.items():
                                if entry.is_expired():
                                    expired_keys.append(key)
                        
                        for key in expired_keys:
                            cache._remove(key)
                        
                        if expired_keys:
                            logger.info(f"Cleaned {len(expired_keys)} expired entries")
                
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
                
                await asyncio.sleep(interval)
        
        self._cleanup_task = asyncio.create_task(cleanup())
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            'memory_cache': self.memory_cache.get_stats(),
            'pattern_cache': self.pattern_cache.get_stats(),
            'visualization_cache': self.visualization_cache.get_stats(),
            'total_memory_mb': sum(
                cache.total_memory_bytes / (1024 * 1024)
                for cache in [self.memory_cache, self.pattern_cache, self.visualization_cache]
            )
        }
    
    def clear_all_caches(self):
        """Clear all caches"""
        self.memory_cache.clear()
        self.pattern_cache.clear()
        self.visualization_cache.clear()
        logger.info("All caches cleared")


class LazyDataFrame:
    """Lazy-loading DataFrame wrapper for large datasets"""
    
    def __init__(self, 
                 data_source: Callable,
                 total_rows: int,
                 columns: List[str] = None,
                 chunk_size: int = 1000,
                 cache_manager: CacheManager = None):
        self.data_source = data_source
        self.total_rows = total_rows
        self.columns = columns
        self.chunk_size = chunk_size
        self.cache_manager = cache_manager
        
        # Loaded chunks cache
        self._chunks: Dict[int, pd.DataFrame] = {}
        self._lock = threading.RLock()
    
    def _get_chunk_index(self, row: int) -> int:
        """Get chunk index for a given row"""
        return row // self.chunk_size
    
    def _load_chunk(self, chunk_idx: int) -> pd.DataFrame:
        """Load a specific chunk"""
        with self._lock:
            # Check if already loaded
            if chunk_idx in self._chunks:
                return self._chunks[chunk_idx]
            
            # Calculate offset and limit
            offset = chunk_idx * self.chunk_size
            limit = min(self.chunk_size, self.total_rows - offset)
            
            # Load from data source
            chunk_data = self.data_source(offset=offset, limit=limit)
            
            # Cache chunk
            self._chunks[chunk_idx] = chunk_data
            
            # Limit cache size
            if len(self._chunks) > 10:
                # Remove oldest chunks
                oldest_idx = min(self._chunks.keys())
                del self._chunks[oldest_idx]
            
            return chunk_data
    
    def iloc(self, rows: slice, cols: slice = None) -> pd.DataFrame:
        """Get data by integer location (lazy loading)"""
        start_row = rows.start or 0
        stop_row = rows.stop or self.total_rows
        
        # Determine required chunks
        start_chunk = self._get_chunk_index(start_row)
        end_chunk = self._get_chunk_index(stop_row - 1)
        
        # Load required chunks
        chunks = []
        for chunk_idx in range(start_chunk, end_chunk + 1):
            chunk = self._load_chunk(chunk_idx)
            
            # Calculate relative positions within chunk
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = chunk_start + len(chunk)
            
            # Slice within chunk
            rel_start = max(0, start_row - chunk_start)
            rel_end = min(len(chunk), stop_row - chunk_start)
            
            if rel_start < rel_end:
                chunk_slice = chunk.iloc[rel_start:rel_end]
                if cols is not None:
                    chunk_slice = chunk_slice.iloc[:, cols]
                chunks.append(chunk_slice)
        
        # Combine chunks
        if chunks:
            return pd.concat(chunks, ignore_index=True)
        else:
            return pd.DataFrame(columns=self.columns)
    
    def head(self, n: int = 5) -> pd.DataFrame:
        """Get first n rows"""
        return self.iloc(slice(0, n))
    
    def tail(self, n: int = 5) -> pd.DataFrame:
        """Get last n rows"""
        return self.iloc(slice(self.total_rows - n, self.total_rows))
    
    def sample(self, n: int = 5, random_state: int = None) -> pd.DataFrame:
        """Get random sample of rows"""
        if random_state is not None:
            np.random.seed(random_state)
        
        # Generate random indices
        indices = np.random.choice(self.total_rows, size=min(n, self.total_rows), replace=False)
        indices.sort()
        
        # Load samples
        samples = []
        for idx in indices:
            chunk_idx = self._get_chunk_index(idx)
            chunk = self._load_chunk(chunk_idx)
            
            # Get relative position
            rel_idx = idx - (chunk_idx * self.chunk_size)
            if rel_idx < len(chunk):
                samples.append(chunk.iloc[rel_idx:rel_idx+1])
        
        if samples:
            return pd.concat(samples, ignore_index=True)
        else:
            return pd.DataFrame(columns=self.columns)
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get shape of full dataset"""
        n_cols = len(self.columns) if self.columns else 0
        return (self.total_rows, n_cols)
    
    def __len__(self) -> int:
        """Get number of rows"""
        return self.total_rows
    
    def __repr__(self) -> str:
        """String representation"""
        return f"LazyDataFrame(rows={self.total_rows}, columns={self.columns}, chunks_loaded={len(self._chunks)})"


# Global cache manager instance
cache_manager = CacheManager()


# Example usage functions
def example_cached_function():
    """Example of using the cache decorator"""
    
    @cache_manager.cached(ttl=3600)
    def expensive_calculation(x: int, y: int) -> int:
        """Simulate expensive calculation"""
        time.sleep(1)  # Simulate work
        return x * y + x ** y
    
    # First call - will be slow
    result1 = expensive_calculation(5, 3)
    
    # Second call - will be fast (cached)
    result2 = expensive_calculation(5, 3)
    
    return result1, result2


def example_lazy_loading():
    """Example of lazy loading large dataset"""
    
    # Simulate data source
    def data_source(offset: int, limit: int) -> pd.DataFrame:
        """Simulate loading data from database"""
        data = {
            'timestamp': pd.date_range(
                start='2024-01-01', 
                periods=limit, 
                freq='1min'
            ) + pd.Timedelta(minutes=offset),
            'value': np.random.randn(limit),
            'volume': np.random.randint(100, 1000, limit)
        }
        return pd.DataFrame(data)
    
    # Create lazy DataFrame
    lazy_df = cache_manager.lazy_load_dataframe(
        data_source=data_source,
        total_rows=1000000,  # 1 million rows
        columns=['timestamp', 'value', 'volume']
    )
    
    # Access data - only loads what's needed
    head = lazy_df.head(10)
    sample = lazy_df.sample(100)
    
    return lazy_df


def example_webgl_optimization():
    """Example of WebGL optimization"""
    
    # Large dataset
    large_data = np.random.randn(100000, 2)
    
    # Optimize for WebGL
    optimized = cache_manager.optimize_for_webgl(
        large_data,
        max_points=5000
    )
    
    print(f"Original size: {optimized['original_size']}")
    print(f"Optimized size: {optimized['optimized_size']}")
    print(f"LOD levels: {len(optimized['lod_levels'])}")
    
    return optimized


if __name__ == "__main__":
    # Start background tasks
    asyncio.create_task(cache_manager.start_cleanup_task())
    
    # Example usage
    print("Cache Manager initialized")
    print(f"Cache stats: {cache_manager.get_cache_stats()}")
    
    # Test caching
    print("\nTesting caching...")
    r1, r2 = example_cached_function()
    print(f"Results: {r1}, {r2}")
    
    # Test lazy loading
    print("\nTesting lazy loading...")
    lazy_df = example_lazy_loading()
    print(f"Lazy DataFrame: {lazy_df}")
    
    # Test WebGL optimization
    print("\nTesting WebGL optimization...")
    optimized = example_webgl_optimization()
