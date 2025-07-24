"""
Dashboard Optimization Module

This module provides performance optimization features for the dashboard,
including caching, lazy loading, and WebGL optimization.
"""

from .cache_manager import (
    CacheManager,
    cache_manager,
    LRUCache,
    LazyDataFrame,
    CacheEntry
)

__all__ = [
    'CacheManager',
    'cache_manager',
    'LRUCache', 
    'LazyDataFrame',
    'CacheEntry'
]
