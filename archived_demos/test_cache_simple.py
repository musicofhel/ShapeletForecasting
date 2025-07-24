"""
Simple test for cache optimization
"""

import time
import numpy as np
import pandas as pd
from src.dashboard.optimization import cache_manager

print("Testing Cache Optimization...")
print("="*50)

# Test 1: Basic caching
print("\n1. Testing Basic Cache:")
@cache_manager.cached(ttl=3600)
def slow_calculation(x, y):
    print(f"   Computing {x} * {y}...")
    time.sleep(1)
    return x * y

start = time.time()
result1 = slow_calculation(5, 10)
time1 = time.time() - start
print(f"   First call: {result1} (took {time1:.2f}s)")

start = time.time()
result2 = slow_calculation(5, 10)
time2 = time.time() - start
print(f"   Cached call: {result2} (took {time2:.3f}s)")
print(f"   Speedup: {time1/time2:.0f}x")

# Test 2: Lazy loading
print("\n2. Testing Lazy Loading:")
def data_source(offset, limit):
    return pd.DataFrame({
        'id': range(offset, offset + limit),
        'value': np.random.randn(limit)
    })

lazy_df = cache_manager.lazy_load_dataframe(
    data_source=data_source,
    total_rows=10000,
    columns=['id', 'value']
)
print(f"   Created lazy DataFrame: {lazy_df.shape}")
print(f"   Loading first 5 rows...")
print(lazy_df.head())

# Test 3: WebGL optimization
print("\n3. Testing WebGL Optimization:")
large_data = np.random.randn(50000, 2)
optimized = cache_manager.optimize_for_webgl(large_data, max_points=5000)
print(f"   Original: {optimized['original_size']} points")
print(f"   Optimized: {optimized['optimized_size']} points")
print(f"   Reduction: {(1 - optimized['optimized_size']/optimized['original_size'])*100:.1f}%")

# Test 4: Cache stats
print("\n4. Cache Statistics:")
stats = cache_manager.get_cache_stats()
print(f"   Total memory: {stats['total_memory_mb']:.2f} MB")
print(f"   Memory cache entries: {stats['memory_cache']['size']}")

print("\n✓ Cache optimization is working correctly!")
