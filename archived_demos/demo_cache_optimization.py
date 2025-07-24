"""
Demo: Cache Manager and Dashboard Optimization

This demo showcases the cache manager's capabilities for optimizing
dashboard performance through caching, lazy loading, and WebGL optimization.
"""

import asyncio
import time
import numpy as np
import pandas as pd
from src.dashboard.optimization import cache_manager, LazyDataFrame
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def demo_basic_caching():
    """Demonstrate basic caching functionality"""
    print("\n" + "="*60)
    print("DEMO 1: Basic Caching")
    print("="*60)
    
    # Define expensive function with caching
    @cache_manager.cached(ttl=3600)
    def expensive_pattern_analysis(ticker: str, period: int) -> dict:
        """Simulate expensive pattern analysis"""
        print(f"  Computing patterns for {ticker} over {period} days...")
        time.sleep(2)  # Simulate expensive computation
        
        return {
            'ticker': ticker,
            'period': period,
            'patterns_found': np.random.randint(5, 20),
            'confidence': np.random.uniform(0.7, 0.95),
            'computation_time': 2.0
        }
    
    # First call - will be slow
    start = time.time()
    result1 = expensive_pattern_analysis('AAPL', 30)
    time1 = time.time() - start
    print(f"  First call took: {time1:.2f} seconds")
    print(f"  Result: {result1}")
    
    # Second call - will be fast (cached)
    start = time.time()
    result2 = expensive_pattern_analysis('AAPL', 30)
    time2 = time.time() - start
    print(f"  Second call took: {time2:.2f} seconds (cached)")
    if time2 > 0:
        print(f"  Speedup: {time1/time2:.1f}x")
    else:
        print(f"  Speedup: >1000x (instant from cache)")
    
    # Different parameters - will be slow again
    start = time.time()
    result3 = expensive_pattern_analysis('GOOGL', 30)
    time3 = time.time() - start
    print(f"  Different ticker took: {time3:.2f} seconds")
    
    # Show cache stats
    stats = cache_manager.get_cache_stats()
    print(f"\n  Cache Stats:")
    print(f"    Memory used: {stats['total_memory_mb']:.2f} MB")
    print(f"    Entries: {stats['memory_cache']['size']}")


def demo_lazy_loading():
    """Demonstrate lazy loading for large datasets"""
    print("\n" + "="*60)
    print("DEMO 2: Lazy Loading Large Datasets")
    print("="*60)
    
    # Simulate large dataset source
    def load_market_data(offset: int, limit: int) -> pd.DataFrame:
        """Simulate loading market data from database"""
        # In real scenario, this would query a database
        dates = pd.date_range(
            start='2020-01-01',
            periods=limit,
            freq='1min'
        ) + pd.Timedelta(minutes=offset)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': 100 + np.random.randn(limit).cumsum() * 0.1,
            'high': 100 + np.random.randn(limit).cumsum() * 0.1 + 0.5,
            'low': 100 + np.random.randn(limit).cumsum() * 0.1 - 0.5,
            'close': 100 + np.random.randn(limit).cumsum() * 0.1,
            'volume': np.random.randint(1000, 10000, limit)
        })
        
        return data
    
    # Create lazy DataFrame for 1 million rows
    print("  Creating lazy DataFrame for 1,000,000 rows...")
    lazy_df = cache_manager.lazy_load_dataframe(
        data_source=load_market_data,
        total_rows=1000000,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    
    print(f"  LazyDataFrame created: {lazy_df}")
    print(f"  Shape: {lazy_df.shape}")
    
    # Access only what we need
    print("\n  Accessing first 10 rows...")
    start = time.time()
    head = lazy_df.head(10)
    print(f"  Time taken: {time.time() - start:.3f} seconds")
    print(f"  Data shape: {head.shape}")
    print(head)
    
    print("\n  Accessing random sample of 100 rows...")
    start = time.time()
    sample = lazy_df.sample(100, random_state=42)
    print(f"  Time taken: {time.time() - start:.3f} seconds")
    print(f"  Sample stats:")
    print(f"    Mean close: {sample['close'].mean():.2f}")
    print(f"    Volume range: {sample['volume'].min()} - {sample['volume'].max()}")
    
    print("\n  Accessing specific slice (rows 50000-50100)...")
    start = time.time()
    slice_data = lazy_df.iloc(slice(50000, 50100))
    print(f"  Time taken: {time.time() - start:.3f} seconds")
    print(f"  Slice shape: {slice_data.shape}")


def demo_webgl_optimization():
    """Demonstrate WebGL optimization for large visualizations"""
    print("\n" + "="*60)
    print("DEMO 3: WebGL Optimization for Large Visualizations")
    print("="*60)
    
    # Generate large dataset
    n_points = 100000
    print(f"  Generating dataset with {n_points:,} points...")
    
    # Simulate high-frequency time series data
    timestamps = pd.date_range('2024-01-01', periods=n_points, freq='1s')
    values = np.cumsum(np.random.randn(n_points)) * 0.01 + 100
    
    data = np.column_stack([np.arange(n_points), values])
    
    # Optimize for WebGL
    print("\n  Optimizing for WebGL rendering...")
    optimized = cache_manager.optimize_for_webgl(data, max_points=5000)
    
    print(f"  Original size: {optimized['original_size']:,} points")
    print(f"  Optimized size: {optimized['optimized_size']:,} points")
    print(f"  Reduction: {(1 - optimized['optimized_size']/optimized['original_size'])*100:.1f}%")
    print(f"  LOD levels: {len(optimized['lod_levels'])}")
    
    for i, lod in enumerate(optimized['lod_levels']):
        print(f"    Level {i}: {lod['points']:,} points")
    
    # Visualize the optimization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('WebGL Optimization Demonstration')
    
    # Original data (subset for visualization)
    ax = axes[0, 0]
    subset = data[:10000]  # Show only first 10k points
    ax.plot(subset[:, 0], subset[:, 1], 'b-', linewidth=0.5, alpha=0.7)
    ax.set_title(f'Original Data (first 10k of {n_points:,} points)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    
    # Optimized data
    ax = axes[0, 1]
    opt_data = optimized['data']
    ax.plot(opt_data[:, 0], opt_data[:, 1], 'r-', linewidth=1, alpha=0.8)
    ax.set_title(f'Optimized for WebGL ({len(opt_data):,} points)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    
    # LOD Level 0
    if optimized['lod_levels']:
        ax = axes[1, 0]
        lod0 = optimized['lod_levels'][0]['data']
        ax.plot(lod0[:, 0], lod0[:, 1], 'g-', linewidth=1.5, alpha=0.8)
        ax.set_title(f'LOD Level 0 ({len(lod0):,} points)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
    
    # Performance comparison
    ax = axes[1, 1]
    sizes = [optimized['original_size'], optimized['optimized_size']]
    sizes.extend([lod['points'] for lod in optimized['lod_levels'][:3]])
    labels = ['Original', 'Optimized'] + [f'LOD {i}' for i in range(min(3, len(optimized['lod_levels'])))]
    
    ax.bar(labels, sizes)
    ax.set_title('Data Points Comparison')
    ax.set_ylabel('Number of Points')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('webgl_optimization_demo.png', dpi=150, bbox_inches='tight')
    print("\n  Visualization saved to: webgl_optimization_demo.png")


def demo_pattern_cache():
    """Demonstrate pattern discovery caching"""
    print("\n" + "="*60)
    print("DEMO 4: Pattern Discovery Caching")
    print("="*60)
    
    # Simulate pattern discovery function
    @cache_manager.cached(ttl=7200, cache_type='pattern')
    def discover_patterns(data: pd.DataFrame, pattern_type: str, min_confidence: float) -> list:
        """Simulate pattern discovery algorithm"""
        print(f"  Discovering {pattern_type} patterns (min confidence: {min_confidence})...")
        time.sleep(3)  # Simulate expensive computation
        
        # Generate mock patterns
        patterns = []
        n_patterns = np.random.randint(3, 8)
        
        for i in range(n_patterns):
            patterns.append({
                'id': f'{pattern_type}_{i}',
                'type': pattern_type,
                'confidence': np.random.uniform(min_confidence, 1.0),
                'start_idx': np.random.randint(0, len(data) - 100),
                'length': np.random.randint(50, 200),
                'metrics': {
                    'strength': np.random.uniform(0.5, 1.0),
                    'frequency': np.random.randint(1, 10)
                }
            })
        
        return patterns
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1h')
    data = pd.DataFrame({
        'timestamp': dates,
        'price': 100 + np.cumsum(np.random.randn(1000)) * 0.5
    })
    
    # First discovery - slow
    start = time.time()
    patterns1 = discover_patterns(data, 'head_and_shoulders', 0.8)
    time1 = time.time() - start
    print(f"  Found {len(patterns1)} patterns in {time1:.2f} seconds")
    
    # Second discovery - fast (cached)
    start = time.time()
    patterns2 = discover_patterns(data, 'head_and_shoulders', 0.8)
    time2 = time.time() - start
    print(f"  Retrieved {len(patterns2)} patterns in {time2:.3f} seconds (cached)")
    print(f"  Cache speedup: {time1/time2:.0f}x")
    
    # Different parameters - slow again
    start = time.time()
    patterns3 = discover_patterns(data, 'double_top', 0.7)
    time3 = time.time() - start
    print(f"  Found {len(patterns3)} different patterns in {time3:.2f} seconds")
    
    # Show pattern cache stats
    stats = cache_manager.get_cache_stats()
    print(f"\n  Pattern Cache Stats:")
    print(f"    Entries: {stats['pattern_cache']['size']}")
    print(f"    Memory: {stats['pattern_cache']['memory_mb']:.2f} MB")
    print(f"    Hit rate: {stats['pattern_cache']['hit_rate']:.1%}")


async def demo_progressive_loading():
    """Demonstrate progressive data loading"""
    print("\n" + "="*60)
    print("DEMO 5: Progressive Data Loading")
    print("="*60)
    
    # Track loading progress
    loaded_chunks = []
    total_rows = 0
    
    async def process_chunk(chunk: pd.DataFrame, offset: int):
        """Process each chunk as it loads"""
        nonlocal total_rows
        loaded_chunks.append(len(chunk))
        total_rows += len(chunk)
        
        # Simulate processing
        await asyncio.sleep(0.1)
        
        # Show progress every 10 chunks
        if len(loaded_chunks) % 10 == 0:
            print(f"    Loaded {len(loaded_chunks)} chunks, {total_rows:,} rows total")
    
    # Data source that returns chunks
    def chunk_data_source(offset: int, limit: int) -> pd.DataFrame:
        """Simulate chunked data source"""
        if offset >= 50000:  # Total of 50k rows
            return None
        
        actual_limit = min(limit, 50000 - offset)
        return pd.DataFrame({
            'id': range(offset, offset + actual_limit),
            'value': np.random.randn(actual_limit)
        })
    
    print("  Starting progressive load of 50,000 rows...")
    
    # Start progressive loading
    cache_manager.progressive_load(
        data_source=chunk_data_source,
        callback=process_chunk,
        chunk_size=1000
    )
    
    # Wait for loading to complete
    await asyncio.sleep(6)
    
    print(f"\n  Progressive loading complete!")
    print(f"  Total chunks: {len(loaded_chunks)}")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Average chunk size: {np.mean(loaded_chunks):.0f} rows")


def demo_cache_management():
    """Demonstrate cache management and cleanup"""
    print("\n" + "="*60)
    print("DEMO 6: Cache Management")
    print("="*60)
    
    # Fill caches with data
    print("  Filling caches with test data...")
    
    # Add to memory cache
    for i in range(20):
        key = f'memory_test_{i}'
        data = np.random.randn(1000, 10)
        cache_manager.memory_cache.put(key, data, ttl=300)
    
    # Add to pattern cache
    for i in range(10):
        key = f'pattern_test_{i}'
        patterns = [{'id': f'p{j}', 'confidence': 0.8} for j in range(5)]
        cache_manager.pattern_cache.put(key, patterns, ttl=600)
    
    # Add to visualization cache
    for i in range(5):
        key = f'viz_test_{i}'
        viz_data = {'data': np.random.randn(5000), 'config': {}}
        cache_manager.visualization_cache.put(key, viz_data, ttl=180)
    
    # Show current stats
    stats = cache_manager.get_cache_stats()
    print("\n  Current Cache Statistics:")
    print(f"    Total memory: {stats['total_memory_mb']:.2f} MB")
    print(f"    Memory cache: {stats['memory_cache']['size']} entries, {stats['memory_cache']['memory_mb']:.2f} MB")
    print(f"    Pattern cache: {stats['pattern_cache']['size']} entries, {stats['pattern_cache']['memory_mb']:.2f} MB")
    print(f"    Viz cache: {stats['visualization_cache']['size']} entries, {stats['visualization_cache']['memory_mb']:.2f} MB")
    
    # Test cache eviction
    print("\n  Adding large entry to trigger eviction...")
    large_data = np.random.randn(10000, 100)  # ~7.6 MB
    cache_manager.memory_cache.put('large_entry', large_data, ttl=300)
    
    stats_after = cache_manager.get_cache_stats()
    print(f"    Memory cache after eviction: {stats_after['memory_cache']['size']} entries")
    
    # Clear specific cache
    print("\n  Clearing visualization cache...")
    cache_manager.visualization_cache.clear()
    
    stats_cleared = cache_manager.get_cache_stats()
    print(f"    Viz cache after clear: {stats_cleared['visualization_cache']['size']} entries")
    
    # Clear all caches
    print("\n  Clearing all caches...")
    cache_manager.clear_all_caches()
    
    final_stats = cache_manager.get_cache_stats()
    print(f"    Total memory after clear: {final_stats['total_memory_mb']:.2f} MB")


async def main():
    """Run all demos"""
    print("\n" + "="*80)
    print("DASHBOARD CACHE OPTIMIZATION DEMO")
    print("="*80)
    
    # Run synchronous demos
    demo_basic_caching()
    demo_lazy_loading()
    demo_webgl_optimization()
    demo_pattern_cache()
    
    # Run async demo
    await demo_progressive_loading()
    
    # Cache management
    demo_cache_management()
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nThe cache manager provides:")
    print("  ✓ Multi-level caching (memory, patterns, visualizations)")
    print("  ✓ LRU eviction with memory limits")
    print("  ✓ Lazy loading for large datasets")
    print("  ✓ WebGL optimization with LOD")
    print("  ✓ Progressive data loading")
    print("  ✓ Automatic cache cleanup")
    print("  ✓ Comprehensive statistics")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
