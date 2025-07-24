"""
Minimal test to check if cache manager imports correctly
"""

try:
    print("Importing cache manager...")
    from src.dashboard.optimization import cache_manager
    print("✓ Import successful!")
    
    print("\nTesting basic functionality...")
    
    # Test cache key generation
    key = cache_manager.cache_key("test", 1, 2, 3)
    print(f"✓ Cache key generated: {key[:8]}...")
    
    # Test cache stats
    stats = cache_manager.get_cache_stats()
    print(f"✓ Cache stats retrieved: {stats['total_memory_mb']:.2f} MB total")
    
    # Test WebGL optimization
    import numpy as np
    data = np.random.randn(1000, 2)
    result = cache_manager.optimize_for_webgl(data, max_points=100)
    print(f"✓ WebGL optimization: {result['original_size']} -> {result['optimized_size']} points")
    
    print("\n✅ All basic tests passed!")
    
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
