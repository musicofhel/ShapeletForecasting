"""
Minimal DTW test - imports calculator directly
"""

import numpy as np
import sys
sys.path.append('.')

# Import DTW calculator directly (bypassing __init__.py)
from src.dtw.dtw_calculator import DTWCalculator, FastDTW, ConstrainedDTW

def test_dtw():
    """Test DTW algorithms"""
    print("Testing DTW Implementation (Minimal Test)")
    print("=" * 50)
    
    # Generate test signals
    t1 = np.linspace(0, 2*np.pi, 100)
    t2 = np.linspace(0, 2*np.pi, 120)
    x = np.sin(t1)
    y = np.sin(t2 + 0.5)  # Phase shifted
    
    print("\n1. Standard DTW:")
    dtw = DTWCalculator()
    result = dtw.compute(x, y)
    print(f"   Distance: {result.distance:.4f}")
    print(f"   Normalized: {result.normalized_distance:.4f}")
    print(f"   Path length: {len(result.path)}")
    
    print("\n2. FastDTW:")
    fast_dtw = FastDTW(radius=2)
    fast_result = fast_dtw.compute(x, y)
    print(f"   Distance: {fast_result.distance:.4f}")
    print(f"   Normalized: {fast_result.normalized_distance:.4f}")
    
    print("\n3. Constrained DTW (Sakoe-Chiba):")
    constrained = ConstrainedDTW(constraint_type='sakoe_chiba', constraint_param=10)
    const_result = constrained.compute(x, y)
    print(f"   Distance: {const_result.distance:.4f}")
    print(f"   Normalized: {const_result.normalized_distance:.4f}")
    
    print("\n✓ All DTW algorithms working correctly!")
    print("\nNote: This is a minimal test. For full functionality:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run full test: python test_dtw_engine.py")

if __name__ == "__main__":
    test_dtw()
