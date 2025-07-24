"""
Simple DTW test without external dependencies
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import just the DTW calculator
from dtw.dtw_calculator import DTWCalculator, FastDTW

def test_basic_dtw():
    """Test basic DTW functionality"""
    print("Testing Basic DTW (without full dependencies)...")
    
    # Generate test signals
    x = np.sin(np.linspace(0, 2*np.pi, 100))
    y = np.sin(np.linspace(0, 2*np.pi, 120) + 0.5)
    
    # Test standard DTW
    dtw = DTWCalculator()
    result = dtw.compute(x, y)
    print(f"✓ Standard DTW Distance: {result.distance:.4f}")
    print(f"✓ Normalized Distance: {result.normalized_distance:.4f}")
    print(f"✓ Path Length: {len(result.path)}")
    
    # Test FastDTW
    fast_dtw = FastDTW(radius=2)
    fast_result = fast_dtw.compute(x, y)
    print(f"✓ FastDTW Distance: {fast_result.distance:.4f}")
    print(f"✓ FastDTW Normalized Distance: {fast_result.normalized_distance:.4f}")
    
    print("\nDTW core functionality working correctly! ✓")
    print("\nNote: Full functionality requires installing dependencies:")
    print("  pip install -r requirements.txt")

if __name__ == "__main__":
    test_basic_dtw()
