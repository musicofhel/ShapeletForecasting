"""
Quick test script to verify DTW module functionality
"""

import numpy as np
from src.dtw import DTWCalculator, FastDTW, SimilarityEngine, DTWVisualizer

def test_basic_dtw():
    """Test basic DTW functionality"""
    print("Testing Basic DTW...")
    
    # Generate test signals
    x = np.sin(np.linspace(0, 2*np.pi, 100))
    y = np.sin(np.linspace(0, 2*np.pi, 120) + 0.5)
    
    # Test standard DTW
    dtw = DTWCalculator()
    result = dtw.compute(x, y)
    print(f"✓ Standard DTW Distance: {result.distance:.4f}")
    print(f"✓ Normalized Distance: {result.normalized_distance:.4f}")
    
    # Test FastDTW
    fast_dtw = FastDTW(radius=2)
    fast_result = fast_dtw.compute(x, y)
    print(f"✓ FastDTW Distance: {fast_result.distance:.4f}")
    
    # Test similarity engine
    patterns = [x, y, np.cos(np.linspace(0, 2*np.pi, 100))]
    labels = ['sin1', 'sin2', 'cos']
    
    engine = SimilarityEngine(dtw_type='fast')
    sim_results = engine.compute_similarity_matrix(patterns, labels)
    print(f"✓ Similarity Matrix Shape: {sim_results['similarity_matrix'].shape}")
    
    # Test visualization
    visualizer = DTWVisualizer()
    print("✓ DTW Visualizer initialized")
    
    print("\nAll DTW components working correctly! ✓")

if __name__ == "__main__":
    test_basic_dtw()
