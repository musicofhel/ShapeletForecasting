"""
Demo script for Pattern Feature Extraction System
"""

import numpy as np
import matplotlib.pyplot as plt
from src.dashboard.pattern_features import PatternFeatureExtractor, FastPatternFeatureExtractor
import time


def generate_test_patterns():
    """Generate various test patterns"""
    t = np.linspace(0, 4*np.pi, 200)
    
    patterns = {
        'sine_wave': np.sin(t),
        'complex_wave': np.sin(t) + 0.5*np.sin(3*t) + 0.2*np.sin(5*t),
        'gaussian_pulse': np.exp(-((t - 2*np.pi)**2) / 2),
        'square_wave': np.sign(np.sin(t)),
        'noisy_sine': np.sin(t) + 0.3*np.random.randn(len(t))
    }
    
    return patterns


def demo_feature_extraction():
    """Demonstrate feature extraction capabilities"""
    print("=" * 80)
    print("PATTERN FEATURE EXTRACTION DEMO")
    print("=" * 80)
    
    # Generate patterns
    patterns = generate_test_patterns()
    
    # Initialize extractor
    extractor = PatternFeatureExtractor(wavelet='db4', normalize=True)
    
    # Extract features for each pattern
    print("\n1. EXTRACTING FEATURES FROM DIFFERENT PATTERNS")
    print("-" * 50)
    
    for name, pattern in patterns.items():
        features = extractor.extract_features(pattern)
        print(f"\n{name.upper()}:")
        print(f"  - Duration: {features.duration}")
        print(f"  - Amplitude range: {features.amplitude_range:.3f}")
        print(f"  - Number of peaks: {features.num_peaks}")
        print(f"  - Dominant frequency: {features.dominant_frequency:.3f}")
        print(f"  - Energy concentration: {features.energy_concentration:.3f}")
        print(f"  - Symmetry: {features.symmetry:.3f}")
    
    # Demonstrate similarity calculation
    print("\n\n2. PATTERN SIMILARITY ANALYSIS")
    print("-" * 50)
    
    base_pattern = patterns['sine_wave']
    
    for name, pattern in patterns.items():
        if name != 'sine_wave':
            similarity = extractor.calculate_similarity(base_pattern, pattern)
            print(f"Similarity between sine_wave and {name}: {similarity:.3f}")
    
    # Demonstrate batch processing
    print("\n\n3. BATCH FEATURE EXTRACTION")
    print("-" * 50)
    
    pattern_list = list(patterns.values())
    start_time = time.time()
    feature_matrix = extractor.extract_batch(pattern_list)
    extraction_time = time.time() - start_time
    
    print(f"Extracted features from {len(pattern_list)} patterns")
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Total extraction time: {extraction_time*1000:.2f}ms")
    print(f"Time per pattern: {extraction_time*1000/len(pattern_list):.2f}ms")
    
    # Demonstrate feature importance
    print("\n\n4. FEATURE IMPORTANCE ANALYSIS")
    print("-" * 50)
    
    # Create patterns with varying frequencies
    freq_patterns = []
    labels = []
    t = np.linspace(0, 2*np.pi, 100)
    
    for freq in [1, 2, 3, 4, 5]:
        for _ in range(5):
            pattern = np.sin(freq * t) + 0.1 * np.random.randn(len(t))
            freq_patterns.append(pattern)
            labels.append(freq)
    
    freq_features = extractor.extract_batch(freq_patterns)
    importance = extractor.get_feature_importance(freq_features, np.array(labels))
    
    # Get top 10 most important features
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("Top 10 most important features for frequency discrimination:")
    for i, (feature_name, score) in enumerate(sorted_features, 1):
        print(f"  {i}. {feature_name}: {score:.4f}")
    
    # Demonstrate fast extraction
    print("\n\n5. FAST FEATURE EXTRACTION")
    print("-" * 50)
    
    # Select only the most important features
    selected_features = ['dominant_frequency', 'frequency_spread', 'high_freq_ratio', 
                        'amplitude_max', 'energy_concentration']
    
    fast_extractor = FastPatternFeatureExtractor(selected_features=selected_features)
    
    # Compare extraction times
    pattern = np.random.randn(1000)
    
    # Full extraction
    start_time = time.time()
    for _ in range(100):
        _ = extractor.extract_features(pattern).to_vector()
    full_time = time.time() - start_time
    
    # Fast extraction
    start_time = time.time()
    for _ in range(100):
        _ = fast_extractor.extract_features_fast(pattern)
    fast_time = time.time() - start_time
    
    print(f"Full extraction (35 features): {full_time*10:.2f}ms per pattern")
    print(f"Fast extraction ({len(selected_features)} features): {fast_time*10:.2f}ms per pattern")
    print(f"Speed improvement: {full_time/fast_time:.1f}x faster")
    
    # Visualize patterns and features
    print("\n\n6. VISUALIZATION")
    print("-" * 50)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, pattern) in enumerate(patterns.items()):
        if idx < 5:
            ax = axes[idx]
            ax.plot(pattern)
            ax.set_title(name.replace('_', ' ').title())
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
    
    # Feature comparison heatmap
    ax = axes[5]
    feature_names = ['Amp Max', 'Peaks', 'Energy', 'Freq', 'Symmetry']
    feature_data = []
    
    for pattern in pattern_list:
        features = extractor.extract_features(pattern)
        feature_data.append([
            features.amplitude_max,
            features.num_peaks,
            features.energy_concentration,
            features.dominant_frequency,
            features.symmetry
        ])
    
    feature_data = np.array(feature_data).T
    im = ax.imshow(feature_data, aspect='auto', cmap='viridis')
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_xticks(range(len(patterns)))
    ax.set_xticklabels(list(patterns.keys()), rotation=45, ha='right')
    ax.set_title('Feature Comparison Heatmap')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('pattern_features_demo.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'pattern_features_demo.png'")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    demo_feature_extraction()
