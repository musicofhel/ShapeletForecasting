"""
Demo script for Pattern Type Classification System
Shows how patterns are classified into meaningful types instead of generic labels
"""

import numpy as np
import matplotlib.pyplot as plt
from src.dashboard.pattern_classifier import PatternClassifier, PatternDefinition
from src.dashboard.wavelet_sequence_analyzer import WaveletSequenceAnalyzer
from src.dashboard.data_utils import data_manager






def analyze_real_market_patterns():
    """Analyze patterns from real market data"""
    print("\n=== Real Market Pattern Analysis ===")
    
    # Download market data using data_manager
    ticker = "AAPL"
    data = data_manager.download_data(ticker, period="1y")
    if data is None or data.empty:
        print(f"Warning: Could not download data for {ticker}")
        return
    prices = data['Close'].values
    
    # Initialize analyzers
    classifier = PatternClassifier()
    wavelet_analyzer = WaveletSequenceAnalyzer()
    
    # Extract patterns using wavelet analysis
    patterns = wavelet_analyzer.extract_patterns(prices, min_pattern_length=20)
    
    print(f"\nAnalyzing {len(patterns)} patterns from {ticker}...")
    
    # Classify each pattern
    pattern_types = {}
    for i, pattern_info in enumerate(patterns[:10]):  # Analyze first 10 patterns
        pattern_data = prices[pattern_info['start']:pattern_info['end']]
        
        # Normalize pattern for better classification
        pattern_normalized = (pattern_data - np.mean(pattern_data)) / (np.std(pattern_data) + 1e-8)
        
        result = classifier.classify_pattern(pattern_normalized)
        
        if result['best_match']:
            pattern_type = result['best_match']['name']
            if pattern_type not in pattern_types:
                pattern_types[pattern_type] = []
            pattern_types[pattern_type].append({
                'index': i,
                'confidence': result['confidence'],
                'start': pattern_info['start'],
                'end': pattern_info['end']
            })
    
    # Print summary
    print("\nPattern Type Distribution:")
    print("-" * 50)
    for pattern_type, occurrences in pattern_types.items():
        avg_confidence = np.mean([o['confidence'] for o in occurrences])
        print(f"{pattern_type}: {len(occurrences)} occurrences (avg confidence: {avg_confidence:.1%})")
    
    # Visualize top patterns by type
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    plot_idx = 0
    for pattern_type, occurrences in list(pattern_types.items())[:4]:
        if plot_idx >= 4:
            break
            
        # Get highest confidence example
        best_example = max(occurrences, key=lambda x: x['confidence'])
        pattern_data = prices[best_example['start']:best_example['end']]
        
        ax = axes[plot_idx]
        ax.plot(pattern_data, 'b-', linewidth=2)
        ax.set_title(f'{pattern_type}\nConfidence: {best_example["confidence"]:.1%}')
        ax.set_xlabel('Days')
        ax.set_ylabel('Price ($)')
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, 4):
        axes[i].axis('off')
    
    plt.suptitle(f'Real Market Patterns Detected in {ticker}', fontsize=16)
    plt.tight_layout()
    plt.savefig('real_market_patterns_classified.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_custom_patterns():
    """Demonstrate adding custom pattern definitions"""
    print("\n=== Custom Pattern Definition Demo ===")
    
    classifier = PatternClassifier()
    
    # Define a custom pattern
    custom_pattern = PatternDefinition(
        name='Market Reversal Shapelet',
        category='custom',
        description='Custom pattern indicating potential market reversal',
        key_points=[
            {'position': 0.0, 'type': 'trend_start'},
            {'position': 0.3, 'type': 'momentum_peak'},
            {'position': 0.6, 'type': 'reversal_point'},
            {'position': 1.0, 'type': 'new_trend'}
        ],
        validation_rules={
            'momentum_change': True,
            'volume_spike': True,
            'reversal_strength': 0.7
        },
        confidence_threshold=0.65
    )
    
    # Add to classifier
    classifier.add_custom_pattern(custom_pattern)
    
    print("Added custom pattern: 'Market Reversal Shapelet'")
    
    # Show updated statistics
    stats = classifier.get_pattern_statistics()
    print(f"\nPattern Library Statistics:")
    print(f"Total patterns: {stats['total_patterns']}")
    print(f"Categories: {list(stats['by_category'].keys())}")
    print(f"Patterns by category:")
    for category, count in stats['by_category'].items():
        print(f"  - {category}: {count} patterns")


def main():
    """Run pattern classification demo"""
    print("=== Pattern Type Classification System Demo ===")
    print("This demo shows how patterns are classified into meaningful types")
    print("instead of generic 'Pattern A', 'Pattern B' labels.\n")
    
    # Initialize classifier
    classifier = PatternClassifier()
    
    # Show available patterns
    stats = classifier.get_pattern_statistics()
    print(f"Pattern library loaded with {stats['total_patterns']} patterns")
    print("\nAvailable pattern types:")
    for pattern in stats['pattern_list'][:10]:  # Show first 10
        print(f"  - {pattern['name']} ({pattern['category']}): {pattern['description']}")
    
    # The system focuses on real market data analysis
    
    # Analyze real market patterns
    try:
        analyze_real_market_patterns()
    except Exception as e:
        print(f"Could not analyze real market data: {e}")
    
    # Demonstrate custom patterns
    demonstrate_custom_patterns()
    
    print("\n=== Demo Complete ===")
    print("The pattern classifier successfully identifies:")
    print("  - Traditional technical analysis patterns (Head & Shoulders, Triangles, etc.)")
    print("  - Fractal patterns with self-similar properties")
    print("  - Unique shapelets and motifs")
    print("  - Custom user-defined patterns")
    print("\nThis provides meaningful pattern names for analysis and forecasting!")


if __name__ == "__main__":
    main()
