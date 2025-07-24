"""
Demo script for Pattern Matcher
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
import sys
sys.path.append('.')

from src.dashboard.pattern_matcher import (
    PatternMatcher, create_example_templates
)


def create_realistic_patterns():
    """Create realistic financial patterns"""
    patterns = {}
    
    # Bull flag pattern
    rise = np.linspace(0, 1, 20)
    consolidation = np.linspace(1, 0.8, 30) + np.sin(np.linspace(0, 2*np.pi, 30)) * 0.05
    patterns['bull_flag'] = np.concatenate([rise, consolidation])
    
    # Head and shoulders
    x = np.linspace(0, 1, 60)
    left_shoulder = np.exp(-((x - 0.2) / 0.1) ** 2) * 0.7
    head = np.exp(-((x - 0.5) / 0.1) ** 2)
    right_shoulder = np.exp(-((x - 0.8) / 0.1) ** 2) * 0.7
    patterns['head_shoulders'] = left_shoulder + head + right_shoulder
    
    # Double bottom
    x = np.linspace(0, 4*np.pi, 60)
    patterns['double_bottom'] = -np.abs(np.sin(x/2)) + 1
    
    # Ascending triangle
    resistance = np.ones(50) * 0.8
    support = np.linspace(0.2, 0.75, 50)
    patterns['ascending_triangle'] = np.minimum(resistance, support + np.random.normal(0, 0.02, 50))
    
    # Cup and handle
    cup = -np.abs(np.sin(np.linspace(0, np.pi, 40))) + 1
    handle = np.linspace(0.7, 0.6, 10) + np.sin(np.linspace(0, np.pi, 10)) * 0.05
    patterns['cup_handle'] = np.concatenate([cup, handle])
    
    return patterns


def demo_pattern_matching():
    """Demonstrate pattern matching capabilities"""
    print("=== Pattern Matcher Demo ===\n")
    
    # Initialize matcher
    print("1. Initializing Pattern Matcher...")
    matcher = PatternMatcher()
    
    # Create synthetic templates
    print("\n2. Creating Pattern Templates...")
    base_patterns = create_realistic_patterns()
    
    # Add variations with realistic outcomes
    for pattern_name, base_pattern in base_patterns.items():
        print(f"   - Creating variations for {pattern_name}")
        
        for i in range(30):  # 30 variations per pattern
            # Create variation
            noise = np.random.normal(0, 0.08, len(base_pattern))
            scale = np.random.uniform(0.8, 1.2)
            varied_pattern = base_pattern * scale + noise
            
            # Length variation
            if np.random.random() > 0.5:
                new_length = int(len(base_pattern) * np.random.uniform(0.85, 1.15))
                x_old = np.linspace(0, 1, len(varied_pattern))
                x_new = np.linspace(0, 1, new_length)
                varied_pattern = np.interp(x_new, x_old, varied_pattern)
            
            # Generate realistic outcomes based on pattern type
            if 'bull' in pattern_name or 'ascending' in pattern_name or 'cup' in pattern_name:
                mean_return = np.random.uniform(0.015, 0.035)  # 1.5-3.5% positive
            elif 'bear' in pattern_name or 'head_shoulders' in pattern_name:
                mean_return = np.random.uniform(-0.025, -0.01)  # Negative returns
            else:
                mean_return = np.random.uniform(-0.01, 0.015)  # Mixed
            
            # Generate return distribution
            returns = np.random.normal(mean_return, 0.008, 25)
            
            matcher.add_template(
                f'{pattern_name}_v{i}',
                varied_pattern,
                {
                    'returns': returns.tolist(),
                    'volatility': np.std(returns),
                    'sharpe': mean_return / np.std(returns) if np.std(returns) > 0 else 0,
                    'win_rate': np.sum(returns > 0) / len(returns),
                    'max_return': np.max(returns),
                    'min_return': np.min(returns)
                },
                {
                    'pattern_type': pattern_name,
                    'variation_id': i,
                    'market_condition': np.random.choice(['trending', 'ranging', 'volatile']),
                    'timeframe': np.random.choice(['1H', '4H', '1D'])
                }
            )
    
    print(f"\n   Total templates created: {len(matcher.template_matcher.templates)}")
    
    # Test pattern matching
    print("\n3. Testing Pattern Matching...")
    
    # Create test queries
    test_queries = {
        'bull_flag_query': create_realistic_patterns()['bull_flag'] + np.random.normal(0, 0.1, 50),
        'double_bottom_query': create_realistic_patterns()['double_bottom'] + np.random.normal(0, 0.12, 60),
        'random_query': np.cumsum(np.random.randn(55)) * 0.1
    }
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Pattern Matching Demo', fontsize=16)
    
    for idx, (query_name, query_pattern) in enumerate(test_queries.items()):
        print(f"\n   Testing {query_name}...")
        
        # Time the matching
        start_time = time.time()
        matches = matcher.match_pattern(query_pattern, top_k=5, min_similarity=0.65)
        match_time = time.time() - start_time
        
        print(f"   - Matching completed in {match_time*1000:.1f}ms")
        print(f"   - Found {len(matches)} matches")
        
        # Plot query
        ax = axes[idx, 0]
        ax.plot(query_pattern, 'b-', linewidth=2)
        ax.set_title(f'Query: {query_name}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Plot best match
        if matches:
            best_match = matches[0]
            template_data = matcher.template_matcher.templates[best_match.template_id]
            
            ax = axes[idx, 1]
            ax.plot(query_pattern / np.std(query_pattern), 'b-', label='Query (normalized)', alpha=0.7)
            ax.plot(template_data['pattern'] / np.std(template_data['pattern']), 
                   'r--', label=f'Best Match: {best_match.template_id}', alpha=0.7)
            ax.set_title(f'Best Match (Similarity: {best_match.similarity_score:.3f})')
            ax.set_xlabel('Time')
            ax.set_ylabel('Normalized Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Show match details
            print(f"\n   Best Match Details:")
            print(f"   - Template: {best_match.template_id}")
            print(f"   - Similarity: {best_match.similarity_score:.3f}")
            print(f"   - DTW Distance: {best_match.dtw_distance:.3f}")
            print(f"   - Correlation: {best_match.correlation:.3f}")
            print(f"   - Pattern Type: {best_match.metadata.get('pattern_type', 'Unknown')}")
            
            # Get forecast
            forecast = matcher.get_forecast_ranges(matches[:3])
            
            ax = axes[idx, 2]
            
            # Plot historical outcomes distribution
            all_returns = []
            for match in matches[:3]:
                if 'returns' in match.historical_outcomes:
                    all_returns.extend(match.historical_outcomes['returns'])
            
            if all_returns:
                ax.hist(all_returns, bins=20, alpha=0.7, color='green', edgecolor='black')
                ax.axvline(forecast['mean_forecast'], color='red', linestyle='--', 
                          label=f"Mean: {forecast['mean_forecast']:.3%}")
                
                # Add confidence intervals
                if '95%' in forecast['confidence_intervals']:
                    ci = forecast['confidence_intervals']['95%']
                    ax.axvspan(ci['lower'], ci['upper'], alpha=0.2, color='red',
                              label=f"95% CI: [{ci['lower']:.3%}, {ci['upper']:.3%}]")
                
                ax.set_title('Forecast Distribution')
                ax.set_xlabel('Returns')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                print(f"\n   Forecast Results:")
                print(f"   - Mean Return: {forecast['mean_forecast']:.3%}")
                print(f"   - 68% CI: [{forecast['confidence_intervals']['68%']['lower']:.3%}, "
                      f"{forecast['confidence_intervals']['68%']['upper']:.3%}]")
                print(f"   - 95% CI: [{forecast['confidence_intervals']['95%']['lower']:.3%}, "
                      f"{forecast['confidence_intervals']['95%']['upper']:.3%}]")
    
    plt.tight_layout()
    plt.savefig('pattern_matcher_demo.png', dpi=150, bbox_inches='tight')
    print("\n   Saved visualization to pattern_matcher_demo.png")
    
    # Performance test
    print("\n4. Performance Testing...")
    
    # Test with different numbers of templates
    template_counts = [100, 500, 1000]
    query_lengths = [30, 50, 100]
    
    print("\n   Matching Speed Test:")
    print("   " + "-" * 50)
    print(f"   {'Templates':<12} {'Query Len':<12} {'Time (ms)':<12} {'Matches':<12}")
    print("   " + "-" * 50)
    
    for n_templates in template_counts:
        # Limit templates for testing
        original_templates = dict(list(matcher.template_matcher.templates.items())[:n_templates])
        matcher.template_matcher.templates = original_templates
        matcher.template_matcher.template_features = {
            tid: matcher.template_matcher._extract_features(tdata['pattern'])
            for tid, tdata in original_templates.items()
        }
        
        for query_len in query_lengths:
            query = np.cumsum(np.random.randn(query_len)) * 0.1
            
            start_time = time.time()
            matches = matcher.match_pattern(query, top_k=5, use_parallel=True)
            elapsed = (time.time() - start_time) * 1000
            
            print(f"   {n_templates:<12} {query_len:<12} {elapsed:<12.1f} {len(matches):<12}")
    
    # Cross-pattern matching test
    print("\n5. Cross-Pattern Matching Test...")
    
    # Create a hybrid pattern (part bull flag, part cup)
    hybrid = np.concatenate([
        create_realistic_patterns()['bull_flag'][:25],
        create_realistic_patterns()['cup_handle'][25:]
    ])
    hybrid += np.random.normal(0, 0.08, len(hybrid))
    
    matches = matcher.match_pattern(hybrid, top_k=10, min_similarity=0.6)
    
    print("\n   Hybrid Pattern Match Results:")
    pattern_types = {}
    for match in matches:
        ptype = match.metadata.get('pattern_type', 'unknown')
        pattern_types[ptype] = pattern_types.get(ptype, 0) + 1
    
    for ptype, count in pattern_types.items():
        print(f"   - {ptype}: {count} matches ({count/len(matches)*100:.1f}%)")
    
    # Memory usage summary
    print("\n6. Memory Usage Summary:")
    template_size = sum(
        len(t['pattern']) * 8 +  # Pattern data
        len(str(t['outcomes'])) +  # Outcomes
        len(str(t.get('metadata', {})))  # Metadata
        for t in matcher.template_matcher.templates.values()
    ) / 1024 / 1024  # Convert to MB
    
    print(f"   - Total templates: {len(matcher.template_matcher.templates)}")
    print(f"   - Estimated memory usage: {template_size:.2f} MB")
    print(f"   - Average bytes per template: {template_size * 1024 * 1024 / len(matcher.template_matcher.templates):.0f}")
    
    print("\n=== Demo Complete ===")


def demo_dtw_alignment():
    """Demonstrate DTW alignment visualization"""
    print("\n=== DTW Alignment Visualization ===\n")
    
    # Create two similar but shifted patterns
    x = np.linspace(0, 2*np.pi, 50)
    pattern1 = np.sin(x)
    pattern2 = np.sin(x + 0.3) * 1.2  # Phase shifted and scaled
    
    # Add some noise
    pattern1 += np.random.normal(0, 0.05, len(pattern1))
    pattern2 += np.random.normal(0, 0.05, len(pattern2))
    
    # Create matcher and compute DTW
    from src.dashboard.pattern_matcher import DTWMatcher
    dtw = DTWMatcher()
    distance, path = dtw.compute_dtw(pattern1, pattern2)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot patterns
    ax1.plot(pattern1, 'b-', label='Pattern 1', linewidth=2)
    ax1.plot(pattern2, 'r-', label='Pattern 2', linewidth=2)
    ax1.set_title('Original Patterns')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot alignment
    ax2.set_title(f'DTW Alignment (Distance: {distance:.3f})')
    
    # Draw alignment connections
    for i, (idx1, idx2) in enumerate(path[::3]):  # Show every 3rd connection
        ax2.plot([idx1, idx2], [pattern1[idx1], pattern2[idx2]], 
                'gray', alpha=0.3, linewidth=0.5)
    
    ax2.plot(pattern1, 'b-', label='Pattern 1', linewidth=2)
    ax2.plot(pattern2, 'r-', label='Pattern 2', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dtw_alignment_demo.png', dpi=150, bbox_inches='tight')
    print("Saved DTW alignment visualization to dtw_alignment_demo.png")


if __name__ == "__main__":
    # Run main demo
    demo_pattern_matching()
    
    # Run DTW alignment demo
    demo_dtw_alignment()
