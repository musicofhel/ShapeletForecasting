"""
Fix Pattern Matching to Use Real YFinance Data
This script creates real pattern templates from actual market data
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pickle
from pathlib import Path
import pywt
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import our modules
from src.dashboard.pattern_matcher import PatternMatcher
from src.dashboard.pattern_classifier import PatternClassifier
from src.dashboard.wavelet_sequence_analyzer import WaveletSequenceAnalyzer
from src.dashboard.data_utils import DataManager

def extract_real_patterns_from_yfinance():
    """Extract real patterns from YFinance data"""
    print("=== Extracting Real Market Patterns ===")
    
    # Initialize data manager
    dm = DataManager()
    
    # Popular tickers to analyze
    tickers = ['BTC-USD', 'ETH-USD', 'AAPL', 'TSLA', 'SPY', 'QQQ', 'GLD', 'NVDA']
    
    all_patterns = []
    pattern_templates = {}
    
    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        
        # Download data
        data = dm.download_data(ticker, period='2y')
        if data is None or data.empty:
            print(f"  ✗ No data for {ticker}")
            continue
            
        prices = data['Close'].values
        print(f"  ✓ Loaded {len(prices)} data points")
        
        # Normalize prices
        scaler = StandardScaler()
        normalized_prices = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        
        # Perform wavelet decomposition
        wavelet = 'db4'
        levels = 5
        coeffs = pywt.wavedec(normalized_prices, wavelet, level=levels)
        
        # Find interesting patterns using multiple methods
        patterns = []
        
        # Method 1: Peak/Valley patterns
        peaks, peak_props = find_peaks(normalized_prices, prominence=0.5, distance=20)
        valleys, valley_props = find_peaks(-normalized_prices, prominence=0.5, distance=20)
        
        # Extract patterns around peaks
        for peak in peaks[:10]:  # Limit to 10 patterns per ticker
            start = max(0, peak - 30)
            end = min(len(prices), peak + 30)
            if end - start >= 20:
                pattern_data = normalized_prices[start:end]
                patterns.append({
                    'ticker': ticker,
                    'type': 'peak_reversal',
                    'data': pattern_data,
                    'start_idx': start,
                    'end_idx': end,
                    'original_prices': prices[start:end],
                    'metadata': {
                        'peak_idx': peak - start,
                        'prominence': peak_props['prominences'][list(peaks).index(peak)]
                    }
                })
        
        # Extract patterns around valleys
        for valley in valleys[:10]:
            start = max(0, valley - 30)
            end = min(len(prices), valley + 30)
            if end - start >= 20:
                pattern_data = normalized_prices[start:end]
                patterns.append({
                    'ticker': ticker,
                    'type': 'valley_reversal',
                    'data': pattern_data,
                    'start_idx': start,
                    'end_idx': end,
                    'original_prices': prices[start:end],
                    'metadata': {
                        'valley_idx': valley - start,
                        'prominence': valley_props['prominences'][list(valleys).index(valley)]
                    }
                })
        
        # Method 2: Wavelet-based patterns
        # Analyze detail coefficients for patterns
        for level in range(1, min(4, len(coeffs))):
            detail_coeffs = coeffs[level]
            
            # Find high-energy regions
            energy = detail_coeffs ** 2
            threshold = np.percentile(energy, 90)
            high_energy_regions = np.where(energy > threshold)[0]
            
            if len(high_energy_regions) > 0:
                # Group consecutive indices
                groups = []
                current_group = [high_energy_regions[0]]
                
                for i in range(1, len(high_energy_regions)):
                    if high_energy_regions[i] - high_energy_regions[i-1] <= 5:
                        current_group.append(high_energy_regions[i])
                    else:
                        if len(current_group) >= 5:
                            groups.append(current_group)
                        current_group = [high_energy_regions[i]]
                
                if len(current_group) >= 5:
                    groups.append(current_group)
                
                # Extract patterns from high-energy regions
                scale_factor = 2 ** level
                for group in groups[:5]:  # Limit patterns per level
                    start_idx = group[0] * scale_factor
                    end_idx = min(group[-1] * scale_factor + scale_factor, len(prices))
                    
                    if end_idx - start_idx >= 20 and end_idx - start_idx <= 200:
                        pattern_data = normalized_prices[start_idx:end_idx]
                        patterns.append({
                            'ticker': ticker,
                            'type': f'wavelet_level_{level}',
                            'data': pattern_data,
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'original_prices': prices[start_idx:end_idx],
                            'metadata': {
                                'wavelet_level': level,
                                'energy': np.sum(energy[group])
                            }
                        })
        
        print(f"  ✓ Found {len(patterns)} patterns")
        all_patterns.extend(patterns)
    
    return all_patterns

def create_pattern_templates(patterns):
    """Create pattern templates for the PatternMatcher"""
    print("\n=== Creating Pattern Templates ===")
    
    # Initialize pattern matcher
    matcher = PatternMatcher()
    
    # Group patterns by type
    pattern_groups = {}
    for pattern in patterns:
        pattern_type = pattern['type']
        if pattern_type not in pattern_groups:
            pattern_groups[pattern_type] = []
        pattern_groups[pattern_type].append(pattern)
    
    # Create templates for each group
    template_count = 0
    for pattern_type, group_patterns in pattern_groups.items():
        print(f"\nProcessing {pattern_type} patterns ({len(group_patterns)} total)")
        
        # Select representative patterns
        # Sort by some quality metric (e.g., prominence for peaks/valleys)
        if 'peak' in pattern_type or 'valley' in pattern_type:
            group_patterns.sort(key=lambda p: p['metadata'].get('prominence', 0), reverse=True)
        else:
            group_patterns.sort(key=lambda p: p['metadata'].get('energy', 0), reverse=True)
        
        # Take top patterns as templates
        for i, pattern in enumerate(group_patterns[:5]):
            template_id = f"{pattern['ticker']}_{pattern_type}_{i}"
            
            # Calculate historical outcomes (returns after pattern)
            returns = []
            if pattern['end_idx'] + 10 < len(pattern['original_prices']):
                future_prices = pattern['original_prices'][pattern['end_idx']:pattern['end_idx']+10]
                current_price = pattern['original_prices'][pattern['end_idx']-1]
                returns = [(p - current_price) / current_price * 100 for p in future_prices]
            
            # Add template
            matcher.add_template(
                template_id=template_id,
                pattern=pattern['data'],
                outcomes={
                    'returns': returns,
                    'avg_return': np.mean(returns) if returns else 0,
                    'std_return': np.std(returns) if returns else 0,
                    'pattern_count': 1
                },
                metadata={
                    'ticker': pattern['ticker'],
                    'type': pattern_type,
                    'length': len(pattern['data']),
                    'extraction_date': datetime.now().isoformat()
                }
            )
            template_count += 1
    
    print(f"\n✓ Created {template_count} pattern templates")
    print(f"✓ Templates saved to {matcher.template_dir}")
    
    return matcher

def test_pattern_matching(matcher):
    """Test pattern matching with real data"""
    print("\n=== Testing Pattern Matching ===")
    
    # Get fresh data for testing
    dm = DataManager()
    test_ticker = 'BTC-USD'
    
    print(f"\nTesting with {test_ticker} recent data...")
    data = dm.download_data(test_ticker, period='1mo')
    
    if data is not None and not data.empty:
        prices = data['Close'].values[-100:]  # Last 100 points
        
        # Normalize
        scaler = StandardScaler()
        normalized_prices = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        
        # Test different segments
        test_segments = [
            (0, 30, "Beginning segment"),
            (35, 65, "Middle segment"),
            (70, 100, "End segment")
        ]
        
        for start, end, desc in test_segments:
            print(f"\n{desc} [{start}:{end}]:")
            query_pattern = normalized_prices[start:end]
            
            # Find matches
            matches = matcher.match_pattern(query_pattern, top_k=3, min_similarity=0.5)
            
            if matches:
                for i, match in enumerate(matches):
                    print(f"  Match {i+1}: {match.template_id}")
                    print(f"    Similarity: {match.similarity_score:.3f}")
                    print(f"    Correlation: {match.correlation:.3f}")
                    print(f"    DTW Distance: {match.dtw_distance:.3f}")
                    if match.historical_outcomes.get('avg_return'):
                        print(f"    Avg Return: {match.historical_outcomes['avg_return']:.2f}%")
            else:
                print("  No matches found")

def visualize_templates(matcher, num_templates=10):
    """Visualize some pattern templates"""
    print("\n=== Visualizing Pattern Templates ===")
    
    templates = list(matcher.template_matcher.templates.items())[:num_templates]
    
    if not templates:
        print("No templates to visualize")
        return
    
    # Create visualization
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, (template_id, template_data) in enumerate(templates):
        if i >= 10:
            break
            
        ax = axes[i]
        pattern = template_data['pattern']
        
        ax.plot(pattern, 'b-', linewidth=2)
        ax.set_title(f"{template_id.split('_')[0]}\n{template_id.split('_')[1]}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-3, 3)
        
        # Add return info
        avg_return = template_data['outcomes'].get('avg_return', 0)
        color = 'green' if avg_return > 0 else 'red'
        ax.text(0.5, 0.95, f"Ret: {avg_return:.1f}%", 
                transform=ax.transAxes, ha='center', va='top',
                color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('real_pattern_templates.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to real_pattern_templates.png")
    plt.close()

def main():
    """Main function to fix pattern matching with real data"""
    print("=" * 60)
    print("FIXING PATTERN MATCHING WITH REAL YFINANCE DATA")
    print("=" * 60)
    
    # Step 1: Extract real patterns
    patterns = extract_real_patterns_from_yfinance()
    print(f"\n✓ Total patterns extracted: {len(patterns)}")
    
    # Step 2: Create pattern templates
    matcher = create_pattern_templates(patterns)
    
    # Step 3: Test pattern matching
    test_pattern_matching(matcher)
    
    # Step 4: Visualize templates
    visualize_templates(matcher)
    
    # Step 5: Create a demo script for the dashboard
    print("\n=== Creating Dashboard Integration ===")
    
    demo_code = '''"""
Demo: Real Pattern Matching with YFinance Data
Run this to test the pattern matching with real market data
"""

import numpy as np
from src.dashboard.pattern_matcher import PatternMatcher
from src.dashboard.data_utils import DataManager
from sklearn.preprocessing import StandardScaler

# Initialize components
matcher = PatternMatcher()
dm = DataManager()

# Get real market data
ticker = 'BTC-USD'
print(f"Loading {ticker} data...")
data = dm.download_data(ticker, period='1d')

if data is not None and not data.empty:
    prices = data['Close'].values[-50:]  # Last 50 points
    
    # Normalize
    scaler = StandardScaler()
    normalized = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
    
    # Find patterns
    print("\\nSearching for patterns...")
    matches = matcher.match_pattern(normalized, top_k=5, min_similarity=0.6)
    
    print(f"\\nFound {len(matches)} matches:")
    for i, match in enumerate(matches):
        print(f"\\n{i+1}. {match.template_id}")
        print(f"   Similarity: {match.similarity_score:.3f}")
        print(f"   Expected return: {match.historical_outcomes.get('avg_return', 0):.2f}%")
else:
    print("Failed to load data")
'''
    
    with open('test_real_pattern_matching.py', 'w') as f:
        f.write(demo_code)
    
    print("✓ Created test_real_pattern_matching.py")
    
    print("\n" + "=" * 60)
    print("PATTERN MATCHING FIXED!")
    print("=" * 60)
    print("\nThe pattern matcher now uses real market data from YFinance.")
    print("Pattern templates have been created from actual price movements.")
    print("\nTo test: python test_real_pattern_matching.py")
    print("\nThe dashboard should now show real patterns instead of 'Pattern A'")

if __name__ == "__main__":
    main()
