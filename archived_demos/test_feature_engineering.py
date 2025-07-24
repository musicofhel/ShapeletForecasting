"""
Test script for feature engineering components
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from features import (
    PatternFeatureExtractor,
    TechnicalIndicators,
    TransitionMatrixBuilder,
    FeaturePipeline,
    FeatureSelector
)


def generate_test_data(n_samples=500):
    """Generate test financial data."""
    dates = pd.date_range(start='2022-01-01', periods=n_samples, freq='D')
    
    # Generate price with trend and noise
    trend = np.linspace(100, 120, n_samples)
    noise = np.random.normal(0, 2, n_samples)
    price = trend + noise
    
    # Create OHLCV data
    df = pd.DataFrame({
        'open': price + np.random.uniform(-1, 1, n_samples),
        'high': price + np.abs(np.random.normal(0, 1, n_samples)),
        'low': price - np.abs(np.random.normal(0, 1, n_samples)),
        'close': price,
        'volume': np.random.lognormal(15, 0.5, n_samples)
    }, index=dates)
    
    # Ensure consistency
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


def test_technical_indicators():
    """Test technical indicators calculation."""
    print("\n=== Testing Technical Indicators ===")
    
    df = generate_test_data()
    calculator = TechnicalIndicators()
    
    # Compute indicators
    indicators = calculator.compute_all_indicators(df)
    
    print(f"✓ Computed {len(indicators.columns)} technical indicators")
    print(f"✓ Sample indicators: {list(indicators.columns[:5])}")
    
    # Check some key indicators
    assert 'sma_20' in indicators.columns
    assert 'rsi_14' in indicators.columns
    assert 'macd' in indicators.columns
    assert not indicators['sma_20'].iloc[20:].isna().any()
    
    print("✓ Technical indicators test passed!")


def test_pattern_features():
    """Test pattern feature extraction."""
    print("\n=== Testing Pattern Features ===")
    
    # Generate windows
    df = generate_test_data()
    window_size = 50
    windows = []
    
    for i in range(window_size, len(df)):
        window = df['close'].iloc[i-window_size:i].values
        windows.append(window)
    
    windows = np.array(windows[:100])  # Use first 100 windows
    
    # Extract features
    extractor = PatternFeatureExtractor(
        n_patterns=5,
        pattern_length=10
    )
    
    features = extractor.fit_transform(windows)
    
    print(f"✓ Extracted {len(features.columns)} pattern features")
    print(f"✓ Feature shape: {features.shape}")
    
    # Check key features
    assert 'wavelet_energy' in features.columns
    assert 'dtw_min_distance' in features.columns
    assert features.shape[0] == len(windows)
    
    print("✓ Pattern features test passed!")


def test_transition_matrix():
    """Test transition matrix builder."""
    print("\n=== Testing Transition Matrix ===")
    
    # Generate pattern sequences
    n_patterns = 5
    sequences = []
    
    for _ in range(50):
        sequence = np.random.randint(0, n_patterns, size=30).tolist()
        sequences.append(sequence)
    
    # Build transition matrix
    builder = TransitionMatrixBuilder(
        n_patterns=n_patterns,
        max_order=2
    )
    
    builder.fit(sequences)
    features = builder.transform(sequences)
    
    print(f"✓ Built transition matrices up to order 2")
    print(f"✓ Extracted {len(features.columns)} transition features")
    
    # Check transition matrix
    trans_matrix = builder.get_transition_matrix(order=1)
    assert trans_matrix.shape == (n_patterns, n_patterns)
    assert np.allclose(trans_matrix.sum(axis=1), 1.0)  # Rows sum to 1
    
    print("✓ Transition matrix test passed!")


def test_feature_pipeline():
    """Test complete feature pipeline."""
    print("\n=== Testing Feature Pipeline ===")
    
    df = generate_test_data()
    
    # Create pipeline
    pipeline = FeaturePipeline(
        use_pattern_features=True,
        use_technical_indicators=True,
        use_transition_features=True,
        n_patterns=5,
        pattern_length=10,
        feature_window=50,
        prediction_horizon=5
    )
    
    # Fit and transform
    print("Fitting pipeline...")
    pipeline.fit(df)
    
    print("Transforming data...")
    features = pipeline.transform(df)
    
    print(f"✓ Total features: {len(features.columns)}")
    print(f"✓ Feature matrix shape: {features.shape}")
    
    # Create target
    target = pipeline.create_target(df, target_type='classification')
    print(f"✓ Created target variable: {target.name}")
    
    print("✓ Feature pipeline test passed!")


def test_feature_selection():
    """Test feature selection."""
    print("\n=== Testing Feature Selection ===")
    
    # Generate simple features
    n_samples = 200
    n_features = 50
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add some correlated features
    X['feature_corr1'] = X['feature_0'] + np.random.normal(0, 0.1, n_samples)
    X['feature_corr2'] = X['feature_1'] * 0.9 + np.random.normal(0, 0.1, n_samples)
    
    # Binary target
    y = (X['feature_0'] + X['feature_1'] > 0).astype(int)
    
    # Feature selection
    selector = FeatureSelector(
        task_type='classification',
        selection_method='rf',
        n_features=10,
        correlation_threshold=0.95
    )
    
    X_selected = selector.fit_transform(X, y)
    
    print(f"✓ Selected {len(selector.selected_features_)} features")
    print(f"✓ Removed {len(selector.removed_correlated_)} correlated features")
    print(f"✓ Top features: {selector.selected_features_[:5]}")
    
    assert X_selected.shape[1] == len(selector.selected_features_)
    assert len(selector.removed_correlated_) > 0  # Should remove some correlated features
    
    print("✓ Feature selection test passed!")


def main():
    """Run all tests."""
    print("Testing Feature Engineering Components")
    print("=" * 50)
    
    try:
        test_technical_indicators()
        test_pattern_features()
        test_transition_matrix()
        test_feature_pipeline()
        test_feature_selection()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
