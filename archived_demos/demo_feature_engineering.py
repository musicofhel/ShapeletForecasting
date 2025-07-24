"""
Feature Engineering Demo Script

This script demonstrates the complete feature engineering pipeline
for financial time series prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import feature engineering components
from features import (
    TechnicalIndicators,
    PatternFeatureExtractor,
    TransitionMatrixBuilder,
    FeaturePipeline,
    FeatureSelector
)

def generate_sample_data(n_samples=1000, n_features=1):
    """Generate sample financial time series data."""
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate synthetic price data with trend and noise
    trend = np.linspace(100, 150, n_samples)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365)
    noise = np.random.normal(0, 5, n_samples)
    
    prices = trend + seasonal + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'high': prices + np.abs(np.random.normal(0, 2, n_samples)),
        'low': prices - np.abs(np.random.normal(0, 2, n_samples)),
        'open': prices + np.random.normal(0, 1, n_samples),
        'volume': np.random.randint(1000000, 5000000, n_samples)
    })
    
    return df

def main():
    print("=" * 80)
    print("FEATURE ENGINEERING DEMO")
    print("=" * 80)
    
    # Generate sample data
    print("\n1. Generating sample financial data...")
    df = generate_sample_data(1000)
    print(f"   Generated {len(df)} samples")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Technical Indicators
    print("\n2. Computing Technical Indicators...")
    tech_indicators = TechnicalIndicators()
    tech_features = tech_indicators.compute_all_indicators(df)
    print(f"   Computed {len(tech_features.columns)} technical indicators")
    print(f"   Sample indicators: {list(tech_features.columns[:5])}")
    
    # Pattern Features
    print("\n3. Extracting Pattern Features...")
    # Create time series matrix for pattern extraction
    n_samples = 100
    window_size = 50
    X = np.array([df['close'].values[i:i+window_size] 
                  for i in range(0, len(df)-window_size, 10)])[:n_samples]
    
    pattern_extractor = PatternFeatureExtractor(n_patterns=10, pattern_length=20)
    pattern_extractor.fit(X)
    pattern_features = pattern_extractor.transform(X)
    print(f"   Extracted {len(pattern_features.columns)} pattern features")
    print(f"   Feature categories: wavelet, DTW similarity, cluster, temporal")
    
    # Transition Matrix
    print("\n4. Building Transition Matrices...")
    # Discretize prices into states
    price_changes = df['close'].pct_change().dropna()
    states = pd.qcut(price_changes, q=5, labels=False)  # Use integer labels
    
    # Convert to list of sequences (simulate multiple sequences)
    sequence_length = 50
    sequences = []
    for i in range(0, len(states) - sequence_length, 10):
        sequences.append(states.iloc[i:i+sequence_length].tolist())
    
    transition_builder = TransitionMatrixBuilder(n_patterns=5, max_order=2)
    transition_builder.fit(sequences)
    trans_features = transition_builder.transform(sequences)
    print(f"   Built transition matrices up to order {transition_builder.max_order}")
    print(f"   Extracted {len(trans_features.columns)} transition features")
    
    # Feature Pipeline
    print("\n5. Running Complete Feature Pipeline...")
    pipeline = FeaturePipeline(
        use_pattern_features=True,
        use_technical_indicators=True,
        use_transition_features=True,
        scaler_type='standard',
        n_patterns=10,
        pattern_length=20,
        feature_window=window_size
    )
    
    # Prepare data for pipeline
    pipeline_df = df.copy()
    all_features = pipeline.fit_transform(pipeline_df)
    print(f"   Total features: {len(all_features.columns)}")
    print(f"   Feature matrix shape: {all_features.shape}")
    
    # Feature Selection
    print("\n6. Performing Feature Selection...")
    # Create a dummy target variable aligned with features
    y = (df['close'].shift(-5) > df['close']).astype(int)
    # Align y with all_features by using the same index
    y_aligned = y.loc[all_features.index].dropna()
    all_features_aligned = all_features.loc[y_aligned.index]
    
    selector = FeatureSelector(
        selection_method='mutual_info',
        n_features=20,
        correlation_threshold=0.95
    )
    
    selected_features = selector.fit_transform(all_features_aligned, y_aligned)
    print(f"   Selected {len(selected_features.columns)} features")
    print(f"   Top 5 features: {list(selected_features.columns[:5])}")
    
    # Feature Importance Analysis
    print("\n7. Feature Importance Analysis...")
    importance_df = selector.get_feature_report()
    if importance_df is not None and not importance_df.empty:
        print("\n   Top 10 Most Important Features:")
        print(importance_df.head(10).to_string())
    
    # Save processed features
    print("\n8. Saving Processed Features...")
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save features
    all_features.to_csv(output_dir / 'all_features.csv', index=False)
    selected_features.to_csv(output_dir / 'selected_features.csv', index=False)
    
    # Save feature importance
    if importance_df is not None and not importance_df.empty:
        importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    # Save pipeline configuration
    pipeline_config = {
        'n_features_total': len(all_features.columns),
        'n_features_selected': len(selected_features.columns),
        'feature_categories': {
            'technical': len([c for c in all_features.columns if 'sma' in c or 'ema' in c or 'rsi' in c]),
            'pattern': len([c for c in all_features.columns if 'wavelet' in c or 'dtw' in c or 'cluster' in c]),
            'transition': len([c for c in all_features.columns if 'trans' in c])
        }
    }
    
    import json
    with open(output_dir / 'pipeline_config.json', 'w') as f:
        json.dump(pipeline_config, f, indent=2)
    
    print(f"\n   Saved all features to: {output_dir / 'all_features.csv'}")
    print(f"   Saved selected features to: {output_dir / 'selected_features.csv'}")
    print(f"   Saved feature importance to: {output_dir / 'feature_importance.csv'}")
    print(f"   Saved pipeline config to: {output_dir / 'pipeline_config.json'}")
    
    # Create visualizations
    print("\n9. Creating Feature Visualizations...")
    
    # Feature correlation heatmap
    plt.figure(figsize=(12, 10))
    corr_matrix = selected_features.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap (Selected Features)')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance plot
    if importance_df is not None and not importance_df.empty:
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(20)
        plt.barh(range(len(top_features)), top_features['importance_score'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score')
        plt.title('Top 20 Feature Importance Scores')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Feature distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(selected_features.columns[:4]):
        axes[i].hist(selected_features[col], bins=50, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved visualizations to {output_dir}")
    
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print(f"- Total samples: {len(df)}")
    print(f"- Total features extracted: {len(all_features.columns)}")
    print(f"- Features after selection: {len(selected_features.columns)}")
    print(f"- Reduction ratio: {(1 - len(selected_features.columns)/len(all_features.columns))*100:.1f}%")
    
    print("\nFEATURE BREAKDOWN:")
    for category, count in pipeline_config['feature_categories'].items():
        print(f"- {category.capitalize()} features: {count}")

if __name__ == "__main__":
    main()
