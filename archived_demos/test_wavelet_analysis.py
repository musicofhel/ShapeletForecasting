"""
Test script for Wavelet Analysis module

Tests all components of the wavelet analysis pipeline including:
- Wavelet transforms
- Motif discovery
- Shapelet extraction
- Pattern visualization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import yaml

# Import our modules
from src.wavelet_analysis import (
    WaveletAnalyzer,
    MotifDiscovery,
    ShapeletExtractor,
    PatternVisualizer
)
from src.data_collection import StorageManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from YAML file."""
    config_path = Path("configs/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def test_wavelet_analyzer():
    """Test WaveletAnalyzer functionality."""
    logger.info("=" * 50)
    logger.info("Testing WaveletAnalyzer")
    logger.info("=" * 50)
    
    # Create sample data
    t = np.linspace(0, 10, 1000)
    # Multi-frequency signal
    signal = (np.sin(2 * np.pi * 2 * t) + 
              0.5 * np.sin(2 * np.pi * 5 * t) + 
              0.3 * np.sin(2 * np.pi * 10 * t) +
              0.1 * np.random.randn(len(t)))
    
    # Initialize analyzer
    analyzer = WaveletAnalyzer(wavelet='morl')
    
    # Test CWT
    logger.info("Testing Continuous Wavelet Transform...")
    coeffs, freqs = analyzer.transform(signal)
    logger.info(f"CWT coefficients shape: {coeffs.shape}")
    logger.info(f"Frequency range: {freqs.min():.3f} - {freqs.max():.3f}")
    
    # Test feature extraction
    logger.info("\nTesting feature extraction...")
    features = analyzer.extract_features(coeffs)
    for name, values in features.items():
        if isinstance(values, np.ndarray):
            logger.info(f"  {name}: shape={values.shape}, mean={np.mean(values):.3f}")
        else:
            logger.info(f"  {name}: {values}")
    
    # Test pattern detection
    logger.info("\nTesting pattern detection...")
    patterns = analyzer.detect_patterns(coeffs, min_duration=10, power_threshold=0.3)
    logger.info(f"Found {len(patterns)} significant patterns")
    for i, pattern in enumerate(patterns[:3]):
        logger.info(f"  Pattern {i+1}: scale={pattern['scale']:.2f}, "
                   f"duration={pattern['duration']}, max_power={pattern['max_power']:.3f}")
    
    # Test multi-resolution analysis
    logger.info("\nTesting multi-resolution analysis...")
    mra_results = analyzer.multi_resolution_analysis(signal, level=4)
    logger.info(f"MRA components: {list(mra_results.keys())}")
    
    # Test denoising
    logger.info("\nTesting wavelet denoising...")
    denoised = analyzer.wavelet_denoising(signal)
    noise_reduction = np.std(signal - denoised)
    logger.info(f"Noise reduction: {noise_reduction:.4f}")
    
    return True


def test_motif_discovery():
    """Test MotifDiscovery functionality."""
    logger.info("\n" + "=" * 50)
    logger.info("Testing MotifDiscovery")
    logger.info("=" * 50)
    
    # Create sample data with repeating patterns
    np.random.seed(42)
    t = np.linspace(0, 100, 2000)
    
    # Base patterns
    pattern1 = np.sin(2 * np.pi * t[:50] / 10) + 0.5 * np.sin(2 * np.pi * t[:50] / 5)
    pattern2 = np.exp(-t[:30] / 10) * np.cos(2 * np.pi * t[:30] / 5)
    
    # Create time series with repeated patterns
    data = np.random.randn(2000) * 0.1
    
    # Insert pattern1 at multiple locations
    for i in [100, 300, 500, 700, 900, 1100]:
        data[i:i+50] += pattern1
    
    # Insert pattern2 at different locations
    for i in [200, 600, 1000, 1400]:
        data[i:i+30] += pattern2
    
    # Add some anomalies
    data[1500:1550] += np.random.randn(50) * 2
    data[1700:1720] = np.random.randn(20) * 3
    
    # Initialize motif discovery
    md = MotifDiscovery(window_size=50, min_distance=25)
    
    # Test matrix profile computation
    logger.info("Computing matrix profile...")
    mp, mp_idx = md.compute_matrix_profile(data)
    logger.info(f"Matrix profile shape: {mp.shape}")
    logger.info(f"Matrix profile mean: {np.mean(mp):.3f}, std: {np.std(mp):.3f}")
    
    # Test motif discovery
    logger.info("\nFinding motifs...")
    motifs = md.find_motifs(data, top_k=5)
    logger.info(f"Found {len(motifs)} motifs")
    for motif in motifs:
        logger.info(f"  Motif {motif['id']}: {motif['num_occurrences']} occurrences, "
                   f"mean_distance={motif['mean_distance']:.3f}")
    
    # Test discord discovery
    logger.info("\nFinding discords (anomalies)...")
    discords = md.find_discords(data, top_k=3)
    logger.info(f"Found {len(discords)} discords")
    for discord in discords:
        logger.info(f"  Discord at index {discord['index']}: "
                   f"anomaly_score={discord['anomaly_score']:.3f}")
    
    # Test chain discovery
    logger.info("\nFinding chains (evolving patterns)...")
    chains = md.find_chains(data)
    logger.info(f"Found {len(chains)} chains")
    for i, chain in enumerate(chains[:3]):
        logger.info(f"  Chain {i+1}: length={chain['length']}, "
                   f"total_evolution={chain['total_evolution']:.3f}")
    
    # Test semantic motifs
    logger.info("\nFinding semantic motifs...")
    # Create simple labels based on value ranges
    labels = np.zeros(len(data), dtype=int)
    labels[data > np.percentile(data, 75)] = 1
    labels[data < np.percentile(data, 25)] = 2
    
    semantic_motifs = md.find_semantic_motifs(data, labels, top_k=3)
    logger.info(f"Found {len(semantic_motifs)} semantic motifs")
    for motif in semantic_motifs:
        logger.info(f"  Label {motif['label']}: {motif['num_occurrences']} occurrences")
    
    return motifs, discords


def test_shapelet_extractor():
    """Test ShapeletExtractor functionality."""
    logger.info("\n" + "=" * 50)
    logger.info("Testing ShapeletExtractor")
    logger.info("=" * 50)
    
    # Create sample data with discriminative patterns
    np.random.seed(42)
    n_samples = 1500
    
    # Class 0: Upward trend followed by plateau
    class0_pattern = np.concatenate([
        np.linspace(0, 1, 20),
        np.ones(10)
    ])
    
    # Class 1: Downward trend followed by recovery
    class1_pattern = np.concatenate([
        np.linspace(1, 0, 15),
        np.linspace(0, 0.5, 15)
    ])
    
    # Generate time series
    data = np.random.randn(n_samples) * 0.1
    labels = np.zeros(n_samples, dtype=int)
    
    # Insert class patterns
    for i in range(0, n_samples - 30, 50):
        if np.random.rand() > 0.5:
            data[i:i+30] += class0_pattern + np.random.randn(30) * 0.05
            labels[i:i+30] = 0
        else:
            data[i:i+30] += class1_pattern + np.random.randn(30) * 0.05
            labels[i:i+30] = 1
    
    # Initialize extractor
    extractor = ShapeletExtractor(
        min_length=15,
        max_length=35,
        num_shapelets=10,
        quality_threshold=0.3
    )
    
    # Extract shapelets
    logger.info("Extracting shapelets...")
    shapelets = extractor.extract_shapelets(data, labels, n_jobs=1)
    logger.info(f"Extracted {len(shapelets)} shapelets")
    
    # Display shapelet information
    for i, shapelet in enumerate(shapelets[:5]):
        logger.info(f"  Shapelet {i+1}: length={len(shapelet['pattern'])}, "
                   f"quality={shapelet['quality']:.3f}, "
                   f"threshold={shapelet['threshold']:.3f}")
    
    # Test transformation
    logger.info("\nTesting shapelet transformation...")
    transformed = extractor.transform(data[:500])
    logger.info(f"Transformed data shape: {transformed.shape}")
    
    # Get shapelet features
    shapelet_features = extractor.get_shapelet_features()
    logger.info(f"\nShapelet features shape: {shapelet_features.shape}")
    logger.info("Shapelet statistics:")
    logger.info(f"  Mean quality: {shapelet_features['quality'].mean():.3f}")
    logger.info(f"  Mean length: {shapelet_features['length'].mean():.1f}")
    
    return shapelets


def test_with_real_data(config):
    """Test wavelet analysis with real financial data."""
    logger.info("\n" + "=" * 50)
    logger.info("Testing with Real Financial Data")
    logger.info("=" * 50)
    
    # Initialize storage manager
    storage = StorageManager(base_path="data", storage_format="hdf5")
    
    # Load some data
    logger.info("Loading financial data...")
    data = storage.load_raw_data(tickers=['TSLA'], timeframes=['1d'])
    
    if not data or 'TSLA' not in data or '1d' not in data['TSLA']:
        logger.warning("No TSLA data found. Skipping real data test.")
        return
    
    # Get TSLA daily data
    df = data['TSLA']['1d']
    logger.info(f"Loaded {len(df)} days of TSLA data")
    
    # Use closing prices
    prices = df['Close'].values
    returns = np.diff(np.log(prices))
    
    # 1. Wavelet Analysis
    logger.info("\n1. Performing wavelet analysis on returns...")
    analyzer = WaveletAnalyzer(wavelet='morl')
    coeffs, freqs = analyzer.transform(returns)
    
    scalogram = analyzer.scalogram(returns)
    features = analyzer.extract_features(coeffs)
    logger.info(f"Dominant scale: {features['dominant_scale']:.2f}")
    
    # 2. Motif Discovery
    logger.info("\n2. Discovering motifs in price movements...")
    md = MotifDiscovery(window_size=20, min_distance=10)
    motifs = md.find_motifs(returns, top_k=5)
    logger.info(f"Found {len(motifs)} motifs in returns")
    
    # 3. Shapelet Extraction
    logger.info("\n3. Extracting shapelets...")
    # Create labels based on future returns
    future_returns = pd.Series(returns).rolling(5).mean().shift(-5).fillna(0)
    labels = (future_returns > 0).astype(int).values
    
    extractor = ShapeletExtractor(
        min_length=10,
        max_length=30,
        num_shapelets=20,
        quality_threshold=0.2
    )
    
    shapelets = extractor.extract_shapelets(returns, labels, n_jobs=1)
    logger.info(f"Extracted {len(shapelets)} shapelets")
    
    # 4. Save results
    logger.info("\n4. Saving analysis results...")
    
    # Save shapelets
    shapelet_data = {
        '1d': [s['pattern'] for s in shapelets]
    }
    storage.save_shapelets(shapelet_data, 'TSLA')
    
    # Save wavelet features
    storage.save_processed_data(
        pd.DataFrame([features]),
        'TSLA_wavelet_features',
        category='wavelet_analysis'
    )
    
    # Save motif features
    motif_features = md.get_motif_features(motifs)
    storage.save_processed_data(
        motif_features,
        'TSLA_motif_features',
        category='wavelet_analysis'
    )
    
    return scalogram, motifs, shapelets


def test_visualization():
    """Test visualization capabilities."""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Visualization")
    logger.info("=" * 50)
    
    # Create sample data
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 5 * t)
    
    # Initialize components
    analyzer = WaveletAnalyzer(wavelet='morl')
    md = MotifDiscovery(window_size=50)
    viz = PatternVisualizer()
    
    # Perform analysis
    coeffs, freqs = analyzer.transform(signal)
    scalogram = analyzer.scalogram(signal)
    motifs = md.find_motifs(signal, top_k=3)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # 1. Wavelet transform plot
    fig1 = viz.plot_wavelet_transform(
        coeffs, 
        analyzer.scales,
        title="Wavelet Transform of Test Signal",
        save_path="results/wavelet_transform.png"
    )
    plt.close(fig1)
    
    # 2. Motif plot
    if motifs:
        fig2 = viz.plot_motifs(
            signal,
            motifs,
            title="Discovered Motifs",
            save_path="results/motifs.png"
        )
        plt.close(fig2)
    
    # 3. Interactive scalogram
    interactive_fig = viz.create_interactive_scalogram(
        coeffs,
        analyzer.scales,
        original_data=signal,
        title="Interactive Wavelet Analysis"
    )
    interactive_fig.write_html("results/interactive_scalogram.html")
    
    logger.info("Visualizations saved to results/ directory")
    
    return True


def main():
    """Main test function."""
    logger.info("Starting Wavelet Analysis Tests")
    logger.info("=" * 70)
    
    # Load configuration
    config = load_config()
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    # Run tests
    try:
        # Test individual components
        test_wavelet_analyzer()
        motifs, discords = test_motif_discovery()
        shapelets = test_shapelet_extractor()
        
        # Test with real data if available
        real_results = test_with_real_data(config)
        
        # Test visualization
        test_visualization()
        
        logger.info("\n" + "=" * 70)
        logger.info("All tests completed successfully!")
        logger.info("=" * 70)
        
        # Summary
        logger.info("\nSummary:")
        logger.info("- WaveletAnalyzer: ✓")
        logger.info("- MotifDiscovery: ✓")
        logger.info("- ShapeletExtractor: ✓")
        logger.info("- PatternVisualizer: ✓")
        logger.info("- Real data analysis: ✓" if real_results else "- Real data analysis: Skipped (no data)")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
