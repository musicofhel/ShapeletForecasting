"""
Comprehensive tests for pattern discovery accuracy
Tests pattern detection, matching, and validation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dashboard.pattern_matcher import PatternMatcher
from src.dashboard.pattern_predictor import PatternPredictor
from src.dashboard.pattern_features import PatternFeatureExtractor
from src.dashboard.wavelet_sequence_analyzer import WaveletSequenceAnalyzer
from src.wavelet_analysis.wavelet_analyzer import WaveletAnalyzer
from src.dtw.dtw_calculator import DTWCalculator


class TestPatternDiscovery:
    """Test pattern discovery accuracy"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data with known patterns"""
        np.random.seed(42)
        n_points = 1000
        t = np.linspace(0, 100, n_points)
        
        # Create base signal with multiple patterns
        base_signal = np.sin(0.1 * t) + 0.5 * np.sin(0.3 * t)
        
        # Add known patterns at specific locations
        pattern1 = np.exp(-0.5 * ((t - 20) / 2) ** 2) * 2  # Gaussian peak at t=20
        pattern2 = np.exp(-0.5 * ((t - 50) / 2) ** 2) * 2  # Gaussian peak at t=50
        pattern3 = np.exp(-0.5 * ((t - 80) / 2) ** 2) * 2  # Gaussian peak at t=80
        
        signal = base_signal + pattern1 + pattern2 + pattern3
        signal += np.random.normal(0, 0.1, n_points)  # Add noise
        
        return pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=n_points, freq='1H'),
            'price': signal + 100,  # Shift to realistic price range
            'volume': np.random.randint(1000, 10000, n_points)
        })
    
    @pytest.fixture
    def pattern_matcher(self):
        """Create pattern matcher instance"""
        return PatternMatcher(
            window_size=50,
            similarity_threshold=0.8,
            min_pattern_length=20
        )
    
    @pytest.fixture
    def wavelet_analyzer(self):
        """Create wavelet analyzer instance"""
        return WaveletAnalyzer(
            wavelet='db4',
            levels=5,
            mode='symmetric'
        )
    
    def test_pattern_detection_accuracy(self, sample_data, pattern_matcher):
        """Test accuracy of pattern detection"""
        # Extract patterns
        patterns = pattern_matcher.find_patterns(sample_data['price'].values)
        
        # Should detect at least 3 patterns (the Gaussian peaks)
        assert len(patterns) >= 3
        
        # Check pattern locations (approximate due to noise)
        pattern_locations = [p['start'] for p in patterns]
        
        # Expected locations around indices 200, 500, 800 (for t=20, 50, 80)
        expected_locations = [200, 500, 800]
        
        for expected in expected_locations:
            # Check if any detected pattern is near expected location
            closest = min(pattern_locations, key=lambda x: abs(x - expected))
            assert abs(closest - expected) < 50  # Within 50 samples
    
    def test_pattern_matching_precision(self, sample_data, pattern_matcher):
        """Test precision of pattern matching"""
        # Create a known pattern template
        template_length = 50
        template = np.exp(-0.5 * ((np.linspace(-2, 2, template_length)) ** 2))
        
        # Find matches in the data
        matches = pattern_matcher.match_template(
            sample_data['price'].values,
            template
        )
        
        # Should find matches near the Gaussian peaks
        assert len(matches) >= 3
        
        # Check match quality
        for match in matches:
            assert match['similarity'] > 0.7  # Good similarity score
            assert match['confidence'] > 0.6  # High confidence
    
    def test_wavelet_pattern_extraction(self, sample_data, wavelet_analyzer):
        """Test wavelet-based pattern extraction"""
        # Perform wavelet decomposition
        coeffs = wavelet_analyzer.decompose(sample_data['price'].values)
        
        # Extract patterns from wavelet coefficients
        patterns = wavelet_analyzer.extract_patterns(coeffs)
        
        # Should identify significant patterns
        assert len(patterns) > 0
        
        # Check pattern properties
        for pattern in patterns:
            assert 'scale' in pattern
            assert 'location' in pattern
            assert 'strength' in pattern
            assert pattern['strength'] > 0.5  # Significant patterns only
    
    def test_pattern_classification_accuracy(self, sample_data):
        """Test pattern classification accuracy"""
        feature_extractor = PatternFeatureExtractor()
        
        # Extract features from known patterns
        pattern_segments = [
            sample_data['price'].values[180:230],  # Around first peak
            sample_data['price'].values[480:530],  # Around second peak
            sample_data['price'].values[780:830]   # Around third peak
        ]
        
        features_list = []
        for segment in pattern_segments:
            features = feature_extractor.extract_features(segment)
            features_list.append(features)
        
        # Features should be similar for similar patterns
        # Calculate pairwise similarities
        from scipy.spatial.distance import cosine
        
        similarities = []
        for i in range(len(features_list)):
            for j in range(i + 1, len(features_list)):
                sim = 1 - cosine(
                    list(features_list[i].values()),
                    list(features_list[j].values())
                )
                similarities.append(sim)
        
        # All patterns should be similar (Gaussian peaks)
        assert all(sim > 0.8 for sim in similarities)
    
    def test_pattern_prediction_accuracy(self, sample_data):
        """Test pattern prediction accuracy"""
        predictor = PatternPredictor()
        
        # Split data for training and testing
        train_size = 800
        train_data = sample_data[:train_size]
        test_data = sample_data[train_size:]
        
        # Train predictor
        predictor.train(train_data)
        
        # Make predictions
        predictions = predictor.predict(test_data)
        
        # Check prediction accuracy
        assert predictions is not None
        assert 'pattern_probability' in predictions
        assert 'next_pattern_type' in predictions
        
        # Probability should be reasonable
        assert 0 <= predictions['pattern_probability'] <= 1
    
    def test_dtw_pattern_matching(self, sample_data):
        """Test DTW-based pattern matching accuracy"""
        dtw_calc = DTWCalculator()
        
        # Extract pattern segments
        pattern1 = sample_data['price'].values[180:230]
        pattern2 = sample_data['price'].values[480:530]
        
        # Calculate DTW distance
        distance, path = dtw_calc.calculate_dtw(pattern1, pattern2)
        
        # Similar patterns should have low DTW distance
        normalized_distance = distance / len(pattern1)
        assert normalized_distance < 0.5  # Low distance for similar patterns
        
        # Path should be close to diagonal for similar patterns
        path_deviation = np.mean(np.abs(path[:, 0] - path[:, 1]))
        assert path_deviation < 5  # Small deviation from diagonal
    
    def test_multi_scale_pattern_detection(self, sample_data):
        """Test multi-scale pattern detection"""
        analyzer = WaveletSequenceAnalyzer()
        
        # Analyze at multiple scales
        scales = [10, 20, 50, 100]
        all_patterns = []
        
        for scale in scales:
            patterns = analyzer.analyze_scale(
                sample_data['price'].values,
                scale=scale
            )
            all_patterns.extend(patterns)
        
        # Should detect patterns at different scales
        assert len(all_patterns) > 5
        
        # Check scale diversity
        detected_scales = set(p['scale'] for p in all_patterns)
        assert len(detected_scales) >= 3  # Patterns at multiple scales
    
    def test_pattern_robustness_to_noise(self):
        """Test pattern detection robustness to noise"""
        matcher = PatternMatcher()
        
        # Create clean pattern
        t = np.linspace(0, 10, 100)
        clean_pattern = np.sin(t) + 0.5 * np.sin(2 * t)
        
        # Test with different noise levels
        noise_levels = [0.0, 0.1, 0.2, 0.5]
        detection_rates = []
        
        for noise_level in noise_levels:
            # Add noise
            noisy_pattern = clean_pattern + np.random.normal(0, noise_level, len(clean_pattern))
            
            # Detect patterns
            patterns = matcher.find_patterns(noisy_pattern)
            
            # Calculate detection rate
            detection_rate = len(patterns) / max(1, len(patterns))  # Normalize
            detection_rates.append(detection_rate)
        
        # Detection should degrade gracefully with noise
        assert detection_rates[0] > detection_rates[-1]  # Less detection with more noise
        assert detection_rates[1] > 0.5  # Still good detection with moderate noise
    
    def test_pattern_temporal_consistency(self, sample_data):
        """Test temporal consistency of pattern detection"""
        matcher = PatternMatcher()
        
        # Detect patterns in overlapping windows
        window_size = 200
        step_size = 50
        
        pattern_counts = []
        for i in range(0, len(sample_data) - window_size, step_size):
            window_data = sample_data['price'].values[i:i + window_size]
            patterns = matcher.find_patterns(window_data)
            pattern_counts.append(len(patterns))
        
        # Pattern counts should be relatively stable
        count_variance = np.var(pattern_counts)
        assert count_variance < 2.0  # Low variance in pattern counts
    
    def test_pattern_feature_stability(self):
        """Test stability of pattern features"""
        extractor = PatternFeatureExtractor()
        
        # Create pattern with small variations
        base_pattern = np.sin(np.linspace(0, 2 * np.pi, 100))
        
        feature_variations = []
        for _ in range(10):
            # Add small random variation
            varied_pattern = base_pattern + np.random.normal(0, 0.05, len(base_pattern))
            features = extractor.extract_features(varied_pattern)
            feature_variations.append(list(features.values()))
        
        # Calculate feature stability
        feature_variations = np.array(feature_variations)
        feature_stds = np.std(feature_variations, axis=0)
        
        # Features should be stable (low standard deviation)
        assert np.mean(feature_stds) < 0.1
        assert np.max(feature_stds) < 0.2


class TestPatternValidation:
    """Test pattern validation and quality metrics"""
    
    def test_pattern_quality_metrics(self):
        """Test pattern quality metric calculation"""
        matcher = PatternMatcher()
        
        # Create high-quality pattern (smooth, clear structure)
        t = np.linspace(0, 10, 100)
        high_quality = np.sin(t) * np.exp(-t / 10)
        
        # Create low-quality pattern (noisy, irregular)
        low_quality = np.random.randn(100)
        
        # Calculate quality metrics
        hq_metrics = matcher.calculate_pattern_quality(high_quality)
        lq_metrics = matcher.calculate_pattern_quality(low_quality)
        
        # High-quality pattern should score better
        assert hq_metrics['smoothness'] > lq_metrics['smoothness']
        assert hq_metrics['significance'] > lq_metrics['significance']
        assert hq_metrics['clarity'] > lq_metrics['clarity']
    
    def test_pattern_statistical_significance(self):
        """Test statistical significance of patterns"""
        # Create data with known significant pattern
        n_samples = 1000
        background = np.random.randn(n_samples) * 0.5
        
        # Insert significant pattern
        pattern_start = 400
        pattern_length = 100
        pattern = np.sin(np.linspace(0, 4 * np.pi, pattern_length)) * 3
        background[pattern_start:pattern_start + pattern_length] += pattern
        
        # Test significance
        matcher = PatternMatcher()
        patterns = matcher.find_patterns(background, test_significance=True)
        
        # Should find the significant pattern
        significant_patterns = [p for p in patterns if p['p_value'] < 0.05]
        assert len(significant_patterns) >= 1
        
        # Check location of significant pattern
        for p in significant_patterns:
            if pattern_start <= p['start'] <= pattern_start + pattern_length:
                assert p['p_value'] < 0.01  # Highly significant
    
    def test_pattern_cross_validation(self):
        """Test pattern detection cross-validation"""
        # Generate multiple similar time series
        n_series = 10
        n_points = 500
        
        all_series = []
        for i in range(n_series):
            t = np.linspace(0, 50, n_points)
            # Same underlying pattern with different noise
            signal = np.sin(0.2 * t) + 0.3 * np.cos(0.5 * t)
            signal += np.random.randn(n_points) * 0.2
            all_series.append(signal)
        
        # Detect patterns in each series
        matcher = PatternMatcher()
        pattern_sets = []
        
        for series in all_series:
            patterns = matcher.find_patterns(series)
            pattern_sets.append(patterns)
        
        # Patterns should be consistent across series
        pattern_counts = [len(ps) for ps in pattern_sets]
        count_std = np.std(pattern_counts)
        
        assert count_std < 2.0  # Low variation in pattern counts
        
        # Check pattern similarity across series
        if all(len(ps) > 0 for ps in pattern_sets):
            # Compare first pattern from each series
            first_patterns = [ps[0] for ps in pattern_sets]
            lengths = [p['end'] - p['start'] for p in first_patterns]
            length_std = np.std(lengths)
            
            assert length_std < 10  # Similar pattern lengths


class TestPerformanceBenchmarks:
    """Test performance benchmarks for pattern discovery"""
    
    def test_pattern_detection_speed(self):
        """Test pattern detection speed"""
        # Generate large dataset
        n_points = 10000
        data = np.random.randn(n_points) + np.sin(np.linspace(0, 100, n_points))
        
        matcher = PatternMatcher()
        
        # Measure detection time
        start_time = time.time()
        patterns = matcher.find_patterns(data)
        detection_time = time.time() - start_time
        
        # Should process 10k points quickly
        assert detection_time < 2.0  # Less than 2 seconds
        
        # Calculate throughput
        throughput = n_points / detection_time
        assert throughput > 5000  # At least 5k points per second
    
    def test_wavelet_analysis_performance(self):
        """Test wavelet analysis performance"""
        # Test different data sizes
        sizes = [1000, 5000, 10000, 50000]
        times = []
        
        analyzer = WaveletAnalyzer()
        
        for size in sizes:
            data = np.random.randn(size)
            
            start_time = time.time()
            coeffs = analyzer.decompose(data)
            analysis_time = time.time() - start_time
            
            times.append(analysis_time)
        
        # Check scaling behavior (should be roughly linear)
        # Calculate scaling factor
        scaling_factors = []
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i-1]
            time_ratio = times[i] / times[i-1]
            scaling_factors.append(time_ratio / size_ratio)
        
        # Scaling should be close to linear (factor close to 1)
        avg_scaling = np.mean(scaling_factors)
        assert 0.8 < avg_scaling < 1.5  # Reasonable scaling
    
    def test_dtw_computation_efficiency(self):
        """Test DTW computation efficiency"""
        dtw_calc = DTWCalculator()
        
        # Test different pattern lengths
        lengths = [50, 100, 200, 500]
        times = []
        
        for length in lengths:
            pattern1 = np.random.randn(length)
            pattern2 = np.random.randn(length)
            
            start_time = time.time()
            distance, path = dtw_calc.calculate_dtw(pattern1, pattern2)
            dtw_time = time.time() - start_time
            
            times.append(dtw_time)
        
        # DTW is O(n²), check that it's not worse
        for i in range(1, len(lengths)):
            length_ratio = lengths[i] / lengths[i-1]
            time_ratio = times[i] / times[i-1]
            
            # Time ratio should be less than quadratic
            assert time_ratio < length_ratio ** 2.5
    
    def test_memory_efficiency(self):
        """Test memory efficiency of pattern discovery"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large dataset
        large_data = np.random.randn(100000)
        matcher = PatternMatcher()
        
        # Run pattern detection multiple times
        for _ in range(10):
            patterns = matcher.find_patterns(large_data)
            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory
            
            # Memory increase should be reasonable
            assert memory_increase < 500  # Less than 500MB increase
    
    def test_parallel_processing_speedup(self):
        """Test parallel processing speedup"""
        # Generate multiple time series
        n_series = 20
        series_length = 1000
        
        all_series = [np.random.randn(series_length) for _ in range(n_series)]
        
        matcher = PatternMatcher()
        
        # Sequential processing
        start_time = time.time()
        sequential_results = []
        for series in all_series:
            patterns = matcher.find_patterns(series)
            sequential_results.append(patterns)
        sequential_time = time.time() - start_time
        
        # Parallel processing (if implemented)
        start_time = time.time()
        parallel_results = matcher.find_patterns_parallel(all_series)
        parallel_time = time.time() - start_time
        
        # Should see speedup with parallel processing
        speedup = sequential_time / parallel_time
        assert speedup > 1.5  # At least 1.5x speedup
        
        # Results should be consistent
        assert len(parallel_results) == len(sequential_results)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
