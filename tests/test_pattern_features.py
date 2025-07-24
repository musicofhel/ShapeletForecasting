"""
Unit tests for Pattern Feature Extraction System
"""

import numpy as np
import pytest
from typing import List
import time
from scipy import signal
import pywt
from sklearn.preprocessing import StandardScaler

from src.dashboard.pattern_features import (
    PatternFeatures,
    PatternFeatureExtractor,
    FastPatternFeatureExtractor
)


class TestPatternFeatures:
    """Test PatternFeatures dataclass"""
    
    def test_pattern_features_creation(self):
        """Test creating PatternFeatures object"""
        features = PatternFeatures(
            wavelet_coeffs_peak=np.array([1.0, 2.0, 3.0]),
            wavelet_energy=10.5,
            wavelet_entropy=0.8,
            duration=100,
            scale=16,
            time_to_peak=25,
            time_from_peak=75,
            amplitude_max=2.0,
            amplitude_min=-1.5,
            amplitude_mean=0.1,
            amplitude_std=0.5,
            amplitude_range=3.5,
            num_peaks=3,
            num_valleys=2,
            sharpness=0.7,
            symmetry=0.85,
            kurtosis=2.5,
            skewness=0.3,
            energy_concentration=0.9,
            energy_dispersion=0.2,
            spectral_centroid=0.15,
            spectral_bandwidth=0.05,
            dominant_frequency=0.1,
            frequency_spread=0.08,
            high_freq_ratio=0.3,
            low_freq_ratio=0.7
        )
        
        assert features.wavelet_energy == 10.5
        assert features.duration == 100
        assert features.num_peaks == 3
        
    def test_to_vector_conversion(self):
        """Test converting features to vector"""
        features = PatternFeatures(
            wavelet_coeffs_peak=np.array([1.0, 2.0, 3.0]),
            wavelet_energy=10.5,
            wavelet_entropy=0.8,
            duration=100,
            scale=16,
            time_to_peak=25,
            time_from_peak=75,
            amplitude_max=2.0,
            amplitude_min=-1.5,
            amplitude_mean=0.1,
            amplitude_std=0.5,
            amplitude_range=3.5,
            num_peaks=3,
            num_valleys=2,
            sharpness=0.7,
            symmetry=0.85,
            kurtosis=2.5,
            skewness=0.3,
            energy_concentration=0.9,
            energy_dispersion=0.2,
            spectral_centroid=0.15,
            spectral_bandwidth=0.05,
            dominant_frequency=0.1,
            frequency_spread=0.08,
            high_freq_ratio=0.3,
            low_freq_ratio=0.7
        )
        
        vector = features.to_vector()
        
        # Check vector length (10 wavelet coeffs + 25 other features)
        assert len(vector) == 35
        
        # Check first few values
        assert vector[0] == 1.0  # First wavelet coeff
        assert vector[1] == 2.0  # Second wavelet coeff
        assert vector[10] == 10.5  # wavelet_energy
        
    def test_wavelet_coeff_padding(self):
        """Test padding of wavelet coefficients"""
        # Test with fewer than 10 coefficients
        features = PatternFeatures(
            wavelet_coeffs_peak=np.array([1.0, 2.0]),
            wavelet_energy=0, wavelet_entropy=0, duration=0, scale=0,
            time_to_peak=0, time_from_peak=0, amplitude_max=0,
            amplitude_min=0, amplitude_mean=0, amplitude_std=0,
            amplitude_range=0, num_peaks=0, num_valleys=0,
            sharpness=0, symmetry=0, kurtosis=0, skewness=0,
            energy_concentration=0, energy_dispersion=0,
            spectral_centroid=0, spectral_bandwidth=0,
            dominant_frequency=0, frequency_spread=0,
            high_freq_ratio=0, low_freq_ratio=0
        )
        
        vector = features.to_vector()
        # Should have padded with zeros
        assert vector[0] == 1.0
        assert vector[1] == 2.0
        assert all(vector[2:10] == 0)
        
        # Test with more than 10 coefficients
        features.wavelet_coeffs_peak = np.arange(15)
        vector = features.to_vector()
        # Should only take first 10
        assert all(vector[:10] == np.arange(10))


class TestPatternFeatureExtractor:
    """Test PatternFeatureExtractor class"""
    
    @pytest.fixture
    def sample_patterns(self):
        """Generate sample patterns for testing"""
        t = np.linspace(0, 4*np.pi, 100)
        
        # Sine wave pattern
        sine_pattern = np.sin(t)
        
        # Gaussian pulse pattern
        gaussian_pattern = np.exp(-((t - 2*np.pi)**2) / 2)
        
        # Complex pattern with multiple frequencies
        complex_pattern = np.sin(t) + 0.5*np.sin(3*t) + 0.2*np.sin(5*t)
        
        # Noisy pattern
        noisy_pattern = sine_pattern + 0.1 * np.random.randn(len(t))
        
        return {
            'sine': sine_pattern,
            'gaussian': gaussian_pattern,
            'complex': complex_pattern,
            'noisy': noisy_pattern
        }
    
    def test_extractor_initialization(self):
        """Test initializing feature extractor"""
        extractor = PatternFeatureExtractor()
        assert extractor.wavelet == 'db4'
        assert extractor.normalize == True
        assert not extractor.is_fitted
        
        # Test with custom parameters
        extractor = PatternFeatureExtractor(
            wavelet='sym5',
            normalize=False,
            scaler_type='standard'
        )
        assert extractor.wavelet == 'sym5'
        assert extractor.normalize == False
        assert isinstance(extractor.scaler, StandardScaler)
    
    def test_feature_extraction(self, sample_patterns):
        """Test extracting features from a single pattern"""
        extractor = PatternFeatureExtractor()
        pattern = sample_patterns['sine']
        
        features = extractor.extract_features(pattern)
        
        # Check that all features are extracted
        assert isinstance(features, PatternFeatures)
        assert features.duration == len(pattern)
        assert features.amplitude_max > 0
        assert features.amplitude_min < 0
        assert features.num_peaks > 0
        assert features.wavelet_energy > 0
        
    def test_wavelet_feature_extraction(self, sample_patterns):
        """Test wavelet-specific feature extraction"""
        extractor = PatternFeatureExtractor()
        pattern = sample_patterns['gaussian']
        
        features = extractor._extract_wavelet_features(pattern)
        
        assert 'coeffs_peak' in features
        assert 'energy' in features
        assert 'entropy' in features
        assert features['energy'] > 0
        assert len(features['coeffs_peak']) > 0
        
    def test_temporal_feature_extraction(self, sample_patterns):
        """Test temporal feature extraction"""
        extractor = PatternFeatureExtractor()
        pattern = sample_patterns['gaussian']
        
        features = extractor._extract_temporal_features(pattern)
        
        assert features['duration'] == len(pattern)
        assert features['time_to_peak'] >= 0
        assert features['time_from_peak'] >= 0
        assert features['time_to_peak'] + features['time_from_peak'] == len(pattern) - 1
        
    def test_amplitude_feature_extraction(self, sample_patterns):
        """Test amplitude feature extraction"""
        extractor = PatternFeatureExtractor()
        pattern = sample_patterns['sine']
        
        features = extractor._extract_amplitude_features(pattern)
        
        assert features['max'] == np.max(pattern)
        assert features['min'] == np.min(pattern)
        assert features['mean'] == pytest.approx(np.mean(pattern), abs=1e-6)
        assert features['std'] == pytest.approx(np.std(pattern), abs=1e-6)
        assert features['range'] == np.ptp(pattern)
        
    def test_shape_feature_extraction(self, sample_patterns):
        """Test shape feature extraction"""
        extractor = PatternFeatureExtractor()
        pattern = sample_patterns['complex']
        
        features = extractor._extract_shape_features(pattern)
        
        assert features['num_peaks'] >= 0
        assert features['num_valleys'] >= 0
        assert -1 <= features['symmetry'] <= 1
        assert isinstance(features['kurtosis'], float)
        assert isinstance(features['skewness'], float)
        
    def test_energy_feature_extraction(self, sample_patterns):
        """Test energy distribution feature extraction"""
        extractor = PatternFeatureExtractor()
        pattern = sample_patterns['gaussian']
        
        features = extractor._extract_energy_features(pattern)
        
        assert 0 <= features['concentration'] <= 1
        assert features['dispersion'] >= 0
        assert features['spectral_centroid'] >= 0
        assert features['spectral_bandwidth'] >= 0
        
    def test_frequency_feature_extraction(self, sample_patterns):
        """Test frequency content feature extraction"""
        extractor = PatternFeatureExtractor()
        pattern = sample_patterns['complex']
        
        features = extractor._extract_frequency_features(pattern)
        
        assert features['dominant'] >= 0
        assert features['spread'] >= 0
        assert 0 <= features['high_ratio'] <= 1
        assert 0 <= features['low_ratio'] <= 1
        assert abs(features['high_ratio'] + features['low_ratio'] - 1.0) < 1e-6
        
    def test_batch_extraction(self, sample_patterns):
        """Test extracting features from multiple patterns"""
        extractor = PatternFeatureExtractor()
        patterns = list(sample_patterns.values())
        
        feature_matrix = extractor.extract_batch(patterns)
        
        assert feature_matrix.shape[0] == len(patterns)
        assert feature_matrix.shape[1] == 35  # Total number of features
        
        # Check normalization
        if extractor.normalize:
            assert extractor.is_fitted
            assert np.all(feature_matrix >= 0)
            assert np.all(feature_matrix <= 1)
            
    def test_similarity_calculation(self, sample_patterns):
        """Test pattern similarity calculation"""
        extractor = PatternFeatureExtractor()
        
        # Similar patterns should have high similarity
        pattern1 = sample_patterns['sine']
        pattern2 = sample_patterns['sine'] + 0.01 * np.random.randn(len(pattern1))
        
        similarity = extractor.calculate_similarity(pattern1, pattern2)
        assert 0 <= similarity <= 1
        
        # Different patterns should have lower similarity
        pattern3 = sample_patterns['gaussian']
        similarity2 = extractor.calculate_similarity(pattern1, pattern3)
        assert similarity2 < similarity
        
    def test_similarity_metrics(self, sample_patterns):
        """Test different similarity metrics"""
        extractor = PatternFeatureExtractor()
        pattern1 = sample_patterns['sine']
        pattern2 = sample_patterns['complex']
        
        # Test different metrics
        euclidean_sim = extractor.calculate_similarity(pattern1, pattern2, metric='euclidean')
        cosine_sim = extractor.calculate_similarity(pattern1, pattern2, metric='cosine')
        correlation_sim = extractor.calculate_similarity(pattern1, pattern2, metric='correlation')
        
        assert 0 <= euclidean_sim <= 1
        assert -1 <= cosine_sim <= 1
        assert -1 <= correlation_sim <= 1
        
    def test_feature_names(self):
        """Test getting feature names"""
        extractor = PatternFeatureExtractor()
        names = extractor.get_feature_names()
        
        assert len(names) == 35
        assert names[0] == 'wavelet_coeff_0'
        assert 'wavelet_energy' in names
        assert 'duration' in names
        assert 'amplitude_max' in names
        
    def test_feature_importance(self, sample_patterns):
        """Test feature importance calculation"""
        extractor = PatternFeatureExtractor()
        patterns = list(sample_patterns.values())
        
        # Extract features
        feature_matrix = extractor.extract_batch(patterns)
        
        # Test unsupervised importance
        importance = extractor.get_feature_importance(feature_matrix)
        assert len(importance) == 35
        assert all(0 <= v <= 1 for v in importance.values())
        assert abs(sum(importance.values()) - 1.0) < 1e-6
        
        # Test supervised importance
        labels = np.array([0, 1, 2, 0])  # Dummy labels
        importance_supervised = extractor.get_feature_importance(feature_matrix, labels)
        assert len(importance_supervised) == 35


class TestFastPatternFeatureExtractor:
    """Test FastPatternFeatureExtractor class"""
    
    @pytest.fixture
    def sample_pattern(self):
        """Generate a sample pattern"""
        t = np.linspace(0, 2*np.pi, 100)
        return np.sin(t) + 0.5*np.sin(3*t)
    
    def test_fast_extractor_initialization(self):
        """Test initializing fast feature extractor"""
        extractor = FastPatternFeatureExtractor()
        assert extractor.selected_features is None
        
        # With selected features
        selected = ['amplitude_max', 'amplitude_mean', 'num_peaks']
        extractor = FastPatternFeatureExtractor(selected_features=selected)
        assert extractor.selected_features == selected
        
    def test_fast_extraction_all_features(self, sample_pattern):
        """Test fast extraction with all features"""
        extractor = FastPatternFeatureExtractor()
        features = extractor.extract_features_fast(sample_pattern)
        
        assert len(features) == 35
        
    def test_fast_extraction_selected_features(self, sample_pattern):
        """Test fast extraction with selected features"""
        selected = ['amplitude_max', 'amplitude_min', 'duration', 'num_peaks']
        extractor = FastPatternFeatureExtractor(selected_features=selected)
        
        features = extractor.extract_features_fast(sample_pattern)
        
        # Should only extract selected features
        assert len(features) == len(selected)


class TestPerformance:
    """Test performance requirements"""
    
    def test_extraction_speed(self):
        """Test that feature extraction is fast enough"""
        extractor = PatternFeatureExtractor()
        pattern = np.random.randn(1000)
        
        # Warm up
        _ = extractor.extract_features(pattern)
        
        # Time extraction
        start_time = time.time()
        features = extractor.extract_features(pattern)
        extraction_time = time.time() - start_time
        
        # Should be less than 10ms
        assert extraction_time < 0.01
        print(f"Extraction time: {extraction_time*1000:.2f}ms")
        
    def test_batch_extraction_speed(self):
        """Test batch extraction performance"""
        extractor = PatternFeatureExtractor()
        patterns = [np.random.randn(100) for _ in range(100)]
        
        start_time = time.time()
        features = extractor.extract_batch(patterns)
        extraction_time = time.time() - start_time
        
        # Should be efficient
        time_per_pattern = extraction_time / len(patterns)
        assert time_per_pattern < 0.01
        print(f"Batch extraction time per pattern: {time_per_pattern*1000:.2f}ms")


class TestFeatureConsistency:
    """Test feature consistency across similar patterns"""
    
    def test_similar_pattern_consistency(self):
        """Test that similar patterns produce similar features"""
        extractor = PatternFeatureExtractor(normalize=False)
        
        # Create similar patterns
        t = np.linspace(0, 2*np.pi, 100)
        base_pattern = np.sin(t)
        
        similar_patterns = []
        for i in range(10):
            noise = 0.01 * np.random.randn(len(t))
            similar_patterns.append(base_pattern + noise)
        
        # Extract features
        features = []
        for pattern in similar_patterns:
            feat = extractor.extract_features(pattern).to_vector()
            features.append(feat)
        
        features = np.array(features)
        
        # Calculate correlations between feature vectors
        correlations = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                # Only calculate correlation for features with variance
                feat_i = features[i]
                feat_j = features[j]
                
                # Find features with non-zero variance
                var_i = np.var(feat_i)
                var_j = np.var(feat_j)
                
                if var_i > 1e-10 and var_j > 1e-10:
                    corr = np.corrcoef(feat_i, feat_j)[0, 1]
                    correlations.append(corr)
        
        # Average correlation should be high
        if len(correlations) > 0:
            avg_corr = np.mean(correlations)
            assert avg_corr > 0.8
            print(f"Average correlation: {avg_corr:.3f}")


class TestNormalization:
    """Test feature normalization and scaling"""
    
    def test_minmax_normalization(self):
        """Test MinMax normalization"""
        extractor = PatternFeatureExtractor(scaler_type='minmax')
        
        # Generate diverse patterns
        patterns = []
        for i in range(20):
            t = np.linspace(0, 2*np.pi, 100)
            pattern = (i+1) * np.sin(t) + i * np.cos(2*t)
            patterns.append(pattern)
        
        features = extractor.extract_batch(patterns)
        
        # Check all features are in [0, 1]
        assert np.all(features >= -1e-10)  # Allow for small numerical errors
        assert np.all(features <= 1 + 1e-10)
        
        # Check that features with variance have good spread
        feature_vars = np.var(features, axis=0)
        varying_features = feature_vars > 1e-10
        
        if np.any(varying_features):
            # For features with variance, check spread
            max_vals = np.max(features[:, varying_features], axis=0)
            min_vals = np.min(features[:, varying_features], axis=0)
            assert np.mean(max_vals) > 0.7
            assert np.mean(min_vals) < 0.3
        
    def test_standard_normalization(self):
        """Test Standard normalization"""
        extractor = PatternFeatureExtractor(scaler_type='standard')
        
        # Generate patterns
        patterns = [np.random.randn(100) for _ in range(50)]
        features = extractor.extract_batch(patterns)
        
        # Check mean and std for features with variance
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0)
        
        # Only check features with non-zero variance
        varying_features = stds > 1e-10
        
        if np.any(varying_features):
            assert np.allclose(means[varying_features], 0, atol=1e-6)
            assert np.allclose(stds[varying_features], 1, atol=1e-6)


class TestRobustness:
    """Test robustness with noisy and edge case patterns"""
    
    def test_noisy_patterns(self):
        """Test extraction with noisy patterns"""
        extractor = PatternFeatureExtractor()
        
        # Clean pattern
        t = np.linspace(0, 2*np.pi, 100)
        clean_pattern = np.sin(t)
        
        # Add different levels of noise
        noise_levels = [0.1, 0.5, 1.0]
        for noise_level in noise_levels:
            noisy_pattern = clean_pattern + noise_level * np.random.randn(len(t))
            
            # Should not crash
            features = extractor.extract_features(noisy_pattern)
            assert features is not None
            assert not np.any(np.isnan(features.to_vector()))
            
    def test_edge_cases(self):
        """Test with edge case patterns"""
        extractor = PatternFeatureExtractor()
        
        # Constant pattern
        constant = np.ones(100)
        features = extractor.extract_features(constant)
        assert features.amplitude_range == 0
        assert features.amplitude_std == 0
        
        # Very short pattern
        short = np.array([1, 2, 3])
        features = extractor.extract_features(short)
        assert features.duration == 3
        
        # Single point
        single = np.array([5.0])
        features = extractor.extract_features(single)
        assert features.duration == 1
        
        # Empty pattern handling
        with pytest.raises(Exception):
            empty = np.array([])
            features = extractor.extract_features(empty)


class TestFeatureImportanceValidation:
    """Test feature importance validation"""
    
    def test_importance_with_known_patterns(self):
        """Test feature importance with patterns where we know what should be important"""
        extractor = PatternFeatureExtractor()
        
        # Create patterns where frequency is the distinguishing feature
        patterns = []
        labels = []
        
        t = np.linspace(0, 2*np.pi, 100)
        for freq in [1, 3, 5]:
            for _ in range(10):
                pattern = np.sin(freq * t) + 0.1 * np.random.randn(len(t))
                patterns.append(pattern)
                labels.append(freq)
        
        features = extractor.extract_batch(patterns)
        labels = np.array(labels)
        
        importance = extractor.get_feature_importance(features, labels)
        
        # Frequency-related features should be important
        freq_features = ['dominant_frequency', 'frequency_spread', 'high_freq_ratio']
        freq_importance = sum(importance[name] for name in freq_features if name in importance)
        
        # Should be relatively high
        assert freq_importance > 0.1
        print(f"Frequency feature importance: {freq_importance:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
