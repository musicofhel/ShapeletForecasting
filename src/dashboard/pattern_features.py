"""
Pattern Feature Extraction System

This module provides comprehensive feature extraction from wavelet patterns,
including energy, duration, amplitude, shape characteristics, and similarity metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import pywt
from scipy import signal, stats
from scipy.spatial.distance import euclidean, cosine
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PatternFeatures:
    """Container for extracted pattern features"""
    # Wavelet features
    wavelet_coeffs_peak: np.ndarray
    wavelet_energy: float
    wavelet_entropy: float
    
    # Temporal features
    duration: int
    scale: int
    time_to_peak: int
    time_from_peak: int
    
    # Amplitude features
    amplitude_max: float
    amplitude_min: float
    amplitude_mean: float
    amplitude_std: float
    amplitude_range: float
    
    # Shape features
    num_peaks: int
    num_valleys: int
    sharpness: float
    symmetry: float
    kurtosis: float
    skewness: float
    
    # Energy distribution
    energy_concentration: float
    energy_dispersion: float
    spectral_centroid: float
    spectral_bandwidth: float
    
    # Frequency content
    dominant_frequency: float
    frequency_spread: float
    high_freq_ratio: float
    low_freq_ratio: float
    
    def to_vector(self) -> np.ndarray:
        """Convert features to a single feature vector"""
        features = []
        
        # Add wavelet coefficients (flatten if needed)
        if len(self.wavelet_coeffs_peak) > 10:
            # Take top 10 coefficients if too many
            features.extend(self.wavelet_coeffs_peak[:10])
        else:
            features.extend(self.wavelet_coeffs_peak)
            # Pad with zeros if less than 10
            features.extend([0] * (10 - len(self.wavelet_coeffs_peak)))
        
        # Add scalar features
        features.extend([
            self.wavelet_energy,
            self.wavelet_entropy,
            self.duration,
            self.scale,
            self.time_to_peak,
            self.time_from_peak,
            self.amplitude_max,
            self.amplitude_min,
            self.amplitude_mean,
            self.amplitude_std,
            self.amplitude_range,
            self.num_peaks,
            self.num_valleys,
            self.sharpness,
            self.symmetry,
            self.kurtosis,
            self.skewness,
            self.energy_concentration,
            self.energy_dispersion,
            self.spectral_centroid,
            self.spectral_bandwidth,
            self.dominant_frequency,
            self.frequency_spread,
            self.high_freq_ratio,
            self.low_freq_ratio
        ])
        
        return np.array(features)


class PatternFeatureExtractor:
    """Extract comprehensive features from wavelet patterns"""
    
    def __init__(self, 
                 wavelet: str = 'db4',
                 normalize: bool = True,
                 scaler_type: str = 'minmax'):
        """
        Initialize feature extractor
        
        Args:
            wavelet: Wavelet type for decomposition
            normalize: Whether to normalize features
            scaler_type: Type of scaler ('minmax' or 'standard')
        """
        self.wavelet = wavelet
        self.normalize = normalize
        
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
            
        self.is_fitted = False
        
    def extract_features(self, pattern: np.ndarray, 
                        scale: Optional[int] = None) -> PatternFeatures:
        """
        Extract all features from a pattern
        
        Args:
            pattern: Input pattern array
            scale: Wavelet scale (optional)
            
        Returns:
            PatternFeatures object
        """
        # Ensure pattern is 1D
        pattern = np.asarray(pattern).flatten()
        
        # Extract individual feature groups
        wavelet_features = self._extract_wavelet_features(pattern)
        temporal_features = self._extract_temporal_features(pattern)
        amplitude_features = self._extract_amplitude_features(pattern)
        shape_features = self._extract_shape_features(pattern)
        energy_features = self._extract_energy_features(pattern)
        frequency_features = self._extract_frequency_features(pattern)
        
        # Combine all features
        features = PatternFeatures(
            wavelet_coeffs_peak=wavelet_features['coeffs_peak'],
            wavelet_energy=wavelet_features['energy'],
            wavelet_entropy=wavelet_features['entropy'],
            duration=temporal_features['duration'],
            scale=scale if scale is not None else temporal_features['estimated_scale'],
            time_to_peak=temporal_features['time_to_peak'],
            time_from_peak=temporal_features['time_from_peak'],
            amplitude_max=amplitude_features['max'],
            amplitude_min=amplitude_features['min'],
            amplitude_mean=amplitude_features['mean'],
            amplitude_std=amplitude_features['std'],
            amplitude_range=amplitude_features['range'],
            num_peaks=shape_features['num_peaks'],
            num_valleys=shape_features['num_valleys'],
            sharpness=shape_features['sharpness'],
            symmetry=shape_features['symmetry'],
            kurtosis=shape_features['kurtosis'],
            skewness=shape_features['skewness'],
            energy_concentration=energy_features['concentration'],
            energy_dispersion=energy_features['dispersion'],
            spectral_centroid=energy_features['spectral_centroid'],
            spectral_bandwidth=energy_features['spectral_bandwidth'],
            dominant_frequency=frequency_features['dominant'],
            frequency_spread=frequency_features['spread'],
            high_freq_ratio=frequency_features['high_ratio'],
            low_freq_ratio=frequency_features['low_ratio']
        )
        
        return features
    
    def _extract_wavelet_features(self, pattern: np.ndarray) -> Dict:
        """Extract wavelet-specific features"""
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(pattern, self.wavelet, level=min(4, int(np.log2(len(pattern)))))
        
        # Get coefficients at peak
        peak_idx = np.argmax(np.abs(pattern))
        coeffs_at_peak = []
        
        for level_coeffs in coeffs:
            if len(level_coeffs) > 0:
                # Map peak index to coefficient index
                coeff_idx = min(peak_idx * len(level_coeffs) // len(pattern), 
                               len(level_coeffs) - 1)
                coeffs_at_peak.append(level_coeffs[coeff_idx])
        
        # Calculate wavelet energy
        energy = sum(np.sum(c**2) for c in coeffs)
        
        # Calculate wavelet entropy
        entropy = 0
        for c in coeffs:
            if len(c) > 0:
                # Normalize coefficients
                c_norm = c**2 / np.sum(c**2) if np.sum(c**2) > 0 else c
                # Calculate entropy
                c_norm = c_norm[c_norm > 0]  # Remove zeros
                if len(c_norm) > 0:
                    entropy -= np.sum(c_norm * np.log(c_norm))
        
        return {
            'coeffs_peak': np.array(coeffs_at_peak),
            'energy': energy,
            'entropy': entropy
        }
    
    def _extract_temporal_features(self, pattern: np.ndarray) -> Dict:
        """Extract time-based features"""
        duration = len(pattern)
        
        # Find peak location
        peak_idx = np.argmax(np.abs(pattern))
        time_to_peak = peak_idx
        time_from_peak = duration - peak_idx - 1
        
        # Estimate scale using autocorrelation
        autocorr = np.correlate(pattern, pattern, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find first minimum after peak
        peaks, _ = signal.find_peaks(autocorr)
        estimated_scale = peaks[0] if len(peaks) > 0 else duration // 4
        
        return {
            'duration': duration,
            'estimated_scale': estimated_scale,
            'time_to_peak': time_to_peak,
            'time_from_peak': time_from_peak
        }
    
    def _extract_amplitude_features(self, pattern: np.ndarray) -> Dict:
        """Extract amplitude-based features"""
        return {
            'max': np.max(pattern),
            'min': np.min(pattern),
            'mean': np.mean(pattern),
            'std': np.std(pattern),
            'range': np.ptp(pattern)  # peak-to-peak
        }
    
    def _extract_shape_features(self, pattern: np.ndarray) -> Dict:
        """Extract shape characteristics"""
        # Find peaks and valleys
        peaks, peak_props = signal.find_peaks(pattern, prominence=0.1*np.std(pattern))
        valleys, valley_props = signal.find_peaks(-pattern, prominence=0.1*np.std(pattern))
        
        # Calculate sharpness (second derivative at peaks)
        if len(pattern) > 2:
            second_deriv = np.gradient(np.gradient(pattern))
            sharpness = np.mean(np.abs(second_deriv[peaks])) if len(peaks) > 0 else 0
        else:
            sharpness = 0
        
        # Calculate symmetry
        mid = len(pattern) // 2
        if mid > 0:
            left_half = pattern[:mid]
            right_half = pattern[mid:2*mid] if len(pattern) >= 2*mid else pattern[mid:]
            right_half_rev = right_half[::-1]
            
            # Ensure same length
            min_len = min(len(left_half), len(right_half_rev))
            if min_len > 0:
                symmetry = 1 - np.mean(np.abs(left_half[:min_len] - right_half_rev[:min_len])) / (np.std(pattern) + 1e-8)
            else:
                symmetry = 0
        else:
            symmetry = 0
        
        # Statistical shape measures
        kurtosis = stats.kurtosis(pattern)
        skewness = stats.skew(pattern)
        
        return {
            'num_peaks': len(peaks),
            'num_valleys': len(valleys),
            'sharpness': sharpness,
            'symmetry': symmetry,
            'kurtosis': kurtosis,
            'skewness': skewness
        }
    
    def _extract_energy_features(self, pattern: np.ndarray) -> Dict:
        """Extract energy distribution features"""
        # Calculate energy distribution
        energy = pattern**2
        total_energy = np.sum(energy)
        
        if total_energy > 0:
            # Energy concentration (how concentrated is the energy)
            cumsum_energy = np.cumsum(energy)
            concentration_idx = np.argmax(cumsum_energy > 0.9 * total_energy)
            concentration = 1 - (concentration_idx / len(pattern))
            
            # Energy dispersion
            energy_norm = energy / total_energy
            dispersion = -np.sum(energy_norm * np.log(energy_norm + 1e-10))
        else:
            concentration = 0
            dispersion = 0
        
        # Spectral features
        fft = np.fft.fft(pattern)
        freqs = np.fft.fftfreq(len(pattern))
        magnitude = np.abs(fft[:len(fft)//2])
        freqs = freqs[:len(freqs)//2]
        
        if np.sum(magnitude) > 0:
            # Spectral centroid
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            
            # Spectral bandwidth
            spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * magnitude) / np.sum(magnitude))
        else:
            spectral_centroid = 0
            spectral_bandwidth = 0
        
        return {
            'concentration': concentration,
            'dispersion': dispersion,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth
        }
    
    def _extract_frequency_features(self, pattern: np.ndarray) -> Dict:
        """Extract frequency content features"""
        # Compute FFT
        fft = np.fft.fft(pattern)
        freqs = np.fft.fftfreq(len(pattern))
        magnitude = np.abs(fft[:len(fft)//2])
        freqs = freqs[:len(freqs)//2]
        
        if len(magnitude) > 0 and np.max(magnitude) > 0:
            # Dominant frequency
            dominant_idx = np.argmax(magnitude)
            dominant_frequency = np.abs(freqs[dominant_idx])
            
            # Frequency spread (weighted standard deviation)
            magnitude_norm = magnitude / np.sum(magnitude)
            mean_freq = np.sum(freqs * magnitude_norm)
            frequency_spread = np.sqrt(np.sum(magnitude_norm * (freqs - mean_freq)**2))
            
            # High/low frequency ratios
            mid_freq = 0.5 * np.max(np.abs(freqs))
            high_freq_power = np.sum(magnitude[np.abs(freqs) > mid_freq]**2)
            low_freq_power = np.sum(magnitude[np.abs(freqs) <= mid_freq]**2)
            total_power = high_freq_power + low_freq_power
            
            if total_power > 0:
                high_freq_ratio = high_freq_power / total_power
                low_freq_ratio = low_freq_power / total_power
            else:
                high_freq_ratio = 0
                low_freq_ratio = 0
        else:
            dominant_frequency = 0
            frequency_spread = 0
            high_freq_ratio = 0
            low_freq_ratio = 0
        
        return {
            'dominant': dominant_frequency,
            'spread': frequency_spread,
            'high_ratio': high_freq_ratio,
            'low_ratio': low_freq_ratio
        }
    
    def extract_batch(self, patterns: List[np.ndarray], 
                     scales: Optional[List[int]] = None) -> np.ndarray:
        """
        Extract features from multiple patterns
        
        Args:
            patterns: List of pattern arrays
            scales: Optional list of scales for each pattern
            
        Returns:
            Feature matrix (n_patterns x n_features)
        """
        if scales is None:
            scales = [None] * len(patterns)
        
        features = []
        for pattern, scale in zip(patterns, scales):
            pattern_features = self.extract_features(pattern, scale)
            features.append(pattern_features.to_vector())
        
        feature_matrix = np.array(features)
        
        # Normalize if requested
        if self.normalize:
            if not self.is_fitted:
                self.scaler.fit(feature_matrix)
                self.is_fitted = True
            feature_matrix = self.scaler.transform(feature_matrix)
        
        return feature_matrix
    
    def calculate_similarity(self, pattern1: np.ndarray, 
                           pattern2: np.ndarray,
                           metric: str = 'euclidean') -> float:
        """
        Calculate similarity between two patterns based on features
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            metric: Similarity metric ('euclidean', 'cosine', 'correlation')
            
        Returns:
            Similarity score
        """
        # Extract features
        features1 = self.extract_features(pattern1).to_vector()
        features2 = self.extract_features(pattern2).to_vector()
        
        # Normalize features for comparison
        if self.normalize and self.is_fitted:
            features1 = self.scaler.transform(features1.reshape(1, -1)).flatten()
            features2 = self.scaler.transform(features2.reshape(1, -1)).flatten()
        
        # Calculate similarity
        if metric == 'euclidean':
            # Convert distance to similarity
            distance = euclidean(features1, features2)
            similarity = 1 / (1 + distance)
        elif metric == 'cosine':
            # Cosine similarity
            similarity = 1 - cosine(features1, features2)
        elif metric == 'correlation':
            # Pearson correlation
            similarity = np.corrcoef(features1, features2)[0, 1]
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return similarity
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features"""
        names = []
        
        # Wavelet coefficient names
        for i in range(10):
            names.append(f'wavelet_coeff_{i}')
        
        # Other feature names
        names.extend([
            'wavelet_energy',
            'wavelet_entropy',
            'duration',
            'scale',
            'time_to_peak',
            'time_from_peak',
            'amplitude_max',
            'amplitude_min',
            'amplitude_mean',
            'amplitude_std',
            'amplitude_range',
            'num_peaks',
            'num_valleys',
            'sharpness',
            'symmetry',
            'kurtosis',
            'skewness',
            'energy_concentration',
            'energy_dispersion',
            'spectral_centroid',
            'spectral_bandwidth',
            'dominant_frequency',
            'frequency_spread',
            'high_freq_ratio',
            'low_freq_ratio'
        ])
        
        return names
    
    def get_feature_importance(self, features: np.ndarray, 
                             labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate feature importance scores
        
        Args:
            features: Feature matrix
            labels: Optional labels for supervised importance
            
        Returns:
            Dictionary of feature names to importance scores
        """
        feature_names = self.get_feature_names()
        
        if labels is not None:
            # Use mutual information for supervised importance
            from sklearn.feature_selection import mutual_info_regression
            importance_scores = mutual_info_regression(features, labels)
        else:
            # Use variance for unsupervised importance
            importance_scores = np.var(features, axis=0)
        
        # Normalize scores
        if np.sum(importance_scores) > 0:
            importance_scores = importance_scores / np.sum(importance_scores)
        
        return dict(zip(feature_names, importance_scores))


class FastPatternFeatureExtractor(PatternFeatureExtractor):
    """Optimized version for fast feature extraction"""
    
    def __init__(self, selected_features: Optional[List[str]] = None, **kwargs):
        """
        Initialize fast extractor with optional feature selection
        
        Args:
            selected_features: List of feature names to extract (None = all)
            **kwargs: Arguments for parent class
        """
        super().__init__(**kwargs)
        self.selected_features = selected_features
        
        # Create feature extraction mapping
        self._feature_extractors = {
            'wavelet': self._extract_wavelet_features,
            'temporal': self._extract_temporal_features,
            'amplitude': self._extract_amplitude_features,
            'shape': self._extract_shape_features,
            'energy': self._extract_energy_features,
            'frequency': self._extract_frequency_features
        }
    
    def extract_features_fast(self, pattern: np.ndarray) -> np.ndarray:
        """
        Fast feature extraction with minimal overhead
        
        Args:
            pattern: Input pattern
            
        Returns:
            Feature vector
        """
        if self.selected_features is None:
            # Extract all features
            return self.extract_features(pattern).to_vector()
        
        # Extract only selected features
        features = []
        pattern = np.asarray(pattern).flatten()
        
        # Determine which feature groups to extract
        groups_needed = set()
        if any('wavelet' in f for f in self.selected_features):
            groups_needed.add('wavelet')
        if any('time' in f or 'duration' in f or 'scale' in f for f in self.selected_features):
            groups_needed.add('temporal')
        if any('amplitude' in f for f in self.selected_features):
            groups_needed.add('amplitude')
        if any(f in ['num_peaks', 'num_valleys', 'sharpness', 'symmetry', 'kurtosis', 'skewness'] 
               for f in self.selected_features):
            groups_needed.add('shape')
        if any('energy' in f or 'spectral' in f for f in self.selected_features):
            groups_needed.add('energy')
        if any('freq' in f for f in self.selected_features):
            groups_needed.add('frequency')
        
        # Extract needed feature groups
        extracted = {}
        for group in groups_needed:
            extracted[group] = self._feature_extractors[group](pattern)
        
        # Build feature vector in correct order
        all_features = self.extract_features(pattern)
        feature_dict = {
            name: getattr(all_features, name) if hasattr(all_features, name) else 0
            for name in self.get_feature_names()
        }
        
        # Select only requested features
        selected_values = []
        for name in self.selected_features:
            if name in feature_dict:
                value = feature_dict[name]
                if isinstance(value, np.ndarray):
                    selected_values.extend(value)
                else:
                    selected_values.append(value)
        
        return np.array(selected_values)
