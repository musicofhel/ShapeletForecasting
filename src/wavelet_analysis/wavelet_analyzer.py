"""
Wavelet Analyzer Module

Implements Continuous Wavelet Transform (CWT) for financial time series analysis.
"""

import numpy as np
import pandas as pd
import pywt
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy import signal
from scipy.stats import zscore
import warnings

logger = logging.getLogger(__name__)


class WaveletAnalyzer:
    """
    Performs Continuous Wavelet Transform analysis on financial time series.
    """
    
    def __init__(self, wavelet: str = 'morl', scales: Optional[np.ndarray] = None):
        """
        Initialize the WaveletAnalyzer.
        
        Args:
            wavelet: Wavelet type ('morl', 'mexh', 'gaus8', 'paul')
            scales: Array of scales for CWT (if None, uses default range)
        """
        self.wavelet = wavelet
        self.scales = scales if scales is not None else self._default_scales()
        
        # Validate wavelet
        if wavelet not in pywt.wavelist(kind='continuous'):
            raise ValueError(f"Invalid wavelet: {wavelet}. Must be a continuous wavelet.")
        
        logger.info(f"Initialized WaveletAnalyzer with {wavelet} wavelet")
    
    def _default_scales(self) -> np.ndarray:
        """Generate default scales for CWT."""
        # Logarithmically spaced scales from 2 to 128
        return np.logspace(np.log2(2), np.log2(128), num=100, base=2)
    
    def transform(self, data: Union[pd.Series, np.ndarray], 
                 normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Continuous Wavelet Transform on the input data.
        
        Args:
            data: Time series data
            normalize: Whether to normalize the input data
            
        Returns:
            Tuple of (coefficients, frequencies)
        """
        # Convert to numpy array if needed
        if isinstance(data, pd.Series):
            values = data.values
        else:
            values = np.asarray(data)
        
        # Handle NaN values
        if np.any(np.isnan(values)):
            logger.warning("NaN values detected, interpolating...")
            values = pd.Series(values).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values
        
        # Normalize if requested
        if normalize:
            values = zscore(values)
        
        # Perform CWT
        logger.debug(f"Performing CWT with {len(self.scales)} scales on {len(values)} data points")
        coefficients, frequencies = pywt.cwt(values, self.scales, self.wavelet)
        
        return coefficients, frequencies
    
    def extract_features(self, coefficients: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract features from wavelet coefficients.
        
        Args:
            coefficients: CWT coefficients
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Energy at each scale
        features['scale_energy'] = np.sum(np.abs(coefficients)**2, axis=1)
        
        # Maximum coefficient at each scale
        features['scale_max'] = np.max(np.abs(coefficients), axis=1)
        
        # Mean coefficient at each scale
        features['scale_mean'] = np.mean(np.abs(coefficients), axis=1)
        
        # Standard deviation at each scale
        features['scale_std'] = np.std(coefficients, axis=1)
        
        # Dominant scale (scale with maximum energy)
        features['dominant_scale'] = self.scales[np.argmax(features['scale_energy'])]
        
        # Time-based features
        features['time_energy'] = np.sum(np.abs(coefficients)**2, axis=0)
        features['time_max'] = np.max(np.abs(coefficients), axis=0)
        
        # Ridge detection (local maxima in scale-time plane)
        features['ridges'] = self._detect_ridges(coefficients)
        
        return features
    
    def _detect_ridges(self, coefficients: np.ndarray, 
                      threshold: float = 0.1) -> List[Tuple[int, int]]:
        """
        Detect ridges in the wavelet transform (local maxima).
        
        Args:
            coefficients: CWT coefficients
            threshold: Threshold for ridge detection
            
        Returns:
            List of (scale_idx, time_idx) tuples
        """
        # Get absolute values
        abs_coeffs = np.abs(coefficients)
        
        # Normalize
        abs_coeffs = abs_coeffs / np.max(abs_coeffs)
        
        # Find local maxima
        ridges = []
        for i in range(1, abs_coeffs.shape[0] - 1):
            for j in range(1, abs_coeffs.shape[1] - 1):
                if abs_coeffs[i, j] > threshold:
                    # Check if it's a local maximum
                    if (abs_coeffs[i, j] > abs_coeffs[i-1, j] and
                        abs_coeffs[i, j] > abs_coeffs[i+1, j] and
                        abs_coeffs[i, j] > abs_coeffs[i, j-1] and
                        abs_coeffs[i, j] > abs_coeffs[i, j+1]):
                        ridges.append((i, j))
        
        return ridges
    
    def multi_resolution_analysis(self, data: Union[pd.Series, np.ndarray],
                                 level: int = 5) -> Dict[str, np.ndarray]:
        """
        Perform multi-resolution analysis using discrete wavelet transform.
        
        Args:
            data: Time series data
            level: Decomposition level
            
        Returns:
            Dictionary with approximation and detail coefficients
        """
        # Convert to numpy array if needed
        if isinstance(data, pd.Series):
            values = data.values
        else:
            values = np.asarray(data)
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(values, 'db4', level=level)
        
        # Organize results
        result = {
            'approximation': coeffs[0],
            'details': coeffs[1:]
        }
        
        # Reconstruct signals at each level
        for i in range(level):
            # Create coefficient list with zeros except for current level
            rec_coeffs = [np.zeros_like(c) for c in coeffs]
            rec_coeffs[i+1] = coeffs[i+1]
            
            # Reconstruct
            result[f'detail_level_{i+1}'] = pywt.waverec(rec_coeffs, 'db4', mode='symmetric')
        
        return result
    
    def scalogram(self, data: Union[pd.Series, np.ndarray],
                 normalize: bool = True) -> Dict[str, np.ndarray]:
        """
        Compute scalogram (power spectrum in scale-time plane).
        
        Args:
            data: Time series data
            normalize: Whether to normalize the input data
            
        Returns:
            Dictionary with scalogram data
        """
        # Perform CWT
        coefficients, frequencies = self.transform(data, normalize)
        
        # Compute power
        power = np.abs(coefficients)**2
        
        # Compute phase
        phase = np.angle(coefficients)
        
        return {
            'power': power,
            'phase': phase,
            'coefficients': coefficients,
            'frequencies': frequencies,
            'scales': self.scales
        }
    
    def detect_patterns(self, coefficients: np.ndarray,
                       min_duration: int = 5,
                       power_threshold: float = 0.5) -> List[Dict]:
        """
        Detect significant patterns in wavelet coefficients.
        
        Args:
            coefficients: CWT coefficients
            min_duration: Minimum duration for a pattern
            power_threshold: Power threshold for pattern detection
            
        Returns:
            List of detected patterns with metadata
        """
        patterns = []
        power = np.abs(coefficients)**2
        
        # Normalize power
        power_norm = power / np.max(power)
        
        # For each scale
        for scale_idx, scale in enumerate(self.scales):
            scale_power = power_norm[scale_idx, :]
            
            # Find regions above threshold
            above_threshold = scale_power > power_threshold
            
            # Find connected regions
            regions = self._find_connected_regions(above_threshold, min_duration)
            
            # Create pattern entries
            for start, end in regions:
                patterns.append({
                    'scale': scale,
                    'scale_idx': scale_idx,
                    'start': start,
                    'end': end,
                    'duration': end - start,
                    'max_power': np.max(scale_power[start:end]),
                    'mean_power': np.mean(scale_power[start:end]),
                    'coefficients': coefficients[scale_idx, start:end]
                })
        
        # Sort by max power
        patterns.sort(key=lambda x: x['max_power'], reverse=True)
        
        return patterns
    
    def _find_connected_regions(self, binary_array: np.ndarray,
                               min_length: int) -> List[Tuple[int, int]]:
        """
        Find connected regions in a binary array.
        
        Args:
            binary_array: Binary array
            min_length: Minimum length for a region
            
        Returns:
            List of (start, end) tuples
        """
        regions = []
        start = None
        
        for i, val in enumerate(binary_array):
            if val and start is None:
                start = i
            elif not val and start is not None:
                if i - start >= min_length:
                    regions.append((start, i))
                start = None
        
        # Handle case where array ends with True
        if start is not None and len(binary_array) - start >= min_length:
            regions.append((start, len(binary_array)))
        
        return regions
    
    def cross_wavelet_transform(self, data1: Union[pd.Series, np.ndarray],
                               data2: Union[pd.Series, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute cross wavelet transform between two time series.
        
        Args:
            data1: First time series
            data2: Second time series
            
        Returns:
            Dictionary with cross wavelet results
        """
        # Perform CWT on both series
        coeffs1, _ = self.transform(data1)
        coeffs2, _ = self.transform(data2)
        
        # Compute cross wavelet transform
        cross_wavelet = coeffs1 * np.conj(coeffs2)
        
        # Compute coherence
        # Smooth using a Gaussian filter
        from scipy.ndimage import gaussian_filter
        
        smooth_cross = gaussian_filter(np.abs(cross_wavelet), sigma=1)
        smooth_power1 = gaussian_filter(np.abs(coeffs1)**2, sigma=1)
        smooth_power2 = gaussian_filter(np.abs(coeffs2)**2, sigma=1)
        
        coherence = smooth_cross / np.sqrt(smooth_power1 * smooth_power2)
        
        # Compute phase difference
        phase_diff = np.angle(cross_wavelet)
        
        return {
            'cross_wavelet': cross_wavelet,
            'coherence': coherence,
            'phase_difference': phase_diff,
            'power': np.abs(cross_wavelet)**2
        }
    
    def wavelet_denoising(self, data: Union[pd.Series, np.ndarray],
                         wavelet: str = 'db4',
                         level: Optional[int] = None,
                         threshold_type: str = 'soft') -> np.ndarray:
        """
        Denoise signal using wavelet thresholding.
        
        Args:
            data: Time series data
            wavelet: Wavelet for denoising
            level: Decomposition level (None for automatic)
            threshold_type: 'soft' or 'hard' thresholding
            
        Returns:
            Denoised signal
        """
        # Convert to numpy array if needed
        if isinstance(data, pd.Series):
            values = data.values
        else:
            values = np.asarray(data)
        
        # Determine level if not specified
        if level is None:
            level = pywt.dwt_max_level(len(values), wavelet)
        
        # Decompose
        coeffs = pywt.wavedec(values, wavelet, level=level)
        
        # Estimate noise level (using MAD of finest detail coefficients)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        
        # Universal threshold
        threshold = sigma * np.sqrt(2 * np.log(len(values)))
        
        # Apply thresholding
        coeffs_thresh = list(coeffs)
        coeffs_thresh[1:] = [
            pywt.threshold(c, threshold, mode=threshold_type)
            for c in coeffs_thresh[1:]
        ]
        
        # Reconstruct
        denoised = pywt.waverec(coeffs_thresh, wavelet, mode='symmetric')
        
        # Ensure same length as input
        if len(denoised) > len(values):
            denoised = denoised[:len(values)]
        
        return denoised


def main():
    """Demonstration of WaveletAnalyzer functionality."""
    # Create sample data
    t = np.linspace(0, 1, 1000)
    # Signal with multiple frequency components
    signal = (np.sin(2 * np.pi * 5 * t) + 
              0.5 * np.sin(2 * np.pi * 20 * t) + 
              0.2 * np.random.randn(len(t)))
    
    # Initialize analyzer
    analyzer = WaveletAnalyzer(wavelet='morl')
    
    # Perform CWT
    coeffs, freqs = analyzer.transform(signal)
    print(f"CWT shape: {coeffs.shape}")
    
    # Extract features
    features = analyzer.extract_features(coeffs)
    print(f"Dominant scale: {features['dominant_scale']:.2f}")
    
    # Detect patterns
    patterns = analyzer.detect_patterns(coeffs)
    print(f"Found {len(patterns)} patterns")
    
    # Denoise
    denoised = analyzer.wavelet_denoising(signal)
    print(f"Noise reduction: {np.std(signal - denoised):.4f}")


if __name__ == "__main__":
    main()
