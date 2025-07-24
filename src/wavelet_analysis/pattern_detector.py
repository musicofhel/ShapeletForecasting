"""
Wavelet Pattern Detector Module

Detects and classifies financial patterns using wavelet analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import signal
from scipy.stats import zscore
import logging

from .wavelet_analyzer import WaveletAnalyzer

logger = logging.getLogger(__name__)


class WaveletPatternDetector:
    """
    Detects and classifies patterns in financial time series using wavelet analysis.
    """
    
    def __init__(self, wavelet: str = 'morl', scales: Optional[np.ndarray] = None):
        """
        Initialize the pattern detector.
        
        Args:
            wavelet: Wavelet type for analysis
            scales: Scales for CWT
        """
        self.wavelet_analyzer = WaveletAnalyzer(wavelet, scales)
        self.pattern_types = ['trend_up', 'trend_down', 'reversal_top', 
                             'reversal_bottom', 'consolidation']
        
        # Pattern detection parameters
        self.min_pattern_length = 20
        self.energy_threshold = 0.3
        self.trend_threshold = 0.02
        
        logger.info("Initialized WaveletPatternDetector")
    
    def detect_patterns(self, data: pd.DataFrame, 
                       price_col: str = 'close',
                       window_size: int = 50) -> List[Dict]:
        """
        Detect patterns in the time series data.
        
        Args:
            data: DataFrame with price data
            price_col: Column name for price data
            window_size: Window size for pattern detection
            
        Returns:
            List of detected patterns
        """
        if price_col not in data.columns:
            raise ValueError(f"Column '{price_col}' not found in data")
        
        prices = data[price_col].values
        
        # Perform wavelet transform
        coeffs, freqs = self.wavelet_analyzer.transform(prices)
        
        # Extract features
        features = self.wavelet_analyzer.extract_features(coeffs)
        
        # Detect patterns using sliding window
        patterns = []
        for i in range(0, len(prices) - window_size, window_size // 2):
            window_data = prices[i:i + window_size]
            window_coeffs = coeffs[:, i:i + window_size]
            
            # Classify pattern
            pattern_type = self._classify_pattern(window_data, window_coeffs)
            
            if pattern_type:
                # Calculate pattern strength
                strength = self._calculate_pattern_strength(window_coeffs)
                
                # Extract pattern features
                pattern_features = self._extract_pattern_features(
                    window_data, window_coeffs
                )
                
                patterns.append({
                    'type': pattern_type,
                    'start_idx': i,
                    'end_idx': i + window_size,
                    'timestamp': data.iloc[i]['timestamp'] if 'timestamp' in data.columns else i,
                    'strength': strength,
                    'features': pattern_features,
                    'price_start': window_data[0],
                    'price_end': window_data[-1],
                    'price_change': (window_data[-1] - window_data[0]) / window_data[0]
                })
        
        return patterns
    
    def _classify_pattern(self, prices: np.ndarray, 
                         coeffs: np.ndarray) -> Optional[str]:
        """
        Classify the pattern type based on price movement and wavelet coefficients.
        
        Args:
            prices: Price data for the window
            coeffs: Wavelet coefficients for the window
            
        Returns:
            Pattern type or None
        """
        # Calculate basic statistics
        price_change = (prices[-1] - prices[0]) / prices[0]
        price_std = np.std(prices) / np.mean(prices)
        
        # Calculate trend
        x = np.arange(len(prices))
        trend = np.polyfit(x, prices, 1)[0]
        trend_normalized = trend / np.mean(prices)
        
        # Calculate energy concentration
        energy = np.sum(np.abs(coeffs)**2, axis=1)
        dominant_scale_idx = np.argmax(energy)
        energy_concentration = energy[dominant_scale_idx] / np.sum(energy)
        
        # Classify based on rules
        if abs(trend_normalized) < self.trend_threshold * 0.5:
            # Low trend - consolidation
            if price_std < 0.02:
                return 'consolidation'
        
        elif trend_normalized > self.trend_threshold:
            # Upward trend
            if self._is_reversal_pattern(prices, 'top'):
                return 'reversal_top'
            else:
                return 'trend_up'
        
        elif trend_normalized < -self.trend_threshold:
            # Downward trend
            if self._is_reversal_pattern(prices, 'bottom'):
                return 'reversal_bottom'
            else:
                return 'trend_down'
        
        # Check for reversal patterns
        mid_point = len(prices) // 2
        first_half_trend = np.polyfit(x[:mid_point], prices[:mid_point], 1)[0]
        second_half_trend = np.polyfit(x[mid_point:], prices[mid_point:], 1)[0]
        
        if first_half_trend > 0 and second_half_trend < 0:
            return 'reversal_top'
        elif first_half_trend < 0 and second_half_trend > 0:
            return 'reversal_bottom'
        
        return None
    
    def _is_reversal_pattern(self, prices: np.ndarray, 
                            reversal_type: str) -> bool:
        """
        Check if the price pattern shows a reversal.
        
        Args:
            prices: Price data
            reversal_type: 'top' or 'bottom'
            
        Returns:
            True if reversal pattern detected
        """
        # Find extremum
        if reversal_type == 'top':
            extremum_idx = np.argmax(prices)
        else:
            extremum_idx = np.argmin(prices)
        
        # Check if extremum is in the middle portion
        if 0.3 * len(prices) < extremum_idx < 0.7 * len(prices):
            # Check trend before and after
            before_trend = np.polyfit(range(extremum_idx), prices[:extremum_idx], 1)[0]
            after_trend = np.polyfit(range(len(prices) - extremum_idx), 
                                   prices[extremum_idx:], 1)[0]
            
            if reversal_type == 'top':
                return before_trend > 0 and after_trend < 0
            else:
                return before_trend < 0 and after_trend > 0
        
        return False
    
    def _calculate_pattern_strength(self, coeffs: np.ndarray) -> float:
        """
        Calculate the strength of a pattern based on wavelet coefficients.
        
        Args:
            coeffs: Wavelet coefficients
            
        Returns:
            Pattern strength (0-1)
        """
        # Calculate energy
        energy = np.sum(np.abs(coeffs)**2)
        
        # Calculate concentration
        scale_energy = np.sum(np.abs(coeffs)**2, axis=1)
        max_scale_energy = np.max(scale_energy)
        concentration = max_scale_energy / np.sum(scale_energy)
        
        # Normalize energy (using empirical bounds)
        normalized_energy = np.tanh(energy / 1000)
        
        # Combine metrics
        strength = 0.7 * normalized_energy + 0.3 * concentration
        
        return float(np.clip(strength, 0, 1))
    
    def _extract_pattern_features(self, prices: np.ndarray,
                                 coeffs: np.ndarray) -> Dict[str, float]:
        """
        Extract features from a pattern.
        
        Args:
            prices: Price data
            coeffs: Wavelet coefficients
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Price-based features
        features['volatility'] = float(np.std(prices) / np.mean(prices))
        features['price_range'] = float((np.max(prices) - np.min(prices)) / np.mean(prices))
        features['trend_strength'] = float(abs(np.polyfit(range(len(prices)), prices, 1)[0]) / np.mean(prices))
        
        # Wavelet-based features
        scale_energy = np.sum(np.abs(coeffs)**2, axis=1)
        features['dominant_scale'] = float(self.wavelet_analyzer.scales[np.argmax(scale_energy)])
        features['energy_concentration'] = float(np.max(scale_energy) / np.sum(scale_energy))
        
        # Time-based energy distribution
        time_energy = np.sum(np.abs(coeffs)**2, axis=0)
        features['energy_skewness'] = float(self._safe_skewness(time_energy))
        features['energy_kurtosis'] = float(self._safe_kurtosis(time_energy))
        
        return features
    
    def _safe_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness safely."""
        if len(data) < 3 or np.std(data) == 0:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        return float(np.mean(((data - mean) / std) ** 3))
    
    def _safe_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis safely."""
        if len(data) < 4 or np.std(data) == 0:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        return float(np.mean(((data - mean) / std) ** 4) - 3)
    
    def extract_pattern_sequences(self, patterns: List[Dict],
                                 sequence_length: int = 5) -> List[List[str]]:
        """
        Extract sequences of patterns for sequence modeling.
        
        Args:
            patterns: List of detected patterns
            sequence_length: Length of sequences to extract
            
        Returns:
            List of pattern sequences
        """
        if len(patterns) < sequence_length:
            return []
        
        # Sort patterns by start index
        sorted_patterns = sorted(patterns, key=lambda x: x['start_idx'])
        
        # Extract sequences
        sequences = []
        for i in range(len(sorted_patterns) - sequence_length + 1):
            sequence = [p['type'] for p in sorted_patterns[i:i + sequence_length]]
            sequences.append(sequence)
        
        return sequences
    
    def get_pattern_statistics(self, patterns: List[Dict]) -> Dict[str, Dict]:
        """
        Calculate statistics for detected patterns.
        
        Args:
            patterns: List of detected patterns
            
        Returns:
            Dictionary of statistics by pattern type
        """
        stats = {}
        
        for pattern_type in self.pattern_types:
            type_patterns = [p for p in patterns if p['type'] == pattern_type]
            
            if type_patterns:
                stats[pattern_type] = {
                    'count': len(type_patterns),
                    'avg_strength': np.mean([p['strength'] for p in type_patterns]),
                    'avg_price_change': np.mean([p['price_change'] for p in type_patterns]),
                    'avg_volatility': np.mean([p['features']['volatility'] for p in type_patterns]),
                    'frequency': len(type_patterns) / len(patterns)
                }
            else:
                stats[pattern_type] = {
                    'count': 0,
                    'avg_strength': 0,
                    'avg_price_change': 0,
                    'avg_volatility': 0,
                    'frequency': 0
                }
        
        return stats
