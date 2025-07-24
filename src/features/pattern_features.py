"""
Pattern Features Module

Extracts features from wavelet patterns for machine learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class PatternFeatureExtractor:
    """
    Extracts features from wavelet patterns for prediction and analysis.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.feature_names = []
        logger.info("Initialized PatternFeatureExtractor")
    
    def extract_features(self, pattern: Dict) -> np.ndarray:
        """
        Extract features from a single pattern.
        
        Args:
            pattern: Pattern dictionary with metadata
            
        Returns:
            Feature vector
        """
        features = []
        
        # Basic pattern features
        features.extend(self._extract_basic_features(pattern))
        
        # Wavelet-specific features
        if 'features' in pattern:
            features.extend(self._extract_wavelet_features(pattern['features']))
        
        # Time-based features
        features.extend(self._extract_temporal_features(pattern))
        
        return np.array(features)
    
    def extract_sequence_features(self, patterns: List[Dict]) -> np.ndarray:
        """
        Extract features from a sequence of patterns.
        
        Args:
            patterns: List of pattern dictionaries
            
        Returns:
            Feature vector for the sequence
        """
        features = []
        
        # Pattern type distribution
        features.extend(self._extract_pattern_distribution(patterns))
        
        # Transition features
        features.extend(self._extract_transition_features(patterns))
        
        # Temporal features
        features.extend(self._extract_sequence_temporal_features(patterns))
        
        # Statistical features
        features.extend(self._extract_statistical_features(patterns))
        
        return np.array(features)
    
    def _extract_basic_features(self, pattern: Dict) -> List[float]:
        """Extract basic features from pattern."""
        features = []
        
        # Pattern type encoding (one-hot)
        pattern_types = ['trend_up', 'trend_down', 'reversal_top', 
                        'reversal_bottom', 'consolidation']
        
        for ptype in pattern_types:
            features.append(1.0 if pattern.get('type') == ptype else 0.0)
        
        # Pattern strength
        features.append(pattern.get('strength', 0.0))
        
        # Price change
        features.append(pattern.get('price_change', 0.0))
        
        # Duration (if available)
        if 'start_idx' in pattern and 'end_idx' in pattern:
            duration = pattern['end_idx'] - pattern['start_idx']
            features.append(float(duration))
        else:
            features.append(0.0)
        
        return features
    
    def _extract_wavelet_features(self, pattern_features: Dict) -> List[float]:
        """Extract wavelet-specific features."""
        features = []
        
        # Core wavelet features
        features.append(pattern_features.get('volatility', 0.0))
        features.append(pattern_features.get('price_range', 0.0))
        features.append(pattern_features.get('trend_strength', 0.0))
        features.append(pattern_features.get('dominant_scale', 0.0))
        features.append(pattern_features.get('energy_concentration', 0.0))
        features.append(pattern_features.get('energy_skewness', 0.0))
        features.append(pattern_features.get('energy_kurtosis', 0.0))
        
        # Volume features if available
        features.append(pattern_features.get('volume_ratio', 1.0))
        
        return features
    
    def _extract_temporal_features(self, pattern: Dict) -> List[float]:
        """Extract time-based features."""
        features = []
        
        # Time of day features (if timestamp available)
        if 'timestamp' in pattern and pd.notna(pattern['timestamp']):
            try:
                ts = pd.Timestamp(pattern['timestamp'])
                # Hour of day (normalized)
                features.append(ts.hour / 24.0)
                # Day of week (normalized)
                features.append(ts.dayofweek / 6.0)
                # Day of month (normalized)
                features.append(ts.day / 31.0)
            except:
                features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        return features
    
    def _extract_pattern_distribution(self, patterns: List[Dict]) -> List[float]:
        """Extract pattern type distribution features."""
        features = []
        
        pattern_types = ['trend_up', 'trend_down', 'reversal_top', 
                        'reversal_bottom', 'consolidation']
        
        # Count each pattern type
        type_counts = {ptype: 0 for ptype in pattern_types}
        for pattern in patterns:
            ptype = pattern.get('type')
            if ptype in type_counts:
                type_counts[ptype] += 1
        
        # Normalize by sequence length
        seq_len = len(patterns) if patterns else 1
        for ptype in pattern_types:
            features.append(type_counts[ptype] / seq_len)
        
        return features
    
    def _extract_transition_features(self, patterns: List[Dict]) -> List[float]:
        """Extract pattern transition features."""
        features = []
        
        if len(patterns) < 2:
            return [0.0, 0.0, 0.0]
        
        # Count transitions
        transitions = 0
        same_type_runs = []
        current_run = 1
        
        for i in range(1, len(patterns)):
            if patterns[i].get('type') != patterns[i-1].get('type'):
                transitions += 1
                same_type_runs.append(current_run)
                current_run = 1
            else:
                current_run += 1
        
        same_type_runs.append(current_run)
        
        # Transition rate
        features.append(transitions / (len(patterns) - 1))
        
        # Average run length
        features.append(np.mean(same_type_runs) if same_type_runs else 0.0)
        
        # Max run length
        features.append(max(same_type_runs) if same_type_runs else 0.0)
        
        return features
    
    def _extract_sequence_temporal_features(self, patterns: List[Dict]) -> List[float]:
        """Extract temporal features from pattern sequence."""
        features = []
        
        if not patterns:
            return [0.0, 0.0, 0.0]
        
        # Pattern frequency (patterns per time unit)
        if len(patterns) >= 2:
            # Estimate time span
            if 'timestamp' in patterns[0] and 'timestamp' in patterns[-1]:
                try:
                    start_time = pd.Timestamp(patterns[0]['timestamp'])
                    end_time = pd.Timestamp(patterns[-1]['timestamp'])
                    time_span = (end_time - start_time).total_seconds() / 3600  # hours
                    
                    if time_span > 0:
                        pattern_frequency = len(patterns) / time_span
                    else:
                        pattern_frequency = 0.0
                except:
                    pattern_frequency = 0.0
            else:
                pattern_frequency = 0.0
        else:
            pattern_frequency = 0.0
        
        features.append(pattern_frequency)
        
        # Pattern strength trend
        strengths = [p.get('strength', 0.0) for p in patterns]
        if len(strengths) >= 2:
            # Linear trend of strength
            x = np.arange(len(strengths))
            trend = np.polyfit(x, strengths, 1)[0]
            features.append(trend)
        else:
            features.append(0.0)
        
        # Recent pattern strength (last pattern)
        features.append(patterns[-1].get('strength', 0.0) if patterns else 0.0)
        
        return features
    
    def _extract_statistical_features(self, patterns: List[Dict]) -> List[float]:
        """Extract statistical features from pattern sequence."""
        features = []
        
        if not patterns:
            return [0.0] * 8
        
        # Price change statistics
        price_changes = [p.get('price_change', 0.0) for p in patterns]
        features.append(np.mean(price_changes))
        features.append(np.std(price_changes) if len(price_changes) > 1 else 0.0)
        features.append(np.min(price_changes))
        features.append(np.max(price_changes))
        
        # Strength statistics
        strengths = [p.get('strength', 0.0) for p in patterns]
        features.append(np.mean(strengths))
        features.append(np.std(strengths) if len(strengths) > 1 else 0.0)
        
        # Volatility statistics (if available)
        volatilities = []
        for p in patterns:
            if 'features' in p and 'volatility' in p['features']:
                volatilities.append(p['features']['volatility'])
        
        if volatilities:
            features.append(np.mean(volatilities))
            features.append(np.std(volatilities) if len(volatilities) > 1 else 0.0)
        else:
            features.extend([0.0, 0.0])
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        names = []
        
        # Basic features
        pattern_types = ['trend_up', 'trend_down', 'reversal_top', 
                        'reversal_bottom', 'consolidation']
        names.extend([f'is_{ptype}' for ptype in pattern_types])
        names.extend(['strength', 'price_change', 'duration'])
        
        # Wavelet features
        names.extend(['volatility', 'price_range', 'trend_strength', 
                     'dominant_scale', 'energy_concentration', 
                     'energy_skewness', 'energy_kurtosis', 'volume_ratio'])
        
        # Temporal features
        names.extend(['hour_of_day', 'day_of_week', 'day_of_month'])
        
        # Sequence features
        names.extend([f'{ptype}_ratio' for ptype in pattern_types])
        names.extend(['transition_rate', 'avg_run_length', 'max_run_length'])
        names.extend(['pattern_frequency', 'strength_trend', 'recent_strength'])
        
        # Statistical features
        names.extend(['mean_price_change', 'std_price_change', 
                     'min_price_change', 'max_price_change',
                     'mean_strength', 'std_strength',
                     'mean_volatility', 'std_volatility'])
        
        return names
    
    def create_feature_matrix(self, pattern_sequences: List[List[Dict]]) -> pd.DataFrame:
        """
        Create feature matrix from multiple pattern sequences.
        
        Args:
            pattern_sequences: List of pattern sequences
            
        Returns:
            DataFrame with features
        """
        features_list = []
        
        for sequence in pattern_sequences:
            features = self.extract_sequence_features(sequence)
            features_list.append(features)
        
        # Create DataFrame
        feature_names = self.get_feature_names()
        
        # Adjust feature names if needed
        n_features = len(features_list[0]) if features_list else 0
        if n_features != len(feature_names):
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        df = pd.DataFrame(features_list, columns=feature_names)
        
        return df
