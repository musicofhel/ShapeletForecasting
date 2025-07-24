"""
Pattern-based Feature Extractor

This module extracts features from wavelet patterns, DTW similarities,
and pattern clusters for use in machine learning models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path
import h5py
import pickle

# Import from previous sprints
import sys
sys.path.append(str(Path(__file__).parent.parent))
from wavelet_analysis import WaveletAnalyzer, ShapeletExtractor
from dtw import DTWCalculator, SimilarityEngine, PatternClusterer

logger = logging.getLogger(__name__)


class PatternFeatureExtractor:
    """
    Extracts pattern-based features from time series using wavelets and DTW.
    
    Features include:
    - Wavelet coefficient statistics
    - Pattern similarity scores
    - Cluster membership and distances
    - Pattern occurrence frequencies
    - Temporal pattern features
    """
    
    def __init__(self,
                 wavelet: str = 'morl',
                 scales: Optional[np.ndarray] = None,
                 dtw_type: str = 'fast',
                 n_patterns: int = 20,
                 pattern_length: int = 30):
        """
        Initialize the pattern feature extractor.
        
        Parameters:
        -----------
        wavelet : str
            Wavelet type for CWT
        scales : np.ndarray, optional
            Scales for wavelet transform
        dtw_type : str
            Type of DTW algorithm ('standard', 'fast', 'constrained')
        n_patterns : int
            Number of top patterns to use for features
        pattern_length : int
            Length of patterns for extraction
        """
        self.wavelet = wavelet
        self.scales = scales if scales is not None else np.arange(2, 64)
        self.dtw_type = dtw_type
        self.n_patterns = n_patterns
        self.pattern_length = pattern_length
        
        # Initialize components
        self.wavelet_analyzer = WaveletAnalyzer()
        self.shapelet_extractor = ShapeletExtractor()
        self.dtw_calculator = DTWCalculator()
        self.similarity_engine = SimilarityEngine()
        self.pattern_clusterer = PatternClusterer(n_clusters=self.n_patterns)
        
        # Storage for fitted patterns
        self.reference_patterns = None
        self.cluster_centers = None
        self.pattern_labels = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Optional[np.ndarray] = None) -> 'PatternFeatureExtractor':
        """
        Fit the feature extractor by learning reference patterns.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_timesteps)
            Time series data
        y : array-like of shape (n_samples,), optional
            Labels for supervised pattern extraction
        
        Returns:
        --------
        self : PatternFeatureExtractor
        """
        logger.info("Fitting pattern feature extractor...")
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Extract patterns from training data
        self.reference_patterns = self._extract_reference_patterns(X, y)
        
        # Compute similarity matrix for reference patterns
        pattern_labels = [f"pattern_{i}" for i in range(len(self.reference_patterns))]
        sim_results = self.similarity_engine.compute_similarity_matrix(
            self.reference_patterns, pattern_labels
        )
        
        # Cluster patterns
        cluster_results = self.pattern_clusterer.fit_predict(
            self.reference_patterns,
            similarity_matrix=sim_results['similarity_matrix']
        )
        
        self.cluster_centers = cluster_results['cluster_centers']
        self.pattern_labels = cluster_results['labels']
        
        logger.info(f"Fitted with {len(self.reference_patterns)} reference patterns "
                   f"in {len(self.cluster_centers)} clusters")
        
        return self
        
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """
        Transform time series into pattern-based features.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_timesteps)
            Time series data to transform
            
        Returns:
        --------
        features : pd.DataFrame
            Extracted features
        """
        if self.reference_patterns is None:
            raise ValueError("Feature extractor must be fitted before transform")
            
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        features_list = []
        
        for i, ts in enumerate(X):
            logger.debug(f"Extracting features for sample {i+1}/{len(X)}")
            
            # Extract features for this time series
            features = {}
            
            # 1. Wavelet features
            wavelet_features = self._extract_wavelet_features(ts)
            features.update(wavelet_features)
            
            # 2. Pattern similarity features
            similarity_features = self._extract_similarity_features(ts)
            features.update(similarity_features)
            
            # 3. Cluster-based features
            cluster_features = self._extract_cluster_features(ts)
            features.update(cluster_features)
            
            # 4. Temporal pattern features
            temporal_features = self._extract_temporal_features(ts)
            features.update(temporal_features)
            
            features_list.append(features)
            
        return pd.DataFrame(features_list)
        
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame],
                     y: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
        
    def _extract_reference_patterns(self, X: np.ndarray, 
                                  y: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """Extract reference patterns from training data."""
        patterns = []
        
        for ts in X:
            # Extract sliding window patterns
            for i in range(0, len(ts) - self.pattern_length + 1, self.pattern_length // 2):
                pattern = ts[i:i + self.pattern_length]
                if len(pattern) == self.pattern_length:
                    patterns.append(pattern)
                    
        # If we have labels, use shapelet extraction
        if y is not None:
            shapelet_extractor = ShapeletExtractor(
                min_length=self.pattern_length,
                max_length=self.pattern_length
            )
            shapelets = shapelet_extractor.extract_shapelets(X, y, n_shapelets=self.n_patterns)
            patterns.extend([s['shapelet'] for s in shapelets])
            
        # Select top patterns based on variance
        if len(patterns) > self.n_patterns:
            variances = [np.var(p) for p in patterns]
            top_indices = np.argsort(variances)[-self.n_patterns:]
            patterns = [patterns[i] for i in top_indices]
            
        return patterns
        
    def _extract_wavelet_features(self, ts: np.ndarray) -> Dict[str, float]:
        """Extract wavelet-based features."""
        # Compute wavelet transform
        coeffs, freqs = self.wavelet_analyzer.transform(ts)
        
        # Extract wavelet features
        wavelet_features = self.wavelet_analyzer.extract_features(coeffs)
        
        # Add additional statistics
        features = {
            'wavelet_energy': wavelet_features.get('total_energy', 0),
            'wavelet_entropy': wavelet_features.get('wavelet_entropy', 0),
            'wavelet_dominant_scale': wavelet_features.get('dominant_scale', 0),
            'wavelet_dominant_freq': wavelet_features.get('dominant_frequency', 0),
        }
        
        # Scale-wise statistics
        scale_powers = wavelet_features.get('scale_powers', [])
        if scale_powers:
            features['wavelet_scale_mean'] = np.mean(scale_powers)
            features['wavelet_scale_std'] = np.std(scale_powers)
            features['wavelet_scale_skew'] = stats.skew(scale_powers)
            features['wavelet_scale_kurt'] = stats.kurtosis(scale_powers)
            
        return features
        
    def _extract_similarity_features(self, ts: np.ndarray) -> Dict[str, float]:
        """Extract DTW similarity features to reference patterns."""
        features = {}
        
        # Extract patterns from current time series
        current_patterns = []
        for i in range(0, len(ts) - self.pattern_length + 1, self.pattern_length // 2):
            pattern = ts[i:i + self.pattern_length]
            if len(pattern) == self.pattern_length:
                current_patterns.append(pattern)
                
        if not current_patterns:
            return features
            
        # Compute similarities to reference patterns
        similarities = []
        for ref_pattern in self.reference_patterns:
            # Find best matching pattern in current series
            best_sim = float('inf')
            for curr_pattern in current_patterns:
                result = self.dtw_calculator.compute(curr_pattern, ref_pattern)
                if result.normalized_distance < best_sim:
                    best_sim = result.normalized_distance
            similarities.append(best_sim)
            
        # Aggregate similarity statistics
        features['dtw_min_distance'] = np.min(similarities)
        features['dtw_mean_distance'] = np.mean(similarities)
        features['dtw_std_distance'] = np.std(similarities)
        features['dtw_median_distance'] = np.median(similarities)
        
        # Top-k similarities
        sorted_sims = sorted(similarities)
        for k in [1, 3, 5]:
            if k <= len(sorted_sims):
                features[f'dtw_top{k}_mean'] = np.mean(sorted_sims[:k])
                
        return features
        
    def _extract_cluster_features(self, ts: np.ndarray) -> Dict[str, float]:
        """Extract cluster-based features."""
        features = {}
        
        if self.cluster_centers is None:
            return features
            
        # Extract patterns from current time series
        current_patterns = []
        for i in range(0, len(ts) - self.pattern_length + 1, self.pattern_length // 2):
            pattern = ts[i:i + self.pattern_length]
            if len(pattern) == self.pattern_length:
                current_patterns.append(pattern)
                
        if not current_patterns:
            return features
            
        # Compute distances to cluster centers
        cluster_distances = []
        for center in self.cluster_centers:
            # Find best matching pattern
            best_dist = float('inf')
            for pattern in current_patterns:
                result = self.dtw_calculator.compute(pattern, center)
                if result.normalized_distance < best_dist:
                    best_dist = result.normalized_distance
            cluster_distances.append(best_dist)
            
        # Cluster membership features
        nearest_cluster = np.argmin(cluster_distances)
        features['nearest_cluster'] = nearest_cluster
        features['nearest_cluster_distance'] = cluster_distances[nearest_cluster]
        
        # Distance to each cluster
        for i, dist in enumerate(cluster_distances):
            features[f'cluster_{i}_distance'] = dist
            
        # Cluster distance statistics
        features['cluster_dist_mean'] = np.mean(cluster_distances)
        features['cluster_dist_std'] = np.std(cluster_distances)
        features['cluster_dist_ratio'] = (cluster_distances[nearest_cluster] / 
                                         np.mean(cluster_distances))
        
        return features
        
    def _extract_temporal_features(self, ts: np.ndarray) -> Dict[str, float]:
        """Extract temporal pattern features."""
        features = {}
        
        # Pattern occurrence over time
        pattern_times = []
        pattern_values = []
        
        for i in range(0, len(ts) - self.pattern_length + 1, self.pattern_length // 4):
            pattern = ts[i:i + self.pattern_length]
            if len(pattern) == self.pattern_length:
                # Find best matching reference pattern
                best_sim = float('inf')
                best_idx = 0
                for j, ref_pattern in enumerate(self.reference_patterns):
                    result = self.dtw_calculator.compute(pattern, ref_pattern)
                    if result.normalized_distance < best_sim:
                        best_sim = result.normalized_distance
                        best_idx = j
                        
                pattern_times.append(i / len(ts))  # Normalized time
                pattern_values.append(best_idx)
                
        if pattern_times:
            # Pattern transition statistics
            transitions = np.diff(pattern_values)
            features['pattern_transitions'] = len(np.where(transitions != 0)[0])
            features['pattern_stability'] = 1 - (features['pattern_transitions'] / 
                                               max(len(pattern_values) - 1, 1))
            
            # Dominant pattern
            unique, counts = np.unique(pattern_values, return_counts=True)
            dominant_idx = unique[np.argmax(counts)]
            features['dominant_pattern'] = dominant_idx
            features['dominant_pattern_freq'] = np.max(counts) / len(pattern_values)
            
            # Pattern diversity
            features['pattern_diversity'] = len(unique) / self.n_patterns
            
            # Temporal trend of patterns
            if len(pattern_times) > 1:
                # Linear trend of pattern indices over time
                slope, _, r_value, _, _ = stats.linregress(pattern_times, pattern_values)
                features['pattern_trend_slope'] = slope
                features['pattern_trend_r2'] = r_value ** 2
                
        return features
        
    def save_reference_patterns(self, filepath: str):
        """Save fitted reference patterns and cluster information."""
        data = {
            'reference_patterns': self.reference_patterns,
            'cluster_centers': self.cluster_centers,
            'pattern_labels': self.pattern_labels,
            'config': {
                'wavelet': self.wavelet,
                'scales': self.scales,
                'dtw_type': self.dtw_type,
                'n_patterns': self.n_patterns,
                'pattern_length': self.pattern_length
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        logger.info(f"Saved reference patterns to {filepath}")
        
    def load_reference_patterns(self, filepath: str):
        """Load fitted reference patterns and cluster information."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.reference_patterns = data['reference_patterns']
        self.cluster_centers = data['cluster_centers']
        self.pattern_labels = data['pattern_labels']
        
        # Update configuration
        config = data['config']
        self.wavelet = config['wavelet']
        self.scales = config['scales']
        self.dtw_type = config['dtw_type']
        self.n_patterns = config['n_patterns']
        self.pattern_length = config['pattern_length']
        
        # Reinitialize components with loaded config
        self.wavelet_analyzer = WaveletAnalyzer(wavelet=self.wavelet, scales=self.scales)
        self.dtw_calculator = DTWCalculator()
        self.similarity_engine = SimilarityEngine(dtw_type=self.dtw_type)
        
        logger.info(f"Loaded reference patterns from {filepath}")
        
    def get_feature_names(self) -> List[str]:
        """Get names of all features that will be extracted."""
        feature_names = []
        
        # Wavelet features
        feature_names.extend([
            'wavelet_energy', 'wavelet_entropy', 'wavelet_dominant_scale',
            'wavelet_dominant_freq', 'wavelet_scale_mean', 'wavelet_scale_std',
            'wavelet_scale_skew', 'wavelet_scale_kurt'
        ])
        
        # Similarity features
        feature_names.extend([
            'dtw_min_distance', 'dtw_mean_distance', 'dtw_std_distance',
            'dtw_median_distance', 'dtw_top1_mean', 'dtw_top3_mean', 'dtw_top5_mean'
        ])
        
        # Cluster features
        if self.cluster_centers is not None:
            n_clusters = len(self.cluster_centers)
            feature_names.extend([
                'nearest_cluster', 'nearest_cluster_distance',
                'cluster_dist_mean', 'cluster_dist_std', 'cluster_dist_ratio'
            ])
            for i in range(n_clusters):
                feature_names.append(f'cluster_{i}_distance')
                
        # Temporal features
        feature_names.extend([
            'pattern_transitions', 'pattern_stability', 'dominant_pattern',
            'dominant_pattern_freq', 'pattern_diversity', 'pattern_trend_slope',
            'pattern_trend_r2'
        ])
        
        return feature_names
