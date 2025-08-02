"""
Advanced Pattern Detector with SAX and Shapelet Integration

Enhances wavelet pattern detection with SAX discretization and shapelet extraction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import signal, stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

from .pattern_detector import WaveletPatternDetector
from ..advanced.time_series_integration import (
    SAXTransformer, SAXConfig,
    TimeSeriesSimilaritySearch, SimilaritySearchConfig,
    AdvancedFeatureExtractor
)

logger = logging.getLogger(__name__)


class ShapeletExtractor:
    """
    Extract discriminative shapelets from time series for pattern classification
    """
    
    def __init__(self, min_length: int = 10, max_length: int = 50, 
                 n_shapelets: int = 20):
        self.min_length = min_length
        self.max_length = max_length
        self.n_shapelets = n_shapelets
        self.shapelets = []
        
    def fit(self, X: List[np.ndarray], y: np.ndarray):
        """
        Extract shapelets from labeled time series
        
        Args:
            X: List of time series
            y: Class labels
        """
        candidates = []
        
        # Generate shapelet candidates
        for series_idx, series in enumerate(X):
            for length in range(self.min_length, 
                              min(self.max_length + 1, len(series) + 1)):
                for start in range(len(series) - length + 1):
                    shapelet = series[start:start + length]
                    
                    # Calculate information gain
                    info_gain = self._calculate_information_gain(
                        shapelet, X, y
                    )
                    
                    candidates.append({
                        'shapelet': shapelet,
                        'info_gain': info_gain,
                        'source_idx': series_idx,
                        'start': start,
                        'length': length
                    })
        
        # Select top shapelets
        candidates.sort(key=lambda x: x['info_gain'], reverse=True)
        self.shapelets = candidates[:self.n_shapelets]
        
        logger.info(f"Extracted {len(self.shapelets)} shapelets")
        
    def _calculate_information_gain(self, shapelet: np.ndarray, 
                                   X: List[np.ndarray], y: np.ndarray) -> float:
        """Calculate information gain of a shapelet"""
        # Calculate distances
        distances = []
        for series in X:
            dist = self._min_distance(shapelet, series)
            distances.append(dist)
            
        distances = np.array(distances)
        
        # Find optimal split threshold
        best_gain = 0
        for threshold in np.percentile(distances, [25, 50, 75]):
            # Split data
            left_mask = distances <= threshold
            right_mask = ~left_mask
            
            # Calculate information gain
            gain = self._entropy(y) - (
                np.sum(left_mask) / len(y) * self._entropy(y[left_mask]) +
                np.sum(right_mask) / len(y) * self._entropy(y[right_mask])
            )
            
            best_gain = max(best_gain, gain)
            
        return best_gain
        
    def _min_distance(self, shapelet: np.ndarray, series: np.ndarray) -> float:
        """Calculate minimum distance between shapelet and series"""
        min_dist = np.inf
        
        for i in range(len(series) - len(shapelet) + 1):
            subsequence = series[i:i + len(shapelet)]
            dist = np.linalg.norm(shapelet - subsequence)
            min_dist = min(min_dist, dist)
            
        return min_dist
        
    def _entropy(self, y: np.ndarray) -> float:
        """Calculate entropy of labels"""
        if len(y) == 0:
            return 0
            
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-10))
        
    def transform(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Transform time series to shapelet-based features
        
        Args:
            X: List of time series
            
        Returns:
            Feature matrix (n_samples, n_shapelets)
        """
        features = np.zeros((len(X), len(self.shapelets)))
        
        for i, series in enumerate(X):
            for j, shapelet_info in enumerate(self.shapelets):
                shapelet = shapelet_info['shapelet']
                features[i, j] = self._min_distance(shapelet, series)
                
        return features


class AdvancedWaveletPatternDetector(WaveletPatternDetector):
    """
    Enhanced pattern detector combining wavelets, SAX, and shapelets
    """
    
    def __init__(self, wavelet: str = 'morl', scales: Optional[np.ndarray] = None,
                 use_sax: bool = True, use_shapelets: bool = True):
        super().__init__(wavelet, scales)
        
        self.use_sax = use_sax
        self.use_shapelets = use_shapelets
        
        # Initialize SAX transformer
        if use_sax:
            self.sax_transformer = SAXTransformer(
                SAXConfig(n_segments=20, alphabet_size=5)
            )
            
        # Initialize shapelet extractor
        if use_shapelets:
            self.shapelet_extractor = ShapeletExtractor()
            
        # Initialize similarity search
        self.similarity_search = TimeSeriesSimilaritySearch(
            SimilaritySearchConfig(method='dtw', top_k=5)
        )
        
        # Pattern motif discovery
        self.motifs = {}
        
    def detect_patterns_advanced(self, data: pd.DataFrame,
                               price_col: str = 'close',
                               window_size: int = 50,
                               extract_motifs: bool = True) -> Dict[str, Any]:
        """
        Advanced pattern detection with multiple techniques
        
        Args:
            data: DataFrame with price data
            price_col: Column name for price data
            window_size: Window size for pattern detection
            extract_motifs: Whether to extract pattern motifs
            
        Returns:
            Dictionary containing patterns, motifs, and analysis results
        """
        # Basic wavelet pattern detection
        patterns = self.detect_patterns(data, price_col, window_size)
        
        # Enhance with SAX analysis
        if self.use_sax:
            patterns = self._enhance_with_sax(patterns, data[price_col].values)
            
        # Extract motifs if requested
        if extract_motifs:
            self.motifs = self._extract_motifs(patterns)
            
        # Build similarity index
        self._build_similarity_index(patterns)
        
        # Cluster patterns
        pattern_clusters = self._cluster_patterns(patterns)
        
        # Extract transition probabilities
        transitions = self._analyze_pattern_transitions(patterns)
        
        return {
            'patterns': patterns,
            'motifs': self.motifs,
            'clusters': pattern_clusters,
            'transitions': transitions,
            'statistics': self.get_pattern_statistics(patterns)
        }
        
    def _enhance_with_sax(self, patterns: List[Dict], 
                         full_series: np.ndarray) -> List[Dict]:
        """Enhance patterns with SAX representations"""
        for pattern in patterns:
            start_idx = pattern['start_idx']
            end_idx = pattern['end_idx']
            
            # Extract pattern data
            pattern_data = full_series[start_idx:end_idx]
            
            # Generate SAX representation
            sax_string = self.sax_transformer.transform(pattern_data)
            pattern['sax'] = sax_string
            
            # Add SAX-based features
            pattern['features']['sax_complexity'] = len(set(sax_string))
            pattern['features']['sax_transitions'] = sum(
                1 for i in range(len(sax_string)-1) 
                if sax_string[i] != sax_string[i+1]
            ) / (len(sax_string) - 1)
            
        return patterns
        
    def _extract_motifs(self, patterns: List[Dict]) -> Dict[str, List[Dict]]:
        """Extract recurring motifs from patterns"""
        motifs = {}
        
        # Group by pattern type
        for pattern_type in self.pattern_types:
            type_patterns = [p for p in patterns if p['type'] == pattern_type]
            
            if len(type_patterns) < 3:
                continue
                
            # Extract SAX strings
            sax_strings = [p.get('sax', '') for p in type_patterns if 'sax' in p]
            
            if not sax_strings:
                continue
                
            # Find frequent SAX patterns
            sax_counts = {}
            for sax in sax_strings:
                # Consider substrings as well
                for length in range(5, len(sax) + 1):
                    for i in range(len(sax) - length + 1):
                        substring = sax[i:i+length]
                        sax_counts[substring] = sax_counts.get(substring, 0) + 1
                        
            # Select frequent motifs
            frequent_motifs = [
                (sax, count) for sax, count in sax_counts.items()
                if count >= 3 and len(sax) >= 5
            ]
            frequent_motifs.sort(key=lambda x: x[1], reverse=True)
            
            motifs[pattern_type] = [
                {
                    'sax': sax,
                    'frequency': count,
                    'length': len(sax),
                    'examples': [p for p in type_patterns if sax in p.get('sax', '')][:3]
                }
                for sax, count in frequent_motifs[:5]
            ]
            
        return motifs
        
    def _build_similarity_index(self, patterns: List[Dict]):
        """Build similarity index for fast pattern retrieval"""
        # Prepare patterns for indexing
        indexed_patterns = []
        for i, pattern in enumerate(patterns):
            if 'price_start' in pattern and 'price_end' in pattern:
                # Reconstruct approximate pattern data
                length = pattern['end_idx'] - pattern['start_idx']
                pattern_data = np.linspace(
                    pattern['price_start'], 
                    pattern['price_end'], 
                    length
                )
                
                indexed_patterns.append({
                    'data': pattern_data,
                    'metadata': {
                        'type': pattern['type'],
                        'strength': pattern['strength'],
                        'timestamp': pattern.get('timestamp', i)
                    }
                })
                
        self.similarity_search.index_patterns(indexed_patterns)
        
    def _cluster_patterns(self, patterns: List[Dict], n_clusters: int = 5) -> Dict:
        """Cluster patterns based on features"""
        if len(patterns) < n_clusters:
            return {}
            
        # Extract features for clustering
        feature_matrix = []
        valid_patterns = []
        
        for pattern in patterns:
            features = pattern.get('features', {})
            if features:
                feature_vector = [
                    features.get('volatility', 0),
                    features.get('trend_strength', 0),
                    features.get('energy_concentration', 0),
                    features.get('dominant_scale', 0),
                    pattern.get('strength', 0)
                ]
                feature_matrix.append(feature_vector)
                valid_patterns.append(pattern)
                
        if len(feature_matrix) < n_clusters:
            return {}
            
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_matrix)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Organize results
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(valid_patterns[i])
            
        # Calculate cluster statistics
        cluster_stats = {}
        for label, cluster_patterns in clusters.items():
            cluster_stats[label] = {
                'size': len(cluster_patterns),
                'avg_strength': np.mean([p['strength'] for p in cluster_patterns]),
                'dominant_type': max(
                    set([p['type'] for p in cluster_patterns]),
                    key=[p['type'] for p in cluster_patterns].count
                ),
                'center': kmeans.cluster_centers_[label].tolist()
            }
            
        return {
            'clusters': clusters,
            'statistics': cluster_stats,
            'n_clusters': n_clusters
        }
        
    def _analyze_pattern_transitions(self, patterns: List[Dict]) -> Dict:
        """Analyze transitions between pattern types"""
        if len(patterns) < 2:
            return {}
            
        # Sort by time
        sorted_patterns = sorted(patterns, key=lambda x: x['start_idx'])
        
        # Count transitions
        transitions = {}
        for i in range(len(sorted_patterns) - 1):
            current_type = sorted_patterns[i]['type']
            next_type = sorted_patterns[i + 1]['type']
            
            if current_type not in transitions:
                transitions[current_type] = {}
            transitions[current_type][next_type] = \
                transitions[current_type].get(next_type, 0) + 1
                
        # Convert to probabilities
        transition_probs = {}
        for from_type, to_counts in transitions.items():
            total = sum(to_counts.values())
            transition_probs[from_type] = {
                to_type: count / total
                for to_type, count in to_counts.items()
            }
            
        return transition_probs
        
    def find_similar_patterns(self, query_pattern: np.ndarray,
                            pattern_type: Optional[str] = None) -> List[Dict]:
        """
        Find patterns similar to a query
        
        Args:
            query_pattern: Query time series pattern
            pattern_type: Optional filter by pattern type
            
        Returns:
            List of similar patterns
        """
        filter_criteria = {'type': pattern_type} if pattern_type else None
        return self.similarity_search.search(query_pattern, filter_criteria)
        
    def train_pattern_classifier(self, labeled_patterns: List[Tuple[np.ndarray, str]]):
        """
        Train shapelet-based pattern classifier
        
        Args:
            labeled_patterns: List of (pattern_data, pattern_type) tuples
        """
        if not self.use_shapelets:
            logger.warning("Shapelet extraction is disabled")
            return
            
        X = [pattern for pattern, _ in labeled_patterns]
        y = np.array([label for _, label in labeled_patterns])
        
        # Extract shapelets
        self.shapelet_extractor.fit(X, y)
        
        logger.info("Trained shapelet-based pattern classifier")
        
    def classify_pattern(self, pattern_data: np.ndarray) -> Dict[str, float]:
        """
        Classify a pattern using multiple methods
        
        Args:
            pattern_data: Time series pattern to classify
            
        Returns:
            Dictionary of pattern type probabilities
        """
        # Wavelet-based classification
        coeffs, _ = self.wavelet_analyzer.transform(pattern_data)
        wavelet_type = self._classify_pattern(pattern_data, coeffs)
        
        # SAX-based similarity
        sax_similarities = {}
        if self.use_sax:
            pattern_sax = self.sax_transformer.transform(pattern_data)
            
            for motif_type, motifs in self.motifs.items():
                max_sim = 0
                for motif in motifs:
                    sim = self._sax_similarity(pattern_sax, motif['sax'])
                    max_sim = max(max_sim, sim)
                sax_similarities[motif_type] = max_sim
                
        # Combine predictions
        predictions = {}
        
        # Base prediction from wavelets
        if wavelet_type:
            predictions[wavelet_type] = 0.6
            
        # Add SAX similarities
        if sax_similarities:
            total_sim = sum(sax_similarities.values())
            if total_sim > 0:
                for ptype, sim in sax_similarities.items():
                    predictions[ptype] = predictions.get(ptype, 0) + 0.4 * (sim / total_sim)
                    
        # Normalize
        total = sum(predictions.values())
        if total > 0:
            predictions = {k: v/total for k, v in predictions.items()}
            
        return predictions
        
    def _sax_similarity(self, sax1: str, sax2: str) -> float:
        """Calculate similarity between SAX strings"""
        if len(sax1) != len(sax2):
            # Pad shorter string
            if len(sax1) < len(sax2):
                sax1 = sax1 + sax1[-1] * (len(sax2) - len(sax1))
            else:
                sax2 = sax2 + sax2[-1] * (len(sax1) - len(sax2))
                
        matches = sum(1 for a, b in zip(sax1, sax2) if a == b)
        return matches / len(sax1)
