"""
Wavelet Sequence Analyzer for Pattern Extraction and Transition Analysis

This module provides comprehensive wavelet pattern analysis including:
- Multi-scale wavelet extraction using CWT
- Pattern clustering and vocabulary creation
- Sequence identification and tracking
- Transition probability calculation
- Pattern template storage and matching
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import pywt
from scipy import signal
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import time
from collections import defaultdict, Counter
import pickle
import json

warnings.filterwarnings('ignore')


@dataclass
class WaveletPattern:
    """Represents a wavelet pattern with its properties"""
    pattern_id: int
    coefficients: np.ndarray
    scale: int
    timestamp: int
    cluster_id: Optional[int] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternSequence:
    """Represents a sequence of patterns"""
    sequence_id: int
    pattern_ids: List[int]
    timestamps: List[int]
    transition_probs: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WaveletSequenceAnalyzer:
    """
    Analyzes wavelet patterns in time series data and builds transition models
    """
    
    def __init__(self, 
                 wavelet: str = 'morl',
                 scales: Optional[np.ndarray] = None,
                 n_clusters: int = 10,
                 clustering_method: str = 'kmeans',
                 min_pattern_length: int = 5,
                 max_pattern_length: int = 50,
                 overlap_ratio: float = 0.5,
                 pca_components: Optional[int] = None,
                 random_state: int = 42):
        """
        Initialize the Wavelet Sequence Analyzer
        
        Args:
            wavelet: Wavelet type for CWT
            scales: Array of scales for CWT (default: 1-32)
            n_clusters: Number of pattern clusters
            clustering_method: 'kmeans' or 'dbscan'
            min_pattern_length: Minimum pattern length
            max_pattern_length: Maximum pattern length
            overlap_ratio: Overlap ratio for sliding window
            pca_components: Number of PCA components (None for no reduction)
            random_state: Random seed for reproducibility
        """
        self.wavelet = wavelet
        self.scales = scales if scales is not None else np.arange(1, 33)
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        self.overlap_ratio = overlap_ratio
        self.pca_components = pca_components
        self.random_state = random_state
        
        # Pattern storage
        self.patterns: List[WaveletPattern] = []
        self.pattern_vocabulary: Dict[int, np.ndarray] = {}
        self.pattern_sequences: List[PatternSequence] = []
        self.transition_matrix: Optional[np.ndarray] = None
        self.cluster_centers: Optional[np.ndarray] = None
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components) if pca_components else None
        
        # Performance tracking
        self.extraction_time = 0
        self.clustering_time = 0
        self.sequence_time = 0
        
    def extract_wavelet_patterns(self, 
                               data: np.ndarray,
                               timestamps: Optional[np.ndarray] = None) -> List[WaveletPattern]:
        """
        Extract wavelet patterns from time series data
        
        Args:
            data: Time series data
            timestamps: Optional timestamps for each data point
            
        Returns:
            List of extracted wavelet patterns
        """
        start_time = time.time()
        
        if timestamps is None:
            timestamps = np.arange(len(data))
            
        patterns = []
        pattern_id = 0
        
        # Perform CWT
        coefficients, frequencies = pywt.cwt(data, self.scales, self.wavelet)
        
        # Extract patterns using sliding window
        for scale_idx, scale in enumerate(self.scales):
            # Adaptive window size based on scale
            window_size = min(max(scale * 2, self.min_pattern_length), 
                            self.max_pattern_length)
            step_size = int(window_size * (1 - self.overlap_ratio))
            
            for start_idx in range(0, len(data) - window_size + 1, step_size):
                end_idx = start_idx + window_size
                
                # Extract pattern coefficients
                pattern_coeffs = coefficients[scale_idx, start_idx:end_idx]
                
                # Check if pattern has sufficient energy
                if np.std(pattern_coeffs) > 0.01:  # Threshold for noise
                    pattern = WaveletPattern(
                        pattern_id=pattern_id,
                        coefficients=pattern_coeffs,
                        scale=scale,
                        timestamp=timestamps[start_idx],
                        metadata={
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'energy': np.sum(pattern_coeffs**2),
                            'frequency': frequencies[scale_idx]
                        }
                    )
                    patterns.append(pattern)
                    pattern_id += 1
        
        self.patterns.extend(patterns)
        self.extraction_time = time.time() - start_time
        
        return patterns
    
    def cluster_patterns(self, patterns: Optional[List[WaveletPattern]] = None) -> Dict[int, List[int]]:
        """
        Cluster similar wavelet patterns
        
        Args:
            patterns: List of patterns to cluster (uses self.patterns if None)
            
        Returns:
            Dictionary mapping cluster IDs to pattern IDs
        """
        start_time = time.time()
        
        if patterns is None:
            patterns = self.patterns
            
        if not patterns:
            raise ValueError("No patterns to cluster")
        
        # Prepare feature matrix
        features = self._prepare_pattern_features(patterns)
        
        # Apply PCA if specified
        if self.pca:
            features = self.pca.fit_transform(features)
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Perform clustering
        if self.clustering_method == 'kmeans':
            clusterer = KMeans(n_clusters=self.n_clusters, 
                              random_state=self.random_state)
        elif self.clustering_method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
        
        cluster_labels = clusterer.fit_predict(features_scaled)
        
        # Store cluster centers for KMeans
        if self.clustering_method == 'kmeans':
            self.cluster_centers = clusterer.cluster_centers_
        
        # Update patterns with cluster IDs
        cluster_mapping = defaultdict(list)
        for idx, (pattern, cluster_id) in enumerate(zip(patterns, cluster_labels)):
            pattern.cluster_id = cluster_id
            cluster_mapping[cluster_id].append(pattern.pattern_id)
        
        # Create pattern vocabulary (cluster representatives)
        self._create_pattern_vocabulary(patterns, cluster_mapping)
        
        self.clustering_time = time.time() - start_time
        
        return dict(cluster_mapping)
    
    def _prepare_pattern_features(self, patterns: List[WaveletPattern]) -> np.ndarray:
        """
        Prepare feature matrix from patterns
        
        Args:
            patterns: List of wavelet patterns
            
        Returns:
            Feature matrix
        """
        features = []
        
        for pattern in patterns:
            coeffs = pattern.coefficients
            
            # Statistical features
            pattern_features = [
                np.mean(coeffs),
                np.std(coeffs),
                np.max(coeffs),
                np.min(coeffs),
                np.percentile(coeffs, 25),
                np.percentile(coeffs, 75),
                pattern.scale,
                pattern.metadata.get('energy', 0),
                pattern.metadata.get('frequency', 0)
            ]
            
            # Add spectral features
            fft_coeffs = np.fft.fft(coeffs)
            pattern_features.extend([
                np.mean(np.abs(fft_coeffs)),
                np.std(np.abs(fft_coeffs)),
                np.argmax(np.abs(fft_coeffs))
            ])
            
            features.append(pattern_features)
        
        return np.array(features)
    
    def _create_pattern_vocabulary(self, 
                                 patterns: List[WaveletPattern],
                                 cluster_mapping: Dict[int, List[int]]) -> None:
        """
        Create pattern vocabulary from cluster representatives
        
        Args:
            patterns: List of patterns
            cluster_mapping: Mapping of cluster IDs to pattern IDs
        """
        pattern_dict = {p.pattern_id: p for p in patterns}
        
        for cluster_id, pattern_ids in cluster_mapping.items():
            if cluster_id == -1:  # Skip noise cluster in DBSCAN
                continue
                
            # Find cluster representative (closest to centroid)
            cluster_patterns = [pattern_dict[pid] for pid in pattern_ids]
            
            if self.cluster_centers is not None and cluster_id < len(self.cluster_centers):
                # Use actual cluster center for KMeans
                representative_coeffs = self._find_closest_pattern_to_center(
                    cluster_patterns, cluster_id
                )
            else:
                # Use medoid for DBSCAN
                representative_coeffs = self._find_medoid_pattern(cluster_patterns)
            
            self.pattern_vocabulary[cluster_id] = representative_coeffs
    
    def _find_closest_pattern_to_center(self, 
                                      patterns: List[WaveletPattern],
                                      cluster_id: int) -> np.ndarray:
        """Find pattern closest to cluster center"""
        features = self._prepare_pattern_features(patterns)
        if self.pca:
            features = self.pca.transform(features)
        features_scaled = self.scaler.transform(features)
        
        center = self.cluster_centers[cluster_id]
        distances = cdist([center], features_scaled)[0]
        closest_idx = np.argmin(distances)
        
        return patterns[closest_idx].coefficients
    
    def _find_medoid_pattern(self, patterns: List[WaveletPattern]) -> np.ndarray:
        """Find medoid pattern (minimizes distance to all others)"""
        # Get all coefficient arrays
        coeffs_list = [p.coefficients for p in patterns]
        
        # Find max length for padding
        max_len = max(len(coeffs) for coeffs in coeffs_list)
        
        # Pad sequences to same length
        padded_coeffs = []
        for coeffs in coeffs_list:
            if len(coeffs) < max_len:
                padded = np.pad(coeffs, (0, max_len - len(coeffs)), mode='constant')
            else:
                padded = coeffs[:max_len]
            padded_coeffs.append(padded)
        
        padded_coeffs = np.array(padded_coeffs)
        
        # Calculate pairwise distances
        distances = cdist(padded_coeffs, padded_coeffs)
        total_distances = np.sum(distances, axis=1)
        medoid_idx = np.argmin(total_distances)
        
        return patterns[medoid_idx].coefficients
    
    def identify_sequences(self, 
                         min_sequence_length: int = 2,
                         max_gap: int = 5) -> List[PatternSequence]:
        """
        Identify pattern sequences in the data
        
        Args:
            min_sequence_length: Minimum number of patterns in a sequence
            max_gap: Maximum timestamp gap between consecutive patterns
            
        Returns:
            List of identified pattern sequences
        """
        start_time = time.time()
        
        if not self.patterns:
            raise ValueError("No patterns available for sequence identification")
        
        # Sort patterns by timestamp
        sorted_patterns = sorted(self.patterns, key=lambda p: p.timestamp)
        
        sequences = []
        current_sequence = []
        sequence_id = 0
        
        for i, pattern in enumerate(sorted_patterns):
            if pattern.cluster_id is None or pattern.cluster_id == -1:
                continue
                
            if not current_sequence:
                current_sequence = [pattern]
            else:
                # Check if pattern continues the sequence
                last_pattern = current_sequence[-1]
                time_gap = pattern.timestamp - (last_pattern.timestamp + 
                                               len(last_pattern.coefficients))
                
                if time_gap <= max_gap:
                    current_sequence.append(pattern)
                else:
                    # Save current sequence if long enough
                    if len(current_sequence) >= min_sequence_length:
                        sequence = PatternSequence(
                            sequence_id=sequence_id,
                            pattern_ids=[p.pattern_id for p in current_sequence],
                            timestamps=[p.timestamp for p in current_sequence],
                            metadata={
                                'cluster_ids': [p.cluster_id for p in current_sequence],
                                'duration': current_sequence[-1].timestamp - 
                                          current_sequence[0].timestamp
                            }
                        )
                        sequences.append(sequence)
                        sequence_id += 1
                    
                    # Start new sequence
                    current_sequence = [pattern]
        
        # Don't forget the last sequence
        if len(current_sequence) >= min_sequence_length:
            sequence = PatternSequence(
                sequence_id=sequence_id,
                pattern_ids=[p.pattern_id for p in current_sequence],
                timestamps=[p.timestamp for p in current_sequence],
                metadata={
                    'cluster_ids': [p.cluster_id for p in current_sequence],
                    'duration': current_sequence[-1].timestamp - 
                              current_sequence[0].timestamp
                }
            )
            sequences.append(sequence)
        
        self.pattern_sequences = sequences
        self.sequence_time = time.time() - start_time
        
        return sequences
    
    def calculate_transition_matrix(self) -> np.ndarray:
        """
        Calculate transition probability matrix between pattern types
        
        Returns:
            Transition probability matrix
        """
        if not self.pattern_sequences:
            raise ValueError("No sequences available for transition matrix calculation")
        
        # Count transitions
        transition_counts = defaultdict(lambda: defaultdict(int))
        
        for sequence in self.pattern_sequences:
            cluster_ids = sequence.metadata['cluster_ids']
            
            for i in range(len(cluster_ids) - 1):
                from_cluster = cluster_ids[i]
                to_cluster = cluster_ids[i + 1]
                transition_counts[from_cluster][to_cluster] += 1
        
        # Get unique cluster IDs
        all_clusters = sorted(set(
            cluster_id 
            for sequence in self.pattern_sequences 
            for cluster_id in sequence.metadata['cluster_ids']
        ))
        
        n_clusters = len(all_clusters)
        cluster_to_idx = {cluster: idx for idx, cluster in enumerate(all_clusters)}
        
        # Build transition matrix
        transition_matrix = np.zeros((n_clusters, n_clusters))
        
        for from_cluster in all_clusters:
            from_idx = cluster_to_idx[from_cluster]
            total_transitions = sum(transition_counts[from_cluster].values())
            
            if total_transitions > 0:
                for to_cluster, count in transition_counts[from_cluster].items():
                    to_idx = cluster_to_idx[to_cluster]
                    transition_matrix[from_idx, to_idx] = count / total_transitions
            else:
                # Self-transition if no other transitions observed
                transition_matrix[from_idx, from_idx] = 1.0
        
        self.transition_matrix = transition_matrix
        
        # Store cluster index mapping
        self.cluster_to_idx = cluster_to_idx
        self.idx_to_cluster = {idx: cluster for cluster, idx in cluster_to_idx.items()}
        
        return transition_matrix
    
    def match_pattern(self, 
                     query_coeffs: np.ndarray,
                     scale: int,
                     threshold: float = 0.8) -> Optional[int]:
        """
        Match a query pattern to the pattern vocabulary
        
        Args:
            query_coeffs: Query wavelet coefficients
            scale: Scale of the query pattern
            threshold: Similarity threshold (0-1)
            
        Returns:
            Matched cluster ID or None
        """
        if not self.pattern_vocabulary:
            raise ValueError("Pattern vocabulary not created yet")
        
        best_match = None
        best_similarity = -1
        
        for cluster_id, template_coeffs in self.pattern_vocabulary.items():
            # Compare patterns of similar length
            min_len = min(len(query_coeffs), len(template_coeffs))
            
            # Truncate to same length
            query_truncated = query_coeffs[:min_len]
            template_truncated = template_coeffs[:min_len]
            
            # Calculate similarity (correlation)
            if np.std(query_truncated) > 0 and np.std(template_truncated) > 0:
                similarity = np.corrcoef(query_truncated, template_truncated)[0, 1]
                
                if similarity > best_similarity and similarity >= threshold:
                    best_similarity = similarity
                    best_match = cluster_id
        
        return best_match
    
    def predict_next_pattern(self, 
                           current_pattern_id: int,
                           n_predictions: int = 1) -> List[Tuple[int, float]]:
        """
        Predict next pattern(s) based on transition probabilities
        
        Args:
            current_pattern_id: Current pattern's cluster ID
            n_predictions: Number of predictions to return
            
        Returns:
            List of (cluster_id, probability) tuples
        """
        if self.transition_matrix is None:
            raise ValueError("Transition matrix not calculated yet")
        
        if current_pattern_id not in self.cluster_to_idx:
            raise ValueError(f"Unknown cluster ID: {current_pattern_id}")
        
        current_idx = self.cluster_to_idx[current_pattern_id]
        transition_probs = self.transition_matrix[current_idx]
        
        # Get top n predictions
        top_indices = np.argsort(transition_probs)[-n_predictions:][::-1]
        
        predictions = []
        for idx in top_indices:
            if transition_probs[idx] > 0:
                cluster_id = self.idx_to_cluster[idx]
                probability = transition_probs[idx]
                predictions.append((cluster_id, probability))
        
        return predictions
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about extracted patterns
        
        Returns:
            Dictionary of statistics
        """
        if not self.patterns:
            return {}
        
        cluster_counts = Counter(p.cluster_id for p in self.patterns 
                               if p.cluster_id is not None and p.cluster_id != -1)
        
        sequence_lengths = [len(seq.pattern_ids) for seq in self.pattern_sequences]
        
        stats = {
            'total_patterns': len(self.patterns),
            'unique_clusters': len(self.pattern_vocabulary),
            'total_sequences': len(self.pattern_sequences),
            'avg_sequence_length': np.mean(sequence_lengths) if sequence_lengths else 0,
            'max_sequence_length': max(sequence_lengths) if sequence_lengths else 0,
            'min_sequence_length': min(sequence_lengths) if sequence_lengths else 0,
            'cluster_distribution': dict(cluster_counts),
            'extraction_time': self.extraction_time,
            'clustering_time': self.clustering_time,
            'sequence_time': self.sequence_time,
            'total_time': self.extraction_time + self.clustering_time + self.sequence_time,
            'memory_usage_mb': self._estimate_memory_usage() / (1024 * 1024)
        }
        
        if self.transition_matrix is not None:
            stats['transition_matrix_shape'] = self.transition_matrix.shape
            stats['transition_matrix_sparsity'] = np.mean(self.transition_matrix == 0)
        
        return stats
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes"""
        memory = 0
        
        # Patterns
        for pattern in self.patterns:
            memory += pattern.coefficients.nbytes
            memory += 100  # Overhead for object and metadata
        
        # Pattern vocabulary
        for coeffs in self.pattern_vocabulary.values():
            memory += coeffs.nbytes
        
        # Transition matrix
        if self.transition_matrix is not None:
            memory += self.transition_matrix.nbytes
        
        # Sequences
        memory += len(self.pattern_sequences) * 200  # Rough estimate
        
        return memory
    
    def extract_patterns(self, 
                        data: np.ndarray,
                        min_pattern_length: Optional[int] = None,
                        timestamps: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Extract patterns from time series data (simplified interface)
        
        Args:
            data: Time series data
            min_pattern_length: Minimum pattern length (uses default if None)
            timestamps: Optional timestamps
            
        Returns:
            List of pattern dictionaries with start, end, and data
        """
        if min_pattern_length:
            self.min_pattern_length = min_pattern_length
            
        # Extract wavelet patterns
        wavelet_patterns = self.extract_wavelet_patterns(data, timestamps)
        
        # Cluster patterns
        if wavelet_patterns:
            self.cluster_patterns(wavelet_patterns)
        
        # Convert to simple format for compatibility
        patterns = []
        for pattern in wavelet_patterns:
            if pattern.cluster_id is not None and pattern.cluster_id != -1:
                patterns.append({
                    'start': pattern.metadata['start_idx'],
                    'end': pattern.metadata['end_idx'],
                    'cluster_id': pattern.cluster_id,
                    'scale': pattern.scale,
                    'energy': pattern.metadata.get('energy', 0),
                    'pattern_id': pattern.pattern_id
                })
        
        return patterns
    
    def save_analyzer(self, filepath: str) -> None:
        """
        Save analyzer state to file
        
        Args:
            filepath: Path to save file
        """
        state = {
            'patterns': self.patterns,
            'pattern_vocabulary': self.pattern_vocabulary,
            'pattern_sequences': self.pattern_sequences,
            'transition_matrix': self.transition_matrix,
            'cluster_centers': self.cluster_centers,
            'cluster_to_idx': getattr(self, 'cluster_to_idx', None),
            'idx_to_cluster': getattr(self, 'idx_to_cluster', None),
            'scaler_params': {
                'mean': self.scaler.mean_ if hasattr(self.scaler, 'mean_') else None,
                'scale': self.scaler.scale_ if hasattr(self.scaler, 'scale_') else None
            },
            'pca_components': self.pca.components_ if self.pca and hasattr(self.pca, 'components_') else None,
            'config': {
                'wavelet': self.wavelet,
                'scales': self.scales,
                'n_clusters': self.n_clusters,
                'clustering_method': self.clustering_method,
                'min_pattern_length': self.min_pattern_length,
                'max_pattern_length': self.max_pattern_length,
                'overlap_ratio': self.overlap_ratio,
                'pca_components': self.pca_components,
                'random_state': self.random_state
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_analyzer(self, filepath: str) -> None:
        """
        Load analyzer state from file
        
        Args:
            filepath: Path to saved file
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Restore state
        self.patterns = state['patterns']
        self.pattern_vocabulary = state['pattern_vocabulary']
        self.pattern_sequences = state['pattern_sequences']
        self.transition_matrix = state['transition_matrix']
        self.cluster_centers = state['cluster_centers']
        
        if state.get('cluster_to_idx'):
            self.cluster_to_idx = state['cluster_to_idx']
            self.idx_to_cluster = state['idx_to_cluster']
        
        # Restore scaler
        if state['scaler_params']['mean'] is not None:
            self.scaler.mean_ = state['scaler_params']['mean']
            self.scaler.scale_ = state['scaler_params']['scale']
            self.scaler.n_features_in_ = len(state['scaler_params']['mean'])
        
        # Restore PCA
        if state['pca_components'] is not None and self.pca:
            self.pca.components_ = state['pca_components']
            self.pca.n_components_ = len(state['pca_components'])
        
        # Restore config
        config = state['config']
        self.wavelet = config['wavelet']
        self.scales = config['scales']
        self.n_clusters = config['n_clusters']
        self.clustering_method = config['clustering_method']
        self.min_pattern_length = config['min_pattern_length']
        self.max_pattern_length = config['max_pattern_length']
        self.overlap_ratio = config['overlap_ratio']
        self.pca_components = config['pca_components']
        self.random_state = config['random_state']


def create_analyzer_pipeline(data: np.ndarray,
                           timestamps: Optional[np.ndarray] = None,
                           config: Optional[Dict[str, Any]] = None) -> Tuple[WaveletSequenceAnalyzer, Dict[str, Any]]:
    """
    Create and run complete analyzer pipeline
    
    Args:
        data: Time series data
        timestamps: Optional timestamps
        config: Optional configuration dictionary
        
    Returns:
        Tuple of (analyzer, results)
    """
    # Default configuration
    default_config = {
        'wavelet': 'morl',
        'scales': np.arange(1, 33),
        'n_clusters': 10,
        'clustering_method': 'kmeans',
        'min_pattern_length': 5,
        'max_pattern_length': 50,
        'overlap_ratio': 0.5,
        'pca_components': 10,
        'random_state': 42,
        'min_sequence_length': 2,
        'max_gap': 5
    }
    
    if config:
        default_config.update(config)
    
    # Create analyzer
    analyzer = WaveletSequenceAnalyzer(
        wavelet=default_config['wavelet'],
        scales=default_config['scales'],
        n_clusters=default_config['n_clusters'],
        clustering_method=default_config['clustering_method'],
        min_pattern_length=default_config['min_pattern_length'],
        max_pattern_length=default_config['max_pattern_length'],
        overlap_ratio=default_config['overlap_ratio'],
        pca_components=default_config['pca_components'],
        random_state=default_config['random_state']
    )
    
    # Run pipeline
    patterns = analyzer.extract_wavelet_patterns(data, timestamps)
    cluster_mapping = analyzer.cluster_patterns(patterns)
    sequences = analyzer.identify_sequences(
        min_sequence_length=default_config['min_sequence_length'],
        max_gap=default_config['max_gap']
    )
    transition_matrix = analyzer.calculate_transition_matrix()
    
    # Get statistics
    stats = analyzer.get_pattern_statistics()
    
    results = {
        'patterns': patterns,
        'cluster_mapping': cluster_mapping,
        'sequences': sequences,
        'transition_matrix': transition_matrix,
        'statistics': stats
    }
    
    return analyzer, results
