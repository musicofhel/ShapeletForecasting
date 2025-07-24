"""
Motif Discovery Module

Implements motif discovery algorithms to find recurring patterns in financial time series.
Uses STUMPY for efficient matrix profile computation.
"""

import numpy as np
import pandas as pd
import stumpy
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)


class MotifDiscovery:
    """
    Discovers recurring patterns (motifs) in financial time series data.
    """
    
    def __init__(self, window_size: int = 50, min_distance: int = 25):
        """
        Initialize the MotifDiscovery.
        
        Args:
            window_size: Size of the sliding window for pattern matching
            min_distance: Minimum distance between motif occurrences
        """
        self.window_size = window_size
        self.min_distance = min_distance
        self.matrix_profile = None
        self.matrix_profile_indices = None
        
        logger.info(f"Initialized MotifDiscovery with window_size={window_size}, min_distance={min_distance}")
    
    def compute_matrix_profile(self, data: Union[pd.Series, np.ndarray],
                             normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the matrix profile for the time series.
        
        Args:
            data: Time series data
            normalize: Whether to normalize the data
            
        Returns:
            Tuple of (matrix_profile, matrix_profile_indices)
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
            scaler = StandardScaler()
            values = scaler.fit_transform(values.reshape(-1, 1)).flatten()
        
        # Compute matrix profile
        logger.debug(f"Computing matrix profile for {len(values)} data points with window size {self.window_size}")
        mp = stumpy.stump(values, m=self.window_size)
        
        # Extract matrix profile and indices
        self.matrix_profile = mp[:, 0]
        self.matrix_profile_indices = mp[:, 1].astype(int)
        
        return self.matrix_profile, self.matrix_profile_indices
    
    def find_motifs(self, data: Union[pd.Series, np.ndarray],
                   top_k: int = 10,
                   normalize: bool = True) -> List[Dict]:
        """
        Find top-k motifs in the time series.
        
        Args:
            data: Time series data
            top_k: Number of top motifs to find
            normalize: Whether to normalize the data
            
        Returns:
            List of motif dictionaries with metadata
        """
        # Compute matrix profile if not already done
        if self.matrix_profile is None:
            self.compute_matrix_profile(data, normalize)
        
        # Convert data to numpy array for consistency
        if isinstance(data, pd.Series):
            values = data.values
        else:
            values = np.asarray(data)
        
        # Find motifs using STUMPY
        motif_indices = stumpy.motifs(
            values,
            self.matrix_profile,
            min_neighbors=1,
            max_distance=2 * np.std(self.matrix_profile),
            cutoff=top_k,
            max_matches=10,
            max_motifs=top_k
        )
        
        # Process motifs
        motifs = []
        for i, (motif_idx, neighbors) in enumerate(motif_indices):
            if len(neighbors) > 0:
                # Get the motif pattern
                motif_pattern = values[motif_idx:motif_idx + self.window_size]
                
                # Calculate motif statistics
                distances = []
                occurrences = [motif_idx] + list(neighbors)
                
                for neighbor_idx in neighbors:
                    neighbor_pattern = values[neighbor_idx:neighbor_idx + self.window_size]
                    dist = euclidean(motif_pattern, neighbor_pattern)
                    distances.append(dist)
                
                motifs.append({
                    'id': i,
                    'primary_index': motif_idx,
                    'pattern': motif_pattern,
                    'occurrences': occurrences,
                    'num_occurrences': len(occurrences),
                    'mean_distance': np.mean(distances) if distances else 0,
                    'std_distance': np.std(distances) if distances else 0,
                    'window_size': self.window_size
                })
        
        # Sort by number of occurrences
        motifs.sort(key=lambda x: x['num_occurrences'], reverse=True)
        
        return motifs[:top_k]
    
    def find_discords(self, data: Union[pd.Series, np.ndarray],
                     top_k: int = 10,
                     normalize: bool = True) -> List[Dict]:
        """
        Find top-k discords (anomalous patterns) in the time series.
        
        Args:
            data: Time series data
            top_k: Number of top discords to find
            normalize: Whether to normalize the data
            
        Returns:
            List of discord dictionaries with metadata
        """
        # Compute matrix profile if not already done
        if self.matrix_profile is None:
            self.compute_matrix_profile(data, normalize)
        
        # Convert data to numpy array for consistency
        if isinstance(data, pd.Series):
            values = data.values
        else:
            values = np.asarray(data)
        
        # Find indices with highest matrix profile values (most anomalous)
        discord_indices = np.argsort(self.matrix_profile)[-top_k:][::-1]
        
        discords = []
        for i, idx in enumerate(discord_indices):
            # Get the discord pattern
            discord_pattern = values[idx:idx + self.window_size]
            
            # Get nearest neighbor
            nn_idx = self.matrix_profile_indices[idx]
            nn_pattern = values[nn_idx:nn_idx + self.window_size]
            nn_distance = self.matrix_profile[idx]
            
            discords.append({
                'id': i,
                'index': idx,
                'pattern': discord_pattern,
                'nearest_neighbor_index': nn_idx,
                'nearest_neighbor_pattern': nn_pattern,
                'distance': nn_distance,
                'anomaly_score': nn_distance / np.mean(self.matrix_profile),
                'window_size': self.window_size
            })
        
        return discords
    
    def find_chains(self, data: Union[pd.Series, np.ndarray],
                   normalize: bool = True) -> List[Dict]:
        """
        Find chains (evolving patterns) in the time series.
        
        Args:
            data: Time series data
            normalize: Whether to normalize the data
            
        Returns:
            List of chain dictionaries with metadata
        """
        # Compute matrix profile if not already done
        if self.matrix_profile is None:
            self.compute_matrix_profile(data, normalize)
        
        # Convert data to numpy array for consistency
        if isinstance(data, pd.Series):
            values = data.values
        else:
            values = np.asarray(data)
        
        # Find all chains
        all_chain_set, unanchored_chain = stumpy.allc(self.matrix_profile_indices)
        
        chains = []
        for i, chain in enumerate(all_chain_set):
            if len(chain) > 1:
                # Get patterns for each element in the chain
                patterns = []
                for idx in chain:
                    pattern = values[idx:idx + self.window_size]
                    patterns.append(pattern)
                
                # Calculate chain statistics
                transitions = []
                for j in range(len(patterns) - 1):
                    dist = euclidean(patterns[j], patterns[j + 1])
                    transitions.append(dist)
                
                chains.append({
                    'id': i,
                    'indices': list(chain),
                    'length': len(chain),
                    'patterns': patterns,
                    'mean_transition': np.mean(transitions) if transitions else 0,
                    'std_transition': np.std(transitions) if transitions else 0,
                    'total_evolution': euclidean(patterns[0], patterns[-1]) if len(patterns) > 1 else 0,
                    'window_size': self.window_size
                })
        
        # Add unanchored chain if it exists
        if len(unanchored_chain) > 1:
            patterns = []
            for idx in unanchored_chain:
                pattern = values[idx:idx + self.window_size]
                patterns.append(pattern)
            
            transitions = []
            for j in range(len(patterns) - 1):
                dist = euclidean(patterns[j], patterns[j + 1])
                transitions.append(dist)
            
            chains.append({
                'id': len(chains),
                'indices': list(unanchored_chain),
                'length': len(unanchored_chain),
                'patterns': patterns,
                'mean_transition': np.mean(transitions) if transitions else 0,
                'std_transition': np.std(transitions) if transitions else 0,
                'total_evolution': euclidean(patterns[0], patterns[-1]) if len(patterns) > 1 else 0,
                'window_size': self.window_size,
                'unanchored': True
            })
        
        # Sort by chain length
        chains.sort(key=lambda x: x['length'], reverse=True)
        
        return chains
    
    def find_semantic_motifs(self, data: Union[pd.Series, np.ndarray],
                           labels: Optional[np.ndarray] = None,
                           top_k: int = 10) -> List[Dict]:
        """
        Find semantic motifs (patterns associated with specific events/labels).
        
        Args:
            data: Time series data
            labels: Event labels (same length as data)
            top_k: Number of top motifs to find
            
        Returns:
            List of semantic motif dictionaries
        """
        # Convert data to numpy array for consistency
        if isinstance(data, pd.Series):
            values = data.values
        else:
            values = np.asarray(data)
        
        # If no labels provided, use change detection
        if labels is None:
            labels = self._detect_regime_changes(values)
        
        # Find motifs for each unique label
        semantic_motifs = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            # Get indices where this label occurs
            label_indices = np.where(labels == label)[0]
            
            # Skip if too few occurrences
            if len(label_indices) < 2:
                continue
            
            # Extract subsequences for this label
            subsequences = []
            valid_indices = []
            
            for idx in label_indices:
                if idx + self.window_size <= len(values):
                    subseq = values[idx:idx + self.window_size]
                    subsequences.append(subseq)
                    valid_indices.append(idx)
            
            if len(subsequences) < 2:
                continue
            
            # Find the most representative pattern (medoid)
            subsequences_array = np.array(subsequences)
            distances = np.zeros((len(subsequences), len(subsequences)))
            
            for i in range(len(subsequences)):
                for j in range(i + 1, len(subsequences)):
                    dist = euclidean(subsequences[i], subsequences[j])
                    distances[i, j] = dist
                    distances[j, i] = dist
            
            # Find medoid (pattern with minimum sum of distances to others)
            sum_distances = np.sum(distances, axis=1)
            medoid_idx = np.argmin(sum_distances)
            
            semantic_motifs.append({
                'label': label,
                'representative_pattern': subsequences[medoid_idx],
                'representative_index': valid_indices[medoid_idx],
                'occurrences': valid_indices,
                'num_occurrences': len(valid_indices),
                'mean_intra_distance': np.mean(distances[medoid_idx, :]),
                'std_intra_distance': np.std(distances[medoid_idx, :]),
                'patterns': subsequences,
                'window_size': self.window_size
            })
        
        # Sort by number of occurrences
        semantic_motifs.sort(key=lambda x: x['num_occurrences'], reverse=True)
        
        return semantic_motifs[:top_k]
    
    def _detect_regime_changes(self, data: np.ndarray,
                             n_regimes: int = 5) -> np.ndarray:
        """
        Detect regime changes in the time series using change point detection.
        
        Args:
            data: Time series data
            n_regimes: Number of regimes to detect
            
        Returns:
            Array of regime labels
        """
        # Use FLUSS (Fast Low-cost Unipotent Semantic Segmentation)
        # to detect regime changes
        mp = stumpy.stump(data, m=self.window_size)
        
        # Get arc curve
        arc_curve = stumpy.fluss(mp[:, 1], L=self.window_size // 2, n_regimes=n_regimes)
        
        # Find regime change points
        change_points = arc_curve[1]
        
        # Create labels
        labels = np.zeros(len(data), dtype=int)
        current_label = 0
        
        change_points = np.sort(change_points)
        change_points = np.append(change_points, len(data))
        
        start = 0
        for cp in change_points:
            labels[start:cp] = current_label
            current_label += 1
            start = cp
        
        return labels
    
    def multidimensional_motifs(self, data: pd.DataFrame,
                               top_k: int = 10) -> List[Dict]:
        """
        Find motifs in multidimensional time series.
        
        Args:
            data: DataFrame with multiple time series columns
            top_k: Number of top motifs to find
            
        Returns:
            List of multidimensional motif dictionaries
        """
        # Convert to numpy array
        values = data.values
        
        # Compute multidimensional matrix profile
        mp, indices = stumpy.mstump(values, m=self.window_size)
        
        # Find multidimensional motifs
        motif_indices = stumpy.mmotifs(
            values,
            mp,
            indices,
            min_neighbors=1,
            max_distance=2 * np.std(mp),
            cutoff=top_k
        )
        
        motifs = []
        for i, (motif_idx, dimensions, neighbors) in enumerate(motif_indices):
            if len(neighbors) > 0:
                # Get the motif pattern
                motif_pattern = values[motif_idx:motif_idx + self.window_size, dimensions]
                
                # Calculate statistics
                occurrences = [motif_idx] + list(neighbors)
                
                motifs.append({
                    'id': i,
                    'primary_index': motif_idx,
                    'dimensions': dimensions,
                    'pattern': motif_pattern,
                    'occurrences': occurrences,
                    'num_occurrences': len(occurrences),
                    'window_size': self.window_size
                })
        
        return motifs[:top_k]
    
    def get_motif_features(self, motifs: List[Dict]) -> pd.DataFrame:
        """
        Extract features from discovered motifs.
        
        Args:
            motifs: List of motif dictionaries
            
        Returns:
            DataFrame with motif features
        """
        features = []
        
        for motif in motifs:
            pattern = motif['pattern']
            
            # Statistical features
            feat = {
                'motif_id': motif['id'],
                'num_occurrences': motif['num_occurrences'],
                'mean': np.mean(pattern),
                'std': np.std(pattern),
                'min': np.min(pattern),
                'max': np.max(pattern),
                'range': np.max(pattern) - np.min(pattern),
                'skewness': pd.Series(pattern).skew(),
                'kurtosis': pd.Series(pattern).kurtosis(),
                'trend': np.polyfit(range(len(pattern)), pattern, 1)[0],
                'energy': np.sum(pattern**2),
                'entropy': -np.sum(pattern * np.log(np.abs(pattern) + 1e-10))
            }
            
            # Add mean distance if available
            if 'mean_distance' in motif:
                feat['mean_distance'] = motif['mean_distance']
                feat['std_distance'] = motif['std_distance']
            
            features.append(feat)
        
        return pd.DataFrame(features)


def main():
    """Demonstration of MotifDiscovery functionality."""
    # Create sample data with repeating patterns
    np.random.seed(42)
    t = np.linspace(0, 100, 1000)
    
    # Base pattern
    pattern = np.sin(2 * np.pi * t[:50] / 10) + 0.5 * np.sin(2 * np.pi * t[:50] / 5)
    
    # Create time series with repeated patterns
    data = np.random.randn(1000) * 0.1
    
    # Insert pattern at multiple locations
    for i in [100, 300, 500, 700]:
        data[i:i+50] += pattern
    
    # Add an anomaly
    data[600:650] += np.random.randn(50) * 2
    
    # Initialize motif discovery
    md = MotifDiscovery(window_size=50)
    
    # Find motifs
    motifs = md.find_motifs(data, top_k=5)
    print(f"Found {len(motifs)} motifs")
    for motif in motifs:
        print(f"  Motif {motif['id']}: {motif['num_occurrences']} occurrences")
    
    # Find discords
    discords = md.find_discords(data, top_k=3)
    print(f"\nFound {len(discords)} discords")
    for discord in discords:
        print(f"  Discord at index {discord['index']}: anomaly score = {discord['anomaly_score']:.2f}")
    
    # Extract features
    features = md.get_motif_features(motifs)
    print(f"\nMotif features shape: {features.shape}")


if __name__ == "__main__":
    main()
