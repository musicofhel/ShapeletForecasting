"""
Shapelet Extractor Module

Implements shapelet extraction algorithms to find discriminative subsequences
in financial time series that can distinguish between different market conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from scipy.spatial.distance import euclidean
from joblib import Parallel, delayed
import warnings

logger = logging.getLogger(__name__)


class ShapeletExtractor:
    """
    Extracts shapelets (discriminative subsequences) from financial time series.
    """
    
    def __init__(self, min_length: int = 10, max_length: int = 100,
                 num_shapelets: int = 50, quality_threshold: float = 0.7):
        """
        Initialize the ShapeletExtractor.
        
        Args:
            min_length: Minimum shapelet length
            max_length: Maximum shapelet length
            num_shapelets: Number of shapelets to extract
            quality_threshold: Minimum quality score for shapelet selection
        """
        self.min_length = min_length
        self.max_length = max_length
        self.num_shapelets = num_shapelets
        self.quality_threshold = quality_threshold
        self.shapelets = []
        
        logger.info(f"Initialized ShapeletExtractor with lengths [{min_length}, {max_length}]")
    
    def extract_shapelets(self, data: Union[pd.Series, np.ndarray],
                         labels: np.ndarray,
                         normalize: bool = True,
                         n_jobs: int = -1) -> List[Dict]:
        """
        Extract shapelets from labeled time series data.
        
        Args:
            data: Time series data
            labels: Class labels for the time series
            normalize: Whether to normalize the data
            n_jobs: Number of parallel jobs (-1 for all cores)
            
        Returns:
            List of shapelet dictionaries with metadata
        """
        # Convert to numpy array if needed
        if isinstance(data, pd.Series):
            values = data.values
        else:
            values = np.asarray(data)
        
        # Normalize if requested
        if normalize:
            scaler = StandardScaler()
            values = scaler.fit_transform(values.reshape(-1, 1)).flatten()
        
        # Generate candidate shapelets
        candidates = self._generate_candidates(values, labels)
        
        # Evaluate candidates in parallel
        logger.info(f"Evaluating {len(candidates)} shapelet candidates...")
        
        if n_jobs == 1:
            # Sequential processing
            evaluated = []
            for candidate in candidates:
                result = self._evaluate_shapelet(candidate, values, labels)
                evaluated.append(result)
        else:
            # Parallel processing
            evaluated = Parallel(n_jobs=n_jobs)(
                delayed(self._evaluate_shapelet)(candidate, values, labels)
                for candidate in candidates
            )
        
        # Filter by quality threshold
        quality_shapelets = [s for s in evaluated if s['quality'] >= self.quality_threshold]
        
        # Sort by quality
        quality_shapelets.sort(key=lambda x: x['quality'], reverse=True)
        
        # Select top shapelets with diversity
        self.shapelets = self._select_diverse_shapelets(quality_shapelets, self.num_shapelets)
        
        logger.info(f"Extracted {len(self.shapelets)} shapelets")
        
        return self.shapelets
    
    def _generate_candidates(self, data: np.ndarray, labels: np.ndarray) -> List[Dict]:
        """
        Generate candidate shapelets from the time series.
        
        Args:
            data: Time series data
            labels: Class labels
            
        Returns:
            List of candidate shapelet dictionaries
        """
        candidates = []
        unique_labels = np.unique(labels)
        
        # Sample subsequences from different regions
        for length in range(self.min_length, min(self.max_length + 1, len(data))):
            # Adaptive step size based on length
            step = max(1, length // 10)
            
            for start in range(0, len(data) - length + 1, step):
                subsequence = data[start:start + length]
                
                # Determine the label for this subsequence
                # (using majority vote if labels are per-point)
                if len(labels) == len(data):
                    label_region = labels[start:start + length]
                    label = np.bincount(label_region).argmax()
                else:
                    # Assume single label for entire series
                    label = labels[0] if len(labels) > 0 else 0
                
                candidates.append({
                    'pattern': subsequence,
                    'start': start,
                    'length': length,
                    'label': label
                })
        
        # Limit candidates if too many
        if len(candidates) > 10000:
            # Random sampling
            indices = np.random.choice(len(candidates), 10000, replace=False)
            candidates = [candidates[i] for i in indices]
        
        return candidates
    
    def _evaluate_shapelet(self, candidate: Dict, data: np.ndarray,
                          labels: np.ndarray) -> Dict:
        """
        Evaluate the quality of a shapelet candidate.
        
        Args:
            candidate: Shapelet candidate dictionary
            data: Full time series data
            labels: Class labels
            
        Returns:
            Evaluated shapelet dictionary with quality metrics
        """
        pattern = candidate['pattern']
        length = len(pattern)
        
        # Calculate distances to all possible positions
        distances = []
        positions = []
        
        for i in range(len(data) - length + 1):
            subsequence = data[i:i + length]
            dist = self._shapelet_distance(pattern, subsequence)
            distances.append(dist)
            positions.append(i)
        
        distances = np.array(distances)
        
        # Calculate information gain or other quality measure
        quality = self._calculate_quality(distances, labels, positions)
        
        # Find best split threshold
        threshold = self._find_best_threshold(distances, labels, positions)
        
        # Calculate additional statistics
        result = candidate.copy()
        result.update({
            'quality': quality,
            'threshold': threshold,
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'coverage': len(distances) / len(data)
        })
        
        return result
    
    def _shapelet_distance(self, shapelet: np.ndarray, subsequence: np.ndarray) -> float:
        """
        Calculate distance between shapelet and subsequence.
        
        Args:
            shapelet: Shapelet pattern
            subsequence: Time series subsequence
            
        Returns:
            Distance value
        """
        # Normalize both sequences
        shapelet_norm = (shapelet - np.mean(shapelet)) / (np.std(shapelet) + 1e-8)
        subseq_norm = (subsequence - np.mean(subsequence)) / (np.std(subsequence) + 1e-8)
        
        # Euclidean distance
        return euclidean(shapelet_norm, subseq_norm)
    
    def _calculate_quality(self, distances: np.ndarray, labels: np.ndarray,
                          positions: List[int]) -> float:
        """
        Calculate quality score for a shapelet using information gain.
        
        Args:
            distances: Distance array
            labels: Class labels
            positions: Positions corresponding to distances
            
        Returns:
            Quality score
        """
        # Map positions to labels
        if len(labels) == len(distances):
            position_labels = labels
        else:
            # Handle case where we have windowed data
            position_labels = []
            for pos in positions:
                # Use label at the start position
                if pos < len(labels):
                    position_labels.append(labels[pos])
                else:
                    position_labels.append(labels[-1])
            position_labels = np.array(position_labels)
        
        # Calculate entropy before split
        unique_labels, counts = np.unique(position_labels, return_counts=True)
        probs = counts / len(position_labels)
        entropy_before = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Find best split point
        best_gain = 0
        sorted_indices = np.argsort(distances)
        
        for i in range(1, len(distances)):
            # Split at this point
            left_labels = position_labels[sorted_indices[:i]]
            right_labels = position_labels[sorted_indices[i:]]
            
            # Calculate entropy after split
            left_entropy = self._calculate_entropy(left_labels)
            right_entropy = self._calculate_entropy(right_labels)
            
            # Weighted entropy
            left_weight = len(left_labels) / len(position_labels)
            right_weight = len(right_labels) / len(position_labels)
            entropy_after = left_weight * left_entropy + right_weight * right_entropy
            
            # Information gain
            gain = entropy_before - entropy_after
            
            if gain > best_gain:
                best_gain = gain
        
        return best_gain
    
    def _calculate_entropy(self, labels: np.ndarray) -> float:
        """Calculate entropy of a label array."""
        if len(labels) == 0:
            return 0
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    def _find_best_threshold(self, distances: np.ndarray, labels: np.ndarray,
                           positions: List[int]) -> float:
        """
        Find the best distance threshold for classification.
        
        Args:
            distances: Distance array
            labels: Class labels
            positions: Positions corresponding to distances
            
        Returns:
            Best threshold value
        """
        # Map positions to labels
        if len(labels) == len(distances):
            position_labels = labels
        else:
            position_labels = []
            for pos in positions:
                if pos < len(labels):
                    position_labels.append(labels[pos])
                else:
                    position_labels.append(labels[-1])
            position_labels = np.array(position_labels)
        
        # Try different thresholds
        sorted_distances = np.sort(distances)
        best_threshold = sorted_distances[len(sorted_distances) // 2]
        best_accuracy = 0
        
        for i in range(1, len(sorted_distances)):
            threshold = (sorted_distances[i-1] + sorted_distances[i]) / 2
            
            # Predict labels based on threshold
            predictions = (distances <= threshold).astype(int)
            
            # Calculate accuracy for binary classification
            # Assuming shapelet is indicative of one class
            accuracy = np.mean(predictions == position_labels)
            accuracy = max(accuracy, 1 - accuracy)  # Handle inverted case
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        return best_threshold
    
    def _select_diverse_shapelets(self, shapelets: List[Dict],
                                 num_select: int) -> List[Dict]:
        """
        Select diverse shapelets to avoid redundancy.
        
        Args:
            shapelets: List of shapelet candidates
            num_select: Number of shapelets to select
            
        Returns:
            Selected diverse shapelets
        """
        if len(shapelets) <= num_select:
            return shapelets
        
        selected = []
        remaining = shapelets.copy()
        
        # Select first shapelet (highest quality)
        selected.append(remaining.pop(0))
        
        # Iteratively select diverse shapelets
        while len(selected) < num_select and remaining:
            max_min_dist = -1
            best_idx = -1
            
            # Find shapelet with maximum minimum distance to selected
            for i, candidate in enumerate(remaining):
                min_dist = float('inf')
                
                for selected_shapelet in selected:
                    # Compare patterns
                    if len(candidate['pattern']) == len(selected_shapelet['pattern']):
                        dist = self._shapelet_distance(
                            candidate['pattern'],
                            selected_shapelet['pattern']
                        )
                        min_dist = min(min_dist, dist)
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i
            
            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
            else:
                # If no valid candidate found, break
                break
        
        return selected
    
    def transform(self, data: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Transform time series data using extracted shapelets.
        
        Args:
            data: Time series data
            
        Returns:
            Feature matrix where each column is the distance to a shapelet
        """
        # Convert to numpy array if needed
        if isinstance(data, pd.Series):
            values = data.values
        else:
            values = np.asarray(data)
        
        if not self.shapelets:
            raise ValueError("No shapelets extracted. Run extract_shapelets first.")
        
        # Calculate distances to all shapelets
        features = []
        
        for shapelet in self.shapelets:
            pattern = shapelet['pattern']
            length = len(pattern)
            
            # Find minimum distance to this shapelet
            min_dist = float('inf')
            
            for i in range(len(values) - length + 1):
                subsequence = values[i:i + length]
                dist = self._shapelet_distance(pattern, subsequence)
                min_dist = min(min_dist, dist)
            
            features.append(min_dist)
        
        return np.array(features)
    
    def extract_multivariate_shapelets(self, data: pd.DataFrame,
                                     labels: np.ndarray,
                                     n_jobs: int = -1) -> List[Dict]:
        """
        Extract shapelets from multivariate time series.
        
        Args:
            data: DataFrame with multiple time series columns
            labels: Class labels
            n_jobs: Number of parallel jobs
            
        Returns:
            List of multivariate shapelet dictionaries
        """
        multivariate_shapelets = []
        
        # Extract shapelets for each dimension
        for column in data.columns:
            logger.info(f"Extracting shapelets for {column}")
            
            column_shapelets = self.extract_shapelets(
                data[column].values,
                labels,
                n_jobs=n_jobs
            )
            
            # Add dimension information
            for shapelet in column_shapelets:
                shapelet['dimension'] = column
            
            multivariate_shapelets.extend(column_shapelets)
        
        # Sort by quality across all dimensions
        multivariate_shapelets.sort(key=lambda x: x['quality'], reverse=True)
        
        # Select top shapelets maintaining diversity
        self.shapelets = self._select_diverse_shapelets(
            multivariate_shapelets,
            self.num_shapelets
        )
        
        return self.shapelets
    
    def save_shapelets(self, filepath: str):
        """
        Save extracted shapelets to file.
        
        Args:
            filepath: Path to save shapelets
        """
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.shapelets, f)
        
        logger.info(f"Saved {len(self.shapelets)} shapelets to {filepath}")
    
    def load_shapelets(self, filepath: str):
        """
        Load shapelets from file.
        
        Args:
            filepath: Path to load shapelets from
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            self.shapelets = pickle.load(f)
        
        logger.info(f"Loaded {len(self.shapelets)} shapelets from {filepath}")
    
    def get_shapelet_features(self) -> pd.DataFrame:
        """
        Get features of extracted shapelets.
        
        Returns:
            DataFrame with shapelet features
        """
        if not self.shapelets:
            return pd.DataFrame()
        
        features = []
        for i, shapelet in enumerate(self.shapelets):
            pattern = shapelet['pattern']
            
            feat = {
                'shapelet_id': i,
                'length': len(pattern),
                'quality': shapelet['quality'],
                'threshold': shapelet['threshold'],
                'mean': np.mean(pattern),
                'std': np.std(pattern),
                'min': np.min(pattern),
                'max': np.max(pattern),
                'range': np.max(pattern) - np.min(pattern),
                'trend': np.polyfit(range(len(pattern)), pattern, 1)[0],
                'start_value': pattern[0],
                'end_value': pattern[-1],
                'coverage': shapelet.get('coverage', 0)
            }
            
            if 'dimension' in shapelet:
                feat['dimension'] = shapelet['dimension']
            
            features.append(feat)
        
        return pd.DataFrame(features)


def main():
    """Demonstration of ShapeletExtractor functionality."""
    # Create sample data with two classes
    np.random.seed(42)
    n_samples = 1000
    
    # Class 0: Upward trend pattern
    class0_pattern = np.linspace(0, 1, 30)
    
    # Class 1: Downward trend pattern
    class1_pattern = np.linspace(1, 0, 30)
    
    # Generate time series
    data = np.random.randn(n_samples) * 0.1
    labels = np.zeros(n_samples, dtype=int)
    
    # Insert patterns
    for i in range(0, n_samples - 30, 100):
        if np.random.rand() > 0.5:
            data[i:i+30] += class0_pattern + np.random.randn(30) * 0.05
            labels[i:i+30] = 0
        else:
            data[i:i+30] += class1_pattern + np.random.randn(30) * 0.05
            labels[i:i+30] = 1
    
    # Initialize extractor
    extractor = ShapeletExtractor(
        min_length=20,
        max_length=40,
        num_shapelets=10,
        quality_threshold=0.5
    )
    
    # Extract shapelets
    shapelets = extractor.extract_shapelets(data, labels, n_jobs=1)
    print(f"Extracted {len(shapelets)} shapelets")
    
    # Get shapelet features
    features = extractor.get_shapelet_features()
    print("\nShapelet features:")
    print(features[['shapelet_id', 'length', 'quality', 'trend']].head())
    
    # Transform data
    transformed = extractor.transform(data)
    print(f"\nTransformed data shape: {transformed.shape}")


if __name__ == "__main__":
    main()
