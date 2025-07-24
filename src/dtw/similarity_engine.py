"""
Similarity Engine for Pattern Matching

Computes similarity matrices between patterns using various DTW algorithms
and provides efficient batch processing capabilities.
"""

import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import pickle
import h5py
from pathlib import Path
import logging

from .dtw_calculator import DTWCalculator, FastDTW, ConstrainedDTW, DTWResult


class SimilarityEngine:
    """
    Engine for computing pairwise similarities between time series patterns
    using DTW algorithms with parallel processing support.
    """
    
    def __init__(self,
                 dtw_type: str = 'standard',
                 distance_metric: str = 'euclidean',
                 normalize: bool = True,
                 n_jobs: int = -1,
                 verbose: bool = True,
                 **dtw_params):
        """
        Initialize Similarity Engine
        
        Parameters:
        -----------
        dtw_type : str
            Type of DTW algorithm ('standard', 'fast', 'constrained')
        distance_metric : str
            Distance metric for DTW computation
        normalize : bool
            Whether to normalize DTW distances
        n_jobs : int
            Number of parallel jobs (-1 for all CPUs)
        verbose : bool
            Whether to show progress bars
        **dtw_params : dict
            Additional parameters for DTW algorithms
        """
        self.dtw_type = dtw_type
        self.distance_metric = distance_metric
        self.normalize = normalize
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        self.verbose = verbose
        self.dtw_params = dtw_params
        
        # Initialize DTW calculator
        self._init_dtw_calculator()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _init_dtw_calculator(self):
        """Initialize the appropriate DTW calculator"""
        if self.dtw_type == 'standard':
            self.dtw_calculator = DTWCalculator(
                distance_metric=self.distance_metric,
                normalize=self.normalize,
                return_path=False,
                return_cost_matrix=False
            )
        elif self.dtw_type == 'fast':
            radius = self.dtw_params.get('radius', 1)
            self.dtw_calculator = FastDTW(
                radius=radius,
                distance_metric=self.distance_metric,
                normalize=self.normalize
            )
        elif self.dtw_type == 'constrained':
            constraint_type = self.dtw_params.get('constraint_type', 'sakoe_chiba')
            constraint_param = self.dtw_params.get('constraint_param', 10)
            self.dtw_calculator = ConstrainedDTW(
                constraint_type=constraint_type,
                constraint_param=constraint_param,
                distance_metric=self.distance_metric,
                normalize=self.normalize
            )
        else:
            raise ValueError(f"Unknown DTW type: {self.dtw_type}")
            
    def compute_similarity_matrix(self,
                                patterns: Union[List[np.ndarray], np.ndarray],
                                labels: Optional[List[str]] = None,
                                save_path: Optional[str] = None) -> Dict:
        """
        Compute pairwise similarity matrix between patterns
        
        Parameters:
        -----------
        patterns : List[np.ndarray] or np.ndarray
            List of time series patterns or 3D array (n_patterns, n_timesteps, n_features)
        labels : List[str], optional
            Labels for each pattern
        save_path : str, optional
            Path to save the similarity matrix
            
        Returns:
        --------
        dict containing:
            - 'distance_matrix': Pairwise DTW distance matrix
            - 'similarity_matrix': Similarity matrix (1 - normalized_distance)
            - 'labels': Pattern labels
            - 'metadata': Additional information
        """
        # Convert to list if needed
        if isinstance(patterns, np.ndarray):
            if patterns.ndim == 2:
                patterns = [patterns[i:i+1] for i in range(len(patterns))]
            elif patterns.ndim == 3:
                patterns = [patterns[i] for i in range(len(patterns))]
                
        n_patterns = len(patterns)
        
        # Generate labels if not provided
        if labels is None:
            labels = [f"Pattern_{i}" for i in range(n_patterns)]
            
        # Initialize matrices
        distance_matrix = np.zeros((n_patterns, n_patterns))
        
        # Compute upper triangle (matrix is symmetric)
        if self.verbose:
            total_comparisons = n_patterns * (n_patterns - 1) // 2
            pbar = tqdm(total=total_comparisons, desc="Computing similarities")
            
        # Prepare jobs for parallel processing
        jobs = []
        for i in range(n_patterns):
            for j in range(i + 1, n_patterns):
                jobs.append((i, j, patterns[i], patterns[j]))
                
        # Parallel computation
        if self.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(self._compute_distance, job) for job in jobs]
                
                for future in futures:
                    i, j, distance = future.result()
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
                    
                    if self.verbose:
                        pbar.update(1)
        else:
            # Sequential computation
            for job in jobs:
                i, j, distance = self._compute_distance(job)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
                
                if self.verbose:
                    pbar.update(1)
                    
        if self.verbose:
            pbar.close()
            
        # Convert to similarity matrix
        # Normalize distances to [0, 1] range
        max_dist = np.max(distance_matrix)
        if max_dist > 0:
            normalized_distances = distance_matrix / max_dist
        else:
            normalized_distances = distance_matrix
            
        similarity_matrix = 1 - normalized_distances
        
        # Prepare results
        results = {
            'distance_matrix': distance_matrix,
            'similarity_matrix': similarity_matrix,
            'labels': labels,
            'metadata': {
                'dtw_type': self.dtw_type,
                'distance_metric': self.distance_metric,
                'n_patterns': n_patterns,
                'normalized': self.normalize,
                'dtw_params': self.dtw_params
            }
        }
        
        # Save if requested
        if save_path:
            self.save_similarity_matrix(results, save_path)
            
        return results
        
    def _compute_distance(self, job: Tuple) -> Tuple[int, int, float]:
        """Compute DTW distance for a single pair"""
        i, j, pattern1, pattern2 = job
        result = self.dtw_calculator.compute(pattern1, pattern2)
        distance = result.normalized_distance if self.normalize else result.distance
        return i, j, distance
        
    def compute_pattern_distances(self,
                                query_patterns: Union[List[np.ndarray], np.ndarray],
                                reference_patterns: Union[List[np.ndarray], np.ndarray],
                                top_k: Optional[int] = None) -> Dict:
        """
        Compute distances from query patterns to reference patterns
        
        Parameters:
        -----------
        query_patterns : List[np.ndarray] or np.ndarray
            Query patterns to match
        reference_patterns : List[np.ndarray] or np.ndarray
            Reference pattern library
        top_k : int, optional
            Return only top-k nearest patterns for each query
            
        Returns:
        --------
        dict containing:
            - 'distances': Distance matrix (n_queries x n_references)
            - 'nearest_indices': Indices of nearest patterns
            - 'nearest_distances': Distances to nearest patterns
        """
        # Convert to lists
        if isinstance(query_patterns, np.ndarray):
            query_patterns = [query_patterns[i] for i in range(len(query_patterns))]
        if isinstance(reference_patterns, np.ndarray):
            reference_patterns = [reference_patterns[i] for i in range(len(reference_patterns))]
            
        n_queries = len(query_patterns)
        n_references = len(reference_patterns)
        
        # Compute distance matrix
        distances = np.zeros((n_queries, n_references))
        
        if self.verbose:
            pbar = tqdm(total=n_queries * n_references, desc="Computing pattern distances")
            
        for i, query in enumerate(query_patterns):
            for j, reference in enumerate(reference_patterns):
                result = self.dtw_calculator.compute(query, reference)
                distances[i, j] = result.normalized_distance if self.normalize else result.distance
                
                if self.verbose:
                    pbar.update(1)
                    
        if self.verbose:
            pbar.close()
            
        # Find nearest patterns
        nearest_indices = np.argsort(distances, axis=1)
        nearest_distances = np.take_along_axis(distances, nearest_indices, axis=1)
        
        if top_k is not None:
            nearest_indices = nearest_indices[:, :top_k]
            nearest_distances = nearest_distances[:, :top_k]
            
        return {
            'distances': distances,
            'nearest_indices': nearest_indices,
            'nearest_distances': nearest_distances
        }
        
    def find_similar_patterns(self,
                            query_pattern: np.ndarray,
                            pattern_library: Union[List[np.ndarray], np.ndarray],
                            threshold: float = 0.1,
                            top_k: Optional[int] = None) -> Dict:
        """
        Find patterns similar to a query pattern
        
        Parameters:
        -----------
        query_pattern : np.ndarray
            Query pattern
        pattern_library : List[np.ndarray] or np.ndarray
            Library of patterns to search
        threshold : float
            Distance threshold for similarity
        top_k : int, optional
            Return only top-k most similar patterns
            
        Returns:
        --------
        dict containing:
            - 'indices': Indices of similar patterns
            - 'distances': Distances to similar patterns
            - 'patterns': The similar patterns themselves
        """
        if isinstance(pattern_library, np.ndarray):
            pattern_library = [pattern_library[i] for i in range(len(pattern_library))]
            
        distances = []
        for pattern in pattern_library:
            result = self.dtw_calculator.compute(query_pattern, pattern)
            distance = result.normalized_distance if self.normalize else result.distance
            distances.append(distance)
            
        distances = np.array(distances)
        
        # Find patterns within threshold
        similar_mask = distances <= threshold
        similar_indices = np.where(similar_mask)[0]
        similar_distances = distances[similar_mask]
        
        # Sort by distance
        sort_idx = np.argsort(similar_distances)
        similar_indices = similar_indices[sort_idx]
        similar_distances = similar_distances[sort_idx]
        
        # Apply top-k if specified
        if top_k is not None and len(similar_indices) > top_k:
            similar_indices = similar_indices[:top_k]
            similar_distances = similar_distances[:top_k]
            
        # Get the actual patterns
        similar_patterns = [pattern_library[idx] for idx in similar_indices]
        
        return {
            'indices': similar_indices,
            'distances': similar_distances,
            'patterns': similar_patterns
        }
        
    def compute_cross_similarity(self,
                               patterns_a: Union[List[np.ndarray], np.ndarray],
                               patterns_b: Union[List[np.ndarray], np.ndarray],
                               labels_a: Optional[List[str]] = None,
                               labels_b: Optional[List[str]] = None) -> Dict:
        """
        Compute cross-similarity matrix between two sets of patterns
        
        Parameters:
        -----------
        patterns_a : List[np.ndarray] or np.ndarray
            First set of patterns
        patterns_b : List[np.ndarray] or np.ndarray
            Second set of patterns
        labels_a : List[str], optional
            Labels for first set
        labels_b : List[str], optional
            Labels for second set
            
        Returns:
        --------
        dict containing cross-similarity information
        """
        # Convert to lists
        if isinstance(patterns_a, np.ndarray):
            patterns_a = [patterns_a[i] for i in range(len(patterns_a))]
        if isinstance(patterns_b, np.ndarray):
            patterns_b = [patterns_b[i] for i in range(len(patterns_b))]
            
        n_a = len(patterns_a)
        n_b = len(patterns_b)
        
        # Generate labels if needed
        if labels_a is None:
            labels_a = [f"A_{i}" for i in range(n_a)]
        if labels_b is None:
            labels_b = [f"B_{i}" for i in range(n_b)]
            
        # Compute cross-similarity matrix
        cross_distances = np.zeros((n_a, n_b))
        
        if self.verbose:
            pbar = tqdm(total=n_a * n_b, desc="Computing cross-similarities")
            
        for i, pattern_a in enumerate(patterns_a):
            for j, pattern_b in enumerate(patterns_b):
                result = self.dtw_calculator.compute(pattern_a, pattern_b)
                cross_distances[i, j] = result.normalized_distance if self.normalize else result.distance
                
                if self.verbose:
                    pbar.update(1)
                    
        if self.verbose:
            pbar.close()
            
        # Normalize and convert to similarity
        max_dist = np.max(cross_distances)
        if max_dist > 0:
            normalized_distances = cross_distances / max_dist
        else:
            normalized_distances = cross_distances
            
        cross_similarities = 1 - normalized_distances
        
        return {
            'distance_matrix': cross_distances,
            'similarity_matrix': cross_similarities,
            'labels_a': labels_a,
            'labels_b': labels_b,
            'shape': (n_a, n_b)
        }
        
    def save_similarity_matrix(self, results: Dict, save_path: str):
        """Save similarity matrix results to file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.suffix == '.h5' or save_path.suffix == '.hdf5':
            # Save as HDF5
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('distance_matrix', data=results['distance_matrix'])
                f.create_dataset('similarity_matrix', data=results['similarity_matrix'])
                
                # Save labels as variable-length strings
                dt = h5py.special_dtype(vlen=str)
                labels_dataset = f.create_dataset('labels', (len(results['labels']),), dtype=dt)
                for i, label in enumerate(results['labels']):
                    labels_dataset[i] = label
                    
                # Save metadata
                for key, value in results['metadata'].items():
                    f.attrs[key] = str(value)
        else:
            # Save as pickle
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
                
        self.logger.info(f"Saved similarity matrix to {save_path}")
        
    def load_similarity_matrix(self, load_path: str) -> Dict:
        """Load similarity matrix results from file"""
        load_path = Path(load_path)
        
        if load_path.suffix == '.h5' or load_path.suffix == '.hdf5':
            # Load from HDF5
            with h5py.File(load_path, 'r') as f:
                results = {
                    'distance_matrix': f['distance_matrix'][:],
                    'similarity_matrix': f['similarity_matrix'][:],
                    'labels': [label.decode() if isinstance(label, bytes) else label 
                              for label in f['labels'][:]],
                    'metadata': {key: value for key, value in f.attrs.items()}
                }
        else:
            # Load from pickle
            with open(load_path, 'rb') as f:
                results = pickle.load(f)
                
        return results
        
    def compute_pattern_statistics(self, similarity_matrix: np.ndarray, 
                                 labels: Optional[List[str]] = None) -> Dict:
        """
        Compute statistics from similarity matrix
        
        Parameters:
        -----------
        similarity_matrix : np.ndarray
            Pairwise similarity matrix
        labels : List[str], optional
            Pattern labels
            
        Returns:
        --------
        dict containing various statistics
        """
        n_patterns = len(similarity_matrix)
        
        # Compute statistics
        stats = {
            'mean_similarity': np.mean(similarity_matrix[np.triu_indices(n_patterns, k=1)]),
            'std_similarity': np.std(similarity_matrix[np.triu_indices(n_patterns, k=1)]),
            'min_similarity': np.min(similarity_matrix[np.triu_indices(n_patterns, k=1)]),
            'max_similarity': np.max(similarity_matrix[np.triu_indices(n_patterns, k=1)]),
            'median_similarity': np.median(similarity_matrix[np.triu_indices(n_patterns, k=1)])
        }
        
        # Most similar pairs
        upper_tri = np.triu_indices(n_patterns, k=1)
        similarities = similarity_matrix[upper_tri]
        sorted_idx = np.argsort(similarities)[::-1]
        
        most_similar_pairs = []
        for idx in sorted_idx[:10]:  # Top 10 pairs
            i, j = upper_tri[0][idx], upper_tri[1][idx]
            pair_info = {
                'indices': (i, j),
                'similarity': similarities[idx]
            }
            if labels:
                pair_info['labels'] = (labels[i], labels[j])
            most_similar_pairs.append(pair_info)
            
        stats['most_similar_pairs'] = most_similar_pairs
        
        # Pattern connectivity (average similarity to other patterns)
        pattern_connectivity = []
        for i in range(n_patterns):
            mask = np.ones(n_patterns, dtype=bool)
            mask[i] = False
            avg_similarity = np.mean(similarity_matrix[i, mask])
            
            conn_info = {
                'index': i,
                'avg_similarity': avg_similarity
            }
            if labels:
                conn_info['label'] = labels[i]
            pattern_connectivity.append(conn_info)
            
        # Sort by connectivity
        pattern_connectivity.sort(key=lambda x: x['avg_similarity'], reverse=True)
        stats['pattern_connectivity'] = pattern_connectivity
        
        return stats
