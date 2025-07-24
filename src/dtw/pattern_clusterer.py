"""
Pattern Clusterer for Time Series

Implements hierarchical clustering and other clustering algorithms
for grouping similar time series patterns based on DTW distances.
"""

import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import logging

from .similarity_engine import SimilarityEngine


class PatternClusterer:
    """
    Clusters time series patterns based on DTW similarity/distance matrices.
    Supports various clustering algorithms with a focus on hierarchical clustering.
    """
    
    def __init__(self,
                 clustering_method: str = 'hierarchical',
                 linkage_method: str = 'average',
                 distance_threshold: Optional[float] = None,
                 n_clusters: Optional[int] = None,
                 verbose: bool = True):
        """
        Initialize Pattern Clusterer
        
        Parameters:
        -----------
        clustering_method : str
            Clustering algorithm ('hierarchical', 'dbscan', 'kmeans')
        linkage_method : str
            Linkage method for hierarchical clustering
            ('single', 'complete', 'average', 'ward')
        distance_threshold : float, optional
            Distance threshold for clustering
        n_clusters : int, optional
            Number of clusters (if not using distance threshold)
        verbose : bool
            Whether to show progress information
        """
        self.clustering_method = clustering_method
        self.linkage_method = linkage_method
        self.distance_threshold = distance_threshold
        self.n_clusters = n_clusters
        self.verbose = verbose
        
        # Validate parameters
        if clustering_method == 'hierarchical' and distance_threshold is None and n_clusters is None:
            raise ValueError("Either distance_threshold or n_clusters must be specified for hierarchical clustering")
            
        self.logger = logging.getLogger(__name__)
        
    def fit_predict(self,
                   patterns: Union[List[np.ndarray], np.ndarray],
                   similarity_matrix: Optional[np.ndarray] = None,
                   distance_matrix: Optional[np.ndarray] = None,
                   similarity_engine: Optional[SimilarityEngine] = None) -> Dict:
        """
        Cluster patterns based on similarity/distance matrix
        
        Parameters:
        -----------
        patterns : List[np.ndarray] or np.ndarray
            Time series patterns to cluster
        similarity_matrix : np.ndarray, optional
            Pre-computed similarity matrix
        distance_matrix : np.ndarray, optional
            Pre-computed distance matrix
        similarity_engine : SimilarityEngine, optional
            Engine to compute similarities if matrices not provided
            
        Returns:
        --------
        dict containing:
            - 'labels': Cluster labels for each pattern
            - 'n_clusters': Number of clusters found
            - 'cluster_centers': Representative patterns for each cluster
            - 'cluster_members': Indices of patterns in each cluster
            - 'linkage_matrix': Linkage matrix (for hierarchical)
            - 'silhouette_score': Clustering quality score
        """
        # Convert patterns to list if needed
        if isinstance(patterns, np.ndarray):
            if patterns.ndim == 3:
                patterns = [patterns[i] for i in range(len(patterns))]
            else:
                patterns = list(patterns)
                
        n_patterns = len(patterns)
        
        # Get distance matrix
        if distance_matrix is None:
            if similarity_matrix is not None:
                # Convert similarity to distance
                distance_matrix = 1 - similarity_matrix
            elif similarity_engine is not None:
                # Compute similarity matrix
                if self.verbose:
                    print("Computing similarity matrix...")
                results = similarity_engine.compute_similarity_matrix(patterns)
                distance_matrix = results['distance_matrix']
                # Normalize distances
                if distance_matrix.max() > 0:
                    distance_matrix = distance_matrix / distance_matrix.max()
            else:
                raise ValueError("Must provide either distance_matrix, similarity_matrix, or similarity_engine")
                
        # Apply clustering algorithm
        if self.clustering_method == 'hierarchical':
            results = self._hierarchical_clustering(patterns, distance_matrix)
        elif self.clustering_method == 'dbscan':
            results = self._dbscan_clustering(patterns, distance_matrix)
        elif self.clustering_method == 'kmeans':
            results = self._kmeans_clustering(patterns, distance_matrix)
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
            
        return results
        
    def _hierarchical_clustering(self, patterns: List[np.ndarray], 
                               distance_matrix: np.ndarray) -> Dict:
        """Perform hierarchical clustering"""
        # Convert distance matrix to condensed form for scipy
        condensed_dist = squareform(distance_matrix, checks=False)
        
        # Compute linkage matrix
        if self.verbose:
            print(f"Computing {self.linkage_method} linkage...")
        linkage_matrix = linkage(condensed_dist, method=self.linkage_method)
        
        # Get cluster labels
        if self.distance_threshold is not None:
            labels = fcluster(linkage_matrix, self.distance_threshold, 
                            criterion='distance') - 1  # Convert to 0-based
        else:
            labels = fcluster(linkage_matrix, self.n_clusters, 
                            criterion='maxclust') - 1  # Convert to 0-based
            
        # Organize results
        results = self._organize_clustering_results(patterns, labels, distance_matrix)
        results['linkage_matrix'] = linkage_matrix
        
        return results
        
    def _dbscan_clustering(self, patterns: List[np.ndarray], 
                          distance_matrix: np.ndarray) -> Dict:
        """Perform DBSCAN clustering"""
        # DBSCAN parameters
        eps = self.distance_threshold if self.distance_threshold is not None else 0.5
        min_samples = max(2, int(0.05 * len(patterns)))  # 5% of patterns
        
        if self.verbose:
            print(f"Running DBSCAN with eps={eps}, min_samples={min_samples}")
            
        # Run DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        labels = dbscan.fit_predict(distance_matrix)
        
        # Handle noise points (-1 labels)
        # Assign noise points to nearest cluster
        noise_mask = labels == -1
        if np.any(noise_mask):
            noise_indices = np.where(noise_mask)[0]
            for idx in noise_indices:
                # Find nearest non-noise point
                distances = distance_matrix[idx]
                non_noise_mask = labels != -1
                if np.any(non_noise_mask):
                    nearest_idx = np.argmin(distances[non_noise_mask])
                    nearest_label = labels[non_noise_mask][nearest_idx]
                    labels[idx] = nearest_label
                    
        # Organize results
        results = self._organize_clustering_results(patterns, labels, distance_matrix)
        results['n_noise_points'] = np.sum(noise_mask)
        
        return results
        
    def _kmeans_clustering(self, patterns: List[np.ndarray], 
                          distance_matrix: np.ndarray) -> Dict:
        """Perform K-means clustering on pattern features"""
        # Extract features from patterns for K-means
        features = self._extract_pattern_features(patterns)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Determine number of clusters
        if self.n_clusters is None:
            # Use elbow method to estimate
            self.n_clusters = self._estimate_n_clusters(features_scaled)
            
        if self.verbose:
            print(f"Running K-means with {self.n_clusters} clusters")
            
        # Run K-means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        
        # Organize results
        results = self._organize_clustering_results(patterns, labels, distance_matrix)
        results['feature_importance'] = self._compute_feature_importance(features, labels)
        
        return results
        
    def _organize_clustering_results(self, patterns: List[np.ndarray], 
                                   labels: np.ndarray, 
                                   distance_matrix: np.ndarray) -> Dict:
        """Organize clustering results into structured format"""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Find cluster members
        cluster_members = {}
        for label in unique_labels:
            cluster_members[int(label)] = np.where(labels == label)[0].tolist()
            
        # Find cluster centers (medoids)
        cluster_centers = self._find_cluster_centers(patterns, labels, distance_matrix)
        
        # Compute silhouette score
        if n_clusters > 1:
            from sklearn.metrics import silhouette_score
            silhouette = silhouette_score(distance_matrix, labels, metric='precomputed')
        else:
            silhouette = 0.0
            
        # Compute cluster statistics
        cluster_stats = self._compute_cluster_statistics(patterns, labels, distance_matrix)
        
        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'cluster_centers': cluster_centers,
            'cluster_members': cluster_members,
            'silhouette_score': silhouette,
            'cluster_stats': cluster_stats
        }
        
    def _find_cluster_centers(self, patterns: List[np.ndarray], 
                            labels: np.ndarray, 
                            distance_matrix: np.ndarray) -> Dict[int, int]:
        """Find medoid (most central pattern) for each cluster"""
        cluster_centers = {}
        
        for label in np.unique(labels):
            # Get indices of patterns in this cluster
            cluster_indices = np.where(labels == label)[0]
            
            if len(cluster_indices) == 1:
                cluster_centers[int(label)] = int(cluster_indices[0])
            else:
                # Find medoid: pattern with minimum sum of distances to others
                cluster_dist_matrix = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
                sum_distances = np.sum(cluster_dist_matrix, axis=1)
                medoid_idx = cluster_indices[np.argmin(sum_distances)]
                cluster_centers[int(label)] = int(medoid_idx)
                
        return cluster_centers
        
    def _compute_cluster_statistics(self, patterns: List[np.ndarray], 
                                  labels: np.ndarray, 
                                  distance_matrix: np.ndarray) -> Dict:
        """Compute statistics for each cluster"""
        stats = {}
        
        for label in np.unique(labels):
            cluster_indices = np.where(labels == label)[0]
            n_members = len(cluster_indices)
            
            # Intra-cluster distances
            if n_members > 1:
                cluster_dist_matrix = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
                upper_tri = np.triu_indices(n_members, k=1)
                intra_distances = cluster_dist_matrix[upper_tri]
                
                cluster_stat = {
                    'size': n_members,
                    'mean_intra_distance': np.mean(intra_distances),
                    'std_intra_distance': np.std(intra_distances),
                    'max_intra_distance': np.max(intra_distances),
                    'compactness': 1 / (1 + np.mean(intra_distances))  # Higher is more compact
                }
            else:
                cluster_stat = {
                    'size': n_members,
                    'mean_intra_distance': 0,
                    'std_intra_distance': 0,
                    'max_intra_distance': 0,
                    'compactness': 1.0
                }
                
            # Inter-cluster distances (to other clusters)
            other_indices = np.where(labels != label)[0]
            if len(other_indices) > 0:
                inter_distances = distance_matrix[np.ix_(cluster_indices, other_indices)].flatten()
                cluster_stat['mean_inter_distance'] = np.mean(inter_distances)
                cluster_stat['separation'] = cluster_stat['mean_inter_distance'] / (cluster_stat['mean_intra_distance'] + 1e-10)
            else:
                cluster_stat['mean_inter_distance'] = 0
                cluster_stat['separation'] = np.inf
                
            stats[int(label)] = cluster_stat
            
        return stats
        
    def _extract_pattern_features(self, patterns: List[np.ndarray]) -> np.ndarray:
        """Extract statistical features from patterns for K-means"""
        features = []
        
        for pattern in patterns:
            # Flatten multivariate patterns
            if pattern.ndim > 1:
                pattern = pattern.flatten()
                
            # Extract features
            feat = [
                np.mean(pattern),
                np.std(pattern),
                np.min(pattern),
                np.max(pattern),
                np.percentile(pattern, 25),
                np.percentile(pattern, 75),
                len(pattern),
                np.mean(np.diff(pattern)),  # Average change
                np.std(np.diff(pattern)),   # Volatility
            ]
            
            # Add spectral features
            fft = np.fft.fft(pattern)
            power_spectrum = np.abs(fft) ** 2
            feat.extend([
                np.argmax(power_spectrum[1:len(pattern)//2]) + 1,  # Dominant frequency
                np.sum(power_spectrum[1:len(pattern)//2])  # Total power
            ])
            
            features.append(feat)
            
        return np.array(features)
        
    def _estimate_n_clusters(self, features: np.ndarray, max_k: int = 10) -> int:
        """Estimate optimal number of clusters using elbow method"""
        inertias = []
        K = range(2, min(max_k + 1, len(features)))
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)
            
        # Find elbow point
        if len(inertias) > 2:
            # Calculate second derivative
            second_diff = np.diff(np.diff(inertias))
            elbow_idx = np.argmax(second_diff) + 2  # +2 because of double diff and 0-indexing
            optimal_k = list(K)[elbow_idx]
        else:
            optimal_k = 2
            
        return optimal_k
        
    def _compute_feature_importance(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute feature importance for clustering"""
        n_features = features.shape[1]
        importance = np.zeros(n_features)
        
        # Compute F-statistic for each feature
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters > 1:
            for i in range(n_features):
                # Between-cluster variance
                cluster_means = [np.mean(features[labels == label, i]) for label in unique_labels]
                overall_mean = np.mean(features[:, i])
                between_var = sum(np.sum(labels == label) * (mean - overall_mean) ** 2 
                                for label, mean in zip(unique_labels, cluster_means))
                
                # Within-cluster variance
                within_var = sum(np.sum((features[labels == label, i] - mean) ** 2)
                               for label, mean in zip(unique_labels, cluster_means))
                
                # F-statistic
                if within_var > 0:
                    importance[i] = (between_var / (n_clusters - 1)) / (within_var / (len(features) - n_clusters))
                    
        return importance / (np.sum(importance) + 1e-10)  # Normalize
        
    def plot_dendrogram(self, linkage_matrix: np.ndarray, 
                       labels: Optional[List[str]] = None,
                       figsize: Tuple[int, int] = (12, 8),
                       save_path: Optional[str] = None):
        """
        Plot hierarchical clustering dendrogram
        
        Parameters:
        -----------
        linkage_matrix : np.ndarray
            Linkage matrix from hierarchical clustering
        labels : List[str], optional
            Labels for patterns
        figsize : Tuple[int, int]
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        plt.figure(figsize=figsize)
        
        # Create dendrogram
        dendrogram_data = dendrogram(
            linkage_matrix,
            labels=labels,
            leaf_rotation=90,
            leaf_font_size=10
        )
        
        plt.title('Hierarchical Clustering Dendrogram', fontsize=16)
        plt.xlabel('Pattern', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        
        # Add distance threshold line if specified
        if self.distance_threshold is not None:
            plt.axhline(y=self.distance_threshold, c='r', linestyle='--', 
                       label=f'Distance threshold = {self.distance_threshold:.3f}')
            plt.legend()
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_cluster_heatmap(self, distance_matrix: np.ndarray, 
                           labels: np.ndarray,
                           pattern_labels: Optional[List[str]] = None,
                           figsize: Tuple[int, int] = (10, 10),
                           save_path: Optional[str] = None):
        """
        Plot heatmap of distance matrix ordered by clusters
        
        Parameters:
        -----------
        distance_matrix : np.ndarray
            Pairwise distance matrix
        labels : np.ndarray
            Cluster labels
        pattern_labels : List[str], optional
            Labels for patterns
        figsize : Tuple[int, int]
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        # Sort patterns by cluster
        sorted_indices = np.argsort(labels)
        sorted_matrix = distance_matrix[np.ix_(sorted_indices, sorted_indices)]
        sorted_labels = labels[sorted_indices]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(sorted_matrix, cmap='viridis_r', square=True, 
                   cbar_kws={'label': 'DTW Distance'})
        
        # Add cluster boundaries
        boundaries = np.where(np.diff(sorted_labels) != 0)[0] + 1
        for boundary in boundaries:
            plt.axhline(y=boundary, color='red', linewidth=2)
            plt.axvline(x=boundary, color='red', linewidth=2)
            
        # Add labels if provided
        if pattern_labels is not None:
            sorted_pattern_labels = [pattern_labels[i] for i in sorted_indices]
            plt.xticks(range(len(sorted_pattern_labels)), sorted_pattern_labels, rotation=90)
            plt.yticks(range(len(sorted_pattern_labels)), sorted_pattern_labels)
            
        plt.title('Clustered Distance Matrix Heatmap', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def save_clustering_results(self, results: Dict, save_path: str):
        """Save clustering results to file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
            
        self.logger.info(f"Saved clustering results to {save_path}")
        
    def load_clustering_results(self, load_path: str) -> Dict:
        """Load clustering results from file"""
        with open(load_path, 'rb') as f:
            results = pickle.load(f)
            
        return results
