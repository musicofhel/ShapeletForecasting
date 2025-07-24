"""
Dynamic Time Warping Calculator

Implements various DTW algorithms for time series similarity measurement.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Union
import warnings
from scipy.spatial.distance import cdist
from dataclasses import dataclass

# Try to import numba, but make it optional
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    warnings.warn("Numba not available. DTW will run without JIT compilation.", UserWarning)
    # Create dummy decorator
    def jit(nopython=True):
        def decorator(func):
            return func
        return decorator


@dataclass
class DTWResult:
    """Container for DTW computation results"""
    distance: float
    path: np.ndarray
    cost_matrix: Optional[np.ndarray] = None
    normalized_distance: Optional[float] = None
    
    
class DTWCalculator:
    """
    Standard Dynamic Time Warping implementation with various distance metrics
    and normalization options.
    """
    
    def __init__(self, 
                 distance_metric: str = 'euclidean',
                 normalize: bool = True,
                 return_path: bool = True,
                 return_cost_matrix: bool = False):
        """
        Initialize DTW Calculator
        
        Parameters:
        -----------
        distance_metric : str
            Distance metric to use ('euclidean', 'manhattan', 'cosine')
        normalize : bool
            Whether to normalize the DTW distance by path length
        return_path : bool
            Whether to return the optimal warping path
        return_cost_matrix : bool
            Whether to return the full cost matrix
        """
        self.distance_metric = distance_metric
        self.normalize = normalize
        self.return_path = return_path
        self.return_cost_matrix = return_cost_matrix
        
    def compute(self, x: np.ndarray, y: np.ndarray) -> DTWResult:
        """
        Compute DTW distance between two time series
        
        Parameters:
        -----------
        x : np.ndarray
            First time series (shape: (n,) or (n, d) for multivariate)
        y : np.ndarray
            Second time series (shape: (m,) or (m, d) for multivariate)
            
        Returns:
        --------
        DTWResult object containing distance and optional path/cost matrix
        """
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        
        if x.shape[0] == 1:
            x = x.T
        if y.shape[0] == 1:
            y = y.T
            
        # Compute distance matrix
        if self.distance_metric == 'euclidean':
            dist_matrix = cdist(x, y, metric='euclidean')
        elif self.distance_metric == 'manhattan':
            dist_matrix = cdist(x, y, metric='cityblock')
        elif self.distance_metric == 'cosine':
            dist_matrix = cdist(x, y, metric='cosine')
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
            
        # Compute DTW
        cost_matrix, distance = self._compute_dtw_matrix(dist_matrix)
        
        # Extract path if requested
        path = None
        if self.return_path:
            path = self._extract_path(cost_matrix)
            
        # Normalize if requested
        normalized_distance = None
        if self.normalize:
            path_length = len(path) if path is not None else len(x) + len(y)
            normalized_distance = distance / path_length
            
        return DTWResult(
            distance=distance,
            path=path,
            cost_matrix=cost_matrix if self.return_cost_matrix else None,
            normalized_distance=normalized_distance
        )
        
    @staticmethod
    @jit(nopython=True)
    def _compute_dtw_matrix(dist_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute DTW cost matrix using dynamic programming
        
        Parameters:
        -----------
        dist_matrix : np.ndarray
            Pairwise distance matrix between time series points
            
        Returns:
        --------
        cost_matrix : np.ndarray
            Accumulated cost matrix
        distance : float
            DTW distance (final cost)
        """
        n, m = dist_matrix.shape
        cost_matrix = np.full((n + 1, m + 1), np.inf)
        cost_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = dist_matrix[i-1, j-1]
                cost_matrix[i, j] = cost + min(
                    cost_matrix[i-1, j],    # insertion
                    cost_matrix[i, j-1],    # deletion
                    cost_matrix[i-1, j-1]   # match
                )
                
        return cost_matrix[1:, 1:], cost_matrix[n, m]
        
    @staticmethod
    def _extract_path(cost_matrix: np.ndarray) -> np.ndarray:
        """
        Extract optimal warping path from cost matrix
        
        Parameters:
        -----------
        cost_matrix : np.ndarray
            DTW cost matrix
            
        Returns:
        --------
        path : np.ndarray
            Optimal warping path as array of (i, j) coordinates
        """
        n, m = cost_matrix.shape
        path = []
        
        i, j = n - 1, m - 1
        path.append((i, j))
        
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                candidates = [
                    (i-1, j, cost_matrix[i-1, j]),
                    (i, j-1, cost_matrix[i, j-1]),
                    (i-1, j-1, cost_matrix[i-1, j-1])
                ]
                i, j, _ = min(candidates, key=lambda x: x[2])
                
            path.append((i, j))
            
        path.reverse()
        return np.array(path)
        

class FastDTW:
    """
    FastDTW implementation for improved performance on long time series.
    Uses a multi-resolution approach with linear time/space complexity.
    """
    
    def __init__(self,
                 radius: int = 1,
                 distance_metric: str = 'euclidean',
                 normalize: bool = True):
        """
        Initialize FastDTW
        
        Parameters:
        -----------
        radius : int
            Radius parameter for FastDTW (controls approximation quality)
        distance_metric : str
            Distance metric to use
        normalize : bool
            Whether to normalize the DTW distance
        """
        self.radius = radius
        self.distance_metric = distance_metric
        self.normalize = normalize
        self.base_dtw = DTWCalculator(
            distance_metric=distance_metric,
            normalize=False,
            return_path=True,
            return_cost_matrix=False
        )
        
    def compute(self, x: np.ndarray, y: np.ndarray) -> DTWResult:
        """
        Compute FastDTW distance
        
        Parameters:
        -----------
        x : np.ndarray
            First time series
        y : np.ndarray
            Second time series
            
        Returns:
        --------
        DTWResult object
        """
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        
        if x.shape[0] == 1:
            x = x.T
        if y.shape[0] == 1:
            y = y.T
            
        # Base case: use standard DTW for small series
        if len(x) < 10 or len(y) < 10:
            return self.base_dtw.compute(x, y)
            
        # Recursive FastDTW
        distance, path = self._fast_dtw(x, y, self.radius)
        
        # Normalize if requested
        normalized_distance = None
        if self.normalize:
            normalized_distance = distance / len(path)
            
        return DTWResult(
            distance=distance,
            path=np.array(path),
            normalized_distance=normalized_distance
        )
        
    def _fast_dtw(self, x: np.ndarray, y: np.ndarray, radius: int) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Recursive FastDTW implementation
        """
        min_size = radius + 2
        
        if len(x) <= min_size or len(y) <= min_size:
            # Base case: compute standard DTW
            result = self.base_dtw.compute(x, y)
            return result.distance, result.path.tolist()
            
        # Downsample
        x_shrunk = self._downsample(x)
        y_shrunk = self._downsample(y)
        
        # Recursive call
        distance, path = self._fast_dtw(x_shrunk, y_shrunk, radius)
        
        # Project path to higher resolution
        window = self._expand_window(path, len(x), len(y), radius)
        
        # Compute DTW in the constrained window
        return self._windowed_dtw(x, y, window)
        
    @staticmethod
    def _downsample(series: np.ndarray) -> np.ndarray:
        """Downsample time series by factor of 2"""
        if len(series) % 2 == 0:
            return (series[::2] + series[1::2]) / 2
        else:
            return (series[:-1:2] + series[1::2]) / 2
            
    @staticmethod
    def _expand_window(path: List[Tuple[int, int]], 
                      len_x: int, 
                      len_y: int, 
                      radius: int) -> List[Tuple[int, int]]:
        """Expand low-resolution path to high-resolution window"""
        window = set()
        
        for i, j in path:
            for x in range(max(0, 2*i - radius), min(len_x, 2*i + radius + 1)):
                for y in range(max(0, 2*j - radius), min(len_y, 2*j + radius + 1)):
                    window.add((x, y))
                    
        return sorted(list(window))
        
    def _windowed_dtw(self, x: np.ndarray, y: np.ndarray, 
                     window: List[Tuple[int, int]]) -> Tuple[float, List[Tuple[int, int]]]:
        """Compute DTW within a constrained window"""
        # Create window mask
        n, m = len(x), len(y)
        mask = np.full((n, m), False)
        for i, j in window:
            mask[i, j] = True
            
        # Compute distances only in window
        if self.distance_metric == 'euclidean':
            dist_func = lambda a, b: np.linalg.norm(a - b)
        elif self.distance_metric == 'manhattan':
            dist_func = lambda a, b: np.sum(np.abs(a - b))
        elif self.distance_metric == 'cosine':
            dist_func = lambda a, b: 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            
        # Dynamic programming with window constraint
        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if mask[i-1, j-1]:
                    d = dist_func(x[i-1], y[j-1])
                    cost[i, j] = d + min(
                        cost[i-1, j],
                        cost[i, j-1],
                        cost[i-1, j-1]
                    )
                    
        # Extract path
        path = []
        i, j = n, m
        while i > 0 and j > 0:
            path.append((i-1, j-1))
            
            candidates = []
            if i > 0 and cost[i-1, j] < np.inf:
                candidates.append((i-1, j, cost[i-1, j]))
            if j > 0 and cost[i, j-1] < np.inf:
                candidates.append((i, j-1, cost[i, j-1]))
            if i > 0 and j > 0 and cost[i-1, j-1] < np.inf:
                candidates.append((i-1, j-1, cost[i-1, j-1]))
                
            if candidates:
                i, j, _ = min(candidates, key=lambda x: x[2])
            else:
                break
                
        path.reverse()
        return cost[n, m], path


class ConstrainedDTW:
    """
    DTW with various constraint types:
    - Sakoe-Chiba band
    - Itakura parallelogram
    - Custom constraints
    """
    
    def __init__(self,
                 constraint_type: str = 'sakoe_chiba',
                 constraint_param: Union[int, float] = 10,
                 distance_metric: str = 'euclidean',
                 normalize: bool = True):
        """
        Initialize Constrained DTW
        
        Parameters:
        -----------
        constraint_type : str
            Type of constraint ('sakoe_chiba', 'itakura', 'custom')
        constraint_param : int or float
            Parameter for constraint (band width for Sakoe-Chiba, 
            slope for Itakura)
        distance_metric : str
            Distance metric to use
        normalize : bool
            Whether to normalize the DTW distance
        """
        self.constraint_type = constraint_type
        self.constraint_param = constraint_param
        self.distance_metric = distance_metric
        self.normalize = normalize
        
    def compute(self, x: np.ndarray, y: np.ndarray, 
                custom_mask: Optional[np.ndarray] = None) -> DTWResult:
        """
        Compute constrained DTW distance
        
        Parameters:
        -----------
        x : np.ndarray
            First time series
        y : np.ndarray
            Second time series
        custom_mask : np.ndarray, optional
            Custom constraint mask (for constraint_type='custom')
            
        Returns:
        --------
        DTWResult object
        """
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        
        if x.shape[0] == 1:
            x = x.T
        if y.shape[0] == 1:
            y = y.T
            
        n, m = len(x), len(y)
        
        # Generate constraint mask
        if self.constraint_type == 'sakoe_chiba':
            mask = self._sakoe_chiba_mask(n, m, self.constraint_param)
        elif self.constraint_type == 'itakura':
            mask = self._itakura_mask(n, m, self.constraint_param)
        elif self.constraint_type == 'custom':
            if custom_mask is None:
                raise ValueError("Custom mask must be provided for custom constraint type")
            mask = custom_mask
        else:
            raise ValueError(f"Unknown constraint type: {self.constraint_type}")
            
        # Compute distance matrix
        if self.distance_metric == 'euclidean':
            dist_matrix = cdist(x, y, metric='euclidean')
        elif self.distance_metric == 'manhattan':
            dist_matrix = cdist(x, y, metric='cityblock')
        elif self.distance_metric == 'cosine':
            dist_matrix = cdist(x, y, metric='cosine')
            
        # Apply mask
        dist_matrix[~mask] = np.inf
        
        # Compute constrained DTW
        cost_matrix, distance = self._compute_constrained_dtw(dist_matrix, mask)
        
        # Extract path
        path = self._extract_constrained_path(cost_matrix, mask)
        
        # Normalize if requested
        normalized_distance = None
        if self.normalize:
            normalized_distance = distance / len(path)
            
        return DTWResult(
            distance=distance,
            path=path,
            cost_matrix=cost_matrix,
            normalized_distance=normalized_distance
        )
        
    @staticmethod
    def _sakoe_chiba_mask(n: int, m: int, band_width: int) -> np.ndarray:
        """Generate Sakoe-Chiba band constraint mask"""
        mask = np.zeros((n, m), dtype=bool)
        
        for i in range(n):
            j_start = max(0, i - band_width)
            j_end = min(m, i + band_width + 1)
            mask[i, j_start:j_end] = True
            
        return mask
        
    @staticmethod
    def _itakura_mask(n: int, m: int, max_slope: float = 2.0) -> np.ndarray:
        """Generate Itakura parallelogram constraint mask"""
        mask = np.zeros((n, m), dtype=bool)
        
        for i in range(n):
            # Upper and lower bounds
            j_min = max(0, int((i / max_slope)))
            j_max = min(m - 1, int((i * max_slope)))
            
            # Additional constraints for parallelogram shape
            j_min = max(j_min, i - (n - m))
            j_max = min(j_max, i + (m - n))
            
            mask[i, j_min:j_max+1] = True
            
        return mask
        
    @staticmethod
    @jit(nopython=True)
    def _compute_constrained_dtw(dist_matrix: np.ndarray, 
                                mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute DTW with constraints"""
        n, m = dist_matrix.shape
        cost_matrix = np.full((n + 1, m + 1), np.inf)
        cost_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if mask[i-1, j-1]:
                    cost = dist_matrix[i-1, j-1]
                    cost_matrix[i, j] = cost + min(
                        cost_matrix[i-1, j],
                        cost_matrix[i, j-1],
                        cost_matrix[i-1, j-1]
                    )
                    
        return cost_matrix[1:, 1:], cost_matrix[n, m]
        
    @staticmethod
    def _extract_constrained_path(cost_matrix: np.ndarray, 
                                 mask: np.ndarray) -> np.ndarray:
        """Extract path with constraints"""
        n, m = cost_matrix.shape
        path = []
        
        i, j = n - 1, m - 1
        path.append((i, j))
        
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                candidates = []
                if i > 0 and mask[i-1, j]:
                    candidates.append((i-1, j, cost_matrix[i-1, j]))
                if j > 0 and mask[i, j-1]:
                    candidates.append((i, j-1, cost_matrix[i, j-1]))
                if i > 0 and j > 0 and mask[i-1, j-1]:
                    candidates.append((i-1, j-1, cost_matrix[i-1, j-1]))
                    
                if candidates:
                    i, j, _ = min(candidates, key=lambda x: x[2])
                else:
                    # Fallback if no valid candidates
                    if i > 0:
                        i -= 1
                    else:
                        j -= 1
                        
            path.append((i, j))
            
        path.reverse()
        return np.array(path)
