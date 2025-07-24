"""
Pattern Matching System for Financial Time Series

This module provides DTW-based pattern matching, template matching algorithms,
similarity scoring, and historical outcome retrieval for financial forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr, spearmanr
import json
import pickle
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PatternMatch:
    """Container for pattern matching results"""
    template_id: str
    similarity_score: float
    dtw_distance: float
    correlation: float
    start_idx: int
    end_idx: int
    template_length: int
    query_length: int
    alignment_path: List[Tuple[int, int]]
    historical_outcomes: Dict[str, Any]
    metadata: Dict[str, Any]


class DTWMatcher:
    """Dynamic Time Warping pattern matcher"""
    
    def __init__(self, window_type: str = 'sakoe_chiba', window_size: int = 10):
        """
        Initialize DTW matcher
        
        Args:
            window_type: Type of warping window ('sakoe_chiba', 'itakura', 'none')
            window_size: Size of warping window
        """
        self.window_type = window_type
        self.window_size = window_size
        
    def compute_dtw(self, query: np.ndarray, template: np.ndarray,
                    return_path: bool = True) -> Tuple[float, Optional[List[Tuple[int, int]]]]:
        """
        Compute DTW distance between query and template
        
        Args:
            query: Query pattern
            template: Template pattern
            return_path: Whether to return alignment path
            
        Returns:
            DTW distance and optional alignment path
        """
        n, m = len(query), len(template)
        
        # Initialize cost matrix
        cost_matrix = np.full((n + 1, m + 1), np.inf)
        cost_matrix[0, 0] = 0
        
        # Apply warping window constraints
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if self._is_valid_window(i, j, n, m):
                    cost = abs(query[i-1] - template[j-1])
                    cost_matrix[i, j] = cost + min(
                        cost_matrix[i-1, j],    # insertion
                        cost_matrix[i, j-1],    # deletion
                        cost_matrix[i-1, j-1]   # match
                    )
        
        distance = cost_matrix[n, m]
        
        if return_path:
            path = self._backtrack_path(cost_matrix, n, m)
            return distance, path
        
        return distance, None
    
    def _is_valid_window(self, i: int, j: int, n: int, m: int) -> bool:
        """Check if position is within warping window"""
        if self.window_type == 'none':
            return True
        elif self.window_type == 'sakoe_chiba':
            return abs(i - j * n / m) <= self.window_size
        elif self.window_type == 'itakura':
            # Itakura parallelogram
            return (j >= 2 * i - self.window_size and 
                    j <= 2 * i + self.window_size and
                    i >= (j - self.window_size) / 2 and 
                    i <= (j + self.window_size) / 2)
        return True
    
    def _backtrack_path(self, cost_matrix: np.ndarray, n: int, m: int) -> List[Tuple[int, int]]:
        """Backtrack optimal alignment path"""
        path = []
        i, j = n, m
        
        while i > 0 and j > 0:
            path.append((i-1, j-1))
            
            # Find minimum cost predecessor
            costs = [
                cost_matrix[i-1, j-1],  # diagonal
                cost_matrix[i-1, j],    # up
                cost_matrix[i, j-1]     # left
            ]
            min_idx = np.argmin(costs)
            
            if min_idx == 0:
                i, j = i-1, j-1
            elif min_idx == 1:
                i = i-1
            else:
                j = j-1
                
        path.reverse()
        return path


class TemplateMatcher:
    """Template-based pattern matching"""
    
    def __init__(self):
        self.templates: Dict[str, Dict[str, Any]] = {}
        self.template_features: Dict[str, np.ndarray] = {}
        
    def add_template(self, template_id: str, pattern: np.ndarray,
                     outcomes: Dict[str, Any], metadata: Dict[str, Any] = None):
        """
        Add a template pattern
        
        Args:
            template_id: Unique template identifier
            pattern: Template pattern data
            outcomes: Historical outcomes for this pattern
            metadata: Additional template metadata
        """
        self.templates[template_id] = {
            'pattern': pattern,
            'outcomes': outcomes,
            'metadata': metadata or {},
            'features': self._extract_features(pattern)
        }
        self.template_features[template_id] = self.templates[template_id]['features']
    
    def _extract_features(self, pattern: np.ndarray) -> np.ndarray:
        """Extract features from pattern for fast matching"""
        features = []
        
        # Statistical features
        features.extend([
            np.mean(pattern),
            np.std(pattern),
            np.min(pattern),
            np.max(pattern),
            np.percentile(pattern, 25),
            np.percentile(pattern, 75)
        ])
        
        # Shape features
        diff = np.diff(pattern)
        features.extend([
            np.sum(diff > 0) / len(diff),  # Proportion of increases
            np.sum(np.abs(diff)),          # Total variation
            len(self._find_peaks(pattern)), # Number of peaks
            len(self._find_valleys(pattern)) # Number of valleys
        ])
        
        return np.array(features)
    
    def _find_peaks(self, pattern: np.ndarray, prominence: float = 0.1) -> List[int]:
        """Find peaks in pattern"""
        peaks = []
        for i in range(1, len(pattern) - 1):
            if pattern[i] > pattern[i-1] and pattern[i] > pattern[i+1]:
                if pattern[i] - min(pattern[i-1], pattern[i+1]) > prominence:
                    peaks.append(i)
        return peaks
    
    def _find_valleys(self, pattern: np.ndarray, prominence: float = 0.1) -> List[int]:
        """Find valleys in pattern"""
        valleys = []
        for i in range(1, len(pattern) - 1):
            if pattern[i] < pattern[i-1] and pattern[i] < pattern[i+1]:
                if max(pattern[i-1], pattern[i+1]) - pattern[i] > prominence:
                    valleys.append(i)
        return valleys
    
    def find_similar_templates(self, query_features: np.ndarray,
                              top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find most similar templates based on features
        
        Args:
            query_features: Features of query pattern
            top_k: Number of top matches to return
            
        Returns:
            List of (template_id, similarity) tuples
        """
        similarities = []
        
        for template_id, features in self.template_features.items():
            # Cosine similarity
            similarity = np.dot(query_features, features) / (
                np.linalg.norm(query_features) * np.linalg.norm(features) + 1e-8
            )
            similarities.append((template_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class PatternMatcher:
    """Main pattern matching system"""
    
    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize pattern matcher
        
        Args:
            template_dir: Directory containing pattern templates
        """
        self.dtw_matcher = DTWMatcher()
        self.template_matcher = TemplateMatcher()
        self.template_dir = template_dir or Path('data/pattern_templates')
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing templates
        self.load_templates()
        
    def load_templates(self):
        """Load pattern templates from disk"""
        template_file = self.template_dir / 'templates.pkl'
        if template_file.exists():
            with open(template_file, 'rb') as f:
                templates = pickle.load(f)
                for template_id, template_data in templates.items():
                    self.template_matcher.add_template(
                        template_id,
                        template_data['pattern'],
                        template_data['outcomes'],
                        template_data.get('metadata', {})
                    )
    
    def save_templates(self):
        """Save pattern templates to disk"""
        template_file = self.template_dir / 'templates.pkl'
        with open(template_file, 'wb') as f:
            pickle.dump(self.template_matcher.templates, f)
    
    def add_template(self, template_id: str, pattern: np.ndarray,
                     outcomes: Dict[str, Any], metadata: Dict[str, Any] = None):
        """Add a new pattern template"""
        self.template_matcher.add_template(template_id, pattern, outcomes, metadata)
        self.save_templates()
    
    def match_pattern(self, query: np.ndarray, top_k: int = 5,
                     min_similarity: float = 0.7,
                     use_parallel: bool = True) -> List[PatternMatch]:
        """
        Find best matching patterns for query
        
        Args:
            query: Query pattern to match
            top_k: Number of top matches to return
            min_similarity: Minimum similarity threshold
            use_parallel: Whether to use parallel processing
            
        Returns:
            List of pattern matches
        """
        start_time = time.time()
        
        # Normalize query
        query_norm = self._normalize_pattern(query)
        query_features = self.template_matcher._extract_features(query_norm)
        
        # Find candidate templates using feature similarity
        candidates = self.template_matcher.find_similar_templates(
            query_features, top_k=min(top_k * 3, len(self.template_matcher.templates))
        )
        
        # Compute detailed matches
        if use_parallel and len(candidates) > 10:
            matches = self._parallel_match(query_norm, candidates, min_similarity)
        else:
            matches = self._sequential_match(query_norm, candidates, min_similarity)
        
        # Sort by similarity score
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Add timing metadata
        for match in matches[:top_k]:
            match.metadata['matching_time'] = time.time() - start_time
            
        return matches[:top_k]
    
    def _normalize_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Normalize pattern to zero mean and unit variance"""
        pattern = np.array(pattern)
        mean = np.mean(pattern)
        std = np.std(pattern)
        if std > 0:
            return (pattern - mean) / std
        return pattern - mean
    
    def _sequential_match(self, query: np.ndarray, candidates: List[Tuple[str, float]],
                         min_similarity: float) -> List[PatternMatch]:
        """Sequential pattern matching"""
        matches = []
        
        for template_id, feature_similarity in candidates:
            template_data = self.template_matcher.templates[template_id]
            template_pattern = self._normalize_pattern(template_data['pattern'])
            
            # Compute DTW distance
            dtw_distance, alignment_path = self.dtw_matcher.compute_dtw(
                query, template_pattern, return_path=True
            )
            
            # Compute correlation
            if len(query) == len(template_pattern):
                correlation, _ = pearsonr(query, template_pattern)
            else:
                # Use aligned sequences for correlation
                query_aligned = [query[i] for i, j in alignment_path]
                template_aligned = [template_pattern[j] for i, j in alignment_path]
                correlation, _ = pearsonr(query_aligned, template_aligned)
            
            # Compute similarity score
            similarity_score = self._compute_similarity_score(
                dtw_distance, correlation, feature_similarity,
                len(query), len(template_pattern)
            )
            
            if similarity_score >= min_similarity:
                match = PatternMatch(
                    template_id=template_id,
                    similarity_score=similarity_score,
                    dtw_distance=dtw_distance,
                    correlation=correlation,
                    start_idx=0,
                    end_idx=len(query) - 1,
                    template_length=len(template_pattern),
                    query_length=len(query),
                    alignment_path=alignment_path,
                    historical_outcomes=template_data['outcomes'],
                    metadata=template_data.get('metadata', {}).copy()
                )
                matches.append(match)
        
        return matches
    
    def _parallel_match(self, query: np.ndarray, candidates: List[Tuple[str, float]],
                       min_similarity: float) -> List[PatternMatch]:
        """Parallel pattern matching"""
        matches = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_template = {
                executor.submit(
                    self._match_single_template,
                    query, template_id, feature_similarity
                ): (template_id, feature_similarity)
                for template_id, feature_similarity in candidates
            }
            
            for future in as_completed(future_to_template):
                match = future.result()
                if match and match.similarity_score >= min_similarity:
                    matches.append(match)
        
        return matches
    
    def _match_single_template(self, query: np.ndarray, template_id: str,
                              feature_similarity: float) -> Optional[PatternMatch]:
        """Match query against single template"""
        try:
            template_data = self.template_matcher.templates[template_id]
            template_pattern = self._normalize_pattern(template_data['pattern'])
            
            # Compute DTW distance
            dtw_distance, alignment_path = self.dtw_matcher.compute_dtw(
                query, template_pattern, return_path=True
            )
            
            # Compute correlation
            if len(query) == len(template_pattern):
                correlation, _ = pearsonr(query, template_pattern)
            else:
                query_aligned = [query[i] for i, j in alignment_path]
                template_aligned = [template_pattern[j] for i, j in alignment_path]
                correlation, _ = pearsonr(query_aligned, template_aligned)
            
            # Compute similarity score
            similarity_score = self._compute_similarity_score(
                dtw_distance, correlation, feature_similarity,
                len(query), len(template_pattern)
            )
            
            return PatternMatch(
                template_id=template_id,
                similarity_score=similarity_score,
                dtw_distance=dtw_distance,
                correlation=correlation,
                start_idx=0,
                end_idx=len(query) - 1,
                template_length=len(template_pattern),
                query_length=len(query),
                alignment_path=alignment_path,
                historical_outcomes=template_data['outcomes'],
                metadata=template_data.get('metadata', {}).copy()
            )
        except Exception as e:
            print(f"Error matching template {template_id}: {e}")
            return None
    
    def _compute_similarity_score(self, dtw_distance: float, correlation: float,
                                 feature_similarity: float, query_len: int,
                                 template_len: int) -> float:
        """
        Compute overall similarity score
        
        Args:
            dtw_distance: DTW distance
            correlation: Pearson correlation
            feature_similarity: Feature-based similarity
            query_len: Length of query pattern
            template_len: Length of template pattern
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize DTW distance
        max_len = max(query_len, template_len)
        dtw_normalized = 1.0 - min(dtw_distance / max_len, 1.0)
        
        # Handle correlation (can be negative)
        correlation_score = (correlation + 1) / 2
        
        # Length similarity
        length_similarity = 1.0 - abs(query_len - template_len) / max_len
        
        # Weighted combination
        weights = {
            'dtw': 0.4,
            'correlation': 0.3,
            'features': 0.2,
            'length': 0.1
        }
        
        similarity = (
            weights['dtw'] * dtw_normalized +
            weights['correlation'] * correlation_score +
            weights['features'] * feature_similarity +
            weights['length'] * length_similarity
        )
        
        return np.clip(similarity, 0, 1)
    
    def get_forecast_ranges(self, matches: List[PatternMatch],
                           confidence_levels: List[float] = [0.68, 0.95]) -> Dict[str, Any]:
        """
        Get forecast ranges based on matched patterns
        
        Args:
            matches: List of pattern matches
            confidence_levels: Confidence levels for prediction intervals
            
        Returns:
            Dictionary with forecast statistics and ranges
        """
        if not matches:
            return {
                'mean_forecast': None,
                'confidence_intervals': {},
                'pattern_outcomes': []
            }
        
        # Collect outcomes from matched patterns
        all_outcomes = []
        weights = []
        
        for match in matches:
            outcomes = match.historical_outcomes
            if 'returns' in outcomes:
                all_outcomes.extend(outcomes['returns'])
                weights.extend([match.similarity_score] * len(outcomes['returns']))
        
        if not all_outcomes:
            return {
                'mean_forecast': None,
                'confidence_intervals': {},
                'pattern_outcomes': []
            }
        
        all_outcomes = np.array(all_outcomes)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Compute weighted statistics
        mean_forecast = np.average(all_outcomes, weights=weights)
        
        # Compute confidence intervals
        confidence_intervals = {}
        for level in confidence_levels:
            lower_percentile = (1 - level) / 2 * 100
            upper_percentile = (1 + level) / 2 * 100
            
            # Weighted percentiles
            sorted_idx = np.argsort(all_outcomes)
            sorted_outcomes = all_outcomes[sorted_idx]
            sorted_weights = weights[sorted_idx]
            cumsum_weights = np.cumsum(sorted_weights)
            
            lower_idx = np.searchsorted(cumsum_weights, lower_percentile / 100)
            upper_idx = np.searchsorted(cumsum_weights, upper_percentile / 100)
            
            confidence_intervals[f'{int(level*100)}%'] = {
                'lower': sorted_outcomes[lower_idx],
                'upper': sorted_outcomes[upper_idx]
            }
        
        # Pattern-specific outcomes
        pattern_outcomes = []
        for match in matches:
            outcomes = match.historical_outcomes
            if 'returns' in outcomes:
                pattern_outcomes.append({
                    'template_id': match.template_id,
                    'similarity': match.similarity_score,
                    'mean_return': np.mean(outcomes['returns']),
                    'std_return': np.std(outcomes['returns']),
                    'num_instances': len(outcomes['returns'])
                })
        
        return {
            'mean_forecast': mean_forecast,
            'confidence_intervals': confidence_intervals,
            'pattern_outcomes': pattern_outcomes,
            'total_instances': len(all_outcomes)
        }
    




if __name__ == "__main__":
    # Example usage
    matcher = PatternMatcher()
    
    # Pattern matcher now works with real market data
    # Load templates from saved patterns or add new ones
    print("Pattern Matcher initialized. Ready to match real market patterns.")
    print(f"Loaded {len(matcher.template_matcher.templates)} templates")
