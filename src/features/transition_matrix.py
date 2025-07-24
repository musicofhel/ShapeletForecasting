"""
Pattern Transition Matrix Builder

This module builds transition matrices from pattern sequences to capture
temporal dependencies and pattern evolution in financial time series.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import sparse
from sklearn.preprocessing import normalize
import logging
from pathlib import Path
import pickle

# Import from previous sprints
import sys
sys.path.append(str(Path(__file__).parent.parent))
from dtw import DTWCalculator

logger = logging.getLogger(__name__)


class TransitionMatrixBuilder:
    """
    Builds transition matrices from pattern sequences.
    
    Features include:
    - Pattern-to-pattern transition probabilities
    - Multi-step transition matrices
    - Conditional transition matrices based on market conditions
    - Transition entropy and stability metrics
    """
    
    def __init__(self,
                 n_patterns: int,
                 pattern_length: int = 30,
                 max_order: int = 3,
                 smoothing: float = 0.01):
        """
        Initialize transition matrix builder.
        
        Parameters:
        -----------
        n_patterns : int
            Number of distinct patterns
        pattern_length : int
            Length of each pattern
        max_order : int
            Maximum order of transition matrices to compute
        smoothing : float
            Laplace smoothing parameter
        """
        self.n_patterns = n_patterns
        self.pattern_length = pattern_length
        self.max_order = max_order
        self.smoothing = smoothing
        
        # Storage for transition matrices
        self.transition_matrices = {}
        self.conditional_matrices = {}
        
        # DTW calculator for pattern matching
        self.dtw_calculator = DTWCalculator()
        
    def fit(self, sequences: List[List[int]], 
            conditions: Optional[List[List[str]]] = None) -> 'TransitionMatrixBuilder':
        """
        Fit transition matrices from pattern sequences.
        
        Parameters:
        -----------
        sequences : List[List[int]]
            List of pattern sequences (pattern indices)
        conditions : List[List[str]], optional
            Market conditions for each pattern in sequences
            
        Returns:
        --------
        self : TransitionMatrixBuilder
        """
        logger.info("Building transition matrices...")
        
        # Build standard transition matrices
        for order in range(1, self.max_order + 1):
            self.transition_matrices[order] = self._build_transition_matrix(
                sequences, order
            )
            
        # Build conditional transition matrices if conditions provided
        if conditions is not None:
            unique_conditions = set()
            for cond_seq in conditions:
                unique_conditions.update(cond_seq)
                
            for condition in unique_conditions:
                self.conditional_matrices[condition] = {}
                # Filter sequences by condition
                filtered_sequences = self._filter_sequences_by_condition(
                    sequences, conditions, condition
                )
                
                for order in range(1, min(self.max_order + 1, 3)):  # Limit order for conditional
                    self.conditional_matrices[condition][order] = \
                        self._build_transition_matrix(filtered_sequences, order)
                        
        logger.info(f"Built transition matrices up to order {self.max_order}")
        
        return self
        
    def transform(self, sequences: List[List[int]], 
                 conditions: Optional[List[List[str]]] = None) -> pd.DataFrame:
        """
        Transform pattern sequences into transition-based features.
        
        Parameters:
        -----------
        sequences : List[List[int]]
            Pattern sequences to transform
        conditions : List[List[str]], optional
            Market conditions for conditional features
            
        Returns:
        --------
        features : pd.DataFrame
            Transition-based features
        """
        if not self.transition_matrices:
            raise ValueError("TransitionMatrixBuilder must be fitted before transform")
            
        features_list = []
        
        for i, sequence in enumerate(sequences):
            features = {}
            
            # Extract transition features
            trans_features = self._extract_transition_features(sequence)
            features.update(trans_features)
            
            # Extract entropy features
            entropy_features = self._extract_entropy_features(sequence)
            features.update(entropy_features)
            
            # Extract stability features
            stability_features = self._extract_stability_features(sequence)
            features.update(stability_features)
            
            # Extract conditional features if conditions provided
            if conditions is not None and i < len(conditions):
                cond_features = self._extract_conditional_features(
                    sequence, conditions[i]
                )
                features.update(cond_features)
                
            features_list.append(features)
            
        return pd.DataFrame(features_list)
        
    def _build_transition_matrix(self, sequences: List[List[int]], 
                                order: int) -> np.ndarray:
        """Build transition matrix of given order."""
        if order == 1:
            # First-order transitions
            matrix = np.zeros((self.n_patterns, self.n_patterns)) + self.smoothing
            
            for sequence in sequences:
                for i in range(len(sequence) - 1):
                    matrix[sequence[i], sequence[i + 1]] += 1
                    
        else:
            # Higher-order transitions
            # Create state space for n-grams
            state_size = self.n_patterns ** order
            matrix = np.zeros((state_size, self.n_patterns)) + self.smoothing
            
            for sequence in sequences:
                for i in range(len(sequence) - order):
                    # Convert n-gram to state index
                    state_idx = 0
                    for j in range(order):
                        state_idx = state_idx * self.n_patterns + sequence[i + j]
                    
                    next_pattern = sequence[i + order]
                    matrix[state_idx, next_pattern] += 1
                    
        # Normalize rows to get probabilities
        matrix = normalize(matrix, norm='l1', axis=1)
        
        return matrix
        
    def _filter_sequences_by_condition(self, sequences: List[List[int]], 
                                     conditions: List[List[str]], 
                                     target_condition: str) -> List[List[int]]:
        """Filter sequences based on market condition."""
        filtered = []
        
        for seq, cond in zip(sequences, conditions):
            if len(seq) != len(cond):
                continue
                
            # Extract subsequences where condition matches
            i = 0
            while i < len(cond):
                if cond[i] == target_condition:
                    # Find continuous segment with target condition
                    j = i
                    while j < len(cond) and cond[j] == target_condition:
                        j += 1
                    
                    if j - i >= 2:  # Need at least 2 patterns for transition
                        filtered.append(seq[i:j])
                    
                    i = j
                else:
                    i += 1
                    
        return filtered
        
    def _extract_transition_features(self, sequence: List[int]) -> Dict[str, float]:
        """Extract transition probability features."""
        features = {}
        
        if len(sequence) < 2:
            return features
            
        # First-order transition features
        trans_probs = []
        for i in range(len(sequence) - 1):
            prob = self.transition_matrices[1][sequence[i], sequence[i + 1]]
            trans_probs.append(prob)
            
        if trans_probs:
            features['trans_prob_mean'] = np.mean(trans_probs)
            features['trans_prob_std'] = np.std(trans_probs)
            features['trans_prob_min'] = np.min(trans_probs)
            features['trans_prob_max'] = np.max(trans_probs)
            
        # Higher-order transition features
        for order in range(2, min(self.max_order + 1, len(sequence))):
            order_probs = []
            
            for i in range(len(sequence) - order):
                # Convert n-gram to state index
                state_idx = 0
                for j in range(order):
                    state_idx = state_idx * self.n_patterns + sequence[i + j]
                    
                if state_idx < len(self.transition_matrices[order]):
                    prob = self.transition_matrices[order][state_idx, sequence[i + order]]
                    order_probs.append(prob)
                    
            if order_probs:
                features[f'trans_prob_order{order}_mean'] = np.mean(order_probs)
                features[f'trans_prob_order{order}_std'] = np.std(order_probs)
                
        # Transition likelihood score
        if len(sequence) >= 2:
            likelihood = 1.0
            for i in range(len(sequence) - 1):
                likelihood *= self.transition_matrices[1][sequence[i], sequence[i + 1]]
            features['sequence_likelihood'] = likelihood ** (1 / (len(sequence) - 1))
            
        return features
        
    def _extract_entropy_features(self, sequence: List[int]) -> Dict[str, float]:
        """Extract entropy-based features from transitions."""
        features = {}
        
        if len(sequence) < 2:
            return features
            
        # Pattern distribution entropy
        unique, counts = np.unique(sequence, return_counts=True)
        probs = counts / len(sequence)
        features['pattern_entropy'] = -np.sum(probs * np.log(probs + 1e-10))
        
        # Transition entropy from current states
        trans_entropies = []
        for pattern in unique:
            # Get transition probabilities from this pattern
            trans_probs = self.transition_matrices[1][pattern]
            # Calculate entropy
            entropy = -np.sum(trans_probs * np.log(trans_probs + 1e-10))
            trans_entropies.append(entropy)
            
        features['mean_transition_entropy'] = np.mean(trans_entropies)
        features['max_transition_entropy'] = np.max(trans_entropies)
        
        # Conditional entropy
        if len(sequence) >= 3:
            cond_entropy = 0
            for i in range(len(sequence) - 2):
                # P(next | current, previous)
                state_idx = sequence[i] * self.n_patterns + sequence[i + 1]
                if state_idx < len(self.transition_matrices[2]):
                    trans_probs = self.transition_matrices[2][state_idx]
                    entropy = -np.sum(trans_probs * np.log(trans_probs + 1e-10))
                    cond_entropy += entropy
                    
            features['conditional_entropy'] = cond_entropy / max(len(sequence) - 2, 1)
            
        return features
        
    def _extract_stability_features(self, sequence: List[int]) -> Dict[str, float]:
        """Extract pattern stability features."""
        features = {}
        
        if len(sequence) < 2:
            return features
            
        # Self-transition probability
        self_trans = []
        for pattern in range(self.n_patterns):
            self_trans.append(self.transition_matrices[1][pattern, pattern])
            
        # Stability of patterns in sequence
        pattern_stability = []
        for pattern in sequence:
            pattern_stability.append(self_trans[pattern])
            
        features['mean_pattern_stability'] = np.mean(pattern_stability)
        features['min_pattern_stability'] = np.min(pattern_stability)
        
        # Sequence stability (consecutive same patterns)
        same_count = 0
        max_same = 0
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i-1]:
                same_count += 1
                max_same = max(max_same, same_count)
            else:
                same_count = 0
                
        features['max_consecutive_patterns'] = max_same + 1
        features['pattern_change_rate'] = sum(sequence[i] != sequence[i-1] 
                                            for i in range(1, len(sequence))) / (len(sequence) - 1)
        
        # Cycle detection
        cycles = self._detect_cycles(sequence)
        features['num_cycles'] = len(cycles)
        if cycles:
            features['avg_cycle_length'] = np.mean([len(c) for c in cycles])
            features['max_cycle_length'] = max(len(c) for c in cycles)
        else:
            features['avg_cycle_length'] = 0
            features['max_cycle_length'] = 0
            
        return features
        
    def _extract_conditional_features(self, sequence: List[int], 
                                    conditions: List[str]) -> Dict[str, float]:
        """Extract features based on conditional transitions."""
        features = {}
        
        if len(sequence) < 2 or len(conditions) != len(sequence):
            return features
            
        # Transition probabilities under different conditions
        for condition in self.conditional_matrices:
            if condition not in self.conditional_matrices:
                continue
                
            cond_probs = []
            for i in range(len(sequence) - 1):
                if conditions[i] == condition:
                    prob = self.conditional_matrices[condition][1][
                        sequence[i], sequence[i + 1]
                    ]
                    cond_probs.append(prob)
                    
            if cond_probs:
                features[f'trans_prob_{condition}_mean'] = np.mean(cond_probs)
                features[f'trans_prob_{condition}_std'] = np.std(cond_probs)
                
        # Condition change impact on patterns
        condition_changes = sum(conditions[i] != conditions[i-1] 
                              for i in range(1, len(conditions)))
        pattern_changes = sum(sequence[i] != sequence[i-1] 
                            for i in range(1, len(sequence)))
        
        features['condition_change_rate'] = condition_changes / (len(conditions) - 1)
        features['pattern_condition_correlation'] = (
            pattern_changes / max(condition_changes, 1) if condition_changes > 0 else 0
        )
        
        return features
        
    def _detect_cycles(self, sequence: List[int], min_length: int = 2, 
                      max_length: int = 10) -> List[List[int]]:
        """Detect repeating cycles in pattern sequence."""
        cycles = []
        
        for length in range(min_length, min(max_length + 1, len(sequence) // 2)):
            for start in range(len(sequence) - 2 * length + 1):
                pattern = sequence[start:start + length]
                
                # Check if pattern repeats
                if sequence[start + length:start + 2 * length] == pattern:
                    cycles.append(pattern)
                    
        # Remove duplicate cycles
        unique_cycles = []
        for cycle in cycles:
            if cycle not in unique_cycles:
                unique_cycles.append(cycle)
                
        return unique_cycles
        
    def get_transition_matrix(self, order: int = 1) -> np.ndarray:
        """Get transition matrix of specified order."""
        if order not in self.transition_matrices:
            raise ValueError(f"No transition matrix of order {order} available")
        return self.transition_matrices[order].copy()
        
    def get_stationary_distribution(self, order: int = 1) -> np.ndarray:
        """Calculate stationary distribution of patterns."""
        if order != 1:
            raise NotImplementedError("Stationary distribution only implemented for order 1")
            
        trans_matrix = self.transition_matrices[1]
        
        # Find eigenvector with eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(trans_matrix.T)
        stationary_idx = np.argmin(np.abs(eigenvalues - 1))
        stationary = np.real(eigenvectors[:, stationary_idx])
        stationary = stationary / stationary.sum()
        
        return stationary
        
    def save(self, filepath: str):
        """Save transition matrices to file."""
        data = {
            'transition_matrices': self.transition_matrices,
            'conditional_matrices': self.conditional_matrices,
            'config': {
                'n_patterns': self.n_patterns,
                'pattern_length': self.pattern_length,
                'max_order': self.max_order,
                'smoothing': self.smoothing
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        logger.info(f"Saved transition matrices to {filepath}")
        
    def load(self, filepath: str):
        """Load transition matrices from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.transition_matrices = data['transition_matrices']
        self.conditional_matrices = data['conditional_matrices']
        
        config = data['config']
        self.n_patterns = config['n_patterns']
        self.pattern_length = config['pattern_length']
        self.max_order = config['max_order']
        self.smoothing = config['smoothing']
        
        logger.info(f"Loaded transition matrices from {filepath}")
        
    def visualize_transition_matrix(self, order: int = 1, 
                                  condition: Optional[str] = None) -> np.ndarray:
        """Get transition matrix for visualization."""
        if condition is not None:
            if condition in self.conditional_matrices:
                return self.conditional_matrices[condition][order]
            else:
                raise ValueError(f"No conditional matrix for condition '{condition}'")
        else:
            return self.get_transition_matrix(order)
