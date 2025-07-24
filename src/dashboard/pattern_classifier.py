"""
Pattern Type Classification System

This module provides functionality for identifying and classifying specific pattern types
including traditional technical patterns, shapelets, motifs, and fractals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.signal import find_peaks
from scipy.stats import pearsonr
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class PatternDefinition:
    """Definition of a pattern type"""
    name: str
    category: str  # 'traditional', 'shapelet', 'fractal', 'custom'
    description: str
    key_points: List[Dict[str, float]]  # Relative positions of key points
    validation_rules: Dict[str, Any]
    confidence_threshold: float = 0.7
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'category': self.category,
            'description': self.description,
            'key_points': self.key_points,
            'validation_rules': self.validation_rules,
            'confidence_threshold': self.confidence_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PatternDefinition':
        """Create from dictionary"""
        return cls(**data)


class PatternClassifier:
    """
    Classifies wavelet patterns into specific types like Head & Shoulders,
    Triangles, Flags, and custom shapelets/motifs.
    """
    
    def __init__(self, pattern_library_path: Optional[Path] = None):
        """
        Initialize pattern classifier
        
        Args:
            pattern_library_path: Path to pattern library JSON file
        """
        self.pattern_library_path = pattern_library_path or Path("data/pattern_library.json")
        self.pattern_definitions = self._load_pattern_library()
        self._initialize_default_patterns()
        
    def _load_pattern_library(self) -> Dict[str, PatternDefinition]:
        """Load pattern definitions from library"""
        if self.pattern_library_path.exists():
            try:
                with open(self.pattern_library_path, 'r') as f:
                    data = json.load(f)
                return {
                    name: PatternDefinition.from_dict(defn)
                    for name, defn in data.items()
                }
            except Exception as e:
                logger.warning(f"Failed to load pattern library: {e}")
        return {}
    
    def _initialize_default_patterns(self):
        """Initialize default technical analysis patterns"""
        default_patterns = {
            'head_and_shoulders': PatternDefinition(
                name='Head and Shoulders',
                category='traditional',
                description='Reversal pattern with three peaks, middle highest',
                key_points=[
                    {'position': 0.0, 'type': 'start'},
                    {'position': 0.2, 'type': 'left_shoulder_peak'},
                    {'position': 0.3, 'type': 'left_valley'},
                    {'position': 0.5, 'type': 'head_peak'},
                    {'position': 0.7, 'type': 'right_valley'},
                    {'position': 0.8, 'type': 'right_shoulder_peak'},
                    {'position': 1.0, 'type': 'end'}
                ],
                validation_rules={
                    'head_higher_than_shoulders': True,
                    'shoulders_similar_height': 0.1,  # 10% tolerance
                    'valleys_similar_height': 0.1,
                    'minimum_duration': 20  # periods
                }
            ),
            
            'ascending_triangle': PatternDefinition(
                name='Ascending Triangle',
                category='traditional',
                description='Continuation pattern with flat top and rising bottom',
                key_points=[
                    {'position': 0.0, 'type': 'start'},
                    {'position': 0.25, 'type': 'resistance_1'},
                    {'position': 0.5, 'type': 'resistance_2'},
                    {'position': 0.75, 'type': 'resistance_3'},
                    {'position': 1.0, 'type': 'breakout'}
                ],
                validation_rules={
                    'resistance_level_flat': 0.02,  # 2% tolerance
                    'support_trend_rising': True,
                    'minimum_touches': 2,
                    'volume_pattern': 'decreasing'
                }
            ),
            
            'bull_flag': PatternDefinition(
                name='Bull Flag',
                category='traditional',
                description='Continuation pattern with sharp rise followed by consolidation',
                key_points=[
                    {'position': 0.0, 'type': 'flagpole_start'},
                    {'position': 0.3, 'type': 'flagpole_end'},
                    {'position': 0.65, 'type': 'flag_middle'},
                    {'position': 1.0, 'type': 'breakout'}
                ],
                validation_rules={
                    'flagpole_angle': {'min': 45, 'max': 90},  # degrees
                    'flag_angle': {'min': -30, 'max': 0},
                    'flag_duration_ratio': {'min': 0.3, 'max': 0.7},
                    'volume_spike_on_breakout': True
                }
            ),
            
            'double_bottom': PatternDefinition(
                name='Double Bottom',
                category='traditional',
                description='Reversal pattern with two similar lows',
                key_points=[
                    {'position': 0.0, 'type': 'start'},
                    {'position': 0.25, 'type': 'first_bottom'},
                    {'position': 0.5, 'type': 'middle_peak'},
                    {'position': 0.75, 'type': 'second_bottom'},
                    {'position': 1.0, 'type': 'breakout'}
                ],
                validation_rules={
                    'bottoms_similar': 0.02,  # 2% tolerance
                    'middle_peak_higher': True,
                    'minimum_separation': 10,  # periods
                    'volume_pattern': 'higher_on_second'
                }
            ),
            
            'cup_and_handle': PatternDefinition(
                name='Cup and Handle',
                category='traditional',
                description='Continuation pattern resembling a tea cup',
                key_points=[
                    {'position': 0.0, 'type': 'cup_start'},
                    {'position': 0.5, 'type': 'cup_bottom'},
                    {'position': 0.8, 'type': 'cup_end'},
                    {'position': 0.9, 'type': 'handle_bottom'},
                    {'position': 1.0, 'type': 'breakout'}
                ],
                validation_rules={
                    'cup_shape': 'u_shaped',
                    'handle_depth': {'max': 0.5},  # Max 50% of cup depth
                    'handle_duration': {'max': 0.3},  # Max 30% of pattern
                    'volume_pattern': 'decreasing_in_handle'
                }
            )
        }
        
        # Add default patterns that don't exist in library
        for name, pattern in default_patterns.items():
            if name not in self.pattern_definitions:
                self.pattern_definitions[name] = pattern
    
    def classify_pattern(self, 
                        pattern_data: np.ndarray,
                        timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Classify a pattern into known types
        
        Args:
            pattern_data: Pattern values
            timestamps: Optional timestamps
            
        Returns:
            Classification results with confidence scores
        """
        results = {
            'best_match': None,
            'confidence': 0.0,
            'all_matches': [],
            'features': self._extract_pattern_features(pattern_data)
        }
        
        # Test against each pattern definition
        for name, definition in self.pattern_definitions.items():
            confidence = self._match_pattern(pattern_data, definition)
            
            if confidence >= definition.confidence_threshold:
                match_info = {
                    'name': definition.name,
                    'category': definition.category,
                    'confidence': confidence,
                    'description': definition.description
                }
                results['all_matches'].append(match_info)
                
                if confidence > results['confidence']:
                    results['best_match'] = match_info
                    results['confidence'] = confidence
        
        # If no traditional pattern matched, try shapelet/motif detection
        if not results['best_match']:
            shapelet_result = self._detect_shapelet(pattern_data)
            if shapelet_result:
                results['best_match'] = shapelet_result
                results['confidence'] = shapelet_result['confidence']
                results['all_matches'].append(shapelet_result)
        
        return results
    
    def _extract_pattern_features(self, pattern_data: np.ndarray) -> Dict[str, float]:
        """Extract key features from pattern"""
        features = {}
        
        # Basic statistics
        features['mean'] = float(np.mean(pattern_data))
        features['std'] = float(np.std(pattern_data))
        features['skew'] = float(self._calculate_skew(pattern_data))
        features['kurtosis'] = float(self._calculate_kurtosis(pattern_data))
        
        # Shape features
        peaks, _ = find_peaks(pattern_data)
        valleys, _ = find_peaks(-pattern_data)
        
        features['num_peaks'] = len(peaks)
        features['num_valleys'] = len(valleys)
        features['peak_valley_ratio'] = len(peaks) / (len(valleys) + 1)
        
        # Trend features
        features['overall_trend'] = float(np.polyfit(range(len(pattern_data)), pattern_data, 1)[0])
        features['volatility'] = float(np.std(np.diff(pattern_data)))
        
        # Energy features
        features['energy'] = float(np.sum(pattern_data ** 2))
        features['entropy'] = float(self._calculate_entropy(pattern_data))
        
        return features
    
    def _match_pattern(self, data: np.ndarray, definition: PatternDefinition) -> float:
        """
        Match data against a pattern definition
        
        Returns:
            Confidence score between 0 and 1
        """
        confidence_scores = []
        
        # Extract key points from data
        key_points = self._extract_key_points(data, definition.key_points)
        
        # Validate against rules
        for rule_name, rule_value in definition.validation_rules.items():
            score = self._validate_rule(key_points, rule_name, rule_value, data)
            confidence_scores.append(score)
        
        # Calculate overall confidence
        if confidence_scores:
            return np.mean(confidence_scores)
        return 0.0
    
    def _extract_key_points(self, data: np.ndarray, 
                           point_definitions: List[Dict]) -> Dict[str, float]:
        """Extract key points from data based on definitions"""
        key_points = {}
        data_len = len(data)
        
        for point_def in point_definitions:
            position = int(point_def['position'] * (data_len - 1))
            point_type = point_def['type']
            
            # Handle different point types
            if 'peak' in point_type:
                # Find nearest peak
                peaks, _ = find_peaks(data)
                if peaks.size > 0:
                    nearest_peak = peaks[np.argmin(np.abs(peaks - position))]
                    key_points[point_type] = data[nearest_peak]
                else:
                    key_points[point_type] = data[position]
            elif 'valley' in point_type or 'bottom' in point_type:
                # Find nearest valley
                valleys, _ = find_peaks(-data)
                if valleys.size > 0:
                    nearest_valley = valleys[np.argmin(np.abs(valleys - position))]
                    key_points[point_type] = data[nearest_valley]
                else:
                    key_points[point_type] = data[position]
            else:
                key_points[point_type] = data[position]
        
        return key_points
    
    def _validate_rule(self, key_points: Dict[str, float], 
                      rule_name: str, rule_value: Any, data: np.ndarray) -> float:
        """Validate a specific rule and return confidence score"""
        
        if rule_name == 'head_higher_than_shoulders':
            if all(k in key_points for k in ['head_peak', 'left_shoulder_peak', 'right_shoulder_peak']):
                head = key_points['head_peak']
                left_shoulder = key_points['left_shoulder_peak']
                right_shoulder = key_points['right_shoulder_peak']
                
                if head > left_shoulder and head > right_shoulder:
                    # Calculate how much higher
                    ratio = min(head / left_shoulder, head / right_shoulder)
                    return min(1.0, ratio - 1.0)  # More height difference = higher confidence
            return 0.0
            
        elif rule_name == 'shoulders_similar_height':
            if all(k in key_points for k in ['left_shoulder_peak', 'right_shoulder_peak']):
                left = key_points['left_shoulder_peak']
                right = key_points['right_shoulder_peak']
                diff = abs(left - right) / max(left, right)
                return 1.0 - min(1.0, diff / rule_value)
            return 0.0
            
        elif rule_name == 'resistance_level_flat':
            resistance_points = [v for k, v in key_points.items() if 'resistance' in k]
            if len(resistance_points) >= 2:
                std_dev = np.std(resistance_points) / np.mean(resistance_points)
                return 1.0 - min(1.0, std_dev / rule_value)
            return 0.0
            
        elif rule_name == 'bottoms_similar':
            if all(k in key_points for k in ['first_bottom', 'second_bottom']):
                first = key_points['first_bottom']
                second = key_points['second_bottom']
                diff = abs(first - second) / min(first, second)
                return 1.0 - min(1.0, diff / rule_value)
            return 0.0
            
        # Add more rule validations as needed
        return 0.5  # Default confidence for unimplemented rules
    
    def _detect_shapelet(self, pattern_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect if pattern is a shapelet/motif
        
        Returns:
            Shapelet classification or None
        """
        # Simple shapelet detection based on pattern characteristics
        features = self._extract_pattern_features(pattern_data)
        
        # Check for fractal characteristics
        if self._has_fractal_properties(pattern_data):
            return {
                'name': f'Fractal Pattern #{self._generate_pattern_id(features)}',
                'category': 'fractal',
                'confidence': 0.8,
                'description': 'Self-similar pattern with fractal properties'
            }
        
        # Check for unique shapelet characteristics
        if self._is_unique_shapelet(features):
            return {
                'name': f'Shapelet #{self._generate_pattern_id(features)}',
                'category': 'shapelet',
                'confidence': 0.75,
                'description': 'Unique time series subsequence pattern'
            }
        
        return None
    
    def _has_fractal_properties(self, data: np.ndarray) -> bool:
        """Check if pattern has fractal/self-similar properties"""
        # Simplified fractal detection
        # Check self-similarity at different scales
        if len(data) < 20:
            return False
            
        # Compare pattern at different scales
        half_len = len(data) // 2
        first_half = data[:half_len]
        second_half = data[half_len:2*half_len]
        
        # Resample to same length
        first_resampled = np.interp(np.linspace(0, 1, 100), 
                                   np.linspace(0, 1, len(first_half)), 
                                   first_half)
        second_resampled = np.interp(np.linspace(0, 1, 100), 
                                    np.linspace(0, 1, len(second_half)), 
                                    second_half)
        
        # Calculate correlation
        corr, _ = pearsonr(first_resampled, second_resampled)
        
        return abs(corr) > 0.7
    
    def _is_unique_shapelet(self, features: Dict[str, float]) -> bool:
        """Determine if pattern is a unique shapelet"""
        # Simple heuristic: unusual combination of features
        unusual_score = 0
        
        # Check for unusual feature combinations
        if features['num_peaks'] >= 3 and features['num_valleys'] >= 3:
            unusual_score += 1
        if abs(features['skew']) > 1.5:
            unusual_score += 1
        if features['kurtosis'] > 3:
            unusual_score += 1
        if features['peak_valley_ratio'] < 0.5 or features['peak_valley_ratio'] > 2:
            unusual_score += 1
            
        return unusual_score >= 2
    
    def _generate_pattern_id(self, features: Dict[str, float]) -> str:
        """Generate unique ID for pattern based on features"""
        # Create a simple hash from key features
        feature_str = f"{features['num_peaks']}{features['num_valleys']}{features['skew']:.2f}"
        return str(abs(hash(feature_str)) % 10000)
    
    def _calculate_skew(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate entropy of data"""
        # Normalize data to probabilities
        data_positive = data - np.min(data) + 1e-10
        probs = data_positive / np.sum(data_positive)
        
        # Calculate entropy
        return -np.sum(probs * np.log(probs + 1e-10))
    
    def add_custom_pattern(self, pattern_definition: PatternDefinition):
        """Add a custom pattern definition"""
        self.pattern_definitions[pattern_definition.name.lower().replace(' ', '_')] = pattern_definition
        self._save_pattern_library()
    
    def _save_pattern_library(self):
        """Save pattern library to file"""
        try:
            self.pattern_library_path.parent.mkdir(parents=True, exist_ok=True)
            
            library_data = {
                name: defn.to_dict()
                for name, defn in self.pattern_definitions.items()
            }
            
            with open(self.pattern_library_path, 'w') as f:
                json.dump(library_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save pattern library: {e}")
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about pattern library"""
        stats = {
            'total_patterns': len(self.pattern_definitions),
            'by_category': {},
            'pattern_list': []
        }
        
        for name, defn in self.pattern_definitions.items():
            category = defn.category
            if category not in stats['by_category']:
                stats['by_category'][category] = 0
            stats['by_category'][category] += 1
            
            stats['pattern_list'].append({
                'name': defn.name,
                'category': category,
                'description': defn.description
            })
        
        return stats
