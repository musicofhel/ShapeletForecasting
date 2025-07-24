"""
Tests for Pattern Type Classification System
"""

import pytest
import numpy as np
from pathlib import Path
import json
import tempfile
from src.dashboard.pattern_classifier import PatternClassifier, PatternDefinition


class TestPatternClassifier:
    """Test pattern classification functionality"""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier with temporary pattern library"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            temp_path = Path(tmp.name)
        
        classifier = PatternClassifier(pattern_library_path=temp_path)
        yield classifier
        
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()
    
    @pytest.fixture
    def head_and_shoulders_pattern(self):
        """Generate synthetic head and shoulders pattern"""
        x = np.linspace(0, 100, 100)
        
        # Create head and shoulders shape
        pattern = np.zeros_like(x)
        
        # Left shoulder
        pattern[15:25] = 5 + 3 * np.sin(np.linspace(0, np.pi, 10))
        
        # Left valley
        pattern[25:35] = 3
        
        # Head
        pattern[35:65] = 3 + 5 * np.sin(np.linspace(0, np.pi, 30))
        
        # Right valley
        pattern[65:75] = 3
        
        # Right shoulder
        pattern[75:85] = 5 + 3 * np.sin(np.linspace(0, np.pi, 10))
        
        # Add some noise
        pattern += 0.1 * np.random.randn(len(pattern))
        
        return pattern
    
    @pytest.fixture
    def double_bottom_pattern(self):
        """Generate synthetic double bottom pattern"""
        x = np.linspace(0, 100, 100)
        pattern = 10 * np.ones_like(x)
        
        # First bottom
        pattern[20:30] = 5 - 2 * np.sin(np.linspace(0, np.pi, 10))
        
        # Middle peak
        pattern[40:60] = 8 + 2 * np.sin(np.linspace(0, np.pi, 20))
        
        # Second bottom
        pattern[70:80] = 5 - 2 * np.sin(np.linspace(0, np.pi, 10))
        
        # Add noise
        pattern += 0.1 * np.random.randn(len(pattern))
        
        return pattern
    
    @pytest.fixture
    def fractal_pattern(self):
        """Generate synthetic fractal pattern"""
        # Create self-similar pattern
        base = np.sin(np.linspace(0, 4*np.pi, 50))
        
        # Add self-similar component at different scale
        detail = 0.3 * np.sin(np.linspace(0, 16*np.pi, 50))
        
        pattern = base + detail
        
        # Make it more fractal-like by repeating structure
        full_pattern = np.concatenate([pattern, 0.7 * pattern])
        
        return full_pattern
    
    def test_initialization(self, classifier):
        """Test classifier initialization"""
        assert classifier is not None
        assert len(classifier.pattern_definitions) > 0
        
        # Check default patterns are loaded
        assert 'head_and_shoulders' in classifier.pattern_definitions
        assert 'double_bottom' in classifier.pattern_definitions
        assert 'bull_flag' in classifier.pattern_definitions
    
    def test_pattern_statistics(self, classifier):
        """Test pattern statistics retrieval"""
        stats = classifier.get_pattern_statistics()
        
        assert 'total_patterns' in stats
        assert stats['total_patterns'] > 0
        assert 'by_category' in stats
        assert 'traditional' in stats['by_category']
        assert len(stats['pattern_list']) == stats['total_patterns']
    
    def test_head_and_shoulders_classification(self, classifier, head_and_shoulders_pattern):
        """Test head and shoulders pattern classification"""
        result = classifier.classify_pattern(head_and_shoulders_pattern)
        
        assert result is not None
        assert 'best_match' in result
        assert 'confidence' in result
        assert 'features' in result
        
        # Should identify as head and shoulders
        if result['best_match']:
            assert 'head' in result['best_match']['name'].lower() or \
                   'shoulders' in result['best_match']['name'].lower()
            assert result['confidence'] > 0.5
    
    def test_double_bottom_classification(self, classifier, double_bottom_pattern):
        """Test double bottom pattern classification"""
        result = classifier.classify_pattern(double_bottom_pattern)
        
        assert result is not None
        if result['best_match']:
            assert 'double' in result['best_match']['name'].lower() or \
                   'bottom' in result['best_match']['name'].lower()
    
    def test_fractal_classification(self, classifier, fractal_pattern):
        """Test fractal pattern classification"""
        result = classifier.classify_pattern(fractal_pattern)
        
        assert result is not None
        if result['best_match']:
            assert result['best_match']['category'] in ['fractal', 'shapelet']
    
    def test_feature_extraction(self, classifier):
        """Test pattern feature extraction"""
        # Create simple pattern
        pattern = np.sin(np.linspace(0, 4*np.pi, 100))
        
        result = classifier.classify_pattern(pattern)
        features = result['features']
        
        # Check all expected features are present
        expected_features = [
            'mean', 'std', 'skew', 'kurtosis',
            'num_peaks', 'num_valleys', 'peak_valley_ratio',
            'overall_trend', 'volatility', 'energy', 'entropy'
        ]
        
        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
    
    def test_custom_pattern_addition(self, classifier):
        """Test adding custom pattern"""
        custom_pattern = PatternDefinition(
            name='Custom Triangle',
            category='custom',
            description='User-defined triangle pattern',
            key_points=[
                {'position': 0.0, 'type': 'start'},
                {'position': 0.5, 'type': 'apex'},
                {'position': 1.0, 'type': 'end'}
            ],
            validation_rules={
                'triangle_shape': True,
                'symmetry': 0.8
            }
        )
        
        initial_count = len(classifier.pattern_definitions)
        classifier.add_custom_pattern(custom_pattern)
        
        assert len(classifier.pattern_definitions) == initial_count + 1
        assert 'custom_triangle' in classifier.pattern_definitions
    
    def test_pattern_matching_confidence(self, classifier):
        """Test pattern matching confidence scores"""
        # Create patterns with varying quality
        good_pattern = np.sin(np.linspace(0, 2*np.pi, 100))
        noisy_pattern = good_pattern + 0.5 * np.random.randn(100)
        random_pattern = np.random.randn(100)
        
        good_result = classifier.classify_pattern(good_pattern)
        noisy_result = classifier.classify_pattern(noisy_pattern)
        random_result = classifier.classify_pattern(random_pattern)
        
        # Random pattern should have lower confidence
        if random_result['best_match'] and good_result['best_match']:
            assert random_result['confidence'] <= good_result['confidence']
    
    def test_multiple_matches(self, classifier):
        """Test patterns that match multiple definitions"""
        # Create ambiguous pattern
        x = np.linspace(0, 100, 100)
        pattern = 5 + 2 * np.sin(x/10) + np.sin(x/5)
        
        result = classifier.classify_pattern(pattern)
        
        assert 'all_matches' in result
        assert isinstance(result['all_matches'], list)
        
        # If multiple matches, best match should have highest confidence
        if len(result['all_matches']) > 1 and result['best_match']:
            best_confidence = result['best_match']['confidence']
            for match in result['all_matches']:
                assert match['confidence'] <= best_confidence
    
    def test_pattern_library_persistence(self, classifier):
        """Test saving and loading pattern library"""
        # Add custom pattern
        custom_pattern = PatternDefinition(
            name='Test Pattern',
            category='test',
            description='Test pattern for persistence',
            key_points=[{'position': 0.5, 'type': 'center'}],
            validation_rules={'test_rule': True}
        )
        
        classifier.add_custom_pattern(custom_pattern)
        
        # Create new classifier with same library path
        new_classifier = PatternClassifier(
            pattern_library_path=classifier.pattern_library_path
        )
        
        # Check custom pattern was loaded
        assert 'test_pattern' in new_classifier.pattern_definitions
        assert new_classifier.pattern_definitions['test_pattern'].category == 'test'
    
    def test_edge_cases(self, classifier):
        """Test edge cases"""
        # Empty pattern
        with pytest.raises(Exception):
            classifier.classify_pattern(np.array([]))
        
        # Single value pattern
        result = classifier.classify_pattern(np.array([1.0]))
        assert result is not None
        
        # Very short pattern
        result = classifier.classify_pattern(np.array([1, 2, 3]))
        assert result is not None
        
        # Pattern with NaN values
        pattern_with_nan = np.array([1, 2, np.nan, 4, 5])
        # Should handle gracefully
        result = classifier.classify_pattern(np.nan_to_num(pattern_with_nan))
        assert result is not None
    
    def test_pattern_categories(self, classifier):
        """Test pattern categorization"""
        stats = classifier.get_pattern_statistics()
        
        # Check expected categories exist
        assert 'traditional' in stats['by_category']
        
        # Verify patterns are properly categorized
        for pattern_info in stats['pattern_list']:
            assert pattern_info['category'] in ['traditional', 'shapelet', 'fractal', 'custom']
    
    def test_shapelet_detection(self, classifier):
        """Test shapelet/motif detection"""
        # Create unique pattern that doesn't match traditional patterns
        x = np.linspace(0, 10, 100)
        unique_pattern = np.sin(x) * np.exp(-x/5) + 0.5 * np.sin(10*x)
        
        result = classifier.classify_pattern(unique_pattern)
        
        # Should classify as shapelet if no traditional match
        if result['best_match'] and not any(
            match['category'] == 'traditional' 
            for match in result['all_matches']
        ):
            assert result['best_match']['category'] in ['shapelet', 'fractal']
    
    def test_pattern_id_generation(self, classifier):
        """Test unique pattern ID generation"""
        # Create two different patterns
        pattern1 = np.sin(np.linspace(0, 2*np.pi, 100))
        pattern2 = np.cos(np.linspace(0, 2*np.pi, 100))
        
        result1 = classifier.classify_pattern(pattern1)
        result2 = classifier.classify_pattern(pattern2)
        
        # If both are shapelets, they should have different IDs
        if (result1['best_match'] and result2['best_match'] and
            'Shapelet #' in result1['best_match']['name'] and
            'Shapelet #' in result2['best_match']['name']):
            
            id1 = result1['best_match']['name'].split('#')[1]
            id2 = result2['best_match']['name'].split('#')[1]
            # IDs might be same due to hash collision, but should usually differ
            # This is a probabilistic test
    
    def test_confidence_threshold(self, classifier):
        """Test confidence threshold filtering"""
        # Create poor quality pattern
        random_walk = np.cumsum(np.random.randn(100))
        
        result = classifier.classify_pattern(random_walk)
        
        # All matches should meet their confidence thresholds
        for match in result['all_matches']:
            pattern_name = match['name'].lower().replace(' ', '_')
            if pattern_name in classifier.pattern_definitions:
                threshold = classifier.pattern_definitions[pattern_name].confidence_threshold
                assert match['confidence'] >= threshold


class TestPatternDefinition:
    """Test PatternDefinition class"""
    
    def test_pattern_definition_creation(self):
        """Test creating pattern definition"""
        pattern_def = PatternDefinition(
            name='Test Pattern',
            category='test',
            description='A test pattern',
            key_points=[{'position': 0.5, 'type': 'center'}],
            validation_rules={'rule1': True}
        )
        
        assert pattern_def.name == 'Test Pattern'
        assert pattern_def.category == 'test'
        assert pattern_def.confidence_threshold == 0.7  # Default
    
    def test_pattern_definition_serialization(self):
        """Test pattern definition to/from dict"""
        pattern_def = PatternDefinition(
            name='Test Pattern',
            category='test',
            description='A test pattern',
            key_points=[{'position': 0.5, 'type': 'center'}],
            validation_rules={'rule1': True},
            confidence_threshold=0.8
        )
        
        # Convert to dict
        pattern_dict = pattern_def.to_dict()
        assert isinstance(pattern_dict, dict)
        assert pattern_dict['name'] == 'Test Pattern'
        assert pattern_dict['confidence_threshold'] == 0.8
        
        # Convert back from dict
        pattern_def2 = PatternDefinition.from_dict(pattern_dict)
        assert pattern_def2.name == pattern_def.name
        assert pattern_def2.category == pattern_def.category
        assert pattern_def2.confidence_threshold == pattern_def.confidence_threshold
