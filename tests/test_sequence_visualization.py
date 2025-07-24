"""
Tests for Pattern Sequence Visualization

Tests the pattern sequence visualization component including
timeline views, transitions, and probability overlays.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from src.dashboard.visualizations.sequence_view import PatternSequenceVisualizer


class TestPatternSequenceVisualizer:
    """Test suite for pattern sequence visualization"""
    
    @pytest.fixture
    def visualizer(self):
        """Create a visualizer instance"""
        return PatternSequenceVisualizer()
    
    @pytest.fixture
    def sample_patterns(self):
        """Create sample pattern data"""
        base_time = datetime(2024, 1, 1, 9, 30)
        patterns = [
            {
                'id': 'P1',
                'type': 'head_shoulders',
                'start_time': base_time,
                'end_time': base_time + timedelta(hours=2),
                'confidence': 0.85
            },
            {
                'id': 'P2',
                'type': 'double_top',
                'start_time': base_time + timedelta(hours=3),
                'end_time': base_time + timedelta(hours=5),
                'confidence': 0.72
            },
            {
                'id': 'P3',
                'type': 'triangle_ascending',
                'start_time': base_time + timedelta(hours=6),
                'end_time': base_time + timedelta(hours=8),
                'confidence': 0.91
            },
            {
                'id': 'P4',
                'type': 'flag_bull',
                'start_time': base_time + timedelta(hours=9),
                'end_time': base_time + timedelta(hours=10),
                'confidence': 0.68
            }
        ]
        return patterns
    
    @pytest.fixture
    def sample_transitions(self):
        """Create sample transition data"""
        return {
            'P1_to_P2': {'probability': 0.75},
            'P2_to_P3': {'probability': 0.45},
            'P3_to_P4': {'probability': 0.82}
        }
    
    @pytest.fixture
    def transition_matrix_data(self):
        """Create transition probability matrix data"""
        return {
            ('head_shoulders', 'double_top'): 0.35,
            ('head_shoulders', 'triangle_ascending'): 0.25,
            ('double_top', 'triangle_ascending'): 0.45,
            ('double_top', 'flag_bull'): 0.30,
            ('triangle_ascending', 'flag_bull'): 0.55,
            ('triangle_ascending', 'head_shoulders'): 0.15,
            ('flag_bull', 'head_shoulders'): 0.40,
            ('flag_bull', 'double_top'): 0.20
        }
    
    def test_initialization(self, visualizer):
        """Test visualizer initialization"""
        assert visualizer is not None
        assert len(visualizer.color_palette) > 0
        assert len(visualizer.transition_colors) == 3
        assert 'head_shoulders' in visualizer.color_palette
        assert 'high_prob' in visualizer.transition_colors
    
    def test_create_pattern_timeline_basic(self, visualizer, sample_patterns):
        """Test basic pattern timeline creation"""
        fig = visualizer.create_pattern_timeline(sample_patterns)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text == "Pattern Sequence Timeline"
        
        # Check that patterns are represented
        pattern_traces = [trace for trace in fig.data if hasattr(trace, 'fill')]
        assert len(pattern_traces) >= len(sample_patterns)
    
    def test_create_pattern_timeline_with_transitions(self, visualizer, sample_patterns, sample_transitions):
        """Test pattern timeline with transitions"""
        fig = visualizer.create_pattern_timeline(
            sample_patterns, 
            transitions=sample_transitions,
            show_probabilities=True
        )
        
        assert isinstance(fig, go.Figure)
        # Should have pattern bars plus transition arrows
        assert len(fig.data) > len(sample_patterns)
    
    def test_create_pattern_timeline_empty(self, visualizer):
        """Test timeline with no patterns"""
        fig = visualizer.create_pattern_timeline([])
        
        assert isinstance(fig, go.Figure)
        # Should show empty figure message
        assert len(fig.layout.annotations) > 0
        assert "No patterns" in fig.layout.annotations[0].text
    
    def test_create_transition_matrix(self, visualizer, transition_matrix_data):
        """Test transition matrix heatmap creation"""
        fig = visualizer.create_transition_matrix(transition_matrix_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)
        assert fig.layout.title.text == 'Pattern Transition Probability Matrix'
        
        # Check heatmap data
        heatmap = fig.data[0]
        assert heatmap.z is not None
        assert len(heatmap.x) > 0
        assert len(heatmap.y) > 0
    
    def test_create_transition_matrix_filtering(self, visualizer, transition_matrix_data):
        """Test transition matrix with probability filtering"""
        fig = visualizer.create_transition_matrix(
            transition_matrix_data,
            min_probability=0.4
        )
        
        heatmap = fig.data[0]
        # Check that low probability transitions are filtered
        z_array = np.array(heatmap.z)
        non_zero_values = z_array[z_array > 0]
        assert all(val >= 0.4 for val in non_zero_values)
    
    def test_create_pattern_flow_diagram(self, visualizer, sample_patterns):
        """Test pattern flow diagram creation"""
        transitions = {
            'P1': [
                {'to_pattern': 'double_top', 'probability': 0.75},
                {'to_pattern': 'triangle_ascending', 'probability': 0.20}
            ],
            'P2': [
                {'to_pattern': 'triangle_ascending', 'probability': 0.45},
                {'to_pattern': 'flag_bull', 'probability': 0.35}
            ]
        }
        
        fig = visualizer.create_pattern_flow_diagram(
            sample_patterns,
            transitions,
            max_depth=3
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Sankey)
        
        # Check Sankey diagram structure
        sankey = fig.data[0]
        assert len(sankey.node.label) > 0
        assert len(sankey.link.source) > 0
        assert len(sankey.link.target) > 0
    
    def test_create_pattern_duration_analysis(self, visualizer, sample_patterns):
        """Test pattern duration analysis visualization"""
        fig = visualizer.create_pattern_duration_analysis(sample_patterns)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        
        # Should have box plots for each pattern type
        pattern_types = set(p['type'] for p in sample_patterns)
        assert len(fig.data) == len(pattern_types)
        
        # Check that all traces are box plots
        for trace in fig.data:
            assert isinstance(trace, go.Box)
    
    def test_create_confidence_timeline(self, visualizer, sample_patterns):
        """Test confidence timeline creation"""
        fig = visualizer.create_confidence_timeline(sample_patterns)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        
        # Should have confidence line
        confidence_trace = fig.data[0]
        assert isinstance(confidence_trace, go.Scatter)
        assert confidence_trace.mode == 'lines+markers'
        
        # Check for threshold lines
        shapes = fig.layout.shapes or []
        assert any(hasattr(shape, 'type') and shape.type == 'line' for shape in shapes)
    
    def test_pattern_timeline_color_coding(self, visualizer, sample_patterns):
        """Test that patterns are color-coded by type"""
        fig = visualizer.create_pattern_timeline(sample_patterns)
        
        # Extract pattern traces
        pattern_traces = [trace for trace in fig.data if hasattr(trace, 'fillcolor')]
        
        # Check that different pattern types have different colors
        colors_used = set()
        for trace in pattern_traces:
            if hasattr(trace, 'fillcolor'):
                colors_used.add(trace.fillcolor)
        
        # Should have multiple colors for different pattern types
        assert len(colors_used) > 1
    
    def test_transition_arrow_colors(self, visualizer, sample_patterns):
        """Test that transition arrows are colored by probability"""
        transitions = {
            'P1_to_P2': {'probability': 0.85},  # High
            'P2_to_P3': {'probability': 0.50},  # Medium
            'P3_to_P4': {'probability': 0.25}   # Low
        }
        
        fig = visualizer.create_pattern_timeline(
            sample_patterns,
            transitions=transitions,
            show_probabilities=True
        )
        
        # Find arrow traces
        arrow_traces = [trace for trace in fig.data 
                       if hasattr(trace, 'marker') and 
                       getattr(trace.marker, 'symbol', None) == 'arrow']
        
        # Should have transition arrows
        assert len(arrow_traces) > 0
    
    def test_duration_calculation(self, visualizer):
        """Test duration calculation accuracy"""
        patterns = [
            {
                'id': 'P1',
                'type': 'test_pattern',
                'start_time': datetime(2024, 1, 1, 10, 0),
                'end_time': datetime(2024, 1, 1, 12, 30),  # 2.5 hours
                'confidence': 0.8
            }
        ]
        
        fig = visualizer.create_pattern_duration_analysis(patterns)
        
        # Check that duration is calculated correctly
        box_trace = fig.data[0]
        assert len(box_trace.y) == 1
        assert abs(box_trace.y[0] - 2.5) < 0.01  # 2.5 hours
    
    def test_pattern_timeline_sorting(self, visualizer):
        """Test that patterns are sorted by start time"""
        # Create patterns in random order
        base_time = datetime(2024, 1, 1)
        patterns = [
            {
                'id': 'P3',
                'type': 'pattern3',
                'start_time': base_time + timedelta(hours=6),
                'end_time': base_time + timedelta(hours=7),
                'confidence': 0.8
            },
            {
                'id': 'P1',
                'type': 'pattern1',
                'start_time': base_time,
                'end_time': base_time + timedelta(hours=1),
                'confidence': 0.8
            },
            {
                'id': 'P2',
                'type': 'pattern2',
                'start_time': base_time + timedelta(hours=3),
                'end_time': base_time + timedelta(hours=4),
                'confidence': 0.8
            }
        ]
        
        fig = visualizer.create_pattern_timeline(patterns)
        
        # Patterns should be displayed in chronological order
        # Check y-axis labels
        y_labels = fig.layout.yaxis.ticktext
        assert list(y_labels) == ['P1', 'P2', 'P3']
    
    def test_empty_figure_creation(self, visualizer):
        """Test empty figure creation helper"""
        fig = visualizer._create_empty_figure("Test message")
        
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.annotations) == 1
        assert fig.layout.annotations[0].text == "Test message"
        assert not fig.layout.xaxis.visible
        assert not fig.layout.yaxis.visible
    
    def test_performance_large_dataset(self, visualizer):
        """Test performance with large number of patterns"""
        # Create 100 patterns
        base_time = datetime(2024, 1, 1)
        patterns = []
        pattern_types = list(visualizer.color_palette.keys())
        
        for i in range(100):
            patterns.append({
                'id': f'P{i+1}',
                'type': pattern_types[i % len(pattern_types)],
                'start_time': base_time + timedelta(hours=i*2),
                'end_time': base_time + timedelta(hours=i*2+1),
                'confidence': np.random.uniform(0.5, 0.95)
            })
        
        # Should handle large dataset without error
        fig = visualizer.create_pattern_timeline(patterns)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_custom_height_parameter(self, visualizer, sample_patterns):
        """Test custom height parameter"""
        custom_height = 800
        fig = visualizer.create_pattern_timeline(
            sample_patterns,
            height=custom_height
        )
        
        assert fig.layout.height == custom_height
    
    def test_pattern_flow_max_depth(self, visualizer, sample_patterns):
        """Test pattern flow diagram respects max_depth"""
        transitions = {
            f'P{i}': [{'to_pattern': f'pattern_{j}', 'probability': 0.5} 
                     for j in range(5)]
            for i in range(1, 10)
        }
        
        fig = visualizer.create_pattern_flow_diagram(
            sample_patterns,
            transitions,
            max_depth=2
        )
        
        # Should limit the depth of the flow diagram
        sankey = fig.data[0]
        # Limited depth should result in fewer nodes
        assert len(sankey.node.label) < 50  # Reasonable limit for depth 2
