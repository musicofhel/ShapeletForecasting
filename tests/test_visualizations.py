"""
Comprehensive tests for visualization components
Tests rendering, interactivity, and performance
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from unittest.mock import Mock, patch
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dashboard.visualizations.timeseries import create_timeseries_plot
from src.dashboard.visualizations.scalogram import create_scalogram_plot
from src.dashboard.visualizations.pattern_gallery import create_pattern_gallery
from src.dashboard.visualizations.pattern_comparison import create_comparison_plots
from src.dashboard.visualizations.analytics_simple import create_analytics_dashboard


class TestTimeseriesVisualization:
    """Test timeseries visualization components"""
    
    @pytest.fixture
    def sample_timeseries(self):
        """Generate sample timeseries data"""
        dates = pd.date_range(start='2024-01-01', periods=500, freq='1H')
        prices = 100 + np.cumsum(np.random.randn(500) * 0.5)
        
        return pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': np.random.randint(1000, 10000, 500)
        })
    
    def test_basic_timeseries_plot(self, sample_timeseries):
        """Test basic timeseries plot creation"""
        fig = create_timeseries_plot(
            sample_timeseries,
            title="Test Timeseries"
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        assert fig.layout.title.text == "Test Timeseries"
        
        # Check data integrity
        trace = fig.data[0]
        assert len(trace.x) == len(sample_timeseries)
        assert len(trace.y) == len(sample_timeseries)
    
    def test_multi_trace_timeseries(self, sample_timeseries):
        """Test timeseries with multiple traces"""
        # Add additional series
        sample_timeseries['sma_20'] = sample_timeseries['price'].rolling(20).mean()
        sample_timeseries['sma_50'] = sample_timeseries['price'].rolling(50).mean()
        
        fig = create_timeseries_plot(
            sample_timeseries,
            columns=['price', 'sma_20', 'sma_50'],
            title="Multi-trace Plot"
        )
        
        assert len(fig.data) == 3
        assert fig.data[0].name == 'price'
        assert fig.data[1].name == 'sma_20'
        assert fig.data[2].name == 'sma_50'
    
    def test_timeseries_with_patterns(self, sample_timeseries):
        """Test timeseries with pattern overlays"""
        patterns = [
            {'start': 50, 'end': 100, 'type': 'bullish', 'confidence': 0.85},
            {'start': 200, 'end': 250, 'type': 'bearish', 'confidence': 0.75},
            {'start': 350, 'end': 400, 'type': 'neutral', 'confidence': 0.65}
        ]
        
        fig = create_timeseries_plot(
            sample_timeseries,
            patterns=patterns,
            show_patterns=True
        )
        
        # Should have main trace plus pattern shapes
        assert len(fig.data) >= 1
        assert len(fig.layout.shapes) == len(patterns)
        
        # Check pattern colors
        for i, shape in enumerate(fig.layout.shapes):
            if patterns[i]['type'] == 'bullish':
                assert 'green' in shape.fillcolor.lower() or 'rgba(0,' in shape.fillcolor
            elif patterns[i]['type'] == 'bearish':
                assert 'red' in shape.fillcolor.lower() or 'rgba(255,' in shape.fillcolor
    
    def test_timeseries_interactivity(self, sample_timeseries):
        """Test interactive features of timeseries plot"""
        fig = create_timeseries_plot(
            sample_timeseries,
            enable_rangeslider=True,
            enable_rangeselector=True
        )
        
        # Check rangeslider
        assert fig.layout.xaxis.rangeslider.visible is True
        
        # Check rangeselector buttons
        assert hasattr(fig.layout.xaxis, 'rangeselector')
        assert len(fig.layout.xaxis.rangeselector.buttons) > 0
        
        # Check hover template
        assert fig.data[0].hovertemplate is not None
    
    def test_timeseries_annotations(self, sample_timeseries):
        """Test timeseries with annotations"""
        annotations = [
            {'x': sample_timeseries['timestamp'][100], 'text': 'Peak', 'y': sample_timeseries['price'][100]},
            {'x': sample_timeseries['timestamp'][200], 'text': 'Trough', 'y': sample_timeseries['price'][200]}
        ]
        
        fig = create_timeseries_plot(
            sample_timeseries,
            annotations=annotations
        )
        
        assert len(fig.layout.annotations) == len(annotations)
        assert fig.layout.annotations[0].text == 'Peak'
        assert fig.layout.annotations[1].text == 'Trough'
    
    def test_timeseries_performance(self):
        """Test timeseries performance with large datasets"""
        # Generate large dataset
        large_dates = pd.date_range(start='2020-01-01', periods=50000, freq='1min')
        large_prices = 100 + np.cumsum(np.random.randn(50000) * 0.1)
        
        large_df = pd.DataFrame({
            'timestamp': large_dates,
            'price': large_prices
        })
        
        import time
        start_time = time.time()
        fig = create_timeseries_plot(large_df, downsample=True)
        creation_time = time.time() - start_time
        
        # Should create plot quickly even with large data
        assert creation_time < 1.0  # Less than 1 second
        
        # Should downsample for performance
        assert len(fig.data[0].x) < len(large_df)  # Downsampled


class TestScalogramVisualization:
    """Test scalogram visualization components"""
    
    @pytest.fixture
    def wavelet_coefficients(self):
        """Generate sample wavelet coefficients"""
        scales = np.arange(1, 65)
        time_points = 500
        
        # Create synthetic wavelet coefficients
        coeffs = np.zeros((len(scales), time_points))
        for i, scale in enumerate(scales):
            # Add some patterns at different scales
            freq = 1.0 / scale
            coeffs[i, :] = np.sin(2 * np.pi * freq * np.arange(time_points))
            coeffs[i, :] *= np.exp(-0.01 * scale)  # Decay with scale
            
        return coeffs, scales
    
    def test_basic_scalogram(self, wavelet_coefficients):
        """Test basic scalogram creation"""
        coeffs, scales = wavelet_coefficients
        
        fig = create_scalogram_plot(
            coeffs,
            scales=scales,
            title="Test Scalogram"
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        assert fig.layout.title.text == "Test Scalogram"
        
        # Check heatmap properties
        heatmap = fig.data[0]
        assert heatmap.type == 'heatmap'
        assert heatmap.z.shape == coeffs.shape
    
    def test_scalogram_colorscales(self, wavelet_coefficients):
        """Test different colorscales for scalogram"""
        coeffs, scales = wavelet_coefficients
        
        colorscales = ['Viridis', 'Plasma', 'RdBu', 'Jet']
        
        for colorscale in colorscales:
            fig = create_scalogram_plot(
                coeffs,
                scales=scales,
                colorscale=colorscale
            )
            
            assert fig.data[0].colorscale == colorscale
    
    def test_scalogram_with_ridges(self, wavelet_coefficients):
        """Test scalogram with ridge detection"""
        coeffs, scales = wavelet_coefficients
        
        # Create synthetic ridges
        ridges = [
            {'scale_idx': 10, 'time_idx': list(range(100, 200))},
            {'scale_idx': 25, 'time_idx': list(range(250, 350))}
        ]
        
        fig = create_scalogram_plot(
            coeffs,
            scales=scales,
            ridges=ridges,
            show_ridges=True
        )
        
        # Should have heatmap plus ridge lines
        assert len(fig.data) > 1
        
        # Check ridge traces
        ridge_traces = [trace for trace in fig.data if trace.mode == 'lines']
        assert len(ridge_traces) == len(ridges)
    
    def test_scalogram_3d_view(self, wavelet_coefficients):
        """Test 3D scalogram visualization"""
        coeffs, scales = wavelet_coefficients
        
        fig = create_scalogram_plot(
            coeffs,
            scales=scales,
            plot_type='3d'
        )
        
        # Should be a 3D surface plot
        assert fig.data[0].type == 'surface'
        assert hasattr(fig.layout, 'scene')
        
        # Check 3D axes labels
        assert fig.layout.scene.xaxis.title.text is not None
        assert fig.layout.scene.yaxis.title.text is not None
        assert fig.layout.scene.zaxis.title.text is not None
    
    def test_scalogram_interactivity(self, wavelet_coefficients):
        """Test interactive features of scalogram"""
        coeffs, scales = wavelet_coefficients
        
        fig = create_scalogram_plot(
            coeffs,
            scales=scales,
            enable_zoom=True,
            enable_pan=True
        )
        
        # Check interactive config
        assert fig.layout.dragmode == 'zoom'
        assert fig.layout.hovermode == 'closest'
        
        # Check hover template
        assert fig.data[0].hovertemplate is not None


class TestPatternGallery:
    """Test pattern gallery visualization"""
    
    @pytest.fixture
    def sample_patterns(self):
        """Generate sample patterns"""
        patterns = []
        for i in range(12):
            t = np.linspace(0, 10, 100)
            if i % 3 == 0:
                pattern_data = np.sin(t) * np.exp(-t/10)
                pattern_type = 'decay'
            elif i % 3 == 1:
                pattern_data = np.sin(t) + 0.5 * np.sin(3*t)
                pattern_type = 'complex'
            else:
                pattern_data = t * np.sin(t)
                pattern_type = 'growth'
            
            patterns.append({
                'id': f'pattern_{i}',
                'data': pattern_data,
                'type': pattern_type,
                'confidence': 0.6 + 0.3 * np.random.rand(),
                'frequency': np.random.randint(5, 50),
                'last_seen': datetime.now() - timedelta(hours=np.random.randint(1, 100))
            })
        
        return patterns
    
    def test_basic_pattern_gallery(self, sample_patterns):
        """Test basic pattern gallery creation"""
        fig = create_pattern_gallery(
            sample_patterns,
            title="Pattern Gallery"
        )
        
        assert isinstance(fig, go.Figure)
        
        # Should create subplot grid
        assert hasattr(fig, '_grid_ref')
        assert len(fig.data) == len(sample_patterns)
    
    def test_pattern_gallery_filtering(self, sample_patterns):
        """Test pattern gallery with filtering"""
        # Filter by type
        fig = create_pattern_gallery(
            sample_patterns,
            filter_type='decay',
            title="Decay Patterns"
        )
        
        decay_patterns = [p for p in sample_patterns if p['type'] == 'decay']
        assert len(fig.data) == len(decay_patterns)
        
        # Filter by confidence
        fig = create_pattern_gallery(
            sample_patterns,
            min_confidence=0.8,
            title="High Confidence Patterns"
        )
        
        high_conf_patterns = [p for p in sample_patterns if p['confidence'] >= 0.8]
        assert len(fig.data) == len(high_conf_patterns)
    
    def test_pattern_gallery_sorting(self, sample_patterns):
        """Test pattern gallery sorting options"""
        # Sort by frequency
        fig = create_pattern_gallery(
            sample_patterns,
            sort_by='frequency',
            ascending=False
        )
        
        # Check that patterns are sorted
        frequencies = [p['frequency'] for p in sample_patterns]
        frequencies.sort(reverse=True)
        
        # Verify order in subplot titles
        for i, subplot in enumerate(fig._grid_ref):
            if i < len(frequencies):
                assert f"Freq: {frequencies[i]}" in fig.layout.annotations[i].text
    
    def test_pattern_gallery_layout(self, sample_patterns):
        """Test pattern gallery layout options"""
        # Test different grid sizes
        for cols in [2, 3, 4, 6]:
            fig = create_pattern_gallery(
                sample_patterns,
                columns=cols
            )
            
            # Check grid dimensions
            rows = (len(sample_patterns) + cols - 1) // cols
            assert len(fig._grid_ref) == rows
            assert len(fig._grid_ref[0]) == cols
    
    def test_pattern_gallery_annotations(self, sample_patterns):
        """Test pattern gallery with annotations"""
        fig = create_pattern_gallery(
            sample_patterns,
            show_stats=True,
            show_confidence=True
        )
        
        # Should have annotations for each pattern
        annotations = fig.layout.annotations
        assert len(annotations) >= len(sample_patterns)
        
        # Check annotation content
        for ann in annotations:
            text = ann.text
            assert 'Confidence:' in text or 'Type:' in text or 'Freq:' in text


class TestPatternComparison:
    """Test pattern comparison visualizations"""
    
    @pytest.fixture
    def comparison_patterns(self):
        """Generate patterns for comparison"""
        t = np.linspace(0, 10, 100)
        
        patterns = {
            'pattern_1': {
                'data': np.sin(t) * np.exp(-t/10),
                'confidence': 0.85,
                'type': 'decay',
                'metrics': {'amplitude': 1.0, 'frequency': 1.0, 'decay_rate': 0.1}
            },
            'pattern_2': {
                'data': np.sin(t) * np.exp(-t/20),
                'confidence': 0.75,
                'type': 'decay',
                'metrics': {'amplitude': 1.0, 'frequency': 1.0, 'decay_rate': 0.05}
            },
            'pattern_3': {
                'data': np.sin(2*t) * np.exp(-t/10),
                'confidence': 0.80,
                'type': 'decay',
                'metrics': {'amplitude': 1.0, 'frequency': 2.0, 'decay_rate': 0.1}
            }
        }
        
        return patterns
    
    def test_side_by_side_comparison(self, comparison_patterns):
        """Test side-by-side pattern comparison"""
        figs = create_comparison_plots(
            comparison_patterns,
            plot_type='side_by_side'
        )
        
        assert 'side_by_side' in figs
        fig = figs['side_by_side']
        
        # Should have subplot for each pattern
        assert len(fig.data) == len(comparison_patterns)
        
        # Check subplot arrangement
        assert hasattr(fig, '_grid_ref')
    
    def test_overlay_comparison(self, comparison_patterns):
        """Test overlay pattern comparison"""
        figs = create_comparison_plots(
            comparison_patterns,
            plot_type='overlay'
        )
        
        assert 'overlay' in figs
        fig = figs['overlay']
        
        # All patterns in same plot
        assert len(fig.data) == len(comparison_patterns)
        
        # Check different colors/styles
        colors = [trace.line.color for trace in fig.data]
        assert len(set(colors)) == len(colors)  # All different colors
    
    def test_similarity_matrix(self, comparison_patterns):
        """Test pattern similarity matrix"""
        figs = create_comparison_plots(
            comparison_patterns,
            plot_type='similarity_matrix'
        )
        
        assert 'similarity_matrix' in figs
        fig = figs['similarity_matrix']
        
        # Should be heatmap
        assert fig.data[0].type == 'heatmap'
        
        # Check matrix dimensions
        n_patterns = len(comparison_patterns)
        assert fig.data[0].z.shape == (n_patterns, n_patterns)
        
        # Diagonal should be 1 (self-similarity)
        z_matrix = fig.data[0].z
        for i in range(n_patterns):
            assert abs(z_matrix[i][i] - 1.0) < 0.001
    
    def test_metric_comparison(self, comparison_patterns):
        """Test pattern metric comparison"""
        figs = create_comparison_plots(
            comparison_patterns,
            plot_type='metrics'
        )
        
        assert 'metrics' in figs
        fig = figs['metrics']
        
        # Should be bar chart or radar chart
        assert fig.data[0].type in ['bar', 'scatterpolar']
        
        # Check all metrics are included
        if fig.data[0].type == 'bar':
            metric_names = set()
            for trace in fig.data:
                metric_names.add(trace.name)
            
            expected_metrics = {'amplitude', 'frequency', 'decay_rate'}
            assert expected_metrics.issubset(metric_names)
    
    def test_evolution_comparison(self, comparison_patterns):
        """Test pattern evolution visualization"""
        # Add temporal information
        for i, (key, pattern) in enumerate(comparison_patterns.items()):
            pattern['timestamp'] = datetime.now() - timedelta(hours=i*24)
            pattern['evolution_score'] = 0.5 + 0.1 * i
        
        figs = create_comparison_plots(
            comparison_patterns,
            plot_type='evolution'
        )
        
        assert 'evolution' in figs
        fig = figs['evolution']
        
        # Should show temporal progression
        assert len(fig.data) > 0
        
        # Check time axis
        assert fig.layout.xaxis.title.text.lower() in ['time', 'timestamp', 'date']


class TestAnalyticsDashboard:
    """Test analytics dashboard components"""
    
    @pytest.fixture
    def analytics_data(self):
        """Generate sample analytics data"""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        
        return {
            'pattern_counts': pd.DataFrame({
                'date': dates,
                'bullish': np.random.randint(5, 20, 30),
                'bearish': np.random.randint(5, 20, 30),
                'neutral': np.random.randint(10, 30, 30)
            }),
            'accuracy_metrics': pd.DataFrame({
                'date': dates,
                'accuracy': 0.7 + 0.2 * np.random.rand(30),
                'precision': 0.6 + 0.3 * np.random.rand(30),
                'recall': 0.65 + 0.25 * np.random.rand(30)
            }),
            'performance_stats': {
                'total_patterns': 1234,
                'avg_confidence': 0.78,
                'success_rate': 0.72,
                'avg_return': 0.045
            }
        }
    
    def test_analytics_dashboard_creation(self, analytics_data):
        """Test analytics dashboard creation"""
        figs = create_analytics_dashboard(analytics_data)
        
        assert isinstance(figs, dict)
        
        # Check required components
        expected_charts = ['pattern_distribution', 'accuracy_trends', 'performance_summary']
        for chart in expected_charts:
            assert chart in figs
            assert isinstance(figs[chart], go.Figure)
    
    def test_pattern_distribution_chart(self, analytics_data):
        """Test pattern distribution visualization"""
        figs = create_analytics_dashboard(analytics_data)
        fig = figs['pattern_distribution']
        
        # Should show distribution over time
        assert len(fig.data) >= 3  # bullish, bearish, neutral
        
        # Check stacking
        if hasattr(fig.data[0], 'stackgroup'):
            assert all(trace.stackgroup is not None for trace in fig.data)
    
    def test_accuracy_trends_chart(self, analytics_data):
        """Test accuracy trends visualization"""
        figs = create_analytics_dashboard(analytics_data)
        fig = figs['accuracy_trends']
        
        # Should show multiple metrics
        assert len(fig.data) >= 3  # accuracy, precision, recall
        
        # Check y-axis range (percentages)
        assert fig.layout.yaxis.range[0] >= 0
        assert fig.layout.yaxis.range[1] <= 1.1
    
    def test_performance_summary_chart(self, analytics_data):
        """Test performance summary visualization"""
        figs = create_analytics_dashboard(analytics_data)
        fig = figs['performance_summary']
        
        # Could be gauge charts or indicator cards
        if fig.data[0].type == 'indicator':
            # Check indicator properties
            assert fig.data[0].mode in ['number+delta', 'gauge+number']
            assert fig.data[0].value is not None
        elif fig.data[0].type == 'bar':
            # Check bar chart properties
            assert len(fig.data[0].y) == len(analytics_data['performance_stats'])
    
    def test_analytics_interactivity(self, analytics_data):
        """Test analytics dashboard interactivity"""
        figs = create_analytics_dashboard(
            analytics_data,
            enable_zoom=True,
            enable_selection=True
        )
        
        for fig in figs.values():
            # Check interactive features
            assert fig.layout.hovermode is not None
            assert fig.layout.dragmode is not None
            
            # Check for range selectors on time-based charts
            if hasattr(fig.layout, 'xaxis') and 'date' in str(fig.layout.xaxis.title.text).lower():
                assert hasattr(fig.layout.xaxis, 'rangeselector') or fig.layout.xaxis.rangeslider.visible


class TestVisualizationPerformance:
    """Test visualization performance metrics"""
    
    def test_large_dataset_rendering(self):
        """Test rendering performance with large datasets"""
        import time
        
        # Generate large dataset
        n_points = 100000
        large_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2020-01-01', periods=n_points, freq='1min'),
            'value': np.cumsum(np.random.randn(n_points))
        })
        
        start_time = time.time()
        fig = create_timeseries_plot(large_data, downsample=True, max_points=5000)
        render_time = time.time() - start_time
        
        # Should render quickly
        assert render_time < 2.0  # Less than 2 seconds
        
        # Should be downsampled
        total_points = sum(len(trace.x) for trace in fig.data)
        assert total_points < 10000  # Reasonably downsampled
    
    def test_memory_usage(self):
        """Test memory usage of visualizations"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple large visualizations
        figs = []
        for i in range(10):
            data = pd.DataFrame({
                'x': range(10000),
                'y': np.random.randn(10000)
            })
            fig = create_timeseries_plot(data)
            figs.append(fig)
        
        current_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = current_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 200  # Less than 200MB for 10 large plots
    
    def test_export_performance(self):
        """Test export performance for different formats"""
        import time
        
        # Create complex visualization
        data = pd.DataFrame({
            'x': range(1000),
            'y1': np.random.randn(1000),
            'y2': np.random.randn(1000),
            'y3': np.random.randn(1000)
        })
        
        fig = create_timeseries_plot(data, columns=['y1', 'y2', 'y3'])
        
        # Test HTML export
        start_time = time.time()
        html_str = fig.to_html()
        html_time = time.time() - start_time
        
        assert html_time < 1.0  # Less than 1 second
        assert len(html_str) > 1000  # Non-empty HTML
        
        # Test JSON export
        start_time = time.time()
        json_str = fig.to_json()
        json_time = time.time() - start_time
        
        assert json_time < 0.5  # Less than 0.5 seconds
        assert len(json_str) > 1000  # Non-empty JSON


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
