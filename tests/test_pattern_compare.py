"""
Tests for Enhanced Pattern Comparison Tool
"""

import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path

from src.dashboard.tools.pattern_compare import EnhancedPatternComparison


class TestEnhancedPatternComparison:
    """Test suite for EnhancedPatternComparison class"""
    
    @pytest.fixture
    def comparison(self):
        """Create EnhancedPatternComparison instance"""
        return EnhancedPatternComparison()
    
    @pytest.fixture
    def sample_patterns(self):
        """Create sample patterns for testing"""
        x = np.linspace(0, 2*np.pi, 100)
        return {
            'sine': np.sin(x),
            'cosine': np.cos(x),
            'damped_sine': np.sin(x) * np.exp(-x/5),
            'noisy_sine': np.sin(x) + np.random.normal(0, 0.1, len(x))
        }
    
    def test_initialization(self, comparison):
        """Test proper initialization"""
        assert comparison.statistical_metrics == {}
        assert comparison.morphing_data == {}
        assert comparison.difference_highlights == {}
        assert hasattr(comparison, 'selected_patterns')
        assert hasattr(comparison, 'pattern_data')
    
    def test_add_pattern(self, comparison, sample_patterns):
        """Test adding patterns"""
        for pattern_id, data in sample_patterns.items():
            comparison.add_pattern(pattern_id, data)
        
        assert len(comparison.selected_patterns) == 4
        assert all(pid in comparison.pattern_data for pid in sample_patterns.keys())
    
    def test_calculate_statistical_comparison_metrics(self, comparison, sample_patterns):
        """Test statistical metrics calculation"""
        # Add patterns
        for pattern_id, data in sample_patterns.items():
            comparison.add_pattern(pattern_id, data)
        
        # Calculate metrics
        metrics = comparison.calculate_statistical_comparison_metrics()
        
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        
        # Check metric structure
        for pair_key, pair_metrics in metrics.items():
            assert 'mean_difference' in pair_metrics
            assert 'pearson_correlation' in pair_metrics
            assert 'rmse' in pair_metrics
            assert 'ks_statistic' in pair_metrics
            assert 'ks_pvalue' in pair_metrics
            assert 'trend_similarity' in pair_metrics
            
            # Check value ranges
            assert -1 <= pair_metrics['pearson_correlation'] <= 1
            assert 0 <= pair_metrics['ks_pvalue'] <= 1
            assert 0 <= pair_metrics['trend_similarity'] <= 1
    
    def test_phase_shift_calculation(self, comparison):
        """Test phase shift calculation"""
        x = np.linspace(0, 2*np.pi, 100)
        data1 = np.sin(x)
        data2 = np.sin(x + np.pi/4)  # 45 degree phase shift
        
        shift = comparison._calculate_phase_shift(data1, data2)
        assert isinstance(shift, int)
    
    def test_trend_similarity_calculation(self, comparison):
        """Test trend similarity calculation"""
        # Same trend
        data1 = np.linspace(0, 10, 100)
        data2 = np.linspace(0, 20, 100)
        similarity = comparison._calculate_trend_similarity(data1, data2)
        assert similarity > 0.9  # Should be very similar
        
        # Opposite trend
        data3 = np.linspace(10, 0, 100)
        similarity2 = comparison._calculate_trend_similarity(data1, data3)
        assert similarity2 < 0.1  # Should be very dissimilar
    
    def test_pattern_morphing_visualization(self, comparison):
        """Test pattern morphing visualization"""
        # Add two patterns
        x = np.linspace(0, 2*np.pi, 50)
        comparison.add_pattern('pattern1', np.sin(x))
        comparison.add_pattern('pattern2', np.cos(x))
        
        # Create morphing visualization
        fig = comparison.create_pattern_morphing_visualization('pattern1', 'pattern2', n_steps=5)
        
        assert fig is not None
        assert hasattr(fig, 'frames')
        assert len(fig.frames) == 5
        assert 'pattern1_to_pattern2' in comparison.morphing_data
        assert len(comparison.morphing_data['pattern1_to_pattern2']) == 5
    
    def test_pattern_morphing_invalid_patterns(self, comparison):
        """Test morphing with invalid pattern IDs"""
        with pytest.raises(ValueError):
            comparison.create_pattern_morphing_visualization('invalid1', 'invalid2')
    
    def test_difference_highlighting_visualization(self, comparison, sample_patterns):
        """Test difference highlighting visualization"""
        # Add patterns
        for pattern_id, data in list(sample_patterns.items())[:3]:
            comparison.add_pattern(pattern_id, data)
        
        # Create visualization
        fig = comparison.create_difference_highlighting_visualization()
        
        assert fig is not None
        assert len(comparison.difference_highlights) > 0
        
        # Check difference data structure
        for diff_key, diff_data in comparison.difference_highlights.items():
            assert 'differences' in diff_data
            assert 'abs_differences' in diff_data
            assert 'max_diff_idx' in diff_data
            assert 'max_diff_value' in diff_data
    
    def test_statistical_comparison_dashboard(self, comparison, sample_patterns):
        """Test statistical comparison dashboard"""
        # Add patterns
        for pattern_id, data in sample_patterns.items():
            comparison.add_pattern(pattern_id, data)
        
        # Calculate metrics
        comparison.calculate_statistical_comparison_metrics()
        
        # Create dashboard
        fig = comparison.create_statistical_comparison_dashboard()
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert len(fig.data) > 0
    
    def test_export_comparison_results_json(self, comparison, sample_patterns):
        """Test JSON export"""
        # Add patterns and calculate metrics
        for pattern_id, data in sample_patterns.items():
            comparison.add_pattern(pattern_id, data)
        comparison.calculate_similarity_metrics()
        comparison.calculate_statistical_comparison_metrics()
        
        # Export as JSON
        json_export = comparison.export_comparison_results('json')
        
        assert isinstance(json_export, str)
        
        # Verify JSON structure
        data = json.loads(json_export)
        assert 'metadata' in data
        assert 'similarity_metrics' in data
        assert 'statistical_metrics' in data
        assert data['metadata']['n_patterns'] == 4
    
    def test_export_comparison_results_csv(self, comparison, sample_patterns):
        """Test CSV export"""
        # Add patterns and calculate metrics
        for pattern_id, data in list(sample_patterns.items())[:2]:
            comparison.add_pattern(pattern_id, data)
        comparison.calculate_statistical_comparison_metrics()
        
        # Export as CSV
        csv_export = comparison.export_comparison_results('csv')
        
        assert isinstance(csv_export, str)
        assert 'pattern_pair' in csv_export
        assert 'pearson_correlation' in csv_export
        assert 'rmse' in csv_export
    
    def test_export_comparison_results_html(self, comparison, sample_patterns):
        """Test HTML export"""
        # Add patterns and calculate metrics
        for pattern_id, data in list(sample_patterns.items())[:2]:
            comparison.add_pattern(pattern_id, data)
        comparison.calculate_statistical_comparison_metrics()
        
        # Export as HTML
        html_export = comparison.export_comparison_results('html')
        
        assert isinstance(html_export, str)
        assert '<html>' in html_export
        assert '<table>' in html_export
        assert 'Pattern Comparison Report' in html_export
    
    def test_inherited_functionality(self, comparison, sample_patterns):
        """Test that inherited functionality still works"""
        # Add patterns
        for pattern_id, data in sample_patterns.items():
            comparison.add_pattern(pattern_id, data)
        
        # Test inherited methods
        comparison.calculate_similarity_metrics()
        assert len(comparison.similarity_metrics) > 0
        
        # Test inherited visualizations
        fig1 = comparison.create_side_by_side_visualization()
        assert fig1 is not None
        
        fig2 = comparison.create_pattern_overlay()
        assert fig2 is not None
        
        fig3 = comparison.create_correlation_heatmap()
        assert fig3 is not None
    
    def test_empty_patterns(self, comparison):
        """Test behavior with no patterns"""
        metrics = comparison.calculate_statistical_comparison_metrics()
        assert metrics == {}
        
        fig = comparison.create_statistical_comparison_dashboard()
        assert fig is not None  # Should still create empty dashboard
    
    def test_single_pattern(self, comparison):
        """Test behavior with single pattern"""
        comparison.add_pattern('single', np.sin(np.linspace(0, 2*np.pi, 100)))
        
        metrics = comparison.calculate_statistical_comparison_metrics()
        assert metrics == {}  # No pairs to compare
    
    def test_patterns_different_lengths(self, comparison):
        """Test comparison of patterns with different lengths"""
        comparison.add_pattern('short', np.sin(np.linspace(0, 2*np.pi, 50)))
        comparison.add_pattern('long', np.sin(np.linspace(0, 2*np.pi, 150)))
        
        metrics = comparison.calculate_statistical_comparison_metrics()
        assert len(metrics) == 1
        assert 'short_vs_long' in metrics
        
        # Should handle length mismatch gracefully
        assert metrics['short_vs_long']['rmse'] >= 0


class TestPatternComparisonIntegration:
    """Integration tests for pattern comparison workflow"""
    
    def test_complete_comparison_workflow(self):
        """Test complete pattern comparison workflow"""
        comparison = EnhancedPatternComparison()
        
        # Generate realistic patterns
        x = np.linspace(0, 4*np.pi, 200)
        patterns = {
            'trend_up': x/10 + np.sin(x),
            'trend_down': -x/10 + np.sin(x),
            'volatile': np.sin(x) + 0.5*np.sin(3*x) + np.random.normal(0, 0.1, len(x)),
            'stable': np.sin(x) * 0.5
        }
        
        # Add all patterns
        for pid, data in patterns.items():
            comparison.add_pattern(pid, data)
        
        # Calculate all metrics
        sim_metrics = comparison.calculate_similarity_metrics()
        stat_metrics = comparison.calculate_statistical_comparison_metrics()
        
        # Verify metrics calculated
        assert len(sim_metrics) > 0
        assert len(stat_metrics) > 0
        
        # Create all visualizations
        figs = []
        figs.append(comparison.create_side_by_side_visualization())
        figs.append(comparison.create_pattern_morphing_visualization('trend_up', 'trend_down'))
        figs.append(comparison.create_difference_highlighting_visualization())
        figs.append(comparison.create_statistical_comparison_dashboard())
        
        # Verify all figures created
        assert all(fig is not None for fig in figs)
        
        # Export results
        exports = {
            'json': comparison.export_comparison_results('json'),
            'csv': comparison.export_comparison_results('csv'),
            'html': comparison.export_comparison_results('html')
        }
        
        # Verify all exports successful
        assert all(export is not None and len(export) > 0 for export in exports.values())
    
    def test_performance_with_many_patterns(self):
        """Test performance with many patterns"""
        comparison = EnhancedPatternComparison()
        
        # Add 10 patterns
        x = np.linspace(0, 2*np.pi, 100)
        for i in range(10):
            pattern = np.sin(x + i*np.pi/5) + np.random.normal(0, 0.05, len(x))
            comparison.add_pattern(f'pattern_{i}', pattern)
        
        # Calculate metrics (should complete reasonably fast)
        import time
        start = time.time()
        
        comparison.calculate_similarity_metrics()
        comparison.calculate_statistical_comparison_metrics()
        
        elapsed = time.time() - start
        assert elapsed < 5.0  # Should complete within 5 seconds
        
        # Should have correct number of comparisons
        n_patterns = 10
        expected_pairs = n_patterns * (n_patterns - 1) // 2
        assert len(comparison.statistical_metrics) == expected_pairs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
