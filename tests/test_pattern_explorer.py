"""
Tests for Pattern Explorer Tool
"""

import pytest
import numpy as np
import pandas as pd
import json
import tempfile
import os
from pathlib import Path

from src.dashboard.tools.pattern_explorer import PatternExplorer, PatternStats


class TestPatternExplorer:
    """Test suite for PatternExplorer class"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def explorer(self, temp_data_dir):
        """Create PatternExplorer instance with temp directory"""
        return PatternExplorer(data_dir=temp_data_dir)
    
    def test_initialization(self, explorer):
        """Test PatternExplorer initialization"""
        assert explorer.data_dir.exists()
        assert explorer.classifier is not None
        assert explorer.analyzer is not None
        assert explorer.matcher is not None
    
    def test_load_pattern_library(self, explorer):
        """Test loading pattern library"""
        library = explorer.load_pattern_library()
        assert isinstance(library, dict)
        assert len(library) > 0
        
        # Check required keys
        for pattern_id, info in library.items():
            assert 'name' in info
            assert 'type' in info
            assert 'reliability' in info
            assert 'avg_return' in info
    
    def test_browse_patterns(self, explorer):
        """Test pattern browsing functionality"""
        patterns_df = explorer.browse_patterns()
        
        assert isinstance(patterns_df, pd.DataFrame)
        assert len(patterns_df) > 0
        
        # Check required columns
        required_cols = ['Pattern', 'Type', 'Frequency', 'Reliability', 'Avg Return']
        for col in required_cols:
            assert col in patterns_df.columns
    
    def test_browse_patterns_with_filter(self, explorer):
        """Test pattern browsing with type filter"""
        patterns_df = explorer.browse_patterns(pattern_type='reversal')
        
        if len(patterns_df) > 0:
            assert all(patterns_df['Type'] == 'reversal')
    
    def test_get_pattern_statistics(self, explorer):
        """Test getting pattern statistics"""
        stats = explorer.get_pattern_statistics("head_and_shoulders")
        
        assert isinstance(stats, dict)
        assert 'pattern_name' in stats
        assert 'type' in stats
        assert 'reliability' in stats
        assert 'avg_return' in stats
    
    def test_find_similar_patterns(self, explorer):
        """Test finding similar patterns"""
        similar = explorer.find_similar_patterns("head_and_shoulders")
        
        assert isinstance(similar, list)
        for pattern in similar:
            assert 'pattern_id' in pattern
            assert 'name' in pattern
            assert 'similarity' in pattern
            assert 0 <= pattern['similarity'] <= 1
    
    def test_track_pattern_evolution(self, explorer):
        """Test pattern evolution tracking"""
        evolution = explorer.track_pattern_evolution("AAPL", "head_and_shoulders")
        
        assert isinstance(evolution, dict)
        assert 'dates' in evolution
        assert 'frequency' in evolution
        assert 'reliability' in evolution
        assert 'avg_return' in evolution
        
        assert len(evolution['dates']) == len(evolution['frequency'])
    
    def test_backtest_pattern(self, explorer):
        """Test pattern backtesting"""
        results = explorer.backtest_pattern("AAPL", "head_and_shoulders")
        
        assert isinstance(results, dict)
        assert 'total_trades' in results
        assert 'win_rate' in results
        assert 'total_return' in results
        
        assert 0 <= results['win_rate'] <= 1
    
    def test_create_pattern_browser_dashboard(self, explorer):
        """Test dashboard creation"""
        fig = explorer.create_pattern_browser_dashboard()
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert len(fig.data) > 0
    
    def test_export_pattern_library(self, explorer):
        """Test pattern library export"""
        # Test JSON export
        json_export = explorer.export_pattern_library('json')
        assert isinstance(json_export, str)
        
        # Verify JSON is valid
        parsed = json.loads(json_export)
        assert isinstance(parsed, dict)
        
        # Test CSV export
        csv_export = explorer.export_pattern_library('csv')
        assert isinstance(csv_export, str)
        assert 'pattern_id' in csv_export
    
    def test_pattern_stats_dataclass(self):
        """Test PatternStats dataclass"""
        stats = PatternStats(
            pattern_type="test_pattern",
            frequency=10,
            avg_duration=20.5,
            avg_confidence=0.85,
            success_rate=0.80,
            avg_return=0.05,
            volatility=0.15,
            reliability_score=0.82
        )
        
        assert stats.pattern_type == "test_pattern"
        assert stats.frequency == 10
        assert stats.avg_confidence == 0.85
    
    def test_empty_pattern_library(self, temp_data_dir):
        """Test behavior with empty pattern library"""
        # Create empty library
        empty_lib = {}
        lib_path = Path(temp_data_dir) / "pattern_library" / "pattern_library.json"
        lib_path.parent.mkdir(exist_ok=True)
        
        with open(lib_path, 'w') as f:
            json.dump(empty_lib, f)
        
        explorer = PatternExplorer(data_dir=temp_data_dir)
        library = explorer.load_pattern_library()
        
        # Should create default library
        assert len(library) > 0
    
    def test_invalid_pattern_type(self, explorer):
        """Test handling of invalid pattern type"""
        stats = explorer.get_pattern_statistics("invalid_pattern")
        assert stats == {}
        
        similar = explorer.find_similar_patterns("invalid_pattern")
        assert similar == []


class TestPatternExplorerIntegration:
    """Integration tests for PatternExplorer"""
    
    def test_full_workflow(self):
        """Test complete workflow"""
        with tempfile.TemporaryDirectory() as tmpdir:
            explorer = PatternExplorer(data_dir=tmpdir)
            
            # Browse patterns
            patterns_df = explorer.browse_patterns()
            assert len(patterns_df) > 0
            
            # Get stats for first pattern
            first_pattern = patterns_df.iloc[0]['Pattern'].lower().replace(' ', '_')
            stats = explorer.get_pattern_statistics(first_pattern)
            assert len(stats) > 0
            
            # Find similar patterns
            similar = explorer.find_similar_patterns(first_pattern)
            assert isinstance(similar, list)
            
            # Create dashboard
            fig = explorer.create_pattern_browser_dashboard()
            assert fig is not None
    
    def test_pattern_backtesting_workflow(self):
        """Test pattern backtesting workflow"""
        with tempfile.TemporaryDirectory() as tmpdir:
            explorer = PatternExplorer(data_dir=tmpdir)
            
            # Backtest a pattern
            results = explorer.backtest_pattern("AAPL", "head_and_shoulders")
            
            # Verify results structure
            required_keys = [
                'total_trades', 'winning_trades', 'losing_trades',
                'win_rate', 'avg_win', 'avg_loss', 'profit_factor',
                'max_drawdown', 'sharpe_ratio', 'total_return'
            ]
            
            for key in required_keys:
                assert key in results
            
            # Verify reasonable values
            assert results['total_trades'] >= 0
            assert 0 <= results['win_rate'] <= 1
            assert results['total_return'] >= -1  # Allow for losses


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
