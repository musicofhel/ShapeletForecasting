"""
Test suite for Forecast Visualization component.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go

from src.dashboard.visualizations.forecast_view import ForecastVisualization
from src.dashboard.pattern_classifier import PatternClassifier
from src.dashboard.pattern_predictor import PatternPredictor


class TestForecastVisualization:
    """Test cases for ForecastVisualization class."""
    
    @pytest.fixture
    def forecast_viz(self):
        """Create ForecastVisualization instance."""
        return ForecastVisualization()
    
    @pytest.fixture
    def sample_price_data(self):
        """Generate sample price data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        base_price = 100
        returns = np.random.normal(0.001, 0.02, 100)
        prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
            'high': prices * (1 + np.random.uniform(0, 0.02, 100)),
            'low': prices * (1 + np.random.uniform(-0.02, 0, 100)),
            'close': prices
        }, index=dates)
        
        return df
    
    @pytest.fixture
    def sample_patterns(self):
        """Generate sample pattern data."""
        patterns = [
            {
                'pattern_type': 'Head and Shoulders',
                'start_idx': 10,
                'end_idx': 25,
                'start_time': 0,
                'duration': 15,
                'confidence': 0.85
            },
            {
                'pattern_type': 'Double Bottom',
                'start_idx': 30,
                'end_idx': 45,
                'start_time': 20,
                'duration': 15,
                'confidence': 0.75
            },
            {
                'pattern_type': 'Bull Flag',
                'start_idx': 50,
                'end_idx': 60,
                'start_time': 40,
                'duration': 10,
                'confidence': 0.90
            }
        ]
        return patterns
    
    @pytest.fixture
    def sample_predictions(self):
        """Generate sample predictions."""
        return [
            {'pattern_type': 'Ascending Triangle', 'probability': 0.75},
            {'pattern_type': 'Bull Flag', 'probability': 0.60},
            {'pattern_type': 'Cup and Handle', 'probability': 0.45},
            {'pattern_type': 'Double Bottom', 'probability': 0.30}
        ]
    
    @pytest.fixture
    def sample_confidence_bands(self):
        """Generate sample confidence bands."""
        time_steps = 20
        return {
            'Ascending Triangle': {
                'upper': [0.75 * (1 - t/time_steps) + 0.1 for t in range(time_steps)],
                'lower': [0.75 * (1 - t/time_steps) - 0.1 for t in range(time_steps)]
            },
            'Bull Flag': {
                'upper': [0.60 * (1 - t/time_steps) + 0.08 for t in range(time_steps)],
                'lower': [0.60 * (1 - t/time_steps) - 0.08 for t in range(time_steps)]
            }
        }
    
    def test_initialization(self, forecast_viz):
        """Test proper initialization."""
        assert forecast_viz is not None
        assert hasattr(forecast_viz, 'pattern_colors')
        assert hasattr(forecast_viz, 'confidence_colors')
        assert len(forecast_viz.pattern_colors) > 0
        assert len(forecast_viz.confidence_colors) == 3
    
    def test_create_current_context_view(self, forecast_viz, sample_price_data, sample_patterns):
        """Test current context visualization."""
        current_pattern = sample_patterns[2]  # Bull Flag
        historical_patterns = sample_patterns[:2]
        
        fig = forecast_viz.create_current_context_view(
            current_pattern=current_pattern,
            historical_patterns=historical_patterns,
            price_data=sample_price_data
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text == 'Pattern Forecast Context'
        assert fig.layout.height == 800
        
        # Check for candlestick trace
        candlestick_found = any(isinstance(trace, go.Candlestick) for trace in fig.data)
        assert candlestick_found
        
        # Check for pattern history bars
        bar_traces = [trace for trace in fig.data if isinstance(trace, go.Bar)]
        assert len(bar_traces) >= len(historical_patterns)
    
    def test_create_prediction_visualization(self, forecast_viz, sample_predictions, sample_confidence_bands):
        """Test prediction visualization with confidence bands."""
        fig = forecast_viz.create_prediction_visualization(
            predictions=sample_predictions,
            confidence_bands=sample_confidence_bands,
            time_horizon=20
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text == 'Pattern Predictions with Confidence Bands'
        
        # Check for prediction lines (top 3)
        line_traces = [trace for trace in fig.data if trace.mode and 'lines' in trace.mode]
        assert len(line_traces) >= 3
        
        # Check for confidence bands
        fill_traces = [trace for trace in fig.data if hasattr(trace, 'fill') and trace.fill == 'toself']
        assert len(fill_traces) >= 2  # At least 2 confidence bands
    
    def test_create_scenario_analysis(self, forecast_viz):
        """Test scenario analysis visualization."""
        scenarios = [
            {
                'name': 'Bullish Scenario',
                'pattern_sequence': ['Bull Flag', 'Ascending Triangle'],
                'price_path': [100, 102, 105, 108, 110, 115],
                'probability': 0.6
            },
            {
                'name': 'Neutral Scenario',
                'pattern_sequence': ['Consolidation', 'Range Bound'],
                'price_path': [100, 101, 100, 99, 100, 101],
                'probability': 0.3
            },
            {
                'name': 'Bearish Scenario',
                'pattern_sequence': ['Bear Flag', 'Double Top'],
                'price_path': [100, 98, 95, 92, 90, 88],
                'probability': 0.1
            }
        ]
        
        fig = forecast_viz.create_scenario_analysis(
            scenarios=scenarios,
            current_price=100
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text == 'Multi-Scenario Pattern Predictions'
        
        # Check for scenario lines
        scatter_traces = [trace for trace in fig.data if isinstance(trace, go.Scatter)]
        assert len(scatter_traces) >= len(scenarios)
        
        # Check for probability bars
        bar_traces = [trace for trace in fig.data if isinstance(trace, go.Bar)]
        assert len(bar_traces) >= 1
    
    def test_create_historical_accuracy_overlay(self, forecast_viz):
        """Test historical accuracy overlay visualization."""
        # Create sample historical data
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        
        predictions_history = pd.DataFrame({
            'pattern_type': np.random.choice(['Bull Flag', 'Bear Flag', 'Double Top'], 50),
            'probability': np.random.uniform(0.3, 0.9, 50),
            'correct': np.random.choice([True, False], 50, p=[0.7, 0.3])
        }, index=dates)
        
        actual_patterns = pd.DataFrame({
            'pattern_type': np.random.choice(['Bull Flag', 'Bear Flag', 'Double Top'], 20),
        }, index=dates[::2][:20])  # Every other day
        
        fig = forecast_viz.create_historical_accuracy_overlay(
            predictions_history=predictions_history,
            actual_patterns=actual_patterns
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text == 'Historical Prediction Accuracy'
        assert fig.layout.height == 800
        
        # Check for both predicted and actual markers
        marker_traces = [trace for trace in fig.data if hasattr(trace, 'mode') and 'markers' in trace.mode]
        assert len(marker_traces) >= 2
    
    def test_confidence_calibration_plot(self, forecast_viz):
        """Test confidence calibration plot."""
        # Create sample prediction data
        n_samples = 1000
        predictions = pd.DataFrame({
            'probability': np.random.uniform(0, 1, n_samples),
            'correct': np.random.choice([True, False], n_samples)
        })
        
        # Make it somewhat calibrated
        predictions.loc[predictions['probability'] > 0.7, 'correct'] = np.random.choice(
            [True, False], sum(predictions['probability'] > 0.7), p=[0.75, 0.25]
        )
        
        fig = forecast_viz.create_confidence_calibration_plot(predictions)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Perfect calibration line + actual
        assert fig.layout.title.text == 'Prediction Confidence Calibration'
        
        # Check for diagonal reference line
        diagonal_found = any(
            trace.x is not None and trace.y is not None and 
            np.array_equal(trace.x, [0, 1]) and np.array_equal(trace.y, [0, 1])
            for trace in fig.data
        )
        assert diagonal_found
    
    def test_empty_data_handling(self, forecast_viz):
        """Test handling of empty data."""
        # Empty DataFrames
        empty_df = pd.DataFrame()
        
        # Should not raise errors
        fig = forecast_viz.create_historical_accuracy_overlay(
            predictions_history=empty_df,
            actual_patterns=empty_df
        )
        assert isinstance(fig, go.Figure)
    
    def test_pattern_color_consistency(self, forecast_viz):
        """Test that pattern colors are consistent across visualizations."""
        pattern_type = 'Head and Shoulders'
        
        # Check color exists
        assert pattern_type in forecast_viz.pattern_colors
        color = forecast_viz.pattern_colors[pattern_type]
        
        # Verify it's a valid color string
        assert isinstance(color, str)
        assert color.startswith('#') or color.startswith('rgb')
    
    def test_confidence_level_colors(self, forecast_viz):
        """Test confidence level color mapping."""
        assert 'high' in forecast_viz.confidence_colors
        assert 'medium' in forecast_viz.confidence_colors
        assert 'low' in forecast_viz.confidence_colors
        
        # All should be RGBA colors with transparency
        for level, color in forecast_viz.confidence_colors.items():
            assert color.startswith('rgba')
            assert '0.3' in color  # Check transparency
    
    def test_rolling_accuracy_calculation(self, forecast_viz):
        """Test rolling accuracy calculation."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        predictions = pd.DataFrame({
            'pattern_type_x': np.random.choice(['A', 'B', 'C'], 100),
            'pattern_type_y': np.random.choice(['A', 'B', 'C'], 100)
        }, index=dates)
        
        actuals = pd.DataFrame({
            'pattern_type': predictions['pattern_type_y']
        }, index=dates)
        
        accuracy = forecast_viz._calculate_rolling_accuracy(predictions, actuals, window=10)
        
        assert isinstance(accuracy, pd.Series)
        assert len(accuracy) == len(predictions)
        assert accuracy.min() >= 0
        assert accuracy.max() <= 1
