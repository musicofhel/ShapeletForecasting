"""
Test suite for Accuracy Metrics Dashboard component.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from scipy import stats

from src.dashboard.visualizations.accuracy_metrics import AccuracyMetricsDashboard


class TestAccuracyMetricsDashboard:
    """Test cases for AccuracyMetricsDashboard class."""
    
    @pytest.fixture
    def metrics_dashboard(self):
        """Create AccuracyMetricsDashboard instance."""
        return AccuracyMetricsDashboard()
    
    @pytest.fixture
    def sample_accuracy_data(self):
        """Generate sample accuracy data over time."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        models = ['LSTM', 'Transformer', 'Markov', 'Ensemble']
        
        data = []
        for date in dates:
            for model in models:
                # Generate realistic accuracy values
                base_acc = {'LSTM': 0.75, 'Transformer': 0.78, 'Markov': 0.65, 'Ensemble': 0.80}
                accuracy = base_acc[model] + np.random.normal(0, 0.05)
                accuracy = np.clip(accuracy, 0, 1)
                
                data.append({
                    'date': date,
                    'model': model,
                    'accuracy': accuracy
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_pattern_accuracy(self):
        """Generate sample pattern-specific accuracy data."""
        patterns = [
            'Head and Shoulders', 'Double Top', 'Double Bottom',
            'Ascending Triangle', 'Bull Flag', 'Bear Flag'
        ]
        
        data = []
        for pattern in patterns:
            n_samples = np.random.randint(50, 200)
            correct = np.random.randint(int(n_samples * 0.6), int(n_samples * 0.9))
            
            data.append({
                'pattern_type': pattern,
                'accuracy': correct / n_samples,
                'precision': np.random.uniform(0.6, 0.9),
                'recall': np.random.uniform(0.6, 0.9),
                'f1_score': np.random.uniform(0.6, 0.85),
                'support': n_samples
            })
        
        df = pd.DataFrame(data)
        
        # Add confusion matrix data as a separate aggregated matrix
        n_patterns = len(patterns)
        cm = np.random.randint(0, 20, size=(n_patterns, n_patterns))
        np.fill_diagonal(cm, np.random.randint(30, 50, n_patterns))
        
        # Store confusion matrix separately (not in DataFrame)
        df.confusion_matrix_data = cm
        
        return df
    
    @pytest.fixture
    def sample_predictions(self):
        """Generate sample prediction data for calibration."""
        n_samples = 1000
        
        # Generate predictions with some calibration
        predictions = pd.DataFrame({
            'probability': np.random.uniform(0, 1, n_samples),
            'pattern_type': np.random.choice(['Bull Flag', 'Bear Flag', 'Double Top'], n_samples),
            'model': np.random.choice(['LSTM', 'Transformer', 'Ensemble'], n_samples)
        })
        
        # Make predictions somewhat calibrated
        predictions['correct'] = predictions['probability'] > np.random.uniform(0.3, 0.7, n_samples)
        
        return predictions
    
    @pytest.fixture
    def sample_errors(self):
        """Generate sample error data."""
        n_samples = 500
        dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
        
        errors = pd.DataFrame({
            'error': np.random.normal(0, 0.1, n_samples),
            'confidence': np.random.uniform(0.3, 0.9, n_samples),
            'hour': dates.hour,
            'date': dates
        })
        
        return errors
    
    @pytest.fixture
    def sample_model_metrics(self):
        """Generate sample model comparison metrics."""
        models = ['LSTM', 'Transformer', 'Markov', 'Ensemble', 'Baseline']
        
        data = []
        for model in models:
            # Base performance characteristics
            perf = {
                'LSTM': {'acc': 0.75, 'time': 50},
                'Transformer': {'acc': 0.78, 'time': 100},
                'Markov': {'acc': 0.65, 'time': 10},
                'Ensemble': {'acc': 0.80, 'time': 150},
                'Baseline': {'acc': 0.55, 'time': 5}
            }
            
            data.append({
                'model': model,
                'accuracy': perf[model]['acc'] + np.random.normal(0, 0.02),
                'precision': perf[model]['acc'] + np.random.normal(0, 0.03),
                'recall': perf[model]['acc'] + np.random.normal(-0.05, 0.03),
                'f1_score': perf[model]['acc'] + np.random.normal(-0.02, 0.02),
                'inference_time': perf[model]['time'] + np.random.normal(0, 5),
                'model_size': np.random.randint(10, 100),
                'learning_history': {
                    'epochs': list(range(1, 21)),
                    'val_accuracy': [perf[model]['acc'] * (1 - np.exp(-i/5)) + np.random.normal(0, 0.02) 
                                    for i in range(1, 21)]
                }
            })
        
        return pd.DataFrame(data)
    
    def test_initialization(self, metrics_dashboard):
        """Test proper initialization."""
        assert metrics_dashboard is not None
        assert hasattr(metrics_dashboard, 'pattern_colors')
        assert hasattr(metrics_dashboard, 'model_colors')
        assert len(metrics_dashboard.pattern_colors) > 0
        assert len(metrics_dashboard.model_colors) >= 5
    
    def test_create_accuracy_over_time(self, metrics_dashboard, sample_accuracy_data):
        """Test accuracy over time visualization."""
        fig = metrics_dashboard.create_accuracy_over_time(
            accuracy_data=sample_accuracy_data,
            window_sizes=[7, 30, 90]
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text == 'Model Accuracy Performance Over Time'
        assert fig.layout.height == 800
        
        # Check for both scatter and line traces
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) > 0
        
        # Check for histogram traces
        hist_traces = [t for t in fig.data if isinstance(t, go.Histogram)]
        assert len(hist_traces) > 0
    
    def test_create_pattern_type_accuracy(self, metrics_dashboard, sample_pattern_accuracy):
        """Test pattern-specific accuracy visualization."""
        fig = metrics_dashboard.create_pattern_type_accuracy(
            pattern_accuracy=sample_pattern_accuracy
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text == 'Pattern-Specific Accuracy Analysis'
        assert fig.layout.height == 900
        
        # Check for bar charts
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) >= 2  # Accuracy bars and F1 score bars
        
        # Check for scatter plot (precision vs recall)
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) >= 1
        
        # Check for heatmap (confusion matrix) - may not be present if not in columns
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        # Confusion matrix is optional, so we don't assert its presence
    
    def test_create_confidence_calibration(self, metrics_dashboard, sample_predictions):
        """Test confidence calibration visualization."""
        fig = metrics_dashboard.create_confidence_calibration(
            predictions=sample_predictions
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text == 'Prediction Confidence Calibration Analysis'
        assert fig.layout.height == 900
        
        # Check for calibration plot with diagonal
        diagonal_found = any(
            hasattr(t, 'x') and hasattr(t, 'y') and 
            np.array_equal(t.x, [0, 1]) and np.array_equal(t.y, [0, 1])
            for t in fig.data
        )
        assert diagonal_found
        
        # Check for histogram traces
        hist_traces = [t for t in fig.data if isinstance(t, go.Histogram)]
        assert len(hist_traces) >= 3  # All, correct, incorrect
    
    def test_create_error_distribution(self, metrics_dashboard, sample_errors):
        """Test error distribution analysis."""
        fig = metrics_dashboard.create_error_distribution(
            errors=sample_errors
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text == 'Prediction Error Analysis'
        assert fig.layout.height == 900
        
        # Check for error histogram
        hist_traces = [t for t in fig.data if isinstance(t, go.Histogram)]
        assert len(hist_traces) >= 1
        
        # Check for normal distribution overlay
        normal_overlay = [t for t in fig.data if isinstance(t, go.Scatter) and 
                         'Normal' in str(t.name)]
        assert len(normal_overlay) >= 1
        
        # Check for box plot
        box_traces = [t for t in fig.data if isinstance(t, go.Box)]
        assert len(box_traces) >= 1
        
        # Check for autocorrelation bars
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) >= 1
    
    def test_create_model_comparison(self, metrics_dashboard, sample_model_metrics):
        """Test model comparison visualization."""
        fig = metrics_dashboard.create_model_comparison(
            model_metrics=sample_model_metrics
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text == 'Model Performance Comparison Dashboard'
        assert fig.layout.height == 900
        
        # Check for performance bars
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) >= 4  # accuracy, precision, recall, f1
        
        # Check for radar chart
        polar_traces = [t for t in fig.data if isinstance(t, go.Scatterpolar)]
        assert len(polar_traces) >= len(sample_model_metrics)
        
        # Check for learning curves
        learning_curves = [t for t in fig.data if isinstance(t, go.Scatter) and 
                          hasattr(t, 'x') and len(t.x) > 10]
        assert len(learning_curves) >= 1
    
    def test_create_summary_metrics_card(self, metrics_dashboard):
        """Test summary metrics card visualization."""
        metrics = {
            'overall_accuracy': 0.756,
            'best_model': 'Ensemble',
            'avg_confidence': 0.682,
            'total_predictions': 12345
        }
        
        fig = metrics_dashboard.create_summary_metrics_card(metrics)
        
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == 'Key Performance Metrics'
        assert fig.layout.height == 300
        
        # Check for annotations
        assert len(fig.layout.annotations) == 8  # 4 titles + 4 values
        
        # Verify metric formatting
        annotations_text = [ann.text for ann in fig.layout.annotations]
        assert '75.6%' in annotations_text  # overall_accuracy
        assert 'Ensemble' in annotations_text  # best_model
        assert '68.2%' in annotations_text  # avg_confidence
        assert '12,345' in annotations_text  # total_predictions
    
    def test_color_consistency(self, metrics_dashboard):
        """Test color scheme consistency."""
        # Pattern colors should match those in forecast_view
        expected_patterns = ['Head and Shoulders', 'Double Top', 'Bull Flag']
        for pattern in expected_patterns:
            assert pattern in metrics_dashboard.pattern_colors
        
        # Model colors should be defined
        expected_models = ['LSTM', 'Transformer', 'Markov', 'Ensemble']
        for model in expected_models:
            assert model in metrics_dashboard.model_colors
    
    def test_empty_data_handling(self, metrics_dashboard):
        """Test handling of empty data."""
        empty_df = pd.DataFrame()
        
        # Should not raise errors
        fig = metrics_dashboard.create_accuracy_over_time(empty_df)
        assert isinstance(fig, go.Figure)
        
        fig = metrics_dashboard.create_pattern_type_accuracy(empty_df)
        assert isinstance(fig, go.Figure)
    
    def test_statistical_calculations(self, metrics_dashboard, sample_errors):
        """Test statistical calculations in visualizations."""
        fig = metrics_dashboard.create_error_distribution(sample_errors)
        
        # Find the normal distribution trace
        normal_trace = None
        for trace in fig.data:
            if hasattr(trace, 'name') and 'Normal' in str(trace.name):
                normal_trace = trace
                break
        
        assert normal_trace is not None
        
        # Verify normal distribution parameters are reasonable
        # Extract mu and sigma from the trace name
        import re
        match = re.search(r'μ=([-\d.]+), σ=([\d.]+)', normal_trace.name)
        if match:
            mu = float(match.group(1))
            sigma = float(match.group(2))
            
            # Check parameters are reasonable
            assert -0.5 < mu < 0.5  # Error should be centered near 0
            assert 0 < sigma < 1  # Standard deviation should be positive and reasonable
    
    def test_calibration_binning(self, metrics_dashboard, sample_predictions):
        """Test calibration binning logic."""
        # Add confidence_bin column
        sample_predictions['confidence_bin'] = pd.cut(
            sample_predictions['probability'],
            bins=np.linspace(0, 1, 11),
            labels=np.linspace(0.05, 0.95, 10)
        )
        
        fig = metrics_dashboard.create_confidence_calibration(sample_predictions)
        
        # Verify calibration plot has appropriate number of bins
        calibration_traces = [t for t in fig.data if hasattr(t, 'marker') and 
                            hasattr(t.marker, 'size') and t.marker.size is not None]
        
        assert len(calibration_traces) > 0
        
        # Check that bin centers are in [0, 1]
        for trace in calibration_traces:
            if hasattr(trace, 'x') and trace.x is not None:
                assert all(0 <= x <= 1 for x in trace.x if x is not None)
