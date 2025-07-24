"""
Plotly Dash Forecasting Dashboard Application

This module implements a comprehensive forecasting dashboard with pattern analysis,
prediction visualization, and performance metrics.
"""

import dash
from dash import Dash, html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import time
from functools import lru_cache
import threading
from queue import Queue

# Import custom modules
from src.dashboard.layouts.forecast_layout import create_forecast_layout
from src.dashboard.callbacks.prediction_callbacks import register_prediction_callbacks
from src.dashboard.pattern_predictor import PatternPredictor
from src.dashboard.pattern_matcher import PatternMatcher
from src.dashboard.wavelet_sequence_analyzer import WaveletSequenceAnalyzer
from src.dashboard.visualizations.sequence_view import PatternSequenceVisualizer
from src.dashboard.pattern_classifier import PatternClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performance monitoring
class PerformanceMonitor:
    """Monitor dashboard performance metrics"""
    
    def __init__(self):
        self.callback_times = []
        self.memory_usage = []
        self.active_sessions = 0
        self.data_points_processed = 0
        
    def record_callback_time(self, duration: float):
        """Record callback execution time"""
        self.callback_times.append(duration)
        # Keep only last 1000 measurements
        if len(self.callback_times) > 1000:
            self.callback_times.pop(0)
    
    def get_average_callback_time(self) -> float:
        """Get average callback execution time"""
        if not self.callback_times:
            return 0
        return np.mean(self.callback_times)
    
    def check_performance(self) -> Dict[str, Any]:
        """Check if performance meets criteria"""
        avg_callback_time = self.get_average_callback_time()
        return {
            'meets_criteria': avg_callback_time < 0.5,  # 500ms
            'avg_callback_time': avg_callback_time,
            'active_sessions': self.active_sessions,
            'data_points_processed': self.data_points_processed
        }

# Initialize performance monitor
performance_monitor = PerformanceMonitor()

# Create Dash app with Bootstrap theme
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ],
    suppress_callback_exceptions=True,
    title="Wavelet Forecast Dashboard"
)

# App is configured via constructor parameters

# Cache for expensive computations
@lru_cache(maxsize=128)
def compute_pattern_predictions(data_hash: str, pattern_length: int) -> Dict[str, Any]:
    """Compute pattern predictions with caching"""
    # This would use actual data in production
    # For now, return mock predictions
    return {
        'next_pattern': 'pattern_A',
        'confidence': 0.85,
        'alternatives': [
            {'pattern': 'pattern_B', 'probability': 0.10},
            {'pattern': 'pattern_C', 'probability': 0.05}
        ]
    }

# Data management class
class DataManager:
    """Manage data loading and caching"""
    
    def __init__(self):
        self.data_cache = {}
        self.pattern_cache = {}
        self.lock = threading.Lock()
        
    def load_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load data with caching"""
        cache_key = f"{symbol}_{timeframe}"
        
        with self.lock:
            if cache_key in self.data_cache:
                return self.data_cache[cache_key]
            
            # Generate sample data for demo
            dates = pd.date_range(end=datetime.now(), periods=10000, freq='1H')
            data = pd.DataFrame({
                'timestamp': dates,
                'price': 100 + np.cumsum(np.random.randn(10000) * 0.5),
                'volume': np.random.randint(1000, 10000, 10000)
            })
            
            self.data_cache[cache_key] = data
            performance_monitor.data_points_processed += len(data)
            
        return data
    
    def get_patterns(self, symbol: str) -> List[Dict[str, Any]]:
        """Get detected patterns"""
        if symbol in self.pattern_cache:
            return self.pattern_cache[symbol]
        
        # Generate sample patterns
        patterns = []
        for i in range(20):
            patterns.append({
                'id': f'pattern_{i}',
                'name': f'Pattern {chr(65 + i % 26)}',
                'frequency': np.random.randint(5, 50),
                'avg_return': np.random.uniform(-2, 5),
                'confidence': np.random.uniform(0.6, 0.95)
            })
        
        self.pattern_cache[symbol] = patterns
        return patterns

# Initialize data manager
data_manager = DataManager()

# Create app layout
app.layout = create_forecast_layout()

# Register callbacks
register_prediction_callbacks(app, data_manager, performance_monitor)

# Additional callbacks for main functionality
@app.callback(
    [Output('main-time-series', 'figure'),
     Output('pattern-overlay-store', 'data')],
    [Input('symbol-dropdown', 'value'),
     Input('timeframe-dropdown', 'value'),
     Input('pattern-toggle', 'value'),
     Input('refresh-interval', 'n_intervals')]
)
def update_main_chart(symbol: str, timeframe: str, show_patterns: List[str], n_intervals: int):
    """Update main time series chart with pattern overlays"""
    start_time = time.time()
    
    try:
        # Load data
        df = data_manager.load_data(symbol, timeframe)
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=('Price with Pattern Overlays', 'Volume')
        )
        
        # Add price trace
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['price'],
                name='Price',
                line=dict(color='#1f77b4', width=1),
                hovertemplate='%{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add pattern overlays if enabled
        pattern_data = []
        if 'patterns' in show_patterns:
            # Add sample pattern overlays
            for i in range(5):
                start_idx = np.random.randint(0, len(df) - 100)
                end_idx = start_idx + np.random.randint(20, 100)
                
                pattern_segment = df.iloc[start_idx:end_idx]
                fig.add_trace(
                    go.Scatter(
                        x=pattern_segment['timestamp'],
                        y=pattern_segment['price'],
                        name=f'Pattern {chr(65 + i)}',
                        line=dict(width=3),
                        opacity=0.7,
                        hovertemplate=f'Pattern {chr(65 + i)}<br>%{{x}}<br>Price: $%{{y:.2f}}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                pattern_data.append({
                    'pattern_id': f'pattern_{i}',
                    'start': start_idx,
                    'end': end_idx,
                    'confidence': np.random.uniform(0.7, 0.95)
                })
        
        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['volume'],
                name='Volume',
                marker_color='rgba(128, 128, 128, 0.5)',
                hovertemplate='%{x}<br>Volume: %{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            margin=dict(l=50, r=50, t=50, b=50),
            hovermode='x unified',
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        # Record performance
        duration = time.time() - start_time
        performance_monitor.record_callback_time(duration)
        
        return fig, pattern_data
        
    except Exception as e:
        logger.error(f"Error updating main chart: {e}")
        return go.Figure(), []

@app.callback(
    Output('pattern-sequence-viz', 'figure'),
    [Input('pattern-overlay-store', 'data'),
     Input('sequence-length-slider', 'value'),
     Input('symbol-dropdown', 'value')]
)
def update_pattern_sequence(pattern_data: List[Dict], sequence_length: int, symbol: str):
    """Update pattern sequence visualization using PatternSequenceVisualizer"""
    start_time = time.time()
    
    try:
        # Initialize visualizer
        visualizer = PatternSequenceVisualizer()
        
        # Initialize pattern classifier for realistic pattern types
        classifier = PatternClassifier()
        
        # Generate realistic pattern sequence data
        base_time = datetime.now() - timedelta(days=7)
        patterns = []
        
        # Pattern types from our classifier
        pattern_types = [
            'head_shoulders', 'double_top', 'double_bottom',
            'triangle_ascending', 'triangle_descending',
            'flag_bull', 'flag_bear', 'wedge_rising', 'wedge_falling'
        ]
        
        # Create pattern sequence based on pattern_data or generate sample
        if pattern_data and len(pattern_data) > 0:
            # Use actual pattern data
            for i, pdata in enumerate(pattern_data[:sequence_length]):
                pattern_type = pattern_types[i % len(pattern_types)]
                patterns.append({
                    'id': pdata.get('pattern_id', f'P{i+1:03d}'),
                    'type': pattern_type,
                    'start_time': base_time + timedelta(hours=i*24),
                    'end_time': base_time + timedelta(hours=(i*24 + np.random.randint(4, 20))),
                    'confidence': pdata.get('confidence', 0.8),
                    'ticker': symbol
                })
        else:
            # Generate sample patterns
            for i in range(min(sequence_length, 10)):
                pattern_type = pattern_types[i % len(pattern_types)]
                duration = np.random.randint(4, 24)  # hours
                
                patterns.append({
                    'id': f'P{i+1:03d}',
                    'type': pattern_type,
                    'start_time': base_time + timedelta(hours=i*30),
                    'end_time': base_time + timedelta(hours=(i*30 + duration)),
                    'confidence': np.random.uniform(0.65, 0.95),
                    'ticker': symbol
                })
        
        # Generate transitions between patterns
        transitions = {}
        for i in range(len(patterns) - 1):
            trans_key = f"{patterns[i]['id']}_to_{patterns[i+1]['id']}"
            transitions[trans_key] = {
                'probability': np.random.uniform(0.3, 0.9)
            }
        
        # Create pattern timeline visualization
        fig = visualizer.create_pattern_timeline(
            patterns=patterns,
            transitions=transitions,
            show_probabilities=True,
            height=400
        )
        
        # Update title to include symbol
        fig.update_layout(
            title=f"Pattern Sequence Timeline - {symbol}",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        # Record performance
        duration = time.time() - start_time
        performance_monitor.record_callback_time(duration)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error updating pattern sequence: {e}")
        # Return empty figure with message
        visualizer = PatternSequenceVisualizer()
        return visualizer._create_empty_figure("Error loading pattern sequence")

@app.callback(
    Output('prediction-display', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('symbol-dropdown', 'value'),
     State('prediction-horizon-slider', 'value')]
)
def update_predictions(n_clicks: int, symbol: str, horizon: int):
    """Update prediction display"""
    if not n_clicks:
        raise PreventUpdate
    
    start_time = time.time()
    
    try:
        # Check for invalid inputs
        if not symbol or horizon is None:
            return dbc.Alert(
                "Please select a symbol and prediction horizon",
                color="warning",
                dismissable=True
            )
        
        # Generate predictions
        predictions = compute_pattern_predictions(symbol, horizon)
        
        # Create prediction cards
        cards = []
        
        # Main prediction card
        main_card = dbc.Card([
            dbc.CardHeader(html.H5("Next Pattern Prediction", className="mb-0")),
            dbc.CardBody([
                html.H2(predictions['next_pattern'], className="text-primary"),
                html.P(f"Confidence: {predictions['confidence']:.1%}", className="text-muted"),
                dbc.Progress(
                    value=predictions['confidence'] * 100,
                    color="success" if predictions['confidence'] > 0.8 else "warning",
                    className="mb-3"
                )
            ])
        ], className="mb-3")
        cards.append(main_card)
        
        # Alternative predictions
        alt_card = dbc.Card([
            dbc.CardHeader(html.H5("Alternative Patterns", className="mb-0")),
            dbc.CardBody([
                html.Div([
                    dbc.Row([
                        dbc.Col(html.P(alt['pattern']), width=6),
                        dbc.Col(
                            dbc.Progress(
                                value=alt['probability'] * 100,
                                label=f"{alt['probability']:.1%}",
                                color="info",
                                className="mb-2"
                            ),
                            width=6
                        )
                    ]) for alt in predictions['alternatives']
                ])
            ])
        ])
        cards.append(alt_card)
        
        # Record performance
        duration = time.time() - start_time
        performance_monitor.record_callback_time(duration)
        
        return cards
        
    except Exception as e:
        logger.error(f"Error updating predictions: {e}")
        return dbc.Alert(
            f"Error generating predictions: {str(e)}",
            color="danger",
            dismissable=True
        )

@app.callback(
    Output('accuracy-metrics', 'figure'),
    [Input('metrics-interval', 'n_intervals'),
     Input('metric-type-dropdown', 'value')]
)
def update_accuracy_metrics(n_intervals: int, metric_type: str):
    """Update accuracy metrics panel"""
    start_time = time.time()
    
    try:
        # Generate sample metrics data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        if metric_type == 'accuracy':
            metrics = pd.DataFrame({
                'date': dates,
                'accuracy': 0.75 + np.random.randn(30) * 0.05,
                'precision': 0.80 + np.random.randn(30) * 0.04,
                'recall': 0.70 + np.random.randn(30) * 0.06
            })
            
            fig = go.Figure()
            for col in ['accuracy', 'precision', 'recall']:
                fig.add_trace(go.Scatter(
                    x=metrics['date'],
                    y=metrics[col],
                    name=col.capitalize(),
                    mode='lines+markers',
                    hovertemplate='%{x}<br>%{y:.1%}<extra></extra>'
                ))
                
        elif metric_type == 'returns':
            metrics = pd.DataFrame({
                'date': dates,
                'predicted_return': np.random.randn(30) * 2,
                'actual_return': np.random.randn(30) * 2.2,
                'cumulative': np.cumsum(np.random.randn(30) * 0.5)
            })
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Daily Returns', 'Cumulative Performance')
            )
            
            # Daily returns
            fig.add_trace(
                go.Bar(x=metrics['date'], y=metrics['predicted_return'],
                       name='Predicted', marker_color='lightblue'),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(x=metrics['date'], y=metrics['actual_return'],
                       name='Actual', marker_color='darkblue', opacity=0.7),
                row=1, col=1
            )
            
            # Cumulative performance
            fig.add_trace(
                go.Scatter(x=metrics['date'], y=metrics['cumulative'],
                          name='Cumulative', line=dict(color='green', width=2)),
                row=2, col=1
            )
            
        else:  # confusion matrix
            patterns = ['A', 'B', 'C', 'D', 'E']
            confusion = np.random.randint(0, 20, size=(5, 5))
            np.fill_diagonal(confusion, np.random.randint(50, 100, 5))
            
            fig = go.Figure(data=go.Heatmap(
                z=confusion,
                x=patterns,
                y=patterns,
                colorscale='Blues',
                text=confusion,
                texttemplate='%{text}',
                textfont={"size": 12},
                hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Pattern Confusion Matrix",
                xaxis_title="Predicted Pattern",
                yaxis_title="Actual Pattern"
            )
        
        fig.update_layout(
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            template='plotly_white',
            hovermode='x unified' if metric_type != 'confusion' else 'closest'
        )
        
        # Record performance
        duration = time.time() - start_time
        performance_monitor.record_callback_time(duration)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error updating metrics: {e}")
        return go.Figure()

@app.callback(
    Output('performance-status', 'children'),
    [Input('performance-interval', 'n_intervals')]
)
def update_performance_status(n_intervals: int):
    """Update performance monitoring status"""
    perf = performance_monitor.check_performance()
    
    if perf['meets_criteria']:
        return dbc.Alert(
            [
                html.I(className="fas fa-check-circle me-2"),
                f"Performance OK - Avg callback: {perf['avg_callback_time']*1000:.0f}ms"
            ],
            color="success",
            className="mb-0"
        )
    else:
        return dbc.Alert(
            [
                html.I(className="fas fa-exclamation-triangle me-2"),
                f"Performance Warning - Avg callback: {perf['avg_callback_time']*1000:.0f}ms"
            ],
            color="warning",
            className="mb-0"
        )

# Error handler
@app.callback(
    Output('error-modal', 'is_open'),
    Output('error-modal-body', 'children'),
    [Input('error-store', 'data')],
    [State('error-modal', 'is_open')]
)
def handle_errors(error_data: Dict, is_open: bool):
    """Handle and display errors"""
    if error_data:
        return True, error_data.get('message', 'An error occurred')
    return False, ""

# Run the app
if __name__ == '__main__':
    # Development mode
    app.run_server(debug=True, host='0.0.0.0', port=8050)
    
    # Production mode (uncomment for deployment)
    # app.run_server(debug=False, host='0.0.0.0', port=8050, threaded=True)
