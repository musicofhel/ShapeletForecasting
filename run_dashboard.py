"""
Wavelet Forecast Dashboard - Production Version with Real YFinance Data Only

This is the main dashboard application that provides real-time financial pattern analysis,
prediction visualization, and performance metrics using exclusively real market data.
"""

import sys
import os
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

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import custom modules
from src.dashboard.layouts.forecast_layout import create_forecast_layout
from src.dashboard.callbacks.prediction_callbacks import register_prediction_callbacks
from src.dashboard.pattern_predictor import PatternPredictor
from src.dashboard.pattern_matcher import PatternMatcher
from src.dashboard.wavelet_sequence_analyzer import WaveletSequenceAnalyzer
from src.dashboard.visualizations.sequence_view import PatternSequenceVisualizer
from src.dashboard.pattern_classifier import PatternClassifier
from src.dashboard.data_utils_yfinance import YFinanceDataManager
from src.dashboard.pattern_detection import PatternDetector
from src.dashboard.model_loader import PatternPredictionModels
from src.models.transformer_predictor import TransformerPredictor
from src.models.xgboost_predictor import XGBoostPredictor

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

# Progress tracking class
class ProgressTracker:
    """Track and manage progress updates"""
    
    def __init__(self):
        self.current_progress = {
            'current_task': 'Initializing...',
            'progress': 0,
            'subtasks': [],
            'estimated_time': 0,
            'terminal_output': '$ Starting Wavelet Forecast Dashboard...\n'
        }
        self.start_time = time.time()
        
    def update(self, task: str, progress: int, subtask_status: Dict[str, str] = None):
        """Update progress status"""
        self.current_progress['current_task'] = task
        self.current_progress['progress'] = progress
        
        # Add to terminal output
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.current_progress['terminal_output'] += f"[{timestamp}] {task}\n"
        
        # Update subtasks if provided
        if subtask_status:
            for subtask, status in subtask_status.items():
                # Find or create subtask
                existing = next((s for s in self.current_progress['subtasks'] if s['task'] == subtask), None)
                if existing:
                    existing['status'] = status
                else:
                    self.current_progress['subtasks'].append({'task': subtask, 'status': status})
        
        # Estimate remaining time
        elapsed = time.time() - self.start_time
        if progress > 0:
            total_estimated = elapsed / (progress / 100)
            remaining = total_estimated - elapsed
            self.current_progress['estimated_time'] = max(0, int(remaining))
        
        return self.current_progress
    
    def get_status(self) -> Dict[str, Any]:
        """Get current progress status"""
        return self.current_progress.copy()

# Global progress tracker
progress_tracker = ProgressTracker()

# Enhanced Data management class with YFinance integration
class DataManager:
    """Manage data loading and caching with real YFinance data"""
    
    def __init__(self):
        self.yfinance_manager = YFinanceDataManager()
        self.data_cache = {}
        self.pattern_cache = {}
        self.lock = threading.Lock()
        self.pattern_matcher = PatternMatcher()
        self.pattern_classifier = PatternClassifier()
        self.wavelet_analyzer = WaveletSequenceAnalyzer()
        self.pattern_detector = PatternDetector()
        self.progress_tracker = progress_tracker
        
    def load_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load real data from YFinance with caching and progress tracking"""
        cache_key = f"{symbol}_{timeframe}"
        
        with self.lock:
            # Check cache first (5 minute expiry)
            if cache_key in self.data_cache:
                cached_data, timestamp = self.data_cache[cache_key]
                if time.time() - timestamp < 300:  # 5 minutes
                    logger.info(f"Using cached data for {symbol}")
                    return cached_data
            
            # Update progress - starting data fetch
            self.progress_tracker.update(
                f"Fetching {symbol} data from YFinance...", 
                10,
                {'Connect to YFinance API': 'in_progress', 'Download data': 'pending', 'Process data': 'pending'}
            )
            
            # Map timeframe to period and interval
            timeframe_map = {
                '1 Hour': {'period': '5d', 'interval': '1h'},    # 5 days of hourly data
                '1 Day': {'period': '1mo', 'interval': '1d'},    # 1 month of daily data
                '1 Week': {'period': '6mo', 'interval': '1wk'},  # 6 months of weekly data
                '1 Month': {'period': '2y', 'interval': '1mo'}   # 2 years of monthly data
            }
            
            config = timeframe_map.get(timeframe, {'period': '1mo', 'interval': '1d'})
            
            # Convert crypto symbols to YFinance format
            yf_symbol = symbol
            if symbol == 'BTCUSD':
                yf_symbol = 'BTC-USD'
            elif symbol == 'ETHUSD':
                yf_symbol = 'ETH-USD'
            
            # Update progress - downloading
            self.progress_tracker.update(
                f"Downloading {yf_symbol} data (period: {config['period']}, interval: {config['interval']})...", 
                30,
                {'Connect to YFinance API': 'complete', 'Download data': 'in_progress', 'Process data': 'pending'}
            )
            
            # Fetch real data from YFinance
            logger.info(f"Fetching {yf_symbol} data from YFinance, period: {config['period']}, interval: {config['interval']}")
            df = self.yfinance_manager.download_data(yf_symbol, period=config['period'], interval=config['interval'])
            
            if df is None or df.empty:
                logger.error(f"No data returned for {symbol}")
                self.progress_tracker.update(
                    f"Failed to fetch data for {symbol}", 
                    0,
                    {'Connect to YFinance API': 'complete', 'Download data': 'failed', 'Process data': 'failed'}
                )
                # Return empty dataframe - no fallback to mock data
                return pd.DataFrame()
            else:
                # Update progress - processing
                self.progress_tracker.update(
                    f"Processing {len(df)} data points...", 
                    60,
                    {'Connect to YFinance API': 'complete', 'Download data': 'complete', 'Process data': 'in_progress'}
                )
                
                # Ensure we have a timestamp column
                if 'timestamp' not in df.columns:
                    df['timestamp'] = df.index
                df['price'] = df['close']
                
                # Update progress - complete
                self.progress_tracker.update(
                    f"Data loaded successfully ({len(df)} points)", 
                    100,
                    {'Connect to YFinance API': 'complete', 'Download data': 'complete', 'Process data': 'complete'}
                )
                
            self.data_cache[cache_key] = (df, time.time())
            performance_monitor.data_points_processed += len(df)
            
        return df
    
    def get_patterns(self, symbol: str, df: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """Get detected patterns from real data with progress tracking"""
        if symbol in self.pattern_cache:
            cached_patterns, timestamp = self.pattern_cache.get(symbol, ([], 0))
            if time.time() - timestamp < 300:  # 5 minutes
                return cached_patterns
        
        patterns = []
        
        try:
            # Only detect patterns if we have real data
            if df is not None and len(df) > 50:
                # Update progress - starting pattern detection
                self.progress_tracker.update(
                    f"Applying wavelet transforms to {symbol} data...", 
                    10,
                    {'Wavelet transform': 'in_progress', 'Pattern detection': 'pending', 'Pattern classification': 'pending'}
                )
                
                # Use pattern detector to find patterns
                price_data = df['close'].values if 'close' in df.columns else df['price'].values
                timestamps = df['timestamp'].values if 'timestamp' in df.columns else df.index.values
                
                # Update progress - detecting patterns
                self.progress_tracker.update(
                    f"Detecting patterns in {symbol} price data...", 
                    40,
                    {'Wavelet transform': 'complete', 'Pattern detection': 'in_progress', 'Pattern classification': 'pending'}
                )
                
                # Detect all patterns using the pattern detector
                detected_patterns = self.pattern_detector.detect_all_patterns(price_data, timestamps)
                
                # Update progress - classifying patterns
                self.progress_tracker.update(
                    f"Classifying {len(detected_patterns)} detected patterns...", 
                    70,
                    {'Wavelet transform': 'complete', 'Pattern detection': 'complete', 'Pattern classification': 'in_progress'}
                )
                
                # Format patterns for dashboard display
                for i, pattern in enumerate(detected_patterns):
                    formatted_pattern = {
                        'id': f'pattern_{i}',
                        'name': pattern['name'],
                        'type': pattern['type'],
                        'start_idx': pattern['start_idx'],
                        'end_idx': pattern['end_idx'],
                        'confidence': pattern['confidence'],
                        'frequency': 1,  # Will be calculated from historical data
                        'avg_return': 0.0  # Will be calculated from historical performance
                    }
                    
                    # Add time information if available
                    if 'start_time' in pattern:
                        formatted_pattern['start_time'] = pattern['start_time']
                    if 'end_time' in pattern:
                        formatted_pattern['end_time'] = pattern['end_time']
                    
                    # Calculate pattern strength
                    formatted_pattern['strength'] = self.pattern_detector.calculate_pattern_strength(
                        pattern, price_data
                    )
                    
                    patterns.append(formatted_pattern)
                
                # Sort patterns by confidence and strength
                patterns.sort(key=lambda p: (p['confidence'] * p['strength']), reverse=True)
                
                # Update progress - complete
                self.progress_tracker.update(
                    f"Pattern analysis complete - found {len(patterns)} patterns", 
                    100,
                    {'Wavelet transform': 'complete', 'Pattern detection': 'complete', 'Pattern classification': 'complete'}
                )
            
            # Log pattern detection results
            if patterns:
                logger.info(f"Detected {len(patterns)} patterns for {symbol}")
            else:
                logger.info(f"No patterns detected for {symbol}")
            
            self.pattern_cache[symbol] = (patterns, time.time())
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            self.progress_tracker.update(
                f"Pattern detection failed: {str(e)}", 
                0,
                {'Wavelet transform': 'failed', 'Pattern detection': 'failed', 'Pattern classification': 'failed'}
            )
            # Return empty list on error
            patterns = []
        
        return patterns

# Initialize data manager
data_manager = DataManager()

# Initialize pattern prediction models
pattern_models = PatternPredictionModels()

# Log model status
model_status = pattern_models.get_model_status()
logger.info(f"Model loader status: {model_status}")

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
        # Load real data from YFinance
        df = data_manager.load_data(symbol, timeframe)
        
        # Check if we have data
        if df.empty:
            # Return empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"No data available for {symbol}. Please check your internet connection.",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20, color="red")
            )
            return fig, []
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{symbol} Price with Pattern Overlays', 'Volume')
        )
        
        # Add candlestick trace if we have OHLC data
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            fig.add_trace(
                go.Candlestick(
                    x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name=symbol,
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ),
                row=1, col=1
            )
        else:
            # Fallback to line chart
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                    y=df['price'] if 'price' in df.columns else df['close'],
                    name='Price',
                    line=dict(color='#1f77b4', width=1),
                    hovertemplate='%{x}<br>Price: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Get real patterns from the data
        patterns = data_manager.get_patterns(symbol, df)
        pattern_data = []
        
        # Add pattern overlays if enabled
        if 'patterns' in show_patterns and patterns:
            for i, pattern in enumerate(patterns[:5]):  # Show top 5 patterns
                if 'start_idx' in pattern and 'end_idx' in pattern:
                    start_idx = pattern['start_idx']
                    end_idx = pattern['end_idx']
                    
                    if start_idx < len(df) and end_idx <= len(df):
                        pattern_segment = df.iloc[start_idx:end_idx]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=pattern_segment['timestamp'] if 'timestamp' in pattern_segment.columns else pattern_segment.index,
                                y=pattern_segment['price'] if 'price' in pattern_segment.columns else pattern_segment['close'],
                                name=pattern['name'],
                                line=dict(width=3),
                                opacity=0.7,
                                hovertemplate=f"{pattern['name']}<br>%{{x}}<br>Price: $%{{y:.2f}}<br>Confidence: {pattern['confidence']:.1%}<extra></extra>"
                            ),
                            row=1, col=1
                        )
                        
                        pattern_data.append({
                            'pattern_id': pattern['id'],
                            'start': start_idx,
                            'end': end_idx,
                            'confidence': pattern['confidence'],
                            'type': pattern.get('type', 'unknown')
                        })
        
        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=df['timestamp'] if 'timestamp' in df.columns else df.index,
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
        
        # Add current price annotation
        if len(df) > 0:
            current_price = df.iloc[-1]['close'] if 'close' in df.columns else df.iloc[-1]['price']
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"Current Price: ${current_price:.2f}",
                showarrow=False,
                font=dict(size=16, color="black"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1
            )
        
        # Record performance
        duration = time.time() - start_time
        performance_monitor.record_callback_time(duration)
        
        return fig, pattern_data
        
    except Exception as e:
        logger.error(f"Error updating main chart: {e}")
        # Return empty figure on error
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading data for {symbol}: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="red")
        )
        return fig, []

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
        
        # Create pattern sequence based on pattern_data
        if pattern_data and len(pattern_data) > 0:
            # Use actual pattern data
            for i, pdata in enumerate(pattern_data[:sequence_length]):
                pattern_type = pdata.get('type', 'unknown')
                patterns.append({
                    'id': pdata.get('pattern_id', f'P{i+1:03d}'),
                    'type': pattern_type,
                    'start_time': base_time + timedelta(hours=i*24),
                    'end_time': base_time + timedelta(hours=(i*24 + 12)),  # Fixed 12 hour duration
                    'confidence': pdata.get('confidence', 0.8),
                    'ticker': symbol
                })
        else:
            # No pattern data available - return empty visualization
            return visualizer._create_empty_figure(f"No patterns detected for {symbol}")
        
        # Generate transitions between patterns
        transitions = {}
        for i in range(len(patterns) - 1):
            trans_key = f"{patterns[i]['id']}_to_{patterns[i+1]['id']}"
            transitions[trans_key] = {
                'probability': 0.5  # Default probability until real calculation implemented
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
    """Update prediction display with trained pattern prediction models"""
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
        
        # Load real data
        df = data_manager.load_data(symbol, '1 Day')
        
        # Check if we have data
        if df.empty:
            return dbc.Alert(
                f"No data available for {symbol}. Please check your internet connection.",
                color="danger",
                dismissable=True
            )
        
        # Prepare data for models
        if len(df) > 100:
            price_data = df['close'].values if 'close' in df.columns else df['price'].values
            last_price = price_data[-1]
            
            # Get detected patterns for pattern sequence prediction
            patterns = data_manager.get_patterns(symbol, df)
            
            # Extract pattern sequence (last 10 patterns)
            pattern_sequence = []
            if patterns:
                # Sort patterns by time/index
                sorted_patterns = sorted(patterns, key=lambda p: p.get('start_idx', 0))
                # Get pattern types for the sequence
                pattern_sequence = [p['type'] for p in sorted_patterns[-10:]]
            
            # Use trained models to predict next pattern
            pattern_prediction = pattern_models.predict_next_pattern(pattern_sequence)
            next_pattern = pattern_prediction['next_pattern']
            pattern_confidence = pattern_prediction['confidence']
            alternatives = pattern_prediction['alternatives']
            
            # Get price predictions based on pattern
            price_predictions = pattern_models.predict_price_movement(
                last_price, next_pattern, horizon
            )
            
            # Extract individual model predictions
            lstm_pred = price_predictions.get('lstm', last_price)
            gru_pred = price_predictions.get('gru', last_price)
            transformer_pred = price_predictions.get('transformer', last_price)
            ensemble_pred = price_predictions.get('ensemble', last_price)
            
            # Overall confidence based on pattern confidence
            confidence = pattern_confidence
            
        else:
            # Not enough data for predictions
            return dbc.Alert(
                f"Insufficient data for {symbol}. Need at least 100 data points for predictions.",
                color="warning",
                dismissable=True
            )
        
        # Create prediction cards
        cards = []
        
        # Current price card
        current_card = dbc.Card([
            dbc.CardHeader(html.H5("Current Price", className="mb-0")),
            dbc.CardBody([
                html.H2(f"${last_price:.2f}", className="text-info"),
                html.P(f"Symbol: {symbol}", className="text-muted")
            ])
        ], className="mb-3")
        cards.append(current_card)
        
        # Model predictions card
        pred_card = dbc.Card([
            dbc.CardHeader(html.H5(f"{horizon}-Step Price Predictions", className="mb-0")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("LSTM Model", className="text-muted"),
                        html.H4(f"${lstm_pred:.2f}", className="text-primary"),
                        html.Small(f"{((lstm_pred - last_price) / last_price * 100):+.2f}%", 
                                 className="text-muted")
                    ], width=3),
                    dbc.Col([
                        html.H6("GRU Model", className="text-muted"),
                        html.H4(f"${gru_pred:.2f}", className="text-success"),
                        html.Small(f"{((gru_pred - last_price) / last_price * 100):+.2f}%", 
                                 className="text-muted")
                    ], width=3),
                    dbc.Col([
                        html.H6("Transformer", className="text-muted"),
                        html.H4(f"${transformer_pred:.2f}", className="text-info"),
                        html.Small(f"{((transformer_pred - last_price) / last_price * 100):+.2f}%", 
                                 className="text-muted")
                    ], width=3),
                    dbc.Col([
                        html.H6("Ensemble", className="text-muted"),
                        html.H4(f"${ensemble_pred:.2f}", className="text-warning"),
                        html.Small(f"{((ensemble_pred - last_price) / last_price * 100):+.2f}%", 
                                 className="text-muted")
                    ], width=3)
                ]),
                html.Hr(),
                html.P(f"Pattern Confidence: {confidence:.1%}", className="text-muted"),
                dbc.Progress(
                    value=confidence * 100,
                    color="success" if confidence > 0.8 else "warning" if confidence > 0.5 else "danger",
                    className="mb-3"
                )
            ])
        ], className="mb-3")
        cards.append(pred_card)
        
        # Pattern prediction card with alternatives
        pattern_card = dbc.Card([
            dbc.CardHeader(html.H5("Next Pattern Prediction", className="mb-0")),
            dbc.CardBody([
                html.H3(next_pattern.replace('_', ' ').title(), className="text-primary"),
                html.P(f"Confidence: {pattern_confidence:.1%}", className="text-muted"),
                html.Hr(),
                html.H6("Alternative Patterns:", className="text-muted"),
                html.Ul([
                    html.Li(f"{alt['pattern'].replace('_', ' ').title()} ({alt['confidence']:.1%})")
                    for alt in alternatives[:3]
                ]) if alternatives else html.P("No high-confidence alternatives", className="text-muted")
            ])
        ], className="mb-3")
        cards.append(pattern_card)
        
        # Model status card
        model_status = pattern_models.get_model_status()
        status_card = dbc.Card([
            dbc.CardHeader(html.H5("Model Status", className="mb-0")),
            dbc.CardBody([
                html.P([
                    html.I(className="fas fa-check-circle text-success me-2") 
                    if model_status['models_loaded'] else 
                    html.I(className="fas fa-exclamation-circle text-warning me-2"),
                    f"Models Loaded: {', '.join(model_status['models_loaded']) if model_status['models_loaded'] else 'None'}"
                ]),
                html.P([
                    html.I(className="fas fa-check-circle text-success me-2") 
                    if model_status['config_loaded'] else 
                    html.I(className="fas fa
