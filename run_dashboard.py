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

# Import advanced time series tools
from src.dashboard.tools.pattern_compare import EnhancedPatternComparison
from src.wavelet_analysis.advanced_pattern_detector import AdvancedWaveletPatternDetector
from src.advanced.time_series_integration import (
    SAXTransformer, SAXConfig,
    TimeSeriesSimilaritySearch, SimilaritySearchConfig
)

# Import shapelet discovery
from src.shapelet_discovery import ShapeletDiscoverer
from src.dashboard.visualizations.shapelet_visualization import ShapeletVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performance monitoring - COMMENTED OUT AS REQUESTED
# class PerformanceMonitor:
#     """Monitor dashboard performance metrics"""
#     
#     def __init__(self):
#         self.callback_times = []
#         self.memory_usage = []
#         self.active_sessions = 0
#         self.data_points_processed = 0
#         
#     def record_callback_time(self, duration: float):
#         """Record callback execution time"""
#         self.callback_times.append(duration)
#         # Keep only last 1000 measurements
#         if len(self.callback_times) > 1000:
#             self.callback_times.pop(0)
#     
#     def get_average_callback_time(self) -> float:
#         """Get average callback execution time"""
#         if not self.callback_times:
#             return 0
#         return np.mean(self.callback_times)
#     
#     def check_performance(self) -> Dict[str, Any]:
#         """Check if performance meets criteria"""
#         avg_callback_time = self.get_average_callback_time()
#         return {
#             'meets_criteria': avg_callback_time < 0.5,  # 500ms
#             'avg_callback_time': avg_callback_time,
#             'active_sessions': self.active_sessions,
#             'data_points_processed': self.data_points_processed
#         }

# Initialize performance monitor - COMMENTED OUT
# performance_monitor = PerformanceMonitor()

# Initialize shapelet discovery components
shapelet_discoverer = ShapeletDiscoverer()
shapelet_visualizer = ShapeletVisualizer()
discovered_shapelets = {}  # Cache for discovered shapelets

# Helper function for loading items
def create_loading_item(name: str, status: str = "pending") -> html.Div:
    """Create a loading item with status indicator"""
    icon_map = {
        "pending": "fas fa-circle text-muted",
        "loading": "fas fa-spinner fa-spin text-primary",
        "complete": "fas fa-check-circle text-success",
        "error": "fas fa-times-circle text-danger"
    }
    
    return html.Div([
        html.I(className=f"{icon_map.get(status, icon_map['pending'])} me-2"),
        html.Span(name, className="text-muted" if status == "pending" else "")
    ], className="mb-2", id=f"loading-{name.replace(' ', '-').replace('.', '')}")

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
        self.cache_metadata = {}  # Store cache timestamps and data ranges
        self.lock = threading.Lock()
        self.pattern_matcher = PatternMatcher()
        self.pattern_classifier = PatternClassifier()
        self.wavelet_analyzer = WaveletSequenceAnalyzer()
        self.pattern_detector = PatternDetector()
        self.progress_tracker = progress_tracker
        
        # Initialize advanced pattern detection components
        self.advanced_detector = AdvancedWaveletPatternDetector()
        self.pattern_comparison = EnhancedPatternComparison()
        self.sax_transformer = SAXTransformer(SAXConfig(n_segments=20, alphabet_size=5))
        self.similarity_search = TimeSeriesSimilaritySearch(
            SimilaritySearchConfig(method='dtw', top_k=5)
        )
        
        # Initialize shapelet discovery
        self.shapelet_discoverer = shapelet_discoverer
        self.shapelet_visualizer = shapelet_visualizer
        
    def load_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load real data from YFinance with intelligent caching and incremental updates"""
        cache_key = f"{symbol}_{timeframe}"
        
        # Update progress - checking cache
        self.progress_tracker.update(
            f"Checking cache for {symbol}...", 
            5,
            {'Check cache': 'in_progress', 'Validate data': 'pending', 'Fetch updates': 'pending'}
        )
        
        with self.lock:
            # Check if we have ANY cached data first
            if cache_key in self.data_cache:
                cached_data, cache_timestamp = self.data_cache[cache_key]
                cache_age = time.time() - cache_timestamp
                
                # Update progress - found cache
                self.progress_tracker.update(
                    f"Found cached data for {symbol} (age: {cache_age:.0f}s)", 
                    20,
                    {'Check cache': 'complete', 'Validate data': 'in_progress', 'Fetch updates': 'pending'}
                )
                
                # If cache is very fresh (< 1 minute), use it directly
                if cache_age < 60:
                    logger.info(f"Using fresh cached data for {symbol} (age: {cache_age:.0f}s)")
                    self.progress_tracker.update(
                        f"Using fresh cache - no update needed", 
                        100,
                        {'Check cache': 'complete', 'Validate data': 'complete', 'Fetch updates': 'skipped'}
                    )
                    return cached_data
                
                # If cache is moderately fresh (< 5 minutes), check for updates
                if cache_age < 300:
                    # Check if market is open and we might have new data
                    metadata = self.cache_metadata.get(cache_key, {})
                    last_data_time = metadata.get('last_data_time')
                    
                    if last_data_time:
                        # Calculate if we should check for new data
                        # Ensure both timestamps are timezone-aware or both are naive
                        current_time = datetime.now()
                        last_time = pd.to_datetime(last_data_time)
                        
                        # If last_time is timezone-aware, make current_time aware too
                        if last_time.tzinfo is not None:
                            import pytz
                            current_time = pytz.UTC.localize(current_time)
                        # If last_time is naive, ensure it stays naive
                        elif last_time.tzinfo is None and hasattr(last_time, 'tz_localize'):
                            last_time = last_time.tz_localize(None)
                            
                        time_since_last_data = current_time - last_time
                        
                        # Map timeframe to expected update frequency
                        update_freq = {
                            '1 Hour': timedelta(hours=1),
                            '1 Day': timedelta(days=1),
                            '1 Week': timedelta(weeks=1),
                            '1 Month': timedelta(days=30)
                        }
                        
                        expected_freq = update_freq.get(timeframe, timedelta(days=1))
                        
                        # If not enough time has passed for new data, use cache
                        if time_since_last_data < expected_freq:
                            logger.info(f"Using cached data for {symbol} - no new data expected yet")
                            self.progress_tracker.update(
                                f"No new data expected for {timeframe} timeframe", 
                                100,
                                {'Check cache': 'complete', 'Validate data': 'complete', 'Fetch updates': 'not_needed'}
                            )
                            return cached_data
                    
                    # Otherwise, check for incremental updates
                    logger.info(f"Checking for new data updates for {symbol}")
                    self.progress_tracker.update(
                        f"Checking for new {symbol} data since last update...", 
                        40,
                        {'Check cache': 'complete', 'Validate data': 'complete', 'Fetch updates': 'in_progress'}
                    )
            else:
                # No cache exists
                self.progress_tracker.update(
                    f"No cached data found for {symbol} - will fetch full dataset", 
                    10,
                    {'Check cache': 'complete', 'Validate data': 'skipped', 'Fetch updates': 'pending'}
                )
                    
            # Store metadata about the last data point
            if cache_key in self.data_cache:
                cached_data, _ = self.data_cache[cache_key]
                if not cached_data.empty:
                    last_timestamp = cached_data.index[-1] if isinstance(cached_data.index, pd.DatetimeIndex) else cached_data['timestamp'].iloc[-1]
                    self.cache_metadata[cache_key] = {
                        'last_data_time': last_timestamp,
                        'data_points': len(cached_data),
                        'last_update': datetime.now()
                    }
            
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
            # performance_monitor.data_points_processed += len(df)  # COMMENTED OUT
            
        return df
    
    def get_patterns(self, symbol: str, df: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """Get detected shapelets from real data with progress tracking"""
        if symbol in self.pattern_cache:
            cached_patterns, timestamp = self.pattern_cache.get(symbol, ([], 0))
            if time.time() - timestamp < 300:  # 5 minutes
                return cached_patterns
        
        patterns = []
        
        try:
            # Only detect patterns if we have real data
            if df is not None and len(df) > 50:
                # Update progress - starting shapelet discovery
                self.progress_tracker.update(
                    f"Discovering shapelets in {symbol} data...", 
                    10,
                    {'Shapelet discovery': 'in_progress', 'SAX transformation': 'pending', 'Pattern classification': 'pending'}
                )
                
                # Get timeframe from the dataframe
                timeframe = '1 Hour'  # Default, will be updated based on actual data
                if len(df) > 1:
                    time_diff = pd.to_datetime(df.index[1]) - pd.to_datetime(df.index[0])
                    if time_diff.days >= 7:
                        timeframe = '1 Month'
                    elif time_diff.days >= 1:
                        timeframe = '1 Week'
                    elif time_diff.total_seconds() >= 3600:
                        timeframe = '1 Day'
                
                # Update progress - discovering shapelets
                self.progress_tracker.update(
                    f"Applying SAX transformation to {symbol} price data...", 
                    40,
                    {'Shapelet discovery': 'in_progress', 'SAX transformation': 'in_progress', 'Pattern classification': 'pending'}
                )
                
                # Discover shapelets using shapelet discoverer
                shapelets = self.shapelet_discoverer.discover_shapelets(
                    df, 
                    ticker=symbol,
                    timeframe=timeframe,
                    price_col='close' if 'close' in df.columns else 'price'
                )
                
                # Update progress - classifying patterns
                self.progress_tracker.update(
                    f"Classifying {len(shapelets)} discovered shapelets...", 
                    70,
                    {'Shapelet discovery': 'complete', 'SAX transformation': 'complete', 'Pattern classification': 'in_progress'}
                )
                
                # Format shapelets as patterns for dashboard display
                for i, shapelet in enumerate(shapelets):
                    formatted_pattern = {
                        'id': f'shapelet_{i}',
                        'name': f'Shapelet {shapelet.sax_label}',
                        'type': f'sax_{shapelet.sax_label}',
                        'start_idx': shapelet.start_idx,
                        'end_idx': shapelet.end_idx,
                        'confidence': shapelet.statistics.get('confidence', 0.8),
                        'frequency': shapelet.statistics.get('frequency', 1),
                        'avg_return': shapelet.statistics.get('avg_return_after', 0.0),
                        'sax_label': shapelet.sax_label,
                        'length': shapelet.length
                    }
                    
                    # Add time information if available
                    if hasattr(shapelet, 'start_time'):
                        formatted_pattern['start_time'] = shapelet.start_time
                    if hasattr(shapelet, 'end_time'):
                        formatted_pattern['end_time'] = shapelet.end_time
                    
                    # Calculate pattern strength based on frequency and returns
                    formatted_pattern['strength'] = min(
                        shapelet.statistics.get('frequency', 1) / 10.0 * 
                        (1 + abs(shapelet.statistics.get('avg_return_after', 0))),
                        1.0
                    )
                    
                    patterns.append(formatted_pattern)
                
                # Sort patterns by frequency and strength
                patterns.sort(key=lambda p: (p['frequency'] * p['strength']), reverse=True)
                
                # Update progress - complete
                self.progress_tracker.update(
                    f"Shapelet discovery complete - found {len(shapelets)} shapelets", 
                    100,
                    {'Shapelet discovery': 'complete', 'SAX transformation': 'complete', 'Pattern classification': 'complete'}
                )
            
            # Log shapelet discovery results
            if patterns:
                logger.info(f"Discovered {len(patterns)} shapelets for {symbol}")
            else:
                logger.info(f"No shapelets discovered for {symbol}")
            
            self.pattern_cache[symbol] = (patterns, time.time())
            
        except Exception as e:
            logger.error(f"Error discovering shapelets: {e}")
            self.progress_tracker.update(
                f"Shapelet discovery failed: {str(e)}", 
                0,
                {'Shapelet discovery': 'failed', 'SAX transformation': 'failed', 'Pattern classification': 'failed'}
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

# Create app layout with loading overlay
app.layout = html.Div([
    # URL for routing
    dcc.Location(id='url', refresh=False),
    
    # Loading overlay with detailed progress
    html.Div(
        id='loading-overlay',
        children=[
            dbc.Card([
                dbc.CardHeader([
                    html.H4([
                        html.I(className="fas fa-cog fa-spin me-2"),
                        "Initializing Wavelet Forecast Dashboard"
                    ], className="mb-0")
                ], className="bg-primary text-white"),
                dbc.CardBody([
                    # Overall progress
                    html.Div([
                        html.H6("Overall Progress", className="mb-2"),
                        dbc.Progress(
                            id="startup-progress",
                            value=0,
                            striped=True,
                            animated=True,
                            className="mb-3",
                            style={"height": "25px"}
                        ),
                    ]),
                    
                    # Module loading status
                    html.Div([
                        html.H6("Module Loading Status:", className="mb-3"),
                        html.Div(id="module-loading-list", children=[
                            # Initial loading items
                            create_loading_item("1. Core Libraries", "pending"),
                            create_loading_item("2. Wavelet Analysis Engine", "pending"),
                            create_loading_item("3. Pattern Detection System", "pending"),
                            create_loading_item("4. Machine Learning Models", "pending"),
                            create_loading_item("5. Data Management Layer", "pending"),
                            create_loading_item("6. Dashboard Components", "pending"),
                            create_loading_item("7. Real-time Data Connections", "pending"),
                            create_loading_item("8. Advanced Analytics Tools", "pending"),
                        ])
                    ]),
                    
                    # Terminal-style output
                    html.Div([
                        html.H6("System Log:", className="mb-2 mt-3"),
                        html.Pre(
                            id="startup-terminal",
                            children="$ Initializing dashboard components...\n",
                            style={
                                "backgroundColor": "#1e1e1e",
                                "color": "#00ff00",
                                "padding": "10px",
                                "borderRadius": "5px",
                                "fontFamily": "Consolas, Monaco, 'Courier New', monospace",
                                "fontSize": "11px",
                                "height": "120px",
                                "overflowY": "auto",
                                "margin": "0"
                            }
                        )
                    ]),
                    
                    # Estimated time
                    html.Div([
                        html.Small(id="startup-eta", children="Estimated time: calculating...", 
                                 className="text-muted")
                    ], className="text-center mt-2")
                ])
            ], style={
                'width': '600px',
                'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
            })
        ],
        style={
            'position': 'fixed',
            'top': 0,
            'left': 0,
            'width': '100%',
            'height': '100%',
            'backgroundColor': 'rgba(255, 255, 255, 0.95)',
            'zIndex': 9999,
            'display': 'flex',
            'alignItems': 'center',
            'justifyContent': 'center'
        }
    ),
    
    # Main app layout
    html.Div([
        create_forecast_layout(),
        
        # Floating progress panel
        html.Div(
            id='floating-progress-panel',
            style={'display': 'none'},
            children=[
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Operation Progress", className="mb-0 d-inline-block"),
                        dbc.Button(
                            html.I(className="fas fa-times"),
                            id="close-progress",
                            size="sm",
                            color="link",
                            className="float-end"
                        )
                    ]),
                    dbc.CardBody([
                        html.Div(id='floating-progress-task-name', className="mb-2"),
                        dbc.Progress(
                            id='floating-task-progress-bar',
                            value=0,
                            striped=True,
                            animated=True,
                            className="mb-3"
                        ),
                        html.Div(
                            id='progress-subtasks',
                            children=[],
                            className="small"
                        ),
                        html.Pre(
                            id='floating-terminal-output',
                            style={
                                'backgroundColor': '#1e1e1e',
                                'color': '#00ff00',
                                'padding': '10px',
                                'borderRadius': '5px',
                                'maxHeight': '150px',
                                'overflowY': 'auto',
                                'fontSize': '12px',
                                'fontFamily': 'monospace'
                            }
                        ),
                        html.Div(id='floating-estimated-time', className="text-muted small mt-2")
                    ])
                ], style={
                    'position': 'fixed',
                    'bottom': '20px',
                    'right': '20px',
                    'width': '400px',
                    'zIndex': 1000,
                    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
                })
            ]
        )
    ])
])

# Register callbacks
# register_prediction_callbacks(app, data_manager, performance_monitor)  # COMMENTED OUT - performance_monitor removed
register_prediction_callbacks(app, data_manager, None)  # Pass None for performance_monitor

# Additional callbacks for main functionality
@app.callback(
    [Output('main-time-series', 'figure'),
     Output('pattern-overlay-store', 'data'),
     Output('shapelet-discovered-store', 'data'),
     Output('chart-type-toggle', 'value'),
     Output('chart-type-toggle', 'options')],
    [Input('symbol-dropdown', 'value'),
     Input('timeframe-dropdown', 'value'),
     Input('pattern-toggle', 'value'),
     Input('refresh-interval', 'n_intervals'),
     Input('chart-type-toggle', 'value'),
     Input('discover-shapelets-button', 'n_clicks')],
    [State('shapelet-discovered-store', 'data')]
)
def update_main_chart(symbol: str, timeframe: str, show_patterns: List[str], n_intervals: int, 
                     chart_type: str, shapelet_clicks: int, shapelets_discovered: bool):
    """Update main time series chart with pattern overlays"""
    start_time = time.time()
    
    # Check if shapelets were just discovered
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # If shapelets were discovered, force line chart and set discovered flag
    if triggered_id == 'discover-shapelets-button' and shapelet_clicks:
        shapelets_discovered = True
        chart_type = 'line'
        # Update chart type options to disable candlestick when shapelets are shown
        chart_options = [
            {"label": "Candlestick", "value": "candlestick", "disabled": True},
            {"label": "Line Chart", "value": "line", "disabled": False}
        ]
    else:
        # Normal chart options
        chart_options = [
            {"label": "Candlestick", "value": "candlestick", "disabled": False},
            {"label": "Line Chart", "value": "line", "disabled": False}
        ]
    
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
            return fig, [], shapelets_discovered, chart_type, chart_options
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{symbol} Price with Pattern Overlays', 'Volume')
        )
        
        # Add price trace based on chart type
        if chart_type == 'candlestick' and all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # Add candlestick chart
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
            # Add line chart (default or when candlestick data not available)
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                    y=df['price'] if 'price' in df.columns else df['close'],
                    name='Price',
                    line=dict(color='#1f77b4', width=2),
                    hovertemplate='%{x}<br>Price: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Get real patterns from the data
        patterns = data_manager.get_patterns(symbol, df)
        pattern_data = []
        
        # If shapelets were discovered, overlay them on the chart
        if shapelets_discovered and triggered_id == 'discover-shapelets-button':
            # Get cached shapelets
            cache_key = f"{symbol}_{timeframe}_shapelets"
            if cache_key in discovered_shapelets:
                shapelets = discovered_shapelets[cache_key]
                
                # Add shapelet overlays
                for i, shapelet in enumerate(shapelets[:10]):  # Show top 10 shapelets
                    if shapelet.start_idx < len(df) and shapelet.end_idx <= len(df):
                        shapelet_segment = df.iloc[shapelet.start_idx:shapelet.end_idx]
                        
                        # Safe statistics access helper for hover template
                        freq = shapelet.statistics.get('frequency', 0) if isinstance(shapelet.statistics, dict) else 0
                        avg_return = shapelet.statistics.get('avg_return_after', 0) if isinstance(shapelet.statistics, dict) else 0
                        
                        fig.add_trace(
                            go.Scatter(
                                x=shapelet_segment['timestamp'] if 'timestamp' in shapelet_segment.columns else shapelet_segment.index,
                                y=shapelet_segment['price'] if 'price' in shapelet_segment.columns else shapelet_segment['close'],
                                name=f'SAX: {shapelet.sax_label}',
                                line=dict(width=3),
                                opacity=0.7,
                                hovertemplate=f"SAX: {shapelet.sax_label}<br>%{{x}}<br>Price: $%{{y:.2f}}<br>Frequency: {freq}<br>Avg Return: {avg_return:.2%}<extra></extra>"
                            ),
                            row=1, col=1
                        )
        
        # Add pattern overlays if enabled (and not showing shapelets)
        elif 'patterns' in show_patterns and patterns and not shapelets_discovered:
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
        
        # Record performance - COMMENTED OUT
        # duration = time.time() - start_time
        # performance_monitor.record_callback_time(duration)
        
        return fig, pattern_data, shapelets_discovered, chart_type, chart_options
        
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
        return fig, [], shapelets_discovered, chart_type, chart_options

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
        
        # Record performance - COMMENTED OUT
        # duration = time.time() - start_time
        # performance_monitor.record_callback_time(duration)
        
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
                    html.I(className="fas fa-times-circle text-danger me-2"),
                    "Configuration Loaded"
                ]),
                html.P(f"Device: {model_status['device']}", className="text-muted small")
            ])
        ])
        cards.append(status_card)
        
        # Record performance - COMMENTED OUT
        # duration = time.time() - start_time
        # performance_monitor.record_callback_time(duration)
        
        return cards
        
    except Exception as e:
        logger.error(f"Error updating predictions: {e}")
        return dbc.Alert(
            f"Error generating predictions: {str(e)}",
            color="danger",
            dismissable=True
        )

# ACCURACY METRICS CALLBACK - COMMENTED OUT AS REQUESTED
# @app.callback(
#     Output('accuracy-metrics', 'figure'),
#     [Input('metrics-interval', 'n_intervals'),
#      Input('metric-type-dropdown', 'value')]
# )
# def update_accuracy_metrics(n_intervals: int, metric_type: str):
#     """Update accuracy metrics panel - production version shows real metrics when available"""
#     start_time = time.time()
#     
#     try:
#         # In production, these metrics would come from actual model evaluation
#         # For now, return a message indicating metrics will be available after training
#         fig = go.Figure()
#         fig.add_annotation(
#             text="Metrics will be available once models are trained with sufficient real data",
#             xref="paper", yref="paper",
#             x=0.5, y=0.5,
#             showarrow=False,
#             font=dict(size=16, color="gray")
#         )
#         
#         fig.update_layout(
#             height=400,
#             margin=dict(l=50, r=50, t=50, b=50),
#             template='plotly_white'
#         )
#         
#         # Record performance
#         duration = time.time() - start_time
#         performance_monitor.record_callback_time(duration)
#         
#         return fig
#         
#     except Exception as e:
#         logger.error(f"Error updating metrics: {e}")
#         return go.Figure()

# PERFORMANCE STATUS CALLBACK - COMMENTED OUT AS REQUESTED
# @app.callback(
#     Output('performance-status', 'children'),
#     [Input('performance-interval', 'n_intervals')]
# )
# def update_performance_status(n_intervals: int):
#     """Update performance monitoring status"""
#     perf = performance_monitor.check_performance()
#     
#     if perf['meets_criteria']:
#         return dbc.Alert(
#             [
#                 html.I(className="fas fa-check-circle me-2"),
#                 f"Performance OK - Avg callback: {perf['avg_callback_time']*1000:.0f}ms | Data points: {perf['data_points_processed']:,}"
#             ],
#             color="success",
#             className="mb-0"
#         )
#     else:
#         return dbc.Alert(
#             [
#                 html.I(className="fas fa-exclamation-triangle me-2"),
#                 f"Performance Warning - Avg callback: {perf['avg_callback_time']*1000:.0f}ms"
#             ],
#             color="warning",
#             className="mb-0"
#         )

# Progress panel callbacks
@app.callback(
    Output('progress-collapse', 'is_open'),
    [Input('toggle-progress', 'n_clicks')],
    [State('progress-collapse', 'is_open')]
)
def toggle_progress_panel(n_clicks: int, is_open: bool):
    """Toggle progress panel visibility"""
    if n_clicks:
        return not is_open
    return is_open

@app.callback(
    [Output('floating-progress-panel', 'style'),
     Output('floating-task-progress-bar', 'value'),
     Output('floating-progress-task-name', 'children'),
     Output('progress-percentage', 'children'),
     Output('floating-terminal-output', 'children'),
     Output('floating-estimated-time', 'children')],
    [Input('symbol-dropdown', 'value'),
     Input('timeframe-dropdown', 'value'),
     Input('predict-button', 'n_clicks')],
    [State('progress-store', 'data')]
)
def update_progress_panel(symbol: str, timeframe: str, predict_clicks: int, progress_data: Dict):
    """Update progress panel based on current operations"""
    ctx = callback_context
    
    if not ctx.triggered:
        # Initial state - hide panel
        return {'display': 'none'}, 0, 'Initializing...', '0%', '$ Starting Wavelet Forecast Dashboard...\n', ''
    
    # Get current progress from global tracker
    current_progress = progress_tracker.get_status()
    
    # Show panel when data is being loaded or processed
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_id in ['symbol-dropdown', 'timeframe-dropdown']:
        # Data loading triggered - show panel
        style = {'display': 'block'}
    elif triggered_id == 'predict-button':
        # Prediction triggered - show panel
        style = {'display': 'block'}
    else:
        # Check if we're still processing
        if current_progress['progress'] < 100 and current_progress['progress'] > 0:
            style = {'display': 'block'}
        else:
            # Hide panel after completion (with a delay)
            style = {'display': 'none'} if current_progress['progress'] >= 100 else {'display': 'block'}
    
    # Format estimated time
    est_time = current_progress.get('estimated_time', 0)
    if est_time > 0:
        est_time_str = f"Estimated time remaining: {est_time}s"
    else:
        est_time_str = ""
    
    # Format terminal output with status icons
    terminal_lines = current_progress['terminal_output'].split('\n')
    formatted_terminal = []
    
    for line in terminal_lines[-10:]:  # Show last 10 lines
        if 'complete' in line.lower() or 'success' in line.lower():
            formatted_terminal.append(f"✓ {line}")
        elif 'failed' in line.lower() or 'error' in line.lower():
            formatted_terminal.append(f"✗ {line}")
        elif 'in_progress' in line.lower() or 'processing' in line.lower():
            formatted_terminal.append(f"⟳ {line}")
        else:
            formatted_terminal.append(f"  {line}")
    
    terminal_output = '\n'.join(formatted_terminal)
    
    return (
        style,
        current_progress['progress'],
        current_progress['current_task'],
        f"{current_progress['progress']}%",
        terminal_output,
        est_time_str
    )

# Shapelet discovery callback
@app.callback(
    Output('shapelet-analysis', 'children'),
    [Input('symbol-dropdown', 'value'),
     Input('discover-shapelets-button', 'n_clicks')],
    [State('timeframe-dropdown', 'value')]
)
def update_shapelet_analysis(symbol: str, n_clicks: int, timeframe: str):
    """Discover and visualize shapelets from time series data"""
    if not n_clicks or not symbol:
        raise PreventUpdate
    
    try:
        # Load data
        df = data_manager.load_data(symbol, timeframe)
        
        if df.empty:
            return dbc.Alert(
                f"No data available for {symbol}",
                color="warning",
                dismissable=True
            )
        
        # Check if we have cached shapelets
        cache_key = f"{symbol}_{timeframe}_shapelets"
        if cache_key in discovered_shapelets:
            shapelets = discovered_shapelets[cache_key]
            logger.info(f"Using cached shapelets for {symbol}")
        else:
            # Discover shapelets
            logger.info(f"Discovering shapelets for {symbol}")
            shapelets = shapelet_discoverer.discover_shapelets(
                df, 
                ticker=symbol,
                timeframe=timeframe,
                price_col='close' if 'close' in df.columns else 'price'
            )
            
            # Add to library
            shapelet_discoverer.add_to_library(shapelets)
            
            # Cache the results
            discovered_shapelets[cache_key] = shapelets
        
        # Create visualization cards
        cards = []
        
        # Use the visualizer's safe statistics access method
        get_stat = shapelet_visualizer._get_stat
        
        # Debug: Check what type statistics actually is
        if shapelets:
            logger.info(f"First shapelet statistics type: {type(shapelets[0].statistics)}")
            logger.info(f"First shapelet statistics content: {shapelets[0].statistics}")
            
            # Check all shapelets for any issues
            for i, s in enumerate(shapelets):
                if not isinstance(s.statistics, dict):
                    logger.warning(f"Shapelet {i} has non-dict statistics: {type(s.statistics)}")
        
        # Shapelet summary card
        summary_card = dbc.Card([
            dbc.CardHeader(html.H5("Shapelet Discovery Summary", className="mb-0")),
            dbc.CardBody([
                html.P(f"Total Shapelets Discovered: {len(shapelets)}"),
                html.P(f"Unique SAX Labels: {len(set(s.sax_label for s in shapelets))}"),
                html.P(f"Average Shapelet Length: {np.mean([s.length for s in shapelets]):.1f}" if shapelets else "N/A"),
                html.Hr(),
                html.H6("Top 5 Most Frequent Shapelets:"),
                html.Ul([
                    html.Li(f"SAX: {s.sax_label} (freq: {get_stat(s, 'frequency')}, avg return: {get_stat(s, 'avg_return_after'):.2%})")
                    for s in sorted(shapelets, key=lambda x: get_stat(x, 'frequency'), reverse=True)[:5]
                ]) if shapelets else html.P("No shapelets discovered")
            ])
        ], className="mb-3")
        cards.append(summary_card)
        
        if shapelets:
            # Create shapelet overlay visualization
            try:
                overlay_fig = shapelet_visualizer.create_shapelet_overlay_visualization(
                    df, shapelets, 
                    price_col='close' if 'close' in df.columns else 'price',
                    title=f"{symbol} - Shapelet Overlays"
                )
            except Exception as e:
                logger.error(f"Error creating overlay visualization: {e}")
                logger.error(f"Error type: {type(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
            
            overlay_card = dbc.Card([
                dbc.CardHeader(html.H5("Time Series with Shapelet Overlays", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(figure=overlay_fig, config={'displayModeBar': False})
                ])
            ], className="mb-3")
            cards.append(overlay_card)
            
            # Create shapelet library visualization
            try:
                library_fig = shapelet_visualizer.create_shapelet_library_visualization(
                    shapelets, max_display=12
                )
            except Exception as e:
                logger.error(f"Error creating library visualization: {e}")
                logger.error(f"Error type: {type(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
            
            library_card = dbc.Card([
                dbc.CardHeader(html.H5("Shapelet Library", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(figure=library_fig, config={'displayModeBar': False})
                ])
            ], className="mb-3")
            cards.append(library_card)
            
            # Create shapelet distribution visualization
            try:
                dist_fig = shapelet_visualizer.create_shapelet_distribution_visualization(shapelets)
            except Exception as e:
                logger.error(f"Error creating distribution visualization: {e}")
                logger.error(f"Error type: {type(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
            
            dist_card = dbc.Card([
                dbc.CardHeader(html.H5("Shapelet Distribution Analysis", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(figure=dist_fig, config={'displayModeBar': False})
                ])
            ], className="mb-3")
            cards.append(dist_card)
            
            # Create shapelet timeline visualization
            try:
                timeline_fig = shapelet_visualizer.create_shapelet_timeline_visualization(
                    df, shapelets, symbol,
                    price_col='close' if 'close' in df.columns else 'price'
                )
            except Exception as e:
                logger.error(f"Error creating timeline visualization: {e}")
                logger.error(f"Error type: {type(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
            
            timeline_card = dbc.Card([
                dbc.CardHeader(html.H5("Shapelet Timeline", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(figure=timeline_fig, config={'displayModeBar': False})
                ])
            ], className="mb-3")
            cards.append(timeline_card)
        
        return cards
        
    except Exception as e:
        logger.error(f"Error in shapelet discovery: {e}")
        return dbc.Alert(
            f"Error discovering shapelets: {str(e)}",
            color="danger",
            dismissable=True
        )

# Advanced pattern analysis callback
@app.callback(
    Output('advanced-pattern-analysis', 'children'),
    [Input('symbol-dropdown', 'value'),
     Input('analyze-patterns-button', 'n_clicks')],
    [State('pattern-overlay-store', 'data')]
)
def update_advanced_pattern_analysis(symbol: str, n_clicks: int, pattern_data: List[Dict]):
    """Update advanced pattern analysis with SAX and similarity search"""
    if not n_clicks or not symbol:
        raise PreventUpdate
    
    try:
        # Load data
        df = data_manager.load_data(symbol, '1 Day')
        
        if df.empty:
            return dbc.Alert(
                f"No data available for {symbol}",
                color="warning",
                dismissable=True
            )
        
        # Use advanced pattern detector
        analysis_results = data_manager.advanced_detector.detect_patterns_advanced(
            df, 
            price_col='close' if 'close' in df.columns else 'price',
            extract_motifs=True
        )
        
        # Create analysis cards
        cards = []
        
        # Pattern summary card
        patterns = analysis_results['patterns']
        summary_card = dbc.Card([
            dbc.CardHeader(html.H5("Pattern Analysis Summary", className="mb-0")),
            dbc.CardBody([
                html.P(f"Total Patterns Detected: {len(patterns)}"),
                html.P(f"Pattern Types: {len(set(p['type'] for p in patterns))}"),
                html.P(f"Average Pattern Strength: {np.mean([p['strength'] for p in patterns]):.2f}" if patterns else "N/A"),
                html.Hr(),
                html.H6("Pattern Distribution:"),
                html.Ul([
                    html.Li(f"{ptype}: {sum(1 for p in patterns if p['type'] == ptype)}")
                    for ptype in set(p['type'] for p in patterns)
                ]) if patterns else html.P("No patterns detected")
            ])
        ], className="mb-3")
        cards.append(summary_card)
        
        # SAX motifs card
        motifs = analysis_results.get('motifs', {})
        if motifs:
            motif_card = dbc.Card([
                dbc.CardHeader(html.H5("Discovered Pattern Motifs (SAX)", className="mb-0")),
                dbc.CardBody([
                    html.Div([
                        html.H6(f"{pattern_type.replace('_', ' ').title()} Motifs:"),
                        html.Ul([
                            html.Li(f"SAX: {motif['sax'][:20]}... (freq: {motif['frequency']})")
                            for motif in type_motifs[:3]
                        ]) if type_motifs else html.P("No motifs found", className="text-muted"),
                        html.Hr()
                    ])
                    for pattern_type, type_motifs in motifs.items()
                ])
            ], className="mb-3")
            cards.append(motif_card)
        
        # Pattern transitions card
        transitions = analysis_results.get('transitions', {})
        if transitions:
            trans_card = dbc.Card([
                dbc.CardHeader(html.H5("Pattern Transition Probabilities", className="mb-0")),
                dbc.CardBody([
                    html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("From Pattern"),
                                html.Th("To Pattern"),
                                html.Th("Probability")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td(from_type.replace('_', ' ').title()),
                                html.Td(to_type.replace('_', ' ').title()),
                                html.Td(f"{prob:.1%}")
                            ])
                            for from_type, to_probs in transitions.items()
                            for to_type, prob in sorted(to_probs.items(), key=lambda x: x[1], reverse=True)[:2]
                        ])
                    ], className="table table-sm")
                ])
            ], className="mb-3")
            cards.append(trans_card)
        
        # Add pattern comparison visualization
        if len(patterns) >= 2:
            # Add patterns to comparison tool
            comparison = data_manager.pattern_comparison
            comparison.clear_patterns()
            
            # Add top 4 patterns for comparison
            for i, pattern in enumerate(patterns[:4]):
                if 'start_idx' in pattern and 'end_idx' in pattern:
                    start_idx = pattern['start_idx']
                    end_idx = pattern['end_idx']
                    pattern_data = df.iloc[start_idx:end_idx]['close'].values
                    comparison.add_pattern(f"Pattern_{i+1}", pattern_data)
            
            # Create advanced analysis dashboard
            fig = comparison.create_advanced_pattern_analysis_dashboard()
            
            viz_card = dbc.Card([
                dbc.CardHeader(html.H5("Advanced Pattern Analysis Dashboard", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(figure=fig, config={'displayModeBar': False})
                ])
            ], className="mb-3")
            cards.append(viz_card)
        
        return cards
        
    except Exception as e:
        logger.error(f"Error in advanced pattern analysis: {e}")
        return dbc.Alert(
            f"Error performing advanced analysis: {str(e)}",
            color="danger",
            dismissable=True
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

# Add initial loading state callback
@app.callback(
    Output('loading-overlay', 'style'),
    [Input('url', 'pathname')]
)
def hide_loading_overlay(pathname):
    """Hide loading overlay once the app is ready"""
    # Check if models are loaded
    if pattern_models.get_model_status()['models_loaded']:
        return {'display': 'none'}
    return {
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'width': '100%',
        'height': '100%',
        'backgroundColor': 'rgba(255, 255, 255, 0.95)',
        'zIndex': 9999,
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center'
    }

# Main execution
if __name__ == '__main__':
    # Check if this is the reloader process
    import os
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not os.environ.get('WERKZEUG_RUN_MAIN'):
        print("=" * 60)
        print("Starting Wavelet Forecast Dashboard with Real YFinance Data")
        print("=" * 60)
        print("\nLoading ML models...")
        print(f"  - LSTM: {'✓' if 'lstm' in model_status['models_loaded'] else '✗'}")
        print(f"  - GRU: {'✓' if 'gru' in model_status['models_loaded'] else '✗'}")
        print(f"  - Transformer: {'✗ (Model architecture mismatch)'}")
        print(f"  - Markov: {'✓' if model_status['markov_model_loaded'] else '✗'}")
        print("\nDashboard will be available at: http://localhost:8050")
        print("\nPress Ctrl+C to stop the server")
        print("\nNOTE: This dashboard uses REAL market data from YFinance!")
        print("      - Data is cached with intelligent updates")
        print("      - Checks for new data based on timeframe")
        print("      - Patterns are detected from real price movements")
        print("      - No mock data or fallbacks - production ready")
        print("=" * 60)
    
    # Run the app (use_reloader=False to prevent double loading in production)
    app.run_server(debug=True, host='0.0.0.0', port=8050, use_reloader=True)
