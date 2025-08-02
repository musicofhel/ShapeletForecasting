"""
Prediction Callbacks Module

This module contains all callback functions for the prediction functionality
of the forecasting dashboard.
"""

from dash import Input, Output, State, callback_context, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger(__name__)

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

def register_prediction_callbacks(app, data_manager, performance_monitor=None):
    """Register all prediction-related callbacks"""
    
    @app.callback(
        Output('pattern-tab-content', 'children'),
        [Input('pattern-tabs', 'active_tab'),
         Input('symbol-dropdown', 'value')]
    )
    def update_pattern_explorer(active_tab: str, symbol: str):
        """Update pattern explorer content based on active tab"""
        start_time = time.time()
        
        try:
            if active_tab == "library":
                # Get patterns from data manager
                patterns = data_manager.get_patterns(symbol)
                
                # Create pattern cards
                pattern_cards = []
                for pattern in patterns[:10]:  # Show top 10
                    card = dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H5(pattern['name'], className="mb-1"),
                                    html.P(f"Frequency: {pattern['frequency']} occurrences", 
                                          className="text-muted mb-0")
                                ], width=8),
                                dbc.Col([
                                    html.Div([
                                        html.H6(f"{pattern['avg_return']:.2f}%", 
                                               className="text-success mb-0"),
                                        html.Small("Avg Return", className="text-muted")
                                    ], className="text-end")
                                ], width=4)
                            ]),
                            dbc.Progress(
                                value=pattern['confidence'] * 100,
                                label=f"{pattern['confidence']:.0%}",
                                color="primary",
                                className="mt-2"
                            )
                        ])
                    ], className="mb-2")
                    pattern_cards.append(card)
                
                content = html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Pattern Library", className="mb-3"),
                            html.Div(pattern_cards)
                        ])
                    ])
                ])
                
            elif active_tab == "details":
                # Pattern details view
                content = html.Div([
                    dbc.Row([
                        dbc.Col([
                            dbc.Select(
                                id="pattern-select",
                                options=[
                                    {"label": f"Pattern {chr(65+i)}", "value": f"pattern_{i}"}
                                    for i in range(10)
                                ],
                                value="pattern_0",
                                className="mb-3"
                            )
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(
                                id="pattern-detail-chart",
                                figure=create_pattern_detail_chart(),
                                config={'displayModeBar': False}
                            )
                        ], lg=6),
                        dbc.Col([
                            html.H6("Pattern Statistics"),
                            create_pattern_stats_table()
                        ], lg=6)
                    ])
                ])
                
            elif active_tab == "shapelets":
                # Shapelet library view
                content = html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Shapelet Library", className="mb-3"),
                            html.P("Discovered shapelets with SAX labels", className="text-muted"),
                            html.Hr(),
                            # Get cached shapelets if available
                            html.Div(id="shapelet-library-content", children=[
                                dbc.Alert(
                                    [
                                        html.I(className="fas fa-info-circle me-2"),
                                        "Click 'Discover Shapelets' button to populate the library"
                                    ],
                                    color="info"
                                )
                            ])
                        ])
                    ])
                ])
                
            elif active_tab == "advanced":
                # Advanced analysis view
                content = html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Advanced Pattern Analysis", className="mb-3"),
                            html.P("Click 'Analyze Patterns' button to perform advanced analysis", 
                                   className="text-muted"),
                            html.Hr(),
                            html.Div(id="advanced-analysis-placeholder")
                        ])
                    ])
                ])
                
            else:  # history tab (fallback)
                # Historical performance view
                content = html.Div([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(
                                id="pattern-history-chart",
                                figure=create_pattern_history_chart(),
                                config={'displayModeBar': False}
                            )
                        ])
                    ])
                ])
            
            # Record performance - only if monitor is provided
            if performance_monitor:
                duration = time.time() - start_time
                performance_monitor.record_callback_time(duration)
            
            return content
            
        except Exception as e:
            logger.error(f"Error updating pattern explorer: {e}")
            return dbc.Alert("Error loading pattern data", color="danger")
    
    @app.callback(
        Output('last-update-time', 'children'),
        [Input('refresh-interval', 'n_intervals')]
    )
    def update_timestamp(n_intervals: int):
        """Update last refresh timestamp"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @app.callback(
        Output('download-chart', 'n_clicks'),
        [Input('download-chart', 'n_clicks')],
        [State('main-time-series', 'figure')]
    )
    def download_chart(n_clicks: int, figure: Dict):
        """Handle chart download"""
        if not n_clicks:
            raise PreventUpdate
        
        # In a real implementation, this would trigger a download
        # For now, we'll just reset the click count
        return 0
    
    @app.callback(
        [Output('pattern-detail-chart', 'figure'),
         Output('pattern-stats-table', 'children')],
        [Input('pattern-select', 'value')],
        prevent_initial_call=True
    )
    def update_pattern_details(pattern_id: str):
        """Update pattern detail view"""
        start_time = time.time()
        
        try:
            # Create detailed pattern visualization
            fig = create_pattern_detail_chart(pattern_id)
            
            # Create statistics table
            stats_table = create_pattern_stats_table(pattern_id)
            
            # Record performance - only if monitor is provided
            if performance_monitor:
                duration = time.time() - start_time
                performance_monitor.record_callback_time(duration)
            
            return fig, stats_table
            
        except Exception as e:
            logger.error(f"Error updating pattern details: {e}")
            return go.Figure(), html.Div("Error loading pattern details")
    
    # Advanced prediction callbacks
    @app.callback(
        Output('prediction-store', 'data'),
        [Input('predict-button', 'n_clicks')],
        [State('symbol-dropdown', 'value'),
         State('timeframe-dropdown', 'value'),
         State('prediction-horizon-slider', 'value')],
        prevent_initial_call=True
    )
    def generate_advanced_predictions(n_clicks: int, symbol: str, 
                                    timeframe: str, horizon: int):
        """Generate advanced predictions using multiple models"""
        if not n_clicks:
            raise PreventUpdate
        
        start_time = time.time()
        
        try:
            # Simulate advanced prediction generation
            predictions = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'timeframe': timeframe,
                'horizon': horizon,
                'predictions': {
                    'lstm': {
                        'pattern': 'pattern_A',
                        'confidence': 0.82,
                        'expected_return': 2.5
                    },
                    'transformer': {
                        'pattern': 'pattern_B',
                        'confidence': 0.78,
                        'expected_return': 1.8
                    },
                    'ensemble': {
                        'pattern': 'pattern_A',
                        'confidence': 0.85,
                        'expected_return': 2.3
                    }
                },
                'risk_metrics': {
                    'var_95': -1.5,
                    'cvar_95': -2.1,
                    'sharpe_ratio': 1.8
                }
            }
            
            # Record performance - only if monitor is provided
            if performance_monitor:
                duration = time.time() - start_time
                performance_monitor.record_callback_time(duration)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return {'error': str(e)}
    
    # Real-time update callbacks
    @app.callback(
        Output('refresh-chart', 'n_clicks'),
        [Input('refresh-chart', 'n_clicks')],
        [State('main-time-series', 'figure')]
    )
    def force_refresh(n_clicks: int, current_figure: Dict):
        """Force refresh of chart data"""
        if not n_clicks:
            raise PreventUpdate
        
        # Clear cache for fresh data
        data_manager.data_cache.clear()
        
        return 0
    
    # Pattern matching callbacks
    @app.callback(
        Output('pattern-match-results', 'children'),
        [Input({'type': 'pattern-match', 'index': ALL}, 'n_clicks')],
        [State('symbol-dropdown', 'value')]
    )
    def handle_pattern_matching(n_clicks: List[int], symbol: str):
        """Handle pattern matching requests"""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        start_time = time.time()
        
        try:
            # Identify which pattern was clicked
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            # Perform pattern matching
            matches = perform_pattern_matching(symbol, button_id)
            
            # Create results display
            results = create_match_results_display(matches)
            
            # Record performance - only if monitor is provided
            if performance_monitor:
                duration = time.time() - start_time
                performance_monitor.record_callback_time(duration)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in pattern matching: {e}")
            return dbc.Alert("Error performing pattern matching", color="danger")

# Helper functions
def create_pattern_detail_chart(pattern_id: Optional[str] = None) -> go.Figure:
    """Create detailed pattern visualization"""
    # Generate sample pattern data
    x = np.linspace(0, 100, 100)
    y = np.sin(x/10) * np.exp(-x/50) + np.random.randn(100) * 0.1
    
    fig = go.Figure()
    
    # Add pattern trace
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Pattern Shape',
        line=dict(color='blue', width=2)
    ))
    
    # Add confidence bands
    upper = y + 0.2
    lower = y - 0.2
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,200,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Band',
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"Pattern {pattern_id.split('_')[1] if pattern_id else 'A'} Shape",
        xaxis_title="Time Steps",
        yaxis_title="Normalized Value",
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
        template='plotly_white'
    )
    
    return fig

def create_pattern_stats_table(pattern_id: Optional[str] = None) -> html.Div:
    """Create pattern statistics table"""
    stats = {
        'Occurrences': np.random.randint(20, 100),
        'Avg Duration': f"{np.random.randint(5, 20)} periods",
        'Success Rate': f"{np.random.uniform(0.6, 0.9):.1%}",
        'Avg Return': f"{np.random.uniform(-2, 5):.2f}%",
        'Max Drawdown': f"{np.random.uniform(-5, -1):.2f}%",
        'Sharpe Ratio': f"{np.random.uniform(0.5, 2.5):.2f}"
    }
    
    rows = []
    for key, value in stats.items():
        rows.append(
            html.Tr([
                html.Td(key, className="text-muted"),
                html.Td(value, className="text-end fw-bold")
            ])
        )
    
    # Wrap the table in a Div as expected by the test
    return html.Div(
        html.Table(
            html.Tbody(rows),
            className="table table-sm table-borderless"
        )
    )

def create_pattern_history_chart() -> go.Figure:
    """Create pattern historical performance chart"""
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Pattern Occurrences', 'Cumulative Performance'),
        row_heights=[0.4, 0.6]
    )
    
    # Pattern occurrences
    occurrences = np.random.poisson(3, 90)
    fig.add_trace(
        go.Bar(
            x=dates,
            y=occurrences,
            name='Occurrences',
            marker_color='lightblue'
        ),
        row=1, col=1
    )
    
    # Cumulative performance
    returns = np.random.randn(90) * 0.02
    cumulative = (1 + returns).cumprod()
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=cumulative,
            name='Cumulative Return',
            line=dict(color='green', width=2)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=40),
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Return", row=2, col=1)
    
    return fig

def perform_pattern_matching(symbol: str, pattern_id: str) -> List[Dict]:
    """Perform pattern matching analysis"""
    # Simulate pattern matching results
    matches = []
    for i in range(5):
        matches.append({
            'date': datetime.now() - timedelta(days=i*10),
            'similarity': np.random.uniform(0.7, 0.95),
            'duration': np.random.randint(5, 20),
            'return': np.random.uniform(-3, 5)
        })
    
    return matches

def create_match_results_display(matches: List[Dict]) -> html.Div:
    """Create pattern match results display"""
    if not matches:
        return dbc.Alert("No matches found", color="info")
    
    cards = []
    for match in matches:
        card = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6(match['date'].strftime("%Y-%m-%d")),
                        html.P(f"Similarity: {match['similarity']:.1%}", 
                              className="mb-0 text-muted")
                    ], width=6),
                    dbc.Col([
                        html.P(f"Duration: {match['duration']} periods", 
                              className="mb-0"),
                        html.P(f"Return: {match['return']:.2f}%",
                              className=f"mb-0 text-{'success' if match['return'] > 0 else 'danger'}")
                    ], width=6, className="text-end")
                ])
            ])
        ], className="mb-2")
        cards.append(card)
    
    return html.Div(cards)

# Async helper functions
async def async_compute_predictions(data: pd.DataFrame, horizon: int) -> Dict:
    """Asynchronously compute predictions"""
    # Simulate async computation
    await asyncio.sleep(0.1)
    
    return {
        'pattern': f'pattern_{np.random.randint(0, 5)}',
        'confidence': np.random.uniform(0.7, 0.95),
        'expected_return': np.random.uniform(-2, 5)
    }

def run_async_prediction(data: pd.DataFrame, horizon: int) -> Dict:
    """Run async prediction in thread pool"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(async_compute_predictions(data, horizon))
    loop.close()
    return result
