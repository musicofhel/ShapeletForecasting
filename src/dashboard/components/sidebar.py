"""
Interactive Sidebar Component for Financial Wavelet Prediction System

This module provides a comprehensive sidebar with advanced ticker selection,
date range controls, pattern filters, and real-time data management.
"""

import dash
from dash import dcc, html, Input, Output, State, callback, MATCH, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Set
import json
from datetime import datetime, timedelta, date
import logging
from dash.exceptions import PreventUpdate
from src.dashboard.data_utils import data_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractiveSidebar:
    """Advanced sidebar component with comprehensive selection controls."""
    
    def __init__(self):
        """Initialize sidebar with default configurations."""
        # Default ticker configurations
        self.default_tickers = ["BTC-USD", "ETH-USD", "SPY"]
        self.ticker_categories = {
            "Crypto": ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "LINK-USD", "UNI-USD"],
            "Stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"],
            "ETFs": ["SPY", "QQQ", "DIA", "IWM", "VTI", "VOO"],
            "Commodities": ["GLD", "SLV", "USO", "UNG", "DBA"],
            "Forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"],
            "Indices": ["^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX"]
        }
        
        # Pattern configurations
        self.pattern_categories = {
            "Wavelets": {
                "Haar": {"min_scale": 1, "max_scale": 8},
                "Daubechies": {"min_scale": 2, "max_scale": 10},
                "Morlet": {"min_scale": 1, "max_scale": 12},
                "Mexican Hat": {"min_scale": 1, "max_scale": 10}
            },
            "Shapelets": {
                "Triangle": {"min_length": 10, "max_length": 50},
                "Rectangle": {"min_length": 15, "max_length": 60},
                "Head & Shoulders": {"min_length": 20, "max_length": 80},
                "Double Top/Bottom": {"min_length": 25, "max_length": 100}
            },
            "Motifs": {
                "Repeating": {"min_occurrences": 2, "max_occurrences": 10},
                "Alternating": {"min_occurrences": 3, "max_occurrences": 15},
                "Seasonal": {"min_period": 7, "max_period": 365},
                "Cyclic": {"min_period": 5, "max_period": 50}
            }
        }
        
        # Quality thresholds
        self.quality_metrics = {
            "confidence": {"min": 0, "max": 1, "default": 0.7},
            "significance": {"min": 0, "max": 1, "default": 0.6},
            "stability": {"min": 0, "max": 1, "default": 0.5},
            "robustness": {"min": 0, "max": 1, "default": 0.8}
        }
        
        # Date range presets
        self.date_presets = {
            "1D": 1, "1W": 7, "1M": 30, "3M": 90,
            "6M": 180, "1Y": 365, "2Y": 730, "5Y": 1825
        }
        
    def create_ticker_selector(self) -> dbc.Card:
        """Create advanced multi-select ticker dropdown with categories."""
        return dbc.Card([
            dbc.CardHeader([
                html.H5("Ticker Selection", className="mb-0"),
                dbc.Button(
                    "⚙️",
                    id="ticker-settings-btn",
                    color="link",
                    size="sm",
                    className="float-end"
                )
            ]),
            dbc.CardBody([
                # Category tabs
                dbc.Tabs(
                    id="ticker-category-tabs",
                    children=[
                        dbc.Tab(label=category, tab_id=f"tab-{category.lower()}")
                        for category in self.ticker_categories.keys()
                    ] + [dbc.Tab(label="Custom", tab_id="tab-custom")],
                    active_tab="tab-crypto"
                ),
                
                # Multi-select dropdown
                html.Div([
                    dbc.Label("Select Tickers:", className="mt-3"),
                    dcc.Dropdown(
                        id="ticker-multi-select",
                        options=[],
                        value=self.default_tickers,
                        multi=True,
                        clearable=True,
                        searchable=True,
                        placeholder="Select or search tickers...",
                        persistence=True,
                        persistence_type="session",
                        style={"width": "100%"}
                    )
                ]),
                
                # Quick actions
                dbc.ButtonGroup([
                    dbc.Button("Select All", id="select-all-tickers", size="sm", color="info"),
                    dbc.Button("Clear", id="clear-tickers", size="sm", color="warning"),
                    dbc.Button("Popular", id="popular-tickers", size="sm", color="success")
                ], className="mt-2"),
                
                # Custom ticker input
                dbc.Collapse([
                    html.Hr(),
                    dbc.Label("Add Custom Ticker:"),
                    dbc.InputGroup([
                        dbc.Input(
                            id="custom-ticker-input",
                            placeholder="Enter ticker symbol",
                            type="text",
                            style={"textTransform": "uppercase"}
                        ),
                        dbc.Button("Add", id="add-custom-ticker", color="primary")
                    ], size="sm"),
                    html.Div(id="ticker-validation-msg", className="mt-2")
                ], id="custom-ticker-collapse", is_open=False),
                
                # Selected tickers display
                html.Div([
                    html.Hr(),
                    html.H6("Selected Tickers:"),
                    html.Div(id="selected-tickers-display", className="mt-2")
                ])
            ])
        ], className="mb-3")
    
    def create_date_range_picker(self) -> dbc.Card:
        """Create comprehensive date range picker with presets."""
        return dbc.Card([
            dbc.CardHeader("Date Range Selection"),
            dbc.CardBody([
                # Preset buttons
                dbc.Label("Quick Select:"),
                dbc.ButtonGroup([
                    dbc.Button(
                        preset,
                        id={"type": "date-preset", "index": preset},
                        size="sm",
                        color="secondary",
                        outline=True
                    )
                    for preset in self.date_presets.keys()
                ], size="sm", className="mb-3"),
                
                # Date range picker
                dbc.Label("Custom Range:"),
                dcc.DatePickerRange(
                    id="date-range-picker",
                    start_date=(datetime.now() - timedelta(days=90)).date(),
                    end_date=datetime.now().date(),
                    display_format="YYYY-MM-DD",
                    style={"width": "100%"},
                    persistence=True,
                    persistence_type="session"
                ),
                
                # Time granularity selector
                html.Hr(),
                dbc.Label("Time Granularity:"),
                dbc.RadioItems(
                    id="time-granularity",
                    options=[
                        {"label": "1m", "value": "1m"},
                        {"label": "5m", "value": "5m"},
                        {"label": "15m", "value": "15m"},
                        {"label": "1h", "value": "1h"},
                        {"label": "4h", "value": "4h"},
                        {"label": "1d", "value": "1d"},
                        {"label": "1wk", "value": "1wk"}
                    ],
                    value="1h",
                    inline=True,
                    persistence=True
                ),
                
                # Date range info
                html.Div(id="date-range-info", className="mt-3 text-muted small")
            ])
        ], className="mb-3")
    
    def create_pattern_filters(self) -> dbc.Card:
        """Create pattern type filters with categories."""
        return dbc.Card([
            dbc.CardHeader([
                html.H5("Pattern Type Filters", className="mb-0"),
                dbc.Badge(id="pattern-count-badge", color="primary", className="float-end")
            ]),
            dbc.CardBody([
                # Pattern category accordion
                dbc.Accordion([
                    dbc.AccordionItem([
                        dbc.Checklist(
                            id=f"{category.lower()}-patterns",
                            options=[
                                {"label": pattern, "value": pattern}
                                for pattern in patterns.keys()
                            ],
                            value=[],
                            inline=False,
                            persistence=True
                        )
                    ], title=category)
                    for category, patterns in self.pattern_categories.items()
                ], id="pattern-accordion", start_collapsed=True),
                
                # Pattern complexity filter
                html.Hr(),
                dbc.Label("Pattern Complexity:"),
                dbc.Select(
                    id="pattern-complexity",
                    options=[
                        {"label": "Simple", "value": "simple"},
                        {"label": "Medium", "value": "medium"},
                        {"label": "Complex", "value": "complex"},
                        {"label": "All", "value": "all"}
                    ],
                    value="medium",
                    persistence=True
                ),
                
                # Pattern actions
                dbc.ButtonGroup([
                    dbc.Button("Select All", id="select-all-patterns", size="sm"),
                    dbc.Button("Clear All", id="clear-all-patterns", size="sm"),
                    dbc.Button("Recommended", id="recommended-patterns", size="sm")
                ], className="mt-3")
            ])
        ], className="mb-3")
    
    def create_quality_thresholds(self) -> dbc.Card:
        """Create quality threshold sliders."""
        return dbc.Card([
            dbc.CardHeader("Quality Thresholds"),
            dbc.CardBody([
                # Create slider for each quality metric
                *[self._create_threshold_slider(metric, config)
                  for metric, config in self.quality_metrics.items()],
                
                # Preset quality levels
                html.Hr(),
                dbc.Label("Preset Levels:"),
                dbc.ButtonGroup([
                    dbc.Button("Conservative", id="quality-conservative", size="sm", color="success"),
                    dbc.Button("Balanced", id="quality-balanced", size="sm", color="warning"),
                    dbc.Button("Aggressive", id="quality-aggressive", size="sm", color="danger")
                ], className="mt-2"),
                
                # Quality summary
                html.Div(id="quality-summary", className="mt-3 small text-muted")
            ])
        ], className="mb-3")
    
    def _create_threshold_slider(self, metric: str, config: Dict) -> html.Div:
        """Create individual threshold slider."""
        return html.Div([
            dbc.Label(f"{metric.capitalize()} Threshold:"),
            dcc.Slider(
                id=f"{metric}-threshold",
                min=config["min"],
                max=config["max"],
                step=0.05,
                value=config["default"],
                marks={
                    config["min"]: f"{config['min']:.0%}",
                    config["max"]: f"{config['max']:.0%}"
                },
                tooltip={"placement": "bottom", "always_visible": True},
                persistence=True
            ),
            html.Div(id=f"{metric}-value", className="text-center small mb-3")
        ])
    
    def create_realtime_toggle(self) -> dbc.Card:
        """Create real-time data toggle with status indicators."""
        return dbc.Card([
            dbc.CardHeader("Data Mode"),
            dbc.CardBody([
                # Real-time toggle
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Real-time Data:"),
                        dbc.Switch(
                            id="realtime-data-toggle",
                            label="",
                            value=False,
                            persistence=True
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Auto-refresh:"),
                        dbc.Switch(
                            id="auto-refresh-toggle",
                            label="",
                            value=False,
                            persistence=True
                        )
                    ], width=6)
                ]),
                
                # Refresh interval
                dbc.Collapse([
                    html.Hr(),
                    dbc.Label("Refresh Interval (seconds):"),
                    dbc.Input(
                        id="refresh-interval",
                        type="number",
                        min=1,
                        max=300,
                        value=30,
                        persistence=True
                    )
                ], id="refresh-settings", is_open=False),
                
                # Connection status
                html.Hr(),
                dbc.Alert(
                    id="connection-status",
                    children=[
                    dbc.Spinner(size="sm", spinner_class_name="me-2"),
                        "Checking connection..."
                    ],
                    color="info",
                    className="mb-0"
                ),
                
                # Data source info
                html.Div(id="data-source-info", className="small text-muted mt-2")
            ])
        ], className="mb-3")
    
    def create_advanced_settings(self) -> dbc.Card:
        """Create advanced settings panel."""
        return dbc.Card([
            dbc.CardHeader([
                "Advanced Settings",
                dbc.Button(
                    "▼",
                    id="advanced-toggle",
                    color="link",
                    size="sm",
                    className="float-end"
                )
            ]),
            dbc.Collapse([
                dbc.CardBody([
                    # Cache settings
                    html.H6("Cache Settings:"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Cache Duration (min):"),
                            dbc.Input(
                                id="cache-duration",
                                type="number",
                                min=1,
                                max=1440,
                                value=60,
                                persistence=True
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Button(
                                "Clear Cache",
                                id="clear-cache-btn",
                                color="warning",
                                size="sm",
                                className="mt-4"
                            )
                        ], width=6)
                    ]),
                    
                    # Performance settings
                    html.Hr(),
                    html.H6("Performance Settings:"),
                    dbc.Checklist(
                        id="performance-options",
                        options=[
                            {"label": "Enable GPU acceleration", "value": "gpu"},
                            {"label": "Use parallel processing", "value": "parallel"},
                            {"label": "Optimize memory usage", "value": "memory"},
                            {"label": "Enable data compression", "value": "compress"}
                        ],
                        value=["parallel", "memory"],
                        persistence=True
                    ),
                    
                    # Export settings
                    html.Hr(),
                    html.H6("Export Settings:"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "Export Config",
                                id="export-config-btn",
                                color="info",
                                size="sm",
                                className="w-100"
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Button(
                                "Import Config",
                                id="import-config-btn",
                                color="info",
                                size="sm",
                                className="w-100"
                            )
                        ], width=6)
                    ]),
                    
                    # Reset button
                    html.Hr(),
                    dbc.Button(
                        "Reset All Settings",
                        id="reset-all-btn",
                        color="danger",
                        size="sm",
                        className="w-100"
                    )
                ])
            ], id="advanced-collapse", is_open=False)
        ], className="mb-3")
    
    def create_action_button(self) -> dbc.Card:
        """Create action button with loading indicator."""
        return dbc.Card([
            dbc.CardBody([
                # GO button
                dbc.Button(
                    [
                        html.I(className="fas fa-play me-2"),
                        "Apply Settings & Analyze"
                    ],
                    id="apply-settings-btn",
                    color="success",
                    size="lg",
                    className="w-100 mb-3",
                    disabled=False
                ),
                
                # Loading progress
                dbc.Collapse([
                    dbc.Progress(
                        id="analysis-progress",
                        value=0,
                        max=100,
                        striped=True,
                        animated=True,
                        className="mb-2",
                        style={"height": "25px"}
                    ),
                    html.Div(
                        id="progress-status",
                        className="text-center small text-muted mb-2"
                    ),
                    html.Div(
                        id="time-estimate",
                        className="text-center small"
                    )
                ], id="progress-collapse", is_open=False),
                
                # Status messages
                html.Div(id="action-status", className="mt-2")
            ])
        ], className="mb-3 border-success")
    
    def create_sidebar_layout(self) -> dbc.Col:
        """Create complete sidebar layout."""
        return dbc.Col([
            html.H4("Control Panel", className="mb-3"),
            
            # Main controls
            self.create_ticker_selector(),
            self.create_date_range_picker(),
            self.create_pattern_filters(),
            self.create_quality_thresholds(),
            self.create_realtime_toggle(),
            self.create_advanced_settings(),
            
            # Action button with progress
            self.create_action_button(),
            
            # Hidden stores for state management
            dcc.Store(id="sidebar-state", storage_type="session"),
            dcc.Store(id="ticker-cache", storage_type="local"),
            dcc.Store(id="pattern-cache", storage_type="local"),
            dcc.Store(id="settings-cache", storage_type="local"),
            dcc.Store(id="analysis-state", storage_type="memory"),
            
            # Intervals for updates
            dcc.Interval(id="realtime-interval", interval=30000, disabled=True),
            dcc.Interval(id="connection-check-interval", interval=60000),
            dcc.Interval(id="progress-interval", interval=500, disabled=True),
            
            # Hidden upload component for config import
            dcc.Upload(id="config-upload", style={"display": "none"})
        ], width=3, className="sidebar-container")
    
    @staticmethod
    def register_callbacks(app):
        """Register all sidebar callbacks."""
        
        # Ticker category selection
        @app.callback(
            Output("ticker-multi-select", "options"),
            [Input("ticker-category-tabs", "active_tab")],
            [State("ticker-cache", "data")]
        )
        def update_ticker_options(active_tab, ticker_cache):
            """Update ticker options based on selected category."""
            sidebar = InteractiveSidebar()
            
            if active_tab == "tab-custom":
                # Load custom tickers from cache
                custom_tickers = ticker_cache.get("custom", []) if ticker_cache else []
                return [{"label": t, "value": t} for t in custom_tickers]
            else:
                # Get category tickers
                category = active_tab.replace("tab-", "").title()
                tickers = sidebar.ticker_categories.get(category, [])
                return [{"label": t, "value": t} for t in tickers]
        
        # Ticker selection actions
        @app.callback(
            Output("ticker-multi-select", "value"),
            [Input("select-all-tickers", "n_clicks"),
             Input("clear-tickers", "n_clicks"),
             Input("popular-tickers", "n_clicks")],
            [State("ticker-multi-select", "options"),
             State("ticker-multi-select", "value")]
        )
        def handle_ticker_actions(select_all, clear, popular, options, current):
            """Handle ticker selection actions."""
            ctx = dash.callback_context
            if not ctx.triggered:
                return current
            
            trigger = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if trigger == "select-all-tickers":
                return [opt["value"] for opt in options]
            elif trigger == "clear-tickers":
                return []
            elif trigger == "popular-tickers":
                popular_list = ["BTC-USD", "ETH-USD", "SPY", "AAPL", "TSLA"]
                return [t for t in popular_list if any(opt["value"] == t for opt in options)]
            
            return current
        
        # Custom ticker addition
        @app.callback(
            [Output("ticker-cache", "data"),
             Output("ticker-validation-msg", "children"),
             Output("custom-ticker-input", "value")],
            [Input("add-custom-ticker", "n_clicks")],
            [State("custom-ticker-input", "value"),
             State("ticker-cache", "data")]
        )
        def add_custom_ticker(n_clicks, ticker, cache):
            """Add and validate custom ticker."""
            if not n_clicks or not ticker:
                return cache, "", ""
            
            ticker = ticker.upper()
            cache = cache or {"custom": []}
            
            # Validate ticker using data_manager
            try:
                # Quick validation using data_manager
                info = data_manager.get_ticker_info(ticker)
                if info and info.get('longName'):  # Check if we got valid info
                    if ticker not in cache.get("custom", []):
                        cache.setdefault("custom", []).append(ticker)
                    msg = dbc.Alert(f"✓ {ticker} added successfully", color="success", dismissable=True)
                    return cache, msg, ""
            except Exception as e:
                logger.warning(f"Failed to validate ticker {ticker}: {e}")
            
            msg = dbc.Alert(f"✗ {ticker} is not a valid ticker", color="danger", dismissable=True)
            return cache, msg, ticker
        
        # Date preset handling
        @app.callback(
            [Output("date-range-picker", "start_date"),
             Output("date-range-picker", "end_date")],
            [Input({"type": "date-preset", "index": ALL}, "n_clicks")]
        )
        def handle_date_presets(n_clicks_list):
            """Handle date preset button clicks."""
            ctx = dash.callback_context
            if not ctx.triggered or not any(n_clicks_list):
                raise PreventUpdate
            
            # Find which button was clicked
            button_id = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])
            preset = button_id["index"]
            
            sidebar = InteractiveSidebar()
            days = sidebar.date_presets.get(preset, 30)
            
            end_date = datetime.now().date()
            start_date = (datetime.now() - timedelta(days=days)).date()
            
            return start_date, end_date
        
        # Date range info update
        @app.callback(
            Output("date-range-info", "children"),
            [Input("date-range-picker", "start_date"),
             Input("date-range-picker", "end_date"),
             Input("time-granularity", "value")]
        )
        def update_date_info(start_date, end_date, granularity):
            """Update date range information display."""
            if not start_date or not end_date:
                return ""
            
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            days = (end - start).days
            
            # Calculate approximate data points
            granularity_map = {
                "1m": 1440, "5m": 288, "15m": 96,
                "1h": 24, "4h": 6, "1d": 1, "1wk": 1/7
            }
            points_per_day = granularity_map.get(granularity, 24)
            total_points = int(days * points_per_day)
            
            return f"Range: {days} days | ~{total_points:,} data points"
        
        # Pattern count update
        @app.callback(
            Output("pattern-count-badge", "children"),
            [Input("wavelets-patterns", "value"),
             Input("shapelets-patterns", "value"),
             Input("motifs-patterns", "value")]
        )
        def update_pattern_count(wavelets, shapelets, motifs):
            """Update pattern count badge."""
            total = len(wavelets or []) + len(shapelets or []) + len(motifs or [])
            return f"{total} selected"
        
        # Pattern selection actions
        @app.callback(
            [Output("wavelets-patterns", "value"),
             Output("shapelets-patterns", "value"),
             Output("motifs-patterns", "value")],
            [Input("select-all-patterns", "n_clicks"),
             Input("clear-all-patterns", "n_clicks"),
             Input("recommended-patterns", "n_clicks")],
            [State("wavelets-patterns", "options"),
             State("shapelets-patterns", "options"),
             State("motifs-patterns", "options")]
        )
        def handle_pattern_actions(select_all, clear_all, recommended, 
                                 wavelets_opt, shapelets_opt, motifs_opt):
            """Handle pattern selection actions."""
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate
            
            trigger = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if trigger == "select-all-patterns":
                return (
                    [opt["value"] for opt in wavelets_opt],
                    [opt["value"] for opt in shapelets_opt],
                    [opt["value"] for opt in motifs_opt]
                )
            elif trigger == "clear-all-patterns":
                return [], [], []
            elif trigger == "recommended-patterns":
                # Select recommended patterns
                return ["Morlet", "Daubechies"], ["Head & Shoulders", "Triangle"], ["Seasonal"]
            
            raise PreventUpdate
        
        # Quality threshold updates
        @app.callback(
            [Output(f"{metric}-value", "children") for metric in InteractiveSidebar().quality_metrics.keys()],
            [Input(f"{metric}-threshold", "value") for metric in InteractiveSidebar().quality_metrics.keys()]
        )
        def update_threshold_displays(*values):
            """Update threshold value displays."""
            return [f"{value:.0%}" for value in values]
        
        # Quality preset handling
        @app.callback(
            [Output(f"{metric}-threshold", "value") for metric in InteractiveSidebar().quality_metrics.keys()],
            [Input("quality-conservative", "n_clicks"),
             Input("quality-balanced", "n_clicks"),
             Input("quality-aggressive", "n_clicks")]
        )
        def handle_quality_presets(conservative, balanced, aggressive):
            """Handle quality preset selections."""
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate
            
            trigger = ctx.triggered[0]["prop_id"].split(".")[0]
            
            presets = {
                "quality-conservative": [0.8, 0.8, 0.7, 0.9],
                "quality-balanced": [0.7, 0.6, 0.5, 0.8],
                "quality-aggressive": [0.5, 0.4, 0.3, 0.6]
            }
            
            return presets.get(trigger, [0.7, 0.6, 0.5, 0.8])
        
        # Real-time toggle handling
        @app.callback(
            [Output("refresh-settings", "is_open"),
             Output("realtime-interval", "disabled"),
             Output("realtime-interval", "interval")],
            [Input("auto-refresh-toggle", "value"),
             Input("refresh-interval", "value")]
        )
        def handle_realtime_settings(auto_refresh, interval):
            """Handle real-time data settings."""
            if auto_refresh:
                return True, False, (interval or 30) * 1000
            return False, True, 30000
        
        # Connection status check
        @app.callback(
            [Output("connection-status", "children"),
             Output("connection-status", "color"),
             Output("data-source-info", "children")],
            [Input("connection-check-interval", "n_intervals"),
             Input("realtime-data-toggle", "value")]
        )
        def check_connection_status(n_intervals, realtime):
            """Check and display connection status."""
            if realtime:
                try:
                    # Test connection using data_manager
                    test_data = data_manager.download_data("SPY", period="1d", use_cache=False)
                    
                    if test_data is not None and not test_data.empty:
                        status = [
                            html.I(className="fas fa-check-circle me-2"),
                            "Connected to real-time data"
                        ]
                        return status, "success", "Data source: Yahoo Finance (Live)"
                    else:
                        raise Exception("No data available")
                except Exception as e:
                    logger.error(f"Connection test failed: {e}")
                    status = [
                        html.I(className="fas fa-exclamation-circle me-2"),
                        "Connection error"
                    ]
                    return status, "danger", "Unable to connect to data source"
            else:
                status = [
                    html.I(className="fas fa-database me-2"),
                    "Using historical data"
                ]
                return status, "info", "Data source: Local cache"
        
        # Advanced settings toggle
        @app.callback(
            Output("advanced-collapse", "is_open"),
            [Input("advanced-toggle", "n_clicks")],
            [State("advanced-collapse", "is_open")]
        )
        def toggle_advanced_settings(n_clicks, is_open):
            """Toggle advanced settings panel."""
            if n_clicks:
                return not is_open
            return is_open
        
        # Sidebar state management
        @app.callback(
            Output("sidebar-state", "data"),
            [Input("ticker-multi-select", "value"),
             Input("date-range-picker", "start_date"),
             Input("date-range-picker", "end_date"),
             Input("time-granularity", "value"),
             Input("wavelets-patterns", "value"),
             Input("shapelets-patterns", "value"),
             Input("motifs-patterns", "value"),
             Input("confidence-threshold", "value"),
             Input("significance-threshold", "value"),
             Input("stability-threshold", "value"),
             Input("robustness-threshold", "value"),
             Input("realtime-data-toggle", "value"),
             Input("auto-refresh-toggle", "value"),
             Input("refresh-interval", "value"),
             Input("pattern-complexity", "value"),
             Input("performance-options", "value"),
             Input("cache-duration", "value")]
        )
        def update_sidebar_state(tickers, start_date, end_date, granularity,
                               wavelets, shapelets, motifs, confidence,
                               significance, stability, robustness, realtime,
                               auto_refresh, refresh_interval, complexity,
                               performance, cache_duration):
            """Update centralized sidebar state."""
            return {
                "tickers": tickers or [],
                "date_range": {
                    "start": start_date,
                    "end": end_date,
                    "granularity": granularity
                },
                "patterns": {
                    "wavelets": wavelets or [],
                    "shapelets": shapelets or [],
                    "motifs": motifs or []
                },
                "thresholds": {
                    "confidence": confidence,
                    "significance": significance,
                    "stability": stability,
                    "robustness": robustness
                },
                "realtime": {
                    "enabled": realtime,
                    "auto_refresh": auto_refresh,
                    "interval": refresh_interval
                },
                "complexity": complexity,
                "performance": performance or [],
                "cache_duration": cache_duration,
                "timestamp": datetime.now().isoformat()
            }
        
        # Selected tickers display
        @app.callback(
            Output("selected-tickers-display", "children"),
            [Input("ticker-multi-select", "value")]
        )
        def update_selected_tickers_display(tickers):
            """Update selected tickers display."""
            if not tickers:
                return html.P("No tickers selected", className="text-muted")
            
            return html.Div([
                dbc.Badge(
                    ticker,
                    color="primary",
                    className="me-1 mb-1",
                    pill=True
                )
                for ticker in tickers
            ])
        
        # Custom ticker collapse toggle
        @app.callback(
            Output("custom-ticker-collapse", "is_open"),
            [Input("ticker-category-tabs", "active_tab")]
        )
        def toggle_custom_ticker_input(active_tab):
            """Toggle custom ticker input based on tab selection."""
            return active_tab == "tab-custom"
        
        # Clear cache button
        @app.callback(
            [Output("ticker-cache", "clear_data"),
             Output("pattern-cache", "clear_data"),
             Output("settings-cache", "clear_data")],
            [Input("clear-cache-btn", "n_clicks")]
        )
        def clear_cache(n_clicks):
            """Clear all cached data."""
            if n_clicks:
                return True, True, True
            raise PreventUpdate
        
        # Export configuration
        @app.callback(
            Output("export-config-btn", "download"),
            [Input("export-config-btn", "n_clicks")],
            [State("sidebar-state", "data")]
        )
        def export_configuration(n_clicks, state):
            """Export current configuration to JSON file."""
            if not n_clicks:
                raise PreventUpdate
            
            config = {
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "configuration": state
            }
            
            return dict(
                content=json.dumps(config, indent=2),
                filename=f"sidebar_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        # Import configuration trigger
        @app.callback(
            Output("config-upload", "contents"),
            [Input("import-config-btn", "n_clicks")]
        )
        def trigger_import(n_clicks):
            """Trigger file upload for configuration import."""
            if n_clicks:
                # This will programmatically click the hidden upload component
                return dash.no_update
            raise PreventUpdate
        
        # Reset all settings
        @app.callback(
            [Output("ticker-multi-select", "value", allow_duplicate=True),
             Output("date-range-picker", "start_date", allow_duplicate=True),
             Output("date-range-picker", "end_date", allow_duplicate=True),
             Output("time-granularity", "value", allow_duplicate=True),
             Output("wavelets-patterns", "value", allow_duplicate=True),
             Output("shapelets-patterns", "value", allow_duplicate=True),
             Output("motifs-patterns", "value", allow_duplicate=True),
             Output("confidence-threshold", "value", allow_duplicate=True),
             Output("significance-threshold", "value", allow_duplicate=True),
             Output("stability-threshold", "value", allow_duplicate=True),
             Output("robustness-threshold", "value", allow_duplicate=True),
             Output("realtime-data-toggle", "value", allow_duplicate=True),
             Output("auto-refresh-toggle", "value", allow_duplicate=True),
             Output("pattern-complexity", "value", allow_duplicate=True),
             Output("performance-options", "value", allow_duplicate=True),
             Output("cache-duration", "value", allow_duplicate=True)],
            [Input("reset-all-btn", "n_clicks")],
            prevent_initial_call=True
        )
        def reset_all_settings(n_clicks):
            """Reset all settings to defaults."""
            if not n_clicks:
                raise PreventUpdate
            
            sidebar = InteractiveSidebar()
            
            return (
                sidebar.default_tickers,
                (datetime.now() - timedelta(days=90)).date(),
                datetime.now().date(),
                "1h",
                [],
                [],
                [],
                sidebar.quality_metrics["confidence"]["default"],
                sidebar.quality_metrics["significance"]["default"],
                sidebar.quality_metrics["stability"]["default"],
                sidebar.quality_metrics["robustness"]["default"],
                False,
                False,
                "medium",
                ["parallel", "memory"],
                60
            )
        
        # Quality summary update
        @app.callback(
            Output("quality-summary", "children"),
            [Input("confidence-threshold", "value"),
             Input("significance-threshold", "value"),
             Input("stability-threshold", "value"),
             Input("robustness-threshold", "value")]
        )
        def update_quality_summary(confidence, significance, stability, robustness):
            """Update quality threshold summary."""
            avg_threshold = (confidence + significance + stability + robustness) / 4
            
            if avg_threshold >= 0.7:
                level = "Conservative"
                color = "success"
            elif avg_threshold >= 0.5:
                level = "Balanced"
                color = "warning"
            else:
                level = "Aggressive"
                color = "danger"
            
            return dbc.Alert(
                f"Quality Level: {level} (Avg: {avg_threshold:.0%})",
                color=color,
                className="mb-0 py-2"
            )
        
        # Apply settings button and progress handling
        @app.callback(
            [Output("analysis-state", "data"),
             Output("progress-collapse", "is_open"),
             Output("progress-interval", "disabled"),
             Output("apply-settings-btn", "disabled"),
             Output("action-status", "children")],
            [Input("apply-settings-btn", "n_clicks")],
            [State("sidebar-state", "data")]
        )
        def handle_apply_settings(n_clicks, sidebar_state):
            """Handle apply settings button click."""
            if not n_clicks:
                raise PreventUpdate
            
            # Validate inputs
            is_valid, errors = validate_sidebar_inputs(sidebar_state)
            if not is_valid:
                error_msg = dbc.Alert(
                    ["⚠️ Please fix the following errors:"] + 
                    [html.Li(error) for error in errors],
                    color="danger",
                    dismissable=True
                )
                return dash.no_update, False, True, False, error_msg
            
            # Calculate estimated time
            tickers = sidebar_state.get("tickers", [])
            patterns = sidebar_state.get("patterns", {})
            total_patterns = sum(len(patterns.get(cat, [])) for cat in ["wavelets", "shapelets", "motifs"])
            
            # Estimate: 2 seconds per ticker + 1 second per pattern
            estimated_time = len(tickers) * 2 + total_patterns * 1
            
            # Initialize analysis state
            analysis_state = {
                "start_time": datetime.now().isoformat(),
                "estimated_time": estimated_time,
                "total_steps": len(tickers) * 3 + total_patterns,  # 3 steps per ticker
                "current_step": 0,
                "status": "initializing"
            }
            
            return analysis_state, True, False, True, ""
        
        # Progress update callback
        @app.callback(
            [Output("analysis-progress", "value"),
             Output("progress-status", "children"),
             Output("time-estimate", "children"),
             Output("analysis-state", "data", allow_duplicate=True)],
            [Input("progress-interval", "n_intervals")],
            [State("analysis-state", "data"),
             State("sidebar-state", "data")],
            prevent_initial_call=True
        )
        def update_progress(n_intervals, analysis_state, sidebar_state):
            """Update progress bar and status."""
            if not analysis_state:
                raise PreventUpdate
            
            # Simulate progress
            current_step = analysis_state.get("current_step", 0)
            total_steps = analysis_state.get("total_steps", 1)
            
            # Progress stages
            tickers = sidebar_state.get("tickers", [])
            patterns = sidebar_state.get("patterns", {})
            
            # Update step
            current_step += 1
            progress = min((current_step / total_steps) * 100, 100)
            
            # Determine current status
            if current_step <= len(tickers):
                status = f"📊 Loading data for {tickers[current_step-1] if current_step <= len(tickers) else 'ticker'}..."
            elif current_step <= len(tickers) * 2:
                ticker_idx = (current_step - len(tickers) - 1) % len(tickers)
                status = f"📈 Analyzing patterns for {tickers[ticker_idx]}..."
            elif current_step <= len(tickers) * 3:
                ticker_idx = (current_step - len(tickers)*2 - 1) % len(tickers)
                status = f"🔍 Extracting features for {tickers[ticker_idx]}..."
            else:
                status = "✨ Finalizing analysis..."
            
            # Calculate time remaining
            start_time = datetime.fromisoformat(analysis_state["start_time"])
            elapsed = (datetime.now() - start_time).total_seconds()
            if current_step > 0:
                time_per_step = elapsed / current_step
                remaining_steps = total_steps - current_step
                time_remaining = int(time_per_step * remaining_steps)
                
                if time_remaining > 60:
                    time_str = f"~{time_remaining // 60}m {time_remaining % 60}s remaining"
                else:
                    time_str = f"~{time_remaining}s remaining"
            else:
                time_str = f"~{analysis_state['estimated_time']}s estimated"
            
            # Update state
            analysis_state["current_step"] = current_step
            
            # Check if complete
            if progress >= 100:
                analysis_state["status"] = "complete"
                time_str = "✅ Analysis complete!"
            
            return progress, status, time_str, analysis_state
        
        # Analysis completion callback
        @app.callback(
            [Output("progress-interval", "disabled", allow_duplicate=True),
             Output("apply-settings-btn", "disabled", allow_duplicate=True),
             Output("action-status", "children", allow_duplicate=True),
             Output("progress-collapse", "is_open", allow_duplicate=True)],
            [Input("analysis-state", "data")],
            prevent_initial_call=True
        )
        def handle_analysis_completion(analysis_state):
            """Handle analysis completion."""
            if not analysis_state or analysis_state.get("status") != "complete":
                raise PreventUpdate
            
            # Calculate total time
            start_time = datetime.fromisoformat(analysis_state["start_time"])
            total_time = (datetime.now() - start_time).total_seconds()
            
            success_msg = dbc.Alert(
                [
                    html.I(className="fas fa-check-circle me-2"),
                    f"Analysis completed successfully in {total_time:.1f} seconds!"
                ],
                color="success",
                dismissable=True
            )
            
            # Re-enable button and hide progress after 3 seconds
            return True, False, success_msg, False


# Helper functions for sidebar state management
def get_sidebar_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and validate sidebar state."""
    if not state:
        sidebar = InteractiveSidebar()
        return {
            "tickers": sidebar.default_tickers,
            "date_range": {
                "start": (datetime.now() - timedelta(days=90)).date().isoformat(),
                "end": datetime.now().date().isoformat(),
                "granularity": "1h"
            },
            "patterns": {
                "wavelets": [],
                "shapelets": [],
                "motifs": []
            },
            "thresholds": {
                "confidence": 0.7,
                "significance": 0.6,
                "stability": 0.5,
                "robustness": 0.8
            },
            "realtime": {
                "enabled": False,
                "auto_refresh": False,
                "interval": 30
            },
            "complexity": "medium",
            "performance": ["parallel", "memory"],
            "cache_duration": 60
        }
    return state


def validate_sidebar_inputs(state: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate sidebar input values."""
    errors = []
    
    # Validate tickers
    if not state.get("tickers"):
        errors.append("At least one ticker must be selected")
    
    # Validate date range
    date_range = state.get("date_range", {})
    if date_range.get("start") and date_range.get("end"):
        start = pd.to_datetime(date_range["start"])
        end = pd.to_datetime(date_range["end"])
        if start >= end:
            errors.append("Start date must be before end date")
    
    # Validate thresholds
    thresholds = state.get("thresholds", {})
    for metric, value in thresholds.items():
        if value < 0 or value > 1:
            errors.append(f"{metric} threshold must be between 0 and 1")
    
    # Validate refresh interval
    realtime = state.get("realtime", {})
    interval = realtime.get("interval", 30)
    if interval < 1 or interval > 300:
        errors.append("Refresh interval must be between 1 and 300 seconds")
    
    return len(errors) == 0, errors


def format_sidebar_summary(state: Dict[str, Any]) -> str:
    """Format sidebar state into a readable summary."""
    tickers = state.get("tickers", [])
    date_range = state.get("date_range", {})
    patterns = state.get("patterns", {})
    thresholds = state.get("thresholds", {})
    realtime = state.get("realtime", {})
    
    # Count total patterns
    total_patterns = sum(len(patterns.get(cat, [])) for cat in ["wavelets", "shapelets", "motifs"])
    
    summary = f"""
    Sidebar Configuration:
    - Tickers: {', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''} ({len(tickers)} total)
    - Date Range: {date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')}
    - Granularity: {date_range.get('granularity', 'N/A')}
    - Patterns: {total_patterns} selected
    - Avg Threshold: {sum(thresholds.values()) / len(thresholds) if thresholds else 0:.0%}
    - Mode: {'Real-time' if realtime.get('enabled') else 'Historical'}
    - Auto-refresh: {'On' if realtime.get('auto_refresh') else 'Off'}
    """
    return summary.strip()
